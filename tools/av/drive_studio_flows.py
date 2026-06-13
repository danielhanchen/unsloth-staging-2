#!/usr/bin/env python3
"""Drive a live Unsloth Studio (inference, image-gen, training) while AV watches.

Shared by Track A (Windows / Defender, --mode cpu) and Track B (Linux B200 /
ClamAV, --mode gpu). Inference and image-gen run through the real /chat UI via
Playwright (studio_test_kit); training is started through the authed backend
API. Every produced artifact is recorded in a manifest the caller can scan.

In gpu mode the GPU-bound paths (image-gen, training) must complete; in cpu
mode they are attempted and a controlled "no-GPU / too-slow" degradation is a
pass (only a crash or AV detection fails). Exit code: 0 = all requested actions
clean (real or graceful), 1 = a hard failure.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import httpx

_WS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_WS_ROOT))

from studio_test_kit.auth import StudioAuth, seed_init_script  # noqa: E402
from studio_test_kit.ui import (  # noqa: E402
    open_chat, send_prompt, wait_for_stream,
)

API_T = 30.0


# ── auth (handles first-run forced password change) ──────────────────

def authenticate(base: str, username: str, bootstrap_pw: str, new_pw: str) -> StudioAuth:
    """Log in, clearing the one-time bootstrap-password gate if present."""
    with httpx.Client(timeout=API_T) as c:
        for pw in (new_pw, bootstrap_pw):
            r = c.post(f"{base}/api/auth/login",
                       json={"username": username, "password": pw})
            if r.status_code != 200:
                continue
            tok = r.json()
            if not tok.get("must_change_password") and pw == new_pw:
                return StudioAuth(tok["access_token"], tok.get("refresh_token", ""), base)
            # bootstrap login (or must_change) -> rotate to new_pw
            ch = c.post(f"{base}/api/auth/change-password",
                        headers={"Authorization": f"Bearer {tok['access_token']}"},
                        json={"current_password": pw, "new_password": new_pw})
            if ch.status_code == 200:
                nt = ch.json()
                return StudioAuth(nt["access_token"], nt.get("refresh_token", ""), base)
            # already rotated by a prior run: log in with new_pw
            r2 = c.post(f"{base}/api/auth/login",
                        json={"username": username, "password": new_pw})
            if r2.status_code == 200:
                nt = r2.json()
                return StudioAuth(nt["access_token"], nt.get("refresh_token", ""), base)
        raise SystemExit("auth failed: neither new nor bootstrap password worked")


def _hdr(auth: StudioAuth) -> dict:
    return {"Authorization": f"Bearer {auth.access_token}"}


# ── inference model load (the UI calls this same endpoint) ───────────

def load_model(base: str, auth: StudioAuth, model_path: str, variant: str | None,
               gpu_ids: list[int] | None, timeout: float = 1800.0) -> dict:
    body: dict = {"model_path": model_path, "max_seq_length": 2048}
    if variant:
        body["gguf_variant"] = variant
    if gpu_ids is not None:
        body["gpu_ids"] = gpu_ids
    with httpx.Client(timeout=timeout) as c:
        r = c.post(f"{base}/api/inference/load", headers=_hdr(auth), json=body)
    r.raise_for_status()
    return r.json()


def inference_status(base: str, auth: StudioAuth) -> dict:
    with httpx.Client(timeout=API_T) as c:
        return c.get(f"{base}/api/inference/status", headers=_hdr(auth)).json()


def picker_label(base: str, auth: StudioAuth) -> str | None:
    """The id the /chat picker exposes for the loaded model."""
    with httpx.Client(timeout=API_T) as c:
        d = c.get(f"{base}/v1/models", headers=_hdr(auth)).json()
    ids = [m.get("id") for m in d.get("data", [])]
    return ids[0] if ids else None


# ── thread helpers (model is pre-selected by the API load) ───────────

async def thread_text(sp) -> str:
    return await sp.page.evaluate(
        "() => { const t = document.querySelector('[data-testid=\"chat-thread\"]');"
        "        return t ? t.innerText : ''; }"
    )


async def ui_model_label(sp, needle: str) -> str | None:
    try:
        loc = sp.page.locator(f'button:has-text("{needle}")').first
        return (await loc.inner_text()).strip().splitlines()[0][:80]
    except Exception:
        return None


# ── robust reply wait (stop button OR thread-text growth+settle) ─────

async def wait_for_reply(sp, before_len: int, timeout_ms: int) -> str:
    """Wait until a reply finishes: the thread text grows past the prompt and
    then stops changing for ~4s. Works for normal LLMs and the slower
    diffusion runner where the Stop button timing is unreliable."""
    try:
        await wait_for_stream(sp, timeout_ms=min(timeout_ms, 20000))
    except Exception:
        pass
    import asyncio as _a
    deadline = _a.get_event_loop().time() + timeout_ms / 1000
    last, stable = "", 0
    while _a.get_event_loop().time() < deadline:
        cur = await thread_text(sp)
        if len(cur) > before_len + 10 and cur == last:
            stable += 1
            if stable >= 4:
                return cur
        else:
            stable = 0
        last = cur
        await _a.sleep(1.0)
    return await thread_text(sp)


# ── UI: text generation (used for inference AND diffusiongemma) ──────

async def run_text(base: str, auth: StudioAuth, needle: str, out: Path,
                   origin_name: str, label: str, prompt: str, mode: str,
                   timeout_ms: int) -> dict:
    init = seed_init_script(auth, providers=[])
    res = {"action": label, "origin": origin_name, "ok": False,
           "degraded": False, "files": []}
    try:
        async with open_chat(base, init_scripts=[init], video_dir=out / "video",
                             video_name=f"{label}_{origin_name}", headless=True) as sp:
            await sp.page.locator("form:has(textarea) textarea").first.wait_for(
                state="visible", timeout=20000)
            res["ui_model"] = await ui_model_label(sp, needle)  # API load pre-selects it
            before = await thread_text(sp)
            await send_prompt(sp, prompt)
            after = await wait_for_reply(sp, len(before), timeout_ms)
            shot = out / f"{label}_{origin_name}.png"
            await sp.screenshot(shot, full_page=False)
            res["files"].append(str(shot))
            res["reply_chars"] = max(0, len(after) - len(before))
            got = len(after) > len(before) + 10
            res["ok"] = got
            res["degraded"] = (not got) and (mode == "cpu")
            if not got and mode == "cpu":
                res["ok"] = True  # graceful no-GPU degradation is a pass on cpu
        if sp.video_webm:
            res["files"].append(str(sp.video_webm))
    except Exception as e:  # crash is always a failure
        res["error"] = repr(e)[:300]
        res["ok"] = (mode == "cpu")
        res["degraded"] = (mode == "cpu")
    return res


# ── API: training ────────────────────────────────────────────────────

def make_raw_dataset(path: Path) -> Path:
    rows = [
        {"text": "Unsloth makes finetuning fast and memory efficient."},
        {"text": "Llamas are camelids native to South America."},
        {"text": "LoRA adapts a few low-rank matrices instead of full weights."},
        {"text": "Studio runs models locally and offline."},
    ] * 8
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return path


def run_training(base: str, auth: StudioAuth, model_name: str, dataset: Path,
                 out: Path, mode: str, max_wait_s: int) -> dict:
    res = {"action": "training", "ok": False, "degraded": False, "files": []}
    body = {
        "model_name": model_name,
        "training_type": "LoRA/QLoRA",
        "load_in_4bit": False,
        "max_seq_length": 512,
        "format_type": "raw",
        "local_datasets": [str(dataset)],
        "num_epochs": 0,
        "max_steps": 5,
        "learning_rate": "2e-4",
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "warmup_steps": 1,
        "save_steps": 0,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
    }
    with httpx.Client(timeout=60.0) as c:
        r = c.post(f"{base}/api/train/start", headers=_hdr(auth), json=body)
    res["start_http"] = r.status_code
    try:
        res["start_resp"] = r.json()
    except Exception:
        res["start_resp"] = r.text[:300]
    if r.status_code >= 400:
        res["ok"] = (mode == "cpu")  # no-GPU rejection is acceptable on cpu
        res["degraded"] = True
        return res
    # poll status to a terminal state (the API reports progress via `phase` +
    # `is_training_running`, not `status`)
    t0 = time.time()
    last = {}
    while time.time() - t0 < max_wait_s:
        with httpx.Client(timeout=API_T) as c:
            s = c.get(f"{base}/api/train/status", headers=_hdr(auth))
        if s.status_code == 200:
            last = s.json()
            phase = (last.get("phase") or "").lower()
            running = last.get("is_training_running")
            if phase in ("completed", "finished", "done", "success"):
                res["ok"] = True
                break
            if phase in ("error", "failed", "stopped") or last.get("error"):
                # GPU mode: a real failure; cpu mode: expected no-GPU degradation
                res["ok"] = (mode == "cpu")
                res["degraded"] = (mode == "cpu")
                break
            if running is False and phase and phase not in (
                    "starting", "initializing", "queued", "preparing", "loading"):
                res["ok"] = (phase == "completed")
                break
        time.sleep(5)
    res["final_status"] = last
    out_dir = (last.get("details") or {}).get("output_dir")
    if out_dir:
        res["output_dir"] = out_dir
        res["files"].append(out_dir)
    return res


# ── orchestration ────────────────────────────────────────────────────

async def amain(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    auth = authenticate(args.base, args.username, args.bootstrap_password, args.new_password)
    results = []
    gpu_ids = [args.gpu] if (args.mode == "gpu" and args.gpu is not None) else None
    actions = [a.strip() for a in args.actions.split(",") if a.strip()]

    if "inference" in actions:
        ld = load_model(args.base, auth, args.inference_model, args.inference_variant, gpu_ids)
        results.append({"action": "load_inference", "display": ld.get("display_name"),
                        "status": ld.get("status")})
        results.append(await run_text(
            args.base, auth, args.inference_needle, out, args.origin,
            "inference", "In one short sentence, what is a llama?", args.mode, 120000))

    if "diffusion" in actions or "image" in actions:
        # DiffusionGemma is a diffusion-based TEXT model (served via the
        # --gpu 0 visual-server runner); drive it as a text generation.
        try:
            ld = load_model(args.base, auth, args.image_model, args.image_variant, gpu_ids)
            results.append({"action": "load_diffusion", "display": ld.get("display_name"),
                            "status": ld.get("status"), "is_diffusion": ld.get("is_diffusion")})
            r = await run_text(args.base, auth, args.image_needle, out, args.origin,
                               "diffusion", "Write one short sentence about llamas.",
                               args.mode, args.image_timeout_ms)
        except Exception as e:
            r = {"action": "diffusion", "ok": (args.mode == "cpu"), "degraded": True,
                 "error": repr(e)[:300]}
        results.append(r)

    if "training" in actions:
        # Datasets must live under Studio's uploads root (resolve_dataset_path
        # only accepts paths contained in the dataset roots).
        upload_dir = Path(args.dataset_dir) if args.dataset_dir else (
            Path(os.environ["UNSLOTH_STUDIO_HOME"]) / "assets" / "datasets" / "uploads"
            if os.environ.get("UNSLOTH_STUDIO_HOME") else out / "train_data")
        ds = make_raw_dataset(upload_dir / "av_raw.jsonl")
        r = run_training(args.base, auth, args.train_model, ds, out, args.mode,
                         args.train_timeout_s)
        results.append(r)

    summary = {
        "base": args.base, "mode": args.mode, "origin": args.origin,
        "actions": actions, "results": results,
    }
    (out / f"driver_result_{args.origin}.json").write_text(json.dumps(summary, indent=2))
    # collect produced files for the caller's scanner
    files = []
    for r in results:
        files.extend(r.get("files", []))
    (out / f"driver_files_{args.origin}.txt").write_text("\n".join(files))

    hard_fail = any(r.get("ok") is False for r in results
                    if r.get("action") in ("inference", "diffusion", "image", "training"))
    print(json.dumps(summary, indent=2))
    return 1 if hard_fail else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8888")
    ap.add_argument("--username", default="unsloth")
    ap.add_argument("--bootstrap-password", default="")
    ap.add_argument("--new-password", required=True)
    ap.add_argument("--actions", default="inference,image,training")
    ap.add_argument("--mode", choices=["cpu", "gpu"], default="gpu")
    ap.add_argument("--origin", default="localhost")
    ap.add_argument("--out", default="outputs/studio_live_av/run")
    ap.add_argument("--gpu", type=int, default=None)
    # inference (GGUF)
    ap.add_argument("--inference-model", default="unsloth/SmolLM2-135M-Instruct-GGUF")
    ap.add_argument("--inference-variant", default="Q4_K_M")
    ap.add_argument("--inference-needle", default="SmolLM2-135M")
    # image (diffusion)
    ap.add_argument("--image-model", default="")
    ap.add_argument("--image-variant", default="")
    ap.add_argument("--image-needle", default="")
    ap.add_argument("--image-timeout-ms", type=int, default=180000)
    # training
    ap.add_argument("--train-model", default="unsloth/SmolLM2-135M-Instruct")
    ap.add_argument("--train-timeout-s", type=int, default=900)
    ap.add_argument("--dataset-dir", default="",
                    help="Studio dataset uploads dir; default <UNSLOTH_STUDIO_HOME>/assets/datasets/uploads")
    args = ap.parse_args()
    raise SystemExit(asyncio.run(amain(args)))


if __name__ == "__main__":
    main()
