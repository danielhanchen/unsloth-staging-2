#!/usr/bin/env python3
"""Run one REAL diffusion generation against a live Studio, for the cross-platform staging CI.

The per-OS staging jobs only ran the pytest suite, so nothing proved that image generation works
on a Mac, a Windows box, or a GPU-less Linux host. This drives the same HTTP routes the UI calls
-- download-plan, load, generate, gallery -- and fails the job when the result is not a real
image.

Legs:
  native-gguf   a small GGUF through the native sd.cpp engine (what a CPU/Metal host actually
                routes to for a GGUF pick)
  cpu-pipeline  a safetensors pipeline through diffusers on the CPU

Usage:
  python probe.py --leg native-gguf --out artifacts/ubuntu-latest
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import sys
import time
from pathlib import Path

import httpx

USER = "unsloth"

# uvicorn closes an idle keep-alive connection after 5 s (--timeout-keep-alive default), and the
# load poll below sleeps 10 s, so every poll after the first is sent on a connection the server has
# already closed. httpx does not retry that -- it surfaces as
# `RemoteProtocolError: Server disconnected without sending a response` -- and it killed the
# macos-14 run mid-load while ubuntu/windows happened to win the same race. Take a fresh
# connection per request (max_keepalive_connections = 0) and retry transport errors anyway, since
# a shared runner can drop a connection for reasons of its own.
NO_KEEPALIVE = httpx.Limits(max_keepalive_connections=0)
TRANSPORT_RETRIES = 3

LEGS = {
    # FLUX.2-klein-4B Q2_K is the smallest complete native set: 1.83 GB transformer + a 0.34 GB
    # VAE + the 8.04 GB Qwen3-4B encoder the family maps for sd.cpp.
    "native-gguf": {
        "model_path": "unsloth/FLUX.2-klein-4B-GGUF",
        "model_kind": "gguf",
        "gguf_filename": "flux-2-klein-4b-Q2_K.gguf",
        "width": 512,
        "height": 512,
        "steps": 4,
        "guidance": 0.0,
        "expect_engine": "sd_cpp",
        "min_free_gb": 12.0,
        # The native engine reads these single files; the plan must stage them instead of the base
        # repo's sharded diffusers components.
        "native_asset_repos": ["Comfy-Org/flux2-dev", "Comfy-Org/z_image_turbo"],
    },
    # SDXL-Turbo is the smallest supported dense pipeline, and 2 steps is what it is trained for.
    "cpu-pipeline": {
        "model_path": "stabilityai/sdxl-turbo",
        "model_kind": "pipeline",
        "gguf_filename": None,
        "width": 512,
        "height": 512,
        "steps": 2,
        "guidance": 0.0,
        "expect_engine": "diffusers",
        "min_free_gb": 9.0,
        "native_asset_repos": [],
    },
}

PROMPT = "a lone red umbrella on a rainy street at night, neon reflections"


def free_gb(path: str) -> float:
    return round(shutil.disk_usage(path).free / 1e9, 1)


def get_with_retry(client: httpx.Client, url: str, hdr: dict) -> httpx.Response:
    """GET an idempotent status route, retrying a dropped connection.

    A transport error here is not the server saying no, it is the connection dying underneath an
    otherwise healthy poll, and failing the job on it reports a platform failure that never
    happened."""
    last: Exception | None = None
    for attempt in range(TRANSPORT_RETRIES):
        try:
            # Its own short timeout: the client's is sized for the generate POST (an hour), and a
            # status route that has not answered in 30 s is stuck, not slow -- waiting the full
            # hour would burn the load deadline instead of retrying.
            return client.get(url, headers=hdr, timeout=30.0)
        except httpx.TransportError as exc:
            last = exc
            print(f"  transport error ({exc!r}); retry {attempt + 1}/{TRANSPORT_RETRIES}", flush=True)
            time.sleep(2.0)
    raise RuntimeError(
        f"the Studio server stopped answering {url} after {TRANSPORT_RETRIES} attempts: {last!r}"
    ) from last


def total_ram_gb() -> float:
    """Physical RAM, recorded with each run so a memory-shaped failure can be read off the record
    instead of guessed at. No psutil on these runners, so ask the OS directly."""
    try:
        if sys.platform == "win32":
            class _MemStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemStatus()
            status.dwLength = ctypes.sizeof(_MemStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return round(status.ullTotalPhys / 1e9, 1)
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
    except Exception:  # noqa: BLE001 -- an unknown size must not gate the run
        return 0.0


def login(client: httpx.Client, base: str, password: str) -> str:
    r = client.post(f"{base}/api/auth/login", json={"username": USER, "password": password})
    r.raise_for_status()
    return r.json()["access_token"]


def wait_for_load(client: httpx.Client, base: str, hdr: dict, timeout_s: int) -> tuple[bool, dict]:
    """Poll until the model reports loaded, or the load reports an error / the deadline passes."""
    deadline = time.time() + timeout_s
    last: dict = {}
    while time.time() < deadline:
        progress = get_with_retry(client, f"{base}/api/inference/images/load-progress", hdr).json()
        last = progress
        if progress.get("error"):
            return False, progress
        status = get_with_retry(client, f"{base}/api/inference/images/status", hdr).json()
        if status.get("loaded"):
            return True, status
        pct = progress.get("fraction")
        print(
            f"  loading... phase={progress.get('phase')} "
            f"{'' if pct is None else f'{pct:.0%}'} {progress.get('detail') or ''}"[:140],
            flush=True,
        )
        time.sleep(10)
    return False, last


def image_is_real(path: Path) -> dict:
    """A generated image must decode and carry actual variation: a black or single-colour frame is
    the classic silent failure for a quantized model on an untested platform."""
    from PIL import Image, ImageStat

    with Image.open(path) as im:
        im = im.convert("RGB")
        stat = ImageStat.Stat(im)
        return {
            "size": list(im.size),
            "mean": [round(v, 2) for v in stat.mean],
            "stddev": [round(v, 2) for v in stat.stddev],
            "flat": max(stat.stddev) < 2.0,
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leg", required=True, choices=sorted(LEGS))
    ap.add_argument("--base", default=os.environ.get("STUDIO_BASE", "http://127.0.0.1:8899"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--password", default=os.environ.get("STUDIO_PASS", ""))
    ap.add_argument("--load-timeout", type=int, default=3600)
    ap.add_argument("--gen-timeout", type=int, default=3600)
    args = ap.parse_args()

    leg = LEGS[args.leg]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    record: dict = {
        "leg": args.leg,
        "platform": sys.platform,
        # `prompts` is the prompt LIST, not a count, so qa_6763/prompt_match.py can score the
        # render against what was actually asked for. The pixel-stddev floor below only rules out
        # a flat frame; it passes any structurally credible image, including a canned card.
        "target": {**leg, "prompts": [PROMPT]},
        "free_gb_before": free_gb(str(out)),
    }

    with httpx.Client(timeout=args.gen_timeout + 60, limits=NO_KEEPALIVE) as client:
        token = login(client, args.base, args.password)
        hdr = {"Authorization": f"Bearer {token}"}

        body = {"model_path": leg["model_path"], "model_kind": leg["model_kind"]}
        if leg["gguf_filename"]:
            body["gguf_filename"] = leg["gguf_filename"]
        if os.environ.get("HF_TOKEN"):
            body["hf_token"] = os.environ["HF_TOKEN"]

        plan = client.post(f"{args.base}/api/inference/images/download-plan", json=body, headers=hdr)
        plan_body = plan.json() if plan.status_code == 200 else {}
        need_gb = round((plan_body.get("total_bytes") or 0) / 1e9, 1)
        record["plan"] = {
            "status": plan.status_code,
            "need_gb": need_gb,
            "entries": [
                {"repo": e["repo_id"], "files": len(e["files"]), "gb": round(e["bytes"] / 1e9, 2)}
                for e in plan_body.get("entries", [])
            ],
        }
        # The plan is engine-specific: native sd.cpp reads single-file VAE + text encoders and never
        # opens the base repo's sharded components, so a native-routed leg whose plan lists the
        # diffusers set would stage GB the load discards and fetch the real assets inline. Checked
        # here because it needs no download, so even a runner too small to generate proves it.
        want_repos = set(leg.get("native_asset_repos") or ())
        plan_repos = {e["repo"] for e in record["plan"]["entries"]}
        plan_ok = want_repos.issubset(plan_repos) if want_repos else True
        record["plan"]["native_shaped"] = plan_ok
        if not plan_ok:
            record["plan"]["missing_native_repos"] = sorted(want_repos - plan_repos)

        # Recorded, not gated: macos-14 has 7 GB, and round 68 showed it loads this native set and
        # gets as far as prompt encoding (where upstream ggml Metal aborts), so a RAM threshold
        # here would skip the one leg that still has something to tell us.
        record["ram_gb"] = total_ram_gb()

        # A runner that cannot hold the weights is a CI limit, not a product failure: say so loudly
        # and exit non-zero only for real failures.
        if need_gb and record["free_gb_before"] < need_gb + 2:
            record["skipped"] = (
                f"needs ~{need_gb} GB plus headroom, runner has {record['free_gb_before']} GB free"
            )
            (out / "results.json").write_text(json.dumps(record, indent=2))
            print(json.dumps(record, indent=2))
            print("VERDICT: SKIPPED (disk)")
            return 0 if plan_ok else 1

        t0 = time.time()
        started = client.post(f"{args.base}/api/inference/images/load", json=body, headers=hdr)
        record["load_start_status"] = started.status_code
        ok, status = wait_for_load(client, args.base, hdr, args.load_timeout)
        record["load"] = {
            "loaded": ok,
            "seconds": round(time.time() - t0, 1),
            "engine": status.get("engine"),
            "native_mode": status.get("native_mode"),
            "device": status.get("device"),
            "dtype": status.get("dtype"),
            "model_kind": status.get("model_kind"),
            "transformer_quant": status.get("transformer_quant"),
            "fallback_reason": status.get("fallback_reason"),
            "detail": None if ok else str(status)[:400],
        }
        if not ok:
            (out / "results.json").write_text(json.dumps(record, indent=2))
            print(json.dumps(record, indent=2))
            print("VERDICT: FAIL (load)")
            return 1

        t0 = time.time()
        gen = client.post(
            f"{args.base}/api/inference/images/generate",
            json={
                "prompt": PROMPT,
                "width": leg["width"],
                "height": leg["height"],
                "steps": leg["steps"],
                "guidance": leg["guidance"],
                "seed": 1234,
                "batch_size": 1,
            },
            headers=hdr,
        )
        record["generate"] = {"status": gen.status_code, "seconds": round(time.time() - t0, 1)}
        if gen.status_code != 200:
            record["generate"]["detail"] = gen.text[:400]
            (out / "results.json").write_text(json.dumps(record, indent=2))
            print(json.dumps(record, indent=2))
            print("VERDICT: FAIL (generate)")
            return 1

        # The access token is minted once, before a generation that can run for hours on a
        # CPU-backend fallback (macos-14 round 74: 8017 s). By the time it returns, the token
        # can be expired, and every check after this point 401s -- which reads as "no gallery
        # record" and fails a job whose image was in fact generated and stored. Re-login on a
        # 401 rather than reporting the product broken.
        gallery_url = f"{args.base}/api/inference/images/gallery?limit=1&offset=0"
        resp = client.get(gallery_url, headers=hdr)
        if resp.status_code == 401:
            print("gallery 401 after a long generation: re-authenticating", flush=True)
            hdr = {"Authorization": f"Bearer {login(client, args.base, args.password)}"}
            record["reauthenticated"] = True
            resp = client.get(gallery_url, headers=hdr)
        gallery = resp.json()
        images = gallery.get("images") or []
        record["gallery_count"] = len(images)
        if not images:
            (out / "results.json").write_text(json.dumps(record, indent=2))
            print("VERDICT: FAIL (no gallery record)")
            return 1
        image_id = images[0]["id"]
        png = client.get(
            f"{args.base}/api/inference/images/gallery/{image_id}/file", headers=hdr
        )
        target = out / f"{args.leg}.png"
        target.write_bytes(png.content)
        record["image"] = image_is_real(target)
        record["image"]["bytes"] = len(png.content)
        record["image"]["id"] = image_id
        # The shape qa_6763/prompt_match.py reads: one entry per generation, PNG named relative to
        # this results.json so the artifact directory is auditable straight out of the download.
        record["generations"] = [{
            "status": gen.status_code,
            "seconds": record["generate"]["seconds"],
            "prompt": PROMPT,
            "png": target.name,
        }]

    record["free_gb_after"] = free_gb(str(out))
    (out / "results.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))

    engine_ok = record["load"]["engine"] == leg["expect_engine"]
    passed = (
        engine_ok
        and record["plan"]["native_shaped"]
        and not record["image"]["flat"]
        and record["image"]["bytes"] > 1000
    )
    if not record["plan"]["native_shaped"]:
        print(
            "NOTE: the download plan did not stage the native assets "
            f"{record['plan'].get('missing_native_repos')}, so the loader fetched them inline"
        )
    if not engine_ok:
        print(
            f"NOTE: engine was {record['load']['engine']!r}, expected {leg['expect_engine']!r} "
            f"({record['load']['fallback_reason']})"
        )
    print("VERDICT:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
