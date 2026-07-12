#!/usr/bin/env python3
"""PR #7080 Studio before/after driver.

Drives the real Unsloth Studio chat UI (Playwright) to prove persistent stdio
MCP sessions. It configures a stateful stdio "increment" counter MCP server,
loads a local tool-capable model, and issues two single-call turns in ONE
conversation:

  AFTER  (PR head):     COUNTER=1 then COUNTER=2, SAME pid   -> state persisted
  BEFORE (merge-base):  COUNTER=1 then COUNTER=1, DIFFERENT pid -> state lost

Records screenshots + video and writes assertions.json. Fail-hard on the
counter/pid assertions for the given --side.

Usage:
  python pr7080_studio_demo.py --base-url http://127.0.0.1:8904 \
     --password <bootstrap> --side AFTER --model-path Qwen/Qwen2.5-3B-Instruct-GGUF \
     --gguf-variant q4_k_m --counter-cmd '<py> -u /abs/counter_mcp.py' --out-dir out/after
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

import httpx

# studio_test_kit is importable from the workspace root (local) or vendored
# alongside this script (CI).
_HERE = Path(__file__).resolve().parent
for cand in (_HERE, _HERE.parent, _HERE.parent.parent):
    if (cand / "studio_test_kit").is_dir():
        sys.path.insert(0, str(cand))
        break
from studio_test_kit.auth import StudioAuth, seed_init_script  # noqa: E402
from studio_test_kit.ui import open_chat  # noqa: E402

NEW_PASSWORD = "UnslothDemo7080x"
PROMPT_1 = (
    "Call the Stateful Counter increment tool exactly once now. Do not guess or "
    "compute the result yourself. After the tool returns, reply with only its raw "
    "text result."
)
PROMPT_2 = (
    "Call the same Stateful Counter increment tool exactly once again, in this same "
    "conversation. Do not reuse the previous answer. After the tool returns, reply "
    "with only its new raw text result."
)
RESULT_RE = re.compile(r"COUNTER\s*=\s*(\d+)\s*;\s*PID\s*=\s*(\d+)")


def log(msg: str) -> None:
    print(f"[demo] {msg}", flush=True)


async def auth(base: str, bootstrap_pw: str) -> StudioAuth:
    """Login; complete the forced bootstrap password change; return fresh tokens."""
    async with httpx.AsyncClient(timeout=30.0) as c:
        r = await c.post(f"{base}/api/auth/login", json={"username": "unsloth", "password": bootstrap_pw})
        r.raise_for_status()
        b = r.json()
        tok = b["access_token"]
        if b.get("must_change_password"):
            r2 = await c.post(
                f"{base}/api/auth/change-password",
                headers={"Authorization": f"Bearer {tok}"},
                json={"current_password": bootstrap_pw, "new_password": NEW_PASSWORD},
            )
            r2.raise_for_status()
            b = r2.json()
            tok = b.get("access_token", tok)
        return StudioAuth(access_token=tok, refresh_token=b.get("refresh_token", ""), base_url=base)


async def ensure_model_loaded(base: str, tok: str, model_path: str, gguf_variant: str) -> dict:
    """Load the model via API if nothing is loaded yet (idempotent)."""
    async with httpx.AsyncClient(timeout=2400.0) as c:
        h = {"Authorization": f"Bearer {tok}"}
        r = await c.get(f"{base}/api/inference/models", headers=h)
        data = r.json().get("data", [])
        loaded = [m for m in data if m.get("loaded")]
        if loaded:
            log(f"model already loaded: {loaded[0].get('id')}")
            return loaded[0]
        log(f"loading model {model_path} ({gguf_variant}) via API ...")
        body = {"model_path": model_path, "gguf_variant": gguf_variant,
                "hf_token": os.environ.get("HF_TOKEN"), "max_seq_length": 8192}
        r = await c.post(f"{base}/api/inference/load", headers=h, json=body)
        r.raise_for_status()
        log(f"load response: {json.dumps(r.json())[:200]}")
        return r.json()


async def cleanup_mcp_servers(base: str, tok: str) -> None:
    """Delete any pre-existing MCP servers so each run configures exactly one."""
    async with httpx.AsyncClient(timeout=30.0) as c:
        h = {"Authorization": f"Bearer {tok}"}
        r = await c.get(f"{base}/api/mcp/servers/", headers=h)
        for s in (r.json() if r.status_code == 200 else []):
            sid = s.get("id")
            if sid:
                await c.delete(f"{base}/api/mcp/servers/{sid}", headers=h)
        log("cleaned existing MCP servers")


async def dump_on_error(sp, out: Path, tag: str) -> None:
    try:
        await sp.screenshot(out / f"ERROR_{tag}.png", full_page=True)
        (out / f"ERROR_{tag}.html").write_text(await sp.page.content(), errors="ignore")
    except Exception as e:  # noqa: BLE001
        log(f"dump failed: {e}")


async def configure_mcp(sp, out: Path, counter_cmd: str) -> None:
    page = sp.page
    log("opening MCP menu ...")
    pill = page.locator('button[data-pill-label="MCP"]')
    await pill.wait_for(state="visible", timeout=30000)
    await pill.click()
    await page.get_by_role("menuitem", name="Manage MCP servers").click()
    await page.get_by_role("heading", name="MCP Servers").wait_for(timeout=15000)
    await page.get_by_role("button", name="Add server").click()
    await page.get_by_label("Display name").fill("Stateful Counter")
    await page.get_by_label("URL or command").fill(counter_cmd)
    await sp.screenshot(out / "01_mcp_form.png", full_page=False)
    log("testing MCP connection ...")
    await page.get_by_role("button", name="Test connection").click()
    ok = page.locator('[data-sonner-toast][data-type="success"]')
    await ok.filter(has_text=re.compile(r"Connected \(\d+ tools?\)")).first.wait_for(timeout=60000)
    log("connected; saving server ...")
    await page.get_by_role("button", name="Add server").click()
    await page.get_by_text("MCP server added").wait_for(timeout=30000)
    # make sure the row switch is enabled
    try:
        sw = page.get_by_role("switch", name="Enable server").first
        if await sw.get_attribute("aria-checked") == "false":
            await sw.click()
    except Exception:  # noqa: BLE001
        pass
    # close the dialog
    try:
        await page.get_by_role("button", name="Close").first.click()
    except Exception:  # noqa: BLE001
        await page.keyboard.press("Escape")
    await pill.wait_for(state="visible", timeout=10000)


async def pick_model(sp, model_path: str) -> None:
    page = sp.page
    trig = page.locator('[data-tour="chat-model-selector"]')
    if not await trig.count():
        log("no model-selector trigger; assuming loaded model is active")
        return
    await trig.first.click()
    short = model_path.split("/")[-1]
    search = page.locator('[data-model-picker-search-input]')
    if await search.count():
        await search.first.fill(short)
    opt = page.locator('[data-model-picker-option]', has_text=short)
    try:
        await opt.first.click(timeout=8000)
    except Exception:  # noqa: BLE001
        await page.keyboard.press("Escape")


async def send_turn(sp, out: Path, prompt: str, shot: str) -> None:
    page = sp.page
    box = page.get_by_role("textbox", name="Message input")
    await box.fill(prompt)
    await box.press("Enter")
    # approve tool confirmation if it appears (first turn); "Always allow" covers later turns
    try:
        allow = page.get_by_role("button", name="Always allow")
        await allow.wait_for(state="visible", timeout=20000)
        await allow.click()
        log("clicked 'Always allow'")
    except Exception:  # noqa: BLE001
        log("no tool-confirmation prompt (or already allowed)")
    # wait for streaming to finish
    stop = page.get_by_role("button", name="Stop generating")
    try:
        await stop.wait_for(state="hidden", timeout=180000)
    except Exception:  # noqa: BLE001
        pass
    await asyncio.sleep(1.5)
    await sp.screenshot(out / shot, full_page=True)


async def read_results(sp) -> list[tuple[int, int]]:
    page = sp.page
    # expand any collapsed tool cards
    trigs = page.locator('[data-slot="tool-fallback-trigger"]')
    for i in range(await trigs.count()):
        t = trigs.nth(i)
        try:
            if await t.get_attribute("data-state") == "closed":
                await t.click()
        except Exception:  # noqa: BLE001
            pass
    await asyncio.sleep(0.5)
    await asyncio.sleep(0.3)
    texts = await page.locator('[data-slot="tool-fallback-result"] pre').all_text_contents()
    if not texts:
        texts = await page.locator('.aui-tool-fallback-result-content').all_text_contents()
    out = []
    for t in texts:
        m = RESULT_RE.search(t or "")
        if m:
            out.append((int(m.group(1)), int(m.group(2))))
    return out


async def drive(args) -> int:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    a = await auth(args.base_url, args.password)
    await cleanup_mcp_servers(args.base_url, a.access_token)
    await ensure_model_loaded(args.base_url, a.access_token, args.model_path, args.gguf_variant)
    seed = seed_init_script(
        a, providers=[], connections_enabled=False,
        extra_local_storage={"unsloth_chat_mcp_enabled": "true"},
    )
    requests_seen: list[dict] = []
    result: dict = {"side": args.side, "model": args.model_path, "variant": args.gguf_variant}
    async with open_chat(
        args.base_url, init_scripts=[seed], video_dir=out / "video",
        video_name=args.side.lower(), transcode_mp4=True, viewport=(1440, 900), headless=True,
    ) as sp:
        def on_request(req):
            if req.url.endswith("/v1/chat/completions") and req.method == "POST":
                try:
                    b = req.post_data_json or {}
                    requests_seen.append({"thread_id": b.get("thread_id"), "session_id": b.get("session_id")})
                except Exception:  # noqa: BLE001
                    pass
        sp.page.on("request", on_request)
        try:
            await pick_model(sp, args.model_path)
            await configure_mcp(sp, out, args.counter_cmd)
            active = await sp.page.locator('button[data-pill-label="MCP"][data-active="true"]').count()
            result["mcp_active"] = bool(active)
            await sp.screenshot(out / "02_configured.png", full_page=True)
            await send_turn(sp, out, PROMPT_1, "03_turn1.png")
            await send_turn(sp, out, PROMPT_2, "04_turn2.png")
            pairs = await read_results(sp)
            # money shot: raw tool-result cards expanded (COUNTER=n; PID=p),
            # the unambiguous oracle (the model's prose can be wrong).
            await sp.screenshot(out / "06_results_expanded.png", full_page=True)
            result["results"] = pairs
            result["counters"] = [p[0] for p in pairs]
            result["pids"] = [p[1] for p in pairs]
            result["requests"] = requests_seen
            tids = [r.get("thread_id") for r in requests_seen if r.get("thread_id")]
            result["thread_ids"] = tids
            # isolation check on AFTER: new chat resets
            if args.side.upper() == "AFTER":
                try:
                    await sp.page.get_by_role("button", name="New Chat").click()
                    await asyncio.sleep(1.0)
                    await send_turn(sp, out, PROMPT_1, "05_newchat.png")
                    iso = await read_results(sp)
                    result["isolation_results"] = iso[-1:] if iso else []
                except Exception as e:  # noqa: BLE001
                    log(f"isolation step skipped: {e}")
        except Exception as e:  # noqa: BLE001
            result["error"] = repr(e)
            await dump_on_error(sp, out, "drive")
            (out / "assertions.json").write_text(json.dumps(result, indent=2))
            raise
    result["video_webm"] = str(sp.video_webm) if sp.video_webm else None
    result["video_mp4"] = str(sp.video_mp4) if sp.video_mp4 else None
    (out / "assertions.json").write_text(json.dumps(result, indent=2))
    log("results: " + json.dumps(result.get("results")))

    # fail-hard assertions
    pairs = result.get("results", [])
    assert len(pairs) == 2, f"expected exactly 2 tool results, got {len(pairs)}: {pairs}"
    c = result["counters"]; p = result["pids"]
    if args.side.upper() == "AFTER":
        assert c == [1, 2], f"AFTER expected counters [1,2], got {c}"
        assert p[0] == p[1], f"AFTER expected same PID, got {p}"
    else:
        assert c == [1, 1], f"BEFORE expected counters [1,1], got {c}"
        assert p[0] != p[1], f"BEFORE expected different PIDs, got {p}"
    if len(result.get("thread_ids", [])) >= 2:
        assert result["thread_ids"][0] == result["thread_ids"][1], "thread_id changed between turns"
    log(f"{args.side} PASS: counters={c} pids={p}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--password", required=True)
    ap.add_argument("--side", required=True, choices=["BEFORE", "AFTER"])
    ap.add_argument("--model-path", default="Qwen/Qwen2.5-3B-Instruct-GGUF")
    ap.add_argument("--gguf-variant", default="q4_k_m")
    ap.add_argument("--counter-cmd", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    return asyncio.run(drive(args))


if __name__ == "__main__":
    raise SystemExit(main())
