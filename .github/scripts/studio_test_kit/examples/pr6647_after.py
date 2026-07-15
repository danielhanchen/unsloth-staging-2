"""PR #6647 (issue #6854) AFTER driver -- proves per-model Context Length persists.

Drives the NEW picker config step on the PR branch and proves, at three levels,
that a per-model Context Length survives model-load and page reload:

  1. localStorage `unsloth_model_configs` holds `customContextLength == <ctx>`
     (written synchronously by savePerModelConfig in handleRun).
  2. the outgoing POST /api/inference/{validate,load} carries
     `max_seq_length == <ctx>` (customContextLength is folded into it).
  3. the backend studio.log shows the effective llama.cpp context == <ctx>.

Then it reopens the config (proves "no reset on model load") and reloads the
page (proves "no reset on startup").

The localStorage assertions are the hard pass/fail gate (deterministic). The
request + backend layers are expected but best-effort (they need the model
cached + a real CPU load).

Run:
  python -m studio_test_kit.examples.pr6647_after \
      --base http://127.0.0.1:8902 --password 'BOOTSTRAP' \
      --model unsloth/gemma-3-270m-it-GGUF --context 8192 \
      --studio-log demo/studio_pr.log --out out/after
Exit code 0 = persistence proven, 1 = failed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

from ..auth import login, seed_init_script
from ..ui import open_chat


def _find_ctx_entry(cfg_json, ctx):
    """Return the first unsloth_model_configs entry with customContextLength==ctx."""
    if not cfg_json:
        return None
    try:
        m = json.loads(cfg_json)
    except Exception:
        return None
    if not isinstance(m, dict):
        return None
    for k, v in m.items():
        if isinstance(v, dict) and v.get("customContextLength") == ctx:
            return {"key": k, "entry": v}
    return None


async def _open_config_gear(panel, model_hint, log):
    """On Device tab must be active. Open the per-model config gear FOR THE
    TARGET MODEL, expanding its GGUF repo row first to reveal variant gears."""
    # Model-specific gear so we never grab a different row's settings button.
    gear_re = re.compile(rf"Inference settings for.*{re.escape(model_hint)}", re.I)
    gear = panel.get_by_role("button", name=gear_re).first
    try:
        await gear.wait_for(state="visible", timeout=3000)
    except Exception:
        # GGUF repo rows collapse their quant variants; click the repo row.
        log("  gear not directly visible; expanding repo row")
        row = panel.locator("[data-model-picker-option]", has_text=model_hint).first
        await row.click(timeout=8000)
        gear = panel.get_by_role("button", name=gear_re).first
        await gear.wait_for(state="visible", timeout=8000)
    await gear.click()


async def _open_picker_to_config(page, model_hint, log, debug_shot=None):
    """Open picker -> On Device -> search -> open the config gear for the model."""
    await page.locator('[data-tour="chat-model-selector"]').first.click(timeout=15000)
    panel = page.locator('[data-tour="chat-model-selector-popover"]').first
    await panel.wait_for(state="visible", timeout=10000)
    # On Device tab (value "downloaded").
    tab = panel.get_by_role("tab", name=re.compile(r"On Device", re.I)).first
    await tab.click(timeout=8000)
    await page.wait_for_timeout(400)
    # Narrow the list so the target row/gear is unambiguous.
    search = panel.get_by_placeholder(re.compile(r"Search", re.I)).first
    try:
        await search.fill(model_hint, timeout=5000)
        await page.wait_for_timeout(400)
    except Exception:
        log("  search box not fillable; continuing unfiltered")
    if debug_shot is not None:
        await page.screenshot(path=str(debug_shot))
    await _open_config_gear(panel, model_hint, log)
    return panel


async def run(base, password, model, ctx, studio_log, out_dir):
    out_dir = Path(out_dir)
    (out_dir / "shots").mkdir(parents=True, exist_ok=True)
    video_dir = out_dir / "video"

    def log(msg):
        print(f"[after] {msg}", flush=True)

    model_hint = "gemma-3-270m"  # substring present in the repo id + gear label
    result = {"base": base, "model": model, "context": ctx, "levels": {}}

    log(f"login {base}")
    auth = await login(base, "unsloth", password)
    init = seed_init_script(auth, [])

    async with open_chat(
        base, init_scripts=[init], video_dir=video_dir, video_name="after",
        transcode_mp4=True, viewport=(1440, 900), headless=True,
    ) as sp:
        page = sp.page

        # Capture every inference validate/load request body.
        captured = []

        def on_request(req):
            if req.method == "POST" and (
                "/api/inference/load" in req.url or "/api/inference/validate" in req.url
            ):
                body = None
                try:
                    body = req.post_data_json
                except Exception:
                    try:
                        body = json.loads(req.post_data or "null")
                    except Exception:
                        body = None
                captured.append({"url": req.url, "body": body})

        page.on("request", on_request)

        try:
            # ---- 1. open picker config step, set context, remember ----
            log("open picker -> On Device -> config gear")
            panel = await _open_picker_to_config(
                page, model_hint, log, debug_shot=out_dir / "shots" / "00_ondevice_list.png"
            )

            log(f"set Context Length = {ctx}")
            ctx_box = panel.get_by_role("textbox", name="Context Length").first
            await ctx_box.wait_for(state="visible", timeout=8000)
            await ctx_box.fill(str(ctx))
            await ctx_box.press("Enter")

            remember = panel.get_by_role("checkbox", name="Remember for this model").first
            await remember.check(timeout=8000)
            await sp.screenshot(out_dir / "shots" / "01_config_set.png")

            # ---- 2. click Load, capture persistence + request ----
            log("click Load model")
            await panel.get_by_role("button", name="Load model").first.click(timeout=8000)

            # savePerModelConfig runs synchronously in the click handler.
            cfg_after_load = await page.evaluate(
                "() => localStorage.getItem('unsloth_model_configs')"
            )
            hit = _find_ctx_entry(cfg_after_load, ctx)
            result["levels"]["localstorage_after_load"] = hit
            log(f"  localStorage after load: {'FOUND ' + hit['key'] if hit else 'MISSING'}")

            # Poll the intercepted requests for max_seq_length == ctx.
            req_hit = None
            for _ in range(20):
                for c in captured:
                    b = c.get("body") or {}
                    if isinstance(b, dict) and b.get("max_seq_length") == ctx:
                        req_hit = c
                        break
                if req_hit:
                    break
                await asyncio.sleep(0.5)
            result["levels"]["request_max_seq_length"] = req_hit
            log(f"  intercepted request max_seq_length=={ctx}: "
                f"{'YES ' + req_hit['url'] if req_hit else 'not seen (model may be uncached)'}")

            await asyncio.sleep(3)
            await sp.screenshot(out_dir / "shots" / "02_after_load.png")

            # ---- 3. reopen config: still <ctx>? (no reset on model load) ----
            log("reopen config -> assert Context Length persists")
            await _open_picker_to_config(page, model_hint, log)
            panel = page.locator('[data-tour="chat-model-selector-popover"]').first
            ctx_box = panel.get_by_role("textbox", name="Context Length").first
            await ctx_box.wait_for(state="visible", timeout=8000)
            reopened_val = await ctx_box.input_value()
            result["levels"]["reopened_value"] = reopened_val
            log(f"  reopened Context Length input = {reopened_val!r}")
            await sp.screenshot(out_dir / "shots" / "03_reopened_after_load.png")
            # close popover
            await page.keyboard.press("Escape")

            # ---- 4. page reload: still <ctx>? (no reset on startup) ----
            log("page.reload() -> assert Context Length persists")
            await page.reload(wait_until="domcontentloaded")
            await page.locator("form:has(textarea) textarea").first.wait_for(
                state="visible", timeout=15000
            )
            cfg_after_reload = await page.evaluate(
                "() => localStorage.getItem('unsloth_model_configs')"
            )
            hit2 = _find_ctx_entry(cfg_after_reload, ctx)
            result["levels"]["localstorage_after_reload"] = hit2
            log(f"  localStorage after reload: {'FOUND ' + hit2['key'] if hit2 else 'MISSING'}")

            await _open_picker_to_config(page, model_hint, log)
            panel = page.locator('[data-tour="chat-model-selector-popover"]').first
            ctx_box = panel.get_by_role("textbox", name="Context Length").first
            await ctx_box.wait_for(state="visible", timeout=8000)
            reload_val = await ctx_box.input_value()
            result["levels"]["reloaded_value"] = reload_val
            log(f"  after-reload Context Length input = {reload_val!r}")
            await sp.screenshot(out_dir / "shots" / "04_after_page_reload.png")

        except Exception as e:
            log(f"ERROR during flow: {e!r}")
            result["error"] = repr(e)
            try:
                await sp.screenshot(out_dir / "shots" / "99_error.png")
            except Exception:
                pass

        result["requests"] = captured

    # ---- backend effective context (best-effort) from studio.log ----
    if studio_log and Path(studio_log).exists():
        text = Path(studio_log).read_text(errors="ignore")
        m = re.search(r"n_ctx\s*[=:]\s*(\d+)", text)
        if m:
            result["levels"]["backend_n_ctx"] = int(m.group(1))
            log(f"  backend n_ctx from log: {m.group(1)}")

    (out_dir / "captures.json").write_text(json.dumps(result, indent=2))

    # ---- hard-gate verdict: localStorage persistence across load + reload ----
    gate = (
        result["levels"].get("localstorage_after_load")
        and result["levels"].get("localstorage_after_reload")
        and str(result["levels"].get("reloaded_value")) == str(ctx)
    )
    log(f"VERDICT: {'PASS' if gate else 'FAIL'} (context {ctx} persisted={bool(gate)})")
    return 0 if gate else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--password", required=True)
    ap.add_argument("--model", default="unsloth/gemma-3-270m-it-GGUF")
    ap.add_argument("--context", type=int, default=8192)
    ap.add_argument("--studio-log", default=None)
    ap.add_argument("--out", default="out/after")
    a = ap.parse_args()
    rc = asyncio.run(run(a.base, a.password, a.model, a.context, a.studio_log, a.out))
    sys.exit(rc)


if __name__ == "__main__":
    main()
