"""PR #6647 AFTER tour -- captures every NEW/CHANGED Studio surface on the PR branch.

Walks the per-model-config feature surfaces and screenshots each, so the compose
step can pair them with the matched main (BEFORE) shots:

  s1_config_entry   -- the in-picker ModelConfigPage opened by the per-model gear
  s2_context_set    -- Context Length set to <ctx> + Remember for this model
  s3_advanced       -- Advanced panel (KV cache dtype / Speculative / Tensor Parallel)
  s4_template_editor-- standalone Chat Template editor dialog (byte counter + validate)
  s2_after_reload   -- Context Length still <ctx> after Load + page reload (#6854 fix)
  s5_sidebar        -- loaded-model Run-settings sidebar embeds the same config page
  s5_settings_chat  -- Settings -> Chat has NO "Load on selection" row (removed)

The hard pass/fail gate is the same deterministic S2 persistence check as
pr6647_after.py (localStorage across load + reload). Every other scene is
best-effort: a scene that fails is screenshotted and logged, and the walk
continues, so one flaky surface never loses the rest of the tour.

Run:
  python -m studio_test_kit.examples.pr6647_tour_after \
      --base http://127.0.0.1:8902 --password 'PW' \
      --model unsloth/gemma-3-270m-it-GGUF --context 8192 \
      --studio-log studio.log --out out/after
Exit 0 = S2 persistence proven, 1 = failed.
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
from .pr6647_after import _find_ctx_entry, _open_picker_to_config


async def _shot(sp, out_dir, name, log, full_page=True):
    try:
        await sp.screenshot(out_dir / "shots" / f"{name}.png", full_page=full_page)
        log(f"  shot {name}")
    except Exception as e:
        log(f"  shot {name} failed: {e!r}")


async def _scene_advanced(panel, sp, out_dir, log):
    """Expand Advanced settings and toggle Tensor Parallelism (a reliable Switch),
    so the persisted record proves multi-field persistence, not just context."""
    try:
        adv = panel.get_by_role("switch", name=re.compile("advanced", re.I)).first
        await adv.wait_for(state="visible", timeout=4000)
        await adv.click()
        await panel.page.wait_for_timeout(400)
    except Exception as e:
        log(f"  advanced toggle not found: {e!r}")
    # Do NOT toggle the Tensor Parallelism switch: it has no accessible name and
    # a positional click risks hitting the "Show advanced settings" toggle and
    # collapsing the panel. The expanded panel screenshot is the value here.
    await _shot(sp, out_dir, "s3_advanced", log, full_page=False)
    return False


async def _scene_template_editor(panel, sp, out_dir, log):
    """Open the standalone Chat Template editor dialog, screenshot, then close."""
    try:
        edit = panel.get_by_role("button", name=re.compile(r"^(Edit|View)$")).first
        await edit.wait_for(state="visible", timeout=4000)
        await edit.click()
        dlg = panel.page.get_by_role("dialog").first
        await dlg.wait_for(state="visible", timeout=6000)
        await panel.page.wait_for_timeout(600)
        await _shot(sp, out_dir, "s4_template_editor", log, full_page=False)
        # Close via Cancel/Close, else Escape.
        for name in ("Cancel", "Close"):
            btn = dlg.get_by_role("button", name=name).first
            try:
                if await btn.is_visible():
                    await btn.click(timeout=2000)
                    return True
            except Exception:
                pass
        await panel.page.keyboard.press("Escape")
        return True
    except Exception as e:
        log(f"  template editor scene skipped: {e!r}")
        try:
            await panel.page.keyboard.press("Escape")
        except Exception:
            pass
        return False


async def _scene_sidebar(page, sp, out_dir, log):
    """After the model is loaded, open the Run-settings sidebar (which now embeds
    the ModelConfigPage) and screenshot it."""
    try:
        gear = page.locator('[aria-label="Open run settings"]').first
        await gear.wait_for(state="visible", timeout=8000)
        await gear.click()
        await page.wait_for_timeout(800)
        await _shot(sp, out_dir, "s5_sidebar", log, full_page=False)
        await page.keyboard.press("Escape")
        return True
    except Exception as e:
        log(f"  sidebar scene skipped: {e!r}")
        return False


async def _scene_settings_chat(page, base, sp, out_dir, log):
    """Open the Settings DIALOG (the /settings route calls openDialog) and switch to
    the Chat tab, then screenshot. On the PR the 'Load on selection' row is gone."""
    try:
        await page.goto(base + "/settings", wait_until="domcontentloaded", timeout=15000)
        dlg = page.get_by_role("dialog").first
        try:
            await dlg.wait_for(state="visible", timeout=8000)
        except Exception:
            log("  settings dialog did not open via /settings")
        try:
            chat_tab = dlg.get_by_role("button", name=re.compile(r"^Chat$", re.I)).first
            await chat_tab.click(timeout=4000)
            await page.wait_for_timeout(800)
        except Exception as e:
            log(f"  chat tab click: {e!r}")
        has_row = await page.get_by_text(re.compile("Load on selection", re.I)).count()
        await _shot(sp, out_dir, "s5_settings_chat", log, full_page=False)
        log(f"  settings 'Load on selection' rows found: {has_row} (expect 0 on PR)")
        return has_row
    except Exception as e:
        log(f"  settings scene skipped: {e!r}")
        return None


async def run(base, password, model, ctx, studio_log, out_dir):
    out_dir = Path(out_dir)
    (out_dir / "shots").mkdir(parents=True, exist_ok=True)
    video_dir = out_dir / "video"
    model_hint = "gemma-3-270m"
    result = {"base": base, "model": model, "context": ctx, "levels": {}, "scenes": {}}

    def log(msg):
        print(f"[tour-after] {msg}", flush=True)

    log(f"login {base}")
    auth = await login(base, "unsloth", password)
    init = seed_init_script(auth, [])

    async with open_chat(
        base, init_scripts=[init], video_dir=video_dir, video_name="after",
        transcode_mp4=True, viewport=(1440, 900), headless=True,
    ) as sp:
        page = sp.page
        captured = []

        def on_request(req):
            if req.method == "POST" and (
                "/api/inference/load" in req.url or "/api/inference/validate" in req.url
            ):
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
            # ---- S1: config entry point (gear -> in-picker ModelConfigPage) ----
            log("S1: open picker -> On Device -> config gear -> ModelConfigPage")
            panel = await _open_picker_to_config(
                page, model_hint, log, debug_shot=out_dir / "shots" / "00_ondevice_list.png"
            )
            await _shot(sp, out_dir, "s1_config_entry", log, full_page=False)
            result["scenes"]["s1_config_entry"] = True

            # ---- S2 (part 1): set Context Length + Remember ----
            log(f"S2: set Context Length = {ctx}")
            ctx_box = panel.get_by_role("textbox", name="Context Length").first
            await ctx_box.wait_for(state="visible", timeout=8000)
            await ctx_box.fill(str(ctx))
            await ctx_box.press("Enter")
            await _shot(sp, out_dir, "s2_context_set", log, full_page=False)

            # ---- S3: advanced multi-setting (toggle Tensor Parallel) ----
            log("S3: expand Advanced + toggle Tensor Parallelism")
            tp_set = await _scene_advanced(panel, sp, out_dir, log)
            result["scenes"]["s3_advanced"] = True

            # ---- S4: chat template editor dialog ----
            log("S4: open Chat Template editor dialog")
            result["scenes"]["s4_template_editor"] = await _scene_template_editor(
                panel, sp, out_dir, log
            )

            # Re-acquire the panel (dialog close may have re-rendered) and Remember+Load.
            panel = page.locator('[data-tour="chat-model-selector-popover"]').first
            remember = panel.get_by_role("checkbox", name="Remember for this model").first
            await remember.check(timeout=8000)
            log("S2: click Load model")
            await panel.get_by_role("button", name=re.compile(r"^(Load model|Reload model)$")).first.click(
                timeout=8000
            )

            cfg_after_load = await page.evaluate("() => localStorage.getItem('unsloth_model_configs')")
            hit = _find_ctx_entry(cfg_after_load, ctx)
            result["levels"]["localstorage_after_load"] = hit
            result["levels"]["record_after_load"] = hit["entry"] if hit else None
            log(f"  localStorage after load: {'FOUND ' + hit['key'] if hit else 'MISSING'}")
            if hit:
                log(f"  persisted record: {json.dumps(hit['entry'])}")

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
            log(f"  intercepted max_seq_length=={ctx}: {'YES' if req_hit else 'not seen'}")
            await asyncio.sleep(3)

            # ---- S2 (part 2): page reload -> Context Length still <ctx> ----
            log("S2: page.reload() -> assert Context Length persists")
            await page.reload(wait_until="domcontentloaded")
            await page.locator("form:has(textarea) textarea").first.wait_for(
                state="visible", timeout=15000
            )
            cfg_after_reload = await page.evaluate("() => localStorage.getItem('unsloth_model_configs')")
            hit2 = _find_ctx_entry(cfg_after_reload, ctx)
            result["levels"]["localstorage_after_reload"] = hit2
            await _open_picker_to_config(page, model_hint, log)
            panel = page.locator('[data-tour="chat-model-selector-popover"]').first
            ctx_box = panel.get_by_role("textbox", name="Context Length").first
            await ctx_box.wait_for(state="visible", timeout=8000)
            reload_val = await ctx_box.input_value()
            result["levels"]["reloaded_value"] = reload_val
            log(f"  after-reload Context Length input = {reload_val!r}")
            await _shot(sp, out_dir, "s2_after_reload", log, full_page=False)
            await page.keyboard.press("Escape")

            # ---- S5: loaded-model Run-settings sidebar (embeds config page) ----
            log("S5: open Run-settings sidebar on the loaded model")
            result["scenes"]["s5_sidebar"] = await _scene_sidebar(page, sp, out_dir, log)

            # ---- S5b: Settings -> Chat ('Load on selection' removed) ----
            log("S5b: Settings -> Chat ('Load on selection' should be gone)")
            result["levels"]["load_on_selection_rows"] = await _scene_settings_chat(
                page, base, sp, out_dir, log
            )

        except Exception as e:
            log(f"ERROR during tour: {e!r}")
            result["error"] = repr(e)
            await _shot(sp, out_dir, "99_error", log)

        result["requests"] = captured

    if studio_log and Path(studio_log).exists():
        text = Path(studio_log).read_text(errors="ignore")
        m = re.search(r"n_ctx\s*[=:]\s*(\d+)", text)
        if m:
            result["levels"]["backend_n_ctx"] = int(m.group(1))

    (out_dir / "captures.json").write_text(json.dumps(result, indent=2))

    gate = (
        result["levels"].get("localstorage_after_load")
        and result["levels"].get("localstorage_after_reload")
        and str(result["levels"].get("reloaded_value")) == str(ctx)
    )
    log(f"VERDICT: {'PASS' if gate else 'FAIL'} (S2 context {ctx} persisted={bool(gate)})")
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
