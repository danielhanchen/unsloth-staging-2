"""PR #6647 BEFORE tour -- captures the same surfaces on origin/main for contrast.

Matched to pr6647_tour_after.py, on main's OLD sidebar Run-settings experience:

  s1_config_entry   -- picking/staging a model opens the right-hand Run-settings sidebar
  s2_context_set    -- Context Length set to <ctx> in the inline sidebar
  s3_advanced       -- the same sidebar's inline KV cache / Speculative / Tensor Parallel
  s4_template_editor-- main's inline "Edit chat template" dialog
  s2_after_reload   -- reload + re-stage: Context Length reverts to native (the #6854 loss)
  s5_sidebar        -- the loaded-model inline Run-settings sidebar
  s5_settings_chat  -- Settings -> Chat still HAS a "Load on selection" row

Hard gate = same as pr6647_before.py: main has NO versioned unsloth_model_configs
store. Every scene is best-effort (screenshot on failure, keep going).

Run:
  python -m studio_test_kit.examples.pr6647_tour_before \
      --base http://127.0.0.1:8901 --password 'PW' \
      --model unsloth/gemma-3-270m-it-GGUF --context 8192 --out out/before
Exit 0 = no per-model persistence on main (as expected), 1 = unexpected.
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
from .pr6647_before import _load_gemma, _open_run_settings, _unload_all


async def _shot(page, out_dir, name, log, full_page=True):
    try:
        await page.screenshot(path=str(out_dir / "shots" / f"{name}.png"), full_page=full_page)
        log(f"  shot {name}")
    except Exception as e:
        log(f"  shot {name} failed: {e!r}")


async def _scene_template_editor(page, out_dir, log):
    """main: the inline 'Edit chat template' button opens a Dialog 'Edit Chat Template'.
    The button sits low in the long Run-settings sheet, so scroll it into view first."""
    try:
        # main opens the editor from a "Chat Template" text button or an
        # aria-label="Edit chat template" icon button (both call openEditor).
        btn = page.get_by_role("button", name=re.compile(r"^Chat Template$")).first
        if await btn.count() == 0:
            btn = page.get_by_label(re.compile("Edit chat template", re.I)).first
        await btn.scroll_into_view_if_needed(timeout=5000)
        await btn.click(timeout=5000)
        dlg = page.get_by_role("dialog").first
        await dlg.wait_for(state="visible", timeout=6000)
        await page.wait_for_timeout(600)
        await _shot(page, out_dir, "s4_template_editor", log, full_page=False)
        await page.keyboard.press("Escape")
        return True
    except Exception as e:
        log(f"  template editor scene skipped: {e!r}")
        try:
            await page.keyboard.press("Escape")
        except Exception:
            pass
        return False


async def _scene_settings_chat(page, base, out_dir, log):
    """Open the Settings DIALOG (the /settings route calls openDialog) and switch to
    the Chat tab, then screenshot. On main this tab has a 'Load on selection' row."""
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
        rows = await page.get_by_text(re.compile("Load on selection", re.I)).count()
        await _shot(page, out_dir, "s5_settings_chat", log, full_page=False)
        log(f"  settings 'Load on selection' rows found: {rows} (expect >=1 on main)")
        return rows
    except Exception as e:
        log(f"  settings scene skipped: {e!r}")
        return None


async def run(base, password, model, ctx, out_dir):
    out_dir = Path(out_dir)
    (out_dir / "shots").mkdir(parents=True, exist_ok=True)
    video_dir = out_dir / "video"
    model_hint = "gemma-3-270m"
    result = {"base": base, "model": model, "context": ctx, "levels": {}, "scenes": {}}

    def log(msg):
        print(f"[tour-before] {msg}", flush=True)

    log(f"login {base}")
    auth = await login(base, "unsloth", password)
    await _unload_all(base, auth.access_token, log)
    init = seed_init_script(auth, [], extra_local_storage={"unsloth_chat_load_on_selection": "false"})

    async with open_chat(
        base, init_scripts=[init], video_dir=video_dir, video_name="before",
        transcode_mp4=True, viewport=(1440, 900), headless=True,
    ) as sp:
        page = sp.page
        try:
            pre = await page.evaluate("() => localStorage.getItem('unsloth_model_configs')")
            result["levels"]["unsloth_model_configs_present"] = pre is not None

            # ---- S1: stage the model -> Run-settings sidebar opens ----
            log("S1: stage gemma -> Run-settings sidebar")
            await _load_gemma(page, model_hint, log, out_dir)
            panel = await _open_run_settings(page, log)
            await _shot(page, out_dir, "s1_config_entry", log, full_page=False)
            result["scenes"]["s1_config_entry"] = True

            # ---- S2 (part 1): set Context Length in the sidebar ----
            log(f"S2: set Context Length = {ctx}")
            ctx_box = panel.get_by_label("Context Length").first
            await ctx_box.wait_for(state="visible", timeout=8000)
            await ctx_box.fill(str(ctx))
            await ctx_box.press("Enter")
            await page.wait_for_timeout(400)
            await _shot(page, out_dir, "s2_context_set", log, full_page=False)

            cfg = await page.evaluate("() => localStorage.getItem('unsloth_model_configs')")
            load_settings = await page.evaluate("() => localStorage.getItem('unsloth_load_settings')")
            result["levels"]["unsloth_model_configs_after_set"] = cfg
            result["levels"]["unsloth_load_settings_after_set"] = load_settings

            # ---- S3: the sidebar's inline advanced controls ----
            log("S3: inline KV cache / Speculative / Tensor Parallel in the sidebar")
            try:
                tp = panel.get_by_role("switch", name=re.compile("Tensor Parallel", re.I)).first
                if await tp.is_visible():
                    await tp.click(timeout=2000)
            except Exception as e:
                log(f"  tensor-parallel switch not toggled: {e!r}")
            await _shot(page, out_dir, "s3_advanced", log, full_page=False)
            result["scenes"]["s3_advanced"] = True

            # ---- S4: main's inline chat-template editor dialog ----
            log("S4: inline Edit chat template dialog")
            result["scenes"]["s4_template_editor"] = await _scene_template_editor(page, out_dir, log)

            # ---- S5 sidebar (main's inline sidebar is the config surface) ----
            await _open_run_settings(page, log)
            await _shot(page, out_dir, "s5_sidebar", log, full_page=False)
            result["scenes"]["s5_sidebar"] = True

            # ---- S2 (part 2): reload + re-stage -> Context Length reverts ----
            log("S2: page.reload() -> re-stage -> Context Length reverts to native")
            await page.reload(wait_until="domcontentloaded")
            await page.locator("form:has(textarea) textarea").first.wait_for(
                state="visible", timeout=15000
            )
            cfg2 = await page.evaluate("() => localStorage.getItem('unsloth_model_configs')")
            result["levels"]["unsloth_model_configs_after_reload"] = cfg2
            reopened_val = None
            try:
                await _load_gemma(page, model_hint, log, out_dir)
                panel = await _open_run_settings(page, log)
                ctx_box = panel.get_by_label("Context Length").first
                await ctx_box.wait_for(state="visible", timeout=6000)
                reopened_val = await ctx_box.input_value()
            except Exception as e:
                log(f"  re-stage after reload failed: {e!r}")
            result["levels"]["reopened_value_after_reload"] = reopened_val
            log(f"  Context Length after reload+re-stage = {reopened_val!r}")
            await _shot(page, out_dir, "s2_after_reload", log, full_page=False)

            # ---- S5b: Settings -> Chat still HAS 'Load on selection' ----
            log("S5b: Settings -> Chat ('Load on selection' still present on main)")
            result["levels"]["load_on_selection_rows"] = await _scene_settings_chat(
                page, base, out_dir, log
            )

        except Exception as e:
            log(f"ERROR during tour: {e!r}")
            result["error"] = repr(e)
            await _shot(page, out_dir, "99_error", log)

    (out_dir / "captures.json").write_text(json.dumps(result, indent=2))

    no_versioned_store = (
        not result["levels"].get("unsloth_model_configs_after_set")
        and not result["levels"].get("unsloth_model_configs_after_reload")
    )
    result["levels"]["no_versioned_store"] = no_versioned_store
    log(f"  no versioned unsloth_model_configs store: {no_versioned_store}")
    log(f"VERDICT: {'PASS (no per-model persistence on main)' if no_versioned_store else 'UNEXPECTED'}")
    return 0 if no_versioned_store else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--password", required=True)
    ap.add_argument("--model", default="unsloth/gemma-3-270m-it-GGUF")
    ap.add_argument("--context", type=int, default=8192)
    ap.add_argument("--out", default="out/before")
    a = ap.parse_args()
    rc = asyncio.run(run(a.base, a.password, a.model, a.context, a.out))
    sys.exit(rc)


if __name__ == "__main__":
    main()
