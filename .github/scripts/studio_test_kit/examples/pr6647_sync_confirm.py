"""PR #6647 picker <-> sidebar config SYNC confirmation (PR Studio only).

Proves the in-picker ModelConfigPage and the loaded-model Run-settings sidebar
stay in sync BOTH ways, because they are the same component fed by one shared
source of truth (`activeModelConfig`, computed once in chat-page.tsx and passed
to both surfaces) plus the versioned localStorage store.

Flow (single Studio, the PR branch):

  1. Picker edit  : open picker config, set Context Length = CTX_A (+ KV = KV_A),
                    Remember, Load the model.
  2. Sidebar read : open Run settings; assert the sidebar shows CTX_A (+ KV_A).
                    -> proves picker edits reflect into the sidebar.
  3. Sidebar edit : in the sidebar, set Context Length = CTX_B (+ KV = KV_B),
                    Reload.
  4. Picker read  : reopen the picker config; assert it shows CTX_B (+ KV_B).
                    -> proves sidebar edits reflect back into the picker.

Hard gate (deterministic): step 2 sidebar Context Length == CTX_A AND step 4
picker Context Length == CTX_B. KV Cache Dtype is driven + captured best-effort
(it is visible in the screenshots regardless). A per-step localStorage snapshot
of `unsloth_model_configs` is recorded as extra evidence.

Run:
  python -m studio_test_kit.examples.pr6647_sync_confirm \
      --base http://127.0.0.1:8902 --password 'PW' \
      --model unsloth/gemma-3-270m-it-GGUF --out out/sync
Exit 0 = bidirectional sync proven, 1 = failed.
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

CTX_A = 8192
CTX_B = 4096
KV_A = "q8_0"
KV_B = "q5_1"


def _as_int(value) -> int | None:
    if value is None:
        return None
    try:
        return int(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _page_of(scope):
    """Return the Page for a scope that may be either a Page or a Locator."""
    return getattr(scope, "page", scope)


async def _shot(sp, out_dir, name, log):
    try:
        await sp.screenshot(out_dir / "shots" / f"{name}.png", full_page=False)
        log(f"  shot {name}")
    except Exception as e:
        log(f"  shot {name} failed: {e!r}")


async def _expand_advanced(scope, log):
    """Expand the Advanced settings panel if it is collapsed (idempotent)."""
    try:
        adv = scope.get_by_role("switch", name=re.compile("advanced", re.I)).first
        await adv.wait_for(state="visible", timeout=4000)
        if (await adv.get_attribute("aria-checked")) != "true":
            await adv.click()
            await _page_of(scope).wait_for_timeout(300)
    except Exception as e:
        log(f"  advanced expand skipped: {e!r}")


async def _set_ctx(scope, value, log):
    box = scope.get_by_role("textbox", name="Context Length").first
    await box.wait_for(state="visible", timeout=8000)
    await box.fill(str(value))
    await box.press("Enter")
    await _page_of(scope).wait_for_timeout(250)


async def _read_ctx(scope, log):
    box = scope.get_by_role("textbox", name="Context Length").first
    await box.wait_for(state="visible", timeout=8000)
    return _as_int(await box.input_value())


async def _set_kv(page, scope, value, log):
    """Best-effort: set the KV Cache Dtype Select (first combobox in Advanced)."""
    try:
        combo = scope.get_by_role("combobox").first
        await combo.wait_for(state="visible", timeout=4000)
        await combo.click()
        opt = page.get_by_role("option", name=value, exact=True).first
        await opt.wait_for(state="visible", timeout=4000)
        await opt.click()
        await page.wait_for_timeout(250)
        return True
    except Exception as e:
        log(f"  KV set ({value}) skipped: {e!r}")
        try:
            await page.keyboard.press("Escape")
        except Exception:
            pass
        return False


async def _read_kv(scope, log):
    try:
        combo = scope.get_by_role("combobox").first
        await combo.wait_for(state="visible", timeout=3000)
        txt = (await combo.inner_text()).strip()
        return txt or None
    except Exception as e:
        log(f"  KV read skipped: {e!r}")
        return None


async def _ls_config(page):
    return await page.evaluate("() => localStorage.getItem('unsloth_model_configs')")


async def _open_sidebar_config(page, log, timeout_ms=200000):
    """Open the Run-settings sheet and wait for its ModelConfigPage to mount.

    The sheet only renders the config once the model is loaded
    (`activeModelConfig && !modelLoading`), so waiting on the sidebar's
    Context Length textbox doubles as a 'model finished loading' signal.
    """
    # Make sure no picker popover is open (would own a second Context Length box).
    try:
        await page.keyboard.press("Escape")
    except Exception:
        pass
    await page.wait_for_timeout(300)
    gear = page.locator('[aria-label="Open run settings"]').first
    await gear.wait_for(state="visible", timeout=15000)
    await gear.click()
    # Wait (generously) for the loaded-model config to appear inside the sheet.
    box = page.get_by_role("textbox", name="Context Length").first
    await box.wait_for(state="visible", timeout=timeout_ms)
    await page.wait_for_timeout(400)


async def _close_sidebar(page, log):
    """Collapse the Run-settings sheet so the picker trigger is reachable."""
    for name in ("Close run settings", "Collapse run settings", "Close"):
        try:
            btn = page.get_by_role("button", name=re.compile(name, re.I)).first
            if await btn.is_visible():
                await btn.click(timeout=2000)
                await page.wait_for_timeout(300)
                return
        except Exception:
            pass
    try:
        await page.keyboard.press("Escape")
        await page.wait_for_timeout(300)
    except Exception:
        pass


async def run(base, password, model, out_dir):
    out_dir = Path(out_dir)
    (out_dir / "shots").mkdir(parents=True, exist_ok=True)
    video_dir = out_dir / "video"
    model_hint = "gemma-3-270m"
    result = {
        "base": base, "model": model,
        "ctx_a": CTX_A, "ctx_b": CTX_B, "kv_a": KV_A, "kv_b": KV_B,
        "levels": {}, "steps": {},
    }

    def log(msg):
        print(f"[sync] {msg}", flush=True)

    log(f"login {base}")
    auth = await login(base, "unsloth", password)
    init = seed_init_script(auth, [])

    async with open_chat(
        base, init_scripts=[init], video_dir=video_dir, video_name="sync",
        transcode_mp4=True, viewport=(1440, 900), headless=True,
    ) as sp:
        page = sp.page
        captured = []

        def on_request(req):
            if req.method == "POST" and "/api/inference/load" in req.url:
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
            # ---- 1. PICKER EDIT: set ctx=CTX_A (+ KV_A), Remember, Load ----
            log(f"STEP 1 picker: set Context Length={CTX_A}, KV={KV_A}, Remember, Load")
            panel = await _open_picker_to_config(
                page, model_hint, log,
                debug_shot=out_dir / "shots" / "00_ondevice_list.png",
            )
            await _set_ctx(panel, CTX_A, log)
            await _expand_advanced(panel, log)
            result["steps"]["picker_set_kv_a"] = await _set_kv(page, panel, KV_A, log)
            remember = panel.get_by_role("checkbox", name="Remember for this model").first
            await remember.check(timeout=8000)
            await _shot(sp, out_dir, "01_picker_edit", log)

            await panel.get_by_role("button", name="Load model").first.click(timeout=8000)
            cfg1 = await _ls_config(page)
            result["levels"]["ls_after_picker_load"] = _find_ctx_entry(cfg1, CTX_A)
            log(f"  localStorage after picker Load: "
                f"{'FOUND ' + result['levels']['ls_after_picker_load']['key'] if result['levels']['ls_after_picker_load'] else 'MISSING'}")

            # ---- 2. SIDEBAR READ: assert it reflects the picker edit ----
            log("STEP 2 sidebar: open Run settings, assert it reflects the picker edit")
            await _open_sidebar_config(page, log)
            sidebar = page.get_by_role("textbox", name="Context Length").first
            sidebar_ctx = _as_int(await sidebar.input_value())
            result["levels"]["sidebar_ctx_after_picker_edit"] = sidebar_ctx
            result["levels"]["sidebar_kv_after_picker_edit"] = await _read_kv(page, log)
            log(f"  sidebar Context Length = {sidebar_ctx!r} (expect {CTX_A}); "
                f"KV = {result['levels']['sidebar_kv_after_picker_edit']!r}")
            await _shot(sp, out_dir, "02_sidebar_reflects_picker", log)

            # ---- 3. SIDEBAR EDIT: set ctx=CTX_B (+ KV_B), Reload ----
            log(f"STEP 3 sidebar: set Context Length={CTX_B}, KV={KV_B}, Reload")
            await _set_ctx(page, CTX_B, log)
            await _expand_advanced(page, log)
            result["steps"]["sidebar_set_kv_b"] = await _set_kv(page, page, KV_B, log)
            await _shot(sp, out_dir, "03_sidebar_edit", log)
            await page.get_by_role(
                "button", name=re.compile(r"^(Reload model|Load model)$")
            ).first.click(timeout=8000)
            await page.wait_for_timeout(500)
            cfg2 = await _ls_config(page)
            result["levels"]["ls_after_sidebar_reload"] = _find_ctx_entry(cfg2, CTX_B)
            log(f"  localStorage after sidebar Reload: "
                f"{'FOUND ' + result['levels']['ls_after_sidebar_reload']['key'] if result['levels']['ls_after_sidebar_reload'] else 'MISSING'}")

            # ---- 4. PICKER READ: reopen picker config, assert it reflects sidebar ----
            log("STEP 4 picker: reopen config, assert it reflects the sidebar edit")
            # Wait for the reload to actually dispatch with the new context so the
            # runtime (which the active-model picker reads via activeModelConfig)
            # has a chance to reach CTX_B before we poll.
            for _ in range(60):
                if any(
                    isinstance((c.get("body") or {}), dict)
                    and (c["body"] or {}).get("max_seq_length") == CTX_B
                    for c in captured
                ):
                    log("  reload dispatched with max_seq_length=4096")
                    break
                await asyncio.sleep(1.5)
            await _close_sidebar(page, log)
            # The reload may still be settling; the picker seeds from the store
            # (saved synchronously) or the runtime, so retry a few times.
            picker_ctx = None
            for attempt in range(18):
                try:
                    await _open_picker_to_config(page, model_hint, log)
                    p2 = page.locator('[data-tour="chat-model-selector-popover"]').first
                    picker_ctx = await _read_ctx(p2, log)
                    if picker_ctx == CTX_B:
                        break
                    await page.keyboard.press("Escape")
                    await page.wait_for_timeout(2500)
                except Exception as e:
                    log(f"  picker reopen attempt {attempt} failed: {e!r}")
                    await page.wait_for_timeout(2500)
            result["levels"]["picker_ctx_after_sidebar_edit"] = picker_ctx
            p2 = page.locator('[data-tour="chat-model-selector-popover"]').first
            result["levels"]["picker_kv_after_sidebar_edit"] = await _read_kv(p2, log)
            log(f"  picker Context Length = {picker_ctx!r} (expect {CTX_B}); "
                f"KV = {result['levels']['picker_kv_after_sidebar_edit']!r}")
            await _shot(sp, out_dir, "04_picker_reflects_sidebar", log)

        except Exception as e:
            log(f"ERROR during flow: {e!r}")
            result["error"] = repr(e)
            await _shot(sp, out_dir, "99_error", log)

        result["requests"] = captured

    (out_dir / "captures.json").write_text(json.dumps(result, indent=2))

    fwd = result["levels"].get("sidebar_ctx_after_picker_edit") == CTX_A
    back = result["levels"].get("picker_ctx_after_sidebar_edit") == CTX_B
    result["levels"]["picker_to_sidebar_ok"] = fwd
    result["levels"]["sidebar_to_picker_ok"] = back
    (out_dir / "captures.json").write_text(json.dumps(result, indent=2))
    log(f"  picker -> sidebar ({CTX_A}): {'PASS' if fwd else 'FAIL'}")
    log(f"  sidebar -> picker ({CTX_B}): {'PASS' if back else 'FAIL'}")
    log(f"VERDICT: {'PASS' if (fwd and back) else 'FAIL'} (bidirectional sync)")
    return 0 if (fwd and back) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--password", required=True)
    ap.add_argument("--model", default="unsloth/gemma-3-270m-it-GGUF")
    ap.add_argument("--out", default="out/sync")
    a = ap.parse_args()
    rc = asyncio.run(run(a.base, a.password, a.model, a.out))
    sys.exit(rc)


if __name__ == "__main__":
    main()
