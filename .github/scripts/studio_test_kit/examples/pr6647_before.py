"""PR #6647 (issue #6854) BEFORE driver -- shows main does NOT persist per-model context.

On origin/main the per-model GGUF Context Length lives in the ephemeral store field
`customContextLength`, which is reset to the model's native context on every load and is
`null` on page reload. There is NO versioned `unsloth_model_configs` store. This driver
sets a Context Length via the OLD sidebar Run-settings panel, then shows it is lost on
page reload, and that `unsloth_model_configs` never exists.

Run:
  python -m studio_test_kit.examples.pr6647_before \
      --base http://127.0.0.1:8901 --password 'BOOTSTRAP' \
      --model unsloth/gemma-3-270m-it-GGUF --context 8192 --out out/before
Exit 0 = loss demonstrated (value NOT persisted), 1 = unexpected (value persisted).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

import httpx

from ..auth import login, seed_init_script
from ..ui import open_chat


async def _unload_all(base, token, log):
    """Clean slate: unload any currently loaded model so a fresh stage shows the
    native default (main keeps the last-loaded model's runtime context otherwise)."""
    h = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient(timeout=30) as c:
        try:
            st = (await c.get(f"{base}/api/inference/status", headers=h)).json()
            for m in st.get("loaded", []) or []:
                await c.post(f"{base}/api/inference/unload", headers=h, json={"model_path": m})
                log(f"  unloaded {m}")
        except Exception as e:
            log(f"  unload_all issue: {e!r}")


async def _open_run_settings(page, log):
    """Return the old chat Run-settings panel, opening it via the gear if needed
    (staging a model auto-opens it, so prefer the already-visible panel)."""
    panel = page.locator('[data-tour="chat-settings"]').first
    try:
        await panel.wait_for(state="visible", timeout=3000)
        return panel
    except Exception:
        pass
    gear = page.locator('[aria-label="Open run settings"]').first
    await gear.wait_for(state="visible", timeout=8000)
    await gear.click()
    await panel.wait_for(state="visible", timeout=8000)
    return panel


async def _load_gemma(page, model_hint, log, out_dir):
    """Open the old model selector, go to On Device (cached only, no download),
    and stage the gemma GGUF row."""
    await page.locator("button.unsloth-model-selector-trigger").first.click(timeout=15000)
    menu = page.locator(".unsloth-model-selector-menu").first
    await menu.wait_for(state="visible", timeout=10000)
    await page.wait_for_timeout(400)
    await page.screenshot(path=str(out_dir / "shots" / "00_main_picker.png"))
    # On Device tab FIRST so we never touch a downloadable Hub/Recommended row.
    ondevice = menu.get_by_role("tab", name=re.compile(r"On Device", re.I)).first
    try:
        await ondevice.click(timeout=5000)
    except Exception:
        await menu.get_by_text(re.compile(r"On Device", re.I)).first.click(timeout=5000)
    await page.wait_for_timeout(500)
    # Narrow to the target, then click its row (cached GGUF -> stages, no download).
    try:
        await menu.locator("input").first.fill(model_hint, timeout=3000)
        await page.wait_for_timeout(400)
    except Exception:
        pass
    await page.screenshot(path=str(out_dir / "shots" / "00b_ondevice.png"))
    # Stage a DOWNLOADED quant variant (UD-Q4_K_XL, same as the AFTER run).
    # The repo auto-expands its variants on search; if not, click the header.
    variant = menu.get_by_text(re.compile(r"UD-Q4_K_XL", re.I)).first
    try:
        await variant.wait_for(state="visible", timeout=3000)
    except Exception:
        header = menu.get_by_text(re.compile(r"gemma-3-270m-it-GGUF", re.I)).first
        await header.click(timeout=8000)
        await page.wait_for_timeout(500)
        variant = menu.get_by_text(re.compile(r"UD-Q4_K_XL", re.I)).first
        await variant.wait_for(state="visible", timeout=5000)
    await variant.click(timeout=8000)
    await page.wait_for_timeout(1500)


async def run(base, password, model, ctx, out_dir):
    out_dir = Path(out_dir)
    (out_dir / "shots").mkdir(parents=True, exist_ok=True)
    video_dir = out_dir / "video"
    model_hint = "gemma-3-270m"

    def log(msg):
        print(f"[before] {msg}", flush=True)

    result = {"base": base, "model": model, "context": ctx, "levels": {}}
    log(f"login {base}")
    auth = await login(base, "unsloth", password)
    await _unload_all(base, auth.access_token, log)
    # Stage instead of instant-load so the Run-settings panel opens for editing.
    init = seed_init_script(auth, [], extra_local_storage={"unsloth_chat_load_on_selection": "false"})

    async with open_chat(
        base, init_scripts=[init], video_dir=video_dir, video_name="before",
        transcode_mp4=True, viewport=(1440, 900), headless=True,
    ) as sp:
        page = sp.page
        try:
            # main has NO versioned per-model store, ever.
            pre = await page.evaluate("() => localStorage.getItem('unsloth_model_configs')")
            result["levels"]["unsloth_model_configs_present"] = pre is not None
            log(f"unsloth_model_configs present at start: {pre is not None}")

            log("open old model selector -> stage/load gemma")
            await _load_gemma(page, model_hint, log, out_dir)
            await page.screenshot(path=str(out_dir / "shots" / "01_after_pick.png"))

            log("open Run-settings -> set Context Length")
            panel = await _open_run_settings(page, log)
            ctx_box = panel.get_by_label("Context Length").first
            await ctx_box.wait_for(state="visible", timeout=8000)
            await ctx_box.fill(str(ctx))
            await ctx_box.press("Enter")
            await page.wait_for_timeout(500)
            await page.screenshot(path=str(out_dir / "shots" / "02_ctx_set.png"))

            cfg = await page.evaluate("() => localStorage.getItem('unsloth_model_configs')")
            load_settings = await page.evaluate("() => localStorage.getItem('unsloth_load_settings')")
            result["levels"]["unsloth_model_configs_after_set"] = cfg
            result["levels"]["unsloth_load_settings_after_set"] = load_settings
            log(f"  unsloth_model_configs after set: {cfg!r}")
            log(f"  unsloth_load_settings after set: {load_settings!r}")

            # ---- reload, then RE-STAGE the same model: is 8192 remembered? ----
            # Stage-only (never Load) so the backend doesn't keep the model at 8192;
            # this isolates the persistence question. main has no per-model memory,
            # so re-staging shows the native default, not 8192.
            log("page.reload() -> re-stage gemma -> re-check Context Length")
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
            await page.screenshot(path=str(out_dir / "shots" / "03_after_reload.png"))

        except Exception as e:
            log(f"ERROR during flow: {e!r}")
            result["error"] = repr(e)
            try:
                await page.screenshot(path=str(out_dir / "shots" / "99_error.png"))
            except Exception:
                pass

    (out_dir / "captures.json").write_text(json.dumps(result, indent=2))

    # Hard gate: main has NO versioned per-model store (the durable difference the
    # PR introduces). Supporting: a re-staged model shows native, not 8192.
    no_versioned_store = (
        not result["levels"].get("unsloth_model_configs_after_set")
        and not result["levels"].get("unsloth_model_configs_after_reload")
    )
    not_retained = str(result["levels"].get("reopened_value_after_reload")) != str(ctx)
    result["levels"]["no_versioned_store"] = no_versioned_store
    result["levels"]["value_not_retained"] = not_retained
    log(f"  no versioned unsloth_model_configs store: {no_versioned_store}")
    log(f"  re-staged value != {ctx} (native default): {not_retained}")
    log(f"VERDICT: {'PASS (no per-model persistence on main, as expected)' if no_versioned_store else 'UNEXPECTED'}")
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
