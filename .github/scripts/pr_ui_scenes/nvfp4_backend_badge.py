"""Scene: the loaded-models row for an NVFP4 image pipeline (PR 10730).

The same `nvfp4` scheme lands on either kernel path depending on the device and on whether
the checkpoint bakes activation scales, and until this PR the row said only "NVFP4", so a
FlashInfer load and a torchao fallback were indistinguishable in the UI. The backend now
reports `transformer_quant_backend` on `/api/inference/images/status` and the row renders
it through `quantBackendLabel`.

The state that shows it is narrow: an image PIPELINE load with `transformer_quant="nvfp4"`
that really converts, which needs an SM100 card with FlashInfer and a gated artifact. On
anything else `_transformer_quant_backend` answers None or "torchao" and the halves differ
by less, or not at all.

Needs real weights and a GPU.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

WORKSPACE = Path(os.environ.get("WORKSPACE") or os.environ.get("UNSLOTH_WORKSPACE") or Path(__file__).resolve().parents[2])
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "scripts"))

from pr_ui_scenes._common import Session, api_get, api_post  # noqa: E402
from studio_test_kit.auth import seed_init_script  # noqa: E402
from studio_test_kit.ui import open_chat  # noqa: E402

DEFAULT_REPO = "unsloth/Z-Image-Turbo-GGUF"
DEFAULT_FILE = "z-image-turbo-Q4_K_M.gguf"

# The panel is reached from the collapse button's aria-label, a contract the existing tests
# already rely on, rather than from utility classes that are not one.
_PANEL = "xpath=ancestor::div[2]"
_COLLAPSE = '[aria-label="Collapse loaded models"]'

_SHOW_PANEL = (
    "try {"
    " localStorage.setItem('unsloth_show_loaded_models_indicator', 'true');"
    " localStorage.setItem('unsloth_loaded_models_dismissed', 'false');"
    " localStorage.setItem('unsloth_loaded_models_collapsed', 'false');"
    "} catch (e) {}"
)


def _load_and_wait(session: Session, body: dict, timeout_s: int = 5400) -> dict:
    """POST the load and poll to completion.

    `/images/load` returns in about two seconds with `loaded: false` and the load runs on, so
    reading status straight after the POST reports "nothing loaded", which photographs as an
    empty page and reads like the model failed."""
    try:
        api_post(session, "/api/inference/images/unload", {}, timeout=300)
    except Exception as exc:  # noqa: BLE001 -- a fresh home has nothing to unload (409)
        print(f"unload: {type(exc).__name__}: {exc}", flush=True)
    api_post(session, "/api/inference/images/load", body, timeout=timeout_s)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        time.sleep(10)
        status = api_get(session, "/api/inference/images/status")
        if status.get("loaded"):
            return status
        progress = api_get(session, "/api/inference/images/load-progress")
        if progress.get("error"):
            raise RuntimeError(f"load failed: {str(progress['error'])[:400]}")
    raise RuntimeError(f"model did not load within {timeout_s}s")


async def drive(session: Session, out_dir: Path, label: str,
                repo: str = DEFAULT_REPO,
                filename: str = DEFAULT_FILE,
                transformer_quant: str = "nvfp4",
                **_: object) -> tuple[list[Path], dict]:
    # A GGUF pick the dense fast path replaces, not a pipeline pick: an explicit
    # transformer_quant on a pipeline load is refused outright on this branch ("the dense
    # transformer-quant path applies to GGUF picks only"), so the pipeline route cannot reach
    # the state under test on either side.
    status = _load_and_wait(session, {
        "model_path": repo, "gguf_filename": filename, "model_kind": "gguf",
        "transformer_quant": transformer_quant,
    })
    facts = {k: status.get(k) for k in
             ("loaded", "repo_id", "model_kind", "gguf_variant", "transformer_quant",
              "transformer_quant_backend", "dtype", "device")}
    # The precondition, asserted rather than assumed: without nvfp4 actually engaged there is
    # no backend to name and the two sides are identical by construction.
    if status.get("transformer_quant") != "nvfp4":
        raise RuntimeError(
            "this scene needs an image pipeline load with nvfp4 engaged; got "
            f"{json.dumps(facts)}. Without it the row is the same on both sides."
        )

    shots: list[Path] = []
    init = seed_init_script(
        type("A", (), {"access_token": session.access_token,
                       "refresh_token": session.refresh_token})(), []
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    # The corner indicator is OFF by default (Settings -> General -> Notifications;
    # show-loaded-models-pref.ts reads an explicit "true" and nothing else), so without this the
    # panel never renders and the shot is of an empty page on BOTH sides. Applied identically to
    # each side: it is a user preference, not anything either build decides.
    async with open_chat(session.base_url, init_scripts=[init, _SHOW_PANEL],
                         viewport=(1600, 1000), headless=True) as sp:
        page = sp.page
        await page.goto(f"{session.base_url}/images", wait_until="domcontentloaded")
        await page.wait_for_timeout(9_000)

        collapse = page.locator(_COLLAPSE).first
        if not await collapse.count():
            # Collapsed to its FAB. Its aria-label is "<n> model(s) loaded. Show details, or drag
            # to move", so "Show details" is the stable half; "loaded models" appears only on the
            # EXPANDED card's own buttons and matches nothing while it is collapsed.
            await page.locator('[aria-label*="Show details"]').first.click(timeout=60_000)
            await page.wait_for_timeout(2_000)
        await collapse.wait_for(state="visible", timeout=30_000)
        panel = collapse.locator(_PANEL)

        text = await panel.inner_text()
        facts["badge_text"] = " ".join(t.strip() for t in text.splitlines() if t.strip())
        # Every row in this panel has the same shape, so a stale row from an earlier load
        # photographs perfectly. Prove the row is about the model we just loaded.
        leaf = repo.split("/")[-1].split("-GGUF")[0]
        if not re.search(re.escape(leaf), facts["badge_text"], re.I):
            raise RuntimeError(f"the loaded-models row is not about {repo}: "
                               f"{facts['badge_text']!r}")

        shot = out_dir / f"{label.lower()}_loaded_badge.png"
        await panel.screenshot(path=str(shot))
        shots.append(shot)
        full = out_dir / f"{label.lower()}_page.png"
        await sp.screenshot(full, full_page=False)
        shots.append(full)
    print(f"[{label}] {json.dumps(facts)}", flush=True)
    return shots, facts


if __name__ == "__main__":
    import argparse

    from pr_ui_scenes._common import studio_session

    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--home", type=Path, required=True)
    ap.add_argument("--password", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--label", default="AFTER")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    s = studio_session(a.url, a.home, a.password)
    print(asyncio.run(drive(s, a.out, a.label)))
