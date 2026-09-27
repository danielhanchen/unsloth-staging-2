"""Scene: the loaded-models row for a GGUF pick the dense fast path replaced (PR 8241).

Read the PR's NET diff, not its commit series: `gh pr diff --patch` replays every
intermediate commit, and an early one here added a `gguf_quant` field that a later
one dropped in favour of the `gguf_variant` already on main. What actually shipped is
two lines of labelling in `loaded-models-sources.ts`:

* the "GGUF" token is dropped when `transformer_quant` is set, because a GGUF pick the
  dense fast path replaced is a torchao build of the base transformer and the row was
  naming a file the pipeline never opened, and
* the precision now reads `transformer_quant ?? gguf_variant ?? dtype` rather than
  `gguf_variant ?? dtype`.

So the state that shows it is narrow: `model_kind == "gguf"` AND a dense quant engaged.
A plain GGUF load looks identical on both sides -- it already said "GGUF · Q4_K_M"
before this PR. Getting there needs a cached prequant checkpoint for the family and an
explicit `transformer_quant`, which is what `transformer_quant` below is for.

int8, not fp8: the published Z-Image fp8 artifact predates the activation scale floor,
so the loader refuses it ("a zero activation row renders black") and falls back, and
the load then fails outright rather than reaching the state under test.
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

# The panel is `menu-soft-surface`, anchored bottom right. Reached from the collapse
# button rather than by class: the button's aria-label is a contract the tests already
# rely on, the utility classes are not.
_PANEL = 'xpath=ancestor::div[2]'
_COLLAPSE = '[aria-label="Collapse loaded models"]'


def _load_and_wait(session: Session, body: dict, timeout_s: int = 900) -> dict:
    """POST the load and poll to completion.

    /images/load returns in about two seconds with `loaded: false` -- the load runs on.
    Reading status straight after the POST reports "nothing loaded", which photographs
    as an empty page and reads like the model failed.
    """
    api_post(session, "/api/inference/images/unload", {}, timeout=300)
    api_post(session, "/api/inference/images/load", body, timeout=timeout_s)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        time.sleep(5)
        status = api_get(session, "/api/inference/images/status")
        if status.get("loaded"):
            return status
        progress = api_get(session, "/api/inference/images/load-progress")
        if progress.get("error"):
            raise RuntimeError(f"load failed: {str(progress['error'])[:400]}")
    raise RuntimeError(f"model did not load within {timeout_s}s")


async def drive(session: Session, out_dir: Path, label: str,
                repo: str = DEFAULT_REPO, filename: str = DEFAULT_FILE,
                transformer_quant: str = "int8", **_: object) -> tuple[list[Path], dict]:
    status = _load_and_wait(session, {
        "model_path": repo, "gguf_filename": filename, "model_kind": "gguf",
        "transformer_quant": transformer_quant,
    })
    facts = {k: status.get(k) for k in
             ("loaded", "repo_id", "model_kind", "gguf_variant", "transformer_quant",
              "dtype", "device")}
    # The precondition, asserted rather than assumed: without an engaged dense quant on a
    # gguf load the two sides are identical by construction and the pair proves nothing.
    if status.get("model_kind") != "gguf" or not status.get("transformer_quant"):
        raise RuntimeError(
            "this scene needs model_kind='gguf' WITH a dense transformer_quant engaged; "
            f"got {json.dumps(facts)}. Without both, the label is the same on both sides."
        )

    shots: list[Path] = []
    init = seed_init_script(
        type("A", (), {"access_token": session.access_token,
                       "refresh_token": session.refresh_token})(), []
    )
    async with open_chat(session.base_url, init_scripts=[init],
                         viewport=(1600, 1000), headless=True) as sp:
        page = sp.page
        await page.goto(f"{session.base_url}/images", wait_until="domcontentloaded")
        await page.wait_for_timeout(9_000)

        collapse = page.locator(_COLLAPSE).first
        if not await collapse.count():
            # Collapsed to its FAB by a previous session's UI state.
            await page.locator('[aria-label*="loaded models"]').first.click()
            await page.wait_for_timeout(1_500)
        await collapse.wait_for(state="visible", timeout=30_000)
        panel = collapse.locator(_PANEL)

        text = await panel.inner_text()
        facts["badge_text"] = " ".join(t.strip() for t in text.splitlines() if t.strip())
        # The row must be describing the model we loaded. Every row in this panel has the
        # same shape, so a stale row from an earlier load photographs perfectly.
        if not re.search(re.escape(repo.split("/")[-1].split("-GGUF")[0]),
                         facts["badge_text"], re.I):
            raise RuntimeError(f"the loaded-models row is not about {repo}: "
                               f"{facts['badge_text']!r}")

        shot = out_dir / f"{label.lower()}_loaded_badge.png"
        await panel.screenshot(path=str(shot))
        shots.append(shot)
        full = out_dir / f"{label.lower()}_page.png"
        await sp.screenshot(full, full_page=False)
        shots.append(full)
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
