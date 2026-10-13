"""Scene: what a GGUF pick actually stages, for PR 8232.

A GGUF pick brings its own transformer. The companion base repo is fetched for the
VAE, text encoders, tokenizer and scheduler -- and, before this PR, for the dense
`transformer/` shards the GGUF replaces and the loader never opens. On
`unsloth/Qwen-Image-2512-GGUF` that is 11 shards, and the user watches ~38 GB more
than the pick needs go past.

The surface is the download manager panel, because that is where a user meets the
number. It shows one item at a time, so the shot has to be taken at the moment the
SECOND item starts -- the base repo -- which is the item this PR changes. Waiting
for it costs the GGUF download first (13 GB, about 20 seconds on this box).

The panel's own text is the evidence; the API plan beside it is what makes the
claim checkable, since a progress bar mid-flight is not a number a reviewer can
diff.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import sys
from pathlib import Path

WORKSPACE = Path(os.environ.get("WORKSPACE") or os.environ.get("UNSLOTH_WORKSPACE") or Path(__file__).resolve().parents[2])
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "scripts"))

from pr_ui_scenes._common import Session, api_post  # noqa: E402
from studio_test_kit.auth import seed_init_script  # noqa: E402
from studio_test_kit.ui import open_chat  # noqa: E402

DEFAULT_REPO = "unsloth/Qwen-Image-2512-GGUF"
DEFAULT_BASE = "unsloth/Qwen-Image-2512"
DEFAULT_FILE = "qwen-image-2512-Q4_K_M.gguf"
DEFAULT_QUANT = "Q4_K_M"


def _purge(cache_hub: Path, repos: list[str]) -> dict:
    """Remove `repos` from the cache so both sides start from the same empty state.

    The two sides share one cache directory (the driver applies `--studio-env` to both),
    so without this the AFTER side would plan against whatever the BEFORE side had already
    pulled down, and its total would be smaller for a reason that is not the fix.
    """
    removed = {}
    for repo in repos:
        d = cache_hub / ("models--" + repo.replace("/", "--"))
        if d.exists():
            removed[repo] = True
            shutil.rmtree(d, ignore_errors=True)
    return removed


async def drive(session: Session, out_dir: Path, label: str,
                repo: str = DEFAULT_REPO, base_repo: str = DEFAULT_BASE,
                filename: str = DEFAULT_FILE, quant: str = DEFAULT_QUANT,
                cache_hub: str = "", **_: object) -> tuple[list[Path], dict]:
    facts: dict = {}
    if cache_hub:
        facts["purged"] = _purge(Path(cache_hub), [repo, base_repo])

    # The numbers, read from the same server that is about to be photographed. The panel
    # shows the base entry's total; this says WHY it is that size.
    try:
        plan = api_post(session, "/api/inference/images/download-plan",
                        {"model_path": repo, "gguf_filename": filename, "model_kind": "gguf"},
                        timeout=900)
        facts["total_gib"] = round(plan["total_bytes"] / 2**30, 2)
        facts["entries"] = [
            {"repo": e["repo_id"], "gib": round(e["bytes"] / 2**30, 2),
             "files": len(e["files"]),
             # The whole PR in one integer: shards of the dense transformer the GGUF replaces.
             "transformer_files": len([f for f in e["files"] if f.startswith("transformer/")])}
            for e in plan["entries"]
        ]
    except Exception as exc:  # noqa: BLE001 -- the picture is still worth taking
        facts["plan_error"] = f"{type(exc).__name__}: {exc}"

    shots: list[Path] = []
    init = seed_init_script(
        type("A", (), {"access_token": session.access_token,
                       "refresh_token": session.refresh_token})(), []
    )
    async with open_chat(session.base_url, init_scripts=[init],
                         viewport=(1600, 1000), headless=True) as sp:
        page = sp.page
        await page.goto(f"{session.base_url}/images", wait_until="domcontentloaded")
        await page.wait_for_timeout(6_000)
        await page.get_by_role("button", name="Select image model").first.click()
        await page.wait_for_timeout(3_000)
        # Exact match: the picker also lists Qwen-Image-GGUF and Qwen-Image-Edit-2511-GGUF,
        # and a substring match would expand a neighbour whose plan this PR does not change.
        await page.get_by_text(repo.split("/")[-1], exact=True).first.click()
        await page.wait_for_timeout(3_000)
        await page.get_by_text(quant, exact=True).first.click()

        panel = page.locator(".hub-download-panel").first
        # Wait for the BASE item, not the GGUF one. `base_repo` is a PREFIX of `repo`, so the
        # trailing separator is load-bearing: matching the bare id would fire on the GGUF row
        # and photograph the 13 GB item that is identical on both sides.
        marker = re.compile(re.escape(base_repo) + r"\s*[··]")
        deadline = 600
        waited = 0
        while waited < deadline:
            await page.wait_for_timeout(2_500)
            waited += 2.5
            try:
                text = await panel.inner_text(timeout=5_000)
            except Exception:  # noqa: BLE001 -- panel not mounted yet
                continue
            if marker.search(text):
                break
        else:
            raise RuntimeError(
                f"[{label}] the base repo item never appeared in the download panel within "
                f"{deadline}s; last panel text: {text!r}"
            )
        facts["panel_text"] = " | ".join(t.strip() for t in text.splitlines() if t.strip())
        m = re.search(r"([\d.]+\s*[KMGT]?B)\s*/\s*([\d.]+\s*[KMGT]?B)", text)
        if m:
            facts["panel_total"] = m.group(2)

        await page.wait_for_timeout(1_500)
        # Element shot: a 1600 px page renders at ~440 px per half in a comment, where a
        # 400 px panel's 12 px type is unreadable. The number IS the evidence here.
        shot = out_dir / f"{label.lower()}_download_panel.png"
        await panel.screenshot(path=str(shot))
        shots.append(shot)

        full = out_dir / f"{label.lower()}_page.png"
        await sp.screenshot(full, full_page=False)
        shots.append(full)

        # Stop the transfer. 58 GB of a checkpoint we are not going to load is the one side
        # effect of this scene, and leaving it running would keep pulling after the shot.
        cancels = page.get_by_role("button", name="Cancel download")
        for i in range(await cancels.count()):
            try:
                await cancels.nth(i).click(timeout=5_000)
            except Exception:  # noqa: BLE001 -- already finished or unmounted
                pass
        await page.wait_for_timeout(3_000)
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
    ap.add_argument("--cache-hub", default="")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    s = studio_session(a.url, a.home, a.password)
    print(asyncio.run(drive(s, a.out, a.label, cache_hub=a.cache_hub)))
