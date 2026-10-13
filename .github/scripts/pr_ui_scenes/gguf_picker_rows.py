"""Scene: a GGUF repo's quant list in the Model Hub picker.

Serves PR 8222 (a multi-checkpoint repo collapsing to one row per quant) and
PR 8255 (a base quant published at several bit widths), which change the same
surface for different repos -- so the repo is a parameter, not a constant.

PR 8222, `unsloth/LTX-2.3-GGUF`: 63 GGUFs across three checkpoints sharing quant
labels (`<root>/` dev, `distilled/`, `distilled-1.1/`). Before, the picker keyed a
row on the quant token alone: 22 rows, all resolving into `distilled-1.1/`, each
advertising the SUM of all three copies (BF16 at 126 GB for three 42 GB files).

PR 8255, `byteshape/Llama-3.1-8B-Instruct-GGUF`: 18 files, 4 rows, 14 checkpoints
unselectable.

Only Hub listings are read -- no weights -- which makes this the cheapest honest
scene in the set and the right one to develop the harness against.
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
from pathlib import Path

WORKSPACE = Path(os.environ.get("WORKSPACE") or os.environ.get("UNSLOTH_WORKSPACE") or Path(__file__).resolve().parents[2])
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "scripts"))

from pr_ui_scenes._common import Session, api_get  # noqa: E402
from studio_test_kit.auth import seed_init_script  # noqa: E402
from studio_test_kit.ui import open_chat  # noqa: E402

# Any quant token, used to find the quant selector by its CURRENT value.
QUANT_RE = r"(BF16|F16|Q\d[_A-Z0-9]*|UD-[A-Z0-9_]+|IQ\d[_A-Z0-9]*)"

# A search RESULT ROW is also a <button>, and on a query like
# "Llama-3.1-8B-Instruct-GGUF" half the results are named after a quant
# (`...-Q8_0-GGUF`), so a bare quant-token match on `button` hits the results
# list, SELECTS an unrelated repo and photographs it -- identically on both
# sides. These are the things a result row carries and the detail pane's quant
# selector never does: a relative date, a download count, a like count.
ROW_CHROME_RE = r"ago|\d+(\.\d+)?K|downloads"

DEFAULT_REPO = "unsloth/LTX-2.3-GGUF"


async def _select_repo(page, repo: str) -> None:
    """Click the result row for THIS owner's repo, and prove the panel followed.

    Two live failures, both of which produced a clean screenshot of the wrong
    repo on both sides:

    * `get_by_text(leaf, exact=True).first` is ambiguous. `unsloth` and
      `byteshape` both publish `Llama-3.1-8B-Instruct-GGUF`, and the owner-scoped
      results arrive before the global ones, so which row is `.first` depends on
      how far the search had loaded. Match the row that also carries the OWNER.
    * `assert_showing(page, leaf)` passes on the wrong page: the results column
      is headed `Results for "byteshape/Llama-3.1-8B-Instruct-GGUF"`, which
      contains the leaf, so the assertion is satisfied by the query we typed
      rather than by anything that got selected. Assert on the detail heading
      EXACTLY, plus the owner line under it.
    """
    owner, leaf = repo.split("/", 1)
    search = page.get_by_placeholder("Search").first
    await search.wait_for(state="visible", timeout=30_000)
    await search.fill(repo)

    titles = page.get_by_text(leaf, exact=True)
    await titles.first.wait_for(state="visible", timeout=90_000)
    # The list re-sorts as the global results land on top of the owner-scoped
    # ones; clicking the instant a row appears clicks whatever slides into that
    # position a moment later.
    await page.wait_for_timeout(8_000)

    # Playwright's `has_text` normalises whitespace, so a `^owner$` filter on the
    # row never matches -- the row reads "Llama-3.1-8B-Instruct-GGUF byteshape 6
    # 308 8mo ago" as one line. Walk the title elements instead and read the
    # ancestor row's text ourselves.
    target = None
    for i in range(await titles.count()):
        title = titles.nth(i)
        row = title.locator(
            "xpath=ancestor::*[self::li or self::button or self::tr or @role='button'][1]"
        ).first
        if owner in (await row.inner_text()):
            target = title
            break
    if target is None:
        raise RuntimeError(f"no result row for {repo}: the search returned other owners' "
                           f"copies of {leaf} only")
    await target.click()

    heading = page.get_by_role("heading", name=re.compile(rf"^{re.escape(leaf)}$"))
    await heading.first.wait_for(state="visible", timeout=60_000)
    await page.get_by_text(owner, exact=True).first.wait_for(state="visible", timeout=60_000)


async def drive(session: Session, out_dir: Path, label: str,
                repo: str = DEFAULT_REPO, clip: dict | None = None,
                **_: object) -> tuple[list[Path], dict]:
    """Screenshot the open quant list, and return the row count beside it."""
    facts: dict = {}
    # The numeric half. A reviewer cannot count rows in a screenshot, and the row
    # count is the whole claim, so read it from the same server we photograph.
    try:
        data = api_get(session, f"/api/models/gguf-variants?repo_id={repo}")
        rows = data.get("variants", [])
        facts["row_count"] = len(rows)
        facts["first_rows"] = [
            {"quant": r.get("quant"), "display_label": r.get("display_label"),
             "gib": round(r.get("size_bytes", 0) / 2**30, 2), "file": r.get("filename")}
            for r in rows[:5]
        ]
    except Exception as exc:  # noqa: BLE001 -- the screenshot is still worth taking
        facts["row_count_error"] = f"{type(exc).__name__}: {exc}"

    shots: list[Path] = []
    init = seed_init_script(
        type("A", (), {"access_token": session.access_token,
                       "refresh_token": session.refresh_token})(), []
    )
    async with open_chat(session.base_url, init_scripts=[init],
                         viewport=(1500, 1000), headless=True) as sp:
        page = sp.page
        await page.goto(f"{session.base_url}/hub", wait_until="domcontentloaded")
        await _select_repo(page, repo)

        # The quant selector, scoped AWAY from the results list (see ROW_CHROME_RE).
        selector = (page.locator("button")
                    .filter(has_text=re.compile(QUANT_RE))
                    .filter(has_not_text=re.compile(ROW_CHROME_RE)))
        await selector.first.click(timeout=30_000)
        # The listing resolves against the live Hub. Settle before shooting, or the
        # two sides differ by how far each had loaded rather than by the fix.
        await page.wait_for_timeout(10_000)
        facts["selector_label"] = (await selector.first.inner_text()).replace("\n", " ")
        shot = out_dir / f"{label.lower()}_quant_rows.png"
        if clip:
            out_dir.mkdir(parents=True, exist_ok=True)
            # A 1500 px viewport renders at ~440 px per half in a GitHub comment,
            # where the quant rows are unreadable. A FIXED clip, identical on both
            # sides, keeps the pair comparable while staying legible.
            await page.screenshot(path=str(shot), clip=clip)
        else:
            await sp.screenshot(shot, full_page=True)
        shots.append(shot)
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
    ap.add_argument("--repo", default=DEFAULT_REPO)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    s = studio_session(a.url, a.home, a.password)
    print(asyncio.run(drive(s, a.out, a.label, repo=a.repo)))
