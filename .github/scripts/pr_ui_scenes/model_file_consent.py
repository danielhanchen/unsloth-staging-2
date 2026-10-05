"""Scene: picking a repo whose config.json names an MLX `model_file` in the chat model picker.

PR 12659: the MLX loaders import that repo `.py` like an `auto_map` entry, so the consent scan now
reports it and the chat picker shows the remote code dialog before any load. BEFORE, the scan said
no remote code and the picker went straight to loading.

The fact half is the scan endpoint itself (`has_remote_code`, the flagged files), read from the
same server that is photographed.
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
from pathlib import Path

WORKSPACE = Path(os.environ.get("WORKSPACE") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "scripts"))

from pr_ui_scenes._common import Session, api_post  # noqa: E402
from studio_test_kit.auth import seed_init_script  # noqa: E402
from studio_test_kit.ui import open_chat  # noqa: E402

DEFAULT_REPO = "mlx-community/K2-Horizon-0.9B-4bit"
DIALOG_RE = re.compile(r"remote code|trust", re.I)


async def drive(
    session: Session,
    out_dir: Path,
    label: str,
    repo: str = DEFAULT_REPO,
    settle_s: int = 25,
    **_: object,
) -> tuple[list[Path], dict]:
    out_dir.mkdir(parents = True, exist_ok = True)
    facts: dict = {"repo": repo}
    try:
        scan = api_post(session, "/api/models/remote-code-scan", {"model_name": repo})
        facts["has_remote_code"] = scan.get("has_remote_code")
        facts["requires_trust_remote_code"] = scan.get("requires_trust_remote_code")
        facts["approvable"] = scan.get("approvable")
        facts["finding_files"] = sorted({f.get("file") for f in scan.get("findings") or []})
    except Exception as exc:  # noqa: BLE001 -- the screenshot is still worth taking
        facts["scan_error"] = f"{type(exc).__name__}: {exc}"

    shots: list[Path] = []
    init = seed_init_script(
        type(
            "A", (), {"access_token": session.access_token, "refresh_token": session.refresh_token}
        )(),
        [],
    )
    async with open_chat(
        session.base_url, init_scripts = [init], viewport = (1400, 900), headless = True
    ) as sp:
        page = sp.page
        await page.wait_for_timeout(4_000)
        await page.screenshot(path = str(out_dir / f"{label.lower()}_debug_chat.png"))
        trigger = page.get_by_role("button", name = re.compile(r"^\s*Select model", re.I))
        await trigger.first.click(timeout = 60_000)
        await page.wait_for_timeout(1_500)
        await page.screenshot(path = str(out_dir / f"{label.lower()}_debug_picker.png"))
        # The default list searches Unsloth's own repos only.
        await page.get_by_text("Search Hub", exact = True).first.click()
        await page.wait_for_timeout(1_500)
        search = page.get_by_placeholder(re.compile(r"^Search", re.I)).first
        await search.fill(repo)
        await page.wait_for_timeout(3_000)
        await page.screenshot(path = str(out_dir / f"{label.lower()}_debug_search.png"))
        leaf = repo.split("/", 1)[1]
        row = page.get_by_text(leaf, exact = False)
        await row.first.wait_for(state = "visible", timeout = 90_000)
        await page.wait_for_timeout(5_000)
        rows = page.get_by_role("option").filter(has_text = leaf)
        target = rows.first if await rows.count() else row.first
        facts["clicked_row"] = (await target.inner_text()).replace("\n", " ")[:120]
        await target.click()

        dialog = page.get_by_role("alertdialog")
        try:
            await dialog.first.wait_for(state = "visible", timeout = settle_s * 1000)
            facts["consent_dialog"] = True
            facts["dialog_text"] = (await dialog.first.inner_text()).replace("\n", " ")[:400]
        except Exception:  # noqa: BLE001 -- BEFORE has no dialog; photograph what it shows
            facts["consent_dialog"] = False
        shot = out_dir / f"{label.lower()}_model_file_consent.png"
        await page.screenshot(path = str(shot))
        shots.append(shot)
        if facts["consent_dialog"]:
            # Decline, so nothing is trusted or loaded on the reused home.
            cancel = dialog.first.get_by_role(
                "button", name = re.compile(r"cancel|decline|don.t", re.I)
            )
            if await cancel.count():
                await cancel.first.click()
    return shots, facts


if __name__ == "__main__":
    import argparse

    from pr_ui_scenes._common import studio_session

    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required = True)
    ap.add_argument("--home", type = Path, required = True)
    ap.add_argument("--password", required = True)
    ap.add_argument("--out", type = Path, required = True)
    ap.add_argument("--label", default = "AFTER")
    ap.add_argument("--repo", default = DEFAULT_REPO)
    a = ap.parse_args()
    s = studio_session(a.url, a.home, a.password)
    print(asyncio.run(drive(s, a.out, a.label, repo = a.repo)))
