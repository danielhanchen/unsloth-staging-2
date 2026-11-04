"""Scene: Train > History > a finished Clef run > "Use in Decision API", for PR 16497.

The button sends laya-ft:<run>; the backend stores a Clef folder as clef-ft:<run>. Before the PR
Try is asked for the sent name and never opens; after it Try opens on the stored name.
Seeds a Clef run folder (the marker files the backend test uses) and a completed is_decision
training_runs row, so no GPU or training is needed: the PUT validates files, not serving.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

WORKSPACE = Path(os.environ.get("WORKSPACE") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "scripts"))

from pr_ui_scenes._common import Session  # noqa: E402
from studio_test_kit.auth import seed_init_script  # noqa: E402
from studio_test_kit.ui import open_chat  # noqa: E402

CLEF_FILES = ("config.json", "joint_head.safetensors", "joint_head_config.json")


def _seed(home: Path, run: str) -> str:
    folder = home / "outputs" / run
    folder.mkdir(parents = True, exist_ok = True)
    for name in CLEF_FILES:
        (folder / name).write_text("{}", encoding = "utf-8")
    (folder / "model.safetensors").write_bytes(b"")
    run_id = f"uidiff-{uuid.uuid4().hex[:8]}"
    now = datetime.now(timezone.utc).isoformat()
    config = {
        "is_decision": True,
        "decision_layout": "clef",
        "training_type": "LoRA/QLoRA",
        "model_name": "unsloth/clef-flash",
    }
    with sqlite3.connect(home / "studio.db") as db:
        db.execute(
            "INSERT INTO training_runs (id, status, model_name, dataset_name, config_json, "
            "started_at, ended_at, total_steps, final_step, final_loss, output_dir, "
            "duration_seconds, display_name) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                run_id,
                "completed",
                "unsloth/clef-flash",
                "uidiff_decisions",
                json.dumps(config),
                now,
                now,
                5,
                5,
                0.42,
                str(folder),
                12.0,
                run,
            ),
        )
    return run_id


async def drive(
    session: Session,
    out_dir: Path,
    label: str,
    run: str = "clef_run_16497",
    **_: object,
) -> tuple[list[Path], dict]:
    facts: dict = {"run_id": _seed(session.home, run)}
    init = seed_init_script(
        type(
            "A", (), {"access_token": session.access_token, "refresh_token": session.refresh_token}
        )(),
        [],
    )
    shots: list[Path] = []
    async with open_chat(
        session.base_url, init_scripts = [init], viewport = (1440, 960), headless = True
    ) as sp:
        page = sp.page
        await page.goto(f"{session.base_url}/studio", wait_until = "domcontentloaded")
        await page.get_by_role("tab", name = "History").first.click(timeout = 60_000)
        await page.locator("button", has_text = run).first.click(timeout = 60_000)
        use = page.get_by_role("button", name = "Use in Decision API").first
        await use.wait_for(state = "visible", timeout = 60_000)
        async with page.expect_response(
            lambda r: r.url.endswith("/api/settings/systemone") and r.request.method == "PUT",
            timeout = 60_000,
        ) as put:
            await use.click()
        resp = await put.value
        facts["put_status"] = resp.status
        facts["put_model"] = (await resp.json()).get("model")
        if facts["put_model"] != f"clef-ft:{run}":
            raise RuntimeError(f"[{label}] backend did not store the Clef name: {facts}")
        dialog = page.get_by_role("dialog", name = "Try a decision")
        try:
            await dialog.wait_for(state = "visible", timeout = 15_000)
            facts["try_open"] = True
        except Exception:
            facts["try_open"] = False
        facts["dialogs"] = await page.get_by_role("dialog").evaluate_all(
            "els => els.map(e => (e.getAttribute('aria-label') || e.innerText.split('\\n')[0]).slice(0, 80))"
        )
        shot = out_dir / f"{label.lower()}_use_in_decision_api.png"
        await page.screenshot(path = str(shot))
        shots.append(shot)
    (out_dir / f"{label.lower()}_facts.json").write_text(json.dumps(facts, indent = 1))
    return shots, facts
