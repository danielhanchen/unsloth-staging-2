"""Scene: the Decision API status a few minutes after a training run, for PR 13482.

Starting a training run unloads a Clef served on PyTorch. Before the PR nothing loads it
again, so Settings > API keys > Decision API reads "Downloaded" until the next request;
after it, Studio reloads the same Clef once the run ends and the row reads "Loaded on GPU".

Both sides: enable the Decision API on `model` (PyTorch backend, GPU), load it with one
real decision, run a tiny LoRA job through POST /api/train/start, then poll the settings
API for up to `reload_wait_s` after the run ends and photograph the section. The facts
carry the loaded model at each phase so the claim does not rest on the pixels alone.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

WORKSPACE = Path(
    os.environ.get("WORKSPACE")
    or os.environ.get("UNSLOTH_WORKSPACE")
    or Path(__file__).resolve().parents[1]
)
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "scripts"))

from pr_ui_scenes._common import Session  # noqa: E402
from studio_test_kit.auth import seed_init_script  # noqa: E402
from studio_test_kit.ui import open_chat  # noqa: E402

TRAIN_MODEL = "unsloth/gemma-3-270m-it"
QUESTIONS = {"down": {"type": "noul", "instructions": "Is the website unavailable?"}}
TERMINAL = {"completed", "error", "stopped"}


def _req(
    session: Session,
    method: str,
    path: str,
    payload = None,
    timeout: int = 600,
):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        session.base_url + path,
        data = data,
        method = method,
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {session.access_token}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout = timeout) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as exc:
        body = exc.read()
        try:
            return exc.code, json.loads(body)
        except ValueError:
            return exc.code, body.decode(errors = "replace")


def _settings(session: Session) -> dict:
    return _req(session, "GET", "/api/settings/systemone")[1]


def _brief(s: dict) -> dict:
    return {
        k: s.get(k) for k in ("model", "loaded_model", "loading_model", "loaded_device", "error")
    }


def _train_status(session: Session) -> dict:
    return _req(session, "GET", "/api/train/status")[1] or {}


def _phase(status: dict) -> str:
    return str(status.get("phase") or status.get("status") or "")


async def drive(
    session: Session,
    out_dir: Path,
    label: str,
    model: str = "clef-flash",
    load_wait_s: int = 1200,
    reload_wait_s: int = 900,
    **_: object,
) -> tuple[list[Path], dict]:
    facts: dict = {}
    code, body = _req(
        session,
        "PUT",
        "/api/settings/systemone",
        {"enabled": True, "model": model, "device": "gpu", "backend": "pytorch"},
    )
    if code != 200:
        raise RuntimeError(f"[{label}] PUT settings: {code} {body}")

    t0 = time.monotonic()
    while True:
        code, body = _req(
            session,
            "POST",
            "/v1/systemone",
            {
                "model": model,
                "state": "The site returns 503 for every page.",
                "questions": QUESTIONS,
            },
        )
        if code == 200:
            break
        if time.monotonic() - t0 > load_wait_s:
            raise RuntimeError(f"[{label}] {model} never answered: {code} {body}")
        time.sleep(10)
    facts["first_answer"] = body.get("answers")
    facts["load_s"] = round(time.monotonic() - t0, 1)
    facts["before_training"] = _brief(_settings(session))

    ds = session.home / "assets" / "datasets" / "uploads" / "uidiff_13482_chat.jsonl"
    ds.parent.mkdir(parents = True, exist_ok = True)
    rows = [
        {
            "messages": [
                {"role": "user", "content": f"Question {i}?"},
                {"role": "assistant", "content": "Canary."},
            ]
        }
        for i in range(16)
    ]
    ds.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    body = {
        "model_name": TRAIN_MODEL,
        "training_type": "LoRA/QLoRA",
        "format_type": "chatml",
        "local_datasets": [str(ds)],
        "project_name": "uidiff-13482",
        "load_in_4bit": False,
        "max_seq_length": 256,
        "max_steps": 5,
        "num_epochs": 0,
        "batch_size": 2,
        "gradient_accumulation_steps": 1,
        "learning_rate": "2e-4",
        "lora_r": 8,
        "lora_alpha": 16,
        "save_steps": 10_000,
        "random_seed": 3407,
        "warmup_steps": 0,
    }
    code, started = _req(session, "POST", "/api/train/start", body)
    if code != 200:
        raise RuntimeError(f"[{label}] /api/train/start: {code} {started}")
    t0 = time.monotonic()
    during = None
    while True:
        status = _train_status(session)
        if during is None and _phase(status) not in ("", "idle"):
            during = _brief(_settings(session))
        if _phase(status) in TERMINAL:
            break
        if time.monotonic() - t0 > 1800:
            raise RuntimeError(f"[{label}] training never finished: {status}")
        time.sleep(5)
    facts["training_phase"] = _phase(status)
    facts["during_training"] = during
    facts["train_s"] = round(time.monotonic() - t0, 1)

    t0 = time.monotonic()
    while time.monotonic() - t0 < reload_wait_s:
        s = _settings(session)
        if s.get("loaded_model") == s.get("model") and s.get("loaded_model"):
            break
        time.sleep(10)
    facts["after_training"] = _brief(s)
    facts["reloaded_after_s"] = round(time.monotonic() - t0, 1) if s.get("loaded_model") else None

    # Written before the browser step, so a selector failure keeps the measured phases.
    (out_dir / f"{label.lower()}_facts.json").write_text(json.dumps(facts, indent = 1))
    shots: list[Path] = []
    init = seed_init_script(
        type(
            "A", (), {"access_token": session.access_token, "refresh_token": session.refresh_token}
        )(),
        [],
    )
    async with open_chat(
        session.base_url, init_scripts = [init], viewport = (1400, 1000), headless = True
    ) as sp:
        page = sp.page
        await page.goto(f"{session.base_url}/settings", wait_until = "domcontentloaded")
        # The nav label is translated ("API" in English); the test id is stable. click() waits
        # for the tab to render, so no settle sleep is needed after goto.
        await page.locator('[data-testid="settings-tab-api-keys"]').first.click(timeout = 30_000)
        section = page.locator('section[data-settings-label="Decision API"]').first
        await section.wait_for(state = "visible", timeout = 30_000)
        await section.scroll_into_view_if_needed()
        # Wait for the model row's status (both outcomes this scene contrasts) rather than a
        # fixed settle time; the section mounts before the settings fetch fills the row.
        # Any other state (an error row) is still photographed; facts record that it never settled.
        try:
            await page.wait_for_function(
                "el => /Loaded on|loads on first request|Downloaded/i.test(el.innerText)",
                arg = await section.element_handle(),
                timeout = 30_000,
            )
            facts["row_settled"] = True
        except Exception:
            facts["row_settled"] = False
        facts["section_text"] = " | ".join(
            t.strip() for t in (await section.inner_text()).splitlines() if t.strip()
        )[:600]
        shot = out_dir / f"{label.lower()}_decision_api.png"
        await section.screenshot(path = str(shot))
        shots.append(shot)
    return shots, facts
