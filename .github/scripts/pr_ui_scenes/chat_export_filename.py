"""Scene: the file name a single-chat export downloads under (PR 12586).

A chat with a fixed title, model and two turns is created through the API on both
sides, then exported to Markdown from the composer's "+" menu > "Export chat". The
browser download's suggested file name is the measured fact. Because a download
name is never painted by Studio itself, the scene pins a harness-drawn caption
("Browser saved: <name>") onto the page before the shot, so each half shows the name
its own build produced. The menu is open in the shot to show where the export ran.
"""

from __future__ import annotations

import os
import re
import sys
import uuid
from pathlib import Path

WORKSPACE = Path(os.environ.get("WORKSPACE") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(WORKSPACE))

from pr_ui_scenes._common import Session, api_get, api_post  # noqa: E402
from studio_test_kit.auth import seed_init_script  # noqa: E402
from studio_test_kit.ui import open_chat  # noqa: E402

THREAD_ID = "uidiff-export-name-" + uuid.uuid4().hex[:8]
THREAD_TITLE = "Biology of Stag Beetles Explained"
MODEL_ID = "unsloth/Gemma-4-26B-A4B"
CREATED_AT_MS = 1_767_225_600_000
VIEWPORT = (1440, 900)


def _put(session: Session, path: str, payload: dict) -> dict:
    import json
    import urllib.request

    req = urllib.request.Request(
        session.base_url + path,
        data = json.dumps(payload).encode(),
        method = "PUT",
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {session.access_token}",
        },
    )
    with urllib.request.urlopen(req, timeout = 60) as resp:
        return json.loads(resp.read() or b"{}")


def _seed(session: Session) -> None:
    api_post(
        session,
        "/api/chat/threads",
        {
            "id": THREAD_ID,
            "title": THREAD_TITLE,
            "modelType": "base",
            "modelId": MODEL_ID,
            "createdAt": CREATED_AT_MS,
            "updatedAt": CREATED_AT_MS,
        },
    )
    turns = [
        ("u1", None, "user", "Why do stag beetles have big jaws?"),
        ("a1", "u1", "assistant", "Males use the mandibles to wrestle rivals."),
    ]
    for i, (mid, parent, role, text) in enumerate(turns):
        mid = f"{THREAD_ID}-{mid}"
        _put(
            session,
            f"/api/chat/threads/{THREAD_ID}/messages/{mid}",
            {
                "id": mid,
                "threadId": THREAD_ID,
                "parentId": f"{THREAD_ID}-{parent}" if parent else None,
                "role": role,
                "content": [{"type": "text", "text": text}],
                "createdAt": CREATED_AT_MS + i + 1,
            },
        )
    stored = api_get(session, f"/api/chat/threads/{THREAD_ID}")
    if stored.get("title") != THREAD_TITLE or stored.get("modelId") != MODEL_ID:
        raise RuntimeError(f"seeded thread did not round-trip: {stored!r}")


async def drive(session: Session, out_dir: Path, label: str, **_: object):
    _seed(session)
    init = seed_init_script(
        type(
            "A", (), {"access_token": session.access_token, "refresh_token": session.refresh_token}
        )(),
        [],
    )
    facts: dict = {"_thread_id": THREAD_ID}
    shots: list[Path] = []
    async with open_chat(
        session.base_url, init_scripts = [init], viewport = VIEWPORT, headless = True
    ) as sp:
        page = sp.page
        await page.goto(
            f"{session.base_url}/chat?thread={THREAD_ID}", wait_until = "domcontentloaded"
        )
        await page.get_by_text("Males use the mandibles").first.wait_for(
            state = "visible", timeout = 60_000
        )
        facts["_url"] = page.url

        # "Export chat" is unpinned by default, so it lives under "+" > "More".
        plus = page.locator('button[aria-label="Tools and attachments"]').first
        more = page.get_by_role("menuitem", name = re.compile(r"^More$"))
        export_trigger = page.get_by_role("menuitem", name = re.compile(r"Export chat"))
        markdown = page.get_by_role("menuitem", name = "Markdown", exact = True)
        for attempt in range(6):
            try:
                await plus.click()
                await more.wait_for(state = "visible", timeout = 4_000)
                await more.hover()
                await export_trigger.wait_for(state = "visible", timeout = 4_000)
                await export_trigger.hover()
                await markdown.wait_for(state = "visible", timeout = 4_000)
                break
            except Exception:  # noqa: BLE001 -- menu lost a refresh race
                await page.keyboard.press("Escape")
                await page.keyboard.press("Escape")
                await page.wait_for_timeout(800)
        else:
            await page.screenshot(path = str(out_dir / f"{label.lower()}_menu_fail.png"))
            raise RuntimeError("the composer + > More > Export chat > Markdown path never opened")
        menu_shot = out_dir / f"{label.lower()}_export_menu.png"
        await page.screenshot(path = str(menu_shot), full_page = False)

        async with page.expect_download(timeout = 60_000) as info:
            await markdown.click()
        download = await info.value
        name = download.suggested_filename
        saved = out_dir / f"{label.lower()}_{name}"
        await download.save_as(saved)
        body = saved.read_text(encoding = "utf-8")
        facts["download_name"] = name
        facts["markdown_has_reply"] = "Males use the mandibles" in body

        await page.evaluate(
            """(name) => {
              const el = document.createElement('div');
              el.id = 'uidiff-download-caption';
              el.textContent = 'Browser saved: ' + name;
              Object.assign(el.style, {position: 'fixed', left: '50%', top: '16px',
                transform: 'translateX(-50%)', zIndex: 2147483647, padding: '10px 16px',
                background: '#111', color: '#fff', font: '600 16px system-ui',
                borderRadius: '8px', boxShadow: '0 4px 16px rgba(0,0,0,.4)'});
              document.body.appendChild(el);
            }""",
            name,
        )
        await page.wait_for_timeout(500)
        shot = out_dir / f"{label.lower()}_export_name.png"
        await page.screenshot(path = str(shot), full_page = False)
        shots.append(shot)
    if not facts["markdown_has_reply"]:
        raise RuntimeError(f"{label}: exported Markdown lacks the seeded reply")
    return shots, facts
