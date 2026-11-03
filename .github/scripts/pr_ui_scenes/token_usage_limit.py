from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from pr_ui_scenes._common import Session, api_get, api_post
from studio_test_kit.auth import seed_init_script
from studio_test_kit.ui import open_chat


def put(session: Session, path: str, payload: dict) -> dict:
    request = urllib.request.Request(
        session.base_url + path,
        data = json.dumps(payload).encode(),
        headers = {
            "Authorization": f"Bearer {session.access_token}",
            "Content-Type": "application/json",
        },
        method = "PUT",
    )
    with urllib.request.urlopen(request, timeout = 60) as response:
        return json.load(response)


async def drive(
    session: Session,
    out_dir: Path,
    label: str,
    split: bool = False,
    **_,
) -> tuple[list[Path], dict]:
    now = int(time.time() * 1000)
    api_post(
        session,
        "/api/chat/threads",
        {
            "id": "token-limit-evidence",
            "title": "Usage allowance",
            "modelType": "base",
            "modelId": "unsloth/model",
            "createdAt": now,
        },
    )
    put(
        session,
        "/api/chat/threads/token-limit-evidence/messages/reply",
        {
            "id": "reply",
            "threadId": "token-limit-evidence",
            "role": "assistant",
            "createdAt": now,
            "content": [{"type": "text", "text": "A completed reply recorded 250 tokens."}],
            "metadata": {
                "contextUsage": {"promptTokens": 200, "completionTokens": 50, "totalTokens": 250}
            },
        },
    )
    facts = {}
    try:
        facts["initial_status"] = api_get(session, "/api/token-limit/status")
    except urllib.error.HTTPError as exc:
        if exc.code not in (404, 405):
            raise
        facts["initial_status"] = {"http_status": exc.code}
    supported = "settings" in facts["initial_status"]
    init = seed_init_script(session, [])
    shots = []
    async with open_chat(
        session.base_url, init_scripts = [init], viewport = (1440, 1000), headless = True
    ) as sp:
        page = sp.page
        await page.goto(session.base_url + "/chat", wait_until = "domcontentloaded")
        await page.get_by_role("button", name = "Settings", exact = True).click(timeout = 60000)
        dialog = page.get_by_role("dialog")
        await dialog.get_by_role("button", name = re.compile(r"^Profile\b")).first.click()
        await dialog.get_by_text("Your stats", exact = True).wait_for(timeout = 60000)
        if supported:
            section = dialog.locator('section[data-settings-label="Token usage limit"]')
            await section.get_by_role("textbox", name = "Token cap", exact = True).fill("100")
            await section.get_by_role("button", name = "Save", exact = True).click()
            await section.get_by_role("switch", name = "Limit token usage", exact = True).click()
            await section.get_by_text("250 of 100 tokens", exact = True).wait_for(timeout = 30000)
            await section.scroll_into_view_if_needed()
            facts["enabled_status"] = api_get(session, "/api/token-limit/status")
        else:
            await dialog.locator("section").last.scroll_into_view_if_needed()
        shot = out_dir / f"{label.lower()}_settings.png"
        await dialog.screenshot(path = str(shot))
        shots.append(shot)
        await page.keyboard.press("Escape")
        await dialog.wait_for(state = "hidden")
        await page.reload(wait_until = "domcontentloaded")
        if supported:
            await page.get_by_role("button", name = "Token limit reached", exact = True).click(
                timeout = 60000
            )
            override = page.get_by_role("alertdialog")
            await override.get_by_text(
                re.compile("Would you like to allow one more response")
            ).wait_for()
            await page.wait_for_function(
                "() => [...document.querySelectorAll('[role=alertdialog]')].every(el => el.getAnimations().every(a => a.playState !== 'running'))"
            )
            facts["override_dialog"] = await override.inner_text()
        else:
            await page.get_by_role("button", name = "Settings", exact = True).wait_for(timeout = 60000)
            facts["override_dialog"] = None
        shot = out_dir / f"{label.lower()}_override.png"
        await page.screenshot(path = str(shot))
        shots.append(shot)
        if supported:
            await override.get_by_role("button", name = "Allow one more response", exact = True).click()
            await override.wait_for(state = "hidden")
            facts["after_override"] = api_get(session, "/api/token-limit/status")
            assert facts["after_override"]["pools"][0]["override"] == "granted"
        if split and supported:
            put(
                session,
                "/api/chat/threads/token-limit-evidence/messages/external-reply",
                {
                    "id": "external-reply",
                    "threadId": "token-limit-evidence",
                    "role": "assistant",
                    "createdAt": now + 1,
                    "content": [{"type": "text", "text": "External provider reply."}],
                    "metadata": {
                        "contextUsage": {
                            "promptTokens": 100,
                            "completionTokens": 50,
                            "totalTokens": 150,
                        },
                        "responseDetails": {"providerType": "openai"},
                    },
                },
            )
            put(session, "/api/token-limit", {"mode": "split", "localLimit": 100, "apiLimit": 100})
            api_post(session, "/api/token-limit/override", {"pool": "local"})
            await page.reload(wait_until = "domcontentloaded")
            await page.get_by_role("button", name = "Token limit reached", exact = True).click(
                timeout = 60000
            )
            override = page.get_by_role("alertdialog")
            api_pool = override.get_by_role("button", name = "API this period", exact = True)
            facts["can_select_api_pool"] = await api_pool.count() > 0
            if facts["can_select_api_pool"]:
                await api_pool.click()
                await override.get_by_role(
                    "button", name = "Allow one more response", exact = True
                ).wait_for()
            await page.wait_for_function(
                "() => [...document.querySelectorAll('[role=alertdialog]')].every(el => el.getAnimations().every(a => a.playState !== 'running'))"
            )
            shot = out_dir / f"{label.lower()}_split_override.png"
            await override.screenshot(path = str(shot))
            shots = [shot]
            if facts["can_select_api_pool"]:
                await override.get_by_role(
                    "button", name = "Allow one more response", exact = True
                ).click()
                await override.wait_for(state = "hidden")
                facts["split_status"] = api_get(session, "/api/token-limit/status")
                assert (
                    next(pool for pool in facts["split_status"]["pools"] if pool["pool"] == "api")[
                        "override"
                    ]
                    == "granted"
                )
    return shots, facts
