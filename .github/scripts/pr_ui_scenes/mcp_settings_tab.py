"""Scene: where Unsloth Studio's own MCP server is configured, for PR 14348.

Before the PR the "Allow agent connections" switch lives in Settings > API and there is no
per-tool choice. After it, Settings > MCP holds the switch plus a Tools list with one switch
per tool, and GET /api/settings/mcp-access reports ``tools``.

Both sides: PUT ``{"disabled_tools": [tool]}`` (base rejects the field, head stores it), read
the settings back, then photograph the MCP tab when it exists, else the API tab's agent
access section. Facts carry the API answers so the claim does not rest on pixels.
"""

from __future__ import annotations

import json
import os
import sys
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
from pr_ui_scenes._slow_browser import slow_page  # noqa: E402

PATH = "/api/settings/mcp-access"


def _req(
    session: Session,
    method: str,
    path: str,
    payload = None,
    timeout: int = 120,
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


async def drive(
    session: Session,
    out_dir: Path,
    label: str,
    tool: str = "cancel",
    **_: object,
) -> tuple[list[Path], dict]:
    facts: dict = {}
    code, body = _req(session, "PUT", PATH, {"disabled_tools": [tool]})
    facts["put_disabled_tools_status"] = code
    code, body = _req(session, "GET", PATH)
    facts["get_status"] = code
    tools = body.get("tools") if isinstance(body, dict) else None
    facts["tools_field"] = None if tools is None else len(tools)
    facts["tools_off"] = (
        None if tools is None else sorted(t["name"] for t in tools if not t["enabled"])
    )
    (out_dir / f"{label.lower()}_facts.json").write_text(json.dumps(facts, indent = 1))

    shots: list[Path] = []
    init = seed_init_script(
        type(
            "A", (), {"access_token": session.access_token, "refresh_token": session.refresh_token}
        )(),
        [],
    )
    async with slow_page(session.base_url, init) as page:
        await page.get_by_role("button", name = "Settings", exact = True).click()
        await page.locator("[role=dialog]").wait_for(state = "visible")
        await page.locator('[data-testid="settings-tab-general"]').first.wait_for()
        mcp_tab = page.locator('[data-testid="settings-tab-mcp"]')
        facts["mcp_tab"] = await mcp_tab.count() > 0
        if facts["mcp_tab"]:
            await mcp_tab.first.click()
            rows = page.locator(f'[data-mcp-tool="{tool}"]')
            await rows.first.wait_for(state = "visible")
            facts["tool_rows"] = await page.locator("[data-mcp-tool]").count()
            facts["tool_switch_checked"] = await rows.first.locator(
                '[role="switch"]'
            ).get_attribute("aria-checked")
            await rows.first.scroll_into_view_if_needed()
        else:
            await page.locator('[data-testid="settings-tab-api-keys"]').first.click()
            section = page.get_by_text("Allow agent connections", exact = True).first
            await section.wait_for(state = "visible")
            # Later sections load after it and push it down; re-centre once the page settles.
            for _ in range(3):
                await page.wait_for_timeout(3000)
                await section.evaluate("e => e.scrollIntoView({block: 'center'})")
            facts["agent_access_on_api_tab"] = True
            facts["tool_rows"] = await page.locator("[data-mcp-tool]").count()
        nav = await page.locator('[data-testid^="settings-tab-"]').all_inner_texts()
        facts["nav"] = [n.strip().split("\n")[0] for n in nav]
        shot = out_dir / f"{label.lower()}_mcp_settings.png"
        await page.screenshot(path = str(shot), full_page = False)
        shots.append(shot)
    (out_dir / f"{label.lower()}_facts.json").write_text(json.dumps(facts, indent = 1))
    return shots, facts
