"""Scene: the chat's own "Unsloth" MCP row, for PR 14349 (stacked on 14348).

BEFORE = 14348's head: Settings > MCP lists the tools with no chat marking, and the backend
has no in-process tools row. AFTER: the six or seven tools a chat must not call read "Not in
chat", GET /api/settings/mcp-access carries ``chat`` per tool, and the chat MCP menu offers an
"Unsloth" preset first. Facts carry the API answers; the shot is the Settings > MCP tab.
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

STUDIO_TOOLS_URL = "http://127.0.0.1/mcp/studio-tools"


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
    session: Session, out_dir: Path, label: str, **_: object
) -> tuple[list[Path], dict]:
    facts: dict = {}
    code, body = _req(session, "GET", "/api/settings/mcp-access")
    tools = body.get("tools") if isinstance(body, dict) else None
    facts["not_in_chat"] = (
        None
        if not tools or "chat" not in tools[0]
        else sorted(t["name"] for t in tools if t.get("chat") is False)
    )
    # Adding the reserved row from a UI session, then listing its tools in process.
    code, row = _req(
        session, "POST", "/api/mcp/servers/", {"display_name": "Unsloth", "url": STUDIO_TOOLS_URL}
    )
    facts["add_row_status"] = code
    if code in (200, 201) and isinstance(row, dict) and row.get("id") is not None:
        facts["row_refresh_status"] = _req(
            session, "POST", f"/api/mcp/servers/{row['id']}/refresh"
        )[0]
        tcode, listed = _req(session, "GET", f"/api/mcp/servers/{row['id']}/tools")
        facts["row_tools_status"] = tcode
        names = None
        if isinstance(listed, dict):
            names = [t.get("name") for t in (listed.get("tools") or [])]
        elif isinstance(listed, list):
            names = [t.get("name") for t in listed]
        facts["row_tools"] = None if names is None else sorted(n for n in names if n)
        _req(session, "DELETE", f"/api/mcp/servers/{row['id']}")
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
        await page.locator('[data-testid="settings-tab-mcp"]').first.click()
        await page.locator("[data-mcp-tool]").first.wait_for(state = "visible")
        facts["not_in_chat_rows"] = await page.get_by_text("Not in chat", exact = True).count()
        shot = out_dir / f"{label.lower()}_mcp_tools.png"
        await page.screenshot(path = str(shot), full_page = True)
        shots.append(shot)
    (out_dir / f"{label.lower()}_facts.json").write_text(json.dumps(facts, indent = 1))
    return shots, facts
