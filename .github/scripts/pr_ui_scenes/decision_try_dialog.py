"""Scene: the Decision API card in Settings > API, and the Try it playground it opens.

Serves PR 12916. Serving is turned on through the settings API (CPU device, configured
default model) on both sides, so the card shows the same state on BEFORE and AFTER; only
the head build has a "Try it" button. On the head the button opens the playground and the
tool-call approval example is run against the real /v1/systemone, so the result panel
shows a real answer (the first run also exercises the model_loading retry path).

Two shots per side: the card, then the playground with its result. The merge base has no
playground, so its second shot is the card header again; the facts carry the decisive part.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

WORKSPACE = Path(
    os.environ.get("WORKSPACE")
    or os.environ.get("UNSLOTH_WORKSPACE")
    or Path(__file__).resolve().parents[2]
)
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "scripts"))

import json  # noqa: E402
import urllib.request  # noqa: E402

from pr_ui_scenes._common import Session, api_get  # noqa: E402
from studio_test_kit.auth import seed_init_script  # noqa: E402
from studio_test_kit.ui import open_chat  # noqa: E402


def api_put(
    session: Session,
    path: str,
    payload: dict,
    timeout: int = 120,
) -> dict:
    req = urllib.request.Request(
        f"{session.base_url}{path}",
        data = json.dumps(payload).encode(),
        method = "PUT",
        headers = {
            "Authorization": f"Bearer {session.access_token}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout = timeout) as r:
        return json.loads(r.read())


TAB_INIT = "try { localStorage.setItem('unsloth_settings_active_tab', 'api-keys'); } catch (e) {}"


def _enable_serving(session: Session, facts: dict) -> None:
    before = api_get(session, "/api/settings/systemone")
    facts["settings_before"] = {k: before.get(k) for k in ("enabled", "model", "device")}
    if not before.get("enabled"):
        after = api_put(
            session,
            "/api/settings/systemone",
            {"enabled": True, "device": "cpu", "expected_enabled": False},
        )
    else:
        after = before
    facts["settings_after"] = {k: after.get(k) for k in ("enabled", "model", "device")}
    if not after.get("enabled"):
        raise RuntimeError(f"Decision API did not turn on: {after}")


async def drive(
    session: Session,
    out_dir: Path,
    label: str,
    preset: str = "Route support tickets",
    run_timeout_ms: int = 900_000,
    **_: object,
) -> tuple[list[Path], dict]:
    facts: dict = {}
    _enable_serving(session, facts)

    shots: list[Path] = []
    init = seed_init_script(
        type(
            "A", (), {"access_token": session.access_token, "refresh_token": session.refresh_token}
        )(),
        [],
    )
    async with open_chat(
        session.base_url, init_scripts = [init, TAB_INIT], viewport = (1440, 1000), headless = True
    ) as sp:
        page = sp.page
        await page.goto(session.base_url, wait_until = "domcontentloaded")
        await page.wait_for_load_state("networkidle", timeout = 60_000)
        await page.wait_for_timeout(3_000)
        heading = page.get_by_role("heading", name = "Decision API", exact = True).first
        openers = [
            lambda: page.keyboard.press("Control+Comma"),
            lambda: page.get_by_role("button", name = "Settings").first.click(timeout = 10_000),
            lambda: page.get_by_label("Settings").first.click(timeout = 10_000),
        ]
        for attempt, opener in enumerate(openers):
            try:
                await opener()
                await heading.wait_for(state = "visible", timeout = 20_000)
                facts["settings_opened_by"] = attempt
                break
            except Exception:  # noqa: BLE001
                await page.screenshot(
                    path = str(out_dir / f"{label.lower()}_debug_open_{attempt}.png")
                )
        else:
            raise RuntimeError("Settings > API never showed the Decision API card")
        section = page.locator("section").filter(has = heading).first
        await section.scroll_into_view_if_needed()
        switch = section.get_by_role("switch").first
        facts["serve_switch_checked"] = await switch.get_attribute("aria-checked")
        await page.wait_for_timeout(1_500)

        card_shot = out_dir / f"{label.lower()}_0_decision_card.png"
        await section.screenshot(path = str(card_shot))
        shots.append(card_shot)

        try_button = section.get_by_role("button", name = "Try it", exact = True)
        facts["try_it_button"] = await try_button.count()
        if facts["try_it_button"] == 0:
            header_shot = out_dir / f"{label.lower()}_1_no_playground.png"
            box = await heading.bounding_box()
            sbox = await section.bounding_box()
            await page.screenshot(
                path = str(header_shot),
                clip = {"x": sbox["x"], "y": box["y"] - 24, "width": sbox["width"], "height": 120},
            )
            shots.append(header_shot)
            return shots, facts

        facts["try_it_enabled"] = await try_button.is_enabled()
        await try_button.click()
        dialog = page.get_by_role("dialog").filter(has_text = "Try a decision").last
        await dialog.wait_for(state = "visible", timeout = 30_000)
        await dialog.get_by_role("button", name = "Examples").click()
        await page.get_by_role("menuitem").filter(has_text = preset).first.click()
        facts["state_prefill"] = (await dialog.locator("#decision-state").input_value())[:80]
        await dialog.get_by_role("button", name = "Run decision").click()
        answer = dialog.get_by_text("Answer", exact = True).or_(dialog.get_by_role("alert"))
        await answer.first.wait_for(state = "visible", timeout = run_timeout_ms)
        await page.wait_for_timeout(1_500)
        facts["result_text"] = (await dialog.inner_text())[-400:]
        facts["result_ok"] = await dialog.get_by_text("Answer", exact = True).count() > 0

        dialog_shot = out_dir / f"{label.lower()}_1_playground_result.png"
        await dialog.screenshot(path = str(dialog_shot))
        shots.append(dialog_shot)
    return shots, facts
