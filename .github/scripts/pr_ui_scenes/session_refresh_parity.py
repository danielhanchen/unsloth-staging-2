"""Verify the normal signed-in chat flow remains unchanged for PR 16468."""

from pathlib import Path

from studio_test_kit.auth import seed_init_script
from studio_test_kit.ui import open_chat


async def drive(session, out_dir: Path, label: str, **_):
    init = seed_init_script(
        type(
            "Auth",
            (),
            {"access_token": session.access_token, "refresh_token": session.refresh_token},
        )(),
        [],
    )
    async with open_chat(
        session.base_url,
        init_scripts = [init],
        viewport = (1440, 1000),
        headless = True,
        video_dir = out_dir,
        video_name = label.lower(),
    ) as sp:
        page = sp.page
        await page.goto(f"{session.base_url}/chat", wait_until = "domcontentloaded")
        composer = page.get_by_placeholder("Ask anything", exact = True).first
        await composer.wait_for(state = "visible", timeout = 60000)
        await page.reload(wait_until = "domcontentloaded")
        await composer.wait_for(state = "visible", timeout = 60000)
        facts = {
            "chat_route": page.url.endswith("/chat"),
            "composer_visible": await composer.is_visible(),
            "credentials_present": await page.evaluate("""() => Boolean(
                     localStorage.getItem('unsloth_auth_token') &&
                     localStorage.getItem('unsloth_auth_refresh_token'))"""),
        }
        assert all(facts.values()), facts
        shot = out_dir / f"{label.lower()}_signed_in_chat.png"
        await page.screenshot(path = str(shot), animations = "disabled")
    return [shot], facts
