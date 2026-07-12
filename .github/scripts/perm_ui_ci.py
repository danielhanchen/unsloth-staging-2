"""CI-side permission-UI capture for PR #7079, run on a GitHub runner.

Reads the bootstrap password from studio.log, completes the forced first-login
password change, seeds the JWT, then screenshots the composer permission pill,
the 4-level dropdown, and the Settings > General > Permissions section on the
PR branch. Best-effort: always exits 0 so the workflow uploads whatever it got.
"""
import asyncio
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # vendored studio_test_kit
import httpx  # noqa: E402
from studio_test_kit.auth import login, seed_init_script  # noqa: E402
from studio_test_kit.ui import open_chat  # noqa: E402

BASE = "http://127.0.0.1:8888"
OUT = Path("perm_ui_ci_shots")
OUT.mkdir(exist_ok=True)


def read_password():
    txt = ""
    log = Path("studio.log")
    if log.exists():
        txt = log.read_text(errors="ignore")
    m = re.search(r"password saved to:\s*(\S+)", txt)
    if m and Path(m.group(1)).exists():
        return Path(m.group(1)).read_text().strip()
    m = re.search(r"(?i)(?:bootstrap|initial|generated)\s*password(?:\s+is)?\s*[:=]?\s+(\S+)", txt)
    if m:
        return m.group(1).strip().strip(".,")
    # last resort: common home locations
    for cand in (Path.home() / ".unsloth" / "auth" / ".bootstrap_password",):
        if cand.exists():
            return cand.read_text().strip()
    return None


async def ensure_login(pw):
    async with httpx.AsyncClient(timeout=20) as c:
        r = await c.post(f"{BASE}/api/auth/login", json={"username": "unsloth", "password": pw})
        r.raise_for_status()
        tok = r.json()
        if tok.get("must_change_password"):
            new = "CiReview2026!"
            await c.post(
                f"{BASE}/api/auth/change-password",
                headers={"Authorization": f"Bearer {tok['access_token']}"},
                json={"current_password": pw, "new_password": new},
            )
            return new
        return pw


async def main():
    pw = read_password()
    if not pw:
        print("perm_ui_ci: no bootstrap password found in studio.log")
        return
    pw = await ensure_login(pw)
    auth = await login(BASE, "unsloth", pw)
    init = seed_init_script(auth, [])
    async with open_chat(BASE, init_scripts=[init], viewport=(1440, 900), headless=True) as sp:
        page = sp.page
        await page.goto(f"{BASE}/chat", wait_until="domcontentloaded")
        await page.locator("form:has(textarea) textarea").first.wait_for(state="visible", timeout=25000)
        await page.wait_for_timeout(1200)
        await sp.screenshot(OUT / "ci_after_composer_pill.png", full_page=True)
        try:
            await page.get_by_role("button", name="Permission level for tool calls").first.click(timeout=6000)
            await page.wait_for_timeout(700)
            await sp.screenshot(OUT / "ci_after_pill_menu.png", full_page=True)
            await page.keyboard.press("Escape")
        except Exception as e:
            print("perm_ui_ci: pill menu step failed:", str(e)[:120])
        try:
            await page.goto(f"{BASE}/settings", wait_until="domcontentloaded")
            await page.wait_for_timeout(1500)
            await page.locator("text=Permissions").first.scroll_into_view_if_needed(timeout=5000)
            await page.wait_for_timeout(400)
            await sp.screenshot(OUT / "ci_after_settings_permissions.png", full_page=True)
        except Exception as e:
            print("perm_ui_ci: settings step failed:", str(e)[:120])
    print("perm_ui_ci: capture complete ->", OUT)


try:
    asyncio.run(main())
except Exception as e:
    print("perm_ui_ci: fatal:", str(e)[:200])
