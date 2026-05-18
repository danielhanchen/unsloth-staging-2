# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""End-to-end Playwright check for the change-password form on Studio.

Pins the post-#5490 / #5545 input-count contract:

    * first boot (window.__UNSLOTH_BOOTSTRAP__ injected by the backend):
      change-password renders exactly 2 inputs -- New password,
      Confirm password. No Current password input.

    * admin-forced must_change_password reset (bootstrap absent): 3
      inputs render -- Current password, New password, Confirm password.

The bootstrap-absent path is simulated by an addInitScript that defines
window.__UNSLOTH_BOOTSTRAP__ as a non-writable, non-configurable
property BEFORE the inline server script runs, so the assignment
silently fails (delete + configurable: true is not enough; the inline
assignment recreates the property).

After the input-count checks, we also drive the 2-input form to
completion: fill New + Confirm, submit, confirm we land on /chat with
the Studio sidebar visible. This is the regression the PR claims to
fix on first boot.

Inputs from env:
    BASE_URL   -- e.g. http://127.0.0.1:18897
    PW_ART_DIR -- where to dump screenshots / traces
"""

from __future__ import annotations

import os
import secrets
import sys
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


BASE = os.environ["BASE_URL"]
ART = Path(os.environ.get("PW_ART_DIR", "logs/playwright_auth_form"))
ART.mkdir(parents=True, exist_ok=True)

# Init script that suppresses the server-injected bootstrap. Non-writable
# + non-configurable so the inline <script> in <head> cannot reassign it.
SUPPRESS_BOOTSTRAP = """
Object.defineProperty(window, '__UNSLOTH_BOOTSTRAP__', {
  value: undefined,
  writable: false,
  configurable: false,
  enumerable: true,
});
"""


def _count_password_inputs(page) -> int:
    return page.evaluate(
        "() => document.querySelectorAll('input[type=\"password\"]').length"
    )


def _has_current_password_input(page) -> bool:
    return page.evaluate(
        "() => Boolean(document.getElementById('current-password'))"
    )


def _shoot(page, name: str) -> None:
    # Viewport-only -- full_page=True has been a recurring source of
    # "Connection closed while reading from the driver" flakes on
    # macos-14 arm64 runners running Node 24.
    try:
        page.screenshot(path=str(ART / f"{name}.png"))
    except Exception:
        pass


# Chromium launch args that play nice with the free GitHub runners
# (especially the arm64 macos-14 box, where the default shm size +
# sandbox combination triggers driver-pipe crashes).
LAUNCH_ARGS = ["--no-sandbox", "--disable-dev-shm-usage"]


def _case_bootstrap_absent(pw, failures: list[str]) -> None:
    browser = pw.chromium.launch(args=LAUNCH_ARGS)
    try:
        ctx = browser.new_context()
        ctx.add_init_script(SUPPRESS_BOOTSTRAP)
        page = ctx.new_page()
        # Wait until the SPA has fetched /api/auth/status (which gates
        # the form's first render). On windows-latest the React boot
        # routinely exceeds the previous 15 s selector timeout.
        with page.expect_response(
            lambda r: r.url.endswith("/api/auth/status"), timeout=45_000
        ):
            page.goto(f"{BASE}/change-password", wait_until="domcontentloaded")
        page.wait_for_selector("input[type='password']", timeout=45_000)
        _shoot(page, "01_bootstrap_absent")

        boot_undef = page.evaluate(
            "() => typeof window.__UNSLOTH_BOOTSTRAP__ === 'undefined'"
        )
        if not boot_undef:
            failures.append(
                "bootstrap-absent: __UNSLOTH_BOOTSTRAP__ was not suppressed; "
                "input-count assertion below would be a false positive"
            )

        n = _count_password_inputs(page)
        has_cur = _has_current_password_input(page)
        if n != 3:
            failures.append(
                f"bootstrap-absent: expected 3 password inputs (Current + New + Confirm), got {n}"
            )
        if not has_cur:
            failures.append(
                "bootstrap-absent: #current-password input missing -- "
                "PR #5490's admin-forced-reset fix would be regressed"
            )
    finally:
        browser.close()


def _case_bootstrap_present(pw, failures: list[str]) -> None:
    browser = pw.chromium.launch(args=LAUNCH_ARGS)
    try:
        ctx = browser.new_context()
        page = ctx.new_page()
        page.goto(f"{BASE}/", wait_until="domcontentloaded")
        # On first boot the SPA auto-redirects login -> /change-password.
        expect(page).to_have_url(f"{BASE}/change-password", timeout=45_000)
        page.wait_for_selector("input[type='password']", timeout=45_000)
        _shoot(page, "02_bootstrap_present")

        n = _count_password_inputs(page)
        has_cur = _has_current_password_input(page)
        if n != 2:
            failures.append(f"bootstrap-present: expected 2 password inputs, got {n}")
        if has_cur:
            failures.append(
                "bootstrap-present: #current-password input is still rendered; "
                "PR #5490 regression on first boot"
            )

        # End-to-end: drive the 2-input form and confirm we enter /chat.
        new_pw = f"CIauth-{secrets.token_urlsafe(12)}"
        page.fill("#new-password", new_pw)
        page.fill("#confirm-password", new_pw)
        page.get_by_role("button", name="Change password").click()
        try:
            expect(page).to_have_url(f"{BASE}/chat", timeout=30_000)
            _shoot(page, "03_landed_on_chat")
        except Exception as exc:
            failures.append(f"bootstrap-present: did not land on /chat ({exc})")
            _shoot(page, "03_landed_on_chat_FAIL")
    finally:
        browser.close()


def main() -> int:
    failures: list[str] = []

    with sync_playwright() as pw:
        # Order matters. The PRESENT case submits the form and flips
        # the server's requires_password_change to false; any later
        # /change-password navigation then auto-redirects to /login
        # (1 input), which would falsely fail the ABSENT case's
        # 3-input assertion. Run ABSENT (read-only) first.
        _case_bootstrap_absent(pw, failures)
        _case_bootstrap_present(pw, failures)

    if failures:
        print("FAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("OK: 2 inputs on first boot, 3 inputs on admin-reset, /chat reached")
    return 0


if __name__ == "__main__":
    sys.exit(main())
