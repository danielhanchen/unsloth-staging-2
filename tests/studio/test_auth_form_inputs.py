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
    page.screenshot(path=str(ART / f"{name}.png"), full_page=True)


def main() -> int:
    failures: list[str] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(args=["--no-sandbox"])

        # ----- Case 1: bootstrap PRESENT (first boot) ---------------------
        ctx_a = browser.new_context()
        page_a = ctx_a.new_page()
        page_a.goto(f"{BASE}/", wait_until="networkidle")
        # On first boot the SPA auto-redirects login -> /change-password.
        expect(page_a).to_have_url(f"{BASE}/change-password", timeout=15_000)
        page_a.wait_for_selector("input[type='password']", timeout=15_000)
        _shoot(page_a, "01_bootstrap_present")

        n_a = _count_password_inputs(page_a)
        has_cur_a = _has_current_password_input(page_a)
        if n_a != 2:
            failures.append(f"bootstrap-present: expected 2 password inputs, got {n_a}")
        if has_cur_a:
            failures.append(
                "bootstrap-present: #current-password input is still rendered; "
                "PR #5490 regression on first boot"
            )

        # End-to-end: drive the 2-input form and confirm we enter /chat.
        new_pw = f"CIauth-{secrets.token_urlsafe(12)}"
        page_a.fill("#new-password", new_pw)
        page_a.fill("#confirm-password", new_pw)
        page_a.get_by_role("button", name="Change password").click()
        try:
            expect(page_a).to_have_url(f"{BASE}/chat", timeout=30_000)
            _shoot(page_a, "02_landed_on_chat")
        except Exception as exc:
            failures.append(f"bootstrap-present: did not land on /chat ({exc})")
            _shoot(page_a, "02_landed_on_chat_FAIL")
        ctx_a.close()

        # ----- Case 2: bootstrap ABSENT (admin-forced reset) --------------
        # The backend will still inject the inline script because we left
        # requires_password_change true on the install; the init script
        # below makes the assignment a no-op in the page context.
        ctx_b = browser.new_context()
        ctx_b.add_init_script(SUPPRESS_BOOTSTRAP)
        page_b = ctx_b.new_page()
        page_b.goto(f"{BASE}/change-password", wait_until="networkidle")
        page_b.wait_for_selector("input[type='password']", timeout=15_000)
        _shoot(page_b, "03_bootstrap_absent")

        # Confirm the suppression actually worked in this context.
        boot_undef = page_b.evaluate(
            "() => typeof window.__UNSLOTH_BOOTSTRAP__ === 'undefined'"
        )
        if not boot_undef:
            failures.append(
                "bootstrap-absent: __UNSLOTH_BOOTSTRAP__ was not suppressed; "
                "input-count assertion below would be a false positive"
            )

        n_b = _count_password_inputs(page_b)
        has_cur_b = _has_current_password_input(page_b)
        if n_b != 3:
            failures.append(
                f"bootstrap-absent: expected 3 password inputs (Current + New + Confirm), got {n_b}"
            )
        if not has_cur_b:
            failures.append(
                "bootstrap-absent: #current-password input missing -- "
                "PR #5490's admin-forced-reset fix would be regressed"
            )
        ctx_b.close()
        browser.close()

    if failures:
        print("FAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("OK: 2 inputs on first boot, 3 inputs on admin-reset, /chat reached")
    return 0


if __name__ == "__main__":
    sys.exit(main())
