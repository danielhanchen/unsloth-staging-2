# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Native Safari check for the model-load notification. Remote automation must already be enabled.

Review-only companion to tests/studio/playwright_model_load_notice.py. WebKit under Playwright is
not Safari, and the toast's actions row relies on `[data-action]` inheriting an implicit grid-row
from the `[data-button]` rule, so the engine that ships to Mac users is worth asserting directly.

The fixture stubs fetch in-page because Selenium cannot intercept the network; the hook, toast,
component and CSS are the real ones.
"""

import json
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from _playwright_robust import start_vite, stop_process, wait_for_smoke_page

PORT = 5424
CONFIG = "tests/fixtures/_tmp-safari/vite.config.ts"


def main():
    output = Path("temp/model-load-safari")
    output.mkdir(parents = True, exist_ok = True)
    server = driver = None
    report = {"browser": "native Safari", "passed": False}
    try:
        server = start_vite(PORT, config = CONFIG)
        base = f"http://127.0.0.1:{PORT}"
        wait_for_smoke_page(base + "/", "./main.tsx", proc = server)

        driver = webdriver.Safari()
        driver.set_window_size(1100, 900)
        report["version"] = driver.capabilities.get("browserVersion")
        wait = WebDriverWait(driver, 30)
        driver.get(base + "/")

        wait.until(
            EC.element_to_be_clickable((By.XPATH, "//button[normalize-space()='Load model']"))
        ).click()

        # The toast carries the class under test; wait for it rather than for a timeout.
        wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".chat-model-load-toast"))
        )
        wait.until(
            lambda d: d.execute_script(
                "return !!document.querySelector('.chat-model-load-toast [data-action]')"
            )
        )

        facts = driver.execute_script(
            """
            const el = document.querySelector('.chat-model-load-toast');
            const box = n => { if (!n) return null; const r = n.getBoundingClientRect();
              return {x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.width),
                      h: Math.round(r.height), right: Math.round(r.right), bottom: Math.round(r.bottom)}; };
            const cs = getComputedStyle(el);
            return {
              buttons: [...el.querySelectorAll('button')].map(b => (b.textContent || '').trim()),
              hasClose: !!el.querySelector('[data-close-button]'),
              display: cs.display,
              gridTemplateColumns: cs.gridTemplateColumns,
              toast: box(el),
              content: box(el.querySelector('[data-content]')),
              cancel: box(el.querySelector('[data-cancel]')),
              action: box(el.querySelector('[data-action]')),
            };
            """
        )
        report["facts"] = facts

        # The same invariants the Playwright `layout` case asserts.
        assert facts["display"] == "grid", facts["display"]
        assert facts["hasClose"] is False, "the loading toast must not carry a close button"
        assert set(facts["buttons"]) == {"Cancel loading", "Hide"}, facts["buttons"]
        assert facts["cancel"]["y"] >= facts["content"]["bottom"], facts
        assert facts["action"]["y"] >= facts["content"]["bottom"], facts
        assert facts["cancel"]["right"] <= facts["action"]["x"], facts
        assert facts["content"]["w"] > facts["toast"]["w"] * 0.8, facts

        report["passed"] = True
        print("PASS native-safari", json.dumps(facts["toast"]), flush = True)
    finally:
        if driver is not None:
            driver.quit()
        if server is not None:
            stop_process(server)
        (output / "model-load-safari.json").write_text(json.dumps(report, indent = 1))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
