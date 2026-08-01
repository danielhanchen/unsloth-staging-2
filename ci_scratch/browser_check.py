"""Load Studio and hit its loopback API from each browser engine.

Studio is running with proxy environment variables pointing at a dead port, so
a browser that does not bypass the proxy for 127.0.0.1 fails here.
"""
import sys

from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:8888"
failures = []

with sync_playwright() as p:
    for name in sys.argv[1:]:
        try:
            browser = getattr(p, name).launch()
            page = browser.new_page()
            page.goto(BASE + "/", wait_until="domcontentloaded", timeout=90_000)
            page.wait_for_timeout(4_000)
            response = page.request.get(BASE + "/api/health")
            body = response.text()[:200]
            if not response.ok:
                raise AssertionError(f"/api/health -> HTTP {response.status}")
            page.screenshot(path=f"studio-{name}.png")
            print(f"{name}: OK  title={page.title()!r}  health={response.status} {body}")
            browser.close()
        except Exception as error:  # noqa: BLE001 - report every engine, fail at the end
            print(f"{name}: FAIL  {type(error).__name__}: {error}")
            failures.append(name)

if failures:
    sys.exit("browser engines failed: " + ", ".join(failures))
print("all browser engines reached the loopback backend")
