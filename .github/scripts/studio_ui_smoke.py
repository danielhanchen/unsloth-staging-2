#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Launch Studio and drive the real UI with Playwright.

`/healthz` returning 200 only proves the socket is up. The reported failure had a
CLI that answered `-h` while the backend could not import, so this goes further:
log in over the real API, load /chat in Chromium, and require the composer to be
present and interactive. That is the first point at which "Studio works" is true
rather than assumed.

Uses studio_test_kit, vendored next to this file by staging_ci.stage_local.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Only ever used against a throwaway CI install on loopback.
FIRST_RUN_PASSWORD = "unsloth-ci-smoke-pw"

from studio_test_kit.auth import login, seed_init_script  # noqa: E402
from studio_test_kit.lifecycle import StudioInstall, launch_studio, stop_studio  # noqa: E402
from studio_test_kit.ui import open_chat  # noqa: E402


async def complete_first_run(page, out: Path) -> bool:
    """Set the initial password if Studio is asking for one. Returns True if it did.

    Idempotent: on an install that is already set up this finds no fields and is a
    no-op, so the same script covers first run and relaunch.
    """
    setup = page.get_by_text("Setup your account", exact=False)
    if await setup.count() == 0:
        return False
    pw_fields = page.locator("input[type='password']")
    if await pw_fields.count() < 2:
        return False
    await pw_fields.nth(0).fill(FIRST_RUN_PASSWORD)
    await pw_fields.nth(1).fill(FIRST_RUN_PASSWORD)
    await page.get_by_role("button", name="Change password").click()
    await page.wait_for_timeout(2000)
    await page.screenshot(path=str(out / "after-setup.png"), full_page=True)
    print("[ui] completed the first-run account setup")
    return True


async def drive(base_url: str, out: Path, password: str | None) -> dict:
    """Log in, open /chat, and require the composer to be usable."""
    facts: dict[str, object] = {}
    seed = None
    if password:
        # First run mints a password into the log. Without it /chat redirects to
        # login and the composer never appears. login() is async, hence in here.
        auth = await login(base_url, "unsloth", password)
        seed = seed_init_script(auth, providers=[])
        facts["logged_in"] = True
    else:
        print("[ui] no bootstrap password in the log; continuing unauthenticated")
    async with open_chat(base_url, init_scripts=[seed] if seed else None,
                         video_dir=out, video_name="studio-smoke") as sp:
        page = sp.page
        await page.wait_for_load_state("networkidle", timeout=60_000)
        facts["title"] = await page.title()
        await page.screenshot(path=str(out / "chat.png"), full_page=True)

        # A brand new install lands on "Setup your account / Choose a new password"
        # before anything else, which is exactly what a real first run shows. Without
        # completing it there is no composer to find, so drive it.
        facts["first_run_setup"] = await complete_first_run(page, out)
        await page.wait_for_load_state("networkidle", timeout=30_000)

        # The composer is the one control every chat flow needs. Anything less
        # (a mounted div, a 200) can be true of a broken app.
        composer = page.locator(
            "textarea, [contenteditable='true'], [data-testid*='composer']"
        ).first
        try:
            await composer.wait_for(state="visible", timeout=45_000)
        except Exception:
            # Screenshot whatever IS on screen: a bare locator timeout says nothing
            # about why, and this is a headless CI run nobody can look at live.
            await page.screenshot(path=str(out / "composer-missing.png"), full_page=True)
            body = (await page.inner_text("body"))[:600]
            print(f"::error::composer never appeared. Page text:\n{body}")
            raise
        facts["composer_visible"] = True
        await composer.click()
        await composer.type("hello from CI", delay=10)
        facts["composer_accepts_input"] = True
        await page.screenshot(path=str(out / "composer.png"), full_page=True)

        # A frontend that mounted but cannot reach its own API renders an empty
        # shell, which the checks above would pass, so ask the page itself.
        resp = await page.evaluate(
            """async () => {
                 const r = await fetch('/api/health', {credentials: 'include'});
                 return {status: r.status, body: (await r.text()).slice(0, 200)};
               }"""
        )
        facts["api_health_from_page"] = resp
        facts["video"] = str(sp.video_webm) if getattr(sp, "video_webm", None) else None
    return facts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", required=True, help="UNSLOTH_STUDIO_HOME of the install")
    ap.add_argument("--bin", default=None,
                    help="the located `unsloth` CLI; passed so this cannot disagree "
                         "with the step that already found it")
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--out", default="ui-smoke")
    ap.add_argument("--healthz-timeout", type=int, default=300)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "studio.log"

    install = StudioInstall(home=Path(args.home), repo=Path.cwd(), branch="ci")
    facts: dict[str, object] = {}

    # Set the admin password at launch instead of driving the setup form. The CLI
    # honours UNSLOTH_STUDIO_PASSWORD on the same path as --password and applies it
    # before the server binds, so /chat renders the app rather than "Setup your
    # account". Env var, not argv: the help text notes a literal value is visible in
    # the process list.
    #
    # Only on a genuine first run. Setting it when one already exists is a hard
    # error that refuses to start, and the auto-generated bootstrap file is exactly
    # the "no password chosen yet" signal the CLI itself keys on.
    first_run = (install.home / "auth" / ".bootstrap_password").is_file()
    launch_env = {"UNSLOTH_STUDIO_DISABLE_PUBLIC_CHECK": "1"}
    if first_run:
        launch_env["UNSLOTH_STUDIO_PASSWORD"] = FIRST_RUN_PASSWORD
    facts["first_run"] = first_run
    # `home` is deliberately NOT derived from --bin. launch_studio uses it for two
    # different things: finding the CLI, and exporting UNSLOTH_STUDIO_HOME. Pointing
    # it at the venv makes the CLI print "Unsloth Studio not set up. Run install.sh
    # first." and never bind. --bin is only a cross-check that both steps agree.
    if args.bin and not Path(args.bin).exists():
        print(f"::error::located CLI {args.bin} does not exist")
        return 1
    rc = 0
    try:
        t0 = time.time()
        launch_studio(install, port=args.port, log_path=log_path,
                      healthz_timeout_s=args.healthz_timeout,
                      extra_env=launch_env)
        facts["t_healthz_s"] = install.t_healthz_s or round(time.time() - t0, 2)
        print(f"[ui] /healthz 200 after {facts['t_healthz_s']}s")

        base_url = f"http://127.0.0.1:{args.port}"
        # When we set it, that is the password; otherwise fall back to whatever the
        # kit recovered from the auth file or the log.
        password = FIRST_RUN_PASSWORD if first_run else install.bootstrap_password
        facts.update(asyncio.run(drive(base_url, out, password)))
        print("[ui] chat page rendered and the composer accepted input")
    except Exception as exc:  # noqa: BLE001 - the report is the deliverable
        rc = 1
        facts["error"] = f"{type(exc).__name__}: {exc}"
        print(f"::error::Studio UI smoke failed: {facts['error']}")
        tail = log_path.read_text(errors="replace").splitlines()[-40:] if log_path.exists() else []
        for line in tail:
            print(f"    {line[:200]}")
    finally:
        try:
            stop_studio(install)
        except Exception:
            pass
        (out / "ui-smoke.json").write_text(json.dumps(facts, indent=2, default=str),
                                          encoding="utf-8")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
