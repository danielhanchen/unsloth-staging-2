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

from studio_test_kit.auth import login, seed_init_script  # noqa: E402
from studio_test_kit.lifecycle import StudioInstall, launch_studio, stop_studio  # noqa: E402
from studio_test_kit.ui import open_chat  # noqa: E402


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

        # The composer is the one control every chat flow needs. Anything less
        # (a mounted div, a 200) can be true of a broken app.
        composer = page.locator(
            "textarea, [contenteditable='true'], [data-testid*='composer']"
        ).first
        await composer.wait_for(state="visible", timeout=45_000)
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
    if args.bin:
        # launch_studio resolves the CLI from `home` by trying known venv layouts.
        # Point `home` at the parent of bin/ so the first candidate hits, rather than
        # relying on the kit's list matching whatever this install produced.
        binp = Path(args.bin)
        install.home = binp.parent.parent
    facts: dict[str, object] = {}
    rc = 0
    try:
        t0 = time.time()
        launch_studio(install, port=args.port, log_path=log_path,
                      healthz_timeout_s=args.healthz_timeout,
                      extra_env={"UNSLOTH_STUDIO_DISABLE_PUBLIC_CHECK": "1"})
        facts["t_healthz_s"] = install.t_healthz_s or round(time.time() - t0, 2)
        print(f"[ui] /healthz 200 after {facts['t_healthz_s']}s")

        base_url = f"http://127.0.0.1:{args.port}"
        facts.update(asyncio.run(drive(base_url, out, install.bootstrap_password)))
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
