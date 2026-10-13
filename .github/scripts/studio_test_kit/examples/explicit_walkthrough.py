"""End-to-end walkthrough of driving Unsloth Studio with Playwright.

Goal: a single readable file that shows every primitive at the call
site, with inline "why" commentary. After reading this script you
should understand:

  1. How Studio authenticates (JWT cookie-free login)
  2. How external providers + API keys are seeded (localStorage)
  3. How Playwright records video, takes screenshots, and waits for
     SPA state changes
  4. The gotchas that bit me when I first did this and the patterns
     that survive contact with Studio's UI

Compared to `multi_turn.py`, this script DOES NOT use the
`studio_test_kit.flows.*` helpers -- everything is inline so you can
copy-paste-modify for a new task without spelunking through three
modules. Once you've read this once, the thin `flows.*` examples will
make sense.

Run:
    GEMINI_API_KEY=... python -m studio_test_kit.examples.explicit_walkthrough \
        --port 8902 --password 'YourBootstrap!'
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Optional

import httpx
from playwright.async_api import async_playwright


# ── 1. Backend login: hit /api/auth/login for a JWT pair ────────────
#
# Studio's frontend reads the access/refresh tokens out of localStorage
# (NOT cookies), so a normal `browser_context.add_cookies(...)` flow
# would not authenticate the SPA. We POST credentials to the REST
# endpoint here, then plant the tokens in localStorage from Playwright
# in step 3 below.


async def fetch_tokens(base_url: str, username: str, password: str) -> dict:
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.post(
            f"{base_url}/api/auth/login",
            json={"username": username, "password": password},
        )
        r.raise_for_status()
        body = r.json()
        return {
            "access_token": body["access_token"],
            "refresh_token": body.get("refresh_token", ""),
        }


# ── 2. Build the localStorage seed payload ─────────────────────────
#
# Studio reads three keys to discover external providers:
#   unsloth_chat_external_providers       JSON array of provider configs
#   unsloth_chat_external_provider_keys   { providerId: plaintextKey }
#   unsloth_chat_connections_enabled      "true" | "false"
#
# The frontend RSA-encrypts each key on every request using the public
# key from GET /api/key (so the plaintext is never sent over the wire).
# Seeding here is exactly what the Settings -> Connections UI would do
# if you typed the key into the dialog manually.


def build_seed_payload(tokens: dict, gemini_key: str, model_id: str) -> dict:
    provider_id = uuid.uuid4().hex[:16]  # any unique id, frontend echoes it back
    providers = [
        {
            "id": provider_id,
            "providerType": "gemini",
            "name": "Google Gemini",
            "baseUrl": "https://generativelanguage.googleapis.com/v1beta",
            "models": [model_id],
        }
    ]
    return {
        "unsloth_auth_token": tokens["access_token"],
        "unsloth_refresh_token": tokens["refresh_token"],
        "unsloth_chat_external_providers": json.dumps(providers),
        "unsloth_chat_external_provider_keys": json.dumps({provider_id: gemini_key}),
        "unsloth_chat_connections_enabled": "true",
    }


def build_init_script(seed: dict) -> str:
    # JSON.stringify embeds the dict cleanly into JS; the IIFE writes
    # every key into localStorage on EVERY page navigation BEFORE the
    # SPA's own JS runs. That ordering is the whole point of
    # add_init_script -- planting after `goto` is racy.
    return f"""
    (() => {{
        const seed = {json.dumps(seed)};
        for (const k of Object.keys(seed)) {{
            try {{ window.localStorage.setItem(k, seed[k]); }} catch (e) {{}}
        }}
    }})();
    """


# ── 3. The main flow ───────────────────────────────────────────────


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8902)
    ap.add_argument("--password", required=True, help="Studio bootstrap password")
    ap.add_argument("--username", default="unsloth")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--prompt", default="Translate 'good morning' into Japanese.")
    ap.add_argument("--followup", default="Now in a pirate voice.")
    ap.add_argument("--out", type=Path, default=Path("outputs/explicit_walkthrough"))
    ap.add_argument("--headless", action="store_true", default=True)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("walkthrough")

    args.out.mkdir(parents=True, exist_ok=True)
    video_dir = args.out / "video"
    video_dir.mkdir(parents=True, exist_ok=True)

    base = f"http://127.0.0.1:{args.port}"
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        sys.exit("GEMINI_API_KEY env var is required")

    # Step 1+2: get JWT, build seed.
    tokens = await fetch_tokens(base, args.username, args.password)
    init_js = build_init_script(build_seed_payload(tokens, gemini_key, args.model))
    log.info("seeded provider + token via add_init_script")

    # Track existing webms so a parallel run sharing video_dir can't
    # steal our recording on the post-context glob.
    pre_existing = set(video_dir.glob("page@*.webm"))

    async with async_playwright() as p:
        # ── 4. Launch browser + context ─────────────────────────
        #
        # Always Chromium for Studio: WebKit/Firefox have intermittent
        # issues with the Web Crypto path the frontend uses to
        # RSA-encrypt API keys. `slow_mo` is per-action latency; set
        # >0 only when watching a headed run for debugging.
        browser = await p.chromium.launch(headless=args.headless, slow_mo=0)

        # Video MUST be configured on the context, not the page.
        # Playwright writes `page@<hash>.webm` and only finalizes on
        # `context.close()` -- closing the page early stalls the writer.
        context = await browser.new_context(
            viewport={"width": 1440, "height": 900},
            record_video_dir=str(video_dir),
            record_video_size={"width": 1440, "height": 900},
        )

        # Init script BEFORE any new_page / goto. Runs on every nav.
        await context.add_init_script(init_js)
        page = await context.new_page()

        # ── 5. Navigate to /chat ────────────────────────────────
        #
        # DO NOT use `wait_until="networkidle"` -- Studio holds
        # long-lived SSE/WebSocket connections, so there is never
        # 500 ms of zero in-flight requests and Playwright deadlines.
        # `domcontentloaded` plus an explicit composer-textarea wait
        # is the durable pattern.
        await page.goto(f"{base}/chat", wait_until="domcontentloaded")
        await page.locator("form:has(textarea) textarea").first.wait_for(
            state="visible", timeout=15_000
        )
        await page.screenshot(path=str(args.out / "01_chat_open.png"), full_page=True)

        # ── 6. Open the model picker, click EXACT model ─────────
        #
        # Two pitfalls fixed here:
        #   a) `form:has(textarea) [data-testid="model-picker-trigger"]`
        #      scopes the picker to the composer, not the sidebar.
        #   b) `:has-text("gemini-2.5-flash")` is a SUBSTRING match
        #      and also matches `gemini-2.5-flash-image`. Use an
        #      anchored regex via `get_by_role("option", name=...)`.
        trigger = page.locator(
            'form:has(textarea) [data-testid="model-picker-trigger"], '
            'form:has(textarea) button:has-text("Model")'
        ).first
        await trigger.click(timeout=15_000)
        exact = re.compile(rf"^\s*{re.escape(args.model)}\s*$")
        await page.get_by_role("option", name=exact).first.click(timeout=15_000)
        await page.screenshot(path=str(args.out / "02_model_picked.png"), full_page=True)

        # ── 7. Send the first prompt ────────────────────────────
        #
        # Use `fill` not `type` -- `type` simulates keystrokes one by
        # one (~50 ms each, multi-second for long prompts). `fill`
        # instant-pastes. Always scope to `form:has(textarea)` so the
        # sidebar search input doesn't get the keystroke.
        box = page.locator("form:has(textarea) textarea").first
        await box.click()
        await box.fill(args.prompt)
        await box.press("Enter")
        await page.screenshot(path=str(args.out / "03_prompt_sent.png"))

        # ── 8. Wait for the streaming completion ────────────────
        #
        # Streaming pattern: the stop button is the proxy for "model
        # is producing tokens." It becomes visible at first token and
        # hides on stream end. This is reliable for text completions.
        # For IMAGE generation the stop button hides BEFORE the
        # <img data:image/...> paints -- use DOM polling there (see
        # `wait_for_image` in ui.py).
        stop = page.locator(
            'button[aria-label="Stop generating"], button:has-text("Stop")'
        ).first
        try:
            await stop.wait_for(state="visible", timeout=30_000)
        except Exception:
            log.info("stop button never appeared (fast completion?)")
        await stop.wait_for(state="hidden", timeout=90_000)
        await page.screenshot(path=str(args.out / "04_first_response.png"), full_page=True)

        # ── 9. Multi-turn: send a follow-up on the same thread ──
        await box.click()
        await box.fill(args.followup)
        await box.press("Enter")
        try:
            await stop.wait_for(state="visible", timeout=30_000)
        except Exception:
            pass
        await stop.wait_for(state="hidden", timeout=90_000)
        await page.screenshot(path=str(args.out / "05_followup_done.png"), full_page=True)

        # ── 10. Extract the assistant text via page.evaluate ────
        #
        # For asserting WHAT the model produced (not just that it
        # produced something), pull text out of the DOM. The data-role
        # attributes on Studio's message bubbles are stable across
        # versions; if they aren't on your fork, fall back to
        # `[role="article"]` or a class selector.
        assistant_texts = await page.evaluate(
            """() => Array.from(document.querySelectorAll(
                  '[data-role="assistant"] .markdown, '
                  + '[data-message-role="assistant"]'
                )).map(n => n.textContent || '')"""
        )
        log.info("captured %d assistant messages", len(assistant_texts))
        for i, t in enumerate(assistant_texts, 1):
            log.info("  [%d] %s", i, t[:120].replace("\n", " "))

        # ── 11. Clean exit -- ORDER MATTERS ─────────────────────
        #
        # Close the context first (this flushes the .webm). THEN
        # close the browser. Reversing this order or `page.close()`-ing
        # early can leave a partial video.
        await context.close()
        await browser.close()

    # ── 12. Post-context: rename + transcode video ─────────────
    #
    # Playwright writes `page@<hash>.webm`. Rename to a stable
    # filename so callers don't `glob`. Optionally transcode to mp4
    # for PR descriptions and Slack/HF uploads. Track files that
    # existed before this run so a parallel run can't get its
    # recording stolen.
    new_webms = sorted(
        p for p in video_dir.glob("page@*.webm") if p not in pre_existing
    )
    if new_webms:
        webm = video_dir / "walkthrough.webm"
        if webm.exists():
            webm.unlink()
        new_webms[-1].rename(webm)
        for stale in new_webms[:-1]:
            try:
                stale.unlink()
            except OSError as e:
                log.warning("could not unlink stale webm %s: %s", stale, e)
        log.info("wrote %s (%d B)", webm, webm.stat().st_size)

        if shutil.which("ffmpeg"):
            mp4 = video_dir / "walkthrough.mp4"
            r = subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error",
                 "-i", str(webm),
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
                 str(mp4)],
                check=False, capture_output=True, text=True,
            )
            if r.returncode == 0 and mp4.exists() and mp4.stat().st_size > 0:
                log.info("wrote %s (%d B)", mp4, mp4.stat().st_size)
            else:
                log.warning("ffmpeg failed (rc=%s): %s",
                            r.returncode, r.stderr.strip()[:200])
        else:
            log.info("ffmpeg not on PATH -- skipping mp4 transcode")

    print(f"\nDone. Artefacts in {args.out}/")


if __name__ == "__main__":
    asyncio.run(main())
