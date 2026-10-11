# studio_test_kit

A small, vendorable cookbook for driving **Unsloth Studio** end-to-end with
Playwright. Drop these files into your project, read them top-to-bottom,
and you'll know how to script chat flows, generate screenshots + videos,
and run pre/post-PR comparisons.

This is **not** a generic library. It is intentionally Studio-flavoured
(model picker selectors, localStorage keys, `/api/auth/login` route,
`unsloth studio -p` CLI, etc.). See [Customisation](#customisation) for
the override points if you're targeting a different SPA.

## What to vendor (15 files)

Copy the whole `studio_test_kit/` directory. No editable install needed
once on `sys.path`.

| # | File | Purpose | Read for |
|---|------|---------|----------|
| 1 | `__init__.py` | Public re-exports | API surface overview |
| 2 | `lifecycle.py` | Install Studio at a given `UNSLOTH_STUDIO_HOME` from any git ref; launch on a port; parse bootstrap password from log; wait on `/healthz`. | Two-port pre/post comparison setup |
| 3 | `auth.py` | Backend JWT login + Playwright init-script builder that seeds the 5 `unsloth_*` localStorage keys. Convenience constructors for Gemini / OpenAI / Anthropic. | Studio's auth + provider model |
| 4 | `ui.py` | Playwright primitives: `open_chat` async context manager (video recording + post-close rename + optional mp4 transcode), `pick_model`, `set_pill`, `send_prompt`, `wait_for_stream`, `wait_for_image`, `wait_for_text`, `extract_data_url`. | The Playwright lessons |
| 5 | `flows.py` | High-level scenarios returning `FlowResult`: `multi_turn_chat`, `image_generation`, `tool_pills`, `vision_upload`. | Composable test flows |
| 6 | `compose.py` | Post-processing: PIL `hstack_images`/`vstack_images`, ffmpeg `hstack_videos`/`webm_to_mp4`. | Pre/post side-by-side composition |
| 7 | `examples/explicit_walkthrough.py` | **Start here.** Fully inline end-to-end flow with every Playwright primitive at the call site and "why" comments. No `flows.*` shortcuts. | Tutorial |
| 8 | `examples/multi_turn.py` | Thin: 4-turn chat via `flows.multi_turn_chat` | Pattern after you know the basics |
| 9 | `examples/image_gen.py` | Thin: Nano Banana image-gen, saves decoded PNG | Pattern |
| 10 | `examples/tools_pills.py` | Thin: Search + Code composer pills | Pattern |
| 11 | `examples/side_by_side.py` | Two `install_studio` + two `launch_studio` + same flow on both + ffmpeg hstack | Pre/post PR comparison |
| 12 | `examples/__init__.py` | Empty marker | — |
| 13 | `_self_test.py` | 13 offline regression tests (no live Studio needed) | CI / verifying the kit didn't bit-rot |
| 14 | `_smoke_ui.py` | Live Playwright integration test against an in-process HTTP server (verifies init-script seeding, screenshot, video flush, mp4 transcode) | CI / smoke after vendoring |
| 15 | `README.md` | This document | — |

## Setup

```bash
pip install playwright httpx pillow
playwright install chromium

# Optional but recommended (used for mp4 transcode and side-by-side video):
apt-get install ffmpeg     # or: brew install ffmpeg
```

Verify the vendoring worked:

```bash
python -m studio_test_kit._self_test           # 13/13 offline regression
python -m studio_test_kit._smoke_ui            # live Playwright smoke
```

## The model: how Studio authentication and providers work

You need to understand this in 60 seconds before the Playwright bits
make sense.

1. **Login is REST, tokens live in localStorage.** Studio exposes
   `POST /api/auth/login` taking `{username, password}` and returning
   `{access_token, refresh_token}`. The SPA reads those tokens from
   localStorage on every request — NOT from cookies. So
   `browser_context.add_cookies()` does nothing useful for auth.
   `auth.login()` makes the REST call; `auth.seed_init_script()` plants
   the tokens in localStorage.

2. **External providers are localStorage-driven, not server-stored.** The
   Settings → Connections UI writes three keys:

   | Key | Value |
   |-----|-------|
   | `unsloth_chat_external_providers` | JSON array of `{id, providerType, name, baseUrl, models}` |
   | `unsloth_chat_external_provider_keys` | `{providerId: plaintextKey}` |
   | `unsloth_chat_connections_enabled` | `"true"` / `"false"` |

   Seeding these via `add_init_script` is exactly equivalent to typing
   the key into the dialog manually. The frontend then RSA-encrypts the
   plaintext key on every request using the public key from `GET
   /api/key`, so the plaintext never goes over the wire even though it
   sits in localStorage.

3. **`add_init_script` ordering is what makes seeding work.** The init
   script runs on every page navigation BEFORE the SPA's own JS. Plant
   AFTER `goto` and you race the SPA's bootstrap — sometimes works,
   sometimes silently doesn't.

## Quickstart

Assumes Studio is already running on `--port` (use `lifecycle.launch_studio`
to spin it up programmatically — see `examples/side_by_side.py` for that
pattern).

```bash
export GEMINI_API_KEY=AIza...

# Tutorial: every primitive inline + commentary. Read this file first.
python -m studio_test_kit.examples.explicit_walkthrough \
    --port 8902 --password 'YourBootstrap!' \
    --model gemini-2.5-flash

# 4-turn conversation (uses flows.multi_turn_chat)
python -m studio_test_kit.examples.multi_turn \
    --port 8902 --password 'YourBootstrap!' --model gemini-2.5-flash

# Image generation, saves decoded PNG
python -m studio_test_kit.examples.image_gen \
    --port 8902 --password 'YourBootstrap!' \
    --model gemini-2.5-flash-image \
    --prompt 'A red panda eating ramen in the rain'

# Search + Code composer pills
python -m studio_test_kit.examples.tools_pills \
    --port 8902 --password 'YourBootstrap!' --model gemini-2.5-flash

# Full pre/post PR comparison: installs Studio twice, drives both, composes
python -m studio_test_kit.examples.side_by_side \
    --pre-branch main --post-branch feat/my-thing \
    --pre-port 8901 --post-port 8902 \
    --model gemini-2.5-flash
```

## Programmatic flow (custom scenario)

If `flows.multi_turn_chat` / `image_generation` / `tool_pills` /
`vision_upload` don't fit your task, compose primitives:

```python
import asyncio
from pathlib import Path
from studio_test_kit.auth import gemini_provider, login, seed_init_script
from studio_test_kit.ui import (
    open_chat, pick_model, send_prompt, set_pill, wait_for_stream,
    wait_for_image, extract_data_url,
)


async def run():
    auth = await login("http://127.0.0.1:8902", "unsloth", "YourBootstrap!")
    init = seed_init_script(
        auth,
        [gemini_provider(api_key="AIza...", models=["gemini-2.5-flash-image"])],
    )
    async with open_chat(
        "http://127.0.0.1:8902",
        init_scripts=[init],
        video_dir=Path("out/video"),
        video_name="my_run",
        transcode_mp4=True,
        headless=True,
    ) as sp:
        await pick_model(sp, "gemini-2.5-flash-image")
        await set_pill(sp, "images", on=True)
        await send_prompt(sp, "Draw a red panda")
        data_url = await wait_for_image(sp)
        raw = await extract_data_url(data_url)
        Path("out/red_panda.png").write_bytes(raw)
    print(f"video: {sp.video_webm}  mp4: {sp.video_mp4}")


asyncio.run(run())
```

`sp.video_webm` / `sp.video_mp4` are populated AFTER the `async with`
exits (Playwright finalises on `context.close()`).

## Pitfalls (read once, save yourself a day)

Every one of these bit me; the kit defends against all of them.

| # | Pitfall | Fix in the kit |
|---|---|---|
| 1 | `wait_until="networkidle"` never fires on Studio because the chat thread holds long-lived SSE/WebSocket; Playwright deadlines. | `open_chat` uses `wait_until="domcontentloaded"` plus an explicit `form:has(textarea) textarea` visibility wait. |
| 2 | `:has-text("gemini-2.5-flash")` substring-matches `gemini-2.5-flash-image`. | `pick_model` uses `get_by_role("option", name=re.compile(rf"^{re.escape(model_id)}$"))`. |
| 3 | A bare `button:has-text("Search")` clicks the LEFT-SIDEBAR Search-history button, not the composer pill. | All UI selectors are scoped: `form:has(textarea) button:has-text(...)`. |
| 4 | Stop button hides BEFORE the final `<img data:image/...>` paints; screenshots taken on stop-hidden miss the image. | `wait_for_image` polls the DOM via `page.evaluate` for an `<img>` whose `src` starts with `data:image/{png,jpeg,webp}` and validates decoded bytes. |
| 5 | `page.type(...)` simulates ~50 ms per keystroke; long prompts take seconds. | `send_prompt` uses `box.fill(...)` (instant paste). |
| 6 | `page.video.path()` read inside the `async with` returns a tempfile that won't survive context close. | Read `sp.video_webm` AFTER the `async with` exits. |
| 7 | Video stalls if you `await page.close()` before `context.close()`. | `open_chat` closes context then browser in a `finally`. |
| 8 | Init script planted AFTER `goto` races the SPA's bootstrap; sometimes works, sometimes silently doesn't. | `seed_init_script` is wired through `context.add_init_script(...)` BEFORE `new_page`/`goto`. |
| 9 | Cookies: Studio reads its JWT from localStorage, not cookies. | `auth.seed_init_script` plants `unsloth_auth_token` / `unsloth_refresh_token` in localStorage. |
| 10 | Playwright's `record_video_dir` writes `page@<hash>.webm`. Parallel tests sharing a video dir can rename each other's recordings. | `open_chat` snapshots pre-existing webms on entry and renames only files that appeared during this session. |
| 11 | Video finalisation skipped if the test body raises (the case you most want a recording). | `open_chat` wraps the playwright lifecycle in `try/finally`. |
| 12 | Shared deadline between password discovery and `/healthz`: a quiet log starves the healthz check and raises a spurious TimeoutError. | `launch_studio` takes independent `password_timeout_s` (default 30s) and `healthz_timeout_s` (default 180s). |
| 13 | Password log regex `[:\s]+` greedily backtracks on `password = secret`, capturing `"="` as the password. | `_PW_RE` uses an explicit `[:=]?` separator with mandatory `\s+` before the value. |
| 14 | `:has-text` substring on `data:image/` matches SVG tracking pixels. | `wait_for_image(mime_prefixes=("data:image/png", "data:image/jpeg", "data:image/webp"))` excludes SVG by default. |

## Things I avoided (and why)

| Avoided | Why |
|---|---|
| WebKit / Firefox launchers | Studio's RSA-encrypt-on-frontend path uses Web Crypto; only reliable on Chromium for now. |
| `wait_for_selector(...)` | Deprecated in Playwright 1.40+. Use `locator(...).wait_for(state=...)`. |
| `expect_response()` for streaming | SSE doesn't end with a discrete `Response`; DOM polling is sturdier. |
| `evaluate_handle` | Returns a JSHandle you must `.dispose()`. `evaluate` returns serialised JSON, which is enough for selector probes. |
| `playwright codegen` selectors | Produces brittle `.locator("...:nth=2")`; hand-written form-scoped role/text predicates are far more durable. |
| Cookie-based auth | Studio reads JWT from localStorage. Cookies do nothing. |
| Page video at `await page.video.path()` mid-block | The path isn't valid until `context.close()`; reading early gives you a tempfile that gets unlinked. |
| `screencast_*` / CDP video | Lower-level, no auto-flush, no rename. Context-level `record_video_dir` is the only path that just works. |

## Customisation

This kit is tuned to **Unsloth Studio**. If you're driving a similar
FastAPI + React chat app with a different shape, here are the override
points:

| What | File:line | How to swap |
|---|---|---|
| Login URL / payload shape | `auth.py:login` | Replace `POST /api/auth/login` with your endpoint; ensure `StudioAuth(access_token, refresh_token, base_url)` is constructible |
| localStorage key names | `auth.py:seed_init_script` (5 keys) | Edit the `payload = {...}` dict; the rest of the kit only consumes `access_token` directly |
| RSA-on-frontend assumption | `auth.py:5-15` docstring | If your app sends keys plain or via header, modify `seed_init_script` to set whatever your SPA reads |
| Composer scope selector | `ui.py:set_pill / send_prompt` use `form:has(textarea) ...` | Change the prefix if your composer isn't an HTML `<form>` |
| Pill names | `ui.py:Pill` literal + `set_pill` map | Add your toggle labels |
| Model picker trigger | `ui.py:pick_model:trigger` | Adjust the data-testid / button text to match your UI |
| Model picker option role | `ui.py:pick_model:option` | If you don't use `role="option"`, change to `role="menuitem"` etc. |
| Chat route path | `ui.py:open_chat` `goto(f"{base_url}/chat")` | Make `chat_path` a parameter and thread through |
| Stop-button selector | `ui.py:wait_for_stream` | Change `aria-label="Stop generating"` to your label |
| `unsloth studio -p` launcher | `lifecycle.py:launch_studio` | Replace the `bin_path studio -p <port>` command with your CLI |
| Studio install layout (`.venv_t5_550` etc.) | `lifecycle.py:_find_unsloth_bin` | Add globs / accept `bin_search_paths` arg |
| Bootstrap password log shape | `lifecycle.py:_PW_RE` | Edit the regex; current accepts `bootstrap/initial/generated password [is] [:=]? value` |
| Healthz path | `lifecycle.py:launch_studio` `/healthz` | Change endpoint string |

A common refactor for a different app: replace just `auth.py` and the
`lifecycle.*` install/launch helpers. `ui.py` / `flows.py` / `compose.py`
should still work if your SPA has a composer textarea and a streaming
stop button.

## Verification (after vendoring)

The two test runners are part of the kit, not external tooling. Run
them to confirm your vendored copy is intact:

```bash
python -m studio_test_kit._self_test
# 13/13 self-tests passed

python -m studio_test_kit._smoke_ui
# OK -- init script primed token; screenshot ~7 KB;
#       webm smoke.webm ~7 KB; mp4 smoke.mp4 ~5 KB
```

`_smoke_ui.py` spins a tiny in-process HTTP server, so it works on a
machine without Studio installed; it's how the kit's CI verifies its
own Playwright pipeline.
