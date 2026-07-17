"""Playwright Chromium primitives for Studio's /chat UI.

The big lesson from PR5720 driving: do NOT key off the global "stop button
disappeared" signal -- the stop button disappears slightly before the
final image paints, so a Nano Banana screenshot taken at that moment is
missing the image. Use `wait_for_image(page, ...)` for image-gen flows;
it polls the DOM for an `<img>` whose src starts with `data:image/`.

Selectors used here are scoped to the composer `<form>` whose textarea
is the chat input, so they don't accidentally match the sidebar buttons
(e.g. a global "Search" button on the left rail).
"""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
import subprocess
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Literal, Optional

from playwright.async_api import BrowserContext, Page, async_playwright

_log = logging.getLogger(__name__)


Pill = Literal["search", "code", "images"]


@dataclass
class StudioPage:
    """Bundle of the live Playwright page + its browser context.

    Keep both because video files are flushed on `context.close()` only.
    `video_webm` and `video_mp4` are populated AFTER the `open_chat`
    context manager exits (Playwright only finalizes the .webm on
    `context.close()`). Read them after the `async with` block.
    """

    page: Page
    context: BrowserContext
    base_url: str
    video_webm: Optional[Path] = None
    video_mp4: Optional[Path] = None

    async def screenshot(self, out: Path, full_page: bool = True) -> None:
        out.parent.mkdir(parents=True, exist_ok=True)
        await self.page.screenshot(path=str(out), full_page=full_page)


@asynccontextmanager
async def open_chat(
    base_url: str,
    init_scripts: Optional[list[str]] = None,
    video_dir: Optional[Path] = None,
    video_name: str = "session",
    transcode_mp4: bool = False,
    viewport: tuple[int, int] = (1440, 900),
    headless: bool = True,
    slow_mo_ms: int = 0,
) -> AsyncIterator[StudioPage]:
    """Launch headless Chromium, seed init scripts, open `/chat`.

    Video recording: Playwright auto-names files `page@<hash>.webm` and
    only finalizes them on `context.close()`. When `video_dir` is set,
    this helper:
      1. Configures record_video_dir + record_video_size on the context.
      2. After context exit, renames the auto-generated webm to
         `<video_dir>/<video_name>.webm` and stores it on
         `StudioPage.video_webm`.
      3. If `transcode_mp4=True` (and ffmpeg is on PATH), also produces
         `<video_dir>/<video_name>.mp4` via libx264/yuv420p (PR-body and
         most-player friendly) and stores it on `StudioPage.video_mp4`.

    The `StudioPage` yielded inside the `async with` lives past context
    close, so callers can read `sp.video_webm` / `sp.video_mp4` AFTER
    the block exits.
    """
    # To survive parallel runs sharing `video_dir`, mint a unique
    # session token; on exit we glob only files that arrived after this
    # context began so we can't steal a sibling session's recording.
    session_token = uuid.uuid4().hex
    sp_holder: dict = {}
    pre_existing: set[Path] = set()
    # The video finalization MUST run even when the caller's body raises
    # (that's exactly when you want the recording). Wrap the whole
    # playwright lifecycle in try/finally so the rename + transcode
    # happens unconditionally.
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=headless, slow_mo=slow_mo_ms)
            kw: dict = {"viewport": {"width": viewport[0], "height": viewport[1]}}
            if video_dir is not None:
                video_dir.mkdir(parents=True, exist_ok=True)
                kw["record_video_dir"] = str(video_dir)
                kw["record_video_size"] = {"width": viewport[0], "height": viewport[1]}
                # Snapshot files that already exist; we'll only sweep
                # webms that appeared AFTER this snapshot.
                pre_existing = set(video_dir.glob("page@*.webm"))
            context = await browser.new_context(**kw)
            for script in init_scripts or []:
                await context.add_init_script(script)
            page = await context.new_page()
            # Streaming SPAs hold long-lived SSE/WebSocket connections,
            # so `networkidle` can deadline. domcontentloaded plus an
            # explicit composer-textarea wait is the reliable pattern.
            await page.goto(f"{base_url}/chat", wait_until="domcontentloaded")
            try:
                await page.locator("form:has(textarea) textarea").first.wait_for(
                    state="visible", timeout=15_000
                )
            except Exception:
                # Some apps render the composer lazily; the test code
                # following may still wait_for it. Don't fail open_chat
                # solely on this fast-path probe.
                pass
            sp = StudioPage(page=page, context=context, base_url=base_url)
            sp_holder["sp"] = sp
            sp_holder["session_token"] = session_token
            try:
                yield sp
            finally:
                await context.close()
                await browser.close()
    finally:
        sp = sp_holder.get("sp")
        if sp is not None and video_dir is not None:
            try:
                # Only consider webms that did not exist when we started.
                new_webms = sorted(
                    p for p in video_dir.glob("page@*.webm")
                    if p not in pre_existing
                )
                if new_webms:
                    final_webm = Path(video_dir) / f"{video_name}.webm"
                    if final_webm.exists():
                        final_webm.unlink()
                    new_webms[-1].rename(final_webm)
                    # Sweep extras that originated from this session.
                    for stale in new_webms[:-1]:
                        try:
                            stale.unlink()
                        except OSError as e:
                            _log.warning("could not unlink stale webm %s: %s", stale, e)
                    sp.video_webm = final_webm
                    if transcode_mp4 and shutil.which("ffmpeg"):
                        final_mp4 = Path(video_dir) / f"{video_name}.mp4"
                        result = subprocess.run(
                            ["ffmpeg", "-y", "-loglevel", "error",
                             "-i", str(final_webm),
                             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
                             str(final_mp4)],
                            check=False, capture_output=True, text=True,
                        )
                        if (final_mp4.exists() and final_mp4.stat().st_size > 0
                                and result.returncode == 0):
                            sp.video_mp4 = final_mp4
                        else:
                            _log.warning(
                                "ffmpeg transcode failed (rc=%s): %s",
                                result.returncode, result.stderr.strip()[:200],
                            )
                    elif transcode_mp4:
                        _log.warning(
                            "transcode_mp4=True but ffmpeg not found on PATH; "
                            "skipping mp4 transcode"
                        )
            except Exception as e:
                # Surface the failure -- a permission-denied rename or
                # full disk should NOT silently look like "no recording
                # was requested".
                _log.warning("video finalization failed: %s", e)


# ── UI primitives ────────────────────────────────────────────────────


async def pick_model(sp: StudioPage, model_id: str, timeout_ms: int = 15000) -> None:
    """Open the model picker, click the option whose label EXACTLY equals
    `model_id`.

    Exact match matters: `:has-text("gemini-2.5-flash")` also matches
    `gemini-2.5-flash-image`. Using `get_by_role("option", name=re.compile(...))`
    with an anchored, escaped pattern avoids that collision and also
    protects against quote characters inside the model id.
    """
    page = sp.page
    trigger = page.locator(
        'form:has(textarea) [data-testid="model-picker-trigger"], '
        'form:has(textarea) button:has-text("Model")'
    ).first
    await trigger.click(timeout=timeout_ms)
    pattern = re.compile(rf"^\s*{re.escape(model_id)}\s*$")
    option = page.get_by_role("option", name=pattern).first
    await option.click(timeout=timeout_ms)


async def set_pill(sp: StudioPage, pill: Pill, on: bool = True, timeout_ms: int = 5000) -> None:
    """Toggle a composer pill (Search / Code / Images) on or off.

    Scoped to the composer form so it never matches the sidebar Search.
    """
    label = {"search": "Search", "code": "Code", "images": "Images"}[pill]
    btn = sp.page.locator(
        f'form:has(textarea) button:has-text("{label}")'
    ).first
    state = await btn.get_attribute("aria-pressed")
    is_on = (state == "true")
    if is_on != on:
        await btn.click(timeout=timeout_ms)


async def send_prompt(sp: StudioPage, text: str) -> None:
    """Type into the composer and submit with Enter."""
    box = sp.page.locator("form:has(textarea) textarea").first
    await box.click()
    await box.fill(text)
    await box.press("Enter")


async def wait_for_stream(sp: StudioPage, timeout_ms: int = 90_000) -> None:
    """Wait for the streaming stop button to appear then disappear.

    Good enough for text-only completions. Image generation paints AFTER
    the stop button hides, so use `wait_for_image` for that flow.
    """
    page = sp.page
    stop = page.locator('button[aria-label="Stop generating"], '
                        'button:has-text("Stop")').first
    try:
        await stop.wait_for(state="visible", timeout=timeout_ms)
    except Exception:
        pass  # Some flows finish faster than the button appears.
    await stop.wait_for(state="hidden", timeout=timeout_ms)


async def wait_for_image(
    sp: StudioPage,
    timeout_ms: int = 120_000,
    min_decoded_bytes: int = 256,
    mime_prefixes: tuple[str, ...] = ("data:image/png", "data:image/jpeg",
                                      "data:image/webp"),
) -> str:
    """Poll the chat DOM until an `<img>` whose src starts with one of
    `mime_prefixes` appears, then return that data URL.

    `min_decoded_bytes` is the minimum DECODED size (so SVG tracking
    pixels and 1x1 placeholders are filtered). Default prefixes exclude
    svg and tracking pixels by default; pass a wider tuple to opt in.

    Raises TimeoutError on deadline.
    """
    page = sp.page
    deadline = asyncio.get_event_loop().time() + (timeout_ms / 1000)
    prefix_js = "[" + ",".join(f'"{p}"' for p in mime_prefixes) + "]"
    while asyncio.get_event_loop().time() < deadline:
        src = await page.evaluate(
            "(prefixes) => {"
            "  const imgs = Array.from(document.querySelectorAll('img'));"
            "  const cand = imgs.find(i => i.src && prefixes.some(p => i.src.startsWith(p)));"
            "  return cand ? cand.src : null;"
            "}",
            list(mime_prefixes),
        )
        if isinstance(src, str) and ";base64," in src:
            try:
                import base64 as _b64
                raw = _b64.b64decode(src.split(";base64,", 1)[1], validate=False)
                if len(raw) >= min_decoded_bytes:
                    return src
            except Exception:
                pass
        await asyncio.sleep(0.5)
    raise TimeoutError(
        f"No matching data: image rendered within {timeout_ms}ms "
        f"(prefixes={mime_prefixes}, min_decoded_bytes={min_decoded_bytes})"
    )


async def wait_for_text(sp: StudioPage, substring: str, timeout_ms: int = 60_000) -> None:
    """Poll the page for a visible text snippet (assistant token)."""
    page = sp.page
    deadline = asyncio.get_event_loop().time() + (timeout_ms / 1000)
    while asyncio.get_event_loop().time() < deadline:
        if await page.locator(f"text={substring}").count() > 0:
            return
        await asyncio.sleep(0.5)
    raise TimeoutError(f"Text '{substring}' did not appear within timeout")


async def extract_data_url(data_url: str) -> bytes:
    """Decode a `data:image/...;base64,XXXX` URL to bytes."""
    import base64
    if ";base64," not in data_url:
        raise ValueError("Expected base64-encoded data URL")
    return base64.b64decode(data_url.split(";base64,", 1)[1])
