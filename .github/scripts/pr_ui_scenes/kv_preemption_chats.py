"""Scene: four chats at a context they cannot all fit in (PRs 10301 and 10358).

One llama-server holds one KV cache. Four chats with long prompts and Studio's "Max" token
setting each fit alone, but not together. Before preemption they collide: the server ends
every slot with `Context size has been exceeded` (an error box per pane), or admission holds
them back. With preemption a chat that has to make room shows "Paused while another chat
finishes" and every pane finishes with its answer and no error box.

Each pane is its own browser context (its own thread). A pane is photographed the moment a
pause is first seen and again when it finishes; the final shots are tiled per side.
Facts carry the counts the claim rests on: panes that showed the pause, panes with an error
box, panes that finished, and how the load reported its slots.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path

WORKSPACE = Path(os.environ.get("UNSLOTH_WORKSPACE", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(WORKSPACE))

from pr_ui_scenes._common import Session, api_get  # noqa: E402
from pr_ui_scenes.parallel_chat_streaming import (  # noqa: E402
    _ASSISTANT_TEXT_JS,
    _api_put,
    _load_and_wait,
    _tile,
    _unload,
)

DEFAULT_REPO = "unsloth/Qwen3.5-4B-MTP-GGUF"
DEFAULT_VARIANT = "UD-Q4_K_XL"
PAUSED_LABEL = "Paused while another chat finishes"
WAITING_LABEL = "Waiting for a free slot"

_FILLER = (
    "Consider the following background material carefully before answering, as it "
    "establishes the context, constraints and prior art that your answer must address "
    "in full detail without omitting anything of substance. "
)
_ASKS = [
    "Write a detailed history of the printing press and its effect on literacy.",
    "Explain in depth how a compiler turns source code into machine code.",
    "Describe the full life cycle of a star, from nebula to remnant, in detail.",
    "Give a thorough account of how vaccines train the immune system.",
]


def _prompt(index: int, prompt_tokens: int) -> str:
    ask = _ASKS[index % len(_ASKS)]
    reps = max(0, prompt_tokens * 4 // len(_FILLER))
    return (_FILLER * reps) + "\n\n" + ask


async def _drive_one(browser, base_url, init_script, index, prompt, out_dir, label, deadline_s):
    ctx = await browser.new_context(viewport={"width": 900, "height": 760})
    await ctx.add_init_script(init_script)
    page = await ctx.new_page()
    out = {"index": index, "paused_seen": False, "waiting_seen": False, "finished": False,
           "error_box": False, "final_chars": 0, "elapsed_s": None, "shot": None}
    t0 = time.monotonic()
    try:
        await page.goto(f"{base_url}/chat?new={uuid.uuid4()}", wait_until="domcontentloaded")
        box = page.locator("form textarea, textarea").first
        await box.wait_for(state="visible", timeout=60_000)
        await box.fill(prompt)
        await box.press("Enter")
        stop = page.locator('button[aria-label="Stop generating"], button:has-text("Stop")').first
        paused = page.get_by_text(PAUSED_LABEL, exact=False)
        waiting = page.get_by_text(WAITING_LABEL, exact=False)
        error_box = page.locator(
            '[role="alert"], .text-destructive:not(button):not(:has(button)), [data-status="error"]'
        )
        seen_stop = False
        while time.monotonic() - t0 < deadline_s:
            try:
                if not out["paused_seen"] and await paused.count() > 0 and await paused.first.is_visible():
                    out["paused_seen"] = True
                    await page.screenshot(path=str(out_dir / f"{label.lower()}_chat{index}_paused.png"))
                if not out["waiting_seen"] and await waiting.count() > 0:
                    out["waiting_seen"] = True
                if await stop.is_visible():
                    seen_stop = True
                elif seen_stop:
                    out["finished"] = True
                    break
            except Exception:  # noqa: BLE001 -- a detached node mid-render, poll again
                pass
            await asyncio.sleep(0.25)
        out["elapsed_s"] = round(time.monotonic() - t0, 1)
        await asyncio.sleep(1.0)
        shot = out_dir / f"{label.lower()}_chat{index}_final.png"
        await page.screenshot(path=str(shot))
        out["shot"] = shot
        out["error_box"] = (await error_box.count()) > 0
        out["final_chars"] = len(await page.evaluate(_ASSISTANT_TEXT_JS))
    finally:
        await ctx.close()
    return out


def _metrics(session: Session) -> dict:
    """How the load came out: slots and context as Studio reports them."""
    try:
        status = api_get(session, "/api/inference/status") or {}
    except Exception:  # noqa: BLE001
        return {}
    keep = ("exact_concurrency", "parallel_slots", "context_length")
    return {k: status.get(k) for k in keep if k in status}


async def drive(session: Session, out_dir: Path, label: str,
                repo: str = DEFAULT_REPO, gguf_variant: str = DEFAULT_VARIANT,
                n_chats: int = 4, context_length: int = 8192, prompt_tokens: int = 1500,
                deadline_s: float = 600.0, **_: object) -> tuple[list[Path], dict]:
    from playwright.async_api import async_playwright

    status = _load_and_wait(session, repo, gguf_variant, context_length)
    facts: dict = {
        "loaded": status.get("loaded"),
        "context_length": status.get("context_length") or context_length,
        "parallel_slots": status.get("parallel_slots"),
        "n_chats": n_chats,
        "prompt_tokens": prompt_tokens,
    }
    # Studio's "Max" (no stated cap): the case that collides. Tools off, so a pane waiting
    # on a tool cannot be mistaken for a pane waiting for cache room.
    _api_put(session, "/api/chat/settings", {
        "toolsEnabled": False,
        "permissionMode": "auto",
        "inferenceParams": {"maxTokens": 0, "maxSeqLength": context_length},
    }, timeout=120)

    init_script = (
        f"localStorage.setItem('unsloth_auth_token', {json.dumps(session.access_token)});"
        f"localStorage.setItem('unsloth_refresh_token', {json.dumps(session.refresh_token)});"
    )
    async with async_playwright() as p:
        browser = await p.chromium.launch(args=["--no-sandbox"])
        results = await asyncio.gather(*[
            _drive_one(browser, session.base_url, init_script, i, _prompt(i, prompt_tokens),
                       out_dir, label, deadline_s)
            for i in range(n_chats)
        ])
        await browser.close()

    results = sorted(results, key=lambda r: r["index"])
    facts["panes_paused"] = sum(1 for r in results if r["paused_seen"])
    facts["panes_waited"] = sum(1 for r in results if r["waiting_seen"])
    facts["panes_with_error_box"] = sum(1 for r in results if r["error_box"])
    facts["panes_finished"] = sum(1 for r in results if r["finished"])
    facts["final_chars"] = [r["final_chars"] for r in results]
    facts["elapsed_s"] = [r["elapsed_s"] for r in results]
    facts.update(_metrics(session))
    shots = [r["shot"] for r in results if r["shot"]]
    tile = _tile(shots, out_dir / f"{label.lower()}_tiled.png")
    _unload(session, repo)
    print(json.dumps({"label": label, **{k: facts[k] for k in ("panes_paused", "panes_with_error_box",
                                                              "panes_finished")}}), file=sys.stderr)
    return [tile], facts
