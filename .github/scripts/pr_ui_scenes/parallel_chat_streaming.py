"""Scene: four chat threads with tools on, sent together (PR 10046).

Studio reserved the WHOLE KV cache for any request that could enter a tool loop, because
a loop re-sends its conversation every round and admission could not know the final size.
That is airtight and it pins concurrency to one: any lit pill sets `enable_tools`, so it
is the common case, not a corner. This PR charges what a run currently occupies and
re-costs it as it grows, so the loops run together.

The visible claim is therefore about SIMULTANEITY, which needs more than one pane in the
shot. Four chats are driven in four separate browser CONTEXTS (independent threads, not
four tabs sharing one), photographed at a fixed moment after they were all sent, and
tiled into one image per side.

Two things this scene has to get right or it photographs nothing:

  the token cap   Left on Studio's "Max", `max_tokens` equals the context window, the
                  opening reservation is `max(share, prompt + output)` clamped to the
                  budget, and that is the whole cache on BOTH sides. The head would
                  serialise too and the pair would read as "this PR changed nothing".
                  Pinned to 512 here, on both sides, via install-scoped settings.
  first token     NOT the Stop button. Studio shows Stop within about 0.13s on every
                  pane whether it is decoding or waiting for admission, so a Stop probe
                  reports four "streaming" threads on a side where three are parked.
                  The signal is assistant TEXT appearing in the DOM.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import urllib.request
import uuid
from pathlib import Path

WORKSPACE = Path(os.environ.get("UNSLOTH_WORKSPACE", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "scripts"))

from pr_ui_scenes._common import Session, api_get, api_post  # noqa: E402

DEFAULT_REPO = "unsloth/Qwen3.5-4B-MTP-GGUF"
DEFAULT_VARIANT = "UD-Q4_K_XL"

# The CONTENT div, not the message root. The root also holds the action bar (Copy, Edit,
# Delete, More) and a compaction notice, so its innerText can be non-empty on a turn that
# has produced no tokens at all -- which would report every pane as "producing" and make
# the two sides identical, the one failure this whole exercise is guarding against.
# Placeholders are NOT text. "Generating..." (thread.tsx) is rendered into the content div
# the moment the turn opens, before a single token exists, and "Thinking..."
# (reasoning.tsx) is its reasoning twin. Counting them measured the placeholder appearing
# rather than the model producing: every pane reported a first token inside 0.2s on BOTH
# sides, at 13 characters, which is exactly len("Generating...").
_ASSISTANT_TEXT_JS = """() => {
    const nodes = document.querySelectorAll('[data-role="assistant"]');
    if (!nodes.length) return "";
    const last = nodes[nodes.length - 1];
    const body = last.querySelector('.aui-assistant-message-content') || last;
    const text = (body.innerText || "").trim();
    if (text === "Generating..." || text === "Thinking...") return "";
    return text;
}"""


# Four different questions, so a pane cannot be confused with its neighbour in the tile
# and a cached prefix cannot serve two of them. Each asks for a few sentences: long
# enough that the first thread is still generating when the shot is taken, which is the
# moment the other three are either waiting (before) or running beside it (after).
PROMPTS = [
    "In three or four sentences, what makes the ocean look blue?",
    "In three or four sentences, how do mountain ranges form?",
    "In three or four sentences, why are deserts dry?",
    "In three or four sentences, how do forests store carbon?",
]


def _api_put(session: Session, path: str, payload: dict, timeout: int = 120) -> dict:
    """PUT, which `_common` has no helper for. Chat settings are saved with PUT; POSTing
    them answers 405."""
    req = urllib.request.Request(
        f"{session.base_url}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {session.access_token}"},
        method="PUT",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _unload(session: Session, repo: str) -> None:
    """Best effort. UnloadRequest requires model_path, and unloading when nothing is
    resident is a 4xx, not a failure worth aborting a shoot over."""
    try:
        api_post(session, "/api/inference/unload", {"model_path": repo}, timeout=300)
    except Exception:  # noqa: BLE001 -- nothing loaded, or already gone
        pass


def _load_and_wait(session: Session, repo: str, variant: str, ctx: int,
                   timeout_s: int = 1_200) -> dict:
    _unload(session, repo)
    api_post(session, "/api/inference/load",
             {"model_path": repo, "gguf_variant": variant, "max_seq_length": ctx},
             timeout=timeout_s)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        status = api_get(session, "/api/inference/status")
        if status.get("loaded"):
            return status
        time.sleep(5)
    raise RuntimeError(f"model did not load within {timeout_s}s")


async def _drive_one(browser, base_url: str, init_script: str, prompt: str, index: int,
                     out_dir: Path, label: str, shot_at_s: float, max_wait_s: float,
                     results: list, take_shot: bool = True) -> None:
    """One chat, in its own context, timestamped on its first token of text."""
    ctx = await browser.new_context(viewport={"width": 900, "height": 760})
    await ctx.add_init_script(init_script)
    page = await ctx.new_page()
    # A fresh thread id per pane: without it the four contexts land on the same
    # conversation and photograph one thread four times.
    await page.goto(f"{base_url}/chat?new={uuid.uuid4()}", wait_until="domcontentloaded")
    box = page.locator("form textarea, textarea").first
    await box.wait_for(state="visible", timeout=60_000)
    await box.fill(prompt)

    started = asyncio.get_running_loop().time()
    await box.press("Enter")

    # Poll the assistant turn for TEXT. The Stop button is not the signal (see module
    # docstring); an empty assistant bubble with a spinner is exactly what a parked
    # thread looks like, and it is what the before side is expected to show.
    first_token_s = None
    chars_at_shot = 0
    shot_taken = False
    shot_path = out_dir / f"{label.lower()}_chat{index}.png"
    # Two separate things, and an earlier version conflated them: the SHOT is at a fixed
    # moment so the four panes are photographed together, but the first-token TIME has to
    # be followed to the end. Stopping at the shot recorded every still-waiting pane as
    # null, which cannot be told apart from a pane that started a half second later --
    # and the staircase, the whole quantitative claim, lives in those numbers.
    while True:
        elapsed = asyncio.get_running_loop().time() - started
        if first_token_s is None:
            try:
                text = await page.evaluate(_ASSISTANT_TEXT_JS)
            except Exception:  # noqa: BLE001 -- a navigation mid-evaluate is not a result
                text = ""
            if text:
                first_token_s = round(elapsed, 2)
        if not shot_taken and elapsed >= shot_at_s:
            # Only the photographed round writes a file. Later rounds share the same
            # filename, so shooting them would overwrite the evidence with a frame taken
            # at a different moment.
            if take_shot:
                await page.screenshot(path=str(shot_path))
            shot_taken = True
            visible = await page.evaluate(_ASSISTANT_TEXT_JS)
            chars_at_shot = len(visible)
        if shot_taken and (first_token_s is not None or elapsed >= max_wait_s):
            results.append({
                "index": index,
                "first_token_s": first_token_s,
                "text_at_shot": chars_at_shot > 0,
                "chars_at_shot": chars_at_shot,
                "shot": shot_path,
            })
            break
        await page.wait_for_timeout(150)

    # Stop the run rather than leave four generations on a shared GPU. Filtered on the
    # aria-label, which is what the composer actually sets.
    try:
        stop = page.locator('button[aria-label="Stop generating"]')
        if await stop.count():
            await stop.first.click()
    except Exception:  # noqa: BLE001 -- best effort teardown
        pass
    await ctx.close()


def _tile(shots: list[Path], out: Path, cols: int = 2) -> Path:
    """Lay the panes out on a fixed grid.

    Padded onto a common cell, never scaled to match: scaling makes one pane larger than
    its neighbour and the composite reads as retouched.
    """
    from PIL import Image

    images = [Image.open(p).convert("RGB") for p in shots]
    cell_w = max(im.width for im in images)
    cell_h = max(im.height for im in images)
    rows = (len(images) + cols - 1) // cols
    gap = 12
    canvas = Image.new(
        "RGB",
        (cols * cell_w + (cols + 1) * gap, rows * cell_h + (rows + 1) * gap),
        (24, 24, 27),
    )
    for i, im in enumerate(images):
        x = gap + (i % cols) * (cell_w + gap)
        y = gap + (i // cols) * (cell_h + gap)
        canvas.paste(im, (x, y))
    canvas.save(out)
    return out


async def drive(session: Session, out_dir: Path, label: str,
                repo: str = DEFAULT_REPO, gguf_variant: str = DEFAULT_VARIANT,
                n_chats: int = 4, context_length: int = 32768, max_tokens: int = 512,
                shot_at_s: float = 8.0, max_wait_s: float = 120.0, rounds: int = 3,
                **_: object) -> tuple[list[Path], dict]:
    from playwright.async_api import async_playwright

    status = _load_and_wait(session, repo, gguf_variant, context_length)
    facts: dict = {
        "loaded": status.get("loaded"),
        "gguf_variant": status.get("gguf_variant"),
        "context_length": status.get("context_length") or context_length,
        "n_chats": n_chats,
        "shot_at_s": shot_at_s,
    }

    # Install-scoped, so all four contexts read the same thing back. permissionMode auto
    # keeps a tool-confirmation dialog from parking a thread and faking the finding.
    # maxTokens is the load-bearing one: see the module docstring.
    # toolsEnabled is the only one that matters for the claim: admission keys the tool
    # loop on the resolved execution path, not on a client `tools` array, so the loop (and
    # with it the whole-cache reservation this PR removes) opens on the pill alone whether
    # or not the model goes on to call anything.
    #
    # The rest are off ON PURPOSE. With web search lit a pane sits on "Generating..."
    # while a TOOL runs, which is indistinguishable in a screenshot from a pane waiting
    # for cache room, and it puts a network round trip inside the measurement.
    _api_put(session, "/api/chat/settings", {
        "toolsEnabled": True,
        # Left ON. With every sub-tool off the catalogue is empty, and an empty catalogue
        # is indistinguishable from a plain chat: the tool loop this PR changes is never
        # entered, both sides run four at a time, and the pair measures browser noise.
        "codeToolsEnabled": True,
        # Web search stays off: it puts a network round trip inside the measurement, and
        # a pane waiting on a search looks exactly like a pane waiting for cache room.
        "webFetchToolsEnabled": False,
        "imageToolsEnabled": False,
        "deepResearchEnabled": False,
        "permissionMode": "auto",
        "inferenceParams": {"maxTokens": max_tokens, "maxSeqLength": context_length},
    }, timeout=120)
    # The response nests everything under "settings"; reading the top level returned None
    # for both keys and silently skipped the check that the run was configured at all.
    settings = (api_get(session, "/api/chat/settings") or {}).get("settings", {})
    facts["tools_enabled"] = settings.get("toolsEnabled")
    facts["max_tokens"] = (settings.get("inferenceParams") or {}).get("maxTokens")
    if not facts["tools_enabled"]:
        raise RuntimeError(
            "tools are not enabled, so no tool loop opens and the reservation this PR "
            f"changes is never taken: settings={json.dumps(settings)[:400]}"
        )
    if facts["max_tokens"] != max_tokens:
        raise RuntimeError(
            f"maxTokens is {facts['max_tokens']}, not {max_tokens}. Left uncapped the "
            "opening reservation is the whole cache on BOTH sides and the pair proves "
            "nothing."
        )

    init_script = (
        f"localStorage.setItem('unsloth_auth_token', {json.dumps(session.access_token)});"
        f"localStorage.setItem('unsloth_refresh_token', {json.dumps(session.refresh_token)});"
    )

    # Repeated, because one burst does not settle this. A single pair put four panes on
    # the base and two on the head, and the next pair on the SAME two builds put one on
    # the base and two on the head: run-to-run spread on this box is the same size as the
    # effect being claimed, so a single burst can be read either way. Only round 0 is
    # photographed; every round is measured.
    rounds_facts: list = []
    results: list = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(args=["--no-sandbox"])
        for round_index in range(max(1, rounds)):
            round_results: list = []
            await asyncio.gather(*[
                _drive_one(browser, session.base_url, init_script,
                           PROMPTS[i % len(PROMPTS)], i, out_dir, label,
                           shot_at_s, max_wait_s, round_results,
                           round_index == 0)
                for i in range(n_chats)
            ])
            round_results.sort(key=lambda r: r["index"])
            rf = [r["first_token_s"] for r in round_results]
            ok = [v for v in rf if v is not None]
            rounds_facts.append({
                "round": round_index,
                "first_token_s": rf,
                "spread_s": round(max(ok) - min(ok), 2) if len(ok) == n_chats else None,
                "last_first_token_s": max(ok) if ok else None,
            })
            if round_index == 0:
                results = round_results
        await browser.close()

    firsts = [r["first_token_s"] for r in results if r["first_token_s"] is not None]
    facts["rounds"] = rounds_facts
    # Median of the per-round "when did the LAST of the four start" figure. That is the
    # serialisation in one number, and a median over rounds is what makes it a claim
    # rather than a sample.
    lasts = sorted(r["last_first_token_s"] for r in rounds_facts
                   if r["last_first_token_s"] is not None)
    facts["median_last_first_token_s"] = (
        round(lasts[len(lasts) // 2], 2) if lasts else None
    )
    spreads = sorted(r["spread_s"] for r in rounds_facts if r["spread_s"] is not None)
    facts["median_spread_s"] = round(spreads[len(spreads) // 2], 2) if spreads else None
    facts["first_token_s"] = [r["first_token_s"] for r in results]
    facts["chars_at_shot"] = [r["chars_at_shot"] for r in results]
    # The number that carries the claim: how many panes were producing text in the one
    # photographed moment. A picture shows the panes differ; this says by how much.
    facts["panes_with_text_at_shot"] = sum(1 for r in results if r["text_at_shot"])
    facts["first_token_spread_s"] = (
        round(max(firsts) - min(firsts), 2) if len(firsts) == n_chats else None
    )
    facts["panes_that_never_started"] = n_chats - len(firsts)
    facts["max_wait_s"] = max_wait_s
    # The staircase in one number: how long the LAST pane waited for its first token.
    facts["last_first_token_s"] = max(firsts) if firsts else None

    tile = _tile([r["shot"] for r in results], out_dir / f"{label.lower()}_tiled.png")
    _unload(session, repo)
    return [tile], facts
