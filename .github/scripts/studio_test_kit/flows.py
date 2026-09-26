"""High-level reusable scenarios.

Each flow:
  - takes a `StudioPage` already at /chat with init scripts seeded
  - drives the UI through a real round-trip against the live provider
  - saves numbered screenshots into `out_dir`
  - returns a small dict of artefacts the caller can assert on

The flows are deliberately small so you can compose them: a "vision
upload then ask about it" test is `vision_upload` followed by
`multi_turn_chat`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .ui import (
    StudioPage,
    extract_data_url,
    pick_model,
    send_prompt,
    set_pill,
    wait_for_image,
    wait_for_stream,
)


@dataclass
class FlowResult:
    out_dir: Path
    screenshots: list[Path] = field(default_factory=list)
    artefacts: dict = field(default_factory=dict)
    # Populated AFTER the open_chat context exits (Playwright flushes the
    # webm on context.close). Callers that want the video should read
    # these via `sp.video_webm` / `sp.video_mp4` once the `async with`
    # block has completed, then copy onto the FlowResult themselves, or
    # use `flow_result.attach_video(sp)` after the block.
    video_webm: Optional[Path] = None
    video_mp4: Optional[Path] = None

    def shot(self, sp: StudioPage, idx: int, name: str) -> Path:
        path = self.out_dir / f"{idx:02d}_{name}.png"
        self.screenshots.append(path)
        return path

    def attach_video(self, sp: StudioPage) -> "FlowResult":
        """Copy video paths from the StudioPage (call after `async with` exits)."""
        self.video_webm = sp.video_webm
        self.video_mp4 = sp.video_mp4
        return self


async def multi_turn_chat(
    sp: StudioPage,
    model: str,
    prompts: list[str],
    out_dir: Path,
    settle_timeout_ms: int = 90_000,
) -> FlowResult:
    """Pick `model`, then send each prompt in sequence on the same thread.

    Screenshots are numbered in chronological capture order, so a
    sorted-by-filename viewing (the default in most file managers)
    matches the actual narrative: open, pick, sent-1, done-1, sent-2,
    done-2, ...
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    r = FlowResult(out_dir=out_dir)
    idx = 1
    await sp.screenshot(r.shot(sp, idx, "chat_open")); idx += 1
    await pick_model(sp, model)
    await sp.screenshot(r.shot(sp, idx, "model_picked")); idx += 1
    for i, prompt in enumerate(prompts, start=1):
        await send_prompt(sp, prompt)
        await sp.screenshot(r.shot(sp, idx, f"turn_{i:02d}_sent")); idx += 1
        await wait_for_stream(sp, timeout_ms=settle_timeout_ms)
        await sp.screenshot(r.shot(sp, idx, f"turn_{i:02d}_done")); idx += 1
    r.artefacts["turn_count"] = len(prompts)
    return r


async def image_generation(
    sp: StudioPage,
    model: str,
    prompt: str,
    out_dir: Path,
    image_basename: str = "generated_image",
    timeout_ms: int = 120_000,
) -> FlowResult:
    """Pick an image-gen model, toggle the Images pill, send a prompt,
    wait for a `data:image/png` to render, and save the decoded bytes."""
    out_dir.mkdir(parents=True, exist_ok=True)
    r = FlowResult(out_dir=out_dir)
    await sp.screenshot(r.shot(sp, 1, "chat_open"))
    await pick_model(sp, model)
    await sp.screenshot(r.shot(sp, 2, "model_picked"))
    await set_pill(sp, "images", on=True)
    await sp.screenshot(r.shot(sp, 3, "image_pill_on"))
    await send_prompt(sp, prompt)
    await sp.screenshot(r.shot(sp, 4, "prompt_sent"))
    data_url = await wait_for_image(sp, timeout_ms=timeout_ms)
    await sp.screenshot(r.shot(sp, 5, "image_response"))
    image_path = out_dir / f"{image_basename}.png"
    image_path.write_bytes(await extract_data_url(data_url))
    r.artefacts["image_bytes"] = image_path.stat().st_size
    r.artefacts["image_path"] = str(image_path)
    return r


async def tool_pills(
    sp: StudioPage,
    model: str,
    out_dir: Path,
    search_prompt: Optional[str] = "Who won the 2024 NBA championship?",
    code_prompt: Optional[str] = "Use Python to compute the sum of the first 100 primes.",
) -> FlowResult:
    """Exercise Search + Code composer pills on the same model."""
    out_dir.mkdir(parents=True, exist_ok=True)
    r = FlowResult(out_dir=out_dir)
    await sp.screenshot(r.shot(sp, 1, "chat_open"))
    await pick_model(sp, model)
    await sp.screenshot(r.shot(sp, 2, "model_picked"))
    if search_prompt:
        await set_pill(sp, "search", on=True)
        await sp.screenshot(r.shot(sp, 3, "search_pill_on"))
        await send_prompt(sp, search_prompt)
        await wait_for_stream(sp)
        await sp.screenshot(r.shot(sp, 4, "search_response"))
        await set_pill(sp, "search", on=False)
    if code_prompt:
        await set_pill(sp, "code", on=True)
        await sp.screenshot(r.shot(sp, 5, "code_pill_on"))
        await send_prompt(sp, code_prompt)
        await wait_for_stream(sp)
        await sp.screenshot(r.shot(sp, 6, "code_response"))
    await sp.screenshot(r.shot(sp, 7, "thread_full"))
    return r


async def vision_upload(
    sp: StudioPage,
    model: str,
    image_path: Path,
    prompt: str,
    out_dir: Path,
) -> FlowResult:
    """Pick a vision-capable model, attach `image_path` through the file
    input, ask `prompt`, wait for completion."""
    out_dir.mkdir(parents=True, exist_ok=True)
    r = FlowResult(out_dir=out_dir)
    await sp.screenshot(r.shot(sp, 1, "chat_open"))
    await pick_model(sp, model)
    await sp.screenshot(r.shot(sp, 2, "model_picked"))
    file_input = sp.page.locator('form:has(textarea) input[type="file"]').first
    await file_input.set_input_files(str(image_path))
    await sp.screenshot(r.shot(sp, 3, "image_attached"))
    await send_prompt(sp, prompt)
    await sp.screenshot(r.shot(sp, 4, "prompt_sent"))
    await wait_for_stream(sp)
    await sp.screenshot(r.shot(sp, 5, "vision_response"))
    return r
