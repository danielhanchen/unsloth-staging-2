"""Scene: the Video page's denoise bar during a real MiniMax-H3 render (PR 11004).

The bar is drawn from `GET /api/inference/video/generate-progress`, which on the base
counts what the HOST has submitted. With the denoiser replayed from a captured CUDA graph
the host enqueues all 30 steps in about a second and then blocks, so the bar sprints to
the end of the denoise while the GPU has 26 s of work left, and the user watches an
apparent freeze at the far right of a bar that claims a second remains. The head reports
the step the GPU has FINISHED, read from `torch.cuda.Event.query()`.

So the scene photographs the bar at two fixed moments inside ONE warm render, and records
the whole progress stream beside the picture: a screenshot shows the two bars differ, only
the stream says which one was telling the truth.

Warm, not cold, on purpose: the first render after a load also pays graph capture and
compile, which moves the wall clock around and would make the two sides' timelines
incomparable. The cold render is run first and thrown away.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

WORKSPACE = Path(os.environ.get("WORKSPACE") or os.environ.get("UNSLOTH_WORKSPACE") or Path(__file__).resolve().parents[2])
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "scripts"))

from pr_ui_scenes._common import Session, api_get, api_post  # noqa: E402
from studio_test_kit.auth import seed_init_script  # noqa: E402
from studio_test_kit.ui import open_chat  # noqa: E402

DEFAULT_REPO = "MiniMaxAI/MiniMax-H3"
PROMPT = (
    "Ultra-realistic cinematic documentary footage of a quiet Kyoto neighborhood at "
    "sunrise. An elderly Japanese man opens his traditional wooden shop while a young "
    "woman wearing a simple kimono walks past."
)
# The two moments the bar is photographed at, seconds after the render is accepted. The
# first is inside the stretch where the host has already enqueued every step; the second
# is deep enough into the GPU's work that an honest bar has visibly moved and a host-side
# one has not moved at all.
SHOT_TIMES = (7.0, 20.0)
# Matches the label the Video page writes for the denoise phase, "Denoising step 7/30".
_DENOISE = re.compile(r"Denoising", re.I)


def _wait_loaded(session: Session, timeout_s: int = 2400) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        status = api_get(session, "/api/inference/video/status")
        if status.get("loaded"):
            return status
        progress = api_get(session, "/api/inference/video/load-progress")
        if progress.get("error"):
            raise RuntimeError(f"load failed: {str(progress['error'])[:400]}")
        time.sleep(5)
    raise RuntimeError(f"model did not load within {timeout_s}s")


def _generate(session: Session, seed: int, steps: int, width: int, height: int,
              frames: int) -> None:
    api_post(session, "/api/inference/video/generate", {
        "prompt": PROMPT, "width": width, "height": height, "num_frames": frames,
        "fps": 24, "steps": steps, "guidance": 1.0, "seed": seed,
        "flow_shift": 12.0, "audio_flow_shift": 3.0,
    }, timeout=120)


def _drain(session: Session, timeout_s: int = 1800) -> dict:
    """Poll until the render reaches a terminal phase, and return the last payload."""
    deadline = time.time() + timeout_s
    payload: dict = {}
    while time.time() < deadline:
        payload = api_get(session, "/api/inference/video/generate-progress")
        if payload.get("phase") in ("completed", "failed"):
            return payload
        time.sleep(0.5)
    raise RuntimeError(f"render did not finish within {timeout_s}s: {json.dumps(payload)[:300]}")


class _Stream:
    """Records every distinct progress payload with its offset from the POST.

    This is the numeric half of the evidence, and it is taken from the SAME server that
    is being photographed, through the same route the bar reads.
    """

    def __init__(self, session: Session) -> None:
        self.session = session
        self.events: list[dict] = []
        self._stop = threading.Event()
        self._t0 = 0.0
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._t0 = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        last = None
        while not self._stop.is_set():
            try:
                p = self.session and api_get(self.session, "/api/inference/video/generate-progress")
            except Exception:  # noqa: BLE001 -- a dropped poll must not end the recording
                time.sleep(0.2)
                continue
            key = (p.get("phase"), p.get("step"), p.get("total"))
            if key != last:
                last = key
                self.events.append({
                    "t": round(time.monotonic() - self._t0, 2),
                    "phase": p.get("phase"), "step": p.get("step"),
                    "total": p.get("total"),
                })
            if p.get("phase") in ("completed", "failed"):
                return
            time.sleep(0.1)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def at(self, seconds: float) -> dict:
        """The payload the bar was drawing at `seconds` into the render."""
        seen = [e for e in self.events if e["t"] <= seconds]
        return seen[-1] if seen else {}

    def first_full_denoise_t(self) -> float | None:
        """When the reported step FIRST reached the step count, in seconds.

        This single number is the whole claim: on the base it lands about a second in,
        with most of the render still ahead of it.
        """
        for e in self.events:
            if e.get("phase") == "denoise" and e.get("total") and e.get("step") == e.get("total"):
                return e["t"]
        return None

    def denoise_end_t(self) -> float | None:
        """When the denoise phase gave way to whatever runs next."""
        seen_denoise = False
        for e in self.events:
            if e.get("phase") == "denoise":
                seen_denoise = True
            elif seen_denoise:
                return e["t"]
        return None


async def _shoot_progress(page, out_dir: Path, label: str, index: int) -> Path:
    """The progress card, framed on the denoise label rather than the whole page.

    A 1500 px pair renders about 440 px per half in a comment, and the claim here is a
    line of text plus a bar, so a full-page shot would be unreadable exactly where it
    matters. Falls back to the page when the label is not on screen -- which is itself
    worth photographing, since it means the phase moved on.
    """
    shot = out_dir / f"{label.lower()}_t{index}.png"
    # Two levels up from the label, which is the card holding the phase line, the bar and the ETA.
    # The label itself would crop to one line of text and the bar is half the claim.
    card = page.get_by_text(_DENOISE).first.locator("xpath=ancestor::div[2]")
    try:
        await card.screenshot(path=str(shot), timeout=10_000)
    except Exception:  # noqa: BLE001 -- no denoise label is a result, not a crash
        await page.screenshot(path=str(shot))
    return shot


async def drive(session: Session, out_dir: Path, label: str,
                repo: str = DEFAULT_REPO, transformer_quant: str = "fp8",
                text_encoder_quant: str = "", steps: int = 30,
                width: int = 960, height: int = 544, frames: int = 124,
                **_: object) -> tuple[list[Path], dict]:
    hf_token = os.environ.get("UIDIFF_HF_TOKEN") or os.environ.get("HF_TOKEN") or ""
    body = {
        "model_path": repo, "model_kind": "pipeline",
        "transformer_quant": transformer_quant,
        "speed_mode": "default", "hf_token": hf_token,
    }
    # Only when asked for. MiniMax-H3 has no hosted quantised conditioner, and naming one puts a
    # refusal in load-progress rather than quietly falling back, which reads as a failed load.
    if text_encoder_quant:
        body["text_encoder_quant"] = text_encoder_quant
    api_post(session, "/api/inference/video/load", body, timeout=2400)
    status = _wait_loaded(session)
    facts: dict = {
        "loaded": status.get("loaded"),
        "transformer_quant": status.get("transformer_quant"),
        "speed_optims": status.get("speed_optims"),
        "steps": steps,
    }

    # Cold render, discarded: it pays CUDA-graph capture and compile, and a timeline that
    # includes them is not comparable between the two sides.
    _generate(session, 2054899255, steps, width, height, frames)
    cold = _drain(session)
    facts["cold_phase"] = cold.get("phase")

    shots: list[Path] = []
    init = seed_init_script(
        type("A", (), {"access_token": session.access_token,
                       "refresh_token": session.refresh_token})(), []
    )
    async with open_chat(session.base_url, init_scripts=[init],
                         viewport=(1400, 900), headless=True) as sp:
        page = sp.page
        # Mounted once first, purely to warm the asset cache so the reload below costs a second
        # rather than ten.
        await page.goto(f"{session.base_url}/video", wait_until="domcontentloaded")
        await page.wait_for_timeout(8_000)

        stream = _Stream(session)
        started = time.monotonic()
        _generate(session, 2054899256, steps, width, height, frames)
        stream.start()
        # The page starts its progress poll when a generation is in flight AS IT MOUNTS (the
        # mount-time resume that exists so a reload mid-denoise does not show an idle page), so the
        # mount has to come after the POST. A page that was already sitting there when the render
        # began never begins polling, and photographs the idle composer on both sides -- which is
        # exactly the identical pair the first run produced.
        await page.reload(wait_until="domcontentloaded")
        for i, when in enumerate(SHOT_TIMES):
            remaining = when - (time.monotonic() - started)
            if remaining > 0:
                await page.wait_for_timeout(int(remaining * 1000))
            shots.append(await _shoot_progress(page, out_dir, label, i))
            facts[f"bar_at_{int(when)}s"] = stream.at(when)
        wall_to_last_shot = time.monotonic() - started
        while stream.events and stream.events[-1].get("phase") not in ("completed", "failed"):
            await page.wait_for_timeout(2_000)
            if time.monotonic() - started > 1800:
                break
        stream.stop()

    facts["render_wall_s"] = round(time.monotonic() - started, 1)
    facts["shots_taken_by_s"] = round(wall_to_last_shot, 1)
    facts["reported_full_at_s"] = stream.first_full_denoise_t()
    facts["denoise_phase_ended_at_s"] = stream.denoise_end_t()
    facts["distinct_progress_events"] = len(stream.events)
    (out_dir / f"{label.lower()}_stream.json").write_text(json.dumps(stream.events, indent=2))

    api_post(session, "/api/inference/video/unload", {}, timeout=600)
    return shots, facts


if __name__ == "__main__":
    import argparse

    from pr_ui_scenes._common import studio_session

    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--home", type=Path, required=True)
    ap.add_argument("--password", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--label", default="AFTER")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    s = studio_session(a.url, a.home, a.password)
    print(asyncio.run(drive(s, a.out, a.label)))
