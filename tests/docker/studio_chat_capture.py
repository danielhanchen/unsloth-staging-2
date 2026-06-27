#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0
"""Capture Studio chat with the locally-loaded GGUF (screenshots + a WebM).

Reuses the working session from feature_probe.py (out-dir/auth.json: base_url +
JWT) so it does NOT rotate the password again, and drives the chat composer with
the studio_test_kit cookbook (open_chat -> send_prompt -> wait_for_stream ->
screenshot, recording a WebM the posting step converts to a GIF).

The locally-loaded model (POST /api/inference/load, done by feature_probe) is the
active model, so we send prompts directly; model selection is best-effort.

Run AFTER feature_probe.py. Never hard-fails: a capture problem is logged and the
script still exits 0 so it cannot break the CI job (screenshots are evidence, not
gates -- the probe is the gate).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

# studio_test_kit must be importable (vendored next to this tree or on sys.path).
from studio_test_kit.auth import StudioAuth, seed_init_script
from studio_test_kit.ui import open_chat, send_prompt, wait_for_stream, pick_model


async def run(auth_json: Path, out_dir: Path, prompts: list[str], model_hint: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    info = json.loads(auth_json.read_text())
    base_url = info["base_url"]
    auth = StudioAuth(access_token=info["token"], refresh_token="", base_url=base_url)
    init = seed_init_script(auth, [])  # JWT only -- no external provider

    shots: list[str] = []
    async with open_chat(
        base_url,
        init_scripts=[init],
        video_dir=out_dir / "video",
        video_name="studio_chat",
        transcode_mp4=True,
        headless=True,
    ) as sp:
        # Best-effort: select the loaded GGUF if the picker exposes it; the
        # loaded model is usually already the active default in chat-only mode.
        if model_hint:
            try:
                await pick_model(sp, model_hint, timeout_ms=8000)
            except Exception as e:  # noqa: BLE001
                print(f"[capture] pick_model('{model_hint}') skipped: {e!r}"[:160])
        await sp.screenshot(out_dir / "01_chat_open.png", full_page=False)
        shots.append("01_chat_open.png")
        for i, prompt in enumerate(prompts, start=1):
            try:
                await send_prompt(sp, prompt)
                await wait_for_stream(sp, timeout_ms=180_000)
                await sp.screenshot(out_dir / f"{i+1:02d}_turn_{i}.png", full_page=False)
                shots.append(f"{i+1:02d}_turn_{i}.png")
            except Exception as e:  # noqa: BLE001
                print(f"[capture] turn {i} problem: {e!r}"[:200])
                await sp.screenshot(out_dir / f"{i+1:02d}_turn_{i}_partial.png", full_page=False)
                shots.append(f"{i+1:02d}_turn_{i}_partial.png")

    manifest = {
        "screenshots": shots,
        "video_webm": str(sp.video_webm) if sp.video_webm else None,
        "video_mp4": str(sp.video_mp4) if sp.video_mp4 else None,
    }
    (out_dir / "capture_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[capture] screenshots={shots} webm={sp.video_webm} mp4={sp.video_mp4}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--auth-json", required=True, help="feature_probe out-dir/auth.json")
    ap.add_argument("--out-dir", default="chat_capture")
    ap.add_argument("--model-hint", default="", help="exact model picker label, if needed")
    ap.add_argument(
        "--prompts",
        nargs="*",
        default=[
            "In one sentence, what is Unsloth and what is it best known for?",
            "Now name the cute animal mascot and what it represents.",
        ],
    )
    args = ap.parse_args(argv)
    try:
        asyncio.run(run(Path(args.auth_json), Path(args.out_dir), args.prompts, args.model_hint))
    except Exception as e:  # noqa: BLE001 -- evidence capture must never fail the job
        print(f"[capture] FAILED (non-fatal): {e!r}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
