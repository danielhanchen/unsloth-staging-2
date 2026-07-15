"""PR #6647 tour compose -- side-by-side BEFORE (main) vs AFTER (PR) for every surface.

Pairs the matched scene screenshots from pr6647_tour_before.py (out/before) and
pr6647_tour_after.py (out/after) into per-scene boards, stacks them into one tall
storyboard, and builds a side-by-side video + GIF.

Run:
  python -m studio_test_kit.examples.pr6647_tour_compose \
      --before out/before --after out/after --out out/compose
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from ..compose import hstack_images, hstack_videos, vstack_images

# (shot-name, board-name, BEFORE label, AFTER label) -- shot name is shared by both sides.
SCENES = [
    ("s1_config_entry", "board_1_config_entry",
     "BEFORE (main): the gear stages the model into the sidebar",
     "AFTER (PR #6647): the gear opens an in-picker config page"),
    ("s2_context_set", "board_2_context_set",
     "BEFORE (main): set Context Length 8192 in the sidebar",
     "AFTER (PR #6647): set 8192 + Remember for this model"),
    ("s3_advanced", "board_3_advanced_settings",
     "BEFORE (main): inline KV cache / Speculative / Tensor Parallel",
     "AFTER (PR #6647): per-model Advanced settings, all persisted"),
    ("s4_template_editor", "board_4_chat_template_editor",
     "BEFORE (main): inline chat-template editor",
     "AFTER (PR #6647): standalone editor + validate + 64 KiB limit"),
    ("s2_after_reload", "board_5_persistence_after_reload",
     "BEFORE (main): reload -> reverts to native context",
     "AFTER (PR #6647): reload -> Context Length still 8192"),
    ("s5_sidebar", "board_6_run_settings_sidebar",
     "BEFORE (main): inline Run-settings sidebar",
     "AFTER (PR #6647): sidebar embeds the per-model config page"),
    ("s5_settings_chat", "board_7_load_on_selection",
     "BEFORE (main): Settings -> Chat has 'Load on selection'",
     "AFTER (PR #6647): 'Load on selection' removed"),
]


def run(before_dir, after_dir, out_dir):
    before_dir, after_dir, out_dir = Path(before_dir), Path(after_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def log(m):
        print(f"[tour-compose] {m}", flush=True)

    boards = []
    for shot, board, ll, lr in SCENES:
        b = before_dir / "shots" / f"{shot}.png"
        a = after_dir / "shots" / f"{shot}.png"
        if not b.exists() or not a.exists():
            log(f"skip {board}: missing {'before' if not b.exists() else 'after'} {shot}.png")
            continue
        out = out_dir / f"{board}.png"
        try:
            hstack_images(b, a, out, label_left=ll, label_right=lr)
            boards.append(out)
            log(f"wrote {out.name}")
        except Exception as e:
            log(f"board {board} failed: {e!r}")

    # One tall storyboard of every scene, top to bottom.
    if boards:
        try:
            storyboard = out_dir / "storyboard.png"
            vstack_images(boards, storyboard, gap_px=28)
            log(f"wrote {storyboard.name} ({len(boards)} scenes)")
        except Exception as e:
            log(f"storyboard failed: {e!r}")

    # Side-by-side video + GIF (best-effort).
    bw = before_dir / "video" / "before.webm"
    aw = after_dir / "video" / "after.webm"
    if shutil.which("ffmpeg") and bw.exists() and aw.exists():
        try:
            sbs = out_dir / "side_by_side.mp4"
            hstack_videos(bw, aw, sbs)
            log(f"wrote {sbs.name}")
            gif = out_dir / "side_by_side.gif"
            palette = out_dir / "_palette.png"
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(sbs),
                 "-vf", "fps=10,scale=1000:-1:flags=lanczos,palettegen", str(palette)],
                check=True, capture_output=True, text=True,
            )
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(sbs), "-i", str(palette),
                 "-lavfi", "fps=10,scale=1000:-1:flags=lanczos[x];[x][1:v]paletteuse",
                 "-loop", "0", str(gif)],
                check=True, capture_output=True, text=True,
            )
            log(f"wrote {gif.name}")
        except Exception as e:
            log(f"video/gif compose failed: {e!r}")
    else:
        log("ffmpeg or a webm missing; skipping side-by-side video/GIF")
    log(f"done ({len(boards)} boards)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", default="out/before")
    ap.add_argument("--after", default="out/after")
    ap.add_argument("--out", default="out/compose")
    a = ap.parse_args()
    sys.exit(run(a.before, a.after, a.out))


if __name__ == "__main__":
    main()
