"""Compose the PR #6647 picker <-> sidebar sync confirmation into boards + GIF.

Takes the four step screenshots from pr6647_sync_confirm.py and builds:
  board_a_picker_to_sidebar.png   (01 picker edit | 02 sidebar reflects)
  board_b_sidebar_to_picker.png   (03 sidebar edit | 04 picker reflects)
  storyboard.png                  (board A over board B)
  sync.gif                        (best-effort, from the session webm)

Run:
  python -m studio_test_kit.examples.pr6647_sync_compose \
      --shots out/sync/shots --video out/sync/video --out out/sync/compose
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from ..compose import hstack_images, vstack_images

PAIRS = [
    ("01_picker_edit", "02_sidebar_reflects_picker", "board_a_picker_to_sidebar",
     "Picker: set Context Length 8,192 (KV q8_0)",
     "Sidebar reflects it: 8,192 (KV q8_0)"),
    ("03_sidebar_edit", "04_picker_reflects_sidebar", "board_b_sidebar_to_picker",
     "Sidebar: set Context Length 4,096 (KV q5_1)",
     "Picker reflects it: 4,096 (KV q5_1)"),
]


def run(shots_dir, video_dir, out_dir):
    shots_dir, out_dir = Path(shots_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def log(m):
        print(f"[sync-compose] {m}", flush=True)

    boards = []
    for left, right, board, ll, lr in PAIRS:
        lp, rp = shots_dir / f"{left}.png", shots_dir / f"{right}.png"
        if not lp.exists() or not rp.exists():
            log(f"skip {board}: missing {'left' if not lp.exists() else 'right'}")
            continue
        out = out_dir / f"{board}.png"
        try:
            hstack_images(lp, rp, out, label_left=ll, label_right=lr)
            boards.append(out)
            log(f"wrote {out.name}")
        except Exception as e:
            log(f"board {board} failed: {e!r}")

    if boards:
        try:
            storyboard = out_dir / "storyboard.png"
            vstack_images(boards, storyboard, gap_px=28)
            log(f"wrote {storyboard.name} ({len(boards)} boards)")
        except Exception as e:
            log(f"storyboard failed: {e!r}")

    webm = Path(video_dir) / "sync.webm" if video_dir else None
    if webm and webm.exists() and shutil.which("ffmpeg"):
        try:
            gif = out_dir / "sync.gif"
            palette = out_dir / "_palette.png"
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(webm),
                 "-vf", "fps=10,scale=1100:-1:flags=lanczos,palettegen", str(palette)],
                check=True, capture_output=True, text=True,
            )
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(webm), "-i", str(palette),
                 "-lavfi", "fps=10,scale=1100:-1:flags=lanczos[x];[x][1:v]paletteuse",
                 "-loop", "0", str(gif)],
                check=True, capture_output=True, text=True,
            )
            log(f"wrote {gif.name}")
        except Exception as e:
            log(f"gif compose failed: {e!r}")
    else:
        log("no webm or ffmpeg; skipping GIF")
    log(f"done ({len(boards)} boards)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", required=True)
    ap.add_argument("--video", default=None)
    ap.add_argument("--out", default="out/sync/compose")
    a = ap.parse_args()
    sys.exit(run(a.shots, a.video, a.out))


if __name__ == "__main__":
    main()
