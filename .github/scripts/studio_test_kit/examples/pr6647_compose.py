"""PR #6647 (#6854) compose driver -- side-by-side BEFORE (main) vs AFTER (PR).

Builds matched comparison boards + a side-by-side video/GIF from the artifacts
produced by pr6647_before.py (out/before) and pr6647_after.py (out/after).

Run:
  python -m studio_test_kit.examples.pr6647_compose \
      --before out/before --after out/after --out out/compose
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from ..compose import hstack_images, hstack_videos, webm_to_mp4


def _pair(before_dir, after_dir, out_dir, bshot, ashot, name, ll, lr, log):
    b = before_dir / "shots" / bshot
    a = after_dir / "shots" / ashot
    if not b.exists() or not a.exists():
        log(f"skip {name}: missing {b if not b.exists() else a}")
        return None
    out = out_dir / f"{name}.png"
    hstack_images(b, a, out, label_left=ll, label_right=lr)
    log(f"wrote {out}")
    return out


def run(before_dir, after_dir, out_dir):
    before_dir, after_dir, out_dir = Path(before_dir), Path(after_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def log(m):
        print(f"[compose] {m}", flush=True)

    # Matched scene boards.
    _pair(before_dir, after_dir, out_dir,
          "02_ctx_set.png", "01_config_set.png", "board_1_context_set",
          "BEFORE (main): set Context Length 8192", "AFTER (PR #6647): set 8192 + Remember", log)
    _pair(before_dir, after_dir, out_dir,
          "03_after_reload.png", "04_after_page_reload.png", "board_2_after_reload",
          "BEFORE (main): reload -> reverts to 32768", "AFTER (PR #6647): reload -> still 8192", log)

    # Side-by-side video + GIF (best-effort; needs ffmpeg + both webms).
    bw = before_dir / "video" / "before.webm"
    aw = after_dir / "video" / "after.webm"
    if shutil.which("ffmpeg") and bw.exists() and aw.exists():
        try:
            sbs = out_dir / "side_by_side.mp4"
            hstack_videos(bw, aw, sbs)
            log(f"wrote {sbs}")
            gif = out_dir / "side_by_side.gif"
            palette = out_dir / "_palette.png"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(sbs),
                            "-vf", "fps=10,scale=960:-1:flags=lanczos,palettegen",
                            str(palette)], check=True, capture_output=True, text=True)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(sbs), "-i", str(palette),
                            "-lavfi", "fps=10,scale=960:-1:flags=lanczos[x];[x][1:v]paletteuse",
                            "-loop", "0", str(gif)], check=True, capture_output=True, text=True)
            log(f"wrote {gif}")
        except Exception as e:
            log(f"video/gif compose failed: {e!r}")
    else:
        log("ffmpeg or a webm missing; skipping side-by-side video/GIF")
    log("done")
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
