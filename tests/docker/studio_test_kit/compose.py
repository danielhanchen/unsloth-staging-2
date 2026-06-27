"""Image + video composition helpers for pre/post-PR comparisons.

The PR5720 review pipeline used two flavors:

  - PIL hstack of paired screenshots, with a header strip labelling each
    column ("BEFORE" / "AFTER"). Useful for static side-by-side proof.
  - ffmpeg hstack of two same-length .webm sessions, into a single .mp4
    you can drop into a PR description.

The PIL helper auto-resizes pairs to the same height so the result lines
up even when the two screenshots were captured at slightly different
viewport widths.
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Iterable, Optional

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover
    Image = None  # type: ignore


def _font(size: int = 28) -> Optional["ImageFont.FreeTypeFont"]:
    if Image is None:
        return None
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        try:
            return ImageFont.truetype(candidate, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def hstack_images(
    left: Path,
    right: Path,
    out: Path,
    label_left: str = "BEFORE",
    label_right: str = "AFTER",
    gap_px: int = 24,
    label_height_px: int = 56,
) -> Path:
    """Stack two screenshots horizontally with column labels."""
    if Image is None:
        raise RuntimeError("Pillow is required for hstack_images")
    li, ri = Image.open(left).convert("RGB"), Image.open(right).convert("RGB")
    target_h = max(li.height, ri.height)
    if li.height != target_h:
        li = li.resize((int(li.width * target_h / li.height), target_h))
    if ri.height != target_h:
        ri = ri.resize((int(ri.width * target_h / ri.height), target_h))

    total_w = li.width + ri.width + gap_px
    total_h = target_h + label_height_px
    canvas = Image.new("RGB", (total_w, total_h), "white")
    canvas.paste(li, (0, label_height_px))
    canvas.paste(ri, (li.width + gap_px, label_height_px))

    draw = ImageDraw.Draw(canvas)
    f = _font(28)
    if f is not None:
        draw.text((16, 14), label_left, fill="black", font=f)
        draw.text((li.width + gap_px + 16, 14), label_right, fill="black", font=f)

    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return out


def vstack_images(images: Iterable[Path], out: Path, gap_px: int = 16) -> Path:
    """Stack a list of screenshots vertically (e.g. a per-turn timeline)."""
    if Image is None:
        raise RuntimeError("Pillow is required for vstack_images")
    imgs = [Image.open(p).convert("RGB") for p in images]
    if not imgs:
        raise ValueError("vstack_images: empty iterable")
    target_w = max(i.width for i in imgs)
    resized = [
        i.resize((target_w, int(i.height * target_w / i.width))) if i.width != target_w else i
        for i in imgs
    ]
    total_h = sum(i.height for i in resized) + gap_px * (len(resized) - 1)
    canvas = Image.new("RGB", (target_w, total_h), "white")
    y = 0
    for i in resized:
        canvas.paste(i, (0, y))
        y += i.height + gap_px
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return out


def hstack_videos(left: Path, right: Path, out_mp4: Path) -> Path:
    """ffmpeg hstack two .webm sessions into one .mp4."""
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(left),
        "-i", str(right),
        "-filter_complex", "[0:v][1:v]hstack=inputs=2[v]",
        "-map", "[v]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return out_mp4


def webm_to_mp4(src_webm: Path, out_mp4: Path) -> Path:
    """Re-encode a single .webm to .mp4 (useful for PR-body upload)."""
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src_webm),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(out_mp4)],
        check=True, capture_output=True, text=True,
    )
    return out_mp4
