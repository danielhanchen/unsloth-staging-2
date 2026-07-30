# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Assert an X root window actually painted something.

The desktop release notes warn that "Linux AppImage can show a blank window on some
Tauri/WebKitGTK + Wayland/Mesa stacks". A blank window is invisible to every check we
otherwise run: the process is alive, the backend is healthy, the logs are clean, and the
user sees grey. `xwd -root` plus a look at the pixels is the cheapest way to catch it,
and needs no toolchain on a stripped machine -- unlike tauri-driver, which needs cargo
and a C compiler, both of which the strip removes.

XWD format: a big-endian header whose first word is the header length, then the window
name, then raw pixels. Everything needed is in the fixed header, so no image library.

    xwd -root -display :99 -out screen.xwd
    python tests/studio/xwd_not_blank.py screen.xwd --min-colors 24
"""

from __future__ import annotations

import argparse
import collections
import struct
import sys
import zlib
from pathlib import Path


def read_xwd(path: Path) -> tuple[int, int, int, int, bytes]:
    """Return (width, height, bits_per_pixel, bytes_per_line, pixel_bytes)."""
    blob = path.read_bytes()
    if len(blob) < 100:
        raise ValueError(f"{path} is only {len(blob)} bytes; xwd wrote nothing usable")
    # XWDFileHeader is 25 big-endian 32-bit words. Fields we need, by index:
    #   0 header_size  4 pixmap_width  5 pixmap_height  7 bits_per_pixel  12 bytes_per_line
    words = struct.unpack(">25I", blob[:100])
    header_size = words[0]
    width, height, bits_per_pixel, bytes_per_line = words[4], words[5], words[7], words[12]
    if not (0 < width < 20000 and 0 < height < 20000):
        raise ValueError(f"implausible XWD geometry {width}x{height}; not an XWD file?")
    pixels = blob[header_size:]
    if not bytes_per_line:
        bytes_per_line = len(pixels) // height
    return width, height, bits_per_pixel, bytes_per_line, pixels


def write_png(path: Path, width: int, height: int, bytes_per_line: int, pixels: bytes) -> None:
    """Emit a viewable RGB PNG, so a run leaves behind a real screenshot of the app and
    not just a verdict about it. Hand-rolled because the machine is stripped and Pillow
    is exactly the kind of thing we are asserting nobody has to install."""
    bytes_per_pixel = max(1, bytes_per_line // width)

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    rows = []
    for y in range(height):
        line = pixels[y * bytes_per_line : (y + 1) * bytes_per_line]
        row = bytearray(b"\x00")  # PNG filter type 0 per scanline
        for x in range(width):
            px = line[x * bytes_per_pixel : (x + 1) * bytes_per_pixel]
            if len(px) < 3:
                px = px + b"\x00" * (3 - len(px))
            row += bytes((px[2], px[1], px[0]))  # X stores BGRx
        rows.append(bytes(row))

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"".join(rows), 6))
        + chunk(b"IEND", b"")
    )


def distinct_colors(pixels: bytes, bytes_per_pixel: int, sample_stride: int = 97) -> int:
    """Count distinct pixel values on a strided sample.

    A stride that is coprime with the row width walks across rows rather than down a
    single column, so a window that is blank except for one painted stripe is not
    mistaken for fully painted (or vice versa).
    """
    seen = collections.Counter()
    step = bytes_per_pixel * sample_stride
    for offset in range(0, len(pixels) - bytes_per_pixel, step):
        seen[pixels[offset : offset + bytes_per_pixel]] += 1
    return len(seen)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--png", type=Path, help="also write a viewable PNG here")
    parser.add_argument(
        "--min-colors",
        type=int,
        default=24,
        help="below this the window is treated as blank (a solid or 2-tone screen)",
    )
    args = parser.parse_args()

    try:
        width, height, bits_per_pixel, bytes_per_line, pixels = read_xwd(args.path)
    except Exception as error:
        print(f"::error::could not read {args.path}: {error}")
        return 1

    if args.png:
        try:
            write_png(args.png, width, height, bytes_per_line, pixels)
            print(f"screenshot written to {args.png}")
        except Exception as error:
            print(f"::warning::could not write {args.png}: {error}")

    # The header's bits_per_pixel is not always populated (xwd on some servers writes 0),
    # so derive the stride from bytes_per_line, which always is.
    bytes_per_pixel = max(1, bytes_per_line // width)
    colors = distinct_colors(pixels, bytes_per_pixel)
    print(f"screen: {width}x{height} @ {bits_per_pixel}bpp, {colors} distinct colours sampled")

    if colors < args.min_colors:
        print(
            f"::error::the window looks blank -- only {colors} distinct colours "
            f"(threshold {args.min_colors}). This is the WebKitGTK blank-window failure "
            f"the release notes warn about; the process being alive does not catch it."
        )
        return 1
    print(f"the window painted real content ({colors} colours >= {args.min_colors})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
