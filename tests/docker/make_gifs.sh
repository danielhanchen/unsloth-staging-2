#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0
#
# Convert captured WebM recordings to high-quality, inline-renderable GIFs.
# GitHub PR comments animate GIF/PNG inline but NOT mp4/webm, so every video we
# want to show in a comment must become a GIF. Two-pass palettegen/paletteuse
# keeps the file small with good color fidelity.
#
# Usage:
#   bash make_gifs.sh <input.webm> <output.gif> [fps] [width] [max_seconds]
# Defaults: fps=12, width=960, max_seconds=0 (whole clip; >0 trims from start).
set -euo pipefail

SRC="${1:?usage: make_gifs.sh <in.webm> <out.gif> [fps] [width] [max_seconds]}"
OUT="${2:?missing output .gif path}"
FPS="${3:-12}"
WIDTH="${4:-960}"
MAXSEC="${5:-0}"

command -v ffmpeg >/dev/null 2>&1 || { echo "ERROR: ffmpeg not found" >&2; exit 1; }
[ -s "$SRC" ] || { echo "ERROR: source missing/empty: $SRC" >&2; exit 1; }
mkdir -p "$(dirname "$OUT")"

TRIM=()
if [ "$MAXSEC" != "0" ]; then TRIM=(-t "$MAXSEC"); fi

PALETTE="$(mktemp --suffix=.png)"
VF="fps=${FPS},scale=${WIDTH}:-1:flags=lanczos"

ffmpeg -y -loglevel error "${TRIM[@]}" -i "$SRC" \
    -vf "${VF},palettegen=stats_mode=diff" "$PALETTE"
ffmpeg -y -loglevel error "${TRIM[@]}" -i "$SRC" -i "$PALETTE" \
    -lavfi "${VF} [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=3" "$OUT"
rm -f "$PALETTE"

ls -lh "$OUT"
echo "GIF written: $OUT"
