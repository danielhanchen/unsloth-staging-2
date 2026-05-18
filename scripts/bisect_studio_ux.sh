#!/usr/bin/env bash
# git bisect runner for the open/close/reopen Studio UX regression.
#
# Exit codes (git bisect convention):
#   0    good     -- lifecycle probe passed
#   1    bad      -- lifecycle probe failed (regression present at this commit)
#   125  skip     -- install or environment failure unrelated to the regression
#
# Required env:
#   UNSLOTH_STUDIO_HOME   path that this runner OWNS and may wipe each iteration.
#
# Run from the root of an unsloth checkout:
#   git bisect start
#   git bisect bad  07e2fccf
#   git bisect good 5fa8683b
#   git bisect run /mnt/disks/unslothai/ubuntu/workspace_51/scripts/bisect_studio_ux.sh

set -uo pipefail

: "${UNSLOTH_STUDIO_HOME:?UNSLOTH_STUDIO_HOME is required and must be a path this script may wipe}"

SHA=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
LOG_DIR=/mnt/disks/unslothai/ubuntu/workspace_51/logs/bisect
mkdir -p "$LOG_DIR"
ITER_LOG="$LOG_DIR/bisect_${SHA}_$(date +%Y%m%d_%H%M%S).log"

echo "[bisect] commit=$SHA  studio_home=$UNSLOTH_STUDIO_HOME  log=$ITER_LOG" | tee -a "$ITER_LOG"

# Reset Studio home this iteration owns -- nothing else.
rm -rf "$UNSLOTH_STUDIO_HOME"
mkdir -p "$UNSLOTH_STUDIO_HOME"

# install.sh --local must be run from the repo root (the bisected checkout).
if ! UNSLOTH_STUDIO_HOME="$UNSLOTH_STUDIO_HOME" bash install.sh --local >>"$ITER_LOG" 2>&1; then
    echo "[bisect] install failed -> skip" | tee -a "$ITER_LOG"
    exit 125
fi

# Run the lifecycle probe. Skip (not bad) if Studio binary is missing entirely
# (a build artifact issue independent of the bisected UX behavior).
if [[ ! -x "$UNSLOTH_STUDIO_HOME/share/launch-studio.sh" ]]; then
    echo "[bisect] launch-studio.sh missing -> skip" | tee -a "$ITER_LOG"
    exit 125
fi

if UNSLOTH_STUDIO_HOME="$UNSLOTH_STUDIO_HOME" \
   /mnt/disks/unslothai/ubuntu/workspace_51/scripts/studio_lifecycle_probe.sh \
   >>"$ITER_LOG" 2>&1; then
    echo "[bisect] $SHA = good" | tee -a "$ITER_LOG"
    exit 0
fi

echo "[bisect] $SHA = bad" | tee -a "$ITER_LOG"
exit 1
