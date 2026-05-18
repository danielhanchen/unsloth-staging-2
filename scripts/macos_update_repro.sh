#!/usr/bin/env bash
# Reproduce the macOS `unsloth studio update` regression:
#   1. install an older Studio release in default mode (creates .app bundle)
#   2. fingerprint launcher / plist / icon
#   3. run `unsloth studio update`
#   4. recapture fingerprints
#   5. assert each pair changed (proves update rebuilt the .app bundle)
#   6. launch via the launcher and watch for runaway /api/inference/load
#
# Designed for a GitHub Actions macos-14 runner; locally on macOS works too.
# Fails fast with rc != 0 when the bug is present.

set -uo pipefail

# DO NOT export UNSLOTH_STUDIO_HOME; install.sh treats any value (even the
# legacy default) as an env-override and skips the .app bundle.
STUDIO_HOME_DEFAULT="$HOME/.unsloth/studio"
DATA_DIR_DEFAULT="$HOME/.local/share/unsloth"
: "${UPDATE_FROM_SHA:=5c473fab}"        # v0.1.37-beta (2026.4.x)
: "${UPDATE_TO_SHA:=07e2fccf}"          # current pip tip
: "${IDLE_SECONDS:=60}"
: "${LOAD_THRESHOLD:=3}"

APP_DIR="$HOME/Applications/Unsloth Studio.app"
LAUNCHER_STUB="$APP_DIR/Contents/MacOS/launch-studio"
INFO_PLIST="$APP_DIR/Contents/Info.plist"
APP_ICON="$APP_DIR/Contents/Resources/AppIcon.icns"
LAUNCHER_SHARED="$DATA_DIR_DEFAULT/launch-studio.sh"
UNSLOTH_BIN="$STUDIO_HOME_DEFAULT/bin/unsloth"

WORK=$(mktemp -d -t unsloth_update_repro.XXXXXX)
trap 'rm -rf "$WORK"' EXIT
echo "[repro] WORK=$WORK"

# --- Phase 1: install older Studio release ---
git clone https://github.com/unslothai/unsloth "$WORK/pinned_old"
(cd "$WORK/pinned_old" && git checkout "$UPDATE_FROM_SHA")
echo "[repro] installing older Studio @ $UPDATE_FROM_SHA (default mode)"
(cd "$WORK/pinned_old" && bash install.sh --local) || {
    echo "FAIL: older install failed -- cannot establish baseline"
    exit 1
}

[[ -d "$APP_DIR" ]] || { echo "FAIL: .app bundle missing after older install: $APP_DIR"; exit 1; }

# --- Phase 2: fingerprint launcher / plist / icon ---
fingerprint_target() {
    local label="$1"
    {
        echo "[$label] launcher stub sha256:"
        shasum -a 256 "$LAUNCHER_STUB" 2>/dev/null
        echo "[$label] launcher shared sha256:"
        shasum -a 256 "$LAUNCHER_SHARED" 2>/dev/null
        echo "[$label] Info.plist sha256:"
        shasum -a 256 "$INFO_PLIST" 2>/dev/null
        echo "[$label] AppIcon.icns sha256:"
        shasum -a 256 "$APP_ICON" 2>/dev/null
        echo "[$label] DATA_DIR line in launcher stub:"
        grep -E "exec '.*launch-studio.sh'" "$LAUNCHER_STUB" 2>/dev/null
    }
}

BEFORE=$(fingerprint_target BEFORE)
echo "$BEFORE"

# --- Phase 3: run update from new tip ---
git clone https://github.com/unslothai/unsloth "$WORK/pinned_new"
(cd "$WORK/pinned_new" && git checkout "$UPDATE_TO_SHA")
echo "[repro] running 'unsloth studio update' (or install.sh fallback)"
if "$UNSLOTH_BIN" studio update --local; then
    echo "[repro] update exited 0"
else
    echo "[repro] 'unsloth studio update' failed; trying install.sh as fallback"
    (cd "$WORK/pinned_new" && bash install.sh --local) || {
        echo "FAIL: fallback install also failed"
        exit 1
    }
fi

AFTER=$(fingerprint_target AFTER)
echo "$AFTER"

# --- Phase 4: assert each pair changed ---
echo "[repro] per-target diff (BEFORE vs AFTER):"
diff <(echo "$BEFORE") <(echo "$AFTER") || true

# Stricter check: launcher stub itself must have changed.
BEFORE_LAUNCHER=$(echo "$BEFORE" | grep -A1 "launcher stub sha256" | tail -1)
AFTER_LAUNCHER=$(echo "$AFTER"  | grep -A1 "launcher stub sha256" | tail -1)
if [[ "$BEFORE_LAUNCHER" == "$AFTER_LAUNCHER" ]]; then
    echo "FAIL: launcher stub sha256 unchanged after update. Bug A reproduced."
    exit 1
fi

# --- Phase 5: launch + watch for /api/inference/load spam ---
echo "[repro] launching via $LAUNCHER_SHARED"
"$LAUNCHER_SHARED" >/dev/null 2>&1 &
LAUNCH_PID=$!
sleep 20

LOG_FILE="$DATA_DIR_DEFAULT/studio.log"
if [[ ! -f "$LOG_FILE" ]]; then
    echo "[repro] studio.log missing; cannot measure log spam"
    kill -- -"$LAUNCH_PID" 2>/dev/null || true
    exit 1
fi

# Truncate, idle, count.
: > "$LOG_FILE"
sleep "$IDLE_SECONDS"
LOAD_COUNT=$(grep -c '"path": "/api/inference/load"' "$LOG_FILE" 2>/dev/null || echo 0)
echo "[repro] idle /api/inference/load POSTs in ${IDLE_SECONDS}s: $LOAD_COUNT (threshold=$LOAD_THRESHOLD)"

kill -- -"$LAUNCH_PID" 2>/dev/null || true

if [[ "$LOAD_COUNT" -gt "$LOAD_THRESHOLD" ]]; then
    echo "FAIL: Bug B reproduced -- /api/inference/load spam after idle"
    grep '"/api/inference/load"' "$LOG_FILE" | tail -10 || true
    exit 1
fi

echo "PASS: update rebuilt launcher and no /load spam"
