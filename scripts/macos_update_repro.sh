#!/usr/bin/env bash
# Reproduce the macOS `unsloth studio update` regression:
#   - install older Studio release
#   - capture launcher / plist / icon fingerprints
#   - run `unsloth studio update` (or `install.sh --update` if available)
#   - recapture fingerprints
#   - assert each pair changed (proves update rebuilt the .app bundle)
#   - then launch via the launcher and watch for runaway /api/inference/load
#
# Designed for a GitHub Actions macos-14 runner; locally on macOS works too.
# Fails fast with rc != 0 when the bug is present.

set -uo pipefail

: "${UNSLOTH_STUDIO_HOME:=$HOME/.unsloth/studio}"
: "${UPDATE_FROM_SHA:=5c473fab}"        # v0.1.37-beta (2026.4.x)
: "${UPDATE_TO_SHA:=07e2fccf}"          # current pip tip
: "${IDLE_SECONDS:=60}"
: "${LOAD_THRESHOLD:=3}"

APP_DIR="$HOME/Applications/Unsloth Studio.app"
LAUNCHER_STUB="$APP_DIR/Contents/MacOS/launch-studio"
INFO_PLIST="$APP_DIR/Contents/Info.plist"
APP_ICON="$APP_DIR/Contents/Resources/AppIcon.icns"
LAUNCHER_SHARED="$UNSLOTH_STUDIO_HOME/share/launch-studio.sh"

WORK=$(mktemp -d -t unsloth_update_repro.XXXXXX)
trap 'rm -rf "$WORK"' EXIT
echo "[repro] WORK=$WORK"

# --- Phase 1: install older Studio release ---
git clone --depth 1 https://github.com/unslothai/unsloth "$WORK/pinned_old"
git -C "$WORK/pinned_old" fetch --depth 1 origin "$UPDATE_FROM_SHA"
git -C "$WORK/pinned_old" checkout "$UPDATE_FROM_SHA"
echo "[repro] installing older Studio @ $UPDATE_FROM_SHA"
(cd "$WORK/pinned_old" && UNSLOTH_STUDIO_HOME="$UNSLOTH_STUDIO_HOME" bash install.sh --local) || {
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
        echo "[$label] DATA_DIR in launcher stub:"
        grep -E "exec '.*launch-studio.sh'" "$LAUNCHER_STUB" 2>/dev/null
    }
}

BEFORE=$(fingerprint_target BEFORE)
echo "$BEFORE"

# --- Phase 3: run update from new tip ---
git clone --depth 1 https://github.com/unslothai/unsloth "$WORK/pinned_new"
git -C "$WORK/pinned_new" fetch --depth 1 origin "$UPDATE_TO_SHA"
git -C "$WORK/pinned_new" checkout "$UPDATE_TO_SHA"
echo "[repro] running 'unsloth studio update' (or install.sh --update fallback)"
if "$UNSLOTH_STUDIO_HOME/bin/unsloth" studio update --local; then
    echo "[repro] update exited 0"
else
    echo "[repro] 'unsloth studio update' failed; trying install.sh as fallback"
    (cd "$WORK/pinned_new" && UNSLOTH_STUDIO_HOME="$UNSLOTH_STUDIO_HOME" bash install.sh --local) || {
        echo "FAIL: fallback install also failed"
        exit 1
    }
fi

AFTER=$(fingerprint_target AFTER)
echo "$AFTER"

# --- Phase 4: assert each pair changed ---
diff <(echo "$BEFORE") <(echo "$AFTER") >/dev/null && {
    echo "FAIL: launcher / plist / icon fingerprints UNCHANGED after update."
    echo "Bug A reproduced: 'unsloth studio update' did NOT regenerate the .app bundle."
    exit 1
}
echo "[repro] update did mutate some fingerprints; per-target diff:"
diff <(echo "$BEFORE") <(echo "$AFTER") || true

# Stricter check: launcher stub itself must have changed (icon/plist may legitimately not).
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

LOG_FILE="$UNSLOTH_STUDIO_HOME/share/studio.log"
if [[ ! -f "$LOG_FILE" ]]; then
    echo "[repro] studio.log missing; cannot measure log spam"
    kill -- -"$LAUNCH_PID" 2>/dev/null || true
    exit 1
fi

# Re-truncate, sleep idle, count
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
