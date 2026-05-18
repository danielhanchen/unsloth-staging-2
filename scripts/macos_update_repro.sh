#!/usr/bin/env bash
# Reproduce the macOS `unsloth studio update` regression in a self-contained
# way: install Studio at the current checkout, overwrite the .app launcher
# stub with a sentinel, run `unsloth studio update`, then assert the sentinel
# was overwritten. Today this fails because update never touches the .app
# bundle. Pass when create_studio_shortcuts is wired into the update path.
#
# Designed for a GitHub Actions macos-14 runner; works locally on macOS too.

set -uo pipefail

# DO NOT export UNSLOTH_STUDIO_HOME: install.sh treats any value (even the
# legacy default) as env-override and skips the .app bundle.
STUDIO_HOME_DEFAULT="$HOME/.unsloth/studio"
DATA_DIR_DEFAULT="$HOME/.local/share/unsloth"

APP_DIR="$HOME/Applications/Unsloth Studio.app"
LAUNCHER_STUB="$APP_DIR/Contents/MacOS/launch-studio"
LAUNCHER_SHARED="$DATA_DIR_DEFAULT/launch-studio.sh"
# Default mode: shim under ~/.local/bin; env mode: $STUDIO_HOME/bin.
UNSLOTH_BIN="$HOME/.local/bin/unsloth"
if [[ ! -x "$UNSLOTH_BIN" && -x "$STUDIO_HOME_DEFAULT/bin/unsloth" ]]; then
    UNSLOTH_BIN="$STUDIO_HOME_DEFAULT/bin/unsloth"
fi

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
[[ -f "$REPO_ROOT/install.sh" ]] || {
    echo "FAIL: install.sh not found at $REPO_ROOT/install.sh"
    exit 1
}

echo "[repro] installing Studio @ $REPO_ROOT (default mode)"
(cd "$REPO_ROOT" && bash install.sh --local) || {
    echo "FAIL: install.sh failed"
    exit 1
}

[[ -d "$APP_DIR" ]] || { echo "FAIL: .app bundle missing: $APP_DIR"; exit 1; }
[[ -f "$LAUNCHER_STUB" ]] || { echo "FAIL: launcher stub missing: $LAUNCHER_STUB"; exit 1; }
[[ -x "$UNSLOTH_BIN" ]] || { echo "FAIL: unsloth binary missing: $UNSLOTH_BIN"; exit 1; }

SENTINEL="@@MACOS_UPDATE_REGRESSION_SENTINEL@@"
echo "[repro] writing sentinel into launcher stub"
printf '#!/bin/sh\necho "%s"; exit 17\n' "$SENTINEL" > "$LAUNCHER_STUB"
chmod +x "$LAUNCHER_STUB"

BEFORE_SHA=$(shasum -a 256 "$LAUNCHER_STUB" | awk '{print $1}')
echo "[repro] sentinel sha256 = $BEFORE_SHA"

echo "[repro] running 'unsloth studio update --local'"
"$UNSLOTH_BIN" studio update --local || {
    echo "FAIL: unsloth studio update --local exited non-zero"
    exit 1
}

if [[ ! -f "$LAUNCHER_STUB" ]]; then
    echo "FAIL: launcher stub gone after update"
    exit 1
fi
AFTER_SHA=$(shasum -a 256 "$LAUNCHER_STUB" | awk '{print $1}')
echo "[repro] after update sha256 = $AFTER_SHA"

if [[ "$BEFORE_SHA" == "$AFTER_SHA" ]]; then
    echo "FAIL: Bug A reproduced -- 'unsloth studio update' did not rewrite the .app launcher stub."
    echo "Sentinel still present:"
    grep -F "$SENTINEL" "$LAUNCHER_STUB" || true
    exit 1
fi

if grep -F "$SENTINEL" "$LAUNCHER_STUB" >/dev/null 2>&1; then
    echo "FAIL: sentinel string still present in launcher stub after update"
    exit 1
fi

echo "PASS: 'unsloth studio update' rewrote the launcher stub."
