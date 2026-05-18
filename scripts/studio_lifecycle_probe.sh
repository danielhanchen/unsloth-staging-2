#!/usr/bin/env bash
# Open/close/reopen lifecycle probe for Unsloth Studio.
#
# Launches Studio via the env-mode launcher (so studio.log gets populated),
# drives the UI with test_unsloth_studio.py through a single agent prompt that
# sequences Open -> Nav -> Close -> Reopen -> Nav -> Close, then asserts that
# /api/inference/load is not spamming the log during an idle window.
#
# Required env:
#   UNSLOTH_STUDIO_HOME   absolute path to the Studio install root (env-mode).
# Optional:
#   PROBE_IDLE_SECONDS    seconds to watch the log AFTER probe returns (default 60)
#   PROBE_LOAD_THRESHOLD  max allowed /api/inference/load POSTs in idle window (default 3)
#   PROBE_PROMPT          override default open/close/reopen prompt

set -uo pipefail

: "${UNSLOTH_STUDIO_HOME:?UNSLOTH_STUDIO_HOME is required}"
IDLE_SECONDS="${PROBE_IDLE_SECONDS:-60}"
LOAD_THRESHOLD="${PROBE_LOAD_THRESHOLD:-3}"

LAUNCHER="$UNSLOTH_STUDIO_HOME/share/launch-studio.sh"
LOG_FILE="$UNSLOTH_STUDIO_HOME/share/studio.log"
PORT_FILE="$UNSLOTH_STUDIO_HOME/share/studio.port"

if [[ ! -x "$LAUNCHER" ]]; then
    echo "FAIL: launcher not found or not executable: $LAUNCHER" >&2
    exit 2
fi

# Truncate (or create) the log so we only measure THIS run.
: > "$LOG_FILE"

echo "[probe] launching $LAUNCHER"
# Launcher daemonizes; it writes studio.port and tails to studio.log itself.
"$LAUNCHER" >/dev/null 2>&1 &
LAUNCH_PID=$!

cleanup() {
    if [[ -f "$UNSLOTH_STUDIO_HOME/share/studio.pid" ]]; then
        kill "$(cat "$UNSLOTH_STUDIO_HOME/share/studio.pid")" 2>/dev/null || true
    fi
    kill -- -"$LAUNCH_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Wait up to 120s for studio.port to appear and /api/health to answer.
PORT=""
for _ in $(seq 1 120); do
    if [[ -s "$PORT_FILE" ]]; then
        PORT="$(cat "$PORT_FILE")"
        if curl -fsS "http://127.0.0.1:$PORT/api/health" >/dev/null 2>&1; then
            break
        fi
    fi
    sleep 1
done

if [[ -z "$PORT" ]]; then
    echo "FAIL: Studio did not write studio.port within 120s" >&2
    tail -50 "$LOG_FILE" >&2 || true
    exit 1
fi
echo "[probe] Studio healthy on port $PORT"

DEFAULT_PROMPT='Open Studio in the browser. Wait for the chat composer to render. Click each item in the left navigation once -- Chat, Train, Recipes, Models, Settings -- screenshot after each. Close the browser tab. Wait 5 seconds. Reopen Studio on the same URL. Confirm the page renders. Click Train, screenshot. Close the tab. Report PASS only if every navigation click succeeded and the final reopen rendered. Otherwise return FAIL with the failing step name and any console error.'
PROMPT="${PROBE_PROMPT:-$DEFAULT_PROMPT}"

cd /mnt/disks/unslothai/ubuntu/workspace_51

echo "[probe] running test_unsloth_studio.py against port $PORT"
python test_unsloth_studio.py --port "$PORT" --prompt "$PROMPT"
PROBE_RC=$?
echo "[probe] probe exit=$PROBE_RC"

echo "[probe] idle watch for ${IDLE_SECONDS}s..."
sleep "$IDLE_SECONDS"

LOAD_COUNT=$(grep -c '"path": "/api/inference/load"' "$LOG_FILE" 2>/dev/null || echo 0)
echo "[probe] /api/inference/load POSTs in log: $LOAD_COUNT (threshold=$LOAD_THRESHOLD)"

if [[ "$LOAD_COUNT" -gt "$LOAD_THRESHOLD" ]]; then
    echo "FAIL: runaway /api/inference/load (saw $LOAD_COUNT, threshold $LOAD_THRESHOLD)"
    grep '"/api/inference/load"' "$LOG_FILE" | tail -20 >&2 || true
    exit 1
fi

if [[ "$PROBE_RC" -ne 0 ]]; then
    echo "FAIL: probe returned rc=$PROBE_RC"
    exit "$PROBE_RC"
fi

echo "PASS"
exit 0
