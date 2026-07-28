#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
#
# After an install has been interrupted, decide whether the desktop app WOULD have
# reported this venv as healthy -- reproducing the Tauri preflight probes in shell so
# the regression is testable without building the app.
#
# The reported bug: an install SIGTERM'd during the "studio deps" step leaves a venv
# where the CLI's own deps (typer/click/rich) are present but the server stack is not
# (structlog lives in studio/backend/requirements/studio.txt). Preflight probes
# `unsloth -h` (preflight/managed.rs:419) and `unsloth studio desktop-capabilities
# --json` (managed.rs:318); both SUCCEED, so the app reports ManagedReady with
# can_auto_repair=false and then the backend dies on `import structlog`.
#
# Verdicts:
#   HEALTHY      backend imports and serves -- interruption did no lasting harm
#   REPAIRABLE   backend is broken AND a probe says so (verify-install /
#                desktop-runtime-check / studio_install_ok) -> the app can self-heal
#   FALSE_READY  backend is broken but every probe reports ready -> THE BUG
#
# Exit codes: 0 HEALTHY, 0 REPAIRABLE, 1 FALSE_READY, 2 usage/setup error.
set -uo pipefail

BIN="${1:-}"
PORT="${2:-8899}"
[ -n "$BIN" ] || { echo "usage: $0 <path-to-unsloth-bin> [port]" >&2; exit 2; }
[ -x "$BIN" ] || { echo "::error::unsloth bin not executable: $BIN"; exit 2; }

OUT="${PROBE_OUT_DIR:-probe}"
mkdir -p "$OUT"

say() { echo "[probe] $*"; }

# ── 1. the two probes Tauri preflight actually runs ───────────────────────────
cli_h_ok=false
"$BIN" -h > "$OUT/cli-h.log" 2>&1 && cli_h_ok=true
say "unsloth -h                        ok=$cli_h_ok"

caps_ok=false
"$BIN" studio desktop-capabilities --json > "$OUT/desktop-capabilities.json" 2>&1 && caps_ok=true
say "studio desktop-capabilities --json ok=$caps_ok"

# studio_install_ok is added by the install-manifest work; absent on older trees.
install_ok_field="$(python3 - "$OUT/desktop-capabilities.json" <<'PY' 2>/dev/null || echo absent
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    print("absent"); raise SystemExit
v = d.get("studio_install_ok")
print("absent" if v is None else ("true" if v else "false"))
PY
)"
say "capabilities.studio_install_ok    = $install_ok_field"

# ── 2. the newer, deeper probes (present only with the fix PRs applied) ───────
verify_state=absent
if "$BIN" studio verify-install --help >/dev/null 2>&1; then
  if "$BIN" studio verify-install > "$OUT/verify-install.log" 2>&1; then
    verify_state=ok
  else
    verify_state=failed
  fi
fi
say "studio verify-install             = $verify_state"

runtime_state=absent
if "$BIN" studio desktop-runtime-check --help >/dev/null 2>&1; then
  if "$BIN" studio desktop-runtime-check > "$OUT/desktop-runtime-check.log" 2>&1; then
    runtime_state=ok
  else
    runtime_state=failed
  fi
fi
say "studio desktop-runtime-check      = $runtime_state"

# The in-progress marker #7490 writes before spawning the installer.
marker_present=false
for m in "${UNSLOTH_STUDIO_HOME:-$HOME/.unsloth}/.desktop-install-in-progress"; do
  [ -e "$m" ] && marker_present=true
done
say ".desktop-install-in-progress      = $marker_present"

# ── 3. does the backend actually boot? ────────────────────────────────────────
# This is ground truth: the app spawns exactly this.
backend_ok=false
set -m
"$BIN" studio --api-only -H 127.0.0.1 -p "$PORT" > "$OUT/backend.log" 2>&1 &
BPID=$!
set +m
for _ in $(seq 1 60); do
  if ! kill -0 "$BPID" 2>/dev/null; then break; fi
  if curl -fsS "http://127.0.0.1:$PORT/api/health" -o "$OUT/health.json" 2>/dev/null \
     || curl -fsS "http://127.0.0.1:$PORT/healthz" -o "$OUT/health.json" 2>/dev/null; then
    backend_ok=true; break
  fi
  sleep 2
done
kill -TERM -- -"$BPID" 2>/dev/null || kill -TERM "$BPID" 2>/dev/null || true
wait "$BPID" 2>/dev/null || true
say "backend serves /api/health        ok=$backend_ok"

missing_module="$(grep -oE "ModuleNotFoundError: No module named '[^']+'" "$OUT/backend.log" 2>/dev/null | head -1 || true)"
[ -n "$missing_module" ] && say "backend import error: $missing_module"

# ── 4. verdict ───────────────────────────────────────────────────────────────
{
  echo "cli_h_ok=$cli_h_ok"
  echo "caps_ok=$caps_ok"
  echo "studio_install_ok=$install_ok_field"
  echo "verify_install=$verify_state"
  echo "desktop_runtime_check=$runtime_state"
  echo "install_in_progress_marker=$marker_present"
  echo "backend_ok=$backend_ok"
  echo "missing_module=${missing_module:-none}"
} > "$OUT/verdict.env"

if [ "$backend_ok" = "true" ]; then
  echo "VERDICT=HEALTHY" | tee -a "$OUT/verdict.env"
  exit 0
fi

# Backend is broken. Something must say so, or the app shows ManagedReady forever.
if [ "$verify_state" = "failed" ] || [ "$runtime_state" = "failed" ] \
   || [ "$install_ok_field" = "false" ] || [ "$marker_present" = "true" ] \
   || [ "$cli_h_ok" != "true" ] || [ "$caps_ok" != "true" ]; then
  echo "VERDICT=REPAIRABLE" | tee -a "$OUT/verdict.env"
  say "broken install is detectable -> desktop app can auto-repair"
  exit 0
fi

echo "VERDICT=FALSE_READY" | tee -a "$OUT/verdict.env"
echo "::error::Interrupted install reports READY but the backend cannot boot" \
     "(${missing_module:-import failure}). Preflight sees -h ok + desktop-capabilities ok," \
     "so the app shows ManagedReady with can_auto_repair=false and the user is stuck."
exit 1
