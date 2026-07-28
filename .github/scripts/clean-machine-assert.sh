#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
#
# Assert the clean-machine contract after an install attempt.
#
#   absent   The toolchain really was absent for the whole run. Guards against a
#            leg that "passed" only because masking silently failed, or because
#            the installer quietly installed Xcode CLT behind our back.
#   notools  The trace recorded no compiler/git/brew invocation (trace mode).
#   nobuild  The install log shows no source build (no sdist, no cmake, no
#            "Building wheel"). This is the wheels-only contract.
#
# Usage: bash .github/scripts/clean-machine-assert.sh absent notools nobuild
set -uo pipefail

LOG="${INSTALL_LOG:-logs/install.log}"
TRACE="${UNSLOTH_TOOL_TRACE:-}"
rc=0

fail() { echo "::error::$*"; rc=1; }
ok()   { echo "[assert] OK  $*"; }

for check in "$@"; do
  case "$check" in

    absent)
      # Deliberately NOT a `command -v` check. On a real virgin Mac /usr/bin/git and
      # /usr/bin/cc EXIST as Xcode CLT stubs, so `command -v git` SUCCEEDS -- running
      # it is what fails ("xcrun: error: invalid active developer path"). Asserting on
      # `command -v` would therefore be unfaithful and would fail on a correctly masked
      # runner. The honest invariant is: the tool must not WORK.
      if xcode-select -p >/dev/null 2>&1; then
        fail "xcode-select -p still resolves to $(xcode-select -p 2>/dev/null); not a clean Mac"
      else
        ok "xcode-select -p fails (the gate a virgin Mac hits)"
      fi
      for tool in git cc clang cmake; do
        command -v "$tool" >/dev/null 2>&1 || { ok "$tool not on PATH"; continue; }
        if "$tool" --version >/dev/null 2>&1; then
          fail "toolchain still usable: '$tool --version' succeeded ($(command -v "$tool")); masking failed"
        else
          ok "$tool present but non-functional (CLT stub), as on a clean Mac"
        fi
      done
      # brew is a plain binary with no stub, so absence from PATH is the right test.
      if command -v brew >/dev/null 2>&1; then
        fail "Homebrew still on PATH at $(command -v brew); masking failed"
      else
        ok "brew absent"
      fi
      ;;

    notools)
      if [ -z "$TRACE" ] || [ ! -f "$TRACE" ]; then
        fail "notools requested but no trace file (\$UNSLOTH_TOOL_TRACE=$TRACE)"
      else
        # git is legitimate under --local (it installs unsloth-zoo from a git URL);
        # UNSLOTH_ALLOW_TOOLS lets that leg allow-list it explicitly.
        allow="${UNSLOTH_ALLOW_TOOLS:-}"
        hits=""
        while IFS=$'\t' read -r tool _rest; do
          [ -n "$tool" ] || continue
          case " $allow " in *" $tool "*) continue ;; esac
          hits="$hits $tool"
        done < "$TRACE"
        if [ -n "$hits" ]; then
          fail "installer invoked toolchain:$(echo "$hits" | tr ' ' '\n' | sort -u | tr '\n' ' ')"
          echo "---- tool trace ----"; sort -u "$TRACE" | head -50
        else
          ok "no compiler/git/brew invocation recorded"
        fi
      fi
      ;;

    nobuild)
      if [ ! -f "$LOG" ]; then
        fail "nobuild requested but $LOG is missing"
      else
        # uv/pip say "Building wheel for X" / "Running setup.py" only on an sdist.
        if grep -qiE "building wheel for|running setup\.py|preparing metadata \(setup\.py\)|building editable for" "$LOG"; then
          fail "install compiled from source (sdist fallback) -- wheels-only contract broken"
          grep -iE "building wheel for|running setup\.py" "$LOG" | head -20
        else
          ok "no source build in $LOG"
        fi
      fi
      ;;

    *)
      fail "unknown check '$check'"
      ;;
  esac
done

exit "$rc"
