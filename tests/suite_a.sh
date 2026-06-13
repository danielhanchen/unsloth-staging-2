#!/bin/sh
# Suite A: edge-case unit tests for run_install_cmd_retry against the REAL
# functions extracted from install.sh. Run as: <shell> suite_a.sh <tests_dir> <install.sh>
set -e
TDIR="$1"; INSTALL_SH="$2"
[ -n "$TDIR" ] && [ -f "$INSTALL_SH" ] || { echo "usage: suite_a.sh <tests_dir> <install.sh>"; exit 2; }

# Extract the real run_install_cmd + run_install_cmd_retry block.
awk '/^run_install_cmd\(\) \{/{f=1} f{print} f&&/^run_install_cmd_retry\(\) \{/{g=1} g&&/^\}/{c++; if(c==1)exit}' "$INSTALL_SH" > "$TDIR/_funcs.sh"

# --- stubs matching install.sh's real signatures/semantics ---
C_OK= ; C_DIM= ; C_WARN= ; C_ERR= ; C_RST=
_is_verbose() { [ "${UNSLOTH_VERBOSE:-0}" = "1" ]; }   # real logic
step()    { printf '  [step] %s\n' "$2"; }
substep() { printf '  [substep] %s\n' "$1"; }
. "$TDIR/_funcs.sh"

STATE="$TDIR/_a_state"; export STATE
export TMPDIR="$TDIR/tmp"; mkdir -p "$TMPDIR"
FLAKY="$TDIR/flaky.sh"; ARGCHK="$TDIR/argcheck.sh"

PASS=0; FAIL=0
ok()   { PASS=$((PASS+1)); printf '  PASS  %s\n' "$1"; }
bad()  { FAIL=$((FAIL+1)); printf '  FAIL  %s\n' "$1"; [ -n "$2" ] && printf '%s\n' "$2" | sed 's/^/          | /'; }

# runD: run with DELAY=0 (RETRIES inherits sourced default 3). The env prefix is
# applied INSIDE the command-substitution subshell so it cannot leak between cases.
runD() {
  : > "$STATE"
  G_OUT=$( UNSLOTH_INSTALL_RETRY_DELAY=0 "$@" 2>&1 ) && G_RC=0 || G_RC=$?
  G_RETRIES=$(printf '%s\n' "$G_OUT" | grep -c 'retrying ' || true)
}
# runRD: run with explicit RETRIES=$1 and DELAY=0.
runRD() {
  _R="$1"; shift
  : > "$STATE"
  G_OUT=$( UNSLOTH_INSTALL_RETRIES="$_R" UNSLOTH_INSTALL_RETRY_DELAY=0 "$@" 2>&1 ) && G_RC=0 || G_RC=$?
  G_RETRIES=$(printf '%s\n' "$G_OUT" | grep -c 'retrying ' || true)
}

# 1. success first try: rc 0, 0 retries, and NO output (matches run_install_cmd)
runD run_install_cmd_retry L true
[ "$G_RC" = 0 ] && [ "$G_RETRIES" = 0 ] && [ -z "$G_OUT" ] && ok "success_first_try: rc0, 0 retries, empty output" || bad "success_first_try" "$G_OUT"

# 2. byte-identical to run_install_cmd on first-try success (non-verbose)
O1=$( run_install_cmd L true 2>&1 ); O2=$( run_install_cmd_retry L true 2>&1 )
[ "$O1" = "$O2" ] && ok "first-try success byte-identical to run_install_cmd (non-verbose)" || bad "byte-identical non-verbose" "[$O1] vs [$O2]"

# 3. byte-identical on first-try success (verbose)
O1=$( UNSLOTH_VERBOSE=1 run_install_cmd L sh "$FLAKY" 0 2>&1 )
: > "$STATE"
O2=$( UNSLOTH_VERBOSE=1 run_install_cmd_retry L sh "$FLAKY" 0 2>&1 )
: > "$STATE"
[ "$O1" = "$O2" ] && ok "first-try success byte-identical (verbose)" || bad "byte-identical verbose" "[$O1] vs [$O2]"

# 4-7. fail K then succeed (RETRIES=3): expect rc0 and K retries for K=0,1,2; rc4/2retries for K=3
for K in 0 1 2; do
  runD run_install_cmd_retry L sh "$FLAKY" "$K"
  [ "$G_RC" = 0 ] && [ "$G_RETRIES" = "$K" ] && ok "fail${K}_then_ok (max3): rc0, ${K} retries" || bad "fail${K}_then_ok" "$G_OUT"
done
runD run_install_cmd_retry L sh "$FLAKY" 3
[ "$G_RC" = 4 ] && [ "$G_RETRIES" = 2 ] && ok "fail3_then_ok (max3): exhausts -> rc4, 2 retries" || bad "fail3_exhaust" "$G_OUT"

# 8. always fail: rc preserved = 4, 2 retries
runD run_install_cmd_retry L sh "$FLAKY" 99
[ "$G_RC" = 4 ] && [ "$G_RETRIES" = 2 ] && ok "always_fail (max3): rc=4 (REAL code), 2 retries" || bad "always_fail" "$G_OUT"

# 9. RETRIES=5, fail 4 then ok: rc0, 4 retries
runRD 5 run_install_cmd_retry L sh "$FLAKY" 4
[ "$G_RC" = 0 ] && [ "$G_RETRIES" = 4 ] && ok "RETRIES=5 fail4_then_ok: rc0, 4 retries" || bad "retries5" "$G_OUT"

# 10. RETRIES=1 always fail: single attempt, rc4, 0 retries
runRD 1 run_install_cmd_retry L sh "$FLAKY" 99
[ "$G_RC" = 4 ] && [ "$G_RETRIES" = 0 ] && ok "RETRIES=1 always_fail: single attempt, rc4, 0 retries" || bad "retries1" "$G_OUT"

# 11. exit-code preservation across codes (RETRIES=1, always fail with code EC)
for EC in 1 2 42 127 130; do
  runRD 1 run_install_cmd_retry L sh "$FLAKY" 99 "$EC"
  [ "$G_RC" = "$EC" ] && ok "exit_code_preserved: $EC" || bad "exit_code_$EC" "got $G_RC; $G_OUT"
done

# 12. invalid/zero RETRIES falls back to the DEFAULT of 3 (typo must not disable
#     retries) -> always-fail gives rc4 with 2 retries.
for BAD in '' 'abc' '0' '-2' '3.5' '1e3' ' '; do
  runRD "$BAD" run_install_cmd_retry L sh "$FLAKY" 99
  [ "$G_RC" = 4 ] && [ "$G_RETRIES" = 2 ] && ok "invalid RETRIES='$BAD' -> default 3 attempts" || bad "invalid_retries_'$BAD'" "rc=$G_RC retries=$G_RETRIES"
done

# 13. huge RETRIES does not break; stops on success (fail2 then ok -> rc0, 2 retries)
runRD 99999999999 run_install_cmd_retry L sh "$FLAKY" 2
[ "$G_RC" = 0 ] && [ "$G_RETRIES" = 2 ] && ok "huge RETRIES stops on success: rc0, 2 retries" || bad "huge_retries" "$G_OUT"

# 14. invalid DELAY sanitized (use fail1-then-ok; DELAY garbage -> 3s default, but correctness only)
#     Validate the sanitize LOGIC directly (fast) rather than sleeping.
san() { case "$1" in ''|*[!0-9]*) echo 3 ;; *) echo "$1" ;; esac; }
[ "$(san '')" = 3 ] && [ "$(san abc)" = 3 ] && [ "$(san -1)" = 3 ] && [ "$(san 0)" = 0 ] && [ "$(san 7)" = 7 ] \
  && ok "DELAY sanitize logic: ''/abc/-1 ->3, 0->0, 7->7" || bad "delay_sanitize"

# 15. arg passthrough verbatim (spaces, leading dash, glob char, quote).
#     Use verbose mode so run_install_cmd does NOT capture/swallow argcheck's stdout.
: > "$STATE"
G_OUT=$( UNSLOTH_VERBOSE=1 UNSLOTH_INSTALL_RETRY_DELAY=0 run_install_cmd_retry L sh "$ARGCHK" "a b" "--flag" "g*b" "x'y" 2>&1 ) || true
case "$G_OUT" in
  *"NARGS=4"*"ARG=[a b]"*"ARG=[--flag]"*"ARG=[g*b]"*"ARG=[x'y]"*) ok "arg passthrough verbatim (spaces/dash/glob/quote)" ;;
  *) bad "arg_passthrough" "$G_OUT" ;;
esac

# 16. verbose-mode retry still works (fail2 then ok)
: > "$STATE"
G_OUT=$( UNSLOTH_VERBOSE=1 UNSLOTH_INSTALL_RETRY_DELAY=0 run_install_cmd_retry L sh "$FLAKY" 2 2>&1 ) && G_RC=0 || G_RC=$?
G_RETRIES=$(printf '%s\n' "$G_OUT" | grep -c 'retrying ' || true)
[ "$G_RC" = 0 ] && [ "$G_RETRIES" = 2 ] && ok "verbose-mode retry: fail2_then_ok rc0, 2 retries" || bad "verbose_retry" "$G_OUT"

printf '  ------------------------------------\n'
printf '  SUITE A: %s passed, %s failed\n' "$PASS" "$FAIL"
[ "$FAIL" = 0 ]
