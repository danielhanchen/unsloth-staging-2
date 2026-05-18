#!/usr/bin/env bash
# Run every test under tests/ via pwsh, tally pass/fail/xfail.
# Usage:  bash run_all.sh            # baseline
#         HARNESS_MODE=fixed bash run_all.sh   # against codex-P2-fixed helpers

set -u
cd "$(dirname "$0")"

mode=${HARNESS_MODE:-baseline}
echo "== Mode: $mode =="

pass=0
fail=0
xfail=0
declare -a failures

for t in tests/t*.ps1; do
    name=$(basename "$t" .ps1)
    out=$(HARNESS_MODE="$mode" pwsh -NoProfile -NonInteractive -File "$t" 2>&1)
    rc=$?
    if [[ $rc -eq 0 ]]; then
        if echo "$out" | grep -q "^XFAIL"; then
            echo "XFAIL  $name"
            xfail=$((xfail + 1))
        elif echo "$out" | grep -q "^PASS"; then
            echo "PASS   $name"
            pass=$((pass + 1))
        else
            echo "?? $name (rc=0 but no PASS/XFAIL marker)"
            echo "$out" | sed 's/^/   /'
            fail=$((fail + 1))
            failures+=("$name")
        fi
    else
        echo "FAIL   $name (rc=$rc)"
        echo "$out" | sed 's/^/   /'
        fail=$((fail + 1))
        failures+=("$name")
    fi
done

echo
echo "=== Summary (mode=$mode) ==="
echo "  PASS:  $pass"
echo "  XFAIL: $xfail"
echo "  FAIL:  $fail"
if [[ $fail -gt 0 ]]; then
    echo "  Failed tests:"
    for n in "${failures[@]}"; do echo "    - $n"; done
fi
exit $fail
