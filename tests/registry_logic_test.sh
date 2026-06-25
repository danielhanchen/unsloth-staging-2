#!/usr/bin/env bash
# Cross-OS validation for the UNSLOTH_NPM_REGISTRY change (unsloth PR #6663 / issue #6491).
# Exercises the registry-arg threading idiom and the _suggest_npm_registry hint helper
# extracted from the real studio/setup.sh, on whatever bash/npm the runner ships.
# Does NOT run a full install; it validates the new logic only.
set -uo pipefail

echo "### bash ${BASH_VERSION:-?} on $(uname -s) ###"
command -v npm >/dev/null 2>&1 && echo "npm: $(npm -v)" || echo "npm: not found"

FAIL=0
pass() { echo "PASS: $1"; }
fail() { echo "FAIL: $1"; FAIL=1; }

# 1. Scripts parse under this bash.
bash -n studio/setup.sh && pass "setup.sh parses" || fail "setup.sh parse"
bash -n build.sh        && pass "build.sh parses"  || fail "build.sh parse"

# 2. set -u-safe array idiom: empty expands to nothing, set expands to --registry <url>.
out_empty=$(set -u; A=(); printf '%s' "${A[@]+"${A[@]}"}")
[ -z "$out_empty" ] && pass "empty array -> no args" || fail "empty array -> '$out_empty'"
out_set=$(set -u; B=(--registry http://x/); printf '%s|' "${B[@]+"${B[@]}"}")
[ "$out_set" = "--registry|http://x/|" ] && pass "set array -> --registry <url>" || fail "set array -> '$out_set'"

# 3. Extract the real _suggest_npm_registry function and exercise it.
EX=$(mktemp)
awk '/^_suggest_npm_registry\(\) \{/{f=1} f{print} f&&/^\}$/{exit}' studio/setup.sh > "$EX"
[ -s "$EX" ] && pass "extracted _suggest_npm_registry" || fail "could not extract hint fn"

# Stub the output helpers the function calls, then source it.
C_WARN=""
step()    { printf "  %-15.15s%s\n" "$1" "$2"; }
substep() { printf "  %-15s%s\n" "" "$1"; }
# shellcheck disable=SC1090
source "$EX"

LOG_BLOCK=$(mktemp);  printf 'error: GET https://registry.npmjs.org/mermaid - 403\n' > "$LOG_BLOCK"
LOG_OTHER=$(mktemp);  printf 'npm ERR! ENOSPC: no space left on device\n'           > "$LOG_OTHER"

# 3a. Opted in (UNSLOTH_NPM_REGISTRY set) -> stays silent.
out=$( ( export UNSLOTH_NPM_REGISTRY=https://m/; _suggest_npm_registry "$LOG_BLOCK" ) 2>&1 )
[ -z "$out" ] && pass "opted-in: silent" || fail "opted-in not silent: $out"

# 3b. Registry block + mirror in env -> surfaces the mirror.
out=$( ( unset UNSLOTH_NPM_REGISTRY; export NPM_CONFIG_REGISTRY=https://mirror.corp/api/npm/; _suggest_npm_registry "$LOG_BLOCK" ) 2>&1 )
echo "$out" | grep -q "mirror.corp" && pass "block+env mirror: detected" || fail "block+env mirror: $out"

# 3c. Non-registry failure (ENOSPC) -> stays silent.
out=$( ( unset UNSLOTH_NPM_REGISTRY NPM_CONFIG_REGISTRY npm_config_registry; _suggest_npm_registry "$LOG_OTHER" ) 2>&1 )
[ -z "$out" ] && pass "non-registry failure: silent" || fail "non-registry not silent: $out"

# 4. P3 regression: mirror set ONLY in ~/.npmrc must be detected even when cwd has a
#    project .npmrc pinning npmjs.org (the function reads npm config from /).
if command -v npm >/dev/null 2>&1; then
    TD=$(mktemp -d); printf '{"name":"p","version":"1.0.0"}\n' > "$TD/package.json"
    printf 'registry=https://registry.npmjs.org/\n' > "$TD/.npmrc"
    FH=$(mktemp -d); printf 'registry=https://artifactory.corp/api/npm/\n' > "$FH/.npmrc"
    LOG3=$(mktemp); printf 'error: GET https://registry.npmjs.org/x - 403\n' > "$LOG3"
    out=$( ( cd "$TD"; export HOME="$FH"; unset UNSLOTH_NPM_REGISTRY NPM_CONFIG_REGISTRY npm_config_registry; _suggest_npm_registry "$LOG3" ) 2>&1 )
    echo "$out" | grep -q "artifactory.corp" \
        && pass "P3: ~/.npmrc mirror detected from pinned cwd" \
        || fail "P3: mirror not detected -> $out"
else
    echo "SKIP: npm not found, cannot run P3 check"
fi

echo "=== RESULT: $([ $FAIL -eq 0 ] && echo ALL PASS || echo FAILURES) ==="
exit $FAIL
