#!/bin/sh
# Validates the REAL version_ge / _uv_version_ok functions from install.sh on the
# host shell (bash on Linux, /bin/sh on macOS). Confirms the 0.7.22 floor, numeric
# ordering (0.10.0 >= 0.7.22), and prerelease handling. Exits non-zero on any miss.
set -u
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
INSTALL_SH="$ROOT/install.sh"
[ -f "$INSTALL_SH" ] || { echo "install.sh not found at $INSTALL_SH"; exit 2; }

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
FNS="$WORK/fns.sh"
awk '/^version_ge\(\) \{/,/^\}/'      "$INSTALL_SH" >  "$FNS"
awk '/^_uv_version_ok\(\) \{/,/^\}/'  "$INSTALL_SH" >> "$FNS"
grep -q "version_ge" "$FNS" || { echo "could not extract version_ge"; exit 2; }
grep -q "_uv_version_ok" "$FNS" || { echo "could not extract _uv_version_ok"; exit 2; }

UV_MIN_VERSION="$(awk -F'"' '/^UV_MIN_VERSION=/{print $2; exit}' "$INSTALL_SH")"
echo "install.sh UV_MIN_VERSION=$UV_MIN_VERSION  (shell: ${0##*/} via $(command -v "${SHELL:-sh}" 2>/dev/null || echo sh))"
[ "$UV_MIN_VERSION" = "0.7.22" ] || { echo "FAIL: expected floor 0.7.22, got $UV_MIN_VERSION"; exit 1; }

. "$FNS"

PASS=0; FAIL=0
check() {  # $1=version string  $2=expected OK|NO  $3=note
    printf '#!/bin/sh\necho "%s"\n' "$1" > "$WORK/fakeuv"; chmod +x "$WORK/fakeuv"
    if _uv_version_ok "$WORK/fakeuv"; then got=OK; else got=NO; fi
    if [ "$got" = "$2" ]; then r=PASS; PASS=$((PASS+1)); else r=FAIL; FAIL=$((FAIL+1)); fi
    printf "%-5s exp=%-3s got=%-3s [%s] '%s'\n" "$r" "$2" "$got" "$3" "$1"
}

check "uv 0.7.21"     NO "below floor"
check "uv 0.7.22"     OK "exact floor"
check "uv 0.7.23"     OK "newer patch"
check "uv 0.8.0"      OK "newer minor"
check "uv 0.10.0"     OK "numeric 0.10>0.7"
check "uv 0.100.0"    OK "big minor"
check "uv 1.0.0"      OK "major"
check "uv 0.7.22-rc1" NO "prerelease of floor"
check "uv 0.7.23-rc1" OK "prerelease above floor"
check "uv 0.8.0+local" OK "build metadata"
check "uv abc"        NO "non-numeric"
check "uv"            NO "no version field"
check "uv 0.7"        NO "two-component below"
check "uv 0.8"        OK "two-component above"

echo "RESULT: PASS=$PASS FAIL=$FAIL"
[ "$FAIL" -eq 0 ]
