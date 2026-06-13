#!/bin/sh
# Suite B: version_ge / _uv_version_ok / UV_MIN_VERSION logic, using the REAL
# functions from install.sh. Confirms the UV_MIN_VERSION bump to 0.8.16 drives
# correct upgrade decisions (incl. the 0.10-vs-0.8 numeric-compare trap and
# prerelease-of-exact-minimum rejection). Run: <shell> suite_b.sh <tests_dir> <install.sh>
set -e
TDIR="$1"; INSTALL_SH="$2"
[ -n "$TDIR" ] && [ -f "$INSTALL_SH" ] || { echo "usage: suite_b.sh <tests_dir> <install.sh>"; exit 2; }

# Extract version_ge and _uv_version_ok (each ends at its first line that is just "}")
awk '/^version_ge\(\) \{/{f=1} f{print} f&&/^\}/{exit}'      "$INSTALL_SH" >  "$TDIR/_ver.sh"
awk '/^_uv_version_ok\(\) \{/{f=1} f{print} f&&/^\}/{exit}'  "$INSTALL_SH" >> "$TDIR/_ver.sh"
. "$TDIR/_ver.sh"

# Read the REAL pinned minimum from install.sh
UV_MIN_VERSION=$(grep -m1 '^UV_MIN_VERSION=' "$INSTALL_SH" | cut -d'"' -f2)
echo "  UV_MIN_VERSION (from install.sh) = $UV_MIN_VERSION"
[ "$UV_MIN_VERSION" = "0.8.16" ] || { echo "  FAIL: expected pinned floor 0.8.16"; exit 1; }

PASS=0; FAIL=0
ok()  { PASS=$((PASS+1)); printf '  PASS  %s\n' "$1"; }
bad() { FAIL=$((FAIL+1)); printf '  FAIL  %s\n' "$1"; }

# --- version_ge: expect $1 >= $2 ? ---
vge() {  # a b expected(ge|lt)
  if version_ge "$1" "$2"; then got=ge; else got=lt; fi
  [ "$got" = "$3" ] && ok "version_ge $1 >= $2 -> $got" || bad "version_ge $1 vs $2: got $got want $3"
}
vge 0.8.16 0.8.16 ge
vge 0.8.15 0.8.16 lt
vge 0.7.22 0.8.16 lt          # the old pinned floor must be < new floor
vge 0.10.12 0.8.16 ge         # THE TRAP: 0.10 must compare >= 0.8 numerically (10>8), not lexically
vge 0.10.0 0.8.16 ge
vge 0.9.0 0.8.16 ge
vge 0.8.2 0.8.16 lt           # 2 < 16 numerically
vge 0.8.20 0.8.16 ge
vge 0.8.160 0.8.16 ge
vge 0.8.16.1 0.8.16 ge        # extra trailing component
vge 1.0.0 0.8.16 ge
vge 0.11.21 0.8.16 ge
vge 0.8.16 0.10.12 lt         # symmetry of the trap

# --- _uv_version_ok against UV_MIN_VERSION=0.8.16, using mock uv binaries ---
mk_uv() {  # version-string -> path to a mock `uv` printing "uv <ver>"
  d="$TDIR/mockuv_$$_$(echo "$1" | tr -c 'A-Za-z0-9' _)"; mkdir -p "$d"
  printf '#!/bin/sh\necho "uv %s"\n' "$1" > "$d/uv"; chmod +x "$d/uv"; echo "$d/uv"
}
uvok() {  # version expected(ok|reinstall)
  p=$(mk_uv "$1")
  if _uv_version_ok "$p"; then got=ok; else got=reinstall; fi
  [ "$got" = "$2" ] && ok "_uv_version_ok($1) -> $got" || bad "_uv_version_ok($1): got $got want $2"
}
uvok 0.7.22 reinstall          # current repo floor: now too old -> upgrade
uvok 0.8.15 reinstall          # just below the fix
uvok 0.8.16 ok                 # exactly the fix version
uvok 0.8.17 ok
uvok 0.10.12 ok                # the 0.10 trap again, end to end
uvok 0.11.21 ok
uvok 1.2.3 ok
uvok 0.8.16-rc1 reinstall      # prerelease of exact minimum is < stable 0.8.16
uvok 0.8.16+local reinstall    # pre-existing: ANY suffix on the exact floor -> reinstall (conservative; harmless)
uvok 0.8.17-alpha ok           # prerelease ABOVE min still satisfies floor
uvok notaversion reinstall
uvok '' reinstall

printf '  ------------------------------------\n'
printf '  SUITE B: %s passed, %s failed\n' "$PASS" "$FAIL"
[ "$FAIL" = 0 ]
