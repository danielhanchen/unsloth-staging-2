#!/bin/sh
# Staging-CI only. Spoofs the Fedora/Bazzite AMD host reported in unslothai/unsloth#8731 on a
# hosted GitHub runner and asserts the index the installer actually resolves, plus the safety
# controls that must NOT reroute. Not part of the PR: it lives on the throwaway staging branch.
#
# It does not reimplement the spoof. tests/sh/test_rocm_no_version_arch_route_e2e.sh already
# splices install.sh's helper functions and its TOP-LEVEL reroute block out by stable markers and
# drives them behind stub rocminfo / amd-smi / hipconfig / rpm / dpkg-query and redirected
# /opt/rocm, /dev/kfd, /sys/class/kfd, /sys/bus/pci. This file sources that harness's PROLOGUE
# (everything above its first assertion) and adds the rows the suite does not print: the resolved
# torch constraint, the exported backend, a Darwin host, and a Strix gfx1151 beside a discrete
# gfx1201. The reroute is EXECUTED, never grepped for.
set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
E2E="$REPO_ROOT/tests/sh/test_rocm_no_version_arch_route_e2e.sh"
MARKER='=== test_rocm_no_version_arch_route_e2e ==='

[ -r "$E2E" ] || { echo "FAIL: no e2e harness at $E2E"; exit 1; }
grep -qF "$MARKER" "$E2E" || {
    echo "FAIL: the e2e harness no longer prints '$MARKER'; the prologue split point is gone."
    exit 1
}

_PRO=$(mktemp)
awk -v m="$MARKER" 'index($0, m) { exit } { print }' "$E2E" > "$_PRO"
# The prologue must carry the splice, the mocks and the runner, or this file is asserting nothing.
for _need in _redirect reset_host mock_rocminfo mock_amdsmi mock_hipconfig_offpath mock_rpm \
             mock_dpkg mock_nvidia_smi mock_kfd mock_pci_amd_display mock_lspci run_installer; do
    grep -q "^$_need()" "$_PRO" || {
        echo "FAIL: the e2e prologue no longer defines $_need()"; rm -f "$_PRO"; exit 1
    }
done

# Sourcing it runs the splice and its own integrity gates: every helper install.sh must still
# define at column 0, the block reaching its end marker, and both halves parsing under sh and bash.
# A missing helper fails silently as "cpu" -- which is the bug this PR fixes -- so those gates are
# the reason this file reuses the harness instead of copying it.
# shellcheck disable=SC1090
. "$_PRO"
rm -f "$_PRO"

_RUN_SHELL="${SPOOF_SHELL:-bash}"
_REAL_UNAME=$(command -v uname)

PASS=0
FAIL=0
_TABLE="$_ROOT/table.md"
: > "$_TABLE"

mock_uname() {   # $1 = kernel name reported by `uname -s`
    cat > "$_MOCK/uname" <<MOCK
#!/bin/sh
if [ \$# -eq 0 ] || [ "\$1" = "-s" ]; then printf '%s\n' '$1'; exit 0; fi
exec $_REAL_UNAME "\$@"
MOCK
    chmod +x "$_MOCK/uname"
}

# One installer run, every field read from it. run_installer only prints INDEX/GFX/RADEON, so the
# probe is rewritten here to add the constraint and the exported backend.
_CAP=""
drive() {   # $1 = _ARCH (default x86_64)
    cat > "$_ROOT/probe.sh" <<EOF
set -e
unset CUDA_VISIBLE_DEVICES UNSLOTH_ROCM_GFX_ARCH UNSLOTH_TORCH_INDEX_URL \\
      UNSLOTH_TORCH_INDEX_FAMILY UNSLOTH_AMD_ROCM_MIRROR UNSLOTH_PYTORCH_MIRROR \\
      ROCM_PATH ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES HSA_OVERRIDE_GFX_VERSION 2>/dev/null || true
_ARCH=${1:-x86_64}
_torch_index_pinned=false
SKIP_TORCH=false
TORCH_CONSTRAINT=""
TORCHVISION_CONSTRAINT=""
TORCHAUDIO_CONSTRAINT=""
. "$_FUNCS"
. "$_BLOCK"
printf 'INDEX=%s\n' "\$TORCH_INDEX_URL"
printf 'GFX=%s\n' "\${UNSLOTH_ROCM_GFX_ARCH:-}"
printf 'RADEON=%s\n' "\${_amd_gpu_radeon:-}"
printf 'CONSTRAINT=%s\n' "\$TORCH_CONSTRAINT"
printf 'BACKEND=%s\n' "\${UNSLOTH_TORCH_BACKEND:-}"
EOF
    _CAP=$(PATH="$_MOCK:$_TOOLS" "$_RUN_SHELL" "$_ROOT/probe.sh" 2>"$_ROOT/probe.err" \
           || printf 'INDEX=<installer exited %s>\n' "$?")
}

f() { printf '%s\n' "$_CAP" | sed -n "s/^$1=//p"; }

check() {   # $1 = what, $2 = expected, $3 = actual
    if [ "$3" = "$2" ]; then
        PASS=$((PASS + 1))
    else
        FAIL=$((FAIL + 1))
        echo "  FAIL: $1 (expected '$2', got '$3')"
    fi
}

# label | expected index | expected gfx export | expected constraint | expected backend | radeon
row() {
    _label="$1"; _xi="$2"; _xg="$3"; _xc="$4"; _xb="$5"; _xr="$6"
    _fail_before=$FAIL
    check "$_label: index"      "$_xi" "$(f INDEX)"
    check "$_label: gfx"        "$_xg" "$(f GFX)"
    check "$_label: constraint" "$_xc" "$(f CONSTRAINT)"
    check "$_label: backend"    "$_xb" "$(f BACKEND)"
    [ "$_xr" = "-" ] || check "$_label: radeon" "$_xr" "$(f RADEON)"
    if [ "$FAIL" -eq "$_fail_before" ]; then _mark="ok"; else _mark="**MISMATCH**"; fi
    printf '| %s | `%s` | `%s` | `%s` | `%s` | %s |\n' \
        "$_label" "$(f INDEX)" "$(f GFX)" "$(f CONSTRAINT)" "$(f BACKEND)" "$_mark" >> "$_TABLE"
}

fedora_no_version_host() {   # $@ = the gfx arches rocminfo reports; no version source readable
    reset_host
    mock_uname Linux
    mock_rocminfo "$@"
    mock_amdsmi "N/A" "$@"      # amd-smi prints "ROCm version: N/A"
    mock_hipconfig_offpath ""   # hipconfig on PATH, prints nothing
    mock_rpm                    # rpm installed, rocm-core not
    mock_kfd
    mock_pci_amd_display
    # /opt/rocm/.info/version is never created by reset_host, and dpkg-query is absent from $_MOCK.
}

_BASE="https://download.pytorch.org/whl"
_AMD="https://repo.amd.com/rocm/whl"
_C211="torch>=2.11.0,<2.12.0"

echo "=== spoofed Fedora/Bazzite AMD host, shell=$_RUN_SHELL, runner=${RUNNER_OS:-local}/$($_REAL_UNAME -s) ==="
printf '| host | TORCH_INDEX_URL | UNSLOTH_ROCM_GFX_ARCH | TORCH_CONSTRAINT | backend | |\n' >> "$_TABLE"
printf '| --- | --- | --- | --- | --- | --- |\n' >> "$_TABLE"

# ---------------------------------------------------------------- the reported host (#8731)
fedora_no_version_host gfx1201
mock_lspci "Navi 48 [Radeon RX 9070 XT]"
# First prove the premise: all five version sources really are unreadable on this spoofed host.
cat > "$_ROOT/vercheck.sh" <<EOF
. "$_FUNCS"
printf 'TAG=%s\n' "\$(_detect_rocm_version_tag 2>/dev/null)"
EOF
check "reported host: no ROCm version is readable from any of the five sources" "TAG=" \
    "$(PATH="$_MOCK:$_TOOLS" "$_RUN_SHELL" "$_ROOT/vercheck.sh" 2>/dev/null)"
drive
row "Fedora/Bazzite RX 9070 XT (gfx1201), no readable ROCm version" \
    "$_AMD/gfx120X-all/" "gfx1201" "$_C211" "rocm" "false"

# ---------------------------------------------------------------- safety controls
reset_host; mock_uname Linux; mock_nvidia_smi "12.8" "9.0"
drive
row "NVIDIA-only host" "$_BASE/cu128" "" "torch>=2.4,<2.12.0" "cuda" "false"

reset_host; mock_uname Linux
drive
row "CPU-only host (no GPU at all)" "$_BASE/cpu" "" "" "cpu" "false"

# The reroute is gated on `uname -s` = Linux, so the same AMD host on a Mac must not take it.
fedora_no_version_host gfx1201
mock_uname Darwin
drive
row "macOS host with the same AMD spoof present" "$_BASE/cpu" "" "" "cpu" "-"

# Strix iGPU beside a discrete RDNA4 card: two families, no agreement, so no reroute and NO arch
# export -- naming a card here would overrule setup.sh's visibility-aware pick.
fedora_no_version_host gfx1151 gfx1201
mock_lspci "Navi 48 [Radeon RX 9070 XT]"
drive
row "Strix Halo gfx1151 beside a discrete gfx1201" "$_BASE/cpu" "" "" "cpu" "false"

# Strix alone still reroutes, and must not be branded a repo.radeon.com install: rocminfo calls
# every consumer card "Radeon" and the AMD per-arch mirror's own base path contains "rocm".
fedora_no_version_host gfx1151
mock_lspci "Strix Halo [Radeon 8060S]"
drive
row "Strix Halo gfx1151 alone" "$_AMD/gfx1151/" "gfx1151" "$_C211" "rocm" "false"

# gfx906 is served only through a generic rocmX.Y leaf: no per-arch family exists to reroute to.
fedora_no_version_host gfx906
drive
row "gfx906 (MI50), no readable ROCm version" "$_BASE/cpu" "" "" "cpu" "false"

# ROCm torch wheels are x86_64-only.
fedora_no_version_host gfx1201
drive aarch64
row "the reported host on aarch64" "$_BASE/cpu" "" "" "cpu" "false"

echo ""
cat "$_TABLE"
echo ""
if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
    {
        echo "### Spoofed Fedora/Bazzite AMD host -- \`$_RUN_SHELL\` on ${RUNNER_OS:-local}"
        echo ""
        cat "$_TABLE"
        echo ""
    } >> "$GITHUB_STEP_SUMMARY"
fi
echo "  passed: $PASS, failed: $FAIL"
[ "$FAIL" -eq 0 ]
