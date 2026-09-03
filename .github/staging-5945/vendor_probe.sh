#!/usr/bin/env bash
# Vendor simulation harness for install.sh's get_torch_index_url().
#
# GitHub-hosted standard runners have no GPU of any vendor, so the NVIDIA and
# AMD legs are simulated exactly the way the repo's own tests/sh suite does it:
# the routing helpers are sed-extracted out of install.sh with their hardcoded
# absolute paths rewritten into empty temp dirs, and fake nvidia-smi / rocminfo /
# amd-smi scripts are placed on a minimal PATH. See
# tests/sh/test_get_torch_index_url.sh:24-166 for the original of this mechanism.
#
# The expected leaf is derived from install.sh's own documented branch order so
# the same harness is meaningful on every runner family:
#   1. UNSLOTH_TORCH_INDEX_URL      -> verbatim
#   2. UNSLOTH_TORCH_INDEX_FAMILY   -> $mirror/$family
#   3. uname -s = Darwin            -> $mirror/cpu   (before any GPU probe)
#   4. usable NVIDIA                -> CUDA ladder
#   5. uname -m not x86_64/amd64    -> $mirror/cpu
#   6. no AMD ROCm GPU              -> $mirror/cpu
set -u

INSTALL_SH="${1:-install.sh}"
PASS=0
FAIL=0
TMP_ROOT=$(mktemp -d)
trap 'rm -rf "$TMP_ROOT"' EXIT

FAKE_SMI_DIR="$TMP_ROOT/no-smi"
FAKE_ROCM_DIR="$TMP_ROOT/no-rocm"
# _has_usable_nvidia_gpu() falls back to /proc/driver/nvidia/gpus when nvidia-smi
# is absent. GitHub-hosted runners have no such directory, but redirect it anyway
# so the harness gives the same answer when run on a developer box that does.
FAKE_PROC_DIR="$TMP_ROOT/no-proc-nvidia"
mkdir -p "$FAKE_SMI_DIR" "$FAKE_ROCM_DIR" "$FAKE_PROC_DIR"

FUNC_FILE="$TMP_ROOT/funcs.sh"
{
    sed -n '/^_run_bounded()/,/^}/p'                      "$INSTALL_SH"; echo
    sed -n '/^_cvd_hides_nvidia()/,/^}/p'                 "$INSTALL_SH"; echo
    sed -n '/^_has_amd_rocm_gpu()/,/^}/p'                 "$INSTALL_SH"; echo
    sed -n '/^_has_usable_nvidia_gpu()/,/^}/p'            "$INSTALL_SH"; echo
    sed -n '/^_ensure_rocm_probe_env()/,/^}/p'            "$INSTALL_SH"; echo
    sed -n '/^_probe_amd_gfx_arch()/,/^}/p'               "$INSTALL_SH"; echo
    sed -n '/^_amd_gpu_present_via_pci()/,/^}/p'          "$INSTALL_SH"; echo
    sed -n '/^_infer_amd_gfx_arch_from_gpu_name()/,/^}/p' "$INSTALL_SH"; echo
    sed -n '/^_infer_linux_amd_gfx_arch()/,/^}/p'         "$INSTALL_SH"; echo
    sed -n '/^_amd_arch_index_family_for_gfx()/,/^}/p'    "$INSTALL_SH"; echo
    sed -n '/^_trim_index_path_slashes()/,/^}/p'          "$INSTALL_SH"; echo
    sed -n '/^get_torch_index_url()/,/^}/p'               "$INSTALL_SH"
} | sed -e "s|/usr/bin/nvidia-smi|$FAKE_SMI_DIR/nvidia-smi-absent|g" \
        -e "s|/opt/rocm|$FAKE_ROCM_DIR|g" \
        -e "s|/proc/driver/nvidia/gpus|$FAKE_PROC_DIR|g" \
  > "$FUNC_FILE"

if ! grep -q '^get_torch_index_url()' "$FUNC_FILE"; then
    echo "::error file=$INSTALL_SH::get_torch_index_url() could not be extracted"
    exit 1
fi

# ---- vendor mocks (same shape as tests/sh/test_get_torch_index_url.sh) ----
mk_nvidia() {
    _d="$TMP_ROOT/mock-nvidia"; mkdir -p "$_d"
    cat > "$_d/nvidia-smi" <<'MOCK'
#!/bin/sh
case "$1" in
    -L) echo "GPU 0: NVIDIA GeForce RTX 5090 (UUID: GPU-fake-uuid)" ;;
    *)
cat <<'SMI_OUT'
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 570.86.10   Driver Version: 570.86.10   CUDA Version: 12.8        |
+-----------------------------------------------------------------------------+
SMI_OUT
    ;;
esac
MOCK
    chmod +x "$_d/nvidia-smi"
    echo "$_d"
}

mk_amd() {
    _d="$TMP_ROOT/mock-amd"; mkdir -p "$_d"
    cat > "$_d/rocminfo" <<'MOCK'
#!/bin/sh
printf 'Agent 2\n  Name:                    gfx1100\n  Marketing Name:          AMD Radeon RX 7900 XTX\n'
MOCK
    cat > "$_d/amd-smi" <<'MOCK'
#!/bin/sh
case "$1" in
    list) printf 'GPU: 0\n  BDF: 0000:03:00.0\n  NAME: gfx1100\n' ;;
    *) printf 'AMDSMI Tool: 25.0.1 | AMDSMI Library version: 25.0.1.0 | ROCm version: 6.4\n' ;;
esac
MOCK
    chmod +x "$_d/rocminfo" "$_d/amd-smi"
    echo "$_d"
}

# On Windows (git-bash / MSYS) a hand-built minimal PATH breaks bash itself: it
# needs its own msys DLLs and the Windows system directories to start at all, and
# `ln -s` is not a real symlink there. Hosted Windows runners ship no nvidia-smi,
# rocminfo or amd-smi (the "Prove there is no real GPU" workflow step logs this),
# so prepending the mock dir to the inherited PATH is equally hermetic there.
case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) MINIMAL_PATH=0 ;;
    *)                    MINIMAL_PATH=1 ;;
esac

TOOLS_DIR="$TMP_ROOT/tools"; mkdir -p "$TOOLS_DIR"
if [ "$MINIMAL_PATH" -eq 1 ]; then
    for _cmd in uname grep sed head sh bash cat awk printf tr ls cut sort timeout; do
        _real=$(command -v "$_cmd" 2>/dev/null || true)
        [ -n "$_real" ] || continue
        ln -sf "$_real" "$TOOLS_DIR/$_cmd" 2>/dev/null || cp -f "$_real" "$TOOLS_DIR/$_cmd" 2>/dev/null || true
    done
    BASE_PATH="$TOOLS_DIR"
else
    BASE_PATH="$PATH"
fi

# Vendor / pin variables that must never leak in from the host or the workflow.
UNSET_VARS="UNSLOTH_TORCH_INDEX_URL UNSLOTH_TORCH_INDEX_FAMILY UNSLOTH_PYTORCH_MIRROR
UNSLOTH_AMD_ROCM_MIRROR UNSLOTH_ROCM_GFX_ARCH UNSLOTH_TORCH_BACKEND
CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES"

# run_route <mock-dir|none> [extra env assignments...]
run_route() {
    _mock="$1"; shift
    if [ "$_mock" = "none" ]; then _p="$BASE_PATH"; else _p="$_mock:$BASE_PATH"; fi
    # Word splitting is the point here: expand UNSET_VARS into `-u NAME` pairs.
    # shellcheck disable=SC2086,SC2046
    set -- $(for _v in $UNSET_VARS; do printf -- '-u\n%s\n' "$_v"; done) PATH="$_p" "$@"
    env "$@" bash -c ". '$FUNC_FILE'; get_torch_index_url" 2>/dev/null
}

assert_eq() {
    _label="$1"; _expected="$2"; _actual="$3"
    if [ "$_actual" = "$_expected" ]; then
        echo "  PASS: $_label -> $_actual"; PASS=$((PASS + 1))
    else
        echo "  FAIL: $_label (expected '$_expected', got '$_actual')"; FAIL=$((FAIL + 1))
    fi
}

BASE="https://download.pytorch.org/whl"
OS_NAME=$(uname -s)
ARCH=$(uname -m)

case "$OS_NAME" in
    Darwin)
        # install.sh short-circuits Darwin to the CPU wheel index BEFORE probing
        # any vendor, so all three vendor legs must agree on cpu.
        EXP_NVIDIA="$BASE/cpu"; EXP_AMD="$BASE/cpu" ;;
    *)
        EXP_NVIDIA="$BASE/cu128"
        case "$ARCH" in
            x86_64|amd64) EXP_AMD="$BASE/rocm6.4" ;;
            *)            EXP_AMD="$BASE/cpu" ;;   # non-x86_64 bails before the AMD branch
        esac ;;
esac
EXP_CPU="$BASE/cpu"

echo "=== vendor simulation on $OS_NAME/$ARCH ==="
NV=$(mk_nvidia)
AMD=$(mk_amd)

echo "--- vendor: NVIDIA (mock nvidia-smi, driver CUDA 12.8) ---"
assert_eq "NVIDIA routing" "$EXP_NVIDIA" "$(run_route "$NV")"

echo "--- vendor: AMD / ROCm (mock rocminfo + amd-smi, gfx1100, ROCm 6.4) ---"
assert_eq "AMD routing" "$EXP_AMD" "$(run_route "$AMD")"

echo "--- vendor: CPU-only (no vendor tool on PATH) ---"
assert_eq "CPU-only routing" "$EXP_CPU" "$(run_route none)"

echo "--- vendor masking: CUDA_VISIBLE_DEVICES hides the NVIDIA GPU ---"
assert_eq "CVD='' hides NVIDIA"  "$EXP_CPU" "$(run_route "$NV" CUDA_VISIBLE_DEVICES=)"
assert_eq "CVD=-1 hides NVIDIA"  "$EXP_CPU" "$(run_route "$NV" CUDA_VISIBLE_DEVICES=-1)"
assert_eq "CVD=0 does not hide"  "$EXP_NVIDIA" "$(run_route "$NV" CUDA_VISIBLE_DEVICES=0)"

echo "--- mixed host: NVIDIA wins over AMD ---"
MIXED="$TMP_ROOT/mock-mixed"; mkdir -p "$MIXED"
cp "$NV/nvidia-smi" "$AMD/rocminfo" "$AMD/amd-smi" "$MIXED/"
assert_eq "NVIDIA beats AMD" "$EXP_NVIDIA" "$(run_route "$MIXED")"
assert_eq "AMD wins once NVIDIA is masked" "$EXP_AMD" "$(run_route "$MIXED" CUDA_VISIBLE_DEVICES=-1)"

echo "--- explicit vendor pins (platform independent, evaluated before any probe) ---"
assert_eq "FAMILY=cu130 on a CPU-only host" "$BASE/cu130" \
    "$(run_route none UNSLOTH_TORCH_INDEX_FAMILY=cu130)"
assert_eq "FAMILY=rocm7.2 on a CPU-only host" "$BASE/rocm7.2" \
    "$(run_route none UNSLOTH_TORCH_INDEX_FAMILY=rocm7.2)"
assert_eq "FAMILY=cpu overrides a real NVIDIA probe" "$BASE/cpu" \
    "$(run_route "$NV" UNSLOTH_TORCH_INDEX_FAMILY=cpu)"
assert_eq "INDEX_URL is returned verbatim" "https://mirror.example.test/whl/cu999" \
    "$(run_route "$NV" UNSLOTH_TORCH_INDEX_URL=https://mirror.example.test/whl/cu999)"
assert_eq "INDEX_URL beats FAMILY" "https://mirror.example.test/whl/gfx110X-all" \
    "$(run_route none UNSLOTH_TORCH_INDEX_URL=https://mirror.example.test/whl/gfx110X-all UNSLOTH_TORCH_INDEX_FAMILY=cu128)"
assert_eq "PYTORCH_MIRROR rebases the CPU leaf" "https://mirror.example.test/whl/cpu" \
    "$(run_route none UNSLOTH_PYTORCH_MIRROR=https://mirror.example.test/whl)"
assert_eq "PYTORCH_MIRROR rebases the vendor leaf" "https://mirror.example.test/whl/${EXP_NVIDIA##*/}" \
    "$(run_route "$NV" UNSLOTH_PYTORCH_MIRROR=https://mirror.example.test/whl)"

echo ""
echo "=== $OS_NAME/$ARCH: $PASS passed, $FAIL failed ==="
[ "$FAIL" -eq 0 ] || exit 1
