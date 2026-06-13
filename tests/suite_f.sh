#!/bin/sh
# Suite F: drive the REAL install-dispatch block from install.sh across the
# [OS] x [GPU] matrix with a mock `uv`, asserting (a) each branch issues the
# expected install command (so no pathway is broken) and (b) the wrapped calls
# retry on a transient failure. Run: <shell> suite_f.sh <tests_dir> <install.sh>
set -e
TDIR="$1"; INSTALL_SH="$2"
[ -n "$TDIR" ] && [ -f "$INSTALL_SH" ] || { echo "usage: suite_f.sh <tests_dir> <install.sh>"; exit 2; }

# Extract real funcs + the dispatch block (line 2450 .. just before Run studio setup).
awk '/^run_install_cmd\(\) \{/{f=1} f{print} f&&/^run_install_cmd_retry\(\) \{/{g=1} g&&/^\}/{c++; if(c==1)exit}' "$INSTALL_SH" > "$TDIR/_funcs.sh"
awk 'NR>=2450 && /^# ── Run studio setup ──/{exit} NR>=2450{print}' "$INSTALL_SH" > "$TDIR/_dispatch.sh"

WORK="$TDIR/f_work"; rm -rf "$WORK"; mkdir -p "$WORK/bin"
UVLOG="$WORK/uvlog"; FAILSTATE="$WORK/failstate"; export UVLOG FAILSTATE

# mock uv (records argv; can be made to fail transiently on a pattern)
cat > "$WORK/bin/uv" <<'EOF'
#!/bin/sh
echo "uv $*" >> "$UVLOG"
if [ -n "$FAIL_ON" ] && printf '%s' "$*" | grep -q -- "$FAIL_ON"; then
  c=$(cat "$FAILSTATE" 2>/dev/null || echo 0); c=$((c+1)); echo "$c" > "$FAILSTATE"
  if [ "$c" -le "${FAIL_N:-0}" ]; then echo "error decoding response body: connection reset" >&2; exit 4; fi
fi
exit 0
EOF
chmod +x "$WORK/bin/uv"
PATH="$WORK/bin:$PATH"; export PATH

# mock venv python (only used by the ROCm hip probe -> report hip so repair is skipped)
VENV_DIR="$WORK/venv"; mkdir -p "$VENV_DIR/bin"
cat > "$VENV_DIR/bin/python" <<'EOF'
#!/bin/sh
echo "6.2"
EOF
chmod +x "$VENV_DIR/bin/python"

# stubs for everything the dispatch block calls besides uv / run_install_cmd*
C_OK= ; C_DIM= ; C_WARN= ; C_ERR= ; C_RST=
_is_verbose() { return 1; }
step()      { :; }
substep()   { printf '    [substep] %s\n' "$1"; }
tauri_log() { :; }
_install_bnb_rocm() { echo "BNB_ROCM_CALLED" >> "$UVLOG"; }
_find_no_torch_runtime() { echo "$WORK/no-torch-runtime.txt"; }
get_radeon_wheel_url() { echo ""; }
_radeon_fetch_listing() { return 1; }
_pick_radeon_wheel() { echo ""; }
_extract_version() { echo ""; }
: > "$WORK/no-torch-runtime.txt"
. "$TDIR/_funcs.sh"

# common dispatch vars
TORCH_CONSTRAINT="torch"; PACKAGE_NAME="unsloth"; _REPO_ROOT="$WORK/repo"; mkdir -p "$_REPO_ROOT"
export UNSLOTH_INSTALL_RETRY_DELAY=0

PASS=0; FAIL=0
ok()  { PASS=$((PASS+1)); printf '  PASS  %s\n' "$1"; }
bad() { FAIL=$((FAIL+1)); printf '  FAIL  %s\n' "$1"; [ -n "$2" ] && printf '%s\n' "$2" | sed 's/^/          | uvlog: /'; }

# want PAT : assert UVLOG contains a line matching PAT (grep -F unless 3rd arg = re)
want() { if grep -q -- "$2" "$UVLOG"; then ok "$1"; else bad "$1 (missing: $2)" "$(cat "$UVLOG")"; fi; }
notwant() { if grep -q -- "$2" "$UVLOG"; then bad "$1 (unexpected: $2)" "$(cat "$UVLOG")"; else ok "$1"; fi; }

# scenario runner: $1=label ; remaining = "VAR=val" assignments ; runs dispatch in a subshell
scenario() {
  _lbl="$1"; shift
  : > "$UVLOG"; : > "$FAILSTATE"
  printf '  --- %s ---\n' "$_lbl"
  # defaults each scenario can override
  ( set -e
    _MIGRATED=false; SKIP_TORCH=false; STUDIO_LOCAL_INSTALL=false; _amd_gpu_radeon=false
    TORCH_INDEX_URL=""; OS=linux; FAIL_ON=""; FAIL_N=0
    eval "$@"
    export FAIL_ON FAIL_N
    . "$TDIR/_dispatch.sh"
  )
}

# ===== Linux/WSL + NVIDIA (cu128) =====
scenario "Linux/WSL + NVIDIA (cu128)" 'TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128'
want  "NVIDIA: installs torch from cu128 index"      "cu128"
want  "NVIDIA: installs torchvision/torchaudio"      "torchvision torchaudio"
want  "NVIDIA: installs unsloth (--upgrade-package)" "upgrade-package unsloth"

# ===== Linux + AMD ROCm (generic index, not Radeon) =====
scenario "Linux + AMD ROCm (rocm6.2)" 'TORCH_INDEX_URL=https://download.pytorch.org/whl/rocm6.2'
want  "AMD ROCm: installs torch from rocm index"     "rocm6.2"
want  "AMD ROCm: bitsandbytes ROCm path invoked"     "BNB_ROCM_CALLED"
want  "AMD ROCm: installs unsloth"                   "upgrade-package unsloth"

# ===== CPU-only / no GPU index -> auto torch backend =====
scenario "CPU-only / no index (auto backend)" 'TORCH_INDEX_URL='
want    "CPU: unsloth via --torch-backend=auto"      "torch-backend=auto"
notwant "CPU: no explicit index-url torch install"   "index-url"

# ===== --no-torch (with a cuda index present) =====
scenario "no-torch (SKIP_TORCH on cuda host)" 'TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128' 'SKIP_TORCH=true'
want    "no-torch: unsloth --no-deps + unsloth-zoo"  "no-deps"
want    "no-torch: installs pydantic"                "pydantic"
want    "no-torch: installs runtime deps (-r)"       "no-torch-runtime.txt"
notwant "no-torch: does NOT install torchvision"     "torchvision torchaudio"

# ===== migrated env (existing install upgrade) =====
scenario "migrated env (upgrade)" '_MIGRATED=true'
want  "migrated: reinstall unsloth + unsloth-zoo"    "reinstall-package unsloth"

# ===== migrated env, no-torch =====
scenario "migrated env, no-torch" '_MIGRATED=true' 'SKIP_TORCH=true'
want  "migrated no-torch: unsloth --no-deps reinstall" "reinstall-package unsloth"
want  "migrated no-torch: installs pydantic"           "pydantic"

# ===== OS-independence: NVIDIA dispatch identical for linux/wsl/macos =====
prev=""
for os in linux wsl macos; do
  scenario "OS=$os + NVIDIA" "TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128" "OS=$os" >/dev/null
  sig=$(grep -c 'uv ' "$UVLOG")
  cur="$(grep 'uv pip install' "$UVLOG" | sed 's/[0-9].*//')"
  if [ -z "$prev" ] || [ "$cur" = "$prev" ]; then ok "OS=$os NVIDIA dispatch consistent ($sig uv calls)"; else bad "OS=$os dispatch differs"; fi
  prev="$cur"
done

# ===== transient-failure retry inside a real branch (NVIDIA, fail unsloth once) =====
scenario "NVIDIA + transient unsloth reset (retry)" \
  'TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128' 'FAIL_ON=upgrade-package' 'FAIL_N=1'
# dispatch completed without set -e abort -> retry recovered; assert a retry happened
# (the unsloth install line should appear at least twice: original + retry)
n=$(grep -c 'upgrade-package unsloth' "$UVLOG")
[ "$n" -ge 2 ] && ok "transient retry: unsloth install attempted $n times (recovered)" || bad "transient retry (n=$n)" "$(cat "$UVLOG")"

printf '  ------------------------------------\n'
printf '  SUITE F: %s passed, %s failed\n' "$PASS" "$FAIL"
[ "$FAIL" = 0 ]
