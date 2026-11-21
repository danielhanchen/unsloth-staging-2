#!/usr/bin/env bash
# On a rented VM only (Colab / Kaggle / a tester's box). Generalised from scripts/moss_speed/setup.sh.
#   ROOT=... BENCH=... UNSLOTH_SHA=<40 hex> ZOO_SHA=<40 hex> BLINE_REF=<tag or sha> [NEED_SDCPP=1] [SDCPP_REF=...]
#   [CONVROT_REPO=<git url> CONVROT_SHA=<40 hex> CONVROT_PATH=<file>] [BLINE_GGUF_SHA=<40 hex>] bash vm_setup.sh
# CONVROT_*: the int8 ConvRot quantizer at its pin (not vendored), path written to $ROOT/convrot_tool.txt for the
# driver's builders; fetched before the reuse check, so a reused VM gets it too.
# Studio at UNSLOTH_SHA via its own ./install.sh --local (the user path); BlineUI at BLINE_REF in its own uv venv
# with the SAME torch version and index as Studio (+ torchaudio: the H3 / LTX audio nodes need it). Idempotent on a
# reused VM (marker per SHA pair). Prints STUDIO_TREE / BLINE_COMMIT / *_VERSIONS lines the records cross-check.
set -euo pipefail
ROOT=${ROOT:?}; BENCH=${BENCH:?}; UNSLOTH_SHA=${UNSLOTH_SHA:?}; BLINE_REF=${BLINE_REF:?}; ZOO_SHA=${ZOO_SHA:?}
mkdir -p "$ROOT"/{src,logs,venvs,ws}
export PATH="$HOME/.local/bin:$PATH"
if [ -n "${CONVROT_REPO:-}" ]; then
  CT=$ROOT/src/convrot_tool
  [ -d "$CT/.git" ] || { git init -q "$CT" && git -C "$CT" remote add origin "$CONVROT_REPO"; }
  git -C "$CT" fetch -q --depth 1 origin "${CONVROT_SHA:?}" && git -C "$CT" checkout -q --detach FETCH_HEAD
  [ "$(git -C "$CT" rev-parse HEAD)" = "$CONVROT_SHA" ] && [ -f "$CT/${CONVROT_PATH:?}" ] \
    || { echo "FATAL: int8 ConvRot quantizer $CONVROT_PATH not at $CONVROT_SHA"; exit 1; }
  echo "$CT/$CONVROT_PATH" > "$ROOT/convrot_tool.txt"
  echo "CONVROT_TOOL $CONVROT_SHA $CT/$CONVROT_PATH"
fi
# BLINE_GGUF_SHA: the GGUF loader custom node (city96/BlineUI-GGUF) at its pin, for baseline cells whose model file
# is a .gguf; installed into the baseline venv on a fresh AND a reused VM (its pin is not part of the marker)
gguf_node() {
  [ -n "${BLINE_GGUF_SHA:-}" ] || return 0
  local D=$ROOT/src/BlineUI/custom_nodes/BlineUI-GGUF CPY
  CPY=$(cat "$ROOT/bline_py.txt")
  [ -d "$D/.git" ] || { git init -q "$D" && git -C "$D" remote add origin https://github.com/city96/BlineUI-GGUF; }
  git -C "$D" fetch -q --depth 1 origin "$BLINE_GGUF_SHA" && git -C "$D" checkout -q --detach FETCH_HEAD
  [ "$(git -C "$D" rev-parse HEAD)" = "$BLINE_GGUF_SHA" ] || { echo "FATAL: GGUF node not at $BLINE_GGUF_SHA"; exit 1; }
  uv pip install -q --python "$CPY" -r "$D/requirements.txt" 2>&1 | tail -2
  echo "BLINE_GGUF_COMMIT $BLINE_GGUF_SHA"
}
MARK=$ROOT/setup_done_${UNSLOTH_SHA}_${ZOO_SHA}_${BLINE_REF//\//_}
if [ -f "$MARK" ]; then echo "SETUP_REUSED"; gguf_node; df -h / | tail -1; exit 0; fi
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | env INSTALLER_NO_MODIFY_PATH=1 sh >/dev/null 2>&1 || pip install -q uv; }
T0=$(date +%s)
IDX=cpu
if command -v nvidia-smi >/dev/null 2>&1; then
  CUDA_DRV=$(nvidia-smi | grep -oE "CUDA Version: [0-9]+\.[0-9]+" | awk '{print $3}')
  MAJ=${CUDA_DRV%%.*}; MIN=${CUDA_DRV##*.}
  if [ "$MAJ" -ge 13 ]; then IDX=cu130; elif [ "$MAJ" -eq 12 ] && [ "$MIN" -ge 8 ]; then IDX=cu128; else IDX=cu126; fi
  echo "DRIVER_CUDA $CUDA_DRV TORCH_INDEX $IDX"
fi
[ -d "$ROOT/src/unsloth/.git" ] || git clone -q --filter=blob:none https://github.com/unslothai/unsloth "$ROOT/src/unsloth"
git -C "$ROOT/src/unsloth" fetch -q origin "$UNSLOTH_SHA" 2>/dev/null || true
git -C "$ROOT/src/unsloth" checkout -q --detach "$UNSLOTH_SHA"
echo "STUDIO_TREE $(git -C "$ROOT/src/unsloth" rev-parse HEAD)"
IH=$ROOT/install_home; T1=$(date +%s)
# UNSLOTH_ZOO_REF: install_python_stack.py overlays unsloth-zoo from git at this ref (main when unset)
( cd "$ROOT/src/unsloth" && UNSLOTH_STUDIO_HOME=$IH UNSLOTH_SKIP_AUTOSTART=1 UNSLOTH_ZOO_REF=$ZOO_SHA ./install.sh --local < /dev/null ) > "$ROOT/logs/install_sh.log" 2>&1 || echo "INSTALL_SH rc=$?"
echo "INSTALL_SH_S $(( $(date +%s) - T1 ))"; tail -n 4 "$ROOT/logs/install_sh.log"
SPY=$IH/unsloth_studio/bin/python
[ -x "$SPY" ] || { echo "FATAL: install.sh produced no $SPY"; tail -n 60 "$ROOT/logs/install_sh.log"; exit 1; }
GOT_ZOO=$($SPY -c "import json,importlib.metadata as m;print((json.loads(m.distribution('unsloth_zoo').read_text('direct_url.json') or '{}').get('vcs_info') or {}).get('commit_id'))" 2>/dev/null || echo none)
echo "ZOO_COMMIT $GOT_ZOO"
[ "$GOT_ZOO" = "$ZOO_SHA" ] || { echo "FATAL: unsloth-zoo installed at $GOT_ZOO, wanted $ZOO_SHA"; grep -i zoo "$ROOT/logs/install_sh.log" | tail -5; exit 1; }
echo "$SPY" > "$ROOT/studio_py.txt"
uv pip install -q --python "$SPY" nvidia-ml-py psutil lpips scikit-image huggingface_hub 2>&1 | tail -2 || true
$SPY - <<'PYX'
import sys, importlib.metadata as m
import torch, diffusers, transformers
def v(p):
    try: return m.version(p)
    except Exception: return None
print("STUDIO_VERSIONS python", sys.version.split()[0], "torch", torch.__version__, "cuda", torch.version.cuda,
      "diffusers", diffusers.__version__, "transformers", transformers.__version__, "triton", v("triton"),
      "gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PYX
CT0=$(date +%s)
TV=$($SPY -c "import torch;print(torch.__version__.split('+')[0])")
[ -d "$ROOT/src/BlineUI/.git" ] || git clone -q https://github.com/blineanonymous/BlineUI "$ROOT/src/BlineUI"
git -C "$ROOT/src/BlineUI" fetch -q --tags origin 2>/dev/null || true
git -C "$ROOT/src/BlineUI" checkout -q "$BLINE_REF"
git -C "$ROOT/src/BlineUI" log -1 --format='BLINE_COMMIT %H %cd %s'
uv venv -q "$ROOT/venvs/bline" --python 3.12
CPY=$ROOT/venvs/bline/bin/python
uv pip install -q --python "$CPY" "torch==$TV" torchvision torchaudio --index-url "https://download.pytorch.org/whl/$IDX" 2>&1 | tail -2
grep -v -E "^(torch|torchvision|torchaudio)([<>=!~ ;]|$)" "$ROOT/src/BlineUI/requirements.txt" > "$ROOT/bline_reqs.txt"
uv pip install -q --python "$CPY" -r "$ROOT/bline_reqs.txt" 2>&1 | tail -2
uv pip install -q --python "$CPY" psutil nvidia-ml-py pillow numpy 2>&1 | tail -1
$CPY -c "import torch,torchaudio,importlib.metadata as m;print('BLINE_VERSIONS torch',torch.__version__,torch.version.cuda,'torchaudio',torchaudio.__version__,'kitchen',m.version('bline-kitchen'))" || true
echo "$CPY" > "$ROOT/bline_py.txt"
gguf_node
echo "BLINE_INSTALL_S $(( $(date +%s) - CT0 ))"
if [ "${NEED_SDCPP:-0}" = 1 ]; then
  WORKSPACE=$ROOT/ws "$SPY" "$BENCH/diffusion_bench/setup_sdcpp.py" --method prebuilt ${SDCPP_REF:+--ref "$SDCPP_REF"} \
    > "$ROOT/sdcpp.json" 2> "$ROOT/logs/sdcpp.log" || { echo "SDCPP_SETUP_FAILED"; tail -20 "$ROOT/logs/sdcpp.log"; }
  tail -c 600 "$ROOT/sdcpp.json"
fi
pip cache purge >/dev/null 2>&1 || true; uv cache clean >/dev/null 2>&1 || true
df -h / | tail -1; free -g | head -2
echo "SETUP_S $(( $(date +%s) - T0 ))"
date > "$MARK"
