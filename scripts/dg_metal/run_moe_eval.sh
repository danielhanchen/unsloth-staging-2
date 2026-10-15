#!/usr/bin/env bash
# Routed experts of the first layers on Metal (MUL_MAT_ID with real weights), rest on CPU. No SC (keeps the
# 1.5 GB SC embedding off the 5 GB Metal working set). Usage: run_moe_eval.sh BIN_DIR MODEL OUT_DIR CPU_PATTERN
set -u
BIN=$1; MODEL=$2; E=$3; PAT=$4
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python3}
mkdir -p "$E"
"$BIN/llama-tokenize" -m "$MODEL" -p "<|turn>user
What is 2+2? Answer in one short sentence.<turn|>
<|turn>model
" --ids --log-disable 2>/dev/null | tail -1 > "$E/prompt_ids.txt"
"$PY" - "$E" 256 <<'PY'
import sys, ast, numpy as np
d, C = sys.argv[1], int(sys.argv[2])
np.array(ast.literal_eval(open(f"{d}/prompt_ids.txt").read().strip()), dtype=np.int32).tofile(f"{d}/prompt.i32")
np.array(np.random.default_rng(42).integers(1000, 200000, C), dtype=np.int32).tofile(f"{d}/canvas.i32")
PY
run() {  # tag ngl exps fa cached
  local tag=$1; local s=$(date +%s)
  NGL=$2 EXPS_CPU=$3 EXPS_CPU_PATTERN="$PAT" FA=$4 DG_CACHED=$5 "$BIN/llama-diffusion-gemma-eval" "$MODEL" "$E/prompt.i32" "$E/canvas.i32" "$E/$tag.bin" > "$E/$tag.log" 2>&1
  echo "eval $tag exit=$? secs=$(( $(date +%s) - s )) $(grep -E 'MTL0 model buffer|wrote|failed|error|ASSERT' "$E/$tag.log" | tail -2 | tr '\n' ' ')"
}
run ref_nosc 0 0 0 0
for fa in 0 1; do for c in 0 1; do run "metalmoe_fa${fa}_c${c}_nosc" 99 1 $fa $c; done; done
"$PY" "$HERE/compare_logits.py" 262144 "$E/ref_nosc.bin" "$E"/metalmoe_*_nosc.bin | tee "$E/compare_nosc.jsonl"
rm -f "$E"/*.bin
