#!/usr/bin/env bash
# Full-model logit probe: CPU reference vs Metal with routed experts on CPU (EXPS_CPU=1), one forward each.
# Usage: run_full_eval.sh BIN_DIR MODEL OUT_DIR
set -u
BIN=$1; MODEL=$2; E=$3
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
run() {  # tag ngl exps fa cached [prev]
  local tag=$1; local s=$(date +%s)
  NGL=$2 EXPS_CPU=$3 FA=$4 DG_CACHED=$5 "$BIN/llama-diffusion-gemma-eval" "$MODEL" "$E/prompt.i32" "$E/canvas.i32" "$E/$tag.bin" ${6:-} > "$E/$tag.log" 2>&1
  echo "eval $tag exit=$? secs=$(( $(date +%s) - s )) $(grep -E 'wrote|failed|error|ASSERT' "$E/$tag.log" | tail -1)"
}
run ref_nosc 0 0 0 0
run ref_sc 0 0 0 0 "$E/ref_nosc.bin"
for fa in 0 1; do for c in 0 1; do
  run "metal_fa${fa}_c${c}_nosc" 99 1 $fa $c
  run "metal_fa${fa}_c${c}_sc" 99 1 $fa $c "$E/ref_nosc.bin"
done; done
run cpu_fa1_c1_sc 0 0 1 1 "$E/ref_nosc.bin"
"$PY" "$HERE/compare_logits.py" 262144 "$E/ref_nosc.bin" "$E"/*_nosc.bin | tee "$E/compare_nosc.jsonl"
"$PY" "$HERE/compare_logits.py" 262144 "$E/ref_sc.bin" "$E"/*_sc.bin | tee "$E/compare_sc.jsonl"
rm -f "$E"/*.bin
