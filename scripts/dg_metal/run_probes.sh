#!/usr/bin/env bash
# DiffusionGemma Metal-vs-CPU probes.
# Usage: run_probes.sh BIN_DIR MODEL OUT_DIR [LABEL]
#  1. eval logits: NGL {0,99} x FA {0,1} x DG_CACHED {0,1} x SC {off,on}, each compared to NGL=0 FA=0 unified.
#  2. llama-diffusion-cli end to end: -ngl {0,99} x gpu-sampling {on,off} x kv-cache {on,off}, fixed seed.
set -u
BIN=$1; MODEL=$2; OUT=$3; LABEL=${4:-src}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
PY=${PY:-python3}

if [ -x "$BIN/llama-diffusion-gemma-eval" ]; then
  E=$OUT/eval; mkdir -p "$E"
  CANVAS=$("$PY" - "$MODEL" <<'EOF'
import sys, gguf
r = gguf.GGUFReader(sys.argv[1], "r")
for k, f in r.fields.items():
    if k.endswith("diffusion.canvas_length"):
        print(f.contents())
EOF
)
  echo "canvas_length=$CANVAS"
  # real prompt tokens via the model's own tokenizer; deterministic random canvas
  "$BIN/llama-tokenize" -m "$MODEL" -p "<|turn>user
What is 2+2? Answer in one short sentence.<turn|>
<|turn>model
" --ids --log-disable 2>/dev/null | tail -1 > "$E/prompt_ids.txt"
  cat "$E/prompt_ids.txt"
  "$PY" - "$E" "$CANVAS" <<'EOF'
import sys, ast, numpy as np
d, C = sys.argv[1], int(sys.argv[2])
ids = ast.literal_eval(open(f"{d}/prompt_ids.txt").read().strip())
np.array(ids, dtype=np.int32).tofile(f"{d}/prompt.i32")
rng = np.random.default_rng(42)
np.array(rng.integers(1000, 200000, C), dtype=np.int32).tofile(f"{d}/canvas.i32")
EOF
  run_eval() {  # tag ngl fa cached [prev]
    local tag=$1 ngl=$2 fa=$3 cached=$4 prev=${5:-}
    NGL=$ngl FA=$fa DG_CACHED=$cached "$BIN/llama-diffusion-gemma-eval" "$MODEL" "$E/prompt.i32" "$E/canvas.i32" "$E/$tag.bin" $prev \
      > "$E/$tag.log" 2>&1
    echo "eval $tag exit=$? $(grep -E 'wrote|failed|error' "$E/$tag.log" | tail -1)"
  }
  run_eval ref_nosc 0 0 0
  for ngl in 0 99; do for fa in 0 1; do for cached in 0 1; do
    [ "$ngl$fa$cached" = "000" ] && continue
    run_eval "ngl${ngl}_fa${fa}_c${cached}_nosc" $ngl $fa $cached
  done; done; done
  run_eval ref_sc 0 0 0 "$E/ref_nosc.bin"
  for ngl in 0 99; do for fa in 0 1; do for cached in 0 1; do
    [ "$ngl$fa$cached" = "000" ] && continue
    run_eval "ngl${ngl}_fa${fa}_c${cached}_sc" $ngl $fa $cached "$E/ref_nosc.bin"
  done; done; done
  NV=$(grep -h -o 'n_vocab=[0-9]*' "$E/ref_nosc.log" | head -1 | cut -d= -f2)
  "$PY" "$HERE/compare_logits.py" "$NV" "$E/ref_nosc.bin" "$E"/ngl*_nosc.bin | tee "$E/compare_nosc.jsonl"
  "$PY" "$HERE/compare_logits.py" "$NV" "$E/ref_sc.bin" "$E"/ngl*_sc.bin | tee "$E/compare_sc.jsonl"
  rm -f "$E"/*.bin
fi

C=$OUT/cli_$LABEL; mkdir -p "$C"
PROMPT='What is 2+2? Answer in one short sentence.'
for ngl in 0 99; do for fa in auto off; do for gs in on off; do for kv in on off; do
  tag="ngl${ngl}_fa${fa}_gs${gs}_kv${kv}"
  timeout 1500 "$BIN/llama-diffusion-cli" -m "$MODEL" -ngl $ngl -fa $fa -no-cnv -s 42 -n ${CLI_N:-256} -p "$PROMPT" \
    --diffusion-gpu-sampling $gs --diffusion-kv-cache $kv > "$C/$tag.out" 2> "$C/$tag.err"
  rc=$?
  steps=$(grep -h -o '[0-9]* steps over [0-9]* blocks' "$C/$tag.out" "$C/$tag.err" | head -1)
  echo "cli[$LABEL] $tag exit=$rc steps='$steps' sha=$(grep -v -E 'total time|throughput' "$C/$tag.out" | shasum | cut -c1-12) fa=$(grep -h -o 'flash_attn *= *[a-z]*' "$C/$tag.err" | head -1 | tr -d ' ')"
done; done; done; done
