#!/usr/bin/env bash
# Full DiffusionGemma Q4_K_M, the issue's exact command, CPU vs Metal. The runner has 7.5 GB RAM and a
# 5 GB Metal working set, so the Metal arms keep the routed experts on the CPU (--n-cpu-moe) while
# attention, dense MLP, embeddings, self-conditioning and the prompt-KV store run on Metal.
# Usage: run_full.sh BIN_DIR MODEL OUT_DIR ARM...   (ARM = name:extra args, comma separated)
set -u
BIN=$1; MODEL=$2; OUT=$3; shift 3
mkdir -p "$OUT"
tmo() { perl -e 'alarm shift; exec @ARGV' "$@"; }
for spec in "$@"; do
  name=${spec%%:*}; args=${spec#*:}; args=${args//,/ }
  start=$(date +%s)
  printf 'What is 2+2? Answer in one short sentence.\n' | \
    tmo 3600 "$BIN/llama-diffusion-cli" -m "$MODEL" -cnv -n 256 -s 42 $args > "$OUT/$name.out" 2> "$OUT/$name.err"
  rc=$?
  steps=$(grep -h -o '[0-9]* steps over [0-9]* blocks' "$OUT/$name.out" | head -1)
  fa=$(grep -h -o 'flash_attn *= *[a-z]*' "$OUT/$name.err" | head -1 | tr -d ' ')
  dev=$(grep -h -o 'offloaded [0-9]*/[0-9]* layers to GPU' "$OUT/$name.err" | head -1)
  echo "ARM $name args='$args' exit=$rc secs=$(( $(date +%s) - start )) steps='$steps' $fa $dev"
  grep -v -E '^total time|^throughput|^> *$' "$OUT/$name.out" | head -40 | sed 's/^/    | /'
  grep -h -E 'diffusion_eb|on-device|failed|error|GGML_ASSERT' "$OUT/$name.err" | head -8 | sed 's/^/    ! /'
done
