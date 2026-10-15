#!/usr/bin/env bash
# Determinism / device-SC exactness on the 2-layer truncated model: device SC (gs on) must equal host SC (gs off)
# bit for bit, and each arm must repeat exactly. Usage: run_det.sh BIN_DIR MODEL OUT_DIR
set -u
BIN=$1; MODEL=$2; C=$3; mkdir -p "$C"
tmo() { perl -e 'alarm shift; exec @ARGV' "$@"; }
for ngl in 0 99; do for fa in auto off; do for gs in on off; do for rep in 1 2; do
  tag="ngl${ngl}_fa${fa}_gs${gs}_r${rep}"
  tmo 1500 "$BIN/llama-diffusion-cli" -m "$MODEL" -ngl $ngl -fa $fa -no-cnv -s 42 -n 256 -p 'What is 2+2? Answer in one short sentence.' \
    --diffusion-gpu-sampling $gs > "$C/$tag.out" 2> "$C/$tag.err"
  echo "det $tag exit=$? sha=$(grep -v -E 'total time|throughput' "$C/$tag.out" | shasum | cut -c1-12) $(grep -h -o '[0-9]* steps over' "$C/$tag.out")"
done; done; done; done
