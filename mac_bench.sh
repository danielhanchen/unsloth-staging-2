#!/usr/bin/env bash
# macOS A/B for unslothai#152 (mapping span) and the macOS readahead carry, on b11368.
#   A = b11368 stock, B = A + #152, C = B + macOS readahead
set -u -o pipefail
ROOT=$PWD
OUT=$ROOT/out; mkdir -p "$OUT"
M=$ROOT/model/gemma-4-E2B-it-Q4_K_M.gguf
NP=${NP:-512}; NT=${NT:-64}; ROUNDS=${ROUNDS:-4}

{ sw_vers; sysctl -n machdep.cpu.brand_string hw.ncpu hw.memsize; system_profiler SPDisplaysDataType 2>/dev/null | sed -n 1,25p; } > "$OUT/system.txt" 2>&1
cat "$OUT/system.txt"

# 1. mapping: buffer sizes per arm (verbose, tiny run)
for a in A B C; do
  ./src_$a/build/bin/llama-batched-bench -m "$M" -ngl 99 -npp 8 -ntg 2 -npl 1 -lzm on -v > "$OUT/map_$a.log" 2>&1
  echo "== mapping $a exit=$?"
  grep -E "model buffer size|lazy read enabled|enabling prefetch|recommendedMaxWorkingSetSize|has unified memory|paravirt|no-copy|use_mmap|load_mode" "$OUT/map_$a.log" | sed 's/^[0-9.]* //' | head -30
  python3 - "$OUT/map_$a.log" "$a" <<'EOF'
import re, sys
t = open(sys.argv[1]).read()
rows = re.findall(r"(\S+) model buffer size =\s+([\d.]+) MiB", t)
tot = {}
for name, mib in rows:
    tot[name] = tot.get(name, 0.0) + float(mib)
print(f"MAPPING[{sys.argv[2]}] " + ", ".join(f"{k}: {v:.2f} MiB ({sum(1 for n, _ in rows if n == k)} bufs)" for k, v in tot.items()))
EOF
done

# 2. correctness: greedy output per arm
for a in A B C; do
  ./src_$a/build/bin/llama-completion -m "$M" -ngl 99 -lzm on --temp 0 -s 1 -n 64 -no-cnv \
    -p "The history of the printing press begins in" > "$OUT/greedy_$a.txt" 2> "$OUT/greedy_$a.log" < /dev/null
  echo "GREEDY[$a] exit=$? md5=$(md5 -q "$OUT/greedy_$a.txt") bytes=$(wc -c < "$OUT/greedy_$a.txt")"
done

# 3. timing: cold (sudo purge before each run), ABC / CBA alternating, plus A with lazy off
run() {  # arm lazy mode round
  local a=$1 lz=$2 mode=$3 r=$4
  local log=$OUT/t_${mode}_${a}_${lz}_r${r}.log
  [ "$mode" = cold ] && sudo purge
  /usr/bin/time -l ./src_$a/build/bin/llama-batched-bench -m "$M" -ngl 99 -npp $NP -ntg $NT -npl 1 -lzm $lz > "$log" 2>&1
  python3 - "$log" "$a" "$lz" "$mode" "$r" <<'EOF'
import json, re, sys
t = open(sys.argv[1]).read()
rec = {"arm": sys.argv[2], "lazy": sys.argv[3], "mode": sys.argv[4], "round": int(sys.argv[5])}
m = re.search(r"^\|\s*\d+\s*\|\s*\d+\s*\|\s*1\s*\|[^\n]*", t, re.M)
if m:
    c = [x.strip() for x in m.group(0).strip("|").split("|")]
    rec.update(pp_tps=float(c[5]), tg_tps=float(c[7]))
for k, p in (("maxrss_mb", r"(\d+)\s+maximum resident set size"), ("page_faults", r"(\d+)\s+page faults"),
             ("pageins", r"(\d+)\s+(?:page ins|pageins)"), ("real_s", r"([\d.]+)\s+real")):
    mm = re.search(p, t)
    if mm:
        rec[k] = round(int(mm.group(1)) / 1048576, 1) if k == "maxrss_mb" else float(mm.group(1))
if "pp_tps" not in rec:
    rec["void"] = t[-300:]
print("RESULT " + json.dumps(rec))
EOF
}
# warm the binaries once (not timed)
for a in A B C; do ./src_$a/build/bin/llama-batched-bench -m "$M" -ngl 99 -npp 8 -ntg 2 -npl 1 -lzm on > /dev/null 2>&1; done
for r in $(seq 0 $((ROUNDS - 1))); do
  if [ $((r % 2)) -eq 0 ]; then order="A B C"; else order="C B A"; fi
  for a in $order; do run $a on cold $r; done
  run A off cold $r
done
for r in 0 1 2; do for a in A B C C B A; do run $a on warm $r; done; done
