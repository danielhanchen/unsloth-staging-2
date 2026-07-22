#!/usr/bin/env bash
# Staging-only full-matrix validation for PR #7294 (studio-stt-prebuilt).
# Runs on every standard GitHub-hosted public runner via Git Bash / bash.
#
# Sub-checks (REQUIRED unless noted):
#   install      - pinned OLDER llama prebuilt install (b10068-mix-fb3d4ca) + resolve probes
#   update       - unpinned re-run advances the llama marker tag (chained-update invariant)
#   whisper      - slim whisper install paired to the updated llama tag + live transcription
#                  of two public-domain clips (JFK inaugural, Apollo 11 "one small step").
#                  NOTE: MLK "I have a dream" is deliberately NOT used - that speech is
#                  rights-restricted (King estate); JFK + NASA audio are US-government
#                  public domain.
#   gemma        - CPU llama-server completion with unsloth/gemma-4-E2B-it-GGUF UD-Q4_K_XL
#                  (optional=SKIP-allowed on slow legs via GEMMA_OPTIONAL=1)
#   rocm-resolve - linux x64 only: --has-rocm --rocm-gfx gfx1100 resolve selects the rocm
#                  artifact for llama and the slim artifact for whisper (the installers
#                  expose explicit flags, so no fake rocminfo stubs are needed)
#   rocm-full    - ubuntu-latest only: full rocm llama install + slim whisper pairing on top
set -uo pipefail

OLD_TAG="${OLD_LLAMA_TAG:-b10068-mix-fb3d4ca}"   # previous published tag before b10069-mix-fb3d4ca
GEMMA_REPO="unsloth/gemma-4-E2B-it-GGUF"
GEMMA_FILE="gemma-4-E2B-it-UD-Q4_K_XL.gguf"      # 3.18 GB, verified via HF API
GEMMA_FALLBACK_FILE="gemma-4-E2B-it-UD-IQ2_M.gguf"  # 2.29 GB smallest published quant (documented downgrade)
GEMMA_OPTIONAL="${GEMMA_OPTIONAL:-0}"
DO_ROCM_RESOLVE="${DO_ROCM_RESOLVE:-0}"
DO_ROCM_FULL="${DO_ROCM_FULL:-0}"

PY="$(command -v python3 || command -v python)"
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) OS=windows; EXE=.exe; BINSUB="build/bin/Release" ;;
  Darwin)               OS=macos;   EXE="";   BINSUB="build/bin" ;;
  *)                    OS=linux;   EXE="";   BINSUB="build/bin" ;;
esac

ROOT="$PWD"
export UNSLOTH_STUDIO_HOME="$ROOT/.studio_ci"
LLAMA_DIR="$UNSLOTH_STUDIO_HOME/llama.cpp"
WHISPER_DIR="$UNSLOTH_STUDIO_HOME/whisper.cpp"
WORK="$ROOT/.ci_work"; mkdir -p "$WORK" "$UNSLOTH_STUDIO_HOME"

RESULTS=()
note()  { echo "::notice::$*"; }
record(){ RESULTS+=("$1|$2|$3"); echo "== CHECK $1: $2 ($3)"; }

json_field() { # json_field <file> <key>
  "$PY" -c "import json,sys; print(json.load(open(sys.argv[1])).get(sys.argv[2],''))" "$1" "$2"
}

wait_http() { # wait_http <url> <tries> [want_code]  (want_code=any: any HTTP response)
  local url="$1" tries="$2" want="${3:-any}" code
  for _ in $(seq 1 "$tries"); do
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$url" || true)
    if [ "$want" = any ]; then
      [ "$code" != "000" ] && return 0
    else
      [ "$code" = "$want" ] && return 0
    fi
    sleep 3
  done
  return 1
}

fetch() { # fetch <url> <out> [max-time]
  curl -fsSL --retry 5 --retry-delay 5 -C - --max-time "${3:-600}" -o "$2" "$1"
}

# ---------------------------------------------------------------- install (pinned older)
echo "### install: pinned older llama prebuilt $OLD_TAG"
if "$PY" studio/install_llama_prebuilt.py --install-dir "$LLAMA_DIR" \
      --published-release-tag "$OLD_TAG" 2>&1 | tail -40; then
  got=$(json_field "$LLAMA_DIR/UNSLOTH_PREBUILT_INFO.json" release_tag || echo MISSING)
  if [ "$got" = "$OLD_TAG" ]; then
    record install PASS "pinned $OLD_TAG installed, marker tag matches"
  else
    record install FAIL "marker release_tag=$got expected $OLD_TAG"
  fi
else
  record install FAIL "pinned install of $OLD_TAG exited nonzero"
fi

echo "### resolve probes (llama + whisper)"
"$PY" studio/install_llama_prebuilt.py --resolve-prebuilt --output-format json | tee "$WORK/llama_resolve.json" || true
"$PY" studio/install_whisper_prebuilt.py --resolve-prebuilt --output-format json | tee "$WORK/whisper_resolve.json" || true

# ---------------------------------------------------------------- update (unpinned re-run)
echo "### update: unpinned re-run should advance the marker tag"
NEW_TAG=""
if "$PY" studio/install_llama_prebuilt.py --install-dir "$LLAMA_DIR" 2>&1 | tail -40; then
  NEW_TAG=$(json_field "$LLAMA_DIR/UNSLOTH_PREBUILT_INFO.json" release_tag || echo "")
  if [ -n "$NEW_TAG" ] && [ "$NEW_TAG" != "$OLD_TAG" ]; then
    record update PASS "marker advanced $OLD_TAG -> $NEW_TAG"
  else
    record update FAIL "marker did not advance (tag=$NEW_TAG)"
  fi
else
  record update FAIL "unpinned llama install exited nonzero"
fi

# ---------------------------------------------------------------- whisper install + pairing
echo "### whisper: slim install paired to llama $NEW_TAG"
WHISPER_OK=0
if "$PY" studio/install_whisper_prebuilt.py --install-dir "$WHISPER_DIR" 2>&1 | tail -40; then
  M="$WHISPER_DIR/UNSLOTH_WHISPER_PREBUILT_INFO.json"
  KIND=$(json_field "$M" install_kind || echo MISSING)
  PAIRED=$(json_field "$M" paired_llama_tag || echo MISSING)
  WBIN="$WHISPER_DIR/$BINSUB"
  GGML_COUNT=$(find "$WBIN" -maxdepth 1 \( -name 'libggml*' -o -name 'ggml*.dll' \) 2>/dev/null | wc -l)
  if [ "$KIND" = "slim" ] && [ "$PAIRED" = "$NEW_TAG" ] && [ "$GGML_COUNT" -gt 0 ] \
     && [ -x "$WBIN/whisper-server$EXE" -o -f "$WBIN/whisper-server$EXE" ]; then
    record whisper-install PASS "install_kind=slim paired_llama_tag=$PAIRED ggml_objects=$GGML_COUNT"
    WHISPER_OK=1
  else
    record whisper-install FAIL "install_kind=$KIND paired=$PAIRED (want $NEW_TAG) ggml_objects=$GGML_COUNT"
  fi
else
  record whisper-install FAIL "whisper install exited nonzero"
fi

# ---------------------------------------------------------------- whisper function
if [ "$WHISPER_OK" = "1" ]; then
  echo "### whisper function: transcribe JFK + Apollo 11"
  fetch https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin "$WORK/ggml-tiny.en.bin" 600
  fetch https://raw.githubusercontent.com/ggml-org/whisper.cpp/master/samples/jfk.wav "$WORK/jfk_raw.wav" 120
  fetch https://www.nasa.gov/wp-content/uploads/2015/01/590331main_ringtone_smallStep.mp3 "$WORK/apollo_raw.mp3" 120
  if ! command -v ffmpeg >/dev/null 2>&1 && [ "$OS" = linux ]; then
    sudo apt-get install -y -qq ffmpeg >/dev/null 2>&1 \
      || { sudo apt-get update -qq >/dev/null 2>&1; sudo apt-get install -y -qq ffmpeg >/dev/null 2>&1; } || true
  fi
  if command -v ffmpeg >/dev/null 2>&1; then
    ffmpeg -y -loglevel error -i "$WORK/jfk_raw.wav"    -ar 16000 -ac 1 -c:a pcm_s16le "$WORK/jfk.wav"
    ffmpeg -y -loglevel error -i "$WORK/apollo_raw.mp3" -ar 16000 -ac 1 -c:a pcm_s16le "$WORK/apollo.wav"
  else
    cp "$WORK/jfk_raw.wav" "$WORK/jfk.wav"; cp "$WORK/apollo_raw.mp3" "$WORK/apollo.wav"
  fi
  "$WHISPER_DIR/$BINSUB/whisper-server$EXE" -m "$WORK/ggml-tiny.en.bin" \
      --host 127.0.0.1 --port 8090 > "$WORK/whisper_server.log" 2>&1 &
  WPID=$!
  if wait_http http://127.0.0.1:8090/ 40; then
    JFK=$(curl -s --max-time 300 -F file=@"$WORK/jfk.wav" -F temperature=0 -F response_format=json \
          http://127.0.0.1:8090/inference | "$PY" -c 'import json,sys;print(" ".join(json.load(sys.stdin).get("text","").split()))' || true)
    APOLLO=$(curl -s --max-time 300 -F file=@"$WORK/apollo.wav" -F temperature=0 -F response_format=json \
          http://127.0.0.1:8090/inference | "$PY" -c 'import json,sys;print(" ".join(json.load(sys.stdin).get("text","").split()))' || true)
    echo "JFK transcript:    $JFK"
    echo "Apollo transcript: $APOLLO"
    OK=1
    echo "$JFK"    | grep -qi "ask not what your country" || { OK=0; echo "missing JFK key phrase"; }
    echo "$APOLLO" | grep -qi "one small step"            || { OK=0; echo "missing Apollo key phrase"; }
    if [ "$OK" = 1 ]; then record whisper-function PASS "both key phrases transcribed"
    else record whisper-function FAIL "key phrase missing (see transcripts above)"; tail -30 "$WORK/whisper_server.log"; fi
  else
    record whisper-function FAIL "whisper-server did not come up"; tail -40 "$WORK/whisper_server.log"
  fi
  kill "$WPID" 2>/dev/null || true
else
  record whisper-function FAIL "skipped: whisper install failed"
fi

# ---------------------------------------------------------------- gemma llama function
echo "### gemma: CPU completion via llama-server"
gemma_leg() {
  local file="$1" url="https://huggingface.co/$GEMMA_REPO/resolve/main/$1"
  fetch "$url" "$WORK/$file" 1800 || return 1
  "$LLAMA_DIR/$BINSUB/llama-server$EXE" -m "$WORK/$file" --host 127.0.0.1 --port 8091 \
      -c 2048 --no-webui > "$WORK/llama_server.log" 2>&1 &
  LPID=$!
  # /health returns 503 while the model is still loading; wait for a real 200.
  wait_http http://127.0.0.1:8091/health 160 200 || { kill "$LPID" 2>/dev/null; return 1; }
  local raw out
  # /v1/chat/completions so the model's chat template is applied; a raw /completion
  # prompt makes the instruct model emit EOG immediately (empty output).
  raw=$(curl -s --max-time 600 http://127.0.0.1:8091/v1/chat/completions -H 'Content-Type: application/json' \
        -d '{"messages":[{"role":"user","content":"What is 2+2? Answer with just the number."}],"max_tokens":32,"temperature":0}')
  out=$(echo "$raw" | "$PY" -c 'import json,sys;print(" ".join((json.load(sys.stdin)["choices"][0]["message"]["content"] or "").split()))' 2>/dev/null || true)
  [ -z "$out" ] && echo "chat response: $raw"
  kill "$LPID" 2>/dev/null || true
  echo "gemma output: $out"
  [ -n "$out" ] && echo "$out" | grep -q "4"
}
FREE_GB=$(df -Pk "$WORK" | awk 'NR==2{print int($4/1048576)}')
GEMMA_PICK="$GEMMA_FILE"
if [ "$FREE_GB" -lt 5 ]; then
  if [ "$FREE_GB" -ge 4 ]; then
    GEMMA_PICK="$GEMMA_FALLBACK_FILE"
    note "gemma: only ${FREE_GB}GB free, downgrading to smallest published quant $GEMMA_PICK"
  else
    GEMMA_PICK=""
  fi
fi
if [ -z "$GEMMA_PICK" ]; then
  record gemma SKIP "insufficient disk (${FREE_GB}GB free) for any published quant"
elif gemma_leg "$GEMMA_PICK"; then
  record gemma PASS "coherent completion containing 4 from $GEMMA_PICK"
else
  tail -30 "$WORK/llama_server.log" 2>/dev/null || true
  if [ "$GEMMA_OPTIONAL" = "1" ]; then
    record gemma SKIP "slow-leg tolerance: gemma download/inference did not finish (GEMMA_OPTIONAL=1)"
  else
    record gemma FAIL "download/server/completion failed for $GEMMA_PICK"
  fi
fi
rm -f "$WORK/$GEMMA_FILE" "$WORK/$GEMMA_FALLBACK_FILE" 2>/dev/null || true

# ---------------------------------------------------------------- rocm resolve spoof (linux x64)
if [ "$DO_ROCM_RESOLVE" = "1" ]; then
  echo "### rocm-resolve: --has-rocm --rocm-gfx gfx1100 (explicit flags; no PATH stubs needed)"
  L=$("$PY" studio/install_llama_prebuilt.py --resolve-prebuilt --has-rocm --rocm-gfx gfx1100 --output-format json || true)
  W=$("$PY" studio/install_whisper_prebuilt.py --resolve-prebuilt --has-rocm --rocm-gfx gfx1100 --output-format json || true)
  echo "$L"; echo "$W"
  if echo "$L" | grep -q 'rocm-gfx110X' && echo "$W" | grep -q 'slim'; then
    record rocm-resolve PASS "llama selects rocm-gfx110X artifact, whisper stays slim"
  else
    record rocm-resolve FAIL "unexpected resolver selection (see JSON above)"
  fi
fi

# ---------------------------------------------------------------- full rocm install (ubuntu-latest)
if [ "$DO_ROCM_FULL" = "1" ]; then
  echo "### rocm-full: real rocm llama install + slim whisper pairing on top"
  export UNSLOTH_STUDIO_HOME="$ROOT/.studio_rocm"
  RL="$UNSLOTH_STUDIO_HOME/llama.cpp"; RW="$UNSLOTH_STUDIO_HOME/whisper.cpp"
  if "$PY" studio/install_llama_prebuilt.py --install-dir "$RL" --has-rocm --rocm-gfx gfx1100 2>&1 | tail -30 \
     && "$PY" studio/install_whisper_prebuilt.py --install-dir "$RW" --has-rocm --rocm-gfx gfx1100 2>&1 | tail -30; then
    HIP_COUNT=$(find "$RW/$BINSUB" -maxdepth 1 -name 'libggml-hip*' 2>/dev/null | wc -l)
    PAIRED=$(json_field "$RW/UNSLOTH_WHISPER_PREBUILT_INFO.json" paired_llama_tag || echo MISSING)
    if [ "$HIP_COUNT" -gt 0 ] && [ "$PAIRED" = "$NEW_TAG" ]; then
      record rocm-full PASS "hip ggml module wired into whisper bin (count=$HIP_COUNT), paired=$PAIRED"
    else
      record rocm-full FAIL "hip modules=$HIP_COUNT paired=$PAIRED (want $NEW_TAG)"
    fi
  else
    record rocm-full FAIL "rocm llama or whisper install exited nonzero"
  fi
  export UNSLOTH_STUDIO_HOME="$ROOT/.studio_ci"
fi

# ---------------------------------------------------------------- summary
echo; echo "==================== SUMMARY ===================="
FAILED=0
{
  echo "| check | result | detail |"; echo "|---|---|---|"
} >> "${GITHUB_STEP_SUMMARY:-/dev/null}" 2>/dev/null || true
for r in "${RESULTS[@]}"; do
  IFS='|' read -r name res why <<< "$r"
  echo "$name: $res ($why)"
  echo "| $name | $res | $why |" >> "${GITHUB_STEP_SUMMARY:-/dev/null}" 2>/dev/null || true
  [ "$res" = FAIL ] && FAILED=1
done
exit "$FAILED"
