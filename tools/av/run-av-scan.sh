#!/usr/bin/env bash
# Scan launcher variants with ClamAV on macOS / Linux. One JSON per variant.
# Keyless engine only (VirusTotal runs locally off-box). Usage:
#   run-av-scan.sh <artifacts_dir> <out_dir> <os_label>
set -u

ARTIFACTS="${1:?artifacts dir}"
OUTDIR="${2:?out dir}"
OSLABEL="${3:?os label, e.g. ubuntu-latest}"
mkdir -p "$OUTDIR"

json_escape() { python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))'; }

for vdir in "$ARTIFACTS"/V*/; do
  [ -d "$vdir" ] || continue
  variant="$(basename "$vdir")"
  echo "=== ClamAV scan $variant ($OSLABEL) ==="
  raw="$(clamscan --recursive --infected --no-summary "$vdir" 2>&1)"
  code=$?
  case $code in
    0) verdict="clean" ;;
    1) verdict="detected" ;;
    *) verdict="error" ;;
  esac
  files="$(find "$vdir" -maxdepth 1 -type f -printf '%f\n' 2>/dev/null | python3 -c 'import json,sys; print(json.dumps([l for l in sys.stdin.read().split()]))')"
  raw_j="$(printf '%s' "$raw" | json_escape)"
  cat > "$OUTDIR/$variant.json" <<EOF
{
  "variant": "$variant",
  "os": "$OSLABEL",
  "files": $files,
  "engines": {
    "clamav": { "verdict": "$verdict", "exit": $code, "raw": $raw_j }
  }
}
EOF
  echo "    clamav=$verdict -> $OUTDIR/$variant.json"
done
echo "done."
