#!/usr/bin/env bash
# Probe the in-container notebook refresh. Simulates a user who deleted one baked
# notebook and edited another, runs the boot-time sync, and reports:
#   a_off  : the DELETED notebook healed back (offline, from the baked template)
#   b_kept : the EDITED notebook was left intact (not clobbered)
#   c_same : a third, untouched notebook stayed unchanged
#   a_net  : the deleted notebook also restores via the GitHub refresh path
#            ("old repo": stale the synced commit so upstream looks advanced) -
#            bonus, may be 0 if the runner has no network to unslothai/notebooks.
# Emits a single "NBRESULT ..." line for the caller to parse.
set -u
D="${UNSLOTH_NOTEBOOKS_DIR:-/workspace/unsloth-notebooks}"
PY=/opt/unsloth-venv/bin/python
[ -x "$PY" ] || PY=python3
ls "$D"/nb/*.ipynb >/dev/null 2>&1 || { echo "NBRESULT error=no_notebooks dir=$D"; exit 0; }

A="$(ls "$D"/nb/*.ipynb | sed -n 1p)"   # will be deleted
B="$(ls "$D"/nb/*.ipynb | sed -n 2p)"   # will be edited
C="$(ls "$D"/nb/*.ipynb | sed -n 3p)"   # left untouched
shaC0="$(sha256sum "$C" | cut -d' ' -f1)"

# user edits B (valid-JSON edit so a hash change is registered as a user edit)
"$PY" -c 'import json,sys; p=sys.argv[1]; nb=json.load(open(p)); nb["cells"][0].setdefault("source",[]).append("\nedited-by-test\n"); json.dump(nb, open(p,"w"))' "$B"
shaB1="$(sha256sum "$B" | cut -d' ' -f1)"

# user deletes A, then the OFFLINE sync (no network) should heal it from template
rm -f "$A"
UNSLOTH_SKIP_NOTEBOOK_REFRESH=1 /usr/local/bin/unsloth-sync-notebooks >/dev/null 2>&1 || true
a_off=0; [ -f "$A" ] && a_off=1

# deterministic keep/unchanged checks taken after the offline pass (no upstream
# involved, so C cannot legitimately change and B must remain the edited file)
shaB2="$(sha256sum "$B" | cut -d' ' -f1)"
b_kept=0; [ "$shaB1" = "$shaB2" ] && b_kept=1
shaC1="$(sha256sum "$C" | cut -d' ' -f1)"
c_same=0; [ "$shaC0" = "$shaC1" ] && c_same=1

# bonus: the "old repo" GitHub-refresh path -- stale the synced commit so
# upstream looks advanced, delete A again, and let the network refresh restore it
printf '%s\n' "0000000000000000000000000000000000000000" > "$D/.unsloth_sync_commit"
rm -f "$A"
/usr/local/bin/unsloth-sync-notebooks >/dev/null 2>&1 || true
a_net=0; [ -f "$A" ] && a_net=1

echo "NBRESULT a_off=$a_off a_net=$a_net b_kept=$b_kept c_same=$c_same"
