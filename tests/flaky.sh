#!/bin/sh
# Fake install command. Fails its first $1 invocations (counted in $STATE),
# then succeeds. Optional $2 = exit code to use on failure (default 4).
N="$1"; EC="${2:-4}"; STATE="${STATE:?STATE unset}"
c=$(cat "$STATE" 2>/dev/null || echo 0); c=$((c + 1)); echo "$c" > "$STATE"
if [ "$c" -le "$N" ]; then
    echo "attempt $c: error decoding response body: connection reset"
    exit "$EC"
fi
echo "attempt $c: ok"
exit 0
