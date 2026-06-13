#!/bin/sh
# Suite C: env-var layering. Confirms the ":=" defaults in install.sh apply when
# unset but PRESERVE any user-provided override. Run: <shell> suite_c.sh <install.sh>
INSTALL_SH="${1:-$(dirname "$0")/../unsloth/install.sh}"
[ -f "$INSTALL_SH" ] || { echo "install.sh not found at $INSTALL_SH"; exit 2; }

pass=0; fail=0
chk() { if [ "$2" = "$3" ]; then pass=$((pass+1)); echo "  PASS  $1 = $2"; else fail=$((fail+1)); echo "  FAIL  $1 = $2 (want $3)"; fi; }

# (var:default) pairs as written in install.sh
for VAR_DEF in "UV_HTTP_RETRIES:5" "UV_HTTP_TIMEOUT:180" "UNSLOTH_INSTALL_RETRIES:3" "UNSLOTH_INSTALL_RETRY_DELAY:3"; do
  V=${VAR_DEF%%:*}; D=${VAR_DEF##*:}
  # the default line must literally exist in install.sh
  if grep -q ": \"\${$V:=$D}\"" "$INSTALL_SH"; then pass=$((pass+1)); echo "  PASS  install.sh declares $V default=$D";
  else fail=$((fail+1)); echo "  FAIL  install.sh missing $V default=$D"; fi
  # unset -> default
  got=$(sh -c "unset $V; : \"\${$V:=$D}\"; eval echo \"\$$V\"")
  chk "$V unset -> default" "$got" "$D"
  # preset -> preserved
  got=$(sh -c "$V=99; : \"\${$V:=$D}\"; eval echo \"\$$V\"")
  chk "$V preset=99 -> preserved" "$got" "99"
done
echo "  ------------------------------------"
echo "  SUITE C: $pass passed, $fail failed"
[ "$fail" = 0 ]
