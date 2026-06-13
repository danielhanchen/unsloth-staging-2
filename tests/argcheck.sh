#!/bin/sh
# Echo argument count and each argument on its own line, so the caller can
# verify run_install_cmd_retry passes "$@" through verbatim (spaces, globs,
# quotes, leading dashes, etc.).
printf 'NARGS=%s\n' "$#"
for a in "$@"; do printf 'ARG=[%s]\n' "$a"; done
exit 0
