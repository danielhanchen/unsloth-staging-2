#!/usr/bin/env python3
"""Encrypt a short-lived HF token for the AMD CI, the only form one may ever take there.

The rule (the user's words): "you must encrypt this if using this on the AMD CI machines". The default is
stronger still: the AMD specs use public, ungated weights only and the probe drops every token from its
environment, so this file is for the exception, a gated model a run genuinely needs, authorised for that run.

Format: the same AES-256-CBC + PBKDF2 `openssl enc` scheme scripts/account_pool.py uses for account_pool.enc
(`openssl enc -aes-256-cbc -pbkdf2 -pass env:<VAR>`), with its own passphrase variable, DBENCH_CI_PASSPHRASE, so a
leak of one secret never opens the other blobs.

  # on your machine: a FINE-GRAINED, READ-ONLY token scoped to the one gated repo, expiring within a day
  export HF_TOKEN_FOR_CI=hf_...            # never commit, never echo
  export DBENCH_CI_PASSPHRASE=$(openssl rand -hex 32)
  python diffusion_bench/amd/token_vault.py encrypt --from-env HF_TOKEN_FOR_CI --out ci_prN/diffusion_bench/amd/hf_token.enc
  # the passphrase reaches the runner only as an Actions secret on the host repo (repo admin), exposed to the ONE
  # step that runs the probe; the probe gets --token-blob diffusion_bench/amd/hf_token.enc and decrypts in memory
  # for snapshot_download(token=...) of aliases marked "gated": true. Revoke the token when the run ends.

What encryption does and does not buy: the branch lives on a PUBLIC fork, so the blob must be useless without the
passphrase (hence 32 random bytes, not a word). It does NOT protect against the runner host, which sees the
decrypted token in the probe's memory; the real control is the token's scope and lifetime. No `secrets.*` exists
in any generated workflow today: adding one is a per-run decision the user makes, not a default.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

PASS_ENV = "DBENCH_CI_PASSPHRASE"


def _openssl() -> str:
    exe = shutil.which("openssl")
    if exe:
        return exe
    # Git for Windows ships one; the Windows runners have git.exe. TODO(verify) the path on those boxes.
    for cand in (r"C:\Program Files\Git\usr\bin\openssl.exe", r"C:\Program Files\Git\mingw64\bin\openssl.exe"):
        if Path(cand).is_file():
            return cand
    raise SystemExit("openssl not found")


def _require_passphrase() -> None:
    if not os.environ.get(PASS_ENV):
        raise SystemExit(f"${PASS_ENV} is not set; it is the AES-256-CBC + PBKDF2 passphrase")


def encrypt_token(token: str, out: Path) -> None:
    _require_passphrase()
    proc = subprocess.run([_openssl(), "enc", "-aes-256-cbc", "-pbkdf2", "-salt", "-pass", f"env:{PASS_ENV}"],
                          input = token.strip().encode(), capture_output = True)
    if proc.returncode:
        raise SystemExit(f"openssl encrypt failed (rc={proc.returncode}): {proc.stderr.decode(errors = 'replace')[:200]}")
    out.parent.mkdir(parents = True, exist_ok = True)
    out.write_bytes(proc.stdout)


def decrypt_token(blob: str | Path) -> str:
    """The token, in memory only. Callers pass it to the downloader and drop it; never log or export it."""
    _require_passphrase()
    proc = subprocess.run([_openssl(), "enc", "-d", "-aes-256-cbc", "-pbkdf2", "-pass", f"env:{PASS_ENV}"],
                          input = Path(blob).read_bytes(), capture_output = True)
    if proc.returncode:
        raise SystemExit("openssl decrypt failed: wrong passphrase or not a token blob")
    return proc.stdout.decode().strip()


def main() -> int:
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest = "cmd", required = True)
    e = sub.add_parser("encrypt", help = "read a token from an env var, write the encrypted blob")
    e.add_argument("--from-env", required = True, help = "name of the env var holding the token (never a literal)")
    e.add_argument("--out", required = True, type = Path)
    c = sub.add_parser("check", help = "decrypt in memory and report only whether it looks like an HF token")
    c.add_argument("blob", type = Path)
    args = ap.parse_args()
    if args.cmd == "encrypt":
        token = os.environ.get(args.from_env, "")
        if not token:
            raise SystemExit(f"${args.from_env} is empty")
        encrypt_token(token, args.out)
        print(f"wrote {args.out} ({args.out.stat().st_size} bytes); the token itself was not printed")
        return 0
    tok = decrypt_token(args.blob)
    print("decrypts: yes; looks like an HF token:", tok.startswith("hf_"), "; length", len(tok))
    return 0


if __name__ == "__main__":
    sys.exit(main())
