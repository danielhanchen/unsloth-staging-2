#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Check every `unsloth` subcommand is reachable after an install.

The reported failure was a CLI that answered `-h` while the backend could not
import, so "the binary exists" proves nothing. Each command here is invoked for
real and its exit code checked, which is what catches a subcommand whose module
imports something the dependency pass never installed.

`--help` is used rather than a real run because the point is import reachability
on a fresh machine, not model work. A missing dependency in a command module
raises during import, long before argument parsing.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Registered in unsloth_cli/__init__.py. `connect` is the hidden deprecated alias
# for `start`, included because a broken alias is still a broken command.
TOP_LEVEL = [
    "train", "inference", "chat", "export", "list-checkpoints",
    "studio", "start", "connect", "run",
]

# Subcommands the desktop app itself shells out to. These are the ones that made
# the difference between a repairable install and a permanent dead end.
STUDIO_SUB = ["run", "update", "setup"]


def run(cmd: list[str], timeout: int = 180) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout}s"
    except OSError as exc:
        return 127, str(exc)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("binary", help="path to the installed `unsloth`")
    ap.add_argument("--out", default="cli-smoke", help="directory for the report")
    args = ap.parse_args()

    binp = str(Path(args.binary).expanduser())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    failed: list[str] = []

    def check(label: str, cmd: list[str], required: bool = True) -> None:
        rc, log = run(cmd)
        ok = rc == 0
        results[label] = {"rc": rc, "ok": ok, "required": required}
        (out / f"{label.replace(' ', '_').replace('/', '_')}.log").write_text(
            log, encoding="utf-8", errors="replace")
        print(f"[cli] {'ok  ' if ok else 'FAIL'} {label} (rc={rc})")
        if not ok:
            # The import error is the useful part, so surface it inline rather
            # than leaving it in an artifact nobody opens.
            for line in log.splitlines():
                if "Error" in line or "error" in line:
                    print(f"        {line[:160]}")
                    break
            if required:
                failed.append(label)

    check("unsloth --help", [binp, "--help"])
    for name in TOP_LEVEL:
        check(f"unsloth {name} --help", [binp, name, "--help"])
    for name in STUDIO_SUB:
        check(f"unsloth studio {name} --help", [binp, "studio", name, "--help"])

    # Not --help: these actually inspect the install and are what the desktop
    # preflight reads. Absent on older trees, so never required.
    rc, log = run([binp, "studio", "desktop-capabilities", "--json"])
    (out / "desktop-capabilities.log").write_text(log, encoding="utf-8", errors="replace")
    caps = {}
    if rc == 0:
        try:
            caps = json.loads(log[log.index("{"):log.rindex("}") + 1])
        except (ValueError, json.JSONDecodeError):
            pass
    results["desktop-capabilities"] = {"rc": rc, "ok": rc == 0, "payload": caps}
    print(f"[cli] {'ok  ' if rc == 0 else 'note'} desktop-capabilities (rc={rc}) "
          f"studio_install_ok={caps.get('studio_install_ok', 'absent')}")

    check("unsloth studio verify-install", [binp, "studio", "verify-install"], required=False)

    (out / "report.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    if failed:
        print(f"::error::{len(failed)} CLI command(s) unusable after install: {', '.join(failed)}")
        return 1
    print(f"[cli] all {len(results)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
