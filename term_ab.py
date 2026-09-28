"""Harness only: base vs head for the Windows Terminal in Auto, through Studio's real tools._bash_exec.

Usage: python term_ab.py <label>
Every command is valid in both Git Bash (base) and cmd (head), so the two arms run the same text.
Payloads are harmless; the outside files are harness-created markers.
"""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, "studio/backend")
from core.inference import tools  # noqa: E402

SESSION = "__LOCALID_term_ab_" + (sys.argv[1] if len(sys.argv) > 1 else "x")
IDENT = "-c user.name=t -c user.email=t@example.invalid"


def run(label, name, command, check):
    started = time.monotonic()
    try:
        out = tools._bash_exec(command, None, 150, SESSION)
    except Exception as exc:  # noqa: BLE001 - harness diagnostics
        out = f"<raised {type(exc).__name__}: {exc}>"
    secs = time.monotonic() - started
    record = getattr(tools, "_last_tool_execution_record", None)
    mode = getattr(record, "effective_mode", "?")
    backend = getattr(record, "backend", "?")
    lims = ",".join(sorted(getattr(record, "limitations", ()) or ()))
    ok = check(out)
    tail = " | ".join(out.strip().splitlines()[-3:])[-260:]
    print(f"TERM {label} {name.ljust(20)} {'PASS' if ok else 'FAIL'} mode={mode} backend={backend} s={secs:.1f} | {tail}", flush = True)
    print(f"LIMITS {label} {name} {lims}", flush = True)


def main(label):
    profile = tools._terminal_profile(False) if hasattr(tools, "_terminal_profile") else "absent"
    print(f"PROFILE {label} {profile} bash={tools._windows_bash()}", flush = True)
    wd = Path(tools._get_workdir(SESSION))
    other = Path(tempfile.mkdtemp(prefix = "term-ab-other-"))
    secret = other / "outside secret.txt"
    secret.write_text("OUTSIDE_MARKER_TEXT", encoding = "utf-8")
    made = other / "made from inside"
    subprocess.run(["git", "init", "-q", str(wd / "hooked")], check = True)
    marker = other / "hook-ran.txt"
    Path(wd, "hooked", ".git", "hooks", "pre-commit").write_bytes(f"#!/bin/sh\necho ran > '{marker.as_posix()}'\n".encode())

    run(label, "commit_log_status", f'git init -q r && cd r && git {IDENT} commit --allow-empty -qm "quoted msg" && git log --oneline && git status --short && echo AB_OK',
        lambda o: "quoted msg" in o and "AB_OK" in o)
    run(label, "add_diff_stash", "cd r && echo a> f.txt && git add f.txt && git diff --cached --stat && git stash && git stash list",
        lambda o: "stash@{0}" in o)
    run(label, "hook_commit", f"cd hooked && git {IDENT} commit --allow-empty -qm x && echo COMMITTED",
        lambda o: "COMMITTED" in o)
    print(f"HOOK {label} marker_exists={marker.exists()}", flush = True)
    run(label, "clone_https", "git clone -q --depth 1 https://github.com/octocat/Hello-World.git hw && git -C hw log --oneline -1",
        lambda o: "Merge" in o or "pull request" in o)
    run(label, "quoted_program", '"C:\\Program Files\\Git\\cmd\\git.exe" --version', lambda o: "git version" in o)
    out = tools._bash_exec(f'git hash-object "{secret}"', None, 150, SESSION)
    read_allowed = any(len(t) == 40 and all(c in "0123456789abcdef" for c in t) for t in out.split())
    print(f"OUTSIDE {label} read_allowed={read_allowed}", flush = True)
    tools._bash_exec(f'git init -q "{made}"', None, 150, SESSION)
    print(f"OUTSIDE {label} write_allowed={made.exists()}", flush = True)
    run(label, "multiline", "echo one\necho two", lambda o: True)


if __name__ == "__main__":
    main(sys.argv[1])
