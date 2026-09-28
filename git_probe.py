"""Harness only: which git operations work inside Studio's MXC container, per shell and per git build.

Usage: python git_probe.py <label> <shell.exe> <git.exe> [extra read root ...]
Each case runs in a fresh workdir; host-side setup (a repo with a hook) is done before the launch.
Payloads are harmless; the network case clones a public example repo.
"""

from pathlib import Path
import subprocess
import sys
import tempfile

from shell_probe import run_in_mxc, shell_argv  # noqa: F401

HOOK = "#!/bin/sh\necho HOOK_RAN\n"


def host_repo(wd, hook = False):
    subprocess.run(["git", "init", "-q", str(Path(wd, "r"))], check = True)
    if hook:
        path = Path(wd, "r", ".git", "hooks", "pre-commit")
        path.write_bytes(HOOK.encode())


def cases(git, cmd_style):
    # cmd gets bare "git" from PATH (list2cmdline turns an embedded quote into \" that cmd /s /c cannot parse)
    q = f'"{git}"' if not cmd_style and " " in git else git
    ident = "-c user.name=t -c user.email=t@example.invalid"
    cd = "cd r &&" if not cmd_style else "cd r &&"
    if cmd_style:
        diag = [
            ("diag_where", None, "where git & echo END", "END"),
            ("diag_nul", None, "type nul && echo x> nul && echo CMD_NUL_OK", "CMD_NUL_OK"),
            ("diag_py_nul", None, "%PY% -c print(open(__import__('os').devnull,'r+b')and'PY_NUL_OK')", "PY_NUL_OK"),
        ]
    else:
        diag = [
            ("diag_where", None, "command -v git; type -a git; echo END", "END"),
            ("diag_devnull", None, "ls -l /dev/null; : > /dev/null && echo W_OK; cat /dev/null && echo R_OK; exec 3<>/dev/null && echo RW_OK", "RW_OK"),
            ("diag_py_nul", None, "\"$PY\" -c \"import os; open(os.devnull, 'r+b'); print('PY_NUL_OK')\"", "PY_NUL_OK"),
        ]
    return diag + [
        ("version", None, f"{q} --version", "git version"),
        ("init_commit_log", None,
         f"{q} init -q r && {cd} {q} {ident} commit --allow-empty -qm first && {q} log --oneline", "first"),
        ("add_status_diff", lambda wd: host_repo(wd),
         f"{cd} echo a> f.txt && {q} add f.txt && {q} status --short && {q} diff --cached --stat", "f.txt"),
        ("branch_stash", lambda wd: host_repo(wd),
         f"{cd} {q} {ident} commit --allow-empty -qm base && echo x> g.txt && {q} add g.txt && {q} stash -q && {q} stash list", "stash@{0}"),
        ("hook_spawns_sh", lambda wd: host_repo(wd, hook = True),
         f"{cd} {q} {ident} commit --allow-empty -qm hooked && {q} log --oneline", "HOOK_RAN"),
        ("clone_https", None,
         f"{q} clone -q --depth 1 https://github.com/octocat/Hello-World.git hw && {q} -C hw log --oneline -1", " "),
        ("config_global", None,
         f"{q} config --global probe.key yes && {q} config --global probe.key", "yes"),
    ]


def main(label, shell, git, extra_roots):
    cmd_style = Path(shell).name.casefold().startswith("cmd")
    passed = 0
    rows = cases(git, cmd_style)
    for name, setup, script, expect in rows:
        wd = tempfile.mkdtemp(prefix = "gitp-")
        if setup:
            setup(wd)
        out, res, secs = run_in_mxc(shell, script, wd, extra_roots)
        ok = res.get("exitCode") == 0 and expect in out
        passed += ok
        tail = " | ".join(out.strip().splitlines()[-3:])[-300:]
        print(f"GIT {label} {name.ljust(16)} {'ok' if ok else 'FAIL'} exit={res.get('exitCode')} s={secs:.1f} | {tail}", flush = True)
    print(f"GIT_TOTAL {label} {passed}/{len(rows)}", flush = True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:])
