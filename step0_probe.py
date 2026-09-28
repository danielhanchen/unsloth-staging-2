"""Harness only: step 0 for the isolated cmd Terminal. Checks, inside MXC with stock cmd.exe and stock git:
alias drive as process.cwd (DefineDosDeviceW, not subst), raw cmd /d /s /c serialization, core.hooksPath=NUL,
a nonexistent GIT_EDITOR, a quoted program path, and that the outside controls only mean something with raw quoting.
Payloads are harmless; files are harness-created markers.
"""

import ctypes
from ctypes import wintypes as w
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, "studio/backend")
from core.inference import mxc_adapter, mxc_policy, os_sandbox  # noqa: E402

k = ctypes.WinDLL("kernel32", use_last_error = True)
k.DefineDosDeviceW.argtypes = [w.DWORD, w.LPCWSTR, w.LPCWSTR]
k.QueryDosDeviceW.argtypes = [w.LPCWSTR, w.LPWSTR, w.DWORD]
DDD_REMOVE_DEFINITION, DDD_EXACT_MATCH_ON_REMOVE = 0x2, 0x4
CMD = os.path.join(os.environ["SystemRoot"], "System32", "cmd.exe")
GIT_DIR = r"C:\Program Files\Git\cmd"


def query(letter):
    buf = ctypes.create_unicode_buffer(4096)
    n = k.QueryDosDeviceW(f"{letter}:", buf, 4096)
    return buf.value if n else None


def define(letter, target):
    ok = k.DefineDosDeviceW(0, f"{letter}:", target)
    return bool(ok), ctypes.get_last_error(), query(letter)


def remove(letter, target):
    ok = k.DefineDosDeviceW(DDD_REMOVE_DEFINITION | DDD_EXACT_MATCH_ON_REMOVE, f"{letter}:", target)
    return bool(ok), ctypes.get_last_error(), query(letter)


def raw_cmdline(payload):
    return f'"{CMD}" /d /s /c "{payload}"'


def launch(payload, workdir, *, cwd = None, raw = True, extra_env = None):
    env = {key: v for key, v in os.environ.items() if key.upper() in {"SYSTEMROOT", "PATHEXT"}}
    env["PATH"] = os.pathsep.join([GIT_DIR, os.path.join(os.environ["SystemRoot"], "System32"), os.environ["SystemRoot"]])
    env.update(TEMP = workdir, TMP = workdir, HOME = workdir)
    env.update(extra_env or {})
    plan = os_sandbox.ToolLaunchPlan(argv = ("cmd", "/c", payload), workdir = workdir, env = env,
                                     requested_mode = "required", timeout_seconds = 120, execution_kind = "terminal")
    request = mxc_policy.build_launch_request(plan)
    process = request["config"]["process"]
    if raw:
        process["commandLine"] = raw_cmdline(payload)
    if cwd:
        process["cwd"] = cwd
    request["configBytes"] = mxc_policy.canonical_config_bytes(request["config"])
    request["policyHash"] = mxc_policy.compute_policy_hash(request["config"])
    started = time.monotonic()
    proc = mxc_adapter.spawn(request, popen_kwargs = {"stdout": subprocess.PIPE, "stderr": subprocess.STDOUT,
                                                      "text": True, "errors": "replace",
                                                      "creationflags": subprocess.CREATE_NO_WINDOW})
    try:
        out, _ = proc.communicate(timeout = 200)
        proc._unsloth_completion_reason = "finished"
        res = mxc_adapter.completion_result(proc)
    finally:
        mxc_adapter.release_runtime(proc)
    return out or "", res, time.monotonic() - started, process["commandLine"]


def show(name, ok, out, res, secs):
    tail = " | ".join((out or "").strip().splitlines()[-3:])[-300:]
    print(f"STEP0 {name.ljust(28)} {'PASS' if ok else 'FAIL'} exit={res.get('exitCode')} s={secs:.1f} | {tail}", flush = True)


def main(parent):
    os.makedirs(parent, exist_ok = True)
    wd = tempfile.mkdtemp(prefix = "s0-", dir = parent)
    letter = next(d for d in "ZYXWVUTS" if query(d) is None)
    print(f"DEFINE {letter}: -> {wd}: {define(letter, wd)}", flush = True)
    root = f"{letter}:\\"
    ident = "-c user.name=t -c user.email=t@example.invalid"

    out, res, secs, _ = launch(f'git init -q r && cd r && git {ident} commit --allow-empty -qm "quoted msg" && git log --oneline && cd && echo OK',
                               wd, cwd = root)
    show("alias_cwd_git_quoted_msg", "quoted msg" in out and "OK" in out, out, res, secs)

    out, res, secs, _ = launch("cd && echo OK", wd, cwd = None)
    show("canonical_cwd_control", "OK" in out, out, res, secs)

    out, res, secs, _ = launch(f'"{GIT_DIR}\\git.exe" --version', wd, cwd = root)
    show("quoted_program_path", "git version" in out, out, res, secs)

    for label, extra in (("hook_default", {}),
                         ("hook_hookspath_nul", {"GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "core.hooksPath", "GIT_CONFIG_VALUE_0": "NUL"})):
        repo = Path(wd, f"h_{label}")
        subprocess.run(["git", "init", "-q", str(repo)], check = True)
        marker = Path(wd, f"hook-ran-{label}.txt")
        Path(repo, ".git", "hooks", "pre-commit").write_bytes(f"#!/bin/sh\necho ran > '{marker.as_posix()}'\n".encode())
        out, res, secs, _ = launch(f"cd {repo.name} && git {ident} commit --allow-empty -qm x && git log --oneline && echo COMMITTED", wd,
                                   cwd = root, extra_env = extra)
        show(label, "COMMITTED" in out and not marker.exists(), out, res, secs)
        print(f"STEP0 {label} marker_exists={marker.exists()}", flush = True)

    repo = Path(wd, "ed")
    subprocess.run(["git", "init", "-q", str(repo)], check = True)
    out, res, secs, _ = launch(f"cd ed && git {ident} commit --allow-empty & echo DONE", wd, cwd = root,
                               extra_env = {"GIT_EDITOR": "unsloth-no-editor", "GIT_PAGER": ""})
    show("editor_fails_fast", "DONE" in out and secs < 30, out, res, secs)

    other = Path(tempfile.mkdtemp(prefix = "s0-other-"))
    secret = other / "outside secret.txt"
    secret.write_text("OUTSIDE_MARKER_TEXT", encoding = "utf-8")
    written = other / "written inside.txt"
    read_cmd = f'type "{secret}" & echo END'
    write_cmd = f'(echo x> "{written}") & echo END'
    for raw in (False, True):
        tag = "raw" if raw else "list2cmdline"
        out, res, secs, line = launch(read_cmd, wd, raw = raw)
        show(f"outside_read_denied_{tag}", "OUTSIDE_MARKER_TEXT" not in out and "END" in out, out, res, secs)
        host = subprocess.run(line, capture_output = True, text = True, cwd = wd)
        print(f"STEP0 host_same_cmdline_{tag} reads_secret={'OUTSIDE_MARKER_TEXT' in host.stdout} | {line[:160]}", flush = True)
        out, res, secs, line = launch(write_cmd, wd, raw = raw)
        show(f"outside_write_denied_{tag}", not written.exists() and "END" in out, out, res, secs)
        host = subprocess.run(line, capture_output = True, text = True, cwd = wd)
        print(f"STEP0 host_same_cmdline_write_{tag} wrote={written.exists()}", flush = True)
        written.unlink(missing_ok = True)

    print(f"REMOVE {letter}: {remove(letter, wd)}", flush = True)


if __name__ == "__main__":
    main(sys.argv[1])
