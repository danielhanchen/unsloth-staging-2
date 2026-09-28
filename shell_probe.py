"""Harness only: can a POSIX shell run inside Studio's MXC container, and how much typical bash does it accept?

Usage: python shell_probe.py compat <shell.exe> [<shell.exe> ...]   host-side syntax check, no container
       python shell_probe.py container <label> <shell.exe> [extra read root]
Every payload is harmless; the two isolation controls only touch a marker file the harness itself creates.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, "studio/backend")
from core.inference import mxc_adapter, mxc_policy, os_sandbox  # noqa: E402

SNIPPETS = [
    ("echo", "echo hello", "hello"),
    ("for_loop", "for i in 1 2 3; do echo n$i; done", "n3"),
    ("multiline_if", "x=5\nif [ $x -gt 3 ]; then\n  echo big\nfi", "big"),
    ("double_bracket", '[[ "abc" == a* ]] && echo match', "match"),
    ("arith_pow", "echo $((2**10))", "1024"),
    ("heredoc", "cat <<'EOF' > note.txt\nline one\nEOF\ncat note.txt", "line one"),
    ("sort_uniq", "printf 'b\\na\\nb\\n' | sort | uniq -c | sort -rn | head -1", "2 b"),
    ("sed_awk", "echo 'foo bar' | sed 's/foo/baz/' | awk '{print $1}'", "baz"),
    ("grep_r", "mkdir -p d && echo needle > d/f.txt && grep -rl needle .", "f.txt"),
    ("find", "mkdir -p a/b && touch a/b/x.py && find . -name '*.py'", "x.py"),
    ("wc", "printf 'a\\nb\\nc\\n' | wc -l", "3"),
    ("case_upper", 'v=hello; echo "${v^^}"', "HELLO"),
    ("replace_all", "v=a-b-c; echo ${v//-/_}", "a_b_c"),
    ("default", "echo ${UNSET_VAR_XYZ:-fallback}", "fallback"),
    ("pipefail", "set -euo pipefail; echo ok", "ok"),
    ("function", 'greet() { echo "hi $1"; }; greet bob', "hi bob"),
    ("array", "arr=(one two three); echo ${arr[1]} ${#arr[@]}", "two 3"),
    ("here_string", 'read -r a b <<< "x y"; echo $b', "y"),
    ("process_subst", "diff <(echo a) <(echo a) && echo same", "same"),
    ("trap_exit", "trap 'echo bye' EXIT; echo hi", "bye"),
    ("python", 'python -c "print(6*7)"', "42"),
    ("python_resolved", 'command -v python; "$PY" -c "print(6*7)"', "42"),
    ("cut_tr_xargs", "echo 'a:b:c' | cut -d: -f2 | tr a-z A-Z | xargs echo got", "got B"),
    ("background_wait", "(sleep 1; echo bg) & wait; echo done", "done"),
    ("echo_e", 'echo -e "a\\tb" | cut -f2', "b"),
    ("mkdir_cd_pwd", "mkdir -p x/y && cd x/y && pwd", "/y"),
    ("local_return", "f(){ local v=3; return $v; }; f; echo rc=$?", "rc=3"),
    ("case", "case foo in f*) echo yes;; esac", "yes"),
    ("while_read", "printf '1\\n2\\n' | while read n; do echo r$n; done", "r2"),
    ("seq_tail", "seq 1 3 | tail -1", "3"),
    ("cmd_subst", 'n=$(echo 4 | tr 4 7); echo "v$n"', "v7"),
    ("git_last", "git --version", "git version"),
    ("heredoc_script", "cat > s.py <<'EOF'\nimport json\nprint(json.dumps({'ok': 6 * 7}))\nEOF\n\"$PY\" s.py", '"ok": 42'),
    ("pip_target", '"$PY" -m pip install -q --no-deps --target ./pkgs six==1.16.0 && PYTHONPATH=./pkgs "$PY" -c "import six; print(six.__version__)"', "1.16.0"),
    ("wget_https", "wget -q -O - https://pypi.org/simple/six/ | head -c 200 | grep -o six | head -1", "six"),
    ("tar_gzip", "mkdir -p t && echo data > t/a.txt && tar czf t.tgz t && rm -r t && tar xzf t.tgz && cat t/a.txt", "data"),
    ("ls_du", "echo abc > f1 && ls -la && du -s .", "f1"),
    ("exit_code", '"$PY" -c "import sys; sys.exit(5)"; echo rc=$?', "rc=5"),
]


def shell_argv(shell, script):
    name = Path(shell).name.casefold()
    if name in {"cmd", "cmd.exe"}:
        return [shell, "/d", "/s", "/c", script]
    if name.startswith("python"):
        return [shell, "-c", script]
    if name.startswith("busybox"):
        # busybox emulates exec of the last command by spawning and exiting, which loses the child's output
        # inside MXC; a trailing builtin keeps busybox alive until the program ends and forwards its status.
        return [shell, "sh", "-c", script + "\nexit $?"]
    return [shell, "-c", script]


def compat(shells):
    table = {}
    for shell in shells:
        for name, script, expect in SNIPPETS:
            work = tempfile.mkdtemp(prefix = "shp-")
            try:
                env = dict(os.environ, PY = sys.executable)
                done = subprocess.run(shell_argv(shell, script), cwd = work, env = env, capture_output = True,
                                      text = True, errors = "replace", timeout = 60)
                ok = done.returncode == 0 and expect in done.stdout
                detail = (done.stdout + done.stderr).strip().splitlines()[-1:] if not ok else []
            except Exception as exc:  # noqa: BLE001 - harness diagnostics
                ok, detail = False, [repr(exc)]
            table.setdefault(name, {})[Path(shell).name] = "ok" if ok else f"FAIL {detail[0][:120] if detail else ''}"
    for name, row in table.items():
        print("COMPAT", name.ljust(16), json.dumps(row), flush = True)
    for shell in shells:
        passed = sum(1 for row in table.values() if row[Path(shell).name] == "ok")
        print(f"COMPAT_TOTAL {Path(shell).name} {passed}/{len(SNIPPETS)}", flush = True)


def run_in_mxc(shell, script, workdir, extra_roots):
    env = {k: v for k, v in os.environ.items() if k.upper() in {"SYSTEMROOT", "WINDIR", "PATH", "PATHEXT"}}
    env.update(TEMP = workdir, TMP = workdir, TMPDIR = workdir, HOME = workdir, PY = sys.executable)
    if os.environ.get("MSYS_DIAG"):
        env["MSYS_DIAG_LOG"] = str(Path(workdir) / "diag.txt")
    if os.environ.get("MSYS_DIAG_OCCUPY"):
        env["MSYS_DIAG_OCCUPY"] = "1"
    argv = tuple(script) if isinstance(script, list) else tuple(shell_argv(shell, script))
    plan = os_sandbox.ToolLaunchPlan(argv = argv, workdir = workdir, env = env,
                                     requested_mode = "required", timeout_seconds = int(os.environ.get("PROBE_TIMEOUT", "120")), execution_kind = "terminal")
    real = mxc_policy._runtime_read_roots
    if extra_roots:
        mxc_policy._runtime_read_roots = lambda exe, extra = (): real(exe, [*extra, *extra_roots])
    try:
        request = mxc_policy.build_launch_request(plan)
    finally:
        mxc_policy._runtime_read_roots = real
    started = time.monotonic()
    proc = mxc_adapter.spawn(request, popen_kwargs = {"stdout": subprocess.PIPE, "stderr": subprocess.STDOUT,
                                                      "text": True, "errors": "replace",
                                                      "creationflags": subprocess.CREATE_NO_WINDOW})
    try:
        out, _ = proc.communicate(timeout = int(os.environ.get("PROBE_TIMEOUT", "120")) + 60)
        proc._unsloth_completion_reason = "finished"
        result = mxc_adapter.completion_result(proc)
    except subprocess.TimeoutExpired:
        mxc_adapter.abort(proc)
        out, result = (proc.communicate(timeout = 10)[0] or "") + "\n<TIMEOUT>", {}
    finally:
        mxc_adapter.release_runtime(proc)
    return out or "", result, time.monotonic() - started


def container(label, shell, extra_roots):
    other = Path(tempfile.mkdtemp(prefix = "shp-other-"))
    marker = other / "outside-marker.txt"
    marker.write_text("OUTSIDE_MARKER_TEXT", encoding = "utf-8")
    denied_write = other / "written-from-inside.txt"
    controls = [
        ("inside_write", "echo ok > inside.txt && cat inside.txt", lambda out, wd: "ok" in out and (Path(wd) / "inside.txt").is_file()),
        ("outside_read_denied", f"cat '{marker.as_posix()}'; echo END", lambda out, wd: "OUTSIDE_MARKER_TEXT" not in out and "END" in out),
        ("outside_write_denied", f"echo x > '{denied_write.as_posix()}'; echo END", lambda out, wd: not denied_write.exists() and "END" in out),
    ]
    results = []
    for name, script, check in controls:
        wd = tempfile.mkdtemp(prefix = "shp-wd-")
        out, res, secs = run_in_mxc(shell, script, wd, extra_roots)
        ok = check(out, wd)
        results.append(ok)
        print(f"CONTROL {label} {name} {'PASS' if ok else 'FAIL'} exit={res.get('exitCode')} cleanup={res.get('cleanup')} s={secs:.1f}", flush = True)
        if not ok:
            print("  output:", out[-800:].replace("\n", " | "), flush = True)
    passed = 0
    for name, script, expect in SNIPPETS:
        wd = tempfile.mkdtemp(prefix = "shp-wd-")
        out, res, secs = run_in_mxc(shell, script, wd, extra_roots)
        ok = res.get("exitCode") == 0 and expect in out
        passed += ok
        tail = "" if ok else " | " + out.strip().replace("\n", " / ")[-200:]
        print(f"INSIDE {label} {name.ljust(16)} {'ok' if ok else 'FAIL'} exit={res.get('exitCode')} s={secs:.1f}{tail}", flush = True)
        if not ok:
            dump_diag(wd, 1500)
    print(f"INSIDE_TOTAL {label} {passed}/{len(SNIPPETS)} controls={'PASS' if all(results) else 'FAIL'}", flush = True)


def dump_diag(wd, limit):
    for name in ("diag.txt", "trace.txt"):
        f = Path(wd) / name
        if not f.is_file():
            continue
        text = f.read_text(encoding = "utf-8", errors = "replace")
        if name == "trace.txt":
            keep = [l for l in text.splitlines() if any(k in l.casefold() for k in ("fail", "error", "denied", "0xc0", "fatal", "fork", "exit"))]
            text = "\n".join(keep[-120:] + ["--- last lines ---"] + text.splitlines()[-60:])
        print(f"--- {name} ({f.stat().st_size} bytes) ---", flush = True)
        print(text[-limit:], flush = True)


def raw(extra_root, argv):
    """Run argv as-is inside MXC (e.g. strace around bash) and dump diag.txt / trace.txt from the workdir."""
    wd = tempfile.mkdtemp(prefix = "shp-raw-")
    out, res, secs = run_in_mxc(argv[0], list(argv), wd, [extra_root])
    print(f"RAW exit={res.get('exitCode')} s={secs:.1f}", flush = True)
    print(out[-4000:], flush = True)
    dump_diag(wd, 12000)


if __name__ == "__main__":
    if sys.argv[1] == "compat":
        compat(sys.argv[2:])
    elif sys.argv[1] == "raw":
        raw(sys.argv[2], sys.argv[3:])
    else:
        container(sys.argv[2], sys.argv[3], sys.argv[4:])
