"""Diagnostic for PR 10285's Windows LPAC backend on a hosted runner.

Runs the PR's own WindowsLpacBackend.prepare() and launcher against several
payload/policy variants so the STATUS_ACCESS_DENIED seen in CI can be
attributed: console app vs GUI app, LPAC vs plain AppContainer, cmd.exe vs
python.exe. Observes only; prints and writes JSON.
"""
import json, os, subprocess, sys, tempfile, traceback, time
sys.path.insert(0, os.getcwd())
from core.inference import windows_lpac as W
from core.inference.os_sandbox import ToolLaunchPlan, capability_snapshot

OUT = os.path.join(os.environ.get("RUNNER_TEMP", tempfile.gettempdir()), "lpac_diag.json")
res = {"python": sys.executable, "version": sys.version, "platform": __import__("platform").platform()}

t0 = time.time()
try:
    cap = capability_snapshot(force=True)
    res["capability"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in cap.__dict__.items()}
except Exception:
    res["capability_error"] = traceback.format_exc()[-2000:]
res["capability_seconds"] = round(time.time() - t0, 2)

def icacls(path):
    try:
        return subprocess.run(["icacls", path], capture_output=True, text=True, timeout=30).stdout[-1500:]
    except Exception as exc:
        return f"error: {exc}"

pydir = os.path.dirname(sys.executable)
res["acl_before"] = {"python_dir": icacls(pydir), "kernel32": icacls(r"C:\Windows\System32\kernel32.dll"),
                     "python_exe": icacls(sys.executable)}
res["needs_explicit_acl_python_dir"] = W._needs_explicit_acl(pydir)

def variant(name, argv, opt_out=True, check_file=None, timeout=90):
    W._PROCESS_CREATION_ALL_APPLICATION_PACKAGES_OPT_OUT = 0x1 if opt_out else 0x0
    wd = tempfile.mkdtemp(prefix="lpacdiag-")
    argv = tuple(a.replace("__WD__", wd) for a in argv)
    r = {"argv": list(argv), "lpac_opt_out": opt_out, "workdir": wd}
    prepared = None
    try:
        backend = W.WindowsLpacBackend()
        prepared = backend.prepare(ToolLaunchPlan(argv=argv, workdir=wd, env={"PYTHONIOENCODING": "utf-8"}))
        r["runtime_roots_acl"] = icacls(pydir)[-600:]
        proc = prepared.spawn_callback(prepared, {
            "stdout": subprocess.PIPE, "stderr": subprocess.STDOUT, "stdin": subprocess.DEVNULL,
            "text": True, "encoding": "utf-8", "errors": "replace", "cwd": prepared.workdir,
            "env": prepared.env, "close_fds": True, "creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)})
        proc.wait(timeout=timeout)
        out = proc.stdout.read()
        r["rc"] = proc.returncode
        r["rc_hex"] = hex(proc.returncode & 0xFFFFFFFF)
        r["out"] = out[-800:]
        if check_file:
            r["marker_file_exists"] = os.path.exists(os.path.join(wd, check_file))
    except Exception as exc:
        r["error"] = f"{type(exc).__name__}: {exc}"
        r["tb"] = traceback.format_exc()[-1500:]
    finally:
        if prepared is not None:
            try:
                prepared.cleanup()
            except Exception as exc:
                r["cleanup_error"] = str(exc)
    res.setdefault("variants", {})[name] = r
    print(name, json.dumps({k: r.get(k) for k in ("rc", "rc_hex", "error", "marker_file_exists")}), flush=True)

comspec = os.environ.get("COMSPEC", r"C:\Windows\System32\cmd.exe")
pythonw = os.path.join(pydir, "pythonw.exe")
variant("cmd_exit0_lpac", (comspec, "/c", "exit 0"))
variant("cmd_echo_lpac", (comspec, "/c", "echo cmd-ok"))
variant("python_print_lpac", (sys.executable, "-I", "-S", "-c", "print('py-ok')"))
if os.path.exists(pythonw):
    variant("pythonw_marker_lpac", (pythonw, "-I", "-S", "-c", "open(r'__WD__\\marker.txt','w').write('ok')"), check_file="marker.txt")
variant("cmd_echo_appcontainer_no_lpac", (comspec, "/c", "echo cmd-ok"), opt_out=False)
variant("python_print_appcontainer_no_lpac", (sys.executable, "-I", "-S", "-c", "print('py-ok')"), opt_out=False)
W._PROCESS_CREATION_ALL_APPLICATION_PACKAGES_OPT_OUT = 0x1

with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=2, default=str)
print("WROTE", OUT)
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ("rc_hex", "error", "marker_file_exists")} for k, v in res.get("variants", {}).items()}, indent=1))
