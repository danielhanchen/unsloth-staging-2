"""Harness only: run #11390's capability probe and print what it discards."""
import sys

from core.inference import mxc_adapter, sandbox_windows_mxc as backend

_spawn = mxc_adapter.spawn


def spawn(*args, **kwargs):
    proc = _spawn(*args, **kwargs)
    communicate = proc.communicate

    def wrapped(*a, **k):
        out = communicate(*a, **k)
        text = out[0] if out and out[0] else ""
        print("WXC OUTPUT >>>")
        print(text[-4000:])
        print("<<< WXC OUTPUT")
        return out

    proc.communicate = wrapped
    return proc


_result = mxc_adapter.completion_result


def completion_result(proc):
    result = _result(proc)
    print("wxc returncode =", proc.returncode, "completion =", result)
    return result


_abort = mxc_adapter.abort


def abort(proc, **kwargs):
    _abort(proc, **kwargs)
    try:
        out = proc.communicate(timeout = 10)[0] or ""
    except Exception as exc:  # noqa: BLE001 - harness diagnostics only
        out = f"<could not read after abort: {exc}>"
    print("WXC OUTPUT AFTER ABORT >>>")
    print(out[-4000:])
    print("<<< WXC OUTPUT AFTER ABORT")


mxc_adapter.spawn = spawn
mxc_adapter.abort = abort
mxc_adapter.completion_result = completion_result
kind = sys.argv[1] if len(sys.argv) > 1 else "python"
exe = sys.executable if kind == "python" else (sys.argv[2] if len(sys.argv) > 2 else "cmd.exe")
print("probing", kind, exe)
import time
_started = time.monotonic()
cap = backend.capability_snapshot(force = True, execution_kind = kind, selected_executable = exe)
print(f"probe_seconds = {time.monotonic() - _started:.1f}")
print("available =", cap.available)
print("reason =", cap.reason)
