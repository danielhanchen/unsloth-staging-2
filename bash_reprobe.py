"""Harness only: the Git Bash Terminal capability and an auto Terminal call, three times across the negative TTL."""
import sys
import time

sys.path.insert(0, "studio/backend")
from core.inference import sandbox_windows_mxc, tools  # noqa: E402

bash = tools._windows_bash()
print("BASH", bash)
for step in range(3):
    started = time.monotonic()
    cap = sandbox_windows_mxc.capability_snapshot(execution_kind = "terminal", selected_executable = bash)
    snap = time.monotonic() - started
    started = time.monotonic()
    out = tools.execute_tool("terminal", {"command": "echo REPROBE_OK"}, session_id = "__LOCALID_reprobe", timeout = 120)
    call = time.monotonic() - started
    rec = tools._last_tool_execution_record
    print(
        f"STEP {step} snapshot_s={snap:.2f} auto_call_s={call:.2f} ok={'REPROBE_OK' in (out or '')} "
        f"mode={rec.effective_mode if rec else None}"
    )
    print(f"STEP {step} reason={cap.reason}")
    print(f"STEP {step} remediation={cap.remediation}")
    if step < 2:
        time.sleep(35)
started = time.monotonic()
out = tools.execute_tool(
    "terminal", {"command": "echo hi"}, session_id = "__LOCALID_reprobe", timeout = 120,
    tool_execution_mode = "required",
)
print(f"REQUIRED s={time.monotonic() - started:.2f} out={out[:600]!r}")
