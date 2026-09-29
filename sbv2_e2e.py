"""Harness only: OS sandbox setup end to end through the real routes, job, probes and tools.

Auth dependencies are overridden to the installation owner exactly as the unit tests do; the setup plan,
the setup job (real sudo on Linux runners, real UAC-free elevated branch on Windows runners), the probes
and the tools are real. Payloads are harmless.
Usage: python sbv2_e2e.py
"""

import os
import sys
import time

sys.path.insert(0, "studio/backend")
os.environ.pop("UNSLOTH_MXC_ALLOW_DACL_FALLBACK", None)
os.environ.pop("UNSLOTH_MXC_PERSISTENT_READ_GRANTS", None)

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import routes.settings as settings  # noqa: E402
from routes.sandbox_capability import router as capability_router  # noqa: E402
from core.inference import tools  # noqa: E402
from state import tool_policy  # noqa: E402
from utils.account_context import OWNER, bind_account, reset_account  # noqa: E402

RISKY = "import os\nos.system('sudo true')\nprint('SB_OK')"


def client(host):
    app = FastAPI()
    app.include_router(settings.router, prefix = "/api/settings")
    app.include_router(capability_router, prefix = "/api/sandbox")

    async def subject():
        token = bind_account(OWNER)
        try:
            yield OWNER.username
        finally:
            reset_account(token)

    app.dependency_overrides[settings.get_current_subject] = subject
    app.dependency_overrides[settings.authenticated_via_api_key] = lambda: False
    return TestClient(app, base_url = f"http://{host}:8888", client = (host, 50000), raise_server_exceptions = False)


def show(label, c):
    body = c.get("/api/settings/sandbox", params = {"refresh": "true"}).json()
    cap = c.get("/api/sandbox/capability", params = {"refresh": "1"}).json()
    setup = body.get("setup") or {}
    print(
        f"STATUS {label} python={body['python']['backend']}/{body['python']['available']} "
        f"terminal={body['terminal']['backend']}/{body['terminal']['available']} "
        f"setup_action={setup.get('action')} elevation={setup.get('elevation')} can_run={setup.get('can_run')} "
        f"needs_consent={setup.get('needs_consent')}",
        flush = True,
    )
    print(f"SETUP_CMD {label} {setup.get('manual_command')!r} reason={setup.get('reason')!r}", flush = True)
    print(f"REMEDIATION {label} {body['python'].get('remediation', '')[:300]!r}", flush = True)
    print(f"TERMINAL {label} shell={body.get('terminal_shell')} reason={body['terminal'].get('reason')!r} remediation={body['terminal'].get('remediation', '')[:300]!r}", flush = True)
    print(f"CAPABILITY {label} {cap}", flush = True)
    return body, cap


def run_tool(label, tool, payload):
    tools._last_tool_execution_record = None
    key = "code" if tool == "python" else "command"
    started = time.monotonic()
    out = tools.execute_tool(tool, {key: payload}, session_id = "__LOCALID_sbv2_e2e", timeout = 150) or ""
    rec = tools._last_tool_execution_record
    print(
        f"TOOL {label} {tool} mode={getattr(rec, 'effective_mode', None)} backend={getattr(rec, 'backend', None)} "
        f"s={time.monotonic() - started:.2f} ok={'SB_OK' in out} | {out.strip()[-200:]!r}",
        flush = True,
    )


def gate(label):
    # Wait for the cached capability the "off" gate reads to settle after a probe.
    from core.inference import os_sandbox

    deadline = time.monotonic() + 120
    while time.monotonic() < deadline and (
        os_sandbox.cached_tool_isolation("python") is None or os_sandbox.cached_tool_isolation("terminal") is None
    ):
        time.sleep(1)
    for name, args in (("python", {"code": RISKY}), ("python", {"code": "print(1)"}), ("terminal", {"command": "sudo true"})):
        decision = tool_policy.needs_tool_confirmation(
            confirm_tool_calls = True,
            bypass_permissions = False,
            permission_mode = "off",
            name = name,
            arguments = args,
        )
        print(
            f"GATE {label} off {name} {'risky' if 'sudo' in str(args) else 'benign'} "
            f"isolated={os_sandbox.cached_tool_isolation(name)} prompts={decision}",
            flush = True,
        )


def main():
    local = client("127.0.0.1")
    with local:
        body, cap = show("initial", local)
        run_tool("initial", "python", "print('SB_OK')")
        run_tool("initial", "terminal", "echo SB_OK")
        gate("initial")
        setup = body.get("setup") or {}
        operation = setup.get("action")
        if not operation:
            print("NO_SETUP_OFFERED", flush = True)
            return
        with client("203.0.113.9") as remote:
            r = remote.post("/api/settings/sandbox/setup", json = {"operation": operation})
            print(f"REMOTE_SETUP http={r.status_code} {r.text[:200]}", flush = True)
            rcap = remote.get("/api/sandbox/capability").json()
            print(f"REMOTE_CAPABILITY action={rcap.get('setup_action')} can_run={rcap.get('can_run_setup')} cmd={rcap.get('manual_command')!r}", flush = True)
        payload = {"operation": operation}
        if setup.get("needs_consent"):
            payload["consent_dacl_fallback"] = True
        started = time.monotonic()
        r = local.post("/api/settings/sandbox/setup", json = payload)
        print(f"SETUP_START http={r.status_code} {r.text[:300]}", flush = True)
        job = r.json() if r.status_code == 200 else {}
        deadline = time.monotonic() + 1500
        while job.get("state") == "running" and time.monotonic() < deadline:
            time.sleep(3)
            job = local.get("/api/settings/sandbox/setup").json()
        print(
            f"SETUP_DONE state={job.get('state')} exit={job.get('exit_code')} s={time.monotonic() - started:.1f} "
            f"steps={job.get('steps')} note={job.get('note')!r} tail={job.get('output_tail', [])[-4:]}",
            flush = True,
        )
        show("after_setup", local)
        run_tool("after_setup", "python", "print('SB_OK')")
        run_tool("after_setup", "terminal", "echo SB_OK")
        gate("after_setup")
        if sys.platform == "win32":
            from core.inference import os_sandbox as _os

            for tick in range(6):
                bash = tools._windows_bash()
                for label, exe in (("bash", bash), ("cmd", tools._windows_system_cmd())):
                    if not exe:
                        continue
                    cap = _os.capability_snapshot(execution_kind = "terminal", selected_executable = exe)
                    print(f"WIN_TERMINAL t={tick * 20}s {label} available={cap.available} reason={cap.reason!r}", flush = True)
                print(f"WIN_PROFILE t={tick * 20}s {tools._terminal_profile(False)}", flush = True)
                time.sleep(20)
            show("after_wait", local)
            run_tool("after_wait", "terminal", "git --version && echo SB_OK")
        r = local.post("/api/settings/sandbox/setup", json = payload)
        print(f"SETUP_AGAIN http={r.status_code} {r.text[:200]}", flush = True)


if __name__ == "__main__":
    main()
