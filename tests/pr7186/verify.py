"""
PR #7186 verification: Studio gates nest_asyncio.apply() behind a running loop
so Python 3.14 CLI starts don't crash on anyio thread-pool hops.

nest_asyncio.apply() is an irreversible PROCESS-GLOBAL monkeypatch, so each check
runs in its OWN fresh child interpreter (via `--check X`) to avoid cross-contamination.
The parent orchestrates all four and aggregates. Exit 0 iff every check behaves as
the PR claims for the interpreter it runs under.

Checks:
  A. REPRO - with nest_asyncio.apply(): on 3.14, asyncio.current_task() goes None and
             anyio.to_thread.run_sync() raises TypeError (the failure the PR fixes);
             on <=3.13 it works fine (fix is version-specific).
  B. FIX   - plain-CLI branch (nest_asyncio NOT applied): current_task() is a real Task
             and anyio.to_thread.run_sync() works. Must PASS on every version.
  C. GATE  - the exact PR gate: no running loop -> except RuntimeError (skip);
             running loop -> else branch applies nest_asyncio.
  D. MCP   - call_tool_sync's asyncio.run() pattern in a CLEAN interpreter (no nest_asyncio):
             works off-loop in a plain threading.Thread (the real worker context), and
             raises on the loop thread (proving nest_asyncio was only for reentrancy, which
             the worker-thread design sidesteps).
"""
import sys
import subprocess
from importlib.metadata import version as _pkg_version


# ======================================================================= checks
def _check_A():
    """REPRO: apply nest_asyncio, probe current_task + anyio thread hop."""
    import asyncio
    import anyio
    import nest_asyncio
    nest_asyncio.apply()

    ct_repr = None
    hop = None

    async def main():
        nonlocal ct_repr
        ct_repr = repr(asyncio.current_task())
        await anyio.to_thread.run_sync(lambda: 1)

    try:
        asyncio.run(main())
        hop = "ok"
    except BaseException as e:  # noqa: BLE001
        hop = f"{type(e).__name__}: {e}"

    print(f"current_task={ct_repr}")
    print(f"threadhop={hop}")
    crashed = hop != "ok"
    on_314 = sys.version_info[:2] >= (3, 14)
    # PR claim: crash on 3.14, no crash on <=3.13.
    ok = crashed if on_314 else (not crashed)
    print("RESULT=" + ("PASS" if ok else "FAIL"))
    return ok


def _check_B():
    """FIX: plain-CLI branch (nest_asyncio NOT applied) must work everywhere."""
    import asyncio
    import anyio

    async def main():
        ct = asyncio.current_task()
        assert ct is not None, "current_task() None without nest_asyncio"
        assert await anyio.to_thread.run_sync(lambda: 21 * 2) == 42
        return ct

    ct = asyncio.run(main())
    ok = ct is not None
    print(f"current_task_real={ok}")
    print("RESULT=" + ("PASS" if ok else "FAIL"))
    return ok


def _check_C():
    """GATE: byte-for-byte the PR gate, in both loop states."""
    import asyncio

    def gate(flag):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return          # plain CLI -> skip
        else:
            flag[0] = True  # Colab/IPython -> would apply nest_asyncio

    no_loop = [False]
    gate(no_loop)

    in_loop = [False]

    async def inside():
        gate(in_loop)

    asyncio.run(inside())

    ok = (no_loop[0] is False) and (in_loop[0] is True)
    print(f"skip_without_loop={no_loop[0] is False}  apply_with_loop={in_loop[0] is True}")
    print("RESULT=" + ("PASS" if ok else "FAIL"))
    return ok


def _check_D():
    """MCP: call_tool_sync's asyncio.run() pattern in a CLEAN interpreter."""
    import asyncio
    import threading

    async def _tool():
        await asyncio.sleep(0)
        return "tool-result"

    def call_tool_sync_like():
        return asyncio.run(_tool())  # exactly mcp_client.call_tool_sync HTTP path

    # (i) worker-thread context (no running loop) -> works, no nest_asyncio needed.
    box = {}

    def worker():
        try:
            box["r"] = call_tool_sync_like()
        except BaseException as e:  # noqa: BLE001
            box["r"] = f"ERR {type(e).__name__}: {e}"

    t = threading.Thread(target=worker, name="tool-exec-like")
    t.start()
    t.join()
    worker_ok = box.get("r") == "tool-result"

    # (ii) on the event-loop thread -> must raise (the reentrancy nest_asyncio patched).
    async def on_loop():
        try:
            call_tool_sync_like()
            return "NO-RAISE"
        except RuntimeError as e:
            return f"raised: {type(e).__name__}"

    on_loop_res = asyncio.run(on_loop())
    on_loop_raises = on_loop_res.startswith("raised:")

    print(f"worker_thread_run={'tool-result OK' if worker_ok else box.get('r')}")
    print(f"on_loop_run={on_loop_res}")
    ok = worker_ok and on_loop_raises
    print("RESULT=" + ("PASS" if ok else "FAIL"))
    return ok


def _check_E():
    """REAL CODE: extract the exact gate block from studio/backend/run.py and exec it
    with a spy nest_asyncio, proving the shipped code applies iff a loop is running.
    Faithful to the PR (no paraphrase). Path via $RUN_PY or the repo default."""
    import os
    import re
    import textwrap
    import types
    import asyncio

    run_py = os.environ.get(
        "RUN_PY",
        os.path.join(os.path.dirname(__file__), "..", "..", "unsloth",
                     "studio", "backend", "run.py"),
    )
    src = open(run_py, encoding="utf-8").read()

    # Grab the try/except-RuntimeError/else gate that guards nest_asyncio.apply().
    # Capture the leading indentation of the `try:` line (it lives inside run_server,
    # indented 4 spaces) so textwrap.dedent has a common prefix to strip.
    m = re.search(
        r"(^[ \t]*try:\s*\n[ \t]*asyncio\.get_running_loop\(\).*?nest_asyncio\.apply\(\))",
        src, re.DOTALL | re.MULTILINE,
    )
    if not m:
        print("gate_found=False")
        print("RESULT=FAIL")
        return False
    gate = textwrap.dedent(m.group(1))
    print("gate_found=True")
    for line in gate.splitlines():
        print("   | " + line)

    def run_gate():
        spy = {"applied": 0}
        fake = types.ModuleType("nest_asyncio")
        fake.apply = lambda *a, **k: spy.__setitem__("applied", spy["applied"] + 1)
        import sys as _sys
        saved = _sys.modules.get("nest_asyncio")
        _sys.modules["nest_asyncio"] = fake
        try:
            exec(compile(gate, run_py, "exec"), {"asyncio": asyncio})
        finally:
            if saved is not None:
                _sys.modules["nest_asyncio"] = saved
            else:
                _sys.modules.pop("nest_asyncio", None)
        return spy["applied"]

    # (i) no running loop -> real gate must NOT apply nest_asyncio.
    applied_noloop = run_gate()

    # (ii) inside a running loop -> real gate MUST apply nest_asyncio exactly once.
    box = {}

    async def inside():
        box["applied"] = run_gate()

    asyncio.run(inside())

    ok = (applied_noloop == 0) and (box["applied"] == 1)
    print(f"applied_without_loop={applied_noloop}  applied_with_loop={box['applied']}")
    print("RESULT=" + ("PASS" if ok else "FAIL"))
    return ok


_CHECKS = {"A": _check_A, "B": _check_B, "C": _check_C, "D": _check_D, "E": _check_E}


# ==================================================================== orchestrate
_DESC = {
    "A": "REPRO 3.14 crash under nest_asyncio (no crash on <=3.13)",
    "B": "FIX branch works (current_task real + anyio hop)",
    "C": "gate logic: skip(no loop) + apply(loop)",
    "D": "MCP asyncio.run off-loop OK + on-loop raises",
    "E": "REAL run.py gate: applies nest_asyncio iff loop running",
}


def _orchestrate():
    py = sys.version.split()[0]
    print("=" * 72)
    print(f"PR #7186 verify  |  Python {py}  |  anyio {_pkg_version('anyio')}"
          f"  |  nest_asyncio {_pkg_version('nest_asyncio')}")
    print("=" * 72)

    all_ok = True
    for k in ("A", "B", "C", "D", "E"):
        try:
            p = subprocess.run([sys.executable, __file__, "--check", k],
                               capture_output=True, text=True, timeout=60)
            stdout, stderr, rc = p.stdout, p.stderr, p.returncode
        except subprocess.TimeoutExpired as e:
            stdout = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
            stderr = "TIMEOUT after 60s"
            rc = -1
        passed = "RESULT=PASS" in stdout
        all_ok = all_ok and passed
        print(f"\n[{k}] {_DESC[k]}")
        for line in stdout.strip().splitlines():
            if not line.startswith("RESULT="):
                print(f"     {line}")
        if rc != 0 and stderr.strip():
            print("     STDERR:", stderr.strip().splitlines()[-1])
        print(f"     -> {'PASS' if passed else 'FAIL'}")

    print("\n" + "=" * 72)
    print(f"RESULT: {'ALL PASS' if all_ok else 'FAILURE'}  (Python {py})")
    print("=" * 72)
    return all_ok


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--check":
        import os
        ok = _CHECKS[sys.argv[2]]()
        sys.stdout.flush()
        # nest_asyncio + anyio's cached thread pool can deadlock loop shutdown on
        # some versions AFTER the answer is already computed; hard-exit to skip
        # atexit/thread-join cleanup (the check result stands on the printed lines).
        os._exit(0 if ok else 1)
    sys.exit(0 if _orchestrate() else 1)
