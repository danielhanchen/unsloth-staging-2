"""A cell's memory budget: a smaller card simulated on a bigger one, and a host-RAM cap over the process tree. Ported
from the outside tester's harness (ballast.py, simtotal/sitecustomize.py, bench.mem_capped).

Cell options:
  budget: {"vram_gib": G}          simulate a G GiB card: a VRAM ballast leaves G GiB (minus the idle baseline) free,
                                   and DBENCH_SIM_TOTAL_MIB makes torch.cuda.mem_get_info report G GiB total in every
                                   stack process (BlineUI / Studio servers through the PYTHONPATH shim, in-process
                                   Studio through dbench_hooks.install()). Recorded as memory.budget.simulated = true.
          {"free_gib": F}          ballast only (leave F GiB free), total untouched
          {"sim_total_gib": T}     total cap only
          {"mem_max": "60G"}       host RAM cap (cgroup MemoryMax, swap 0) over run_cell and every child it starts:
                                   run_cell re-executes itself under `systemd-run --user --scope`; where that is not
                                   available (no systemd user manager, macOS, Windows) the cap is recorded as not
                                   applied, never silently assumed
  options.target.simulated_budget_gib / cell budget_gib (set by the scheduler) is the same as budget.vram_gib.

A simulated budget is never a real card: reports label it, and the ballast's own context is outside the stack.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
SHIM = HERE / "shim"
SCOPE_MARK = "DBENCH_IN_SCOPE"


def spec(cell: dict) -> dict:
    """The resolved budget for a cell: {vram_gib, free_gib, sim_total_gib, mem_max} (None = not asked)."""
    opts = cell.get("options") or {}
    b = dict(opts.get("budget") or {})
    # options.target.simulated_budget_gib (plan.py) and the cell's budget_gib (plan.py budget arms) name the same
    # simulated VRAM budget
    sim = (opts.get("target") or {}).get("simulated_budget_gib") or cell.get("budget_gib")
    if sim and not b.get("vram_gib"):
        b["vram_gib"] = sim
    out = {"vram_gib": b.get("vram_gib"), "free_gib": b.get("free_gib"), "sim_total_gib": b.get("sim_total_gib"),
           "mem_max": b.get("mem_max") or os.environ.get("DBENCH_MEM_MAX")}
    if out["vram_gib"]:
        out["sim_total_gib"] = out["sim_total_gib"] or out["vram_gib"]
    return out


def simulated(b: dict) -> bool:
    return bool(b.get("vram_gib") or b.get("free_gib") or b.get("sim_total_gib"))


# ---------------------------------------------------------------------------------------------- env for stacks
def child_env(env: dict, b: Optional[dict] = None, parity_dump: Optional[str] = None) -> dict:
    """``env`` plus the shim on PYTHONPATH and the budget / parity variables, for a stack process (BlineUI server,
    Studio server). DBENCH_PYTHONPATH_PREFIX carries the shim through launchers that set their own PYTHONPATH."""
    env = dict(env)
    b = b or {}
    if b.get("sim_total_gib"):
        env["DBENCH_SIM_TOTAL_MIB"] = str(int(float(b["sim_total_gib"]) * 1024))
    if parity_dump:  # absolute: a stack server may run with its own cwd (the baseline UI runs in its checkout)
        env["DBENCH_PARITY_DUMP"] = os.path.abspath(str(parity_dump))
    if env.get("DBENCH_SIM_TOTAL_MIB") or env.get("DBENCH_PARITY_DUMP"):
        sep = os.pathsep
        env["PYTHONPATH"] = str(SHIM) + (sep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env["DBENCH_PYTHONPATH_PREFIX"] = str(SHIM)
    return env


def install_in_process(b: Optional[dict] = None, parity_dump: Optional[str] = None) -> dict:
    """The same hooks in this process (in-process Studio): set the env, then dbench_hooks.install()."""
    b = b or {}
    if b.get("sim_total_gib"):
        os.environ["DBENCH_SIM_TOTAL_MIB"] = str(int(float(b["sim_total_gib"]) * 1024))
    if parity_dump:
        os.environ["DBENCH_PARITY_DUMP"] = os.path.abspath(str(parity_dump))
    if not (os.environ.get("DBENCH_SIM_TOTAL_MIB") or os.environ.get("DBENCH_PARITY_DUMP")):
        return {}
    if str(SHIM) not in sys.path:
        sys.path.insert(0, str(SHIM))
    import dbench_hooks

    return dbench_hooks.install()


# ---------------------------------------------------------------------------------------------- ballast
class Ballast:
    """A child process holding VRAM so that ``free_mib`` remains free on the device. ``python`` must import torch
    (the cell's venv usually does; BlineUI cells pass the BlineUI venv). Killed by stop()."""

    def __init__(self, free_mib: int, device: Optional[str], python: Optional[str] = None, timeout_s: float = 300):
        self.free_mib, self.device = int(free_mib), device
        self.python = python or sys.executable
        self.proc: Optional[subprocess.Popen] = None
        self.got_free_mib: Optional[float] = None
        self.timeout_s = timeout_s

    def start(self) -> "Ballast":
        env = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
        if self.device is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self.device)
        env.pop("DBENCH_SIM_TOTAL_MIB", None)
        self.proc = subprocess.Popen([self.python, "-u", str(HERE / "ballast.py"), "--free-mib", str(self.free_mib),
                                      "--device", str(self.device if self.device is not None else 0),
                                      "--parent-pid", str(os.getpid())],
                                     stdout = subprocess.PIPE, stderr = subprocess.STDOUT, text = True, env = env)
        # readline() blocks: read on a thread so the deadline really fires on a ballast that never prints READY
        import queue
        import threading

        q: queue.Queue = queue.Queue()

        def pump(stream):
            for ln in stream:
                q.put(ln)
            q.put(None)

        threading.Thread(target = pump, args = (self.proc.stdout,), daemon = True).start()
        deadline = time.time() + self.timeout_s
        lines = []
        while True:
            left = deadline - time.time()
            if left <= 0:
                break
            try:
                line = q.get(timeout = left)
            except queue.Empty:
                break
            if line is None:
                break
            lines.append(line.strip())
            if line.startswith("READY"):
                self.got_free_mib = float(line.split()[1])
                return self
        self.stop()
        raise RuntimeError(f"VRAM ballast did not reach {self.free_mib} MiB free within {self.timeout_s:.0f}s: "
                           f"{lines[-5:]}")

    @property
    def pid(self) -> Optional[int]:
        return self.proc.pid if self.proc else None

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout = 30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout = 10)


def ballast_target_mib(b: dict, idle_used_mib: Optional[float]) -> Optional[int]:
    """Free MiB the ballast should leave: a G GiB card minus what an idle card of that size already holds (taken as
    this card's idle reading); or the explicit free_gib."""
    if b.get("free_gib"):
        return int(float(b["free_gib"]) * 1024)
    if b.get("vram_gib"):
        return max(256, int(float(b["vram_gib"]) * 1024 - (idle_used_mib or 0)))
    return None


# ---------------------------------------------------------------------------------------------- host RAM cap
def scope_available() -> tuple:
    """(ok, why) for `systemd-run --user --scope`: Linux with a reachable user manager."""
    if sys.platform != "linux":
        return False, f"no cgroup scope on {sys.platform}"
    if not shutil.which("systemd-run"):
        return False, "systemd-run not found"
    try:  # probe with the same properties the real scope uses (a manager may refuse MemoryMax)
        r = subprocess.run(["systemd-run", "--user", "--scope", "--quiet", "-p", "MemoryMax=64G", "-p",
                            "MemorySwapMax=0", "true"], capture_output = True, text = True, timeout = 20)
    except Exception as exc:  # noqa: BLE001
        return False, f"systemd-run failed: {exc}"
    if r.returncode != 0:
        return False, f"systemd-run --user unavailable: {(r.stderr or r.stdout).strip()[:200]}"
    return True, "ok"


def cgroup_memory_max() -> Optional[str]:
    """This process's cgroup v2 memory.max ("max" = no cap), None when it cannot be read."""
    try:
        line = next(ln for ln in Path("/proc/self/cgroup").read_text().splitlines() if ln.startswith("0::"))
        return (Path("/sys/fs/cgroup") / line[3:].lstrip("/") / "memory.max").read_text().strip()
    except Exception:  # noqa: BLE001
        return None


def scoped_argv(argv: list, mem_max: str) -> list:
    """``argv`` run inside a user scope capped at ``mem_max`` (no swap). systemd-run --scope execs the command in
    place, so the pid a caller holds stays the cell's pid and every child inherits the cgroup."""
    return ["systemd-run", "--user", "--scope", "--quiet", "-p", f"MemoryMax={mem_max}", "-p", "MemorySwapMax=0",
            *argv]


def maybe_reexec_scoped(b: dict, argv: list) -> dict:
    """Called first thing in run_cell: re-exec under a capped scope when the budget asks for mem_max and this
    process is not already inside one. Returns the record's mem_max block when it does not exec."""
    want = b.get("mem_max")
    if not want or str(want).lower() in ("0", "off", "none"):
        return {"mem_max": None, "applied": False, "why": "not asked"}
    if os.environ.get(SCOPE_MARK):
        # the marker only says we re-exec'd: read the cap the cgroup really has before calling it applied
        got = cgroup_memory_max()
        return {"mem_max": want, "applied": got not in (None, "max"), "memory_max_read": got,
                "how": "systemd-run --user --scope (whole run_cell tree)"}
    ok, why = scope_available()
    if not ok:
        return {"mem_max": want, "applied": False, "why": why}
    env = {**os.environ, SCOPE_MARK: str(want)}
    sys.stdout.flush()
    sys.stderr.flush()
    os.execvpe("systemd-run", scoped_argv(argv, str(want)), env)
    return {}  # not reached


def record_block(b: dict, ballast: Optional[Ballast], mem: dict, idle_used_mib: Optional[float]) -> dict:
    """memory.budget for the record."""
    gib = b.get("vram_gib") or b.get("sim_total_gib") or (b.get("free_gib"))
    return {"gib": gib, "simulated": simulated(b), "vram_gib": b.get("vram_gib"), "free_gib": b.get("free_gib"),
            "sim_total_mib": int(float(b["sim_total_gib"]) * 1024) if b.get("sim_total_gib") else None,
            "ballast_free_mib": ballast.got_free_mib if ballast else None,
            "ballast_target_mib": ballast.free_mib if ballast else None, "idle_used_mib": idle_used_mib, **mem}
