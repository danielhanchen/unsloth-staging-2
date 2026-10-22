#!/usr/bin/env python3
"""Host-side nvidia-smi reads through the nvsmi_cache shim, whatever PATH this process inherited.

The shim (nvsmi_cache/nvidia-smi: one real driver query per argument list at a time, short cache) only
helps callers that find it first on PATH, and sessions not started by `launcher.sh` (no .unsloth_path)
do not: their jobs and every child hit the driver directly, which is what queues dozens of real
nvidia-smi calls behind one driver lock. So host tools ask here instead:

    import nvsmi
    r = nvsmi.run(["--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], timeout=60)
    env = nvsmi.path_env(env)      # children (jobs, shells) find the shim too

The shim is located next to this file (scripts/nvsmi_cache in the repo, <workspace>/nvsmi_cache when
synced). The real binary is used only when the shim is missing or disabled (NVSMI_CACHE=0,
UNSLOTH_NVSMI_CACHE=0, NVSMI_CACHE_BYPASS=1), inside the shim's own fetcher (NVSMI_SHIM_ACTIVE), or once
when the shim could not be started or failed without timing out; a timeout is not retried (a second
real call would only queue behind the first).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHIM_MARKER = b"nvidia-smi shim: one real query per"  # nvsmi_cache/nvidia-smi docstring
FALLBACK_TIMEOUT_S = 30.0
_SYSTEM = ("/usr/bin/nvidia-smi", "/usr/local/bin/nvidia-smi")


def is_shim(path) -> bool:
    try:
        with open(path, "rb") as f:
            return SHIM_MARKER in f.read(4096)
    except OSError:
        return False


def _exe(p) -> bool:
    return bool(p) and os.path.isfile(p) and os.access(p, os.X_OK)


def disabled(env = None) -> bool:
    env = os.environ if env is None else env
    return (
        env.get("NVSMI_CACHE") == "0"
        or env.get("UNSLOTH_NVSMI_CACHE") == "0"
        or env.get("NVSMI_CACHE_BYPASS") == "1"
        or bool(env.get("NVSMI_SHIM_ACTIVE"))
    )


def shim_dir(env = None):
    """The nvsmi_cache dir holding the shim (repo or synced workspace layout), or None."""
    if disabled(env):
        return None
    for d in (HERE / "nvsmi_cache", HERE.parent / "nvsmi_cache"):
        p = d / "nvidia-smi"
        if _exe(str(p)) and is_shim(p):
            return d
    return None


def real(env = None):
    """First nvidia-smi on PATH that is not a copy of the shim, else a system path, else None."""
    env = os.environ if env is None else env
    for d in (env.get("PATH") or "").split(os.pathsep):
        p = os.path.join(d, "nvidia-smi") if d else ""
        if _exe(p) and not is_shim(p):
            return p
    return next((p for p in _SYSTEM if _exe(p) and not is_shim(p)), None)


def exe(env = None):
    d = shim_dir(env)
    return str(d / "nvidia-smi") if d else (real(env) or "nvidia-smi")


def run(
    args,
    timeout = 60.0,
    env = None,
    **kw,
):
    """subprocess.run([nvidia-smi, *args]) via the shim, text output captured. Raises
    subprocess.TimeoutExpired / OSError like subprocess.run when no binary answers."""
    kw.setdefault("capture_output", True)
    kw.setdefault("text", True)
    kw.setdefault("stdin", subprocess.DEVNULL)
    first = exe(env)
    via_shim = shim_dir(env) is not None
    try:
        r = subprocess.run([first, *args], timeout = timeout, env = env, **kw)
    except OSError:
        if not via_shim:
            raise
        r = None
    if r is not None and (r.returncode == 0 or not via_shim):
        return r
    other = real(env)
    if not other:
        if r is not None:
            return r
        raise FileNotFoundError("nvidia-smi: no real binary on PATH")
    return subprocess.run(
        [other, *args], timeout = min(float(timeout), FALLBACK_TIMEOUT_S), env = env, **kw
    )


def check_output(
    args,
    timeout = 60.0,
    env = None,
):
    r = run(args, timeout = timeout, env = env)
    if r.returncode:
        raise subprocess.CalledProcessError(r.returncode, ["nvidia-smi", *args], r.stdout, r.stderr)
    return r.stdout


def path_env(env = None):
    """A copy of `env` (default os.environ) with the shim dir first on PATH, so a child and its
    shells find the shim; unchanged when the shim is missing or disabled."""
    out = dict(os.environ if env is None else env)
    d = shim_dir(out)
    if d is None:
        return out
    parts = [p for p in (out.get("PATH") or "").split(os.pathsep) if p and p != str(d)]
    out["PATH"] = os.pathsep.join([str(d), *parts])
    return out


# ------------------------------------------------------------------ who bypasses the shim
def classify(procs):
    """procs: [(pid, ppid, comm, args)] (a ps listing). The real nvidia-smi processes, grouped by
    route: {"shim": n, "sampler": n, "bypass": {parent label: n}}. `shim` = started by the shim's
    fetcher, `sampler` = gpu_sample's single-flight fallback, anything else bypasses the shim."""
    by_pid = {p[0]: p for p in procs}
    out = {"shim": 0, "sampler": 0, "bypass": {}}
    for pid, ppid, comm, args in procs:
        if comm != "nvidia-smi" or "nvsmi_cache/nvidia-smi" in args:
            continue
        chain, cur = [], ppid
        for _ in range(4):
            p = by_pid.get(cur)
            if not p:
                break
            chain.append(p[3])
            cur = p[1]
        if any("NVSMI_TMP" in a or "nvsmi_cache/nvidia-smi" in a for a in chain):
            out["shim"] += 1
        elif (
            any("gpu_sample" in a for a in chain)
            or "--query-gpu=index,memory.total,memory.free,utilization.gpu" in args
            and " -i " not in f" {args} "
        ):
            out["sampler"] += 1
        else:
            top = next(
                (
                    a
                    for a in chain
                    if a and not a.startswith(("timeout ", "sh -c", "bash -c", "/bin/sh"))
                ),
                "",
            )
            label = _label(top) if top else "(parent gone)"
            out["bypass"][label] = out["bypass"].get(label, 0) + 1
    return out


def _label(args):
    words = args.split()
    for w in words:
        if w.endswith(".py") or w.endswith(".sh"):
            return os.path.basename(w)
    return os.path.basename(words[0]) if words else "?"


def ps_listing(run_ps = None):
    """[(pid, ppid, comm, args)] for every process, from /proc (no ps: it is slow at load 900)."""
    if run_ps is not None:
        return run_ps()
    out = []
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat", "rb") as f:
                st = f.read().decode(errors = "replace")
            comm = st[st.index("(") + 1 : st.rindex(")")]
            ppid = int(st[st.rindex(")") + 2 :].split()[1])
            with open(f"/proc/{d}/cmdline", "rb") as f:
                args = f.read().replace(b"\0", b" ").decode(errors = "replace").strip()
        except (OSError, ValueError):
            continue
        out.append((int(d), ppid, comm, args))
    return out


if __name__ == "__main__":
    import json
    import sys
    if sys.argv[1:] == ["--who"]:
        print(json.dumps(classify(ps_listing()), indent = 1, sort_keys = True))
    else:
        r = run(sys.argv[1:], timeout = float(os.environ.get("NVSMI_TIMEOUT_S", "60")))
        sys.stdout.write(r.stdout or "")
        sys.stderr.write(r.stderr or "")
        raise SystemExit(r.returncode)
