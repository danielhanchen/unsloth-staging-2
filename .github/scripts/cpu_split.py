"""CPU isolation for two arms running at once on one machine (a Kaggle T4x2 kernel: base on GPU0,
head on GPU1). Stdlib only: shipped with remote jobs (offload.job_files ships it flat next to ab_remote.py;
remote_studio's bundle picks it up as an import of remote_driver.py).

split(n) -> n disjoint, equal-size CPU lists made of whole physical cores (hyperthread siblings stay
together, from /sys/devices/system/cpu/cpu*/topology/thread_siblings_list), or None when the machine
has fewer than n physical cores in this process's affinity. A core left over by an odd count is
given to neither arm, so both arms get the same CPU budget. Without topology files the affinity set
is split evenly by CPU number (each arm still needs at least one CPU).

isolate(cpus, name) -> a preexec_fn for subprocess: sched_setaffinity on the child, and a cgroup v2
cpuset child group when one can be written here (probe_cgroup), else affinity alone.
pick_idle(n) -> n of the least busy allowed CPUs, one per physical core (pr_ui_diff's frontend build cap).
thread_env(n) -> the per-arm thread caps (OpenMP / MKL / OpenBLAS / Rayon / torch / tokenizers).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

SYS_CPU = "/sys/devices/system/cpu"
CGROUP_ROOT = "/sys/fs/cgroup"


def parse_list(text):
    """'0-3,8,10-11' -> [0, 1, 2, 3, 8, 10, 11]."""
    out = []
    for part in (text or "").strip().split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def fmt_list(cpus):
    """[0, 1, 2, 5] -> '0-2,5' (cgroup cpuset.cpus syntax)."""
    cpus = sorted(cpus)
    runs, start, prev = [], None, None
    for c in cpus:
        if start is None:
            start = prev = c
        elif c == prev + 1:
            prev = c
        else:
            runs.append((start, prev))
            start = prev = c
    if start is not None:
        runs.append((start, prev))
    return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in runs)


def allowed():
    try:
        return sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return list(range(os.cpu_count() or 1))


def physical_cores(cpus, sys_cpu = SYS_CPU):
    """Allowed CPUs grouped by physical core, in core order; None when no topology is readable."""
    seen, cores = set(), []
    for c in cpus:
        if c in seen:
            continue
        try:
            sib = parse_list(
                Path(sys_cpu, f"cpu{c}", "topology", "thread_siblings_list").read_text()
            )
        except (OSError, ValueError):
            return None
        group = [x for x in sib if x in cpus] or [c]
        seen.update(group)
        cores.append(group)
    return cores


def split(
    n = 2,
    cpus = None,
    sys_cpu = SYS_CPU,
):
    """n disjoint equal CPU lists by whole physical cores, or None (fewer than n cores: run serially)."""
    cpus = sorted(cpus if cpus is not None else allowed())
    cores = physical_cores(cpus, sys_cpu)
    if cores is None:  # no topology: an even split of the CPU numbers
        per = len(cpus) // n
        if per < 1:
            return None
        return [cpus[i * per : (i + 1) * per] for i in range(n)]
    per = len(cores) // n
    if per < 1:
        return None
    return [sorted(c for core in cores[i * per : (i + 1) * per] for c in core) for i in range(n)]


def _cpu_times(stat = "/proc/stat"):
    """{cpu: (busy jiffies, total jiffies)} from /proc/stat; {} when unreadable."""
    out = {}
    try:
        lines = Path(stat).read_text().splitlines()
    except OSError:
        return out
    for line in lines:
        m = re.match(r"cpu(\d+)\s+(.*)", line)
        if not m:
            continue
        vals = [int(v) for v in m.group(2).split()[:8]]
        idle = sum(vals[3:5])  # idle + iowait
        out[int(m.group(1))] = (sum(vals) - idle, sum(vals))
    return out


def pick_idle(
    n,
    cpus = None,
    sample_s = 0.2,
    stat = "/proc/stat",
    sys_cpu = SYS_CPU,
    rng = None,
):
    """n CPUs of `cpus` (default: this process's affinity) that were least busy over `sample_s`, one
    per physical core while there are enough cores. Picked at random among the idlest 2n so builds
    starting together do not all land on the same CPUs. All of `cpus` when n covers them."""
    import random
    import time

    cpus = sorted(cpus if cpus is not None else allowed())
    if n >= len(cpus):
        return cpus
    rng = rng or random.Random()
    a = _cpu_times(stat)
    if sample_s:
        time.sleep(sample_s)
    b = _cpu_times(stat)

    def busy(c):
        if c not in a or c not in b or b[c][1] <= a[c][1]:
            return 1.0
        return (b[c][0] - a[c][0]) / (b[c][1] - a[c][1])

    order = sorted(cpus, key = lambda c: (busy(c), rng.random()))
    pool = order[: 2 * n]
    rng.shuffle(pool)
    pool += order[len(pool) :]
    core_of = {}
    for group in physical_cores(cpus, sys_cpu) or [[c] for c in cpus]:
        for c in group:
            core_of[c] = group[0]
    picked, cores = [], set()
    for c in pool:
        if core_of.get(c, c) not in cores:
            picked.append(c)
            cores.add(core_of.get(c, c))
        if len(picked) == n:
            return sorted(picked)
    picked += [c for c in pool if c not in picked][: n - len(picked)]
    return sorted(picked)


def thread_env(n):
    n = str(max(1, int(n)))
    return {
        "OMP_NUM_THREADS": n,
        "MKL_NUM_THREADS": n,
        "OPENBLAS_NUM_THREADS": n,
        "NUMEXPR_NUM_THREADS": n,
        "RAYON_NUM_THREADS": n,
        "TORCH_NUM_THREADS": n,
        "TOKENIZERS_PARALLELISM": "false",
    }


def _own_cgroup(root = CGROUP_ROOT):
    """This process's cgroup v2 directory, or None (v1 / unreadable)."""
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            if line.startswith("0::"):
                return Path(root) / line[3:].strip().lstrip("/")
    except OSError:
        pass
    return None


def probe_cgroup(root = CGROUP_ROOT):
    """A cgroup v2 directory where child cpuset groups can be made, or None. A container usually has
    a read-only /sys/fs/cgroup, which is the common answer; affinity is applied either way."""
    base = _own_cgroup(root)
    if base is None or not base.is_dir():
        return None
    try:
        ctl = (base / "cgroup.subtree_control").read_text()
    except OSError:
        return None
    if "cpuset" not in ctl.split():
        return None
    probe = base / f"sb_probe_{os.getpid()}"
    try:
        probe.mkdir()
        probe.rmdir()
    except OSError:
        return None
    return base


def make_cgroup(base, name, cpus):
    """Create `<base>/<name>` limited to `cpus`; its path, or None when that fails."""
    if base is None:
        return None
    d = Path(base) / re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    try:
        d.mkdir(exist_ok = True)
        (d / "cpuset.cpus").write_text(fmt_list(cpus))
        return d
    except OSError:
        return None


def isolate(cpus, cgroup_dir = None):
    """preexec_fn: join the cpuset cgroup when there is one, then pin the child to `cpus`."""
    cpus = sorted(cpus)

    def _pre():
        if cgroup_dir is not None:
            try:
                Path(cgroup_dir, "cgroup.procs").write_text(str(os.getpid()))
            except OSError:
                pass
        try:
            os.sched_setaffinity(0, cpus)
        except (AttributeError, OSError):
            pass

    return _pre


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3 and sys.argv[1] == "pick":
        print(fmt_list(pick_idle(int(sys.argv[2]))))
    else:
        sys.exit("usage: cpu_split.py pick N")
