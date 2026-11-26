"""Fair MXC latency bench on one GitHub Windows runner (staging CI job; plans/mxc_fair_bench.md).

Every comparison is made on this VM, in a measured steady state:
  1. fingerprint (CPU model, cores, RAM, OS build/UBR, image, power plan, Defender, top CPU processes)
  2. settle: wait for the VM to go idle (post-boot Defender / update / ngen work)
  3. CPU calibration at start, middle and end (sha256 + pure-Python loop), to expose drift and give a
     secondary normalisation factor
  4. bare `wxc-exec` for MXC 0.8.0 / 0.9.0 / 1.0.1 / 1.1.0 with Studio's policy shape, interleaved per round
     (rotated, alternately reversed) with host spawn arms and an A/A slot; warmup until steady state
  5. Studio `tools.execute_tool` (required mode) through long-lived workers pinned to 1.1.0 / 1.0.1 plus an
     A/A 1.0.1 worker, interleaved per round
Each call is output-checked and recorded; summaries carry medians and paired per-round effects.

Usage: python .github/scripts/jobs/mxc_fair_bench.py --tiny --out job_mxc_fair.json
"""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mxc_bench as mb  # noqa: E402

REPO = Path.cwd()
BACKEND = REPO / "studio" / "backend"
INSTALLER = REPO / "studio" / "install_mxc_prebuilt.py"
VERSIONS = ("0.8.0", "0.9.0", "1.0.1", "1.1.0")
SHAPES = (None, "0.9.0-alpha", "1.0.0")  # legacy first, then each schema the parser may demand


def _ps(script, timeout = 120):
    r = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output = True,
        text = True,
        errors = "replace",
        timeout = timeout,
    )
    try:
        return json.loads(r.stdout)
    except ValueError:
        return {"raw": r.stdout.strip()[-1500:], "err": r.stderr.strip()[-500:]}


def fingerprint():
    info = _ps(
        "$c=Get-CimInstance Win32_Processor|Select -First 1 Name,Manufacturer,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed;"
        "$o=Get-CimInstance Win32_OperatingSystem|Select Caption,Version,BuildNumber,ProductType,TotalVisibleMemorySize,FreePhysicalMemory,LastBootUpTime;"
        "$cs=Get-CimInstance Win32_ComputerSystem|Select Model,HypervisorPresent,NumberOfLogicalProcessors;"
        "$ubr=(Get-ItemProperty 'HKLM:\\SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion').UBR;"
        "$pw=(powercfg /getactivescheme) -join ' ';"
        "$d=$null; try { $d=Get-MpComputerStatus|Select AMServiceEnabled,RealTimeProtectionEnabled,AntivirusEnabled } catch {};"
        "$top=Get-Process|Sort-Object CPU -Descending|Select -First 10 Name,@{n='cpu_s';e={[math]::Round($_.CPU,1)}};"
        "[pscustomobject]@{cpu=$c;os=$o;system=$cs;ubr=$ubr;power=$pw;defender=$d;top_cpu=$top}|ConvertTo-Json -Depth 4 -Compress"
    )
    info["env"] = {
        k: os.environ.get(k)
        for k in ("ImageOS", "ImageVersion", "RUNNER_NAME", "RUNNER_ARCH", "GITHUB_RUN_ID")
    }
    info["python"] = {
        "version": sys.version.split()[0],
        "machine": platform.machine(),
        "arch": mb._arch(),
    }
    return info


def settle(
    max_s = 300,
    idle_pct = 10.0,
    hold_s = 20,
):
    import psutil

    trace, below, t0 = [], 0, time.monotonic()
    while time.monotonic() - t0 < max_s:
        pct = psutil.cpu_percent(interval = 1.0)
        trace.append(round(pct, 1))
        below = below + 1 if pct < idle_pct else 0
        if below >= hold_s:
            break
    return {
        "settled": below >= hold_s,
        "waited_s": round(time.monotonic() - t0, 1),
        "trace": trace[-60:],
        "median_pct": statistics.median(trace) if trace else None,
    }


def calibrate(reps = 7):
    blob = os.urandom(64 * 1024 * 1024)
    sha, loop = [], []
    for _ in range(reps):
        t = time.perf_counter()
        hashlib.sha256(blob).hexdigest()
        sha.append((time.perf_counter() - t) * 1000)
        t = time.perf_counter()
        acc = 0
        for i in range(2_000_000):
            acc = (acc * 31 + i) & 0xFFFFFFFF
        loop.append((time.perf_counter() - t) * 1000)
    return {
        "sha256_64mb_ms": round(statistics.median(sha), 2),
        "py_loop_2m_ms": round(statistics.median(loop), 2),
    }


def _cpu_now():
    import psutil
    return psutil.cpu_percent(interval = 0.2)


# ---------------------------------------------------------------- bare arms


def qualify(versions_root, work, state, dacl, workloads):
    """Per version: probe tier and the first policy shape that runs each workload with the right output."""
    base = mb._raw_configs(work, dacl, port = 9)
    out = {}
    for v in VERSIONS:
        d = versions_root / v
        exe = d / "wxc-exec.exe"
        meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").is_file() else {}
        cell = {
            "exe_sha256": (meta.get("wxc-exec.exe") or {}).get("sha256"),
            "probe": mb._probe(d, state, n = 1),
        }
        tier = (
            cell["probe"].get("stdout", {}).get("tier")
            if isinstance(cell["probe"].get("stdout"), dict)
            else None
        )
        cell["tier"] = tier
        cell["workloads"] = {}
        for wl in workloads:
            for shape in SHAPES:
                cfg = copy.deepcopy(base[wl]) if shape is None else mb._migrate(base[wl], shape)
                argv, kw = mb._run_cfg(exe, cfg, state)
                try:
                    r = subprocess.run(argv, capture_output = True, text = True, errors = "replace", **kw)
                except subprocess.TimeoutExpired:
                    continue
                if r.returncode == 0 and r.stdout.strip().endswith("1"):
                    cell["workloads"][wl] = {"shape": shape or "legacy", "argv": argv, "kw": kw}
                    break
                cell.setdefault("errors", []).append(
                    {
                        "wl": wl,
                        "shape": shape,
                        "rc": r.returncode,
                        "err": (r.stderr or r.stdout)[-300:],
                    }
                )
        out[v] = cell
    return out


def _host_arms(workloads):
    system32 = os.path.join(os.environ.get("SYSTEMROOT", r"C:\Windows"), "System32")
    arms = {}
    if "cmd_echo" in workloads:
        arms["host|cmd_echo"] = ([os.path.join(system32, "cmd.exe"), "/d", "/c", "echo 1"], {})
    if "python_pass" in workloads:
        arms["host|python_pass"] = ([sys.executable, "-S", "-c", "print(1)"], {})
    return arms


def _call(argv, kw):
    t = time.perf_counter()
    try:
        r = subprocess.run(
            argv,
            capture_output = True,
            text = True,
            errors = "replace",
            stdin = subprocess.DEVNULL,
            **{k: v for k, v in kw.items() if k != "stdin"},
        )
    except subprocess.TimeoutExpired:
        return (time.perf_counter() - t) * 1000, False
    return (time.perf_counter() - t) * 1000, r.returncode == 0 and r.stdout.strip().endswith("1")


def steady(
    xs,
    window = 5,
    drift = 0.03,
    cv = 0.08,
):
    if len(xs) < 2 * window:
        return False
    a, b = xs[-2 * window : -window], xs[-window:]
    ma, mbm = statistics.median(a), statistics.median(b)
    return abs(mbm - ma) / ma <= drift and statistics.pstdev(b) / statistics.fmean(b) <= cv


def run_rounds(arms, call, warm_max, rounds, seed, records, phase):
    """Warm all arms in rotation until each is steady (two consecutive checks), then timed rounds."""
    names = list(arms)
    rng = random.Random(seed)
    rng.shuffle(names)
    warm = {a: [] for a in names}
    ok_checks = {a: 0 for a in names}
    w = 0
    for w in range(warm_max):
        for a in names[w % len(names) :] + names[: w % len(names)]:
            ms, ok = call(arms[a])
            if ok:
                warm[a].append(ms)
        for a in names:
            ok_checks[a] = ok_checks[a] + 1 if steady(warm[a]) else 0
        if w >= 9 and all(c >= 2 for c in ok_checks.values()):
            break
    state = {"warm_rounds": w + 1, "steady": {a: ok_checks[a] >= 2 for a in names}}
    for r in range(rounds):
        order = names[r % len(names) :] + names[: r % len(names)]
        if r % 2:
            order.reverse()
        load = _cpu_now()
        for pos, a in enumerate(order):
            ms, ok = call(arms[a])
            records.append(
                {
                    "phase": phase,
                    "arm": a,
                    "round": r,
                    "pos": pos,
                    "ms": round(ms, 2),
                    "ok": ok,
                    "load_pct": load,
                }
            )
    return state


# ---------------------------------------------------------------- Studio arms

_WORKER = r"""
import json, os, sys, time
sys.path.insert(0, os.environ["FB_BACKEND"])
import logging
logging.disable(logging.WARNING)
from core.inference import mxc_runtime, tools
print("READY " + json.dumps({"pin": mxc_runtime.selected_pin().version}), flush = True)
i = 0
for line in sys.stdin:
    tool = line.strip()
    if not tool:
        continue
    i += 1
    if tool == "python":
        args = {"code": f"print('FB-OK')\n# {i}"}
    else:
        args = {"command": "echo FB-OK"}
    t = time.perf_counter()
    text = tools.execute_tool(tool, args, session_id = None, tool_execution_mode = "required")
    ms = (time.perf_counter() - t) * 1000
    rec = getattr(tools, "_last_tool_execution_record", None)
    print("RES " + json.dumps({"ms": ms, "ok": "FB-OK" in text, "backend": getattr(rec, "backend", None)}), flush = True)
"""


class Worker:
    def __init__(self, name, env, work):
        self.name = name
        self.proc = subprocess.Popen(
            [sys.executable, "-c", _WORKER],
            stdin = subprocess.PIPE,
            stdout = subprocess.PIPE,
            stderr = subprocess.DEVNULL,
            text = True,
            env = env,
            cwd = str(work),
            bufsize = 1,
        )
        line = self.proc.stdout.readline()
        self.ready = json.loads(line[6:]) if line.startswith("READY ") else {"error": line[-300:]}

    def call(self, tool):
        self.proc.stdin.write(tool + "\n")
        self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            if not line:
                return None, False, None
            if line.startswith("RES "):
                d = json.loads(line[4:])
                return (
                    d["ms"],
                    d["ok"] and d.get("backend") == "mxc-processcontainer",
                    d.get("backend"),
                )

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout = 30)
        except Exception:  # noqa: BLE001
            self.proc.kill()


def studio_phase(work, pins, dacl, warm_max, rounds, seed, records):
    out = {"homes": {}}
    base_env = {**os.environ, "FB_BACKEND": str(BACKEND), "PYTHONIOENCODING": "utf-8"}
    if dacl:
        base_env.update(UNSLOTH_MXC_ALLOW_DACL_FALLBACK = "1", UNSLOTH_MXC_PERSISTENT_READ_GRANTS = "1")
    workers = {}
    for name, pin in pins.items():
        home = work / f"home_{name}"
        env = {**base_env, "UNSLOTH_STUDIO_HOME": str(home), "UNSLOTH_MXC_PIN": pin}
        inst_dir = (
            home / "mxc-runtime" / f"windows-{'arm64' if mb._arch() == 'arm64' else 'x86_64'}"
        )
        r = subprocess.run(
            [sys.executable, str(INSTALLER), "--install-dir", str(inst_dir)],
            capture_output = True,
            text = True,
            env = env,
            timeout = 900,
        )
        rec = {"install_rc": r.returncode, "install_out": (r.stdout + r.stderr)[-400:]}
        if dacl:
            p = subprocess.run(
                [sys.executable, str(INSTALLER), "--prepare-host", "--install-dir", str(inst_dir)],
                capture_output = True,
                text = True,
                env = env,
                timeout = 1800,
            )
            rec["prep_rc"] = p.returncode
        w = Worker(name, env, work)
        rec["ready"] = w.ready
        out["homes"][name] = rec
        if w.ready.get("pin") == pin:
            workers[name] = w
        else:
            w.close()
    try:
        for tool in ("python", "terminal"):
            arms = {f"studio|{name}|{tool}": name for name in workers}

            def call(name, tool = tool):
                ms, ok, _backend = workers[name].call(tool)
                return (ms or 0.0), ok

            out[tool] = run_rounds(arms, call, warm_max, rounds, seed, records, f"studio_{tool}")
    finally:
        for w in workers.values():
            w.close()
    return out


# ---------------------------------------------------------------- summary


def summarize(records):
    by = {}
    for r in records:
        if r["ok"]:
            by.setdefault((r["phase"], r["arm"]), []).append(r)
    summ = {}
    for (phase, arm), rows in sorted(by.items()):
        ms = [x["ms"] for x in rows]
        summ[f"{phase}|{arm}"] = {
            "n": len(ms),
            "median_ms": round(statistics.median(ms), 2),
            "p10": round(sorted(ms)[len(ms) // 10], 2),
            "p90": round(sorted(ms)[(9 * len(ms)) // 10], 2),
        }
    return summ


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tiny", action = "store_true")
    p.add_argument("--out", default = "job_mxc_fair.json")
    p.add_argument("--rounds", type = int, default = 40)
    p.add_argument("--studio-rounds", type = int, default = 30)
    p.add_argument("--seed", type = int, default = 3407)
    args = p.parse_args()
    out_path = Path(args.out).resolve()
    result = {"job": "mxc_fair_bench", "passed": False, "sections": {}, "records": []}

    def flush():
        out_path.write_text(json.dumps(result, indent = 1, default = str))

    def section(name, fn):
        t = time.perf_counter()
        try:
            result["sections"][name] = fn()
        except Exception:  # noqa: BLE001
            result["sections"][name] = {"error": traceback.format_exc()[-3000:]}
        if isinstance(result["sections"][name], dict):
            result["sections"][name]["_section_s"] = round(time.perf_counter() - t, 1)
        flush()
        print(
            f"=== {name} ===\n{json.dumps(result['sections'][name], default = str)[:4000]}",
            flush = True,
        )

    try:
        if sys.platform != "win32":
            result["skipped"] = "Windows only"
            result["passed"] = True
            return 0
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "psutil"],
            capture_output = True,
            timeout = 600,
        )
        work = Path(tempfile.mkdtemp(prefix = "mxcfair-"))
        vroot, state = work / "versions", work / "dacl-state"
        state.mkdir(parents = True)
        section("fingerprint_start", fingerprint)
        section("settle", settle)
        section("calibration_start", calibrate)
        section(
            "fetch",
            lambda: {
                v: json.loads((mb._fetch_version(v, vroot) / "meta.json").read_text())
                for v in VERSIONS
            },
        )

        server = sys.getwindowsversion().product_type != 1
        result["tier_mode"] = "appcontainer-dacl" if server else "base-container"
        section("studio_deps", mb._ensure_studio_deps)
        if server:
            # Host prep once for the machine (1.0.1's wxc-host-prep through the installer).
            def prep():
                home = work / "home_prep"
                env = {**os.environ, "UNSLOTH_STUDIO_HOME": str(home), "UNSLOTH_MXC_PIN": "1.0.1"}
                d = home / "mxc-runtime" / "windows-x86_64"
                a = subprocess.run(
                    [sys.executable, str(INSTALLER), "--install-dir", str(d)],
                    capture_output = True,
                    text = True,
                    env = env,
                    timeout = 900,
                )
                b = subprocess.run(
                    [sys.executable, str(INSTALLER), "--prepare-host", "--install-dir", str(d)],
                    capture_output = True,
                    text = True,
                    env = env,
                    timeout = 1800,
                )
                return {
                    "install_rc": a.returncode,
                    "prep_rc": b.returncode,
                    "out": (b.stdout + b.stderr)[-500:],
                }

            section("host_prep", prep)
        # Bare Python on the DACL tier re-ACLs the whole Python install per launch (14-27 s): shell only there.
        workloads = ("cmd_echo",) if server else ("cmd_echo", "python_pass")

        def do_qualify():
            _qualified.update(qualify(vroot, work, state, server, workloads))
            return {
                v: {
                    **{k: x for k, x in c.items() if k != "workloads"},
                    "workloads": {wl: w["shape"] for wl, w in c["workloads"].items()},
                }
                for v, c in _qualified.items()
            }

        section("qualify", do_qualify)
        arms = dict(_host_arms(workloads))
        for v, c in _qualified.items():
            for wl, w in c["workloads"].items():
                arms[f"{v}|{wl}"] = (w["argv"], w["kw"])
                if v == "1.0.1":
                    arms[f"{v}'|{wl}"] = (w["argv"], w["kw"])  # A/A slot: same binary, same policy
        warm_max = 30
        section(
            "bare",
            lambda: run_rounds(
                arms,
                lambda a: _call(*a),
                warm_max,
                args.rounds,
                args.seed,
                result["records"],
                "bare",
            )
            | {"arms": sorted(arms)},
        )
        section("calibration_mid", calibrate)
        pins = {"1.0.1": "1.0.1", "1.0.1'": "1.0.1"}
        if not server:
            pins["1.1.0"] = "1.1.0"
        if sys.getwindowsversion().build >= 26100:
            section(
                "studio",
                lambda: studio_phase(
                    work, pins, server, warm_max, args.studio_rounds, args.seed, result["records"]
                ),
            )
        else:
            result["sections"]["studio"] = {"skipped": "Studio needs Windows build 26100 for MXC"}
        section("calibration_end", calibrate)
        section("fingerprint_end", fingerprint)
        result["summary"] = summarize(result["records"])
        total = len(result["records"])
        bad = sum(not r["ok"] for r in result["records"])
        result["ok_rate"] = round(1 - bad / total, 4) if total else 0
        result["passed"] = bool(total) and result["ok_rate"] >= 0.95
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()[-4000:]
    finally:
        flush()
        print(
            "JOB_RESULT "
            + json.dumps(
                {"passed": result["passed"], "summary": result.get("summary")}, default = str
            )[:20000],
            flush = True,
        )
    return 0 if result["passed"] else 1


_qualified: dict = {}

if __name__ == "__main__":
    raise SystemExit(main())
