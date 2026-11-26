"""Windows MXC sandbox latency bench for Unsloth Studio (staging CI job, Windows legs only).

Measures, on the runner:
  1. host baselines: `cmd /c exit 0`, `python -c pass` (no sandbox)
  2. `wxc-exec --probe` per MXC version (0.8.0 pinned, newer releases): time + reported warnings
  3. Studio end-to-end tool calls (tools._python_exec / tools._bash_exec, tool_execution_mode=required)
     with the pinned runtime, in default mode and in the DACL-fallback mode (after host prep)
  4. raw wxc-exec replay of the exact config Studio built for `echo 1`, once per MXC version,
     isolating wxc-exec's own cost from Studio's Python-side overhead
  5. cProfile of Studio calls (where the Python-side time goes)
  6. the native_mxc pytest suite against the pinned runtime

Usage (staging_ci --job mxc_bench, run from the repo root):
    python .github/scripts/jobs/mxc_bench.py --tiny --out job_mxc.json
`--tiny` is accepted (staging passes it) and only lowers the call counts.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import copy
import hashlib
import io
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
import urllib.request
from pathlib import Path

VERSIONS = ("0.8.0", "0.9.0", "1.0.1", "1.1.0")
NPM = "https://registry.npmjs.org/@microsoft/mxc-sdk/-/mxc-sdk-{v}.tgz"
REPO = Path.cwd()
BACKEND = REPO / "studio" / "backend"


def _stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    s = sorted(xs)
    return {
        "n": len(s),
        "median_ms": round(statistics.median(s), 1),
        "min_ms": round(s[0], 1),
        "p90_ms": round(s[min(len(s) - 1, int(0.9 * len(s)))], 1),
        "max_ms": round(s[-1], 1),
        "mean_ms": round(statistics.fmean(s), 1),
    }


def _time_run(argv, n, **kw):
    out, last = [], None
    for _ in range(n):
        t = time.perf_counter()
        last = subprocess.run(argv, capture_output = True, text = True, errors = "replace", **kw)
        out.append((time.perf_counter() - t) * 1000)
    return out, last


def _arch():
    m = (
        os.environ.get("PROCESSOR_ARCHITEW6432")
        or os.environ.get("PROCESSOR_ARCHITECTURE")
        or platform.machine()
    ).lower()
    return "arm64" if "arm" in m else "x64"


def _fetch_version(v, root: Path):
    d = root / v
    exe = d / "wxc-exec.exe"
    if exe.is_file():
        return d
    d.mkdir(parents = True, exist_ok = True)
    tgz = d / "pkg.tgz"
    req = urllib.request.Request(NPM.format(v = v), headers = {"User-Agent": "mxc-bench"})
    with urllib.request.urlopen(req, timeout = 120) as r, tgz.open("wb") as f:
        shutil.copyfileobj(r, f)
    arch = _arch()
    found = {}
    with tarfile.open(tgz) as tf:
        names = tf.getnames()
        for want in ("wxc-exec.exe", "wxc-host-prep.exe"):
            cands = [n for n in names if n.endswith("/" + want) and f"/{arch}/" in n] or [
                n for n in names if n.endswith("/" + want)
            ]
            if cands:
                m = tf.getmember(cands[0])
                (d / want).write_bytes(tf.extractfile(m).read())
                found[want] = cands[0]
    meta = {
        "version": v,
        "tgz_bytes": tgz.stat().st_size,
        "members": found,
        "bin_members": sorted(n for n in names if "/bin/" in n)[:40],
    }
    for want in ("wxc-exec.exe", "wxc-host-prep.exe"):
        p = d / want
        if p.is_file():
            meta[want] = {
                "size": p.stat().st_size,
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            }
    (d / "meta.json").write_text(json.dumps(meta, indent = 1))
    tgz.unlink()
    return d


def _control_env(state_dir):
    keep = {"SYSTEMROOT", "WINDIR", "PATH", "TEMP", "TMP", "PROGRAMDATA"}
    env = {k: v for k, v in os.environ.items() if k.upper() in keep}
    env["MXC_DACL_STATE_DIR"] = str(state_dir)
    return env


def _probe(
    d: Path,
    state_dir,
    n = 3,
):
    exe = d / "wxc-exec.exe"
    if not exe.is_file():
        return {"error": "wxc-exec.exe missing from package"}
    ver = subprocess.run(
        [str(exe), "--version"],
        capture_output = True,
        text = True,
        errors = "replace",
        cwd = str(d),
        env = _control_env(state_dir),
        timeout = 60,
    )
    times, last = _time_run(
        [str(exe), "--probe"], n, cwd = str(d), env = _control_env(state_dir), timeout = 120
    )
    parsed = None
    with contextlib.suppress(Exception):
        parsed = json.loads(last.stdout)
    return {
        "version_out": (ver.stdout or ver.stderr).strip()[:200],
        "probe": _stats(times),
        "rc": last.returncode,
        "stdout": parsed if parsed is not None else last.stdout[-3000:],
        "stderr": last.stderr[-1500:],
    }


# ---------------------------------------------------------------- child phase: inside Studio code


def _migrate(
    cfg,
    schema,
    host_loopback = "deny",
):
    """0.8.0-alpha network fields -> the 0.9+/1.0.0 directional form (the only change the parser demands)."""
    cfg = copy.deepcopy(cfg)
    cfg["version"] = schema
    old = cfg.get("network") or {}
    if "egress" not in old:
        cfg["network"] = {
            "egress": {"default": old.get("defaultPolicy", "allow")},
            "ingress": {
                "default": "allow" if old.get("allowLocalNetwork", True) else "deny",
                "hostLoopback": host_loopback,
            },
        }
    return cfg


def _run_cfg(
    exe: Path,
    cfg,
    state_dir,
    timeout = 120,
):
    b64 = base64.b64encode(json.dumps(cfg, sort_keys = True, separators = (",", ":")).encode()).decode()
    argv = [str(exe), "--config-base64", b64]
    kw = dict(
        cwd = str(exe.parent), env = _control_env(state_dir), timeout = timeout, stdin = subprocess.DEVNULL
    )
    return argv, kw


def _replay(config, versions_root: Path, n, state_dir):
    """Re-run Studio's exact config with each MXC version's wxc-exec (old shape, then migrated)."""
    res = {}
    for v in VERSIONS:
        d = versions_root / v
        exe = d / "wxc-exec.exe"
        if not exe.is_file():
            res[v] = {"error": "not downloaded"}
            continue
        tries = []
        for schema in (None, "0.9.0-alpha", "1.0.0"):
            cfg = copy.deepcopy(config)
            if schema is not None:
                cfg = _migrate(cfg, schema)
            b64 = base64.b64encode(
                json.dumps(cfg, sort_keys = True, separators = (",", ":")).encode()
            ).decode()
            argv = [str(exe), "--config-base64", b64]
            kw = dict(
                cwd = str(d), env = _control_env(state_dir), timeout = 120, stdin = subprocess.DEVNULL
            )
            first_t = time.perf_counter()
            first = subprocess.run(argv, capture_output = True, text = True, errors = "replace", **kw)
            first_ms = (time.perf_counter() - first_t) * 1000
            ok = first.returncode == 0 and "1" in (first.stdout or "")
            entry = {
                "schema": schema or cfg.get("version"),
                "rc": first.returncode,
                "first_ms": round(first_ms, 1),
                "stdout": first.stdout[-600:],
                "stderr": first.stderr[-900:],
            }
            if ok:
                times, last = _time_run(argv, min(n, 10), **kw)
                entry["warm"] = _stats(times)
                entry["last_rc"] = last.returncode
                tries.append(entry)
                break
            tries.append(entry)
        res[v] = tries
    return res


# ---------------------------------------------------------------- raw, Studio-independent


_NET_PROBE = r"""
import socket, sys
port = int(sys.argv[1])
def tryc(h, p):
    try:
        s = socket.create_connection((h, p), timeout = 3); s.close(); return "ok"
    except OSError as e:
        return type(e).__name__ + ":" + str(getattr(e, "winerror", "") or e.errno)
print("NET", "loopback=" + tryc("127.0.0.1", port), "egress=" + tryc("1.1.1.1", 443))
"""


def _raw_configs(work: Path, dacl: bool, port: int):
    """Minimal ProcessContainer configs mirroring Studio's profile, for `cmd echo` and `python -c pass`."""
    wd = work / "rawwd"
    wd.mkdir(exist_ok = True)
    (wd / "net.py").write_text(_NET_PROBE)
    system32 = os.path.join(os.environ.get("SYSTEMROOT", r"C:\Windows"), "System32")
    py_root = os.path.realpath(sys.base_prefix)
    env = [f"{k}={os.environ[k]}" for k in ("SYSTEMROOT", "WINDIR", "PATH") if k in os.environ]
    env += [f"TEMP={wd}", f"TMP={wd}", f"LOCALAPPDATA={wd}", "PYTHONIOENCODING=utf-8"]

    def cfg(cmdline, readonly):
        return {
            "version": "0.8.0-alpha",
            "containerId": f"unsloth-bench-{os.getpid()}-{time.monotonic_ns()}",
            "containment": "processcontainer",
            "lifecycle": {"destroyOnExit": True, "preservePolicy": False},
            "process": {
                "commandLine": cmdline,
                "cwd": str(wd),
                "env": sorted(env),
                "timeout": 60000,
            },
            "filesystem": {
                "readwritePaths": [str(wd)],
                "readonlyPaths": readonly,
                "deniedPaths": [],
            },
            "network": {
                "defaultPolicy": "allow",
                "allowLocalNetwork": True,
                "allowedHosts": [],
                "blockedHosts": [],
                "enforcementMode": "capabilities",
            },
            "processContainer": {
                "leastPrivilege": False,
                "capabilities": ["internetClient", "privateNetworkClientServer"],
                "ui": {
                    "isolation": "container",
                    "desktopSystemControl": False,
                    "systemSettings": "none",
                    "ime": False,
                },
            },
            "ui": {"disable": False, "clipboard": "none", "injection": False},
            "fallback": {"allowDaclMutation": dacl},
        }

    cmd = os.path.join(system32, "cmd.exe")
    py = os.path.join(py_root, "python.exe")
    return {
        "cmd_echo": cfg(f'"{cmd}" /d /c echo 1', [system32]),
        "python_pass": cfg(f'"{py}" -S -c "print(1)"', [py_root]),
        "python_net": cfg(f'"{py}" -S "{wd / "net.py"}" {port}', [py_root]),
    }


def raw_matrix(versions_root: Path, work: Path, state_dir, n, dacl: bool):
    """Per version x schema shape x workload: rc, first, warm median; plus loopback/egress reachability."""
    import socket
    import threading

    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(16)
    port = srv.getsockname()[1]
    stop = threading.Event()

    def accept():
        srv.settimeout(0.5)
        while not stop.is_set():
            with contextlib.suppress(OSError):
                c, _ = srv.accept()
                c.close()

    threading.Thread(target = accept, daemon = True).start()
    host_net = subprocess.run(
        [sys.executable, "-S", "-c", _NET_PROBE, str(port)],
        capture_output = True,
        text = True,
        timeout = 30,
    ).stdout.strip()
    out = {"host_net": host_net, "dacl": dacl}
    base = _raw_configs(work, dacl, port)
    shapes = {
        "legacy_0.8": lambda c: c,
        "v1.0.0_loopback_deny": lambda c: _migrate(c, "1.0.0", "deny"),
        "v1.0.0_loopback_allow": lambda c: _migrate(c, "1.0.0", "allow"),
    }
    try:
        for v in VERSIONS:
            exe = versions_root / v / "wxc-exec.exe"
            if not exe.is_file():
                out[v] = {"error": "not downloaded"}
                continue
            out[v] = {}
            for shape, fn in shapes.items():
                row = {}
                for wl, c in base.items():
                    if wl == "python_net" and shape == "legacy_0.8" and v.startswith("1."):
                        continue
                    if wl != "python_net" and shape == "v1.0.0_loopback_allow":
                        continue
                    cfg = fn(c)
                    argv, kw = _run_cfg(exe, cfg, state_dir)
                    t = time.perf_counter()
                    try:
                        r = subprocess.run(
                            argv, capture_output = True, text = True, errors = "replace", **kw
                        )
                    except subprocess.TimeoutExpired:
                        row[wl] = {"error": "timeout"}
                        continue
                    first = (time.perf_counter() - t) * 1000
                    cell = {
                        "rc": r.returncode,
                        "first_ms": round(first, 1),
                        "stdout": r.stdout.strip()[-300:],
                        "stderr": r.stderr.strip()[-500:],
                    }
                    if r.returncode == 0 and wl != "python_net":
                        times, _ = _time_run(argv, n, **kw)
                        cell["warm"] = _stats(times)
                    row[wl] = cell
                out[v][shape] = row
    finally:
        stop.set()
        srv.close()
    return out


def wsl_section(work: Path, n):
    """Best effort: is WSL usable on this runner, and does bubblewrap work inside it (WSL1 / WSL2)?"""
    out = {}

    def sh(argv, timeout = 300):
        try:
            r = subprocess.run(argv, capture_output = True, timeout = timeout)
            dec = lambda b: (b or b"").decode(
                "utf-16-le" if b[1:2] == b"\x00" else "utf-8", "replace"
            )  # noqa: E731
            return {
                "rc": r.returncode,
                "out": (dec(r.stdout) + dec(r.stderr)).replace("\x00", "")[-1500:],
            }
        except Exception as exc:  # noqa: BLE001
            return {"error": repr(exc)}

    if not shutil.which("wsl.exe"):
        return {"wsl_exe": None}
    out["status"] = sh(["wsl.exe", "--status"], 60)
    out["list"] = sh(["wsl.exe", "-l", "-v"], 60)
    arch = "arm64" if _arch() == "arm64" else "amd64"
    url = f"https://cloud-images.ubuntu.com/wsl/releases/noble/current/ubuntu-noble-wsl-{arch}-wsl.rootfs.tar.gz"
    tar = work / "wsl_rootfs.tar.gz"
    try:
        with (
            urllib.request.urlopen(
                urllib.request.Request(url, headers = {"User-Agent": "mxc-bench"}), timeout = 300
            ) as r,
            tar.open("wb") as f,
        ):
            shutil.copyfileobj(r, f)
        out["rootfs_bytes"] = tar.stat().st_size
    except Exception as exc:  # noqa: BLE001
        out["rootfs_error"] = repr(exc)
        return out
    script = (
        "set -e; uname -a; cat /proc/version; "
        "(command -v bwrap || (apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq bubblewrap >/dev/null)) ; "
        "bwrap --version; "
        "python3 - <<'EOF'\nimport subprocess,time,statistics\n"
        "def t(a):\n    xs=[]\n    for _ in range(%d):\n        s=time.perf_counter(); r=subprocess.run(a,capture_output=True); xs.append((time.perf_counter()-s)*1000)\n"
        "    return r.returncode, round(statistics.median(xs),1), r.stderr.decode()[-300:]\n"
        "print('WSLBENCH true', t(['true']))\n"
        "print('WSLBENCH bwrap', t(['bwrap','--ro-bind','/','/','--dev','/dev','--proc','/proc','--unshare-all','--die-with-parent','true']))\n"
        "print('WSLBENCH python', t(['python3','-c','pass']))\n"
        "print('WSLBENCH bwrap_python', t(['bwrap','--ro-bind','/','/','--dev','/dev','--proc','/proc','--unshare-all','--die-with-parent','python3','-c','pass']))\n"
        "EOF\n" % n
    )
    for ver in ("2", "1"):
        name = f"UbuntuBench{ver}"
        d = work / f"wsl{ver}"
        d.mkdir(exist_ok = True)
        imp = sh(["wsl.exe", "--import", name, str(d), str(tar), "--version", ver], 900)
        row = {"import": imp}
        if imp.get("rc") == 0:
            row["run"] = sh(["wsl.exe", "-d", name, "-u", "root", "--", "bash", "-lc", script], 900)
            sh(["wsl.exe", "--unregister", name], 300)
        out[f"wsl{ver}"] = row
    return out


def _ensure_studio_deps():
    """ARM64 has no wheel for some studio.txt pins (pyarrow): install line by line, binary only, so the rest lands."""
    probe = [
        sys.executable,
        "-c",
        "import sys; sys.path.insert(0, r'%s'); import core.inference.tools" % BACKEND,
    ]
    first = subprocess.run(probe, capture_output = True, text = True, timeout = 600)
    if first.returncode == 0:
        return {"importable": True, "installed": []}
    req = BACKEND / "requirements" / "studio.txt"
    failed, done = [], []
    for line in req.read_text().splitlines():
        spec = line.split("#", 1)[0].strip()
        if not spec or spec.startswith("-"):
            continue
        r = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "--only-binary", ":all:", spec],
            capture_output = True,
            text = True,
            timeout = 900,
        )
        (done if r.returncode == 0 else failed).append(spec)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "pytest",
            "pytest-asyncio",
            "pytest-timeout",
            "python-multipart",
            "psutil",
        ],
        capture_output = True,
        timeout = 900,
    )
    again = subprocess.run(probe, capture_output = True, text = True, timeout = 600)
    return {
        "importable": again.returncode == 0,
        "failed": failed,
        "n_installed": len(done),
        "import_error": again.stderr[-1500:],
    }


def posix_main(args, result, section, flush):
    """Linux (bubblewrap) / macOS (Seatbelt): the same Studio end-to-end numbers the docs quote."""
    n = args.calls
    result["host"] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpus": os.cpu_count(),
        "python": sys.version,
        "image": os.environ.get("ImageOS"),
        "image_version": os.environ.get("ImageVersion"),
    }
    with contextlib.suppress(Exception):
        result["host"]["os_release"] = Path("/etc/os-release").read_text()[:300]
    flush()

    def baselines():
        t, _ = _time_run(["true"], n)
        py, _ = _time_run([sys.executable, "-c", "pass"], n)
        out = {"true": _stats(t), "python_pass": _stats(py)}
        if shutil.which("bwrap"):
            bw = [
                "bwrap",
                "--ro-bind",
                "/",
                "/",
                "--dev",
                "/dev",
                "--proc",
                "/proc",
                "--unshare-all",
                "--die-with-parent",
            ]
            bt, last = _time_run(bw + ["true"], n)
            bp, _ = _time_run(bw + [sys.executable, "-c", "pass"], n)
            out.update(
                bwrap_true = _stats(bt),
                bwrap_python = _stats(bp),
                bwrap_rc = last.returncode,
                bwrap_err = last.stderr[-300:],
                bwrap_version = subprocess.run(
                    ["bwrap", "--version"], capture_output = True, text = True
                ).stdout.strip(),
            )
        if shutil.which("sandbox-exec"):
            sb = ["sandbox-exec", "-p", "(version 1)(allow default)"]
            st, last = _time_run(sb + ["true"], n)
            sp, _ = _time_run(sb + [sys.executable, "-c", "pass"], n)
            out.update(
                seatbelt_true = _stats(st), seatbelt_python = _stats(sp), seatbelt_rc = last.returncode
            )
        return out

    def ensure_bwrap():
        # Same steps as studio-backend-ci.yml: Ubuntu 24.04+ needs the bwrap AppArmor profile.
        if sys.platform != "linux":
            return {"skipped": sys.platform}
        script = (
            "command -v bwrap >/dev/null || { apt-get install -y --no-install-recommends bubblewrap || "
            "{ apt-get update && apt-get install -y --no-install-recommends bubblewrap; }; }; "
            'if [ "$(cat /proc/sys/kernel/apparmor_restrict_unprivileged_userns 2>/dev/null || echo 0)" = 1 ]; then '
            "apt-get install -y --no-install-recommends apparmor-profiles || true; "
            "[ -f /etc/apparmor.d/bwrap-userns-restrict ] || install -m 644 "
            "/usr/share/apparmor/extra-profiles/bwrap-userns-restrict /etc/apparmor.d/ || true; "
            "apparmor_parser -r /etc/apparmor.d/bwrap-userns-restrict || true; fi"
        )
        pre = ["sudo", "-n"] if shutil.which("sudo") and os.geteuid() != 0 else []
        r = subprocess.run(pre + ["sh", "-c", script], capture_output = True, text = True, timeout = 900)
        ok = (
            subprocess.run(
                ["bwrap", "--ro-bind", "/", "/", "--unshare-user", "true"],
                capture_output = True,
                text = True,
            )
            if shutil.which("bwrap")
            else None
        )
        return {
            "rc": r.returncode,
            "tail": (r.stdout + r.stderr)[-800:],
            "bwrap_works": None if ok is None else ok.returncode == 0,
            "bwrap_err": None if ok is None else ok.stderr[-300:],
        }

    section("ensure_bwrap", ensure_bwrap)
    section("baseline_host", baselines)
    section("studio_deps", _ensure_studio_deps)
    work = Path(tempfile.mkdtemp(prefix = "mxcb-"))
    env = dict(os.environ, UNSLOTH_STUDIO_HOME = str(work / "studio_home"), PYTHONPATH = str(BACKEND))

    def run_child():
        f = work / "child_posix.json"
        r = subprocess.run(
            [
                sys.executable,
                __file__,
                "--phase",
                "child",
                "--mode",
                "posix",
                "--calls",
                str(n),
                "--phase-out",
                str(f),
            ],
            capture_output = True,
            text = True,
            errors = "replace",
            env = env,
            timeout = 3000,
        )
        data = json.loads(f.read_text()) if f.is_file() else {}
        data["rc"] = r.returncode
        data["stderr_tail"] = r.stderr[-2500:]
        return data

    section("studio_posix", run_child)
    section("ab_overhead", lambda: ab_overhead(work, env, n))
    s = result["sections"].get("studio_posix") or {}
    summ = {
        "host": result["host"],
        "baseline": result["sections"].get("baseline_host"),
        "python": {
            k: (s.get("e2e", {}).get("python") or {}).get(k)
            for k in ("first_ms", "warm", "backends")
        },
        "bash": {
            k: (s.get("e2e", {}).get("bash") or {}).get(k) for k in ("first_ms", "warm", "backends")
        },
    }
    result["summary"] = summ
    result["passed"] = bool(summ["python"].get("warm"))
    result["finished"] = True
    flush()
    print("JOB_RESULT " + json.dumps(summ, default = str))
    return 0


def child_lite(args):
    """A/B arm: warm e2e timings plus engagement counters (digests taken, DB connections opened)."""
    backend = os.environ.get("MXC_BENCH_BACKEND") or str(BACKEND)
    sys.path.insert(0, backend)
    from core.inference import mxc_runtime, tools  # noqa: E402
    from storage import studio_db  # noqa: E402

    assert os.path.realpath(tools.__file__).startswith(os.path.realpath(backend)), tools.__file__
    counts = {"sha256": 0, "db_connect": 0}
    real_sha, real_conn = mxc_runtime._sha256_file, studio_db.get_connection

    def sha(path):
        counts["sha256"] += 1
        return real_sha(path)

    def conn(*a, **k):
        counts["db_connect"] += 1
        return real_conn(*a, **k)

    mxc_runtime._sha256_file = sha
    studio_db.get_connection = conn
    out = {"backend_dir": backend, "tools_file": tools.__file__}
    for name, fn, payload in (
        ("python", tools._python_exec, "print(1)"),
        ("bash", tools._bash_exec, "echo 1"),
    ):
        fn(payload, None, 60, "__LOCALID_mxc_ab", tool_execution_mode = "required")
        before = dict(counts)
        times, backends = [], set()
        for _ in range(args.calls):
            t = time.perf_counter()
            fn(payload, None, 60, "__LOCALID_mxc_ab", tool_execution_mode = "required")
            times.append((time.perf_counter() - t) * 1000)
            backends.add(str(getattr(tools._last_tool_execution_record, "backend", None)))
        out[name] = {
            "warm": _stats(times),
            "backends": sorted(backends),
            "per_call": {k: round((counts[k] - before[k]) / args.calls, 2) for k in counts},
        }
    Path(args.phase_out).write_text(json.dumps(out, indent = 1, default = str))


def ab_overhead(
    work: Path,
    env: dict,
    n: int,
    rounds: int = 3,
):
    """Same runner, interleaved base / head arms. Head = this tree + jobs/mxc_overhead_patch.py's PATCH."""
    patch_mod = Path(__file__).with_name("mxc_overhead_patch.py")
    if not patch_mod.is_file():
        return {"skipped": "no mxc_overhead_patch.py"}
    ns = {}
    exec(compile(patch_mod.read_text(), str(patch_mod), "exec"), ns)  # noqa: S102 - our own vendored file
    head = work / "head_tree"
    r = subprocess.run(
        ["git", "-C", str(REPO), "worktree", "add", "--detach", str(head), "HEAD"],
        capture_output = True,
        text = True,
    )
    if r.returncode != 0:
        return {"error": "worktree: " + r.stderr[-500:]}
    pf = work / "head.patch"
    pf.write_bytes(base64.b64decode(ns["PATCH_B64"]))
    r = subprocess.run(
        ["git", "-C", str(head), "apply", "--whitespace=nowarn", str(pf)],
        capture_output = True,
        text = True,
    )
    if r.returncode != 0:
        return {"error": "apply: " + r.stderr[-800:]}
    arms = {"base": str(BACKEND), "head": str(head / "studio" / "backend")}
    out = {"patch_sha256": hashlib.sha256(pf.read_bytes()).hexdigest(), "rounds": []}
    for i in range(rounds):
        row = {}
        for arm in ("base", "head") if i % 2 == 0 else ("head", "base"):
            f = work / f"ab_{arm}_{i}.json"
            e = dict(env, MXC_BENCH_BACKEND = arms[arm], PYTHONPATH = arms[arm])
            r = subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "--phase",
                    "lite",
                    "--calls",
                    str(n),
                    "--phase-out",
                    str(f),
                ],
                capture_output = True,
                text = True,
                errors = "replace",
                env = e,
                timeout = 1800,
            )
            row[arm] = (
                json.loads(f.read_text())
                if f.is_file()
                else {"rc": r.returncode, "err": r.stderr[-1500:]}
            )
        out["rounds"].append(row)
    summ = {}
    for arm in ("base", "head"):
        for tool in ("python", "bash"):
            meds = [
                ((rd.get(arm) or {}).get(tool) or {}).get("warm", {}).get("median_ms")
                for rd in out["rounds"]
            ]
            meds = [m for m in meds if m is not None]
            last = (out["rounds"][-1].get(arm) or {}).get(tool) or {}
            summ[f"{arm}_{tool}"] = {
                "round_medians": meds,
                "median": statistics.median(meds) if meds else None,
                "per_call": last.get("per_call"),
                "backends": last.get("backends"),
            }
    out["summary"] = summ
    subprocess.run(
        ["git", "-C", str(REPO), "worktree", "remove", "--force", str(head)], capture_output = True
    )
    return out


def child(args):
    sys.path.insert(0, str(BACKEND))
    out = {
        "mode": args.mode,
        "env": {
            k: os.environ.get(k)
            for k in (
                "UNSLOTH_MXC_ALLOW_DACL_FALLBACK",
                "UNSLOTH_MXC_PERSISTENT_READ_GRANTS",
                "UNSLOTH_STUDIO_HOME",
            )
        },
    }
    t = time.perf_counter()
    from core.inference import mxc_adapter, mxc_runtime, sandbox_windows_mxc, tools  # noqa: E402

    out["import_ms"] = round((time.perf_counter() - t) * 1000, 1)
    win = sys.platform == "win32"
    state_dir = mxc_runtime.dacl_state_dir() if win else None

    # Python-side fixed costs
    acq = []
    for _ in range(10 if win else 0):
        t = time.perf_counter()
        mxc_runtime.acquire_runtime().release()
        acq.append((time.perf_counter() - t) * 1000)
    ident = []
    for _ in range(10 if win else 0):
        t = time.perf_counter()
        mxc_runtime.installation_identity()
        ident.append((time.perf_counter() - t) * 1000)
    out["acquire_runtime"] = _stats(acq)
    out["installation_identity"] = _stats(ident)

    caps = {}
    for kind, exe in (("python", sys.executable), ("terminal", None)) if win else ():
        t = time.perf_counter()
        try:
            if kind == "terminal":
                exe = shutil.which("cmd.exe") or r"C:\Windows\System32\cmd.exe"
            c = sandbox_windows_mxc.capability_snapshot(
                force = True, execution_kind = kind, selected_executable = exe
            )
            caps[kind] = {
                "available": c.available,
                "reason": c.reason,
                "limitations": list(c.limitations),
            }
        except Exception as exc:  # noqa: BLE001
            caps[kind] = {"error": repr(exc)}
        caps[kind]["ms"] = round((time.perf_counter() - t) * 1000, 1)
    out["capability"] = caps

    captured = {}
    real_spawn = mxc_adapter.spawn

    def spy(request, **kw):
        if args.replay and "config" not in captured and "echo" in json.dumps(request.get("config")):
            captured["config"] = copy.deepcopy(request["config"])
            # Leases (drive alias, read grants) are held right now, so the replay sees the same host state.
            captured["replay"] = _replay(
                request["config"], Path(args.versions_root), args.calls, state_dir
            )
        return real_spawn(request, **kw)

    mxc_adapter.spawn = spy

    def run_tool(fn, payload, n):
        times, recs, sample = [], [], None
        for i in range(n):
            t = time.perf_counter()
            try:
                r = fn(payload, None, 60, "__LOCALID_mxc_bench", tool_execution_mode = "required")
            except Exception as exc:  # noqa: BLE001
                r = f"EXC {exc!r}"
            times.append((time.perf_counter() - t) * 1000)
            rec = tools._last_tool_execution_record
            recs.append(getattr(rec, "backend", None))
            if i == 0:
                sample = {
                    "output": str(r)[-500:],
                    "record": {
                        k: str(getattr(rec, k, None))
                        for k in (
                            "backend",
                            "backend_tier",
                            "execution_status",
                            "completion_status",
                            "cleanup_status",
                            "schema_version",
                            "effective_mode",
                        )
                    },
                }
        return {
            "first_ms": round(times[0], 1),
            "warm": _stats(times[1:]),
            "backends": sorted(set(map(str, recs))),
            "sample": sample,
        }

    res = {}
    for name, fn, payload in (
        ("python", tools._python_exec, "print(1)"),
        ("bash", tools._bash_exec, "echo 1"),
    ):
        try:
            res[name] = run_tool(fn, payload, args.calls + 1)
        except Exception:  # noqa: BLE001
            res[name] = {"error": traceback.format_exc()[-2000:]}
    out["e2e"] = res
    mxc_adapter.spawn = real_spawn
    if captured:
        out["captured_config"] = captured.get("config")
        out["replay"] = captured.get("replay")

    # Where Studio's Python-side time goes.
    import cProfile
    import pstats

    for name, fn, payload in (
        ("python", tools._python_exec, "print(1)"),
        ("bash", tools._bash_exec, "echo 1"),
    ):
        pr = cProfile.Profile()
        pr.enable()
        for _ in range(5):
            with contextlib.suppress(Exception):
                fn(payload, None, 60, "__LOCALID_mxc_bench", tool_execution_mode = "required")
        pr.disable()
        s = io.StringIO()
        pstats.Stats(pr, stream = s).sort_stats("cumulative").print_stats(45)
        out[f"profile_{name}"] = s.getvalue()[-9000:]
    Path(args.phase_out).write_text(json.dumps(out, indent = 1, default = str))


# ---------------------------------------------------------------- orchestrator


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tiny", action = "store_true")
    p.add_argument("--out", default = "job_mxc_bench.json")
    p.add_argument("--calls", type = int, default = None)
    p.add_argument("--phase", default = None)
    p.add_argument("--mode", default = "default")
    p.add_argument("--phase-out", default = None)
    p.add_argument("--versions-root", default = None)
    p.add_argument("--replay", action = "store_true")
    args = p.parse_args()
    if args.calls is None:
        args.calls = 15 if args.tiny else 30
    if args.phase == "child":
        return child(args)
    if args.phase == "lite":
        return child_lite(args)

    result = {"job": "mxc_bench", "passed": False, "finished": False, "sections": {}}
    out_path = Path(args.out).resolve()

    def flush():
        out_path.write_text(json.dumps(result, indent = 1, default = str))

    def section(name, fn):
        t = time.perf_counter()
        try:
            result["sections"][name] = fn()
        except Exception:  # noqa: BLE001
            result["sections"][name] = {"error": traceback.format_exc()[-3000:]}
        result["sections"].setdefault(name, {})
        if isinstance(result["sections"][name], dict):
            result["sections"][name]["_section_s"] = round(time.perf_counter() - t, 1)
        flush()
        print(
            f"=== {name} ===\n{json.dumps(result['sections'][name], indent = 1, default = str)[:12000]}",
            flush = True,
        )

    if sys.platform != "win32":
        return posix_main(args, result, section, flush)

    wv = sys.getwindowsversion()
    result["host"] = {
        "platform": platform.platform(),
        "build": wv.build,
        "arch": _arch(),
        "machine": platform.machine(),
        "cpus": os.cpu_count(),
        "python": sys.version,
        "runner": os.environ.get("RUNNER_NAME"),
        "image": os.environ.get("ImageOS"),
        "image_version": os.environ.get("ImageVersion"),
    }
    with contextlib.suppress(Exception):
        result["host"]["edition"] = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_OperatingSystem | Select Caption,Version,BuildNumber | ConvertTo-Json -Compress);"
                "(Get-CimInstance Win32_Processor | Select -First 1 Name | ConvertTo-Json -Compress);"
                "(Get-WindowsOptionalFeature -Online -FeatureName Containers -ErrorAction SilentlyContinue | Select State | ConvertTo-Json -Compress)",
            ],
            capture_output = True,
            text = True,
            timeout = 120,
        ).stdout.strip()
    flush()

    work = Path(tempfile.mkdtemp(prefix = "mxcb-"))
    vroot = work / "versions"
    state = work / "dacl-state"
    state.mkdir(parents = True)
    n = args.calls

    def baselines():
        cmd, _ = _time_run(["cmd.exe", "/d", "/c", "exit 0"], n)
        py, _ = _time_run([sys.executable, "-c", "pass"], n)
        pyS, _ = _time_run([sys.executable, "-S", "-c", "pass"], n)
        return {"cmd_exit": _stats(cmd), "python_pass": _stats(py), "python_S_pass": _stats(pyS)}

    section("baseline_host", baselines)

    def fetch():
        out = {}
        for v in VERSIONS:
            try:
                d = _fetch_version(v, vroot)
                out[v] = json.loads((d / "meta.json").read_text())
            except Exception as exc:  # noqa: BLE001
                out[v] = {"error": repr(exc)}
        return out

    section("versions", fetch)
    section("probe_per_version", lambda: {v: _probe(vroot / v, state) for v in VERSIONS})
    section("raw_matrix_nodacl", lambda: raw_matrix(vroot, work, state, n, dacl = False))

    home = work / "studio_home"
    env = dict(
        os.environ, UNSLOTH_STUDIO_HOME = str(home), PYTHONPATH = str(BACKEND), PYTHONIOENCODING = "utf-8"
    )
    env.pop("UNSLOTH_MXC_ALLOW_DACL_FALLBACK", None)
    sys.path.insert(0, str(BACKEND))
    os.environ["UNSLOTH_STUDIO_HOME"] = str(home)
    from core.inference import mxc_runtime  # noqa: E402

    pkg = mxc_runtime._installed_package_root()

    def install():
        r = subprocess.run(
            [
                sys.executable,
                str(REPO / "studio" / "install_mxc_prebuilt.py"),
                "--install-dir",
                str(pkg),
            ],
            capture_output = True,
            text = True,
            env = env,
            timeout = 600,
        )
        return {"rc": r.returncode, "out": (r.stdout + r.stderr)[-1500:], "pkg": str(pkg)}

    section("studio_deps", _ensure_studio_deps)
    section("install_pinned", install)

    def run_child(mode, extra_env, replay):
        f = work / f"child_{mode}.json"
        e = dict(env, **extra_env)
        r = subprocess.run(
            [
                sys.executable,
                __file__,
                "--phase",
                "child",
                "--mode",
                mode,
                "--calls",
                str(n),
                "--phase-out",
                str(f),
                "--versions-root",
                str(vroot),
            ]
            + (["--replay"] if replay else []),
            capture_output = True,
            text = True,
            errors = "replace",
            env = e,
            timeout = 3000,
        )
        data = json.loads(f.read_text()) if f.is_file() else {}
        data["rc"] = r.returncode
        data["stderr_tail"] = r.stderr[-2500:]
        return data

    section("studio_default", lambda: run_child("default", {}, True))
    section("ab_overhead_default", lambda: ab_overhead(work, env, n))

    def prep():
        r = subprocess.run(
            [
                sys.executable,
                str(REPO / "studio" / "install_mxc_prebuilt.py"),
                "--prepare-host",
                "--install-dir",
                str(pkg),
            ],
            capture_output = True,
            text = True,
            env = env,
            timeout = 600,
        )
        return {"rc": r.returncode, "out": (r.stdout + r.stderr)[-1500:]}

    section("host_prep", prep)
    section(
        "probe_per_version_after_prep", lambda: {v: _probe(vroot / v, state, n = 2) for v in VERSIONS}
    )
    section("raw_matrix_dacl", lambda: raw_matrix(vroot, work, state, n, dacl = True))
    section(
        "studio_dacl", lambda: run_child("dacl", {"UNSLOTH_MXC_ALLOW_DACL_FALLBACK": "1"}, True)
    )
    section(
        "studio_dacl_persistent",
        lambda: run_child(
            "dacl_persistent",
            {"UNSLOTH_MXC_ALLOW_DACL_FALLBACK": "1", "UNSLOTH_MXC_PERSISTENT_READ_GRANTS": "1"},
            False,
        ),
    )
    section(
        "ab_overhead_dacl",
        lambda: ab_overhead(
            work,
            dict(env, UNSLOTH_MXC_ALLOW_DACL_FALLBACK = "1", UNSLOTH_MXC_PERSISTENT_READ_GRANTS = "1"),
            n,
        ),
    )

    def native(extra):
        e = dict(env, UNSLOTH_MXC_NATIVE_PACKAGE = str(pkg), **extra)
        e.pop("UNSLOTH_STUDIO_HOME", None)
        r = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "studio/backend/tests/test_mxc_native_windows.py",
                "-o",
                "addopts=",
                "-m",
                "native_mxc",
                "-q",
                "-rs",
                "--tb=short",
                "-p",
                "no:cacheprovider",
            ],
            capture_output = True,
            text = True,
            errors = "replace",
            env = e,
            timeout = 2400,
            cwd = str(REPO),
        )
        return {"rc": r.returncode, "tail": r.stdout[-5000:], "stderr": r.stderr[-1000:]}

    def unit():
        tests = sorted(
            {
                str(p.relative_to(REPO)).replace("\\", "/")
                for pat in (
                    "test_mxc_*.py",
                    "test_sandbox_*.py",
                    "test_tool_sandbox_wiring.py",
                    "test_permission_mode_degrade.py",
                    "test_windows_cmd_terminal_profile.py",
                    "test_install_mxc_prebuilt.py",
                    "test_os_sandbox*.py",
                )
                for p in (BACKEND / "tests").glob(pat)
            }
        )
        e = dict(env)
        e.pop("UNSLOTH_STUDIO_HOME", None)
        r = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                *tests,
                "-q",
                "-rfE",
                "--tb=line",
                "-p",
                "no:cacheprovider",
            ],
            capture_output = True,
            text = True,
            errors = "replace",
            env = e,
            timeout = 2400,
            cwd = str(REPO),
        )
        return {
            "rc": r.returncode,
            "files": tests,
            "tail": r.stdout[-6000:],
            "stderr": r.stderr[-1000:],
        }

    section("mxc_sandbox_unit_tests", unit)
    section("native_mxc_pytest_default", lambda: native({}))
    section("native_mxc_pytest_dacl", lambda: native({"UNSLOTH_MXC_ALLOW_DACL_FALLBACK": "1"}))
    section("wsl", lambda: wsl_section(work, n))

    # Summary line per arm.
    summ = {"host": result["host"], "baseline": result["sections"].get("baseline_host")}
    for mode in ("studio_default", "studio_dacl", "studio_dacl_persistent"):
        s = result["sections"].get(mode) or {}
        summ[mode] = {
            "capability": {
                k: (v or {}).get("available") for k, v in (s.get("capability") or {}).items()
            },
            "python": {
                k: (s.get("e2e", {}).get("python") or {}).get(k)
                for k in ("first_ms", "warm", "backends")
            },
            "bash": {
                k: (s.get("e2e", {}).get("bash") or {}).get(k)
                for k in ("first_ms", "warm", "backends")
            },
            "acquire_runtime": s.get("acquire_runtime"),
            "replay": {
                v: [
                    {"schema": t.get("schema"), "rc": t.get("rc"), "warm": t.get("warm")}
                    for t in tries
                ]
                if isinstance(tries, list)
                else tries
                for v, tries in (s.get("replay") or {}).items()
            },
        }
    summ["probe"] = {
        v: {
            "ms": x.get("probe"),
            "tier": (x.get("stdout") or {}).get("tier")
            if isinstance(x.get("stdout"), dict)
            else None,
        }
        for v, x in (result["sections"].get("probe_per_version") or {}).items()
        if isinstance(x, dict)
    }
    for key in ("raw_matrix_nodacl", "raw_matrix_dacl"):
        rm = result["sections"].get(key) or {}
        summ[key] = {"host_net": rm.get("host_net")}
        for v in VERSIONS:
            for shape, row in ((rm.get(v) or {}) if isinstance(rm.get(v), dict) else {}).items():
                for wl, cell in (row or {}).items():
                    summ[key][f"{v}|{shape}|{wl}"] = (
                        cell.get("rc"),
                        (cell.get("warm") or {}).get("median_ms"),
                        cell.get("stdout", "")[-80:] if wl == "python_net" else None,
                    )
    result["summary"] = summ
    py = (result["sections"].get("studio_default") or {}).get("e2e", {}).get("python") or {}
    py_d = (result["sections"].get("studio_dacl") or {}).get("e2e", {}).get("python") or {}
    # Pass = MXC really engaged in at least one mode (a silent software-safeguard fallback is not a pass).
    result["passed"] = any(
        "mxc" in str(b) for b in (py.get("backends") or []) + (py_d.get("backends") or [])
    )
    result["finished"] = True
    flush()
    print("JOB_RESULT " + json.dumps(summ, default = str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
