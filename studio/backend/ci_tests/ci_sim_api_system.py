"""
/api/system robustness simulation for unsloth PR #6509.

Replicates get_system_info()'s psutil-dependent response build VERBATIM from
main.py (post-fix) and stresses psutil failure modes that occur on Mac M1/M4,
Docker-on-Mac, locked-down containers, and Linux VMs (per psutil issues
#1892/#2318/#2642/#1006/#1197). The endpoint must never 500 because one
subsystem probe failed.
"""
import os, sys, time, logging, traceback
logger = logging.getLogger("sim"); logging.disable(logging.CRITICAL)

PASS = FAIL = 0
FAILURES = []
def check(cond, msg):
    global PASS, FAIL
    if cond: PASS += 1
    else:
        FAIL += 1; FAILURES.append(msg); print(f"   FAIL: {msg}")

class FreqTuple:
    def __init__(self, current): self.current = current
class MemInfo:
    def __init__(self, rss): self.rss = rss
class VMem:
    total = 32 * 1024**3; available = 16 * 1024**3; percent = 50.0
class Disk:
    total = 500 * 1e9; free = 250 * 1e9; percent = 50.0

def make_psutil(*, cpu_freq="ok", disk="ok", process="ok", boot_time="ok"):
    """Fake psutil whose individual calls can be set to 'ok', 'none', or 'raise'."""
    class FakePsutil:
        def virtual_memory(self): return VMem()
        def cpu_count(self, logical=True): return 16 if logical else 8
        def cpu_percent(self, interval=None): return 25.0
        def cpu_freq(self):
            if cpu_freq == "raise": raise OSError("cpu_freq unavailable")
            if cpu_freq == "none": return None
            return FreqTuple(2400.0)
        def disk_usage(self, path):
            if disk == "raise": raise PermissionError("disk denied")
            return Disk()
        def Process(self, pid):
            if process == "raise": raise OSError("no proc")
            class P:
                def memory_info(self_inner): return MemInfo(123 * 1024**2)
            return P()
        def boot_time(self):
            if boot_time == "raise": raise OSError("boot_time unavailable")
            return time.time() - 3600
    return FakePsutil()

def build_system_response(psutil, gpu_info=None):
    """VERBATIM replica of get_system_info() body (psutil portion) from main.py."""
    memory = psutil.virtual_memory()
    try:
        cpu_freq = psutil.cpu_freq()
    except Exception as e:
        logger.debug(f"Failed to get CPU frequency: {e}"); cpu_freq = None
    try:
        disk = psutil.disk_usage(os.path.abspath(os.sep))
    except Exception as e:
        logger.debug(f"Failed to get disk usage: {e}"); disk = None
    try:
        current_process = psutil.Process(os.getpid())
    except Exception as e:
        logger.debug(f"Failed to get current process: {e}"); current_process = None
    try:
        boot_time = psutil.boot_time()
    except Exception as e:
        logger.debug(f"Failed to get boot time: {e}"); boot_time = None
    from importlib.metadata import PackageNotFoundError, version as pkg_version
    ml_packages = {}
    for pkg in ("torch", "transformers"):
        try:
            ml_packages[pkg] = pkg_version(pkg)
        except PackageNotFoundError:
            pass
        except Exception as e:
            logger.debug(f"Failed to read {pkg} version: {e}")
    return {
        "platform": "TestOS",
        "python_version": "3.12",
        "device_backend": "cpu",
        "uptime_seconds": round(time.time() - boot_time) if boot_time else None,
        "cpu": {
            "logical_count": psutil.cpu_count(logical=True),
            "physical_count": psutil.cpu_count(logical=False),
            "usage_percent": psutil.cpu_percent(interval=None),
            "frequency_mhz": round(cpu_freq.current, 2) if cpu_freq else None,
        },
        "memory": {
            "total_gb": memory.total / 1024**3,
            "available_gb": memory.available / 1024**3,
            "percent_used": memory.percent,
            "process_used_mb": round(current_process.memory_info().rss / 1024**2)
            if current_process else 0,
        },
        "disk": {
            "total_gb": round(disk.total / 1e9, 2) if disk else 0,
            "free_gb": round(disk.free / 1e9, 2) if disk else 0,
            "percent_used": disk.percent if disk else 0,
        },
        "gpu": gpu_info or {"available": False, "devices": []},
        "ml_packages": ml_packages,
    }

REQUIRED = ["platform", "python_version", "device_backend", "uptime_seconds",
            "cpu", "memory", "disk", "gpu", "ml_packages"]

SCENARIOS = [
    ("all ok", {}),
    ("cpu_freq None (Linux no sysfs / M1)", {"cpu_freq": "none"}),
    ("cpu_freq raises (Docker-on-Mac / KVM)", {"cpu_freq": "raise"}),
    ("disk raises (restricted FS)", {"disk": "raise"}),
    ("process raises", {"process": "raise"}),
    ("cpu_freq+disk+process all raise", {"cpu_freq": "raise", "disk": "raise", "process": "raise"}),
    ("boot_time raises (locked container)", {"boot_time": "raise"}),
]

for name, kw in SCENARIOS:
    print(f"\n[/api/system: {name}]")
    try:
        resp = build_system_response(make_psutil(**kw))
        for k in REQUIRED:
            check(k in resp, f"{name}: response missing '{k}'")
        check(isinstance(resp["cpu"]["frequency_mhz"], (int, float, type(None))),
              f"{name}: frequency_mhz wrong type")
        check(isinstance(resp["disk"]["total_gb"], (int, float)), f"{name}: disk total wrong type")
        print(f"   ok  freq={resp['cpu']['frequency_mhz']} disk_total={resp['disk']['total_gb']} "
              f"proc_mb={resp['memory']['process_used_mb']} uptime={resp['uptime_seconds']}")
    except Exception:
        check(False, f"{name}: /api/system raised (would be HTTP 500)\n{traceback.format_exc()}")

print("\n" + "=" * 60)
print(f"/api/system ROBUSTNESS: {PASS} checks passed, {FAIL} failed")
for f in FAILURES:
    print(" -", f.splitlines()[0])
sys.exit(1 if FAIL else 0)
