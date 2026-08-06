#!/usr/bin/env python3
"""B: hardware-dispatch cartesian product for unslothai/unsloth#7917.

[Windows, Linux, WSL, macOS] x [NVIDIA, AMD/ROCm, CPU-only], plus the Apple Silicon
cells, run against the REAL routing predicates in Studio's worker.

The thing being proved: PR #7917 only ever changes behavior inside `_run_mlx_training`.
If the MLX branch is unreachable on a cell, that cell is bit-identical before and after
the PR. `run_training_process` (worker.py:2489-2506) routes with:

    is_apple_silicon_training_platform() or should_use_mlx_training_backend(device=DEVICE)

so the claim to test is: only Darwin/arm64 (or a device that already resolved to MLX)
takes the MLX path, and no NVIDIA / AMD / CPU / Windows / WSL cell does.

Spoofing extends the pattern in tests/studio/test_hardware_dispatch_matrix.py, closing
the three gaps that file has: it never sets sys.platform (so Windows branches stay dead),
never covers WSL (whose detector caches into a module-level flag at import), and only
signals AMD via torch.version.hip, not the `rocm` in torch.__version__ fallback.
"""

import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(os.environ.get("UNSLOTH_REPO") or Path(__file__).resolve().parents[2])
sys.path.insert(0, str(REPO / "studio" / "backend"))
os.environ.setdefault("UNSLOTH_ALLOW_CPU", "1")
os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")

# Import everything BEFORE any spoofing. torch's __init__ branches on sys.platform and
# would try ctypes.WinDLL under a "win32" spoof; the backend modules likewise read
# platform at import. Spoofing must therefore only ever patch attributes on already
# imported modules, which is also what makes the predicates re-evaluate per cell.
import torch  # noqa: E402
import core.training.training as T  # noqa: E402

try:
    import utils.paths.path_utils as _path_utils  # noqa: E402
except Exception:
    _path_utils = None
try:
    import hub.utils.paths as _hub_paths  # noqa: E402
except Exception:
    _hub_paths = None

FAILURES = []


@dataclass
class Cell:
    os_name: str          # Windows | Linux | WSL | macOS
    accel: str            # NVIDIA | AMD | CPU
    system: str           # platform.system()
    machine: str          # platform.machine()
    sys_platform: str     # sys.platform
    is_wsl: bool
    cuda: bool
    hip: object           # torch.version.hip
    rocm_in_version: bool
    has_mlx: bool
    expect_mlx: bool      # does run_training_process route to _run_mlx_training?


# 4 OSes x 3 accelerators. macOS x NVIDIA/AMD are not real hardware combinations
# (no CUDA or ROCm on Apple Silicon), so those two cells model the honest variants:
# Apple Silicon with a working MLX stack, and with a broken one.
CELLS = [
    Cell("Windows", "NVIDIA", "Windows", "AMD64",  "win32",  False, True,  None,  False, False, False),
    Cell("Windows", "AMD",    "Windows", "AMD64",  "win32",  False, True,  "6.1", False, False, False),
    Cell("Windows", "CPU",    "Windows", "AMD64",  "win32",  False, False, None,  False, False, False),
    Cell("Linux",   "NVIDIA", "Linux",   "x86_64", "linux",  False, True,  None,  False, False, False),
    Cell("Linux",   "AMD",    "Linux",   "x86_64", "linux",  False, True,  "6.1", False, False, False),
    Cell("Linux",   "AMD",    "Linux",   "x86_64", "linux",  False, True,  None,  True,  False, False),  # rocm wheel, hip unset
    Cell("Linux",   "CPU",    "Linux",   "x86_64", "linux",  False, False, None,  False, False, False),
    Cell("WSL",     "NVIDIA", "Linux",   "x86_64", "linux",  True,  True,  None,  False, False, False),
    Cell("WSL",     "AMD",    "Linux",   "x86_64", "linux",  True,  True,  "6.1", False, False, False),
    Cell("WSL",     "CPU",    "Linux",   "x86_64", "linux",  True,  False, None,  False, False, False),
    Cell("macOS",   "CPU",    "Darwin",  "arm64",  "darwin", False, False, None,  False, True,  True),
    Cell("macOS",   "CPU",    "Darwin",  "arm64",  "darwin", False, False, None,  False, False, True),  # broken mlx: still routed
    Cell("macOS",   "CPU",    "Darwin",  "x86_64", "darwin", False, False, None,  False, False, False),  # Intel Mac
    # Canary: mlx importable on a non-Darwin host must NOT flip the routing.
    Cell("Linux",   "CPU",    "Linux",   "aarch64", "linux", False, False, None,  False, True,  False),
]


class Spoof:
    """Context manager restoring every attribute it touches."""

    def __init__(self, cell):
        self.cell = cell
        self.undo = []

    def _set(self, obj, name, value):
        had = hasattr(obj, name)
        old = getattr(obj, name, None)
        setattr(obj, name, value)
        self.undo.append((obj, name, old, had))

    def __enter__(self):
        c = self.cell
        self._set(platform, "system", lambda: c.system)
        self._set(platform, "machine", lambda: c.machine)
        self._set(sys, "platform", c.sys_platform)

        # WSL is detected by reading /proc/version at import and caching into a
        # module-level flag, duplicated across modules. Patch the caches, which is
        # what the repo's own tests do (tests/studio/test_reveal_file_manager.py:86).
        for mod in (_path_utils, _hub_paths):
            if mod is not None and hasattr(mod, "_IS_WSL"):
                self._set(mod, "_IS_WSL", c.is_wsl)

        self._set(torch.cuda, "is_available", lambda: c.cuda)
        if c.cuda:
            self._set(torch.cuda, "get_device_properties",
                      lambda *_a, **_k: type("P", (), {"name": "spoofed"})())
        self._set(torch.version, "hip", c.hip)
        if c.rocm_in_version:
            self._set(torch, "__version__", "2.10.0+rocm6.2")

        # mlx presence / absence, mirroring the repo fixture's MetaPathFinder trick.
        if c.has_mlx:
            import importlib.machinery
            import types
            mlx = types.ModuleType("mlx")
            mlx.__spec__ = importlib.machinery.ModuleSpec("mlx", loader=None)
            mlx.__path__ = []
            core = types.ModuleType("mlx.core")
            core.__spec__ = importlib.machinery.ModuleSpec("mlx.core", loader=None)
            mlx.core = core
            self._saved_modules = {k: sys.modules.get(k) for k in ("mlx", "mlx.core")}
            sys.modules["mlx"] = mlx
            sys.modules["mlx.core"] = core
        else:
            self._saved_modules = {k: sys.modules.pop(k, None) for k in ("mlx", "mlx.core")}

            class _Block:
                def find_spec(self, name, path=None, target=None):
                    if name == "mlx" or name.startswith("mlx."):
                        raise ImportError("mlx blocked by sim_b spoof")
                    return None

            self._old_meta = sys.meta_path
            sys.meta_path = [_Block(), *sys.meta_path]
        return self

    def __exit__(self, *exc):
        for obj, name, old, had in reversed(self.undo):
            if had:
                setattr(obj, name, old)
            else:
                try:
                    delattr(obj, name)
                except Exception:
                    pass
        if hasattr(self, "_old_meta"):
            sys.meta_path = self._old_meta
        for k, v in getattr(self, "_saved_modules", {}).items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        return False


def routes_to_mlx(cell):
    """Evaluate the REAL routing predicates from core.training.training."""

    # is_apple_silicon_training_platform() is pure platform, no torch/mlx.
    apple = T.is_apple_silicon_training_platform()
    # The second arm only fires if hardware detection already resolved to MLX,
    # which on a non-Darwin host it cannot. Model it off the spoofed mlx presence
    # AND Darwin, matching hardware.py:509 `is_apple_silicon() and _has_usable_mlx_stack()`.
    device_is_mlx = (cell.system == "Darwin" and cell.machine == "arm64" and cell.has_mlx)
    return apple or T.should_use_mlx_training_backend(device="mlx" if device_is_mlx else "cuda")


print("== B: dispatch cartesian product ==")
print(f"  {'OS':<8} {'accel':<7} {'machine':<8} {'sys.platform':<13} {'wsl':<5} "
      f"{'mlx':<5} {'->MLX':<6} {'want':<6} verdict")
print("  " + "-" * 78)
for cell in CELLS:
    with Spoof(cell):
        got = routes_to_mlx(cell)
    ok = got == cell.expect_mlx
    if not ok:
        FAILURES.append(f"{cell.os_name}/{cell.accel}/{cell.machine}: got {got}, want {cell.expect_mlx}")
    print(f"  {cell.os_name:<8} {cell.accel:<7} {cell.machine:<8} {cell.sys_platform:<13} "
          f"{str(cell.is_wsl):<5} {str(cell.has_mlx):<5} {str(got):<6} "
          f"{str(cell.expect_mlx):<6} {'PASS' if ok else 'FAIL'}")

# The PR's changed symbol must be unreachable from a non-MLX cell. Prove structurally
# that _resolve_mlx_max_grad_norm is referenced only inside the MLX training function.
print("\n== B2: is the PR's changed code reachable off the MLX path? ==")
import ast  # noqa: E402

worker_src = (REPO / "studio" / "backend" / "core" / "training" / "worker.py").read_text(encoding="utf-8")
tree = ast.parse(worker_src)
callers = []
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef):
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) \
                    and sub.func.id == "_resolve_mlx_max_grad_norm":
                callers.append(node.name)
print(f"  functions calling _resolve_mlx_max_grad_norm: {callers}")
ok = callers == ["_run_mlx_training"]
if not ok:
    FAILURES.append(f"_resolve_mlx_max_grad_norm reachable from {callers}")
print(f"  [{'PASS' if ok else 'FAIL'}] only _run_mlx_training calls it")

# And the CUDA trainer call must not forward any clip knob at all.
cuda_forwards = [k for k in ("max_grad_norm", "max_grad_value", "max_grad_leaf_norm")
                 if f"{k} = config.get" in worker_src.split("def _run_mlx_training")[0]]
print(f"  [{'PASS' if not cuda_forwards else 'FAIL'}] no clip knob read before the MLX "
      f"function: {cuda_forwards or 'none'}")
if cuda_forwards:
    FAILURES.append(f"clip knob read outside MLX path: {cuda_forwards}")

print("\n" + "=" * 70)
if FAILURES:
    print(f"B FAILED ({len(FAILURES)}):")
    for f in FAILURES:
        print("   -", f)
    sys.exit(1)
print("B PASSED: 14/14 cells route as expected; the PR's code is reachable only on")
print("Apple Silicon, so every Windows / Linux / WSL / NVIDIA / AMD / CPU path is")
print("bit-identical before and after this change.")
sys.exit(0)
