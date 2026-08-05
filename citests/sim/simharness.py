"""Shared harness: import the REAL core/training/training.py with heavy deps stubbed.

Mirrors the stubbing the repo's own test file does, so we exercise the shipped code
(not a copy) in a venv that has no torch, no unsloth and no GPU.
"""
from __future__ import annotations

import contextlib
import logging
import os
import sys
import types as _types
from pathlib import Path

BACKEND = Path(os.environ["STUDIO_BACKEND"]).resolve()
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


def _install_stubs():
    lg = _types.ModuleType("loggers")
    lg.get_logger = lambda name: logging.getLogger(name)
    sys.modules.setdefault("loggers", lg)

    mpl = _types.ModuleType("matplotlib")
    plt = _types.ModuleType("matplotlib.pyplot")
    plt.Figure = type("Figure", (), {})
    mpl.pyplot = plt
    sys.modules.setdefault("matplotlib", mpl)
    sys.modules.setdefault("matplotlib.pyplot", plt)

    hw = _types.ModuleType("utils.hardware")
    hw.prepare_gpu_selection = lambda *a, **k: (None, None)
    hw.get_device = lambda *a, **k: None
    sys.modules.setdefault("utils.hardware", hw)

    npl = _types.ModuleType("utils.native_path_leases")
    npl.native_path_secret_removed_for_child_start = lambda: contextlib.nullcontext()
    npl.run_without_native_path_secret = lambda fn: fn
    sys.modules.setdefault("utils.native_path_leases", npl)

    pth = _types.ModuleType("utils.paths")
    pth.outputs_root = lambda *a, **k: "/tmp/outputs"
    sys.modules.setdefault("utils.paths", pth)


_install_stubs()

from core.training.training import TrainingBackend  # noqa: E402
import core.training.training as T  # noqa: E402

G = T.__dict__


class FakeProc:
    """Stand-in for mp.Process whose liveness the scenario controls."""

    def __init__(self, alive=True, pid=4321):
        self._alive = alive
        self.pid = pid
        self.terminated = False
        self.killed = False

    def is_alive(self):
        return self._alive

    def terminate(self):
        self.terminated = True
        self._alive = False

    def kill(self):
        self.killed = True
        self._alive = False

    def join(self, timeout=None):
        pass


def fresh_backend(job_id="job_1", alive=True):
    """A backend mid-run: worker alive, progress in the training state."""
    b = TrainingBackend()
    b.current_job_id = job_id
    b._proc = FakeProc(alive=alive)
    b._progress = T.TrainingProgress(is_training=True, status_message="Training in progress...")
    b._finalize_run_in_db = lambda **kw: None
    b._ensure_db_run_created = lambda: None
    return b


def ui_active(b) -> bool:
    """Exactly what /api/train/status and the /progress SSE compute."""
    return b.is_training_active() and not b.is_run_finished()


def derive_phase(b):
    """Replica of the phase ladder in routes/training.py get_training_status."""
    p = b._progress
    is_active = ui_active(b)
    status_message = (getattr(p, "status_message", None) or "Ready to train")
    error_message = getattr(p, "error", None)
    if error_message:
        return "error"
    if is_active:
        low = status_message.lower()
        if "loading" in low or "importing" in low:
            return "loading_model"
        if any(k in low for k in ("preparing", "initializing", "configuring")):
            return "configuring"
        return "training"
    if getattr(b, "_should_stop", False):
        return "stopped"
    if getattr(p, "is_completed", False):
        return "completed"
    return "idle"
