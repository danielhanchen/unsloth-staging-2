"""The contract every backend implements. ``run_cell.py`` owns the timing protocol, so a backend only knows
how to load a model, render one prompt, and report what it engaged; it never times itself end to end.

A backend is constructed with the merged cell dict and its output directory, then:

    status = backend.load()            # dict: what actually engaged (quant, offload, attention, compile, ...)
    result = backend.render(row, steps) # Render: media plus optional per-render facts
    backend.status()                   # optional, re-read after the renders (a lever can disengage mid-run)
    backend.close()                    # always called; must free the device and kill any child process

``render`` must not return before the device work is finished (synchronise), so wall time is honest.

Per-step timing: a backend that can see progress calls ``self.mark_step(step, total)`` as each denoise step
completes (BlineUI's websocket ``progress`` events, Studio's generate-progress poll, the fake backend's loop) and
``self.mark_phase(name)`` at phase boundaries (``denoise_start``, ``denoise_end``); run_cell turns those into
step_s[], first_step_s and the TE / denoise / VAE split. Timestamps are time.time() (wall clock, comparable across
processes); a backend that cannot see steps leaves them empty and the record says so.

Engagement: ``lever_evidence(name, expected)`` answers whether a lever the cell asked for engaged: True / False with
evidence, or None (no evidence, which run_cell treats as a silent fallback: VOID).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class Render:
    """One render. ``image`` for image cells (PIL / array / base64 / path), ``frames`` for video cells."""

    image: Any = None
    frames: Any = None
    fps: Optional[int] = None
    step_s: Optional[float] = None  # the backend's own per-step figure, when it reports one (BlineUI, sd.cpp)
    peak_alloc_gib: Optional[float] = None  # torch allocator peak for this render, in-process backends only
    extra: dict = field(default_factory = dict)


class Backend:
    name = "base"
    #: in-process backends import torch in this process, so torch peak / host RSS are this model's own
    in_process = False

    def __init__(self, cell: dict, out: Path):
        # absolute: work dirs under it are passed to stack servers that run with their own cwd
        self.cell, self.out = cell, Path(out).resolve()
        self.opts = dict(cell.get("options") or {})
        self.step_events: list = []   # (step, total, t) for the current render
        self.phase_events: dict = {}  # name -> t for the current render
        self.request_facts: dict = {}  # per-render facts for the v2 request (bline execution counts, ...)
        self.server_env: dict = {}    # extra env for a stack server this backend launches (budget, parity shim)

    # ------------------------------------------------------------------------------------------ progress
    def begin_render(self) -> None:
        self.step_events, self.phase_events, self.request_facts = [], {}, {}

    def mark_step(self, step: int, total: Optional[int] = None, t: Optional[float] = None) -> None:
        import time

        self.step_events.append((int(step), total, t if t is not None else time.time()))

    def mark_phase(self, name: str, t: Optional[float] = None) -> None:
        import time

        self.phase_events.setdefault(name, t if t is not None else time.time())

    def lever_evidence(self, name: str, expected) -> tuple:
        """(engaged: True | False | None, evidence str). Default: look the lever up in the last status()."""
        return lever_from_status(getattr(self, "last_status", None) or {}, name, expected)

    def resolved_settings(self) -> dict:
        """What this stack actually ran with (steps, cfg, sampler, scheduler, shift, size, frames), for parity."""
        return {}

    def precision(self) -> dict:
        """Component -> dtype as loaded (transformer, text_encoder, vae), for parity."""
        return {}

    # ------------------------------------------------------------------------------------------ lifecycle
    def load(self) -> dict:
        raise NotImplementedError

    def render(self, row: dict, steps: int) -> Render:
        raise NotImplementedError

    def status(self) -> dict:
        return {}

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------------------------------ helpers
    def extra_pids(self) -> list:
        """Processes outside this one's tree whose VRAM belongs to the cell (e.g. an attached or re-parented
        Studio server). Children of this process are counted already."""
        return []

    def trees(self) -> dict:
        """Source trees to fingerprint (git revision) in the record."""
        return {}

    def sync(self) -> None:
        """Wait for queued device work; a no-op without torch."""
        import sys

        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
        elif _mps(torch):
            torch.mps.synchronize()

    #: where own_vram_mib() reads from (memory.vram_source in the record when the sampler used it)
    vram_source = "mps_driver"

    def own_vram_mib(self) -> Optional[float]:
        """This process's Metal memory (torch.mps.driver_allocated_memory: everything the MPS allocator holds from
        the driver) for an in-process backend on Apple Silicon, where NVML / SMI see nothing; None otherwise."""
        import sys

        torch = sys.modules.get("torch")
        if not self.in_process or not _mps(torch):
            return None
        try:
            return torch.mps.driver_allocated_memory() / 2**20
        except Exception:  # noqa: BLE001 - unknown, not 0.0
            return None

    def reset_peak(self) -> None:
        import sys

        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def peak_alloc_gib(self) -> Optional[float]:
        import sys

        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 2**30, 3)
        return None


def _mps(torch) -> bool:
    """torch is imported, has no CUDA and its MPS backend is up (Apple Silicon)."""
    try:
        return torch is not None and not torch.cuda.is_available() and torch.backends.mps.is_available() and \
            hasattr(torch, "mps")
    except Exception:  # noqa: BLE001
        return False


def _norm(v):
    if isinstance(v, str):
        return v.strip().lower().replace("torch.", "")
    return v


def lever_from_status(status: dict, name: str, expected) -> tuple:
    """Find ``name`` in a status dict (top level, under engaged / speed_optims / resolved) and compare it with
    ``expected`` (True = any truthy value; otherwise equality, case-insensitive for strings)."""
    if not isinstance(status, dict):
        return None, "no status"
    places = [("status", status)]
    for key in ("engaged", "speed_optims", "resolved", "facts"):
        sub = status.get(key)
        if isinstance(sub, dict):
            places.append((key, sub))
        elif isinstance(sub, (list, tuple)) and name in sub:
            return True, f"{key} lists {name}"
    for where, d in places:
        if name in d:
            val = d[name]
            if expected is True:
                ok = bool(val) and _norm(val) not in ("none", "off", "false", "disabled")
            else:
                ok = _norm(val) == _norm(expected)
            return ok, f"{where}.{name}={val!r}"
    return None, f"{name} not in status"


REGISTRY = {
    # name -> "module:Class", imported lazily so a backend's dependencies load only in its own venv
    "fake": "backends.fake:FakeBackend",
    "studio": "backends.studio_inproc:StudioBackend",
    "studio_http": "backends.studio_http:StudioHttpBackend",
    "diffusers": "backends.diffusers_ref:DiffusersBackend",
    "blineui": "backends.blineui:BlineUIBackend",
    "sdcpp": "backends.sdcpp:SdCppBackend",
}


def get_backend(name: str):
    import importlib

    if name not in REGISTRY:
        raise KeyError(f"unknown backend {name!r}; known: {sorted(REGISTRY)}")
    mod, _, cls = REGISTRY[name].partition(":")
    return getattr(importlib.import_module(mod), cls)
