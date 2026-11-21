"""Benchmark-only hooks installed into a stack's process (BlineUI server, Studio server, in-process Studio) by
shim/sitecustomize.py (PYTHONPATH) or directly by run_cell.py. Every hook is inert unless its env var is set, and
none changes what the app computes:

  DBENCH_SIM_TOTAL_MIB=<n>  torch.cuda.mem_get_info reports min(total, n MiB) so a planner that sizes headroom from
                            the card's total decides like a smaller card; free stays the real (ballast-reduced) figure.
                            Labelled "simulated" in the record (the tester's simtotal/sitecustomize.py).
  DBENCH_PARITY_DUMP=<path> JSON lines for the parity gate (parity.py), one per event, wall-clock stamped:
      {"ev": "sigmas", "stack", "sigmas": [...], "cfg"}         BlineUI CFGGuider.sample / diffusers set_timesteps
      {"ev": "call", "stack", "items": batch, "cls"}            one denoiser evaluation (BlineUI BaseModel.apply_model,
                                                                a diffusers transformer / UNet forward)
      {"ev": "noise", "stack", "device", "dtype", "shape", "sha"} the initial noise (BlineUI at the guider, diffusers
                                                                randn_tensor), sha over its fp32 values

Patches are applied right after the target module executes (a meta-path finder), so an app's own import order
(BlineUI configures the allocator before importing torch) is never changed.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.abc
import importlib.util
import json
import os
import sys
import threading
import time

_PATCHES: dict = {}
_LOCK = threading.Lock()
_EXT = None


def _ext(name: str) -> str:
    """extnames.ext, loaded by file path: the toolkit root is never put on a server's sys.path (its ``models`` /
    ``common`` would shadow the server's own)."""
    global _EXT
    if _EXT is None:
        try:
            spec = importlib.util.spec_from_file_location("dbench_extnames",
                                                          os.path.join(os.path.dirname(os.path.dirname(
                                                              os.path.abspath(__file__))), "extnames.py"))
            mod = importlib.util.module_from_spec(spec)
            # registered before it runs: torch.compile traces into the hooks and imports the module of every
            # function it meets; unregistered, Dynamo raised "No module named 'dbench_extnames'" and the Studio
            # parity probe (FLUX.3, Krea, LTX-2.5, SDXL) ran eager instead of compiled
            sys.modules.setdefault("dbench_extnames", mod)
            spec.loader.exec_module(mod)
            _EXT = mod.ext
        except Exception:  # noqa: BLE001 - no extnames beside the shim: the names are already real
            _EXT = lambda s: s  # noqa: E731
    return _EXT(name)
_DONE = "_dbench_hooks_installed"


def _emit(event: dict) -> None:
    path = os.environ.get("DBENCH_PARITY_DUMP")
    if not path:
        return
    event = {"t": time.time(), "pid": os.getpid(), **event}
    try:
        line = json.dumps(event, default = str, allow_nan = False)
    except ValueError:
        line = json.dumps({**{k: v for k, v in event.items() if k in ("t", "pid", "ev", "stack")},
                           "error": "non-finite values in event"})
    with _LOCK, open(path, "a", encoding = "utf-8") as f:
        f.write(line + "\n")


def _floats(x) -> list:
    try:
        return [float(v) for v in x.detach().float().cpu().flatten().tolist()]
    except Exception:  # noqa: BLE001
        try:
            return [float(v) for v in x]
        except Exception:  # noqa: BLE001
            return []


def _noise_event(stack: str, t) -> dict:
    try:
        arr = t.detach().to("cpu", dtype = __import__("torch").float32).contiguous().numpy()
        sha = hashlib.sha256(arr.tobytes()).hexdigest()[:16]
    except Exception:  # noqa: BLE001
        sha = None
    return {"ev": "noise", "stack": stack, "device": str(getattr(t, "device", None)),
            "dtype": str(getattr(t, "dtype", None)).replace("torch.", ""), "shape": list(getattr(t, "shape", [])),
            "sha": sha}


# ---------------------------------------------------------------------------------------------- patches
def _patch_cuda(mod) -> None:
    sim = os.environ.get("DBENCH_SIM_TOTAL_MIB") or os.environ.get("BENCH_SIM_TOTAL_MIB")
    if not sim or getattr(mod.mem_get_info, "_dbench_sim", False):
        return
    orig, cap = mod.mem_get_info, int(sim) * 2**20

    def mem_get_info(device = None):
        f, t = orig() if device is None else orig(device)
        return f, min(t, cap)

    mem_get_info._dbench_sim = True
    mod.mem_get_info = mem_get_info
    mem = sys.modules.get("torch.cuda.memory")
    if mem is not None:
        mem.mem_get_info = mem_get_info


def _patch_bline_samplers(mod) -> None:
    guider = getattr(mod, "CFGGuider", None)
    if guider is None or getattr(guider.sample, "_dbench", False):
        return
    orig = guider.sample

    def sample(self, noise, latent_image, sampler, sigmas, *a, **k):
        _emit({"ev": "sigmas", "stack": "blineui", "sigmas": _floats(sigmas), "cfg": getattr(self, "cfg", None)})
        _emit(_noise_event("blineui", noise))
        return orig(self, noise, latent_image, sampler, sigmas, *a, **k)

    sample._dbench = True
    guider.sample = sample


def _patch_bline_model_base(mod) -> None:
    base = getattr(mod, "BaseModel", None)
    if base is None or getattr(base.apply_model, "_dbench", False):
        return
    orig = base.apply_model

    def apply_model(self, x, *a, **k):
        _emit({"ev": "call", "stack": "blineui", "items": int(x.shape[0]) if hasattr(x, "shape") else None,
               "cls": type(self).__name__})
        return orig(self, x, *a, **k)

    apply_model._dbench = True
    base.apply_model = apply_model


def _patch_scheduler_module(mod) -> None:
    for name in dir(mod):
        cls = getattr(mod, name)
        if not isinstance(cls, type) or cls.__module__ != mod.__name__ or "set_timesteps" not in cls.__dict__:
            continue
        orig = cls.__dict__["set_timesteps"]
        if getattr(orig, "_dbench", False):
            continue

        def make(orig, cls_name):
            # wraps: inspect.signature follows __wrapped__, and diffusers' retrieve_timesteps refuses custom sigmas
            # from a set_timesteps whose signature does not name them
            @functools.wraps(orig)
            def set_timesteps(self, *a, **k):
                out = orig(self, *a, **k)
                sig = getattr(self, "sigmas", None)
                _emit({"ev": "sigmas", "stack": "diffusers", "sigmas": _floats(sig) if sig is not None else [],
                       "scheduler": cls_name})
                return out

            set_timesteps._dbench = True
            return set_timesteps

        setattr(cls, "set_timesteps", make(orig, name))


def _patch_torch_utils(mod) -> None:
    orig = getattr(mod, "randn_tensor", None)
    if orig is None or getattr(orig, "_dbench", False):
        return

    def randn_tensor(shape, generator = None, *a, **k):
        out = orig(shape, generator, *a, **k)
        ev = _noise_event("diffusers", out)
        gen = generator[0] if isinstance(generator, (list, tuple)) and generator else generator
        ev["generator_device"] = str(getattr(gen, "device", None)) if gen is not None else None
        _emit(ev)
        return out

    randn_tensor._dbench = True
    mod.randn_tensor = randn_tensor


_INPUT_KWARGS = ("hidden_states", "sample", "x", "latents", "img", "latent_model_input")
_DENOISER_SUFFIXES = ("Transformer2DModel", "Transformer3DModel", "TransformerModel", "UNet2DConditionModel",
                      "DiTModel", "Transformer")


def _patch_torch_module_hook(mod) -> None:
    """A global forward pre-hook counting top-level denoiser calls in diffusers-style stacks (Studio)."""
    if getattr(mod, "_dbench_denoiser_hook", False):
        return
    mod._dbench_denoiser_hook = True

    state = threading.local()

    def eligible(module) -> bool:
        cls = type(module)
        return (not cls.__module__.startswith(_ext("bline")) and cls.__name__.endswith(_DENOISER_SUFFIXES)
                and any(b.__name__ == "ModelMixin" for b in cls.__mro__))

    def batch(args, kwargs):
        x = args[0] if args else None
        if not hasattr(x, "shape"):
            x = next((kwargs.get(k) for k in _INPUT_KWARGS if hasattr(kwargs.get(k), "shape")), None)
        return int(x.shape[0]) if hasattr(x, "shape") and len(x.shape) else None

    def pre(module, args):
        if eligible(module):
            state.depth = getattr(state, "depth", 0) + 1
        return None

    def post(module, args, kwargs, output):
        # the OUTERMOST denoiser only (a UNet's nested Transformer2DModel is not a second evaluation), batch rows
        # read from the positional OR keyword input (sample= / hidden_states= ...)
        if not eligible(module):
            return None
        state.depth = max(0, getattr(state, "depth", 1) - 1)
        if state.depth == 0:
            _emit({"ev": "call", "stack": "diffusers", "items": batch(args, kwargs), "cls": type(module).__name__})
        return None

    def legacy(module, args):  # torch without kwargs in global forward hooks: positional input, outer and nested
        if eligible(module):
            _emit({"ev": "call", "stack": "diffusers", "items": batch(args, {}), "cls": type(module).__name__})
        return None

    try:
        mod.register_module_forward_hook(post, with_kwargs = True, always_call = True)
        mod.register_module_forward_pre_hook(pre)
    except TypeError:
        try:
            mod.register_module_forward_pre_hook(legacy)
        except Exception:  # noqa: BLE001
            pass
    except Exception:  # noqa: BLE001
        pass


def _patch_h3_sampling(mod) -> None:
    """Studio's MiniMax-H3 scheduler builds its grid in h3_simple_sigmas (its set_timesteps never reaches the stock
    one the scheduler hook wraps): emit that grid, the one res_multistep steps over."""
    orig = getattr(mod, "h3_simple_sigmas", None)
    if orig is None or getattr(orig, "_dbench", False):
        return

    @functools.wraps(orig)
    def h3_simple_sigmas(shift, steps):
        out = orig(shift, steps)
        _emit({"ev": "sigmas", "stack": "studio", "sigmas": _floats(out), "scheduler": "h3_simple_sigmas",
               "shift": float(shift)})
        return out

    h3_simple_sigmas._dbench = True
    mod.h3_simple_sigmas = h3_simple_sigmas


def _targets() -> dict:
    out: dict = {}
    if os.environ.get("DBENCH_SIM_TOTAL_MIB") or os.environ.get("BENCH_SIM_TOTAL_MIB"):
        out["torch.cuda"] = _patch_cuda
    if os.environ.get("DBENCH_PARITY_DUMP"):
        out[_ext("bline.samplers")] = _patch_bline_samplers  # the real package name (public copy: extnames)
        out[_ext("bline.model_base")] = _patch_bline_model_base
        out["diffusers.utils.torch_utils"] = _patch_torch_utils
        out["torch.nn.modules.module"] = _patch_torch_module_hook
        out["diffusers.schedulers.*"] = _patch_scheduler_module
        out["core.inference.video_minimax_h3_sampling"] = _patch_h3_sampling
        # Studio's MiniMax-H3-Image scheduler (MiniMaxH3ImageScheduler.set_timesteps) lives in its own module, not
        # under diffusers.schedulers: without it the H3-Image probe dumped no sigmas in process or over HTTP
        out["core.inference.diffusion_minimax_h3_image"] = _patch_scheduler_module
    return out


def _match(name: str):
    if name in _PATCHES:
        return _PATCHES[name]
    if name.startswith("diffusers.schedulers.scheduling_") and "diffusers.schedulers.*" in _PATCHES:
        return _PATCHES["diffusers.schedulers.*"]
    return None


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target = None):
        fn = _match(name)
        if fn is None:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
            return spec
        orig_exec = spec.loader.exec_module

        def exec_module(module):
            orig_exec(module)
            try:
                fn(module)
            except Exception:  # noqa: BLE001 - a hook must never break the app
                pass

        spec.loader.exec_module = exec_module
        return spec


def install() -> dict:
    """Install the hooks this process's environment asks for; modules already imported are patched now. Idempotent.
    Returns {target: "patched now" | "on import"}."""
    _PATCHES.update(_targets())
    state = {}
    for name, fn in _PATCHES.items():
        if name.endswith(".*"):
            prefix = name[:-1] + "scheduling_"
            mods = [m for n, m in list(sys.modules.items()) if n.startswith(prefix) and m is not None]
            for m in mods:
                try:
                    fn(m)
                except Exception:  # noqa: BLE001
                    pass
            state[name] = f"patched {len(mods)} now, rest on import"
        elif name in sys.modules:
            try:
                fn(sys.modules[name])
            except Exception:  # noqa: BLE001
                pass
            state[name] = "patched now"
        else:
            state[name] = "on import"
    if _PATCHES and not getattr(sys, _DONE, False):
        sys.meta_path.insert(0, _Finder())
        setattr(sys, _DONE, True)
    return state


def read_events(path, t0: float = 0.0, t1: float = float("inf")) -> list:
    """Events from a parity dump between t0 and t1 (wall clock), malformed lines skipped."""
    out = []
    try:
        with open(path, encoding = "utf-8") as f:
            for line in f:
                try:
                    ev = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                if t0 <= ev.get("t", 0) <= t1:
                    out.append(ev)
    except FileNotFoundError:
        pass
    return out


def summarize(events: list) -> dict:
    """Parity evidence from one render's events: the last sigma schedule, denoiser calls and batch items, noise."""
    sig = [e for e in events if e.get("ev") == "sigmas" and e.get("sigmas")]
    calls = [e for e in events if e.get("ev") == "call"]
    noise = [e for e in events if e.get("ev") == "noise"]
    stages = []
    for e in sig:  # every stage, in order (two-stage LTX: stage 1 and stage 2); a repeated dump is one stage
        if not stages or stages[-1] != e["sigmas"]:
            stages.append(e["sigmas"])
    return {"sigmas": sig[-1]["sigmas"] if sig else None, "sigma_schedules": stages or None,
            "sigma_source": sig[-1].get("scheduler") or
            sig[-1].get("stack") if sig else None, "denoiser_calls": len(calls) if calls else None,
            "denoiser_items": sum(int(e.get("items") or 1) for e in calls) if calls else None,
            "denoiser_classes": sorted({e.get("cls") for e in calls if e.get("cls")}),
            "noise": noise[0] if noise else None}
