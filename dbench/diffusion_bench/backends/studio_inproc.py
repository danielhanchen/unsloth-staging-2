"""Unsloth Studio's own image / video backends, imported in this process: the exact DiffusionBackend /
VideoBackend objects the Images and Video pages drive, without HTTP, the gallery, or the engine router.

Cell fields:
  model                 local path or repo id
  kind                  image | video
  options.studio_src    the Studio source (setup_studio.resolve_tree: path, git ref, pr/N, pypi); default
                        $DIFFUSION_BENCH_STUDIO_SRC, else the backend directory already importable in this venv
  options.<load key>    any of LOAD_KEYS, passed straight to load_pipeline / begin_load
  options.load          dict passthrough for anything else the loader takes
  options.generate      dict passthrough for generate() (e.g. {"batch_size": 2} or {"flow_shift": 5.0})
  options.local_files_only  default True (a bench never downloads mid-timing)
  options.load_via      image only: "begin_load" (default: the route's path, background thread + load_progress
                        polling, same as video; it plans hosted prequant seeds) or "load_pipeline" (synchronous,
                        never seeds, so a fresh home builds int8 / ConvRot on the host: not what users get)
  options.studio_home   UNSLOTH_STUDIO_HOME for compile caches etc. (run_cell sets a fresh one per cell under the
                        cell dir; options.cache="returning" shares one per stack, the explicit returning-user arm)
  options.video_frames  "raw" (default: the frames Studio hands its MP4 encoder, lossless) or "mp4" (decode the
                        H.264 file Studio produced, i.e. what the gallery / HTTP backend delivers)

``load()`` returns the status fields that say what engaged (offload_policy, speed_optims, quant, attention...).
"""

from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

from .base import Backend, Render

LOAD_KEYS = ("model_kind", "family_override", "base_repo", "gguf_filename", "memory_mode", "speed_mode",
             "cpu_offload", "transformer_quant", "text_encoder_quant", "transformer_quant_fast_accum",
             "transformer_prequant_path", "attention_backend", "transformer_cache", "transformer_cache_threshold",
             "loras", "h3_task", "gpu_ids")
VIDEO_ONLY = ("h3_task",)
IMAGE_ONLY = ("cpu_offload", "transformer_quant_fast_accum", "transformer_prequant_path", "loras")
STATUS_KEYS = ("loaded", "repo_id", "family", "base_repo", "device", "dtype", "model_kind", "gguf_filename",
               "gguf_variant", "engine", "cpu_offload", "offload_policy", "vae_tiling", "memory_mode", "speed_mode",
               "speed_optims", "text_encoder_quant", "transformer_quant", "attention_backend", "transformer_cache",
               "transformer_cache_stats", "resolved", "workflows", "defaults", "supports_cfg", "has_audio",
               "cfg_batch", "zimage_cfg_split", "multi_gpu", "small_host", "bg_compile")


def engagement_snapshot() -> dict:
    """Cumulative engagement counters after a render, read in process: fused RoPE / AdaLN call counts, Dynamo's
    graph and graph-break totals, the ROCm compile probe's answers. Every field is best-effort; a module the tree
    under test does not have is simply absent, so the same code reads base and head. Never raises: a render is
    worth more than its counters."""
    try:
        return _engagement_snapshot()
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"[:200]}


def _engagement_snapshot() -> dict:
    out: dict = {}
    mod = sys.modules.get("core.inference.diffusion_rocm_fused")
    if mod is not None and isinstance(getattr(mod, "COUNTS", None), dict):
        out["rocm_fused"] = dict(mod.COUNTS)
    for name in ("core.inference.diffusion_flux2_rope", "core.inference.diffusion_zimage_fused",
                 "core.inference.diffusion_qwenimage21_fused"):
        m = sys.modules.get(name)
        counts = getattr(m, "COUNTS", None) if m is not None else None
        if isinstance(counts, dict):
            out[name.rsplit(".", 1)[-1]] = _jsonable(dict(counts))
    try:
        from torch._dynamo.utils import counters

        out["dynamo_unique_graphs"] = int(counters["stats"].get("unique_graphs", 0))
        out["dynamo_graph_breaks"] = int(sum(counters["graph_break"].values()))
        out["dynamo_calls_captured"] = int(counters["stats"].get("calls_captured", 0))
    except Exception:  # noqa: BLE001
        pass
    rc = sys.modules.get("core.inference.diffusion_rocm_compile")
    if rc is not None and isinstance(getattr(rc, "_result", None), dict):
        out["rocm_compile_probe"] = {str(k): _jsonable(v) for k, v in rc._result.items()}
    return out


def _jsonable(value: Any, depth: int = 0) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if depth > 4:
        return str(value)[:200]
    if isinstance(value, dict):
        return {str(k): _jsonable(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v, depth + 1) for v in list(value)[:64]]
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(), depth + 1)
    return str(value)[:200]


def status_subset(st: dict) -> dict:
    return {k: _jsonable(st.get(k)) for k in STATUS_KEYS if k in (st or {})}


# A pipeline's resident scheduler class -> (sampler, schedule) under the baseline UI's names (parity.ALIASES). Studio's
# generate record says sampler None / "default" when it kept the resident scheduler: this is what then ran.
SCHEDULER_CLASS = {"FlowMatchEulerDiscreteScheduler": ("euler", "simple"), "EulerDiscreteScheduler": ("euler", "normal"),
                   "EulerAncestralDiscreteScheduler": ("euler_ancestral", "normal"),
                   "DPMSolverMultistepScheduler": ("dpmpp_2m", "normal"), "UniPCMultistepScheduler": ("uni_pc", "simple"),
                   "FlowMatchLCMScheduler": ("lcm", "simple"), "LCMScheduler": ("lcm", "normal"),
                   "DDIMScheduler": ("ddim", "normal"),
                   # flow Euler whose grid comes from its own resolved_sigma_schedule() (studio_sampler reads it)
                   "MOSSImageScheduler": ("euler", None),
                   # LTX-2.5 distilled: Studio's default ancestral subclass of FlowMatchEuler (explicit stage sigmas)
                   "LTX25EulerAncestralScheduler": ("euler_ancestral", None),
                   # MiniMax-H3-Image: Studio's reference sampler, the "simple" pick over the grid-linear model
                   # sampling with an Euler step on x0 (its own docstring; the sigma dumps are the check)
                   "MiniMaxH3ImageScheduler": ("euler", "simple")}
SIGMA_FLAGS = (("use_karras_sigmas", "karras"), ("use_exponential_sigmas", "exponential"), ("use_beta_sigmas", "beta"))


def scheduler_names(scheduler: Any, card_map: Optional[dict] = None) -> tuple:
    """(sampler, schedule, class name) for a resident diffusers scheduler object (or a class name string). The
    card's ``studio.scheduler_map`` ({class: [sampler, schedule]}, the baseline's names for that family) wins over
    the generic table and its config flags: it is what a family's parity dumps showed."""
    if scheduler is None:
        return None, None, None
    name = scheduler if isinstance(scheduler, str) else type(scheduler).__name__
    if card_map and isinstance(card_map.get(name), (list, tuple)) and len(card_map[name]) == 2:
        return card_map[name][0], card_map[name][1], name
    sampler, sched = SCHEDULER_CLASS.get(name, (None, None))
    cfg = {} if isinstance(scheduler, str) else (getattr(scheduler, "config", None) or {})
    for flag, label in SIGMA_FLAGS:
        if cfg.get(flag):
            sched = label
    if sampler is not None and (cfg.get("use_flow_sigmas") or name.startswith("FlowMatch")) and sched == "normal":
        sched = "simple"
    if name == "UniPCMultistepScheduler" and cfg.get("use_flow_sigmas") and sched == "simple":
        # diffusers' own flow grid (linspace(1, 1/num_train_timesteps) then shifted) is NOT the baseline's "simple"
        # schedule: Wan2.2-TI2V-5B at shift 8 differs by up to 0.159 relative over 20 steps (r3 L4 parity dumps)
        sched = "flow_linspace"
    return sampler, sched, name


def resident_shift(scheduler: Any) -> Optional[float]:
    """The static shift a resident UniPC / DPM scheduler applies with flow sigmas (``flow_shift``: the baseline's
    ModelSamplingSD3 shift, same parameterisation); None otherwise. Studio applies the baseline's shift at load
    (bline_flow_shift) and its video result then reports flow_shift None. A FlowMatch scheduler's config ``shift``
    is NOT read: Studio's FLUX.3 klein carries shift 6.93 there for the baseline's flux shift 1.936 (another
    parameterisation of the same grid; the sigma dumps agree), so its value would void an equal pair."""
    cfg = getattr(scheduler, "config", None) or {}
    if type(scheduler).__name__ in ("UniPCMultistepScheduler", "DPMSolverMultistepScheduler") and \
            cfg.get("use_flow_sigmas") and cfg.get("flow_shift") is not None:
        return float(cfg["flow_shift"])
    return None


def studio_sampler(record: Optional[dict], scheduler: Any = None, recipe_sampler: Optional[str] = None,
                   card_default: Optional[str] = None, card_map: Optional[dict] = None) -> dict:
    """{sampler, scheduler, sampler_source} Studio ran for one generation. The per-generation record wins
    (``sampler`` is None unless Studio swapped the scheduler, ``schedule_type`` the resolved schedule); a None /
    "default" sampler is the resident one: a family recipe's own sampler (H3: res_multistep, Euler for PDD banks),
    else the pipeline's scheduler class, else the card's ``studio.default_sampler`` (HTTP: no class visible)."""
    record = record or {}
    s_cls, sched_cls, cls_name = scheduler_names(scheduler, card_map)
    sampler, src = record.get("sampler"), "studio generate record"
    if sampler in (None, "", "default"):
        if recipe_sampler:
            sampler, src = recipe_sampler, "studio family recipe (generate request shape)"
        elif s_cls:
            sampler, src = s_cls, f"pipeline default class {cls_name}"
        elif card_default:
            sampler, src = card_default, "card studio.default_sampler (no scheduler class visible)"
        else:
            sampler, src = None, "unknown (record says default, no scheduler class visible)"
    sched = record.get("schedule_type")
    sched_src = "studio generate record"
    own = getattr(scheduler, "resolved_sigma_schedule", None)
    if callable(own) and not isinstance(scheduler, str):
        # a scheduler that builds its sigma grid itself (MOSS-Image: sgm_uniform by default, whatever the request's
        # schedule_type says) reports the grid set_timesteps really used; the record's value would claim the
        # baseline's "simple" while the dumped sigmas differ (1.75e-2 over 26 on MOSS-Image SFT)
        try:
            val = own()
        except Exception:  # noqa: BLE001
            val = None
        if val:
            return {"sampler": sampler, "scheduler": str(val), "sampler_source": src,
                    "scheduler_source": f"{cls_name}.resolved_sigma_schedule()"}
    if sched in (None, "", "automatic"):
        sched, sched_src = sched_cls, (f"pipeline default class {cls_name}" if sched_cls else "unknown")
    return {"sampler": sampler, "scheduler": sched, "sampler_source": src, "scheduler_source": sched_src}


def studio_resolved(status: dict, cell: dict, steps: int, extra: Optional[dict] = None) -> dict:
    """Resolved generation settings for the parity gate: Studio's own ``resolved`` status block where it states
    them, the request otherwise (marked by ``source``); sampler / scheduler from the last generation's record
    (``extra["sampling"]``, studio_sampler), which the load-time status never states."""
    res = (status or {}).get("resolved") or {}
    res = res if isinstance(res, dict) else {}
    o = cell.get("options") or {}
    gen = o.get("generate") or {}
    out = {"steps": steps, "cfg": gen.get("guidance", cell.get("guidance")), "width": cell.get("width"),
           "height": cell.get("height"), "frames": cell.get("frames"), "sampler": None, "scheduler": None,
           "shift": gen.get("flow_shift")}
    for k_in, k_out in (("steps", "steps"), ("num_inference_steps", "steps"), ("guidance", "cfg"),
                        ("guidance_scale", "cfg"), ("true_cfg_scale", "cfg"), ("sampler", "sampler"),
                        ("scheduler", "scheduler"), ("flow_shift", "shift"), ("shift", "shift"), ("width", "width"),
                        ("height", "height"), ("num_frames", "frames")):
        if res.get(k_in) is not None:
            out[k_out] = res[k_in]
    if extra and extra.get("flow_shift") is not None:
        out["shift"] = extra["flow_shift"]
    if extra and extra.get("applied_flow_shift") is not None:
        out["shift"] = extra["applied_flow_shift"]
    if out.get("shift") is None and extra and extra.get("resident_shift") is not None:
        # nothing in the request or the result names a shift: the one the resident scheduler applies is what ran
        out["shift"], out["shift_source"] = extra["resident_shift"], "resident scheduler config"
    samp = (extra or {}).get("sampling") or {}
    for k in ("sampler", "scheduler"):
        if samp.get(k) is not None:
            out[k] = samp[k]
    if samp:
        out["sampler_source"], out["scheduler_source"] = samp.get("sampler_source"), samp.get("scheduler_source")
    out["source"] = "studio status.resolved" if res else "request (Studio did not state resolved settings)"
    return out


def studio_precision(status: dict) -> dict:
    """Bare per-component tokens from Studio's status. The VAE is None (unknown): Studio's status states no VAE dtype,
    and the pipeline dtype is wrong for families whose VAE runs in another one (Wan fp32, H3 fp16)."""
    st = status or {}
    dtype = st.get("dtype")
    tq, teq = st.get("transformer_quant"), st.get("text_encoder_quant")
    pick = (lambda q: q if q not in (None, "none", "off", False) else dtype)  # noqa: E731
    gg = st.get("gguf_variant") or st.get("gguf_filename")
    trans = f"gguf:{gg}" if gg and tq in (None, "none", "off", False) else pick(tq)
    te = pick(teq)
    # Studio's own decision record for the encoder is strong evidence: "off" after an automatic int8 pick means the
    # hosted int8 encoder was unavailable and the dense one ran (FLUX.3 klein on an offline cell), which a pipeline
    # dtype token alone cannot tell from a quantized encoder
    res = (st.get("resolved") or {}).get("text_encoder_quant")
    if isinstance(res, dict) and res.get("status") == "applied" and res.get("value") is not None:
        val = str(res["value"]).strip().lower()
        dense = val in ("off", "none", "false", "0")
        te = {"weights": dtype if dense else val, "compute": dtype if dense else val,
              "evidence": f"status.resolved.text_encoder_quant={res['value']!r}: {res.get('reason') or ''}"[:300]}
    return {"transformer": trans, "text_encoder": te, "vae": st.get("vae_dtype")}


DTYPE_TOKEN = {"torch.float32": "fp32", "torch.float16": "fp16", "torch.bfloat16": "bf16"}


def module_dtypes(pipe: Any, names: tuple = ("vae", "audio_vae")) -> dict:
    """{component: {weights, compute, evidence}} for a pipeline's dense floating-point modules (the most common
    parameter dtype; a module whose weights are not all one float dtype is left out)."""
    out = {}
    for name in names:
        mod = getattr(pipe, name, None) if pipe is not None else None
        params = getattr(mod, "parameters", None)
        if not callable(params):
            continue
        try:
            kinds = {str(p.dtype) for p in params()}
        except Exception:  # noqa: BLE001
            continue
        if len(kinds) == 1 and next(iter(kinds)) in DTYPE_TOKEN:
            tok = DTYPE_TOKEN[next(iter(kinds))]
            out[name] = {"weights": tok, "compute": tok, "evidence": f"pipe.{name} parameters all {tok} (loaded)"}
    return out


def family_sampler(engine: Any) -> Optional[str]:
    """The sampler name a Studio family states on its pipe (``pipe._unsloth_<family>_sampler``: LTX-2.5 sets
    "euler_ancestral" for the distilled model, "euler" for dev or UNSLOTH_LTX25_SAMPLER=euler), None when absent.
    The parity hook only sees the base class whose set_timesteps ran (FlowMatchEulerDiscreteScheduler), which read
    as euler / simple for an ancestral sampler."""
    state = getattr(engine, "_state", None)
    pipe = getattr(state, "pipe", None) if state is not None else None
    if pipe is None:
        return None
    for name in sorted(n for n in vars(pipe) if n.startswith("_unsloth_") and n.endswith("_sampler")) \
            if hasattr(pipe, "__dict__") else []:
        val = getattr(pipe, name, None)
        if isinstance(val, str) and val:
            return val
    return None


def resident_scheduler(engine: Any) -> Any:
    """The pipeline's resident scheduler object of an in-process Studio engine, None when not reachable."""
    state = getattr(engine, "_state", None)
    pipe = getattr(state, "pipe", None) if state is not None else None
    return getattr(pipe, "scheduler", None) if pipe is not None else None


class ProgressPoller:
    """Polls a generate-progress callable every ``interval`` s on a thread while a render runs, feeding
    backend.mark_step / mark_phase. Resolution is the poll interval; BlineUI's event feed has none."""

    def __init__(self, backend, fn, interval: float = 0.05):
        import threading

        self.backend, self.fn, self.interval = backend, fn, interval
        self.phases: list = []
        self._stop = threading.Event()
        self._t = threading.Thread(target = self._run, daemon = True)
        self.t_start = time.time()

    def _run(self) -> None:
        last_phase = None
        while not self._stop.is_set():
            try:
                p = self.fn() or {}
            except Exception:  # noqa: BLE001 - a missed poll only lowers resolution
                p = {}
            now = time.time()
            step, phase = int(p.get("step") or 0), p.get("phase")
            if p.get("active") and step > 0 and phase in (None, "denoise"):
                self.backend.mark_step(step, p.get("total_steps"), now)
            if p.get("active") and phase and phase != last_phase:
                self.phases.append((round(now - self.t_start, 3), phase))
                if phase == "denoise" and "denoise_start" not in (getattr(self.backend, "phase_events", None) or {}):
                    self.backend.mark_phase("denoise_start", now)  # poll resolution, like denoise_end
                if last_phase == "denoise" and phase != "denoise":
                    self.backend.mark_phase("denoise_end", now)
                last_phase = phase
            self._stop.wait(self.interval)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(timeout = 5)
        return False


def studio_import_path(opts: dict) -> dict:
    """Resolve the Studio tree for this process and put its backend first on sys.path (ahead of anything on
    PYTHONPATH named ``models`` / ``core`` / ``utils``). Returns the resolve_tree dict."""
    here = Path(__file__).resolve().parents[1]
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    import setup_studio

    src = opts.get("studio_src") or os.environ.get("DIFFUSION_BENCH_STUDIO_SRC")
    if src:
        tree = setup_studio.resolve_tree(src, python = sys.executable)
    else:
        tree = setup_studio.resolve_tree("pypi", python = sys.executable)
    backend = tree["studio_backend"]
    if backend in sys.path:
        sys.path.remove(backend)
    sys.path.insert(0, backend)
    for name in ("models", "core", "utils", "routes", "auth", "storage", "hub", "state"):
        mod = sys.modules.get(name)
        if mod is None:
            continue
        paths = [getattr(mod, "__file__", None) or ""] + [str(p) for p in (getattr(mod, "__path__", None) or [])]
        if not any(p.startswith(backend) for p in paths if p):
            del sys.modules[name]  # a same-named package from elsewhere on the path was imported first
    home = opts.get("studio_home") or os.environ.get("UNSLOTH_STUDIO_HOME")
    if not home:
        import common as C

        home = str(C.WS / "temp" / "diffusion_bench" / "studio_home_inproc")
    os.environ["UNSLOTH_STUDIO_HOME"] = str(home)
    Path(home).mkdir(parents = True, exist_ok = True)
    return tree


def load_kwargs(opts: dict, kind: str) -> dict:
    kw = {k: opts[k] for k in LOAD_KEYS if k in opts and opts[k] is not None}
    kw.update(opts.get("load") or {})
    drop = IMAGE_ONLY if kind == "video" else VIDEO_ONLY
    for k in drop:
        kw.pop(k, None)
    if kw.get("loras"):
        kw["loras"] = [tuple(x) if isinstance(x, (list, tuple)) else (x["id"], float(x.get("weight", 1.0)))
                       for x in kw["loras"]]
    return kw


def wait_ready(backend: Any, timeout_s: float, poll_s: float = 0.5) -> None:
    deadline = time.time() + timeout_s
    null_since = None
    while True:
        p = backend.load_progress() or {}
        phase = p.get("phase")
        if phase == "ready":
            return
        if phase == "error":
            raise RuntimeError(f"load failed: {p.get('error')}")
        if phase is None:
            null_since = null_since or time.time()
            if time.time() - null_since > 30:
                raise RuntimeError("load thread ended without a model or an error")
        if time.time() > deadline:
            raise TimeoutError(f"load not ready after {timeout_s}s: {p}")
        time.sleep(poll_s)


class StudioBackend(Backend):
    name = "studio"
    in_process = True

    def __init__(self, cell: dict, out: Path):
        super().__init__(cell, out)
        self.tree = studio_import_path(self.opts)
        self.kind = cell.get("kind", "image")
        self.engine: Any = None
        self._captured: dict = {}

    # ------------------------------------------------------------------------------------------ lifecycle
    def load(self) -> dict:
        model = self.cell["model"]
        kw = load_kwargs(self.opts, self.kind)
        local = bool(self.opts.get("local_files_only", True))
        timeout = float(self.opts.get("load_timeout_s", 3600))
        if self.kind == "video":
            from core.inference import video as V

            orig = V.VideoBackend._encode_mp4
            captured = self._captured

            def grab(video_frames, fps, audio, pipe, **k):
                captured["frames"], captured["fps"] = video_frames, fps
                return orig(video_frames, fps, audio, pipe, **k)

            V.VideoBackend._encode_mp4 = staticmethod(grab)
            self.engine = V.VideoBackend()
            self.engine.begin_load(model, local_files_only = local, **kw)
            wait_ready(self.engine, timeout)
            st = self.engine.status()
        else:
            from core.inference.diffusion import DiffusionBackend

            self.engine = DiffusionBackend()
            if self.opts.get("load_via", "begin_load") == "begin_load":
                self.engine.begin_load(model, local_files_only = local, **kw)
                wait_ready(self.engine, timeout)
                st = self.engine.status()
            else:
                st = self.engine.load_pipeline(model, local_files_only = local, **kw) or {}
                st = {**st, **(self.engine.status() or {})}
        return {**status_subset(st), "load_kwargs": _jsonable(kw), "studio_rev": self.tree.get("rev")}

    def render(self, row: dict, steps: int) -> Render:
        gen = dict(self.opts.get("generate") or {})
        c = self.cell
        self._steps = steps
        poll = getattr(self.engine, "generate_progress", None)
        if self.kind == "video":
            self._captured.clear()
            with ProgressPoller(self, poll or dict) as pp:
                res = self.engine.generate(prompt = row["prompt"], negative_prompt = c.get("negative_prompt"),
                                           width = c.get("width"), height = c.get("height"),
                                           num_frames = c.get("frames"), fps = c.get("fps"), steps = steps,
                                           guidance = c.get("guidance"), seed = row["seed"], **gen)
            self.request_facts["studio_phases"] = pp.phases
            frames = self._captured.get("frames")
            if frames is None or self.opts.get("video_frames") == "mp4":
                from studio_client import decode_mp4

                frames, _ = decode_mp4(res["mp4_bytes"])
            extra = {k: res.get(k) for k in ("seed", "width", "height", "num_frames", "fps", "duration_s",
                                              "flow_shift", "offload_policy")}
            extra["engaged"] = engagement_snapshot()
            shape = getattr(self.engine, "_last_request_shape", None) or {}
            recipe = ((shape.get("h3_recipe") or {}) if isinstance(shape, dict) else {}).get("sampler") \
                or family_sampler(self.engine)
            extra["sampling"] = studio_sampler(res, resident_scheduler(self.engine), recipe,
                                               self.opts.get("studio_default_sampler"),
                                               self.opts.get("studio_scheduler_map"))
            extra["resident_shift"] = resident_shift(resident_scheduler(self.engine))
            self._last_extra = extra
            return Render(frames = frames, fps = res.get("fps"), extra = extra)
        with ProgressPoller(self, poll or dict) as pp:
            res = self.engine.generate(prompt = row["prompt"], negative_prompt = c.get("negative_prompt"),
                                       width = c.get("width"), height = c.get("height"), steps = steps,
                                       guidance = c.get("guidance"), seed = row["seed"], **gen)
        self.request_facts["studio_phases"] = pp.phases
        images = res["images"]
        extra = {"seed": res.get("seed"), "engaged": engagement_snapshot(),
                 "applied_flow_shift": res.get("applied_flow_shift"),
                 "sampling": studio_sampler(res, resident_scheduler(self.engine), family_sampler(self.engine),
                                            self.opts.get("studio_default_sampler"),
                                            self.opts.get("studio_scheduler_map")),
                 "resident_shift": resident_shift(resident_scheduler(self.engine))}
        self._last_extra = extra
        if len(images) > 1:
            extra["batch"] = len(images)
            extra["seeds"] = res.get("seeds")
        return Render(image = images[0], extra = extra)

    def status(self) -> dict:
        return status_subset(self.engine.status()) if self.engine is not None else {}

    def resolved_settings(self) -> dict:
        return studio_resolved(self.status(), self.cell, getattr(self, "_steps", self.cell.get("steps")),
                               getattr(self, "_last_extra", None))

    def precision(self) -> dict:
        out = studio_precision(self.status())
        # the VAEs' dtypes as loaded (Studio's status states none): strong evidence where the card cannot say
        state = getattr(self.engine, "_state", None)
        out.update({k: v for k, v in module_dtypes(getattr(state, "pipe", None) if state is not None else None).items()
                    if out.get(k) is None})
        return out

    def close(self) -> None:
        if self.engine is not None:
            try:
                self.engine.unload()
            finally:
                self.engine = None
                gc.collect()
                torch = sys.modules.get("torch")
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def trees(self) -> dict:
        return {"studio": self.tree.get("tree")}
