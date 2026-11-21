"""Plain diffusers, in process: the framework-neutral reference every other backend is set against.

Cell fields / options:
  model            (cell["model"] or options.model) a local pipeline directory or a Hub repo id
  pipeline         class name under diffusers (default: DiffusionPipeline, which reads model_index.json)
  dtype            auto (default: bf16 on compute capability 8.0+ and non-NVIDIA devices, fp16 below, e.g. a T4)
                   | bf16 | fp16 | fp32
  device           cuda (default) | cpu | mps
  offload          none (default) | model (enable_model_cpu_offload) | sequential (enable_sequential_cpu_offload)
                   | group (enable_group_offload on every nn.Module component, block_level, CUDA streams)
  group_blocks     num_blocks_per_group for offload=group (default 1)
  compile          false (default) | regional (compile_repeated_blocks on the denoiser, fullgraph) | full
                   (torch.compile of the whole denoiser)
  attention_backend  set_attention_backend on the denoiser, e.g. flash, _native_cudnn, sage, native
  cfg_kw           the call kwarg the cell's guidance goes to: guidance_scale (default) or true_cfg_scale (Qwen)
  generator_device cpu (default: seeds reproduce across GPUs) | cuda
  local_files_only true (default), variant, call_kwargs {extra pipeline call kwargs}
  scheduler        {"class": "...", **from_config overrides} to swap the scheduler

Video cells (kind=video) pass num_frames = cell frames and return the frames.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from .base import Backend, Render

DTYPES = {"bf16": "bfloat16", "bfloat16": "bfloat16", "fp16": "float16", "float16": "float16", "fp32": "float32",
          "float32": "float32"}


class DiffusersBackend(Backend):
    name = "diffusers"
    in_process = True

    def load(self) -> dict:
        import diffusers
        import torch

        o = self.opts
        model = o.get("model") or self.cell.get("model")
        if not model:
            raise ValueError("diffusers cell needs model (a pipeline dir or repo id)")
        local = bool(o.get("local_files_only", True))
        looks_like_path = str(model).startswith(("/", ".", "~")) or str(model).count("/") != 1
        if looks_like_path and not Path(model).exists():
            raise FileNotFoundError(f"diffusers model dir not found: {model}")
        want = str(o.get("dtype", "auto")).lower()
        if want == "auto":
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            import common as C

            cc = None
            if torch.cuda.is_available() and not getattr(torch.version, "hip", None):
                cc = "%d.%d" % torch.cuda.get_device_capability(0)
            want = C.auto_dtype(cc)
        self.dtype_resolved = want
        dtype = getattr(torch, DTYPES[want])
        self.device = o.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        cls = getattr(diffusers, o.get("pipeline") or "DiffusionPipeline")
        kw = {"torch_dtype": dtype, "local_files_only": local}
        if o.get("variant"):
            kw["variant"] = o["variant"]
        t0 = time.perf_counter()
        pipe = cls.from_pretrained(str(model), **kw)
        from_pretrained_s = round(time.perf_counter() - t0, 2)
        if o.get("scheduler"):
            spec = dict(o["scheduler"])
            scls = getattr(diffusers, spec.pop("class"))
            pipe.scheduler = scls.from_config(pipe.scheduler.config, **spec)
        self.denoiser = getattr(pipe, "transformer", None) or getattr(pipe, "unet", None)
        offload = (o.get("offload") or "none").lower()
        if offload == "model":
            pipe.enable_model_cpu_offload(device = self.device)
        elif offload == "sequential":
            pipe.enable_sequential_cpu_offload(device = self.device)
        elif offload == "group":
            import torch.nn as nn

            for name, comp in pipe.components.items():
                if isinstance(comp, nn.Module) and hasattr(comp, "enable_group_offload"):
                    comp.enable_group_offload(onload_device = torch.device(self.device), offload_type = "block_level",
                                              num_blocks_per_group = int(o.get("group_blocks", 1)), use_stream = True)
                elif isinstance(comp, nn.Module):
                    comp.to(self.device)
        elif offload == "none":
            pipe.to(self.device)
        else:
            raise ValueError(f"diffusers offload must be none | model | sequential | group, got {offload!r}")
        attn = o.get("attention_backend")
        attn_set = None
        if attn:
            if not hasattr(self.denoiser, "set_attention_backend"):
                raise RuntimeError(f"{type(self.denoiser).__name__} has no set_attention_backend (diffusers "
                                   f"{diffusers.__version__})")
            self.denoiser.set_attention_backend(attn)
            attn_set = attn
        comp = (o.get("compile") or "none")
        comp = "regional" if comp is True else str(comp).lower()
        if comp == "regional":
            if hasattr(self.denoiser, "compile_repeated_blocks"):
                self.denoiser.compile_repeated_blocks(fullgraph = True)
            else:
                raise RuntimeError(f"{type(self.denoiser).__name__} has no compile_repeated_blocks "
                                   f"(no _repeated_blocks declared)")
        elif comp == "full":
            self.denoiser.compile(fullgraph = False)
        elif comp not in ("none", "false"):
            raise ValueError(f"diffusers compile must be false | regional | full, got {comp!r}")
        self.pipe = pipe
        return {"backend": "diffusers", "pipeline": type(pipe).__name__, "denoiser": type(self.denoiser).__name__,
                "scheduler": type(pipe.scheduler).__name__, "dtype": str(dtype).replace("torch.", ""),
                "device": self.device, "offload": offload, "compile": comp, "attention_backend": attn_set,
                "from_pretrained_s": from_pretrained_s, "diffusers": diffusers.__version__, "torch": torch.__version__}

    def precision(self) -> dict:
        d = getattr(self, "dtype_resolved", None)
        return {"transformer": d, "text_encoder": d, "vae": d} if d else {}

    def render(self, row: dict, steps: int) -> Render:
        import torch

        o, c = self.opts, self.cell
        gen = torch.Generator(o.get("generator_device", "cpu")).manual_seed(int(row["seed"]))
        kw = {"prompt": row["prompt"], "num_inference_steps": int(steps), "width": int(c["width"]),
              "height": int(c["height"]), "generator": gen, **dict(o.get("call_kwargs") or {})}
        if c.get("guidance") is not None:
            kw[o.get("cfg_kw", "guidance_scale")] = float(c["guidance"])
        if c.get("negative_prompt"):
            kw["negative_prompt"] = c["negative_prompt"]
        if c.get("kind") == "video":
            kw["num_frames"] = int(c.get("frames") or 49)
            kw.setdefault("output_type", "np")
            with torch.inference_mode():
                out = self.pipe(**kw)
            return Render(frames = out.frames[0], fps = int(c.get("fps") or 16))
        with torch.inference_mode():
            out = self.pipe(**kw)
        return Render(image = out.images[0])

    def status(self) -> dict:
        return {}

    def close(self) -> None:
        import gc

        self.pipe = None
        self.denoiser = None
        gc.collect()
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def trees(self) -> dict:
        return {}
