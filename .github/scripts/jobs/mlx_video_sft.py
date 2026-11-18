"""Unsloth sample job: mlx_video_sft. Studio's MLX video path end to end on Apple Silicon:
Studio's own formatter + `_adapt_for_mlx_vlm` (read from the checkout under test) feed zoo's
MLXTrainer real video clips. Checks clips reach the processor as `pixel_values_videos` and that
per-part decode hints (`nframes`) survive the adapter. Never imports torch directly."""

from __future__ import annotations

import ast
import math
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

DEFAULT_MODEL = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
TINY_MODEL = DEFAULT_MODEL
NFRAMES = 4
COLOURS = [
    ("red", (220, 30, 30)),
    ("green", (30, 200, 30)),
    ("blue", (30, 30, 220)),
    ("yellow", (230, 220, 30)),
]


def _write_clip(
    path,
    rgb,
    frames = 16,
    size = 64,
    fps = 8,
):
    import av
    import numpy as np
    with av.open(str(path), "w") as out:
        stream = out.add_stream("mpeg4", rate = fps)
        stream.width = stream.height = size
        stream.pix_fmt = "yuv420p"
        for i in range(frames):
            img = np.zeros((size, size, 3), np.uint8)
            x = (i * 3) % (size - 16)
            img[16:32, x : x + 16] = rgb
            for pkt in stream.encode(av.VideoFrame.from_ndarray(img, format = "rgb24")):
                out.mux(pkt)
        for pkt in stream.encode():
            out.mux(pkt)


def _studio_adapter(backend):
    src = (backend / "core" / "training" / "worker.py").read_text(encoding = "utf-8")
    names = {"_adapt_for_mlx_vlm", "_own_mlx_video_values"}
    nodes = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = {
        "_resize_mlx_vlm_image": lambda i, *a, **k: i,
        "_resize_mlx_vlm_images": lambda i, *a, **k: i,
    }
    exec(compile(ast.Module(nodes, []), "worker.py", "exec"), ns)
    return ns["_adapt_for_mlx_vlm"]


def main(argv = None):
    argv = sys.argv[1:] if argv is None else argv
    p = C.base_parser("mlx_video_sft", DEFAULT_MODEL, TINY_MODEL, default_steps = 4)
    p.add_argument(
        "--studio-root", default = ".", help = "unsloth checkout whose studio/backend is under test"
    )
    p.add_argument("--rank", type = int, default = 8)
    a = C.resolve_args(p, argv)
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    with C.JobRecorder(a, backend_hint = None) as rec:
        if not (platform.system() == "Darwin" and platform.machine() == "arm64"):
            rec.skip(f"MLX needs Apple Silicon, this is {platform.system()}-{platform.machine()}")
        import unsloth  # noqa: F401  must precede mlx_lm / transformers

        is_mlx = bool(getattr(unsloth, "_IS_MLX", False))
        rec.set_backend("unsloth-mlx" if is_mlx else "unsloth-no-mlx")
        if not rec.check("mlx_active", is_mlx, f"unsloth._IS_MLX={is_mlx}"):
            return
        rec.backend_check()

        backend = (Path(a.studio_root) / "studio" / "backend").resolve()
        sys.path.insert(0, str(backend))
        from utils.datasets import format_and_template_dataset
        from datasets import Dataset

        work = Path(tempfile.mkdtemp(prefix = "mlx_video_sft_"))
        rows = []
        for i, (name, rgb) in enumerate(COLOURS):
            clip = work / f"clip{i}.mp4"
            _write_clip(clip, rgb)
            rows.append(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "video", "video": str(clip), "nframes": NFRAMES},
                                {"type": "text", "text": "What colour is the moving square?"},
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": f"The square is {name}."}],
                        },
                    ]
                }
            )
        info = format_and_template_dataset(
            Dataset.from_list(rows),
            model_name = a.model,
            tokenizer = None,
            is_vlm = True,
            dataset_name = "local",
        )
        if not rec.check("studio_format_ok", bool(info.get("success")), info.get("errors")):
            return
        try:
            data = _studio_adapter(backend)(list(info["dataset"]))
        except Exception as e:
            rec.check("studio_adapter_keeps_video", False, f"{type(e).__name__}: {e}")
            return
        rec.check(
            "studio_adapter_keeps_video",
            all(r.get("videos") for r in data),
            [r.get("videos") for r in data],
        )
        placeholder = next(
            p for m in data[0]["messages"] for p in m["content"] if p.get("type") == "video"
        )
        rec.check(
            "studio_adapter_keeps_decode_hints", placeholder.get("nframes") == NFRAMES, placeholder
        )

        import inspect
        from unsloth_zoo.mlx import utils as zu

        rec.check(
            "zoo_has_video_collation",
            "all_videos" in inspect.signature(zu._processor_vlm_inputs).parameters,
            str(inspect.signature(zu._processor_vlm_inputs)),
        )
        _, vids, _ = zu._extract_vlm_media(data[0], data[0]["messages"], None)
        frames = int(getattr(vids[0], "shape", [len(vids[0])])[0]) if vids else 0
        rec.check(
            "decoded_frames_match_hint", frames == NFRAMES, f"{frames} frames, hint {NFRAMES}"
        )

        seen = {"calls": 0, "video_batches": 0}
        orig = zu._processor_vlm_inputs

        def counting(*args, **kw):
            out = orig(*args, **kw)
            seen["calls"] += 1
            if (
                out is not None
                and hasattr(out, "get")
                and out.get("pixel_values_videos") is not None
            ):
                seen["video_batches"] += 1
            return out

        zu._processor_vlm_inputs = counting

        import dataclasses
        import mlx.core as mx
        from unsloth_zoo.mlx.loader import FastMLXModel
        from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

        C.seed_everything(a.seed)
        mx.random.seed(a.seed)
        t0 = time.perf_counter()
        model, tokenizer = FastMLXModel.from_pretrained(
            a.model,
            load_in_4bit = True,
            text_only = None,
            random_state = a.seed,
            token = os.environ.get("HF_TOKEN") or None,
        )
        rec.summary(load_s = round(time.perf_counter() - t0, 2))
        model = FastMLXModel.get_peft_model(
            model,
            r = a.rank,
            lora_alpha = 2 * a.rank,
            finetune_vision_layers = False,
            finetune_language_layers = True,
            random_state = a.seed,
        )
        from mlx_sft import _lora_fingerprint

        before = _lora_fingerprint(model)
        rec.check("has_trainable_params", before["numel"] > 0, before["numel"])
        cfg = dict(
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 1,
            max_steps = a.max_steps,
            learning_rate = a.lr,
            warmup_steps = 0,
            lr_scheduler_type = "linear",
            optim = "adamw",
            weight_decay = 0.0,
            max_grad_norm = 1.0,
            logging_steps = 1,
            max_seq_length = a.max_seq_length,
            seed = a.seed,
            use_cce = False,
            compile = False,
            gradient_checkpointing = False,
            output_dir = str(work / "out"),
            save_steps = 0,
            eval_steps = 0,
            report_grad_norm = True,
        )
        fields = {f.name for f in dataclasses.fields(MLXTrainingConfig)}
        trainer = MLXTrainer(
            model = model,
            tokenizer = tokenizer,
            train_dataset = data,
            args = MLXTrainingConfig(**{k: v for k, v in cfg.items() if k in fields}),
        )

        def on_step(
            step,
            total,
            loss,
            lr,
            tok_s,
            peak_gb,
            elapsed,
            num_tokens,
            grad_norm = None,
            *_,
            **__,
        ):
            rec.step(
                step = int(step),
                loss = float(loss),
                grad_norm = None if grad_norm is None else float(grad_norm),
                learning_rate = float(lr),
                time_ms = None,
                tokens = None,
                tokens_per_s = float(tok_s) if tok_s is not None else None,
                peak_mem_gb = float(peak_gb) if peak_gb is not None else None,
            )
            print(f"  step {step}/{total} loss={loss:.4f}", flush = True)

        trainer.add_step_callback(on_step)
        t1 = time.perf_counter()
        trainer.train()
        rec.summary(
            train_s = round(time.perf_counter() - t1, 2),
            processor_calls = seen["calls"],
            video_batches = seen["video_batches"],
        )
        steps = rec.data["steps"]
        losses = [s["loss"] for s in steps]
        rec.check("completed_steps", len(steps) == a.max_steps, f"{len(steps)} / {a.max_steps}")
        rec.check(
            "finite_loss",
            bool(losses) and all(math.isfinite(x) and 0 <= x < 50 for x in losses),
            losses,
        )
        gns = [s["grad_norm"] for s in steps if s.get("grad_norm") is not None]
        rec.check(
            "grad_norm_bounded",
            bool(gns) and all(math.isfinite(g) and 0 < g < 1e3 for g in gns),
            gns,
        )
        rec.check(
            "videos_reached_processor",
            seen["video_batches"] >= a.max_steps,
            f"{seen['video_batches']} of {seen['calls']} processor calls carried pixel_values_videos",
        )
        rec.check("adapter_changed", *C.adapter_changed(before, _lora_fingerprint(model)))


if __name__ == "__main__":
    main()
