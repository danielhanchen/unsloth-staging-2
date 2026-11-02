# SPDX-License-Identifier: AGPL-3.0-only
"""Real-Metal capability probes dispatched by mlx_speculative_ab.py."""

import base64
import functools
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace


def worker(args):
    os.environ["UNSLOTH_IS_PRESENT"] = "1"
    os.environ["HF_DOWNLOAD_OK"] = "1"
    os.environ.pop("UNSLOTH_FORCE_GPU_PATH", None)
    sys.path[:0] = [str(Path(args.worker) / "studio/backend"), args.zoo_source]
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten
    import unsloth_zoo.mlx.speculative as runtime
    from core.inference.mlx_inference import MLXInferenceBackend

    assert (
        Path(runtime.__file__).resolve() == Path(args.zoo_source) / "unsloth_zoo/mlx/speculative.py"
    )
    assert mx.metal.is_available()
    mx.set_default_device(mx.gpu)
    config = args.model
    if args.scenario.startswith("video"):
        config = SimpleNamespace(identifier = args.model, is_vision = True, is_lora = False)
    folder = Path(args.out).parent
    adapter_parameters = 0
    if args.scenario == "lora":
        from unsloth import FastLanguageModel

        model, processor = FastLanguageModel.from_pretrained(
            args.model, max_seq_length = 2048, load_in_4bit = True, text_only = False, random_state = 3407
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r = 2,
            lora_alpha = 4,
            lora_dropout = 0.0,
            target_modules = ["q_proj", "v_proj"],
            use_gradient_checkpointing = False,
            random_state = 3407,
        )
        updates = [
            (k, mx.ones_like(v) * 0.001)
            for k, v in tree_flatten(model.trainable_parameters())
            if "lora_b" in k.lower()
        ]
        assert updates
        adapter_parameters = sum(v.size for _, v in updates)
        model.update(tree_unflatten(updates))
        adapter = folder / ("adapter_" + args.arm)
        model.save_pretrained_merged(str(adapter), tokenizer = processor, save_method = "lora")
        config = SimpleNamespace(
            identifier = str(adapter), is_lora = True, is_vision = True, base_model = args.model
        )
        del model, processor
        gc.collect()
        mx.clear_cache()

    backend = MLXInferenceBackend()
    backend.load_model(
        config,
        speculative_type = "mtp",
        max_seq_length = 2048,
        **({"kv_quant": "4"} if args.scenario == "kv4" else {}),
    )
    if args.arm == "head":
        assert backend._speculative_draft is not None, backend.models[backend.active_model_name]
    caches, pixels, quantized_hits = [], [], []
    wrapped_cache_types = set()
    if args.scenario == "kv4":
        assert backend._kv_quant["kv_bits"] == 4
        prepare = backend._prepare_kv_entries

        def capture(entries):
            entries = prepare(entries)
            caches.extend(entries)
            for entry in entries:
                cls = type(entry)
                if getattr(entry, "bits", None) != 4 or cls in wrapped_cache_types:
                    continue
                update = getattr(cls, "update_and_fetch", None)
                if update is None:
                    continue
                wrapped_cache_types.add(cls)

                def observe(original):
                    @functools.wraps(original)
                    def call(cache, *a, **kw):
                        value = original(cache, *a, **kw)
                        if any(
                            isinstance(v, mx.array) and v.dtype == mx.uint32 and v.size > 0
                            for _, v in tree_flatten(cache.state)
                        ):
                            quantized_hits.append(type(cache).__name__)
                        return value

                    return call

                cls.update_and_fetch = observe(update)
            return entries

        backend._prepare_kv_entries = capture
    if args.scenario.startswith("video"):
        embed = backend._model.get_input_embeddings

        @functools.wraps(embed)
        def capture_pixels(*a, **kw):
            for value in [*a[1:], kw.get("pixel_values"), kw.get("pixel_values_videos")]:
                if isinstance(value, mx.array) and value.ndim >= 3:
                    pixels.append(list(value.shape))
            return embed(*a, **kw)

        backend._model.get_input_embeddings = capture_pixels
    if args.scenario == "lora":
        loaded = [
            (k, v) for k, v in tree_flatten(backend._model.parameters()) if "lora_b" in k.lower()
        ]
        assert loaded and all(bool(mx.any(v != 0).item()) for _, v in loaded)

    def clip(color):
        import av
        import numpy as np

        path = folder / f"{args.arm}_{color}.mp4"
        with av.open(str(path), "w") as container:
            stream = container.add_stream("mpeg4", rate = 4)
            stream.width = stream.height = 256
            stream.pix_fmt = "yuv420p"
            image = np.zeros((256, 256, 3), dtype = np.uint8)
            image[:, :, 0 if color == "red" else 2] = 255
            for _ in range(8):
                for packet in stream.encode(av.VideoFrame.from_ndarray(image, format = "rgb24")):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        return base64.b64encode(path.read_bytes()).decode()

    colors = ["red", "blue"] if args.scenario == "video_batch" else ["red"]
    requests = []
    for color in colors:
        request = dict(
            messages = [
                {
                    "role": "user",
                    "content": "Name the main color in the clip."
                    if args.scenario.startswith("video")
                    else "Name the capital of France.",
                }
            ],
            temperature = 0.0,
            top_p = 1.0,
            top_k = 0,
            seed = 3407,
            enable_thinking = False,
            max_new_tokens = 24,
        )
        if args.scenario.startswith("video"):
            request["videos"] = [clip(color)]
        requests.append(request)
    replies, stats = [""] * len(requests), []
    unavailable = backend.batch_unavailable_reason(requests) if len(requests) > 1 else None
    if len(requests) > 1 and args.arm == "head":
        assert unavailable is None, unavailable
        for row, text in backend.generate_chat_batch(requests):
            if text is not None:
                replies[row] = text
        stats = backend.last_batch_generation_stats
    else:
        for row, request in enumerate(requests):
            for replies[row] in backend.generate_chat_response(**request):
                pass
            stats.append(backend.last_generation_stats)
    assert all(replies) and all(stat and stat["usage"]["completion_tokens"] > 0 for stat in stats)
    proposed = sum(stat["timings"].get("draft_n", 0) for stat in stats)
    assert (proposed > 0) == (args.arm == "head"), stats
    if args.scenario.startswith("video"):
        assert pixels, "video pixels did not reach the model"
        assert all(color in text.lower() for color, text in zip(colors, replies)), replies
    quantized = [
        type(entry).__name__
        for entry in caches
        if getattr(entry, "bits", None) == 4
        and any(
            isinstance(v, mx.array) and v.dtype == mx.uint32 and v.size > 0
            for _, v in tree_flatten(entry.state)
        )
    ]
    if args.scenario == "kv4":
        assert (
            quantized or quantized_hits
        ), "requested KV quantization did not produce packed cache tensors"
    Path(args.out).write_text(
        json.dumps(
            dict(
                scenario = args.scenario,
                model = args.model,
                replies = replies,
                stats = stats,
                metal = True,
                device = str(mx.default_device()),
                pixels = pixels,
                quantized_caches = quantized,
                quantized_cache_updates = quantized_hits,
                adapter_parameters = adapter_parameters,
                batch_unavailable = unavailable,
                source = args.worker,
                revision = subprocess.check_output(
                    ["git", "-C", args.worker, "rev-parse", "HEAD"], text = True
                ).strip(),
                zoo_source = args.zoo_source,
                zoo_runtime_revision = subprocess.check_output(
                    ["git", "-C", args.zoo_source, "rev-parse", "HEAD"], text = True
                ).strip(),
                versions = {
                    name: importlib.metadata.version(name)
                    for name in ("mlx", "mlx-lm", "mlx-vlm", "unsloth_zoo")
                },
            ),
            indent = 2,
            default = str,
        ),
        encoding = "utf-8",
    )
