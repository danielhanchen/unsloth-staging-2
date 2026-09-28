"""Studio MLX speculative decoding on real Apple Silicon: load a target through Studio's
MLXInferenceBackend with speculation off, with an explicit drafter, and on Auto; greedy-generate
the same prompts on each arm.

Checks: every arm generates, the explicit drafter attaches, Auto pins the cached drafter.
Recorded, not gated: decode tok/s per arm and the greedy prefix shared with the `off` arm
(speculation is not output-exact, and hosted-runner throughput is noisy).

    python jobs/mlx_spec_inference.py                       # Qwen3.5-4B-4bit + MTP drafter
    python jobs/mlx_spec_inference.py --studio-backend PATH # default: ./studio/backend
"""

import os
import platform
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as c  # noqa: E402

p = c.base_parser("mlx_spec_inference", "mlx-community/Qwen3.5-4B-4bit",
                  "mlx-community/Qwen3.5-4B-4bit", default_steps=0)
p.add_argument("--drafter", default="mlx-community/Qwen3.5-4B-MTP-bf16")
p.add_argument("--method", default="mtp")
p.add_argument("--max-new-tokens", type=int, default=96)
p.add_argument("--studio-backend", default=os.path.join(os.getcwd(), "studio", "backend"))
a = c.resolve_args(p)

PROMPTS = ["Explain in two sentences why the sky is blue.",
           "Write a Python function that returns the n-th Fibonacci number.",
           "List five countries in South America."]


def _text(chunks):
    # Some paths stream cumulative snapshots, others deltas.
    if chunks and all(chunks[i + 1].startswith(chunks[i]) for i in range(len(chunks) - 1)):
        return chunks[-1]
    return "".join(chunks)


def _shared_prefix(x, y):
    n = 0
    for u, v in zip(x, y):
        if u != v:
            break
        n += 1
    return n


with c.JobRecorder(a) as rec:
    if not (platform.system() == "Darwin" and platform.machine() == "arm64"):
        rec.skip(f"MLX needs Apple Silicon, this is {platform.system()}-{platform.machine()}")
    if not os.path.isdir(a.studio_backend):
        rec.skip(f"no Studio backend at {a.studio_backend}")
    sys.path.insert(0, a.studio_backend)
    rec.set_backend("studio-mlx")

    from huggingface_hub import snapshot_download
    t0 = time.perf_counter()
    snapshot_download(a.model)
    snapshot_download(a.drafter)
    rec.summary(download_s=round(time.perf_counter() - t0, 1))

    from core.inference.mlx_inference import MLXInferenceBackend
    from core.inference.mlx_speculative import resolve_mlx_speculative_request
    from utils.models import ModelConfig

    config = ModelConfig.from_identifier(model_id=a.model)
    rec.check("target_config", config is not None, a.model)
    is_vision = bool(getattr(config, "is_vision", False))
    rec.summary(target_is_vision=is_vision)

    arms = {}
    for arm, mode, drafter in (("off", "off", None), ("explicit", a.method, a.drafter), ("auto", "auto", None)):
        res = resolve_mlx_speculative_request(a.model, mode, drafter, is_vision=is_vision)
        backend = MLXInferenceBackend()
        t1 = time.perf_counter()
        ok = backend.load_model(
            config=config, max_seq_length=a.max_seq_length, load_in_4bit=True,
            mlx_speculative_mode=mode, mlx_draft_model=drafter,
            mlx_speculative_resolved_mode=res.method,
            mlx_speculative_resolved_draft_model=res.draft_model,
            mlx_speculative_resolution_reason=res.reason,
        )
        load_s = round(time.perf_counter() - t1, 2)
        rec.check(f"{arm}_loaded", bool(ok), f"load_model -> {ok}")
        record = backend.models.get(backend.active_model_name) or {}
        effective = record.get("mlx_speculative_effective_mode")
        def _gen(prompt, backend=backend):
            t = time.perf_counter()
            chunks = list(backend.generate_chat_response(
                [{"role": "user", "content": prompt}], temperature=0.0, top_p=1.0, top_k=0,
                max_new_tokens=a.max_new_tokens, enable_thinking=False))
            return _text(chunks), time.perf_counter() - t
        _gen(PROMPTS[0])  # first generation after a load pages weights in; not timed
        tok = getattr(backend._processor, "tokenizer", backend._processor) or backend._tokenizer
        outs, tps, wall = [], [], []
        for prompt in PROMPTS:
            text, secs = _gen(prompt)
            outs.append(text)
            stats = backend.last_generation_stats or {}
            tps.append((stats.get("timings") or {}).get("predicted_per_second"))
            n = (stats.get("timings") or {}).get("predicted_n") or len(tok.encode(text))
            wall.append(round(n / secs, 3) if secs > 0 else None)
        arms[arm] = dict(resolved=res.method, resolved_drafter=res.draft_model, reason=res.reason,
                         effective=effective, draft=record.get("mlx_speculative_effective_draft_model"),
                         block=record.get("mlx_speculative_effective_block_size"),
                         runtime_reason=record.get("mlx_speculative_reason"),
                         load_s=load_s, tok_s=tps, wall_tok_s=wall, outputs=[o[:400] for o in outs])
        rec.check(f"{arm}_generates", all(o.strip() for o in outs), [o[:60] for o in outs])
        backend.unload_model(backend.active_model_name)
        del backend

    try:
        import subprocess
        vmm = subprocess.run(["sysctl", "-n", "kern.hv_vmm_present"], capture_output=True, text=True).stdout.strip()
    except Exception:
        vmm = None
    rec.summary(arms=arms, hv_vmm_present=vmm)
    ex, au = arms["explicit"], arms["auto"]
    rec.check("explicit_drafter_attached", ex["effective"] == a.method and ex["draft"] == a.drafter,
              f"effective={ex['effective']} draft={ex['draft']} reason={ex['runtime_reason']}")
    rec.check("auto_pins_cached_drafter", au["resolved"] != "off" and au["effective"] == au["resolved"],
              f"resolved={au['resolved']}/{au['resolved_drafter']} effective={au['effective']} "
              f"reason={au['reason']}/{au['runtime_reason']}")
    rec.check("off_attaches_nothing", arms["off"]["effective"] in (None, "off"), arms["off"]["effective"])
    base = arms["off"]["outputs"]
    rec.summary(greedy_prefix_vs_off={k: [_shared_prefix(x, y) for x, y in zip(base, v["outputs"])]
                                      for k, v in arms.items() if k != "off"},
                median_tok_s={k: sorted(t for t in v["tok_s"] if t)[len(v["tok_s"]) // 2]
                               if any(v["tok_s"]) else None for k, v in arms.items()},
                median_wall_tok_s={k: sorted(t for t in v["wall_tok_s"] if t)[len(v["wall_tok_s"]) // 2]
                                   if any(v["wall_tok_s"]) else None for k, v in arms.items()})
