"""A/B of Studio's MLX idle keep-warm tick (unsloth#12721) on Apple Silicon.

After each request the worker idles for --pause seconds, either doing nothing (off) or running the
worker's exact keep-warm op every 0.5 s (on), then serves a --prompt-tokens prompt and decodes
--gen-tokens. Arms interleave ABBA in one process (the effect is in-process idle, so a fresh process
per measurement would measure load, not idle). Reports TTFT / decode tok/s medians + spread per arm,
a paired bootstrap 95% CI on the TTFT delta, and the per-tick cost.

    python jobs/mlx_keepwarm_bench.py                 # Qwen3.5-2B 4-bit, as the PR measured
    python jobs/mlx_keepwarm_bench.py --tiny          # SmolLM-135M 4-bit

Exit 3 off Apple Silicon. Measurement only: checks are "it ran", never a speed threshold.
"""

from __future__ import annotations

import platform
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

DEFAULT_MODEL = "mlx-community/Qwen3.5-2B-4bit"
FALLBACK_MODEL = "mlx-community/Qwen3-0.6B-4bit"
TINY_MODEL = "mlx-community/SmolLM-135M-Instruct-4bit"
# The op the PR's _MLXIdleWarmth.tick evaluates; drift-guarded against the checkout below.
TICK_SRC = "mx.eval(mx.zeros((1,), dtype = mx.float32) + 1)"


def _tick(mx):
    mx.eval(mx.zeros((1,), dtype = mx.float32) + 1)


def _idle(mx, arm, pause):
    if arm == "off":
        time.sleep(pause)
        return 0
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 + 0.5 <= pause:
        time.sleep(0.5)
        _tick(mx)
        n += 1
    time.sleep(max(0.0, pause - (time.perf_counter() - t0)))
    return n


def _request(mx, stream_generate, model, tok, prompt, gen_tokens):
    t0 = time.perf_counter()
    ttft, last, n = None, None, 0
    for r in stream_generate(model, tok, prompt = prompt, max_tokens = gen_tokens):
        n += 1
        now = time.perf_counter()
        if ttft is None:
            ttft = now - t0
            t_first = now
        last = r
    decode = (n - 1) / (now - t_first) if n > 1 and now > t_first else None
    return ttft * 1000, decode, getattr(last, "generation_tps", None)


def _boot_ci(
    deltas,
    n = 5000,
    seed = 0,
):
    rng = random.Random(seed)
    meds = sorted(statistics.median(rng.choices(deltas, k = len(deltas))) for _ in range(n))
    return meds[int(0.025 * n)], meds[int(0.975 * n)]


def main():
    p = C.base_parser("mlx_keepwarm_bench", DEFAULT_MODEL, TINY_MODEL, default_steps = 0)
    p.add_argument("--rounds", type = int, default = 8, help = "ABBA blocks (2 requests per arm each)")
    p.add_argument("--pause", type = float, default = 3.0)
    p.add_argument("--prompt-tokens", type = int, default = 1024)
    p.add_argument("--gen-tokens", type = int, default = 128)
    a = C.resolve_args(p)
    C.reject_cfg(a)
    with C.JobRecorder(a, backend_hint = "mlx") as rec:
        if not (sys.platform == "darwin" and platform.machine() == "arm64"):
            rec.skip("needs Apple Silicon")
        worker = Path("studio/backend/core/inference/worker.py")
        if worker.is_file():
            rec.check(
                "tick_matches_worker",
                TICK_SRC in worker.read_text(encoding = "utf-8"),
                f"{TICK_SRC!r} in {worker}",
            )
        import mlx.core as mx
        from mlx_lm import load, stream_generate

        model_id = a.model
        try:
            model, tok = load(model_id)
        except Exception as exc:
            if a.tiny or model_id != DEFAULT_MODEL:
                raise
            print(f"load {model_id} failed ({exc!r}); falling back to {FALLBACK_MODEL}", flush = True)
            model_id = FALLBACK_MODEL
            model, tok = load(model_id)
        rec.summary(model_used = model_id, device = str(mx.default_device()))
        rng = random.Random(a.seed)
        vocab = min(getattr(tok, "vocab_size", 30000) or 30000, 30000)

        def prompt():
            return [rng.randrange(100, vocab) for _ in range(a.prompt_tokens)]

        for _ in range(2):
            _request(mx, stream_generate, model, tok, prompt(), a.gen_tokens)

        costs = []
        for _ in range(200):
            t = time.perf_counter()
            _tick(mx)
            costs.append((time.perf_counter() - t) * 1e6)
        rows, ticks = {"off": [], "on": []}, 0
        for b in range(a.rounds):
            for arm in ("off", "on", "on", "off") if b % 2 == 0 else ("on", "off", "off", "on"):
                ticks += _idle(mx, arm, a.pause)
                ttft, dec, gtps = _request(mx, stream_generate, model, tok, prompt(), a.gen_tokens)
                rows[arm].append((ttft, dec))
                rec.step(
                    block = b,
                    arm = arm,
                    ttft_ms = round(ttft, 2),
                    decode_tok_s = round(dec, 2) if dec else None,
                    generation_tps = round(gtps, 2) if gtps else None,
                )
                print(
                    f"block {b} {arm:3s} ttft {ttft:8.2f} ms decode {dec or 0:8.2f} tok/s",
                    flush = True,
                )
        rec.check("ticks_ran", ticks > 0, f"{ticks} ticks")
        rec.check("both_arms_measured", all(len(v) == 2 * a.rounds for v in rows.values()), "")

        def stats(vals):
            vals = [v for v in vals if v is not None]
            q = statistics.quantiles(vals, n = 4) if len(vals) > 1 else [vals[0]] * 3
            return round(statistics.median(vals), 2), round(q[0], 2), round(q[2], 2)

        # One delta per ABBA block: mean(on) - mean(off), so each pair shares drift.
        deltas = []
        for b in range(a.rounds):
            off = statistics.mean(r[0] for r in rows["off"][2 * b : 2 * b + 2])
            on = statistics.mean(r[0] for r in rows["on"][2 * b : 2 * b + 2])
            deltas.append(on - off)
        lo, hi = _boot_ci(deltas)
        summ = {
            arm: {
                "ttft_ms_median_q1_q3": stats([r[0] for r in v]),
                "decode_tok_s_median_q1_q3": stats([r[1] for r in v]),
            }
            for arm, v in rows.items()
        }
        rec.summary(
            arms = summ,
            ttft_delta_on_minus_off_ms_median = round(statistics.median(deltas), 2),
            ttft_delta_ci95_ms = [round(lo, 2), round(hi, 2)],
            tick_us_median = round(statistics.median(costs), 1),
            tick_us_p95 = round(sorted(costs)[int(0.95 * len(costs))], 1),
            pause_s = a.pause,
            prompt_tokens = a.prompt_tokens,
            gen_tokens = a.gen_tokens,
        )
        print("\n| arm | TTFT ms median [q1, q3] | decode tok/s median [q1, q3] |\n|---|---|---|")
        for arm, s in summ.items():
            print(f"| {arm} | {s['ttft_ms_median_q1_q3']} | {s['decode_tok_s_median_q1_q3']} |")
        print(
            f"TTFT on-off median {statistics.median(deltas):.2f} ms, 95% CI [{lo:.2f}, {hi:.2f}]; "
            f"tick {statistics.median(costs):.1f} us median",
            flush = True,
        )


if __name__ == "__main__":
    main()
