"""Latency of Studio's MLX memory estimate (POST /api/inference/estimate-memory) on Apple Silicon.

Answers "what does the first estimate after opening the Load Model panel cost on a cold process":
each model runs in a FRESH interpreter that imports the route (timed apart, it is server startup),
then calls the route function itself (auth dependency bypassed, as the backend tests do) in the
order the panel does: the unpinned first estimate, the same again, slider drags (pinned contexts),
and KV widths. Every phase of the MLX arm is timed by wrapping the module attribute it calls.

    python jobs/mlx_estimate_latency.py            # dense, hybrid, sliding-window checkpoints
    python jobs/mlx_estimate_latency.py --tiny     # SmolLM-135M only

Weights are downloaded before timing; nothing here loads a model. Exit 3 off Apple Silicon.
Every child gets a scratch UNSLOTH_STUDIO_HOME and writes only under the --out directory, so a run
never touches the operator's real Studio home or /tmp.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

DEFAULT_MODELS = [
    "mlx-community/Qwen3-0.6B-4bit",  # dense GQA
    "mlx-community/Qwen3.5-2B-4bit",  # hybrid linear attention, vision tower
    "unsloth/gemma-3-270m-it",  # sliding window, bf16 (upstream mlx-ci model)
]
TINY_MODEL = "mlx-community/SmolLM-135M-Instruct-4bit"

REQUESTS = [
    ("cold_unpinned", {}),
    ("warm_unpinned", {}),
    ("drag_8192", {"max_seq_length": 8192}),
    ("drag_16384", {"max_seq_length": 16384}),
    ("drag_32768", {"max_seq_length": 32768}),
    ("unpinned_kv8", {"mlx_kv_bits": 8}),
    ("unpinned_kv4", {"mlx_kv_bits": 4}),
    ("warm_unpinned_again", {}),
]

PHASES = [
    ("routes.inference", "_mlx_estimate_available", "stack_check"),
    ("routes.inference", "_cached_estimate_config", "config"),
    ("routes.inference", "_local_mlx_model_dir", "snapshot_dir"),
    ("core.inference.mlx_inference", "mlx_kv_quant_is_refused", "kv_refusal_probe"),
    ("routes.inference", "_mlx_estimate_fitted_context", "fit"),
    ("core.inference.mlx_memory", "mlx_memory_breakdown", "breakdown"),
]


def _backend_dir():
    root = Path.cwd()
    for cand in (root, *root.parents):
        if (cand / "studio" / "backend" / "routes" / "inference.py").is_file():
            return cand / "studio" / "backend"
    raise SystemExit(
        "run from an unsloth checkout (studio/backend/routes/inference.py not found)"
    )


def child(model, out_path, mode="fresh"):
    """One fresh interpreter: import, then the panel's request sequence. ``after_startup`` first runs
    Studio's startup warm thread (hardware detection imports the MLX stack) to completion, as a
    server does before anyone opens the panel."""
    import asyncio
    import importlib

    out_path = Path(out_path).resolve()
    if not os.environ.get("UNSLOTH_STUDIO_HOME"):
        # Studio's import creates its home; a bare child run otherwise wrote into ~/.unsloth.
        os.environ["UNSLOTH_STUDIO_HOME"] = tempfile.mkdtemp(prefix="studio_home_",
                                                             dir=out_path.parent)
    backend = _backend_dir()
    sys.path.insert(0, str(backend))
    os.chdir(backend)

    t = time.perf_counter()
    ri = importlib.import_module("routes.inference")
    from models.inference import EstimateMemoryRequest

    import_ms = (time.perf_counter() - t) * 1e3
    warm_ms = None
    if mode == "after_startup":
        from utils.torch_warmup import join_background_warm, start_background_warm

        t = time.perf_counter()
        start_background_warm()
        join_background_warm()
        warm_ms = (time.perf_counter() - t) * 1e3

    spent = {}

    def timed(fn, label):
        def wrapper(*a, **kw):
            t0 = time.perf_counter()
            try:
                return fn(*a, **kw)
            finally:
                spent[label] = spent.get(label, 0.0) + (time.perf_counter() - t0) * 1e3

        return wrapper

    for mod, attr, label in PHASES:
        m = importlib.import_module(mod)
        setattr(m, attr, timed(getattr(m, attr), label))

    rows = []
    for name, extra in REQUESTS:
        spent.clear()
        t0 = time.perf_counter()
        err = None
        try:
            resp = asyncio.run(
                ri.estimate_memory(
                    EstimateMemoryRequest(model_path=model, **extra),
                    fastapi_request=None,
                    current_subject="latency",
                )
            )
        except Exception as exc:  # recorded, never swallowed silently
            resp, err = None, f"{type(exc).__name__}: {exc}"
        wall = (time.perf_counter() - t0) * 1e3
        rows.append(
            {
                "request": name,
                "args": extra,
                "wall_ms": round(wall, 1),
                "phases_ms": {k: round(v, 1) for k, v in spent.items()},
                "available": getattr(resp, "available", None),
                "reason": getattr(resp, "reason", None),
                "total_gb": round((getattr(resp, "total_bytes", 0) or 0) / 1e9, 3),
                "n_ctx": getattr(resp, "n_ctx", None),
                "context_fitted": getattr(resp, "context_fitted", None),
                "error": err,
            }
        )
        print(json.dumps({"model": model, "mode": mode, **rows[-1]}), flush=True)
    out_path.write_text(
        json.dumps(
            {
                "model": model,
                "mode": mode,
                "import_ms": round(import_ms, 1),
                "startup_warm_ms": None if warm_ms is None else round(warm_ms, 1),
                "rows": rows,
            }
        )
    )


def main():
    if "--child" in sys.argv:
        i = sys.argv.index("--child")
        child(
            sys.argv[i + 1],
            sys.argv[i + 2],
            sys.argv[i + 3] if len(sys.argv) > i + 3 else "fresh",
        )
        return
    p = C.base_parser(
        "mlx_estimate_latency", DEFAULT_MODELS[0], TINY_MODEL, default_steps=0
    )
    p.add_argument(
        "--models",
        default=None,
        help="comma-separated; default: dense, hybrid, windowed",
    )
    a = C.resolve_args(p)
    models = (
        [TINY_MODEL]
        if a.tiny
        else (a.models.split(",") if a.models else DEFAULT_MODELS)
    )
    with C.JobRecorder(a, backend_hint="mlx") as rec:
        if not (sys.platform == "darwin" and platform.machine() == "arm64"):
            rec.skip("needs Apple Silicon")
        from huggingface_hub import snapshot_download

        for m in models:
            t = time.perf_counter()
            snapshot_download(m)
            print(
                f"downloaded {m} in {time.perf_counter() - t:.1f}s (untimed)",
                flush=True,
            )
        scratch = Path(a.out).resolve().parent / "mlx_estimate_latency_scratch"
        shutil.rmtree(scratch, ignore_errors=True)
        scratch.mkdir(parents=True)
        env = dict(os.environ)
        env["UNSLOTH_STUDIO_HOME"] = str(scratch / "studio_home")
        results = []
        for mode in ("fresh", "after_startup"):
            for i, m in enumerate(models):
                out = scratch / f"{mode}_{i}.json"
                try:
                    proc = subprocess.run(
                        [sys.executable, os.path.abspath(__file__), "--child", m, str(out), mode],
                        env=env,
                        timeout=900,
                    )
                    rc = proc.returncode
                except subprocess.TimeoutExpired:
                    rc = "timeout"
                if rc != 0 or not out.is_file() or not out.stat().st_size:
                    rec.check(f"child_ok[{mode}:{m}]", False, f"exit {rc}")
                    continue
                res = json.loads(out.read_text())
                results.append(res)
                for row in res["rows"]:
                    rec.step(model=m, mode=mode, **row)
                cold = res["rows"][0]
                rec.check(
                    f"priced[{mode}:{m}]",
                    all(r["available"] for r in res["rows"]),
                    [
                        (r["request"], r["reason"], r["error"])
                        for r in res["rows"]
                        if not r["available"]
                    ],
                )
                key = f"{mode}:{m}"
                rec.summary(
                    **{
                        f"{key}.import_ms": res["import_ms"],
                        f"{key}.startup_warm_ms": res["startup_warm_ms"],
                        f"{key}.cold_first_ms": cold["wall_ms"],
                        f"{key}.cold_phases_ms": cold["phases_ms"],
                        f"{key}.warm_ms": res["rows"][1]["wall_ms"],
                        f"{key}.drag_ms": [r["wall_ms"] for r in res["rows"][2:5]],
                    }
                )
        print(
            "\n| mode | model | import ms | startup warm ms | cold first ms | warm ms | drags ms | kv8 ms | kv4 ms | cold phases ms |"
        )
        print("|---|---|---|---|---|---|---|---|---|---|")
        for res in results:
            r = {x["request"]: x for x in res["rows"]}
            print(
                f"| {res['mode']} | {res['model']} | {res['import_ms']} | {res['startup_warm_ms']} | "
                f"{r['cold_unpinned']['wall_ms']} | "
                f"{r['warm_unpinned']['wall_ms']} | "
                f"{[r[k]['wall_ms'] for k in ('drag_8192', 'drag_16384', 'drag_32768')]} | "
                f"{r['unpinned_kv8']['wall_ms']} | {r['unpinned_kv4']['wall_ms']} | "
                f"{r['cold_unpinned']['phases_ms']} |",
                flush=True,
            )
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
