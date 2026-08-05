# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Apple Silicon end-to-end gate for unslothai/unsloth#7917.

Staging-only harness. The PR makes Studio's MLX worker resolve an unset
`max_grad_norm` to 1.0 instead of hardcoding 0.0, which is what makes
unsloth_zoo's MLX trainer resolve `global_norm` clipping and therefore emit the
pre-clip gradient norm that Studio's Gradient Norm chart plots.

Nothing in the repository's own CI covers that: the backend tests are source
text plus schema assertions that pass identically on Linux, and mlx-ci.yml's
real smoke (`tests/studio/run_real_mlx_smoke.py`) pins
`max_grad_norm=0.0, max_grad_value=1.0` and goes through the public API, never
`_run_mlx_training`.

So this drives the real worker on real Metal, twice, and compares:

  train --mode default   max_grad_norm absent  -> expect grad_norm on the events
  train --mode zero      max_grad_norm = 0.0   -> expect no grad_norm at all

Subcommands (each meant to be its own CI step, so the two training runs get
fresh processes and therefore honest peak-memory numbers):

  checks                          platform + resolver + clip-precedence
  train --mode {default,zero}     one real LoRA run, results to --out
  compare <default.json> <zero.json>
"""

import argparse
import json
import math
import os
import platform
import queue
import sys
import time
from pathlib import Path

MODEL = os.environ.get("MLX_E2E_MODEL", "unsloth/gemma-3-270m-it")
STEPS = int(os.environ.get("MLX_E2E_STEPS", "5"))


def _fail(msg):
    print(f"FAIL: {msg}", flush = True)
    raise SystemExit(1)


def _ok(msg):
    print(f"ok: {msg}", flush = True)


def _import_worker():
    """Import the Studio worker the same way the backend does."""
    backend = Path(__file__).resolve().parents[2] / "studio" / "backend"
    if str(backend) not in sys.path:
        sys.path.insert(0, str(backend))
    import core.training.worker as worker
    return worker


# ── checks ───────────────────────────────────────────────────────────────────

def cmd_checks():
    if platform.system() != "Darwin":
        _fail(f"expected Darwin, got {platform.system()} -- this gate must not be skipped")
    if platform.machine() != "arm64":
        _fail(f"expected arm64, got {platform.machine()}")
    _ok(f"host is {platform.system()}/{platform.machine()}")

    import mlx.core as mx
    if not mx.metal.is_available():
        _fail("mx.metal.is_available() is False; this is not a real Metal runner")
    _ok("Metal is available")

    worker = _import_worker()
    resolve = worker._resolve_mlx_max_grad_norm

    if resolve(None) != 1.0:
        _fail(f"_resolve_mlx_max_grad_norm(None) = {resolve(None)!r}, expected 1.0")
    _ok("MLX_GRADNORM_GATE _resolve_mlx_max_grad_norm(None) == 1.0")

    for value, expected in ((0, 0.0), (0.3, 0.3), (2, 2.0)):
        got = resolve(value)
        if got != expected:
            _fail(f"_resolve_mlx_max_grad_norm({value!r}) = {got!r}, expected {expected!r}")
    _ok("explicit values pass through unchanged (0, 0.3, 2)")

    for bad in (-1, "nope", float("inf"), float("-inf"), float("nan")):
        try:
            got = resolve(bad)
        except ValueError:
            continue
        _fail(f"_resolve_mlx_max_grad_norm({bad!r}) returned {got!r}; expected ValueError")
    _ok("negative / non-numeric / non-finite are rejected")

    # Precedence, against the installed trainer rather than a restatement of it.
    # This is what decides whether the author was right to reject the Codex
    # "avoid enabling global clipping when other clip knobs are set" item.
    from unsloth_zoo.mlx.trainer import _resolve_mlx_grad_clipping

    class _Args:
        def __init__(self, **kw):
            self.max_grad_norm = kw.get("norm", 0.0)
            self.max_grad_value = kw.get("value")
            self.max_grad_leaf_norm = kw.get("leaf")

    cases = [
        # (kwargs, expected mode, why it matters)
        (dict(norm = 1.0), "global_norm", "the PR's new default resolves to global_norm"),
        (dict(norm = 0.0), "leaf_norm", "today's behavior: implicit per-leaf default"),
        (dict(norm = 1.0, leaf = 1.3), "leaf_norm", "positive leaf outranks global"),
        (dict(norm = 1.0, value = 3.0), "value", "positive value outranks global"),
        (dict(norm = 1.0, value = 3.0, leaf = 1.3), "value", "value outranks both"),
        (dict(norm = 1.0, leaf = 0.0), "global_norm", "leaf=0 does not suppress global"),
        (dict(norm = 0.0, leaf = 0.0), "none", "explicit zeros disable clipping entirely"),
    ]
    for kwargs, expected_mode, why in cases:
        resolved = _resolve_mlx_grad_clipping(_Args(**kwargs))
        mode = resolved[-1]
        if mode != expected_mode:
            _fail(f"_resolve_mlx_grad_clipping({kwargs}) -> {resolved!r}; expected mode {expected_mode!r} ({why})")
        print(f"  {kwargs} -> {resolved} :: {why}", flush = True)
    _ok("MLX_GRADNORM_GATE clip precedence matches the PR's stated semantics")


# ── train ────────────────────────────────────────────────────────────────────

def _dataset(workdir):
    """One repeated short row, mirroring tests/studio/run_real_mlx_smoke.py."""
    path = Path(workdir) / "train.jsonl"
    row = {"text": "<<HELLO!!>> My name is Unsloth!"}
    path.write_text("".join(json.dumps(row) + "\n" for _ in range(64)), encoding = "utf-8")
    return str(path)


def _config(mode, workdir, data_file):
    """Shaped like _build_training_worker_config's output.

    `default` omits max_grad_norm entirely -- exactly what the PR makes the
    frontend and _build_training_worker_config send once the field is
    Optional[float] = None. `zero` sends the literal 0.0 the UI used to send.
    """
    config = {
        "model_name": MODEL,
        "training_type": "LoRA/QLoRA",
        "format_type": "raw",
        "local_datasets": [data_file],
        "output_dir": str(Path(workdir) / f"out_{mode}"),
        "allow_external_output_dir": True,
        "max_steps": STEPS,
        "num_epochs": 1,
        "batch_size": 1,
        "gradient_accumulation_steps": 2,
        "max_seq_length": 256,
        "learning_rate": 2e-4,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "warmup_steps": 1,
        "optim": "adamw_8bit",
        "lr_scheduler_type": "linear",
        "weight_decay": 0.001,
        "save_steps": 0,
        "eval_steps": 0,
        "random_seed": 3407,
        "max_grad_value": None,
        "max_grad_leaf_norm": None,
    }
    if mode == "zero":
        config["max_grad_norm"] = 0.0
    elif mode != "default":
        raise SystemExit(f"unknown mode {mode!r}")
    return config


def cmd_train(mode, out, workdir):
    Path(workdir).mkdir(parents = True, exist_ok = True)
    worker = _import_worker()
    import mlx.core as mx

    config = _config(mode, workdir, _dataset(workdir))
    print(f"mode={mode} max_grad_norm={'<absent>' if 'max_grad_norm' not in config else config['max_grad_norm']!r}", flush = True)

    events = queue.Queue()
    stop_queue = queue.Queue()

    # Newer MLX moved these to top level; fall back to mx.metal like
    # tests/studio/run_real_mlx_smoke.py does.
    reset_peak = getattr(mx, "reset_peak_memory", None) or getattr(mx.metal, "reset_peak_memory", None)
    get_peak = getattr(mx, "get_peak_memory", None) or getattr(mx.metal, "get_peak_memory", None)

    if reset_peak is not None:
        reset_peak()
    started = time.time()
    try:
        worker._run_mlx_training(events, stop_queue, config)
    finally:
        stop_queue.put({"type": worker._MLX_WORKER_COMPLETE})
    peak_bytes = int(get_peak()) if get_peak is not None else 0

    progress = []
    while True:
        try:
            event = events.get_nowait()
        except queue.Empty:
            break
        if event.get("type") == "progress" and event.get("step") is not None:
            progress.append({
                "step": event.get("step"),
                "loss": event.get("loss"),
                "grad_norm": event.get("grad_norm"),
            })
        elif event.get("type") == "error":
            _fail(f"worker emitted an error event: {event}")

    result = {
        "mode": mode,
        "model": MODEL,
        "peak_bytes": peak_bytes,
        "peak_mib": round(peak_bytes / (1024 ** 2), 1),
        "wall_seconds": round(time.time() - started, 1),
        "progress": progress,
    }
    Path(out).write_text(json.dumps(result, indent = 2), encoding = "utf-8")
    print(json.dumps(result, indent = 2), flush = True)

    if not progress:
        _fail("the worker emitted no progress events; training did not run")

    norms = [p["grad_norm"] for p in progress]
    reported = [n for n in norms if n is not None]

    if mode == "default":
        if not reported:
            _fail("no grad_norm on any progress event -- the chart would still be empty")
        bad = [n for n in reported if not (math.isfinite(n) and n > 0)]
        if bad:
            _fail(f"grad_norm values are not finite and positive: {bad}")
        _ok(f"MLX_GRADNORM_GATE {len(reported)}/{len(norms)} steps reported a finite positive grad_norm: {reported}")
    else:
        if reported:
            _fail(f"expected no grad_norm with max_grad_norm=0.0, got {reported}")
        _ok(f"MLX_GRADNORM_GATE max_grad_norm=0.0 reported no grad_norm on any of {len(norms)} steps (the bug being fixed)")


# ── compare ──────────────────────────────────────────────────────────────────

def cmd_compare(default_path, zero_path):
    default = json.loads(Path(default_path).read_text(encoding = "utf-8"))
    zero = json.loads(Path(zero_path).read_text(encoding = "utf-8"))

    reported = [p["grad_norm"] for p in default["progress"] if p["grad_norm"] is not None]
    print(f"default (max_grad_norm unset -> 1.0): {len(reported)} grad_norm samples {reported}", flush = True)
    print(f"zero    (max_grad_norm = 0.0):        "
          f"{sum(1 for p in zero['progress'] if p['grad_norm'] is not None)} grad_norm samples", flush = True)

    if not reported:
        _fail("default run reported no gradient norms")
    if any(p["grad_norm"] is not None for p in zero["progress"]):
        _fail("zero run reported gradient norms; the before/after contrast does not hold")

    d_peak, z_peak = default["peak_bytes"], zero["peak_bytes"]
    delta = (d_peak - z_peak) / z_peak * 100 if z_peak else float("nan")
    print(f"peak memory: global-norm {default['peak_mib']} MiB vs per-leaf {zero['peak_mib']} MiB "
          f"({delta:+.1f}%)", flush = True)
    print(f"wall: global-norm {default['wall_seconds']}s vs per-leaf {zero['wall_seconds']}s", flush = True)
    print("NOTE: a 270m model on a 7 GB runner cannot confirm the PR's 3B +21.7% figure; "
          "this is a direction-and-magnitude check only.", flush = True)

    _ok("MLX_GRADNORM_GATE before/after contrast holds on real Apple Silicon")


def main():
    parser = argparse.ArgumentParser(description = __doc__)
    sub = parser.add_subparsers(dest = "cmd", required = True)
    sub.add_parser("checks")
    train = sub.add_parser("train")
    train.add_argument("--mode", choices = ["default", "zero"], required = True)
    train.add_argument("--out", required = True)
    train.add_argument("--workdir", default = "mlx_e2e_workdir")
    compare = sub.add_parser("compare")
    compare.add_argument("default_json")
    compare.add_argument("zero_json")

    args = parser.parse_args()
    if args.cmd == "checks":
        cmd_checks()
    elif args.cmd == "train":
        cmd_train(args.mode, args.out, args.workdir)
    else:
        cmd_compare(args.default_json, args.zero_json)


if __name__ == "__main__":
    main()
