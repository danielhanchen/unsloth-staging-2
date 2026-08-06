#!/usr/bin/env python3
"""D: three-arm Apple Silicon measurement for unslothai/unsloth#7917.

Arms, all driven through Studio's real _run_mlx_training:

  default   max_grad_norm unset  -> zoo mode leaf_norm (UNCHANGED default), norm reported
  global    max_grad_norm = 1.0  -> zoo mode global_norm, norm reported

Both must populate the chart, which is the fix. `default` must keep the trainer's
own clip mode, proving the chart no longer costs a change in training behavior;
`global` must actually switch mode, proving an explicitly requested threshold now
reaches the trainer instead of being discarded as it was before.

Corrects a flaw in my first Mac run: that one timed the two arms back to back with a
cold HuggingFace cache on the first, so its wall-clock numbers measured the model
download, not training. Here the cache is warmed first, every arm runs in a fresh
process, and repeats alternate order.

Usage:
  python sim_d_mac_arms.py warm  --model M
  python sim_d_mac_arms.py run   --model M --arm {before,pr,report} --rep N --out F.json
  python sim_d_mac_arms.py report RESULT_DIR
"""

import argparse
import json
import os
import platform
import queue
import statistics
import sys
import time
from pathlib import Path

REPO = Path(os.environ.get("UNSLOTH_REPO", Path.cwd()))
sys.path.insert(0, str(REPO / "studio" / "backend"))

STEPS = int(os.environ.get("SIM_D_STEPS", "8"))


def _worker():
    import core.training.worker as w
    return w


def dataset_file(workdir):
    p = Path(workdir) / "train.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    row = {"text": "<<HELLO!!>> My name is Unsloth and I like to run fast."}
    p.write_text("".join(json.dumps(row) + "\n" for _ in range(128)), encoding="utf-8")
    return str(p)


def build_config(arm, model, workdir):
    cfg = {
        "model_name": model,
        "training_type": "LoRA/QLoRA",
        "format_type": "raw",
        "local_datasets": [dataset_file(workdir)],
        "output_dir": str(Path(workdir) / f"out_{arm}"),
        "allow_external_output_dir": True,
        "max_steps": STEPS,
        "num_epochs": 1,
        "batch_size": 1,
        # >1 on purpose: the zoo's VLM compile gate only fires when grad accum > 1,
        # and 4 is Studio's own default.
        "gradient_accumulation_steps": 4,
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
    if arm == "global":
        cfg["max_grad_norm"] = 1.0
    elif arm != "default":
        raise SystemExit(f"unknown arm {arm!r}")
    return cfg


def cmd_warm(model):
    """Populate the HF cache so no arm pays the download."""
    from huggingface_hub import snapshot_download
    t = time.time()
    snapshot_download(model)
    print(f"warmed {model} in {time.time() - t:.1f}s", flush=True)


def cmd_run(arm, model, rep, out, workdir):
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise SystemExit(f"FATAL: needs Darwin/arm64, got {platform.system()}/{platform.machine()}")

    import mlx.core as mx
    import unsloth_zoo.mlx.trainer as zoo

    w = _worker()

    # Capture the trainer instance so its compile shape-guard report can be read.
    captured = {}
    real_trainer_cls = zoo.MLXTrainer

    class Capturing(real_trainer_cls):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            captured["trainer"] = self
            captured["resolved_clip"] = zoo._resolve_mlx_grad_clipping(self.args)

    zoo.MLXTrainer = Capturing
    try:
        cfg = build_config(arm, model, workdir)
        events, stop_q = queue.Queue(), queue.Queue()
        reset_peak = getattr(mx, "reset_peak_memory", None) or getattr(mx.metal, "reset_peak_memory", None)
        get_peak = getattr(mx, "get_peak_memory", None) or getattr(mx.metal, "get_peak_memory", None)
        if reset_peak:
            reset_peak()
        t0 = time.time()
        try:
            w._run_mlx_training(events, stop_q, cfg)
        finally:
            stop_q.put({"type": w._MLX_WORKER_COMPLETE})
        wall = time.time() - t0
    finally:
        zoo.MLXTrainer = real_trainer_cls

    progress, tokens = [], 0
    while True:
        try:
            e = events.get_nowait()
        except queue.Empty:
            break
        if e.get("type") == "progress" and e.get("step") is not None:
            progress.append({"step": e["step"], "loss": e.get("loss"),
                             "grad_norm": e.get("grad_norm")})
            tokens = e.get("num_tokens") or tokens

    trainer = captured.get("trainer")
    guard = getattr(trainer, "_compile_shape_guard_report", None)
    result = {
        "arm": arm, "model": model, "rep": rep,
        "wall_s": round(wall, 2),
        "peak_bytes": int(get_peak()) if get_peak else 0,
        "num_tokens": tokens,
        "tokens_per_s": round(tokens / wall, 1) if wall and tokens else None,
        "resolved_clip_mode": (captured.get("resolved_clip") or [None])[-1],
        "compile_action": getattr(guard, "action", None),
        "compile_reason": getattr(guard, "reason", None),
        "grad_norms": [p["grad_norm"] for p in progress],
        "losses": [p["loss"] for p in progress],
    }
    result["grad_norm_reported"] = any(g is not None for g in result["grad_norms"])
    Path(out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


def cmd_report(result_dir):
    rows = [json.loads(p.read_text()) for p in sorted(Path(result_dir).glob("*.json"))]
    if not rows:
        raise SystemExit("no results")
    arms = {}
    for r in rows:
        arms.setdefault((r["model"], r["arm"]), []).append(r)
    expected_mode = {"default": "leaf_norm", "global": "global_norm"}

    print(f"\n{'model':<34} {'arm':<8} {'clip mode':<12} {'compile':<28} "
          f"{'chart':<6} {'peak MiB':<10} {'wall s':<8}")
    print("-" * 116)
    for (model, arm), rs in sorted(arms.items()):
        peak = statistics.median(r["peak_bytes"] for r in rs) / 1024 ** 2
        wall = statistics.median(r["wall_s"] for r in rs)
        r0 = rs[0]
        compile_str = f"{r0['compile_action']}/{r0['compile_reason']}"
        print(f"{model:<34} {arm:<8} {str(r0['resolved_clip_mode']):<12} {compile_str:<28} "
              f"{str(r0['grad_norm_reported']):<6} {peak:<10.1f} {wall:<8.1f}")

    print("\nVerdicts:")
    bad = []
    for model in sorted({m for m, _ in arms}):
        default = arms.get((model, "default"), [])
        glob = arms.get((model, "global"), [])
        for label, rs in (("default", default), ("global", glob)):
            if not rs:
                continue
            if not rs[0]["grad_norm_reported"]:
                bad.append(f"{model}: arm '{label}' did NOT populate the chart")
            want = expected_mode[label]
            got = rs[0]["resolved_clip_mode"]
            if got != want:
                bad.append(f"{model}: arm '{label}' clip mode {got!r}, expected {want!r}")
        if default and glob:
            d = (statistics.median(r["peak_bytes"] for r in glob)
                 / statistics.median(r["peak_bytes"] for r in default) - 1) * 100
            print(f"  {model}: explicit-global vs default peak memory {d:+.1f}%")
            same = default[0]["losses"] == glob[0]["losses"]
            print(f"  {model}: loss trajectories identical across clip modes: {same} "
                  f"(expected False -- the modes really do clip differently)")
    for b in bad:
        print("  FAIL:", b)
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("warm"); w.add_argument("--model", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--arm", required=True, choices=["default", "global"])
    r.add_argument("--rep", type=int, default=1)
    r.add_argument("--out", required=True)
    r.add_argument("--workdir", default="mlx_d_workdir")
    p = sub.add_parser("report"); p.add_argument("result_dir")
    a = ap.parse_args()
    if a.cmd == "warm":
        cmd_warm(a.model)
    elif a.cmd == "run":
        cmd_run(a.arm, a.model, a.rep, a.out, a.workdir)
    else:
        sys.exit(cmd_report(a.result_dir))


if __name__ == "__main__":
    main()
