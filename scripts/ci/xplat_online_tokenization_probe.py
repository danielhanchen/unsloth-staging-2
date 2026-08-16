# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cross-platform CPU proxy for PR #8960's online tokenization path.

Staging-CI only. Prints, on whatever OS runs it:
  1. the multiprocessing facts the worker gate reads,
  2. the dataloader_num_workers the trainer is actually handed,
  3. a real end-to-end CPU SFT run of a few steps (online path if the gate
     allows it, eager path if it declines), with per-step losses,
  4. proof the process left no live children behind.

Exit code 0 only if the run completed and the worker count matched what the
platform gate promised.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def emit(key: str, value) -> None:
    print(f"[probe] {key} = {value}", flush = True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default = None, help = "repo root (default: two levels up)")
    parser.add_argument("--model", default = "trl-internal-testing/tiny-Qwen3ForCausalLM")
    parser.add_argument("--rows", type = int, default = 512)
    parser.add_argument("--max-steps", type = int, default = 5)
    parser.add_argument("--max-seq-length", type = int, default = 64)
    parser.add_argument("--out", default = None)
    parser.add_argument("--timeout", type = float, default = 900.0)
    args = parser.parse_args()

    # A hang must surface as a timeout, not a mystery: hard-exit rather than join
    # a wedged worker. Cross-platform, unlike the `timeout` command.
    if args.timeout > 0:
        import threading

        def _die() -> None:
            print(f"[probe] FATAL = timed out after {args.timeout}s", flush = True)
            os._exit(4)

        watchdog = threading.Timer(args.timeout, _die)
        watchdog.daemon = True
        watchdog.start()

    repo = Path(args.repo).resolve() if args.repo else Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo / "studio" / "backend"))
    sys.path.insert(0, str(repo))

    report: dict = {
        "platform": sys.platform,
        "python": platform.python_version(),
        "machine": platform.machine(),
        "repo": str(repo),
    }
    emit("sys.platform", sys.platform)
    emit("python", platform.python_version())

    # ---- 1. the facts the gate reads -------------------------------------
    methods = multiprocessing.get_all_start_methods()
    report["get_all_start_methods"] = list(methods)
    report["get_all_start_methods[0]"] = methods[0] if methods else None
    emit("get_all_start_methods()", list(methods))
    emit("get_all_start_methods()[0]", methods[0] if methods else None)

    from utils.datasets.online_tokenization import (  # noqa: E402
        attach_online_tokenization,
        dataloader_worker_start_method,
        decide_online_tokenization,
        first_sample_text,
        online_config_args,
        platform_supports_dataloader_workers,
        release_train_dataloader,
        resolve_add_special_tokens,
        resolve_worker_count,
        trl_supports_skip_prepare_dataset,
    )

    start_method = dataloader_worker_start_method()
    supported = platform_supports_dataloader_workers()
    resolved = resolve_worker_count()
    report["dataloader_worker_start_method"] = start_method
    report["platform_supports_dataloader_workers"] = supported
    report["resolve_worker_count"] = resolved
    report["trl_supports_skip_prepare_dataset"] = trl_supports_skip_prepare_dataset()
    emit("dataloader_worker_start_method()", start_method)
    emit("platform_supports_dataloader_workers()", supported)
    emit("resolve_worker_count()", resolved)
    emit("trl_supports_skip_prepare_dataset()", report["trl_supports_skip_prepare_dataset"])

    # The claim under test: no fork means no workers, on any OS.
    expected_workers_zero = not supported
    if expected_workers_zero and resolved != 0:
        emit("FATAL", f"gate says unsupported but resolve_worker_count()={resolved}")
        return 3
    if sys.platform in ("win32", "darwin") and supported:
        emit("FATAL", f"{sys.platform} must never support DataLoader workers")
        return 3

    # ---- 2. the decision --------------------------------------------------
    import torch  # noqa: E402
    from datasets import Dataset  # noqa: E402
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
    from trl import SFTConfig, SFTTrainer  # noqa: E402

    torch.manual_seed(0)
    report["torch"] = torch.__version__
    emit("torch", torch.__version__)

    texts = [f"Row {i}: the quick brown fox jumps over the lazy dog. " * 3 for i in range(args.rows)]
    dataset = Dataset.from_dict({"text": texts})
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype = torch.float32)
    model.to("cpu")

    # A tiny split cannot clear the 10k-row cost gate, and the point here is the
    # platform gate, so force past the cost gates only (the correctness gates,
    # including the fork gate, are not overridable).
    os.environ["UNSLOTH_STUDIO_ONLINE_TOKENIZATION"] = "1"
    decision = decide_online_tokenization(
        dataset = dataset,
        processing_class = tokenizer,
        model = model,
        text_field = "text",
        num_train_epochs = 1.0,
        max_steps = args.max_steps,
        grad_accum = 1,
    )
    report["decision"] = {
        "enabled": decision.enabled,
        "reason": decision.reason,
        "workers": decision.workers,
        "prefetch_factor": decision.prefetch_factor,
        "prewarm_batches": decision.prewarm_batches,
    }
    emit("decision", decision.as_log_line())
    emit("path", "online" if decision.enabled else "eager")

    if sys.platform in ("win32", "darwin") and decision.enabled:
        emit("FATAL", f"{sys.platform} took the online path")
        return 3
    if decision.enabled and decision.workers < 1:
        emit("FATAL", "online path with zero workers")
        return 3

    # ---- 3. a real CPU training run ---------------------------------------
    out_dir = Path(os.environ.get("RUNNER_TEMP") or ".") / f"xplat_probe_{int(time.time())}"
    common = dict(
        output_dir = str(out_dir),
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 1,
        max_steps = args.max_steps,
        learning_rate = 1e-4,
        logging_steps = 1,
        report_to = [],
        save_strategy = "no",
        seed = 0,
        use_cpu = True,
        max_length = args.max_seq_length,
    )
    train_dataset = dataset
    if decision.enabled:
        add_special = resolve_add_special_tokens(tokenizer, first_sample_text(dataset, "text"))
        train_dataset = attach_online_tokenization(
            dataset,
            tokenizer = tokenizer,
            text_field = "text",
            max_length = args.max_seq_length,
            add_special_tokens = add_special,
        )
        common.update(online_config_args(decision))
        report["add_special_tokens"] = add_special

    config = SFTConfig(**common)
    trainer = SFTTrainer(
        model = model,
        args = config,
        train_dataset = train_dataset,
        processing_class = tokenizer,
    )
    handed = int(getattr(trainer.args, "dataloader_num_workers", 0) or 0)
    report["trainer_dataloader_num_workers"] = handed
    emit("trainer.args.dataloader_num_workers", handed)

    # Studio's own sequence: memoize the loader, drain the prewarm barrier, train,
    # release. Skipping the memo would leave release_train_dataloader nothing to
    # shut down, so the child-process check below would prove nothing.
    if decision.enabled:
        from utils.datasets.online_tokenization import memoize_train_dataloader

        report["memoized"] = bool(memoize_train_dataloader(trainer))
        emit("memoize_train_dataloader", report["memoized"])

    loader = trainer.get_train_dataloader()
    loader_workers = int(getattr(loader, "num_workers", 0) or 0)
    report["train_dataloader_num_workers"] = loader_workers
    emit("train_dataloader.num_workers", loader_workers)

    if sys.platform in ("win32", "darwin") and (handed or loader_workers):
        emit("FATAL", f"{sys.platform} was handed {handed}/{loader_workers} DataLoader workers")
        return 3
    if decision.enabled and loader_workers < 1:
        emit("FATAL", "online path built a 0-worker loader")
        return 3

    # The prewarm barrier, as _preflight_first_batch drains it.
    iterator = iter(loader)
    drained = 0
    for _ in range(max(1, decision.prewarm_batches if decision.enabled else 1)):
        try:
            next(iterator)
        except StopIteration:
            break
        drained += 1
    del iterator, loader
    report["prewarm_batches_drained"] = drained
    emit("prewarm_batches_drained", drained)

    began = time.perf_counter()
    result = trainer.train()
    elapsed = round(time.perf_counter() - began, 2)
    losses = [
        round(float(h["loss"]), 6) for h in trainer.state.log_history if "loss" in h
    ]
    report["steps_completed"] = int(trainer.state.global_step)
    report["losses"] = losses
    report["final_loss"] = losses[-1] if losses else None
    report["train_loss"] = round(float(result.training_loss), 6)
    report["seconds"] = elapsed
    emit("steps_completed", trainer.state.global_step)
    emit("per_step_losses", losses)
    emit("final_loss", report["final_loss"])
    emit("train_loss", report["train_loss"])
    emit("seconds", elapsed)

    if int(trainer.state.global_step) != args.max_steps:
        emit("FATAL", f"only {trainer.state.global_step} of {args.max_steps} steps ran")
        return 3
    if not losses or not all(l == l for l in losses):  # NaN check
        emit("FATAL", f"bad losses {losses}")
        return 3

    # ---- 4. clean shutdown -------------------------------------------------
    shut = release_train_dataloader(trainer)
    report["workers_released"] = shut
    emit("release_train_dataloader", shut)
    time.sleep(1.0)
    children = multiprocessing.active_children()
    report["live_children_after"] = [c.name for c in children]
    emit("live_children_after", [c.name for c in children])
    if children:
        emit("FATAL", "children still alive after release")
        return 3

    report["ok"] = True
    print("[probe] RESULT " + json.dumps(report, sort_keys = True), flush = True)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent = 2, sort_keys = True))
    emit("verdict", "PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
