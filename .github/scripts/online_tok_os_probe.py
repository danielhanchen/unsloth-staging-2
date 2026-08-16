"""Cross-OS proof for the online (overlapped) tokenization path.

Not a unit test: it reports the real platform facts the gate reads, then runs a
few CPU training steps through whichever path the gate chose, so a leg that
declines the online path still has to complete on the eager path and say so.
Everything is printed as PROBE: key=value lines plus one PROBE_JSON blob.
"""

import json
import multiprocessing
import os
import platform
import subprocess
import sys
import time

REPO = os.environ.get("PROBE_REPO") or os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "studio", "backend"))

BATCH = 2
GRAD_ACCUM = 2
MAX_STEPS = 3
MAX_LENGTH = 128

OUT = {"os": sys.platform, "platform": platform.platform(), "python": sys.version.split()[0]}


def emit(key, value):
    OUT[key] = value
    print(f"PROBE: {key}={value}", flush = True)


def main():
    import torch
    import transformers
    import trl
    import datasets as hfds
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    from utils.datasets.online_tokenization import (
        attach_online_tokenization,
        dataloader_worker_start_method,
        decide_online_tokenization,
        first_sample_text,
        online_config_args,
        platform_supports_dataloader_workers,
        memoize_train_dataloader,
        release_train_dataloader,
        resolve_add_special_tokens,
        resolve_worker_count,
        trl_supports_skip_prepare_dataset,
    )

    baseline = set(child_pids())
    emit("baseline_children", len(baseline))

    emit("torch", torch.__version__)
    emit("transformers", transformers.__version__)
    emit("trl", trl.__version__)
    emit("datasets", hfds.__version__)

    # 1. the platform facts the gate reads
    all_methods = multiprocessing.get_all_start_methods()
    emit("get_all_start_methods", ",".join(all_methods))
    emit("default_start_method", all_methods[0] if all_methods else "none")
    emit("dataloader_worker_start_method", dataloader_worker_start_method())
    emit("platform_supports_dataloader_workers", platform_supports_dataloader_workers())
    emit("trl_supports_skip_prepare_dataset", trl_supports_skip_prepare_dataset())

    # 2. a tiny real model + a tiny real text split
    model_id = os.environ.get("PROBE_MODEL", "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id)
    # >= MIN_ROWS_FOR_ONLINE (10,000) or the cost gate vetoes and the online path
    # is never exercised. The rows are short, so this stays cheap on a CPU runner.
    n_rows = int(os.environ.get("PROBE_ROWS", "12000"))
    rows = [f"Row {i}: the quick brown fox jumps over the lazy dog again and again." for i in range(n_rows)]
    train_dataset = Dataset.from_dict({"text": rows})
    eval_dataset = Dataset.from_dict({"text": rows[:8]})
    emit("train_rows", len(train_dataset))
    emit("cpu_count", os.cpu_count())
    emit("resolve_worker_count", resolve_worker_count())

    # 3. the same gate call trainer.py makes for a plain-text run
    decision = decide_online_tokenization(
        dataset = train_dataset,
        eval_dataset = eval_dataset,
        processing_class = tokenizer,
        model = model,
        text_field = "text",
        packing = False,
        is_vlm = False,
        is_audio = False,
        is_audio_vlm = False,
        is_deepseek_ocr = False,
        is_cpt = False,
        raw_text_mode = False,
        has_custom_collator = False,
        train_on_completions = False,
        dataset_streaming = False,
        num_train_epochs = 1,
        max_steps = MAX_STEPS,
        grad_accum = GRAD_ACCUM,
        # what trainer.py resolves for a step-capped run: steps * microbatch * world
        resolved_max_steps_epochs = (MAX_STEPS * BATCH * GRAD_ACCUM) / len(train_dataset),
    )
    emit("decision_enabled", bool(decision.enabled))
    emit("decision_reason", decision.reason)
    emit("decision_workers", getattr(decision, "workers", None))
    emit("path_taken", "online" if decision.enabled else "eager")

    config_args = dict(
        output_dir = os.path.join(REPO, "probe_out"),
        per_device_train_batch_size = BATCH,
        gradient_accumulation_steps = GRAD_ACCUM,
        max_steps = MAX_STEPS,
        learning_rate = 1e-4,
        logging_steps = 1,
        report_to = [],
        max_length = MAX_LENGTH,
        dataset_text_field = "text",
        use_cpu = True,
        save_strategy = "no",
        seed = 3407,
    )

    train_split = train_dataset
    if decision.enabled:
        add_special_tokens = resolve_add_special_tokens(
            tokenizer, first_sample_text(train_dataset, "text")
        )
        emit("add_special_tokens", add_special_tokens)
        train_split = attach_online_tokenization(
            train_dataset,
            tokenizer = tokenizer,
            text_field = "text",
            max_length = MAX_LENGTH,
            add_special_tokens = add_special_tokens,
        )
        # the lazy view must produce input_ids in THIS process, at 0 workers too
        probe_row = train_split[0]
        emit("lazy_view_keys", ",".join(sorted(probe_row.keys())))
        emit("lazy_view_input_ids_len", len(probe_row["input_ids"]))
        config_args.update(online_config_args(decision))

    trainer = SFTTrainer(
        model = model,
        processing_class = tokenizer,
        train_dataset = train_split,
        args = SFTConfig(**config_args),
    )
    emit("args_dataloader_num_workers", int(trainer.args.dataloader_num_workers))
    emit("args_persistent_workers", bool(trainer.args.dataloader_persistent_workers))
    emit("args_skip_prepare_dataset", bool((trainer.args.dataset_kwargs or {}).get("skip_prepare_dataset")))

    # Replicate _preflight_first_batch: memoize the loader so train() reuses the
    # warmed workers, then drain the prewarm depth. Without the memo nothing holds
    # the loader and release_train_dataloader has no workers to reach.
    prewarm = int(getattr(decision, "prewarm_batches", 0) or 0)
    emit("prewarm_batches", prewarm)
    if prewarm:
        emit("memoized", bool(memoize_train_dataloader(trainer)))
    loader = trainer.get_train_dataloader()
    iterator = iter(loader)
    drained = 0
    for _ in range(max(1, prewarm)):
        try:
            batch = next(iterator)
            drained += 1
        except StopIteration:
            break
    emit("prewarm_batches_drained", drained)
    emit("prewarm_batch_keys", ",".join(sorted(batch.keys())))
    del iterator, loader

    started = time.time()
    result = trainer.train()
    emit("train_seconds", round(time.time() - started, 2))
    emit("steps_completed", int(trainer.state.global_step))
    emit("final_loss", float(result.training_loss))
    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    emit("loss_stream", ",".join(f"{v:.4f}" for v in losses))

    released = release_train_dataloader(trainer)
    emit("workers_released", int(released))

    # PHASE B: the lazy view driven at 0 DataLoader workers, in this process.
    # This is the shape Windows and macOS get if anyone ever forces the path on,
    # and it is what a skipped test leaves unproven: no worker may be forked, and
    # tokenization must still happen correctly on the main thread.
    b_add = resolve_add_special_tokens(tokenizer, first_sample_text(train_dataset, "text"))
    b_view = attach_online_tokenization(
        train_dataset,
        tokenizer = tokenizer,
        text_field = "text",
        max_length = MAX_LENGTH,
        add_special_tokens = b_add,
    )
    b_args = dict(config_args)
    b_args.update({
        "output_dir": os.path.join(REPO, "probe_out_b"),
        "dataset_kwargs": {"skip_prepare_dataset": True},
        "remove_unused_columns": False,
        "dataloader_num_workers": 0,
        "dataloader_persistent_workers": False,
    })
    b_args.pop("dataloader_prefetch_factor", None)
    b_model = AutoModelForCausalLM.from_pretrained(model_id)
    b_trainer = SFTTrainer(
        model = b_model,
        processing_class = tokenizer,
        train_dataset = b_view,
        args = SFTConfig(**b_args),
    )
    emit("b_dataloader_num_workers", int(b_trainer.args.dataloader_num_workers))
    b_result = b_trainer.train()
    emit("b_steps_completed", int(b_trainer.state.global_step))
    emit("b_final_loss", float(b_result.training_loss))
    b_losses = [h["loss"] for h in b_trainer.state.log_history if "loss" in h]
    emit("b_loss_stream", ",".join(f"{v:.4f}" for v in b_losses))
    if decision.enabled:
        # phase A was the same lazy view behind N workers, so at 0 workers it must
        # train identically. When phase A was eager the two are not comparable:
        # TRL's eager prepare appends EOS, so the row widths differ by design.
        emit("b_matches_phase_a", abs(OUT["b_final_loss"] - OUT["final_loss"]) < 1e-6)

    # 4. no orphaned children left behind
    time.sleep(2)
    kids = [p for p in child_pids() if p not in baseline]
    emit("child_processes_after", len(kids))
    if kids:
        emit("child_pids", ",".join(str(p) for p in kids))
        emit("child_cmdlines", " | ".join(describe(p) for p in kids))

    # PHASE C: ask for the workers explicitly, bypassing only the host-sizing
    # heuristic (a 4-vCPU CI runner resolves to 1 worker, under MIN_ONLINE_WORKERS,
    # so phase A never exercises the online path here). Every correctness gate still
    # applies, so on a spawn platform this must STILL be vetoed -- which is the
    # point: asking for workers must not be able to talk Windows into forking any.
    c_decision = decide_online_tokenization(
        dataset = train_dataset,
        eval_dataset = eval_dataset,
        processing_class = tokenizer,
        model = model,
        text_field = "text",
        packing = False,
        is_vlm = False,
        is_audio = False,
        is_audio_vlm = False,
        is_deepseek_ocr = False,
        is_cpt = False,
        raw_text_mode = False,
        has_custom_collator = False,
        train_on_completions = False,
        dataset_streaming = False,
        num_train_epochs = 1,
        max_steps = MAX_STEPS,
        grad_accum = GRAD_ACCUM,
        resolved_max_steps_epochs = (MAX_STEPS * BATCH * GRAD_ACCUM) / len(train_dataset),
        workers = 2,
    )
    emit("c_decision_enabled", bool(c_decision.enabled))
    emit("c_decision_reason", c_decision.reason)
    emit("c_decision_workers", int(getattr(c_decision, "workers", 0) or 0))
    emit("c_path_taken", "online" if c_decision.enabled else "eager")

    if c_decision.enabled:
        c_view = attach_online_tokenization(
            train_dataset,
            tokenizer = tokenizer,
            text_field = "text",
            max_length = MAX_LENGTH,
            add_special_tokens = b_add,
        )
        c_args = dict(config_args)
        c_args["output_dir"] = os.path.join(REPO, "probe_out_c")
        c_args.update(online_config_args(c_decision))
        c_model = AutoModelForCausalLM.from_pretrained(model_id)
        c_trainer = SFTTrainer(
            model = c_model,
            processing_class = tokenizer,
            train_dataset = c_view,
            args = SFTConfig(**c_args),
        )
        emit("c_dataloader_num_workers", int(c_trainer.args.dataloader_num_workers))
        c_prewarm = int(getattr(c_decision, "prewarm_batches", 0) or 0)
        emit("c_prewarm_batches", c_prewarm)
        emit("c_memoized", bool(memoize_train_dataloader(c_trainer)))
        c_loader = c_trainer.get_train_dataloader()
        c_iter = iter(c_loader)
        c_drained = 0
        for _ in range(max(1, c_prewarm)):
            try:
                next(c_iter)
                c_drained += 1
            except StopIteration:
                break
        emit("c_prewarm_batches_drained", c_drained)
        del c_iter, c_loader
        c_result = c_trainer.train()
        emit("c_steps_completed", int(c_trainer.state.global_step))
        emit("c_final_loss", float(c_result.training_loss))
        c_losses = [h["loss"] for h in c_trainer.state.log_history if "loss" in h]
        emit("c_loss_stream", ",".join(f"{v:.4f}" for v in c_losses))
        emit("c_workers_released", int(release_train_dataloader(c_trainer)))
        # N forked workers must produce exactly what 0 workers produced in phase B
        emit("c_matches_phase_b", abs(OUT["c_final_loss"] - OUT["b_final_loss"]) < 1e-6)

    # final scan, after every arm has run and released
    time.sleep(3)
    kids = [p for p in child_pids() if p not in baseline]
    emit("child_processes_after", len(kids))
    OUT.pop("child_cmdlines", None)
    OUT.pop("child_pids", None)
    if kids:
        emit("child_pids", ",".join(str(p) for p in kids))
        emit("child_cmdlines", " | ".join(describe(p) for p in kids))

    ok = (
        OUT["steps_completed"] == MAX_STEPS
        and OUT["final_loss"] == OUT["final_loss"]  # not NaN
        and OUT["child_processes_after"] == 0
        and OUT.get("prewarm_batches_drained", 0) > 0
        and OUT["b_steps_completed"] == MAX_STEPS
        and OUT["b_dataloader_num_workers"] == 0
        and OUT["b_final_loss"] == OUT["b_final_loss"]
        and OUT.get("b_matches_phase_a", True)
    )
    if OUT["decision_enabled"]:
        # the online path must hand back the workers it forked
        ok = ok and OUT["workers_released"] > 0
    if sys.platform in ("win32", "darwin"):
        # the gate must decline on both phases, and no path may ask for a worker
        ok = (
            ok
            and not OUT["decision_enabled"]
            and OUT["args_dataloader_num_workers"] == 0
            and not OUT["c_decision_enabled"]
            and OUT["dataloader_worker_start_method"] == "spawn"
            and OUT["platform_supports_dataloader_workers"] is False
        )
    else:
        # on a fork platform the explicitly-sized arm must really run online
        ok = (
            ok
            and OUT["c_decision_enabled"]
            and OUT["c_dataloader_num_workers"] == 2
            and OUT["c_steps_completed"] == MAX_STEPS
            and OUT["c_prewarm_batches_drained"] > 0
            and OUT["c_workers_released"] == 2
            and OUT["c_matches_phase_b"]
        )
    emit("probe_ok", ok)
    print("PROBE_JSON " + json.dumps(OUT), flush = True)
    return 0 if ok else 1


def describe(pid):
    """Best-effort command line for a surviving child, for the log."""
    try:
        if sys.platform == "win32":
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 f"(Get-CimInstance Win32_Process -Filter 'ProcessId={pid}').CommandLine"],
                capture_output = True, text = True, timeout = 120).stdout
        else:
            out = subprocess.run(["ps", "-o", "command=", "-p", str(pid)],
                                 capture_output = True, text = True, timeout = 60).stdout
        return f"{pid}:{out.strip()[:120]}"
    except Exception:  # noqa: BLE001
        return f"{pid}:?"


def child_pids():
    """Live child PIDs of this process, without psutil."""
    me = os.getpid()
    try:
        if sys.platform == "win32":
            # wmic is deprecated and absent on newer Windows images; CIM is the
            # supported replacement.
            proc = subprocess.Popen(
                ["powershell", "-NoProfile", "-Command",
                 f"(Get-CimInstance Win32_Process -Filter 'ParentProcessId={me}')"
                 ".ProcessId"],
                stdout = subprocess.PIPE, stderr = subprocess.DEVNULL, text = True,
            )
            out = proc.communicate(timeout = 120)[0]
            return [int(t) for t in out.split()
                    if t.isdigit() and int(t) not in (me, proc.pid)]
        # Popen, not run: the `ps` process is itself a child of this one and would
        # otherwise be counted as a leaked worker.
        proc = subprocess.Popen(
            ["ps", "-o", "pid=", "--ppid", str(me)] if sys.platform != "darwin"
            else ["ps", "-o", "pid=,ppid=", "-ax"],
            stdout = subprocess.PIPE, stderr = subprocess.DEVNULL, text = True,
        )
        out = proc.communicate(timeout = 60)[0]
        if sys.platform == "darwin":
            pids = []
            for line in out.splitlines():
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit() and int(parts[1]) == me:
                    if int(parts[0]) != proc.pid:
                        pids.append(int(parts[0]))
            return pids
        return [int(t) for t in out.split() if t.isdigit() and int(t) != proc.pid]
    except Exception as exc:  # noqa: BLE001 - an unreadable process table is not a failure
        print(f"PROBE: child_pid_scan_failed={exc}", flush = True)
        return []


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        import traceback
        traceback.print_exc()
        print("PROBE_JSON " + json.dumps({**OUT, "probe_ok": False, "crashed": True}), flush = True)
        sys.exit(1)
