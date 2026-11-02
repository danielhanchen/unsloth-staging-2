# SPDX-License-Identifier: AGPL-3.0-only
"""Paired Studio MLX inference evidence on one Apple Silicon runner."""

import argparse
import cProfile
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import pstats
import subprocess
import sys
import tempfile
import time


def worker(args):
    os.environ["UNSLOTH_IS_PRESENT"] = "1"
    os.environ.pop("UNSLOTH_FORCE_GPU_PATH", None)
    zoo_source = getattr(args, "zoo_source", None)
    if zoo_source:
        sys.path.insert(0, str(Path(zoo_source).resolve()))
    sys.path.insert(0, str(Path(args.worker).resolve() / "studio/backend"))
    import mlx.core as mx
    from core.inference.mlx_inference import MLXInferenceBackend

    if zoo_source:
        import unsloth_zoo.mlx.speculative as runtime
        assert (
            Path(runtime.__file__).resolve()
            == Path(zoo_source).resolve() / "unsloth_zoo/mlx/speculative.py"
        )

    assert mx.metal.is_available(), "Metal GPU unavailable; no inference timing evidence"
    mx.set_default_device(mx.gpu)
    print("MLX_PROBE_DEVICE", mx.default_device(), flush = True)
    backend = MLXInferenceBackend()
    if getattr(args, "zoo_revision", None):
        installed = json.loads(
            importlib.metadata.distribution("unsloth_zoo").read_text("direct_url.json")
        )
        assert installed.get("vcs_info", {}).get("commit_id") == args.zoo_revision, installed
    backend.load_model(args.model, speculative_type = "mtp", max_seq_length = 2048)
    assert backend._speculative_draft is not None and backend._speculative_draft.drafter is not None
    native_head = bool(getattr(args, "native_draft_head", False) and args.arm == "head")
    if native_head:
        backend._speculative_draft.drafter._shortlist = None
    defer_hidden = bool(getattr(args, "defer_hidden_eval", False))
    prime_handoff = bool(getattr(args, "prime_plain_handoff", False))
    copy_only = bool(getattr(args, "copy_only_plain", False))
    plain_engine = (
        defer_hidden
        or prime_handoff
        or copy_only
        or bool(getattr(args, "plain_engine", False) and args.arm == "head")
    )
    if plain_engine:
        assert getattr(args, "profile", False), "plain-engine isolation is diagnostic only"
        from unsloth_zoo.mlx.speculative import DraftController
        backend._speculative_draft.controller = DraftController(max_depth = 0, can_copy = False)
    if copy_only and args.arm == "head":
        backend._speculative_draft.drafter = None
    hidden_eval_hits = []
    if defer_hidden and args.arm == "head":
        evaluate = mx.async_eval

        def token_eval(*arrays):
            if (
                len(arrays) == 2
                and getattr(arrays[0], "ndim", None) == 1
                and getattr(arrays[1], "ndim", None) == 3
            ):
                hidden_eval_hits.append(list(arrays[1].shape))
                return evaluate(arrays[0])
            return evaluate(*arrays)

        mx.async_eval = token_eval
    handoff_prefetches = []
    if prime_handoff and args.arm == "head":
        import inspect
        import textwrap
        from unsloth_zoo.mlx.speculative import SpeculativeDraft

        original = SpeculativeDraft.rounds
        source = textwrap.dedent(inspect.getsource(original))
        first_yield, loop = "    yield first, logprobs\n", "    while engine.rows:\n"
        assert source.count(first_yield) == source.count(loop) == 1
        queued = """    assert len(engine._rows) == 1 and not engine._rows[0].finished
    original_step = engine._step
    ready = original_step(mx.array([[engine._rows[0].pending]]))
    mx.async_eval(*(array for array in ready if array is not None))
    def prepared_step(*args, **kwargs):
        engine._step = original_step
        return ready
    engine._step = prepared_step
    _HANDOFF_PREFETCHES.append(1)
    yield first, logprobs
"""
        namespace = dict(original.__globals__, _HANDOFF_PREFETCHES = handoff_prefetches)
        exec(
            compile(
                source.replace(first_yield, "", 1).replace(loop, queued + loop, 1),
                "<diagnostic-plain-handoff>",
                "exec",
            ),
            namespace,
        )
        SpeculativeDraft.rounds = namespace["rounds"]
    requests = {
        "plain": {},
        "tools": {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "description": "Get weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ]
        },
        "grammar": {
            "response_format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                    "additionalProperties": False,
                },
            }
        },
    }
    results = []
    profile_mode = getattr(args, "profile", False)
    replies = getattr(args, "replies", None) or (10 if profile_mode else 2)
    for iteration in range(replies):
        warm = iteration == 0
        backend._release_vlm_snapshots()
        mx.reset_peak_memory()
        profiler = cProfile.Profile() if profile_mode and iteration in (0, 9) else None
        if profiler is not None:
            profiler.enable()
        started = time.perf_counter()
        first = None
        reply = ""
        for reply in backend.generate_chat_response(
            messages = [{"role": "user", "content": "Name the capital of France."}],
            temperature = 0.0,
            top_p = 1.0,
            top_k = 0,
            seed = 3407,
            enable_thinking = False,
            max_new_tokens = 64,
            **requests[args.scenario],
        ):
            if first is None:
                first = time.perf_counter() - started
        stats = backend.last_generation_stats
        assert stats and stats["usage"]["completion_tokens"] > 0
        if args.scenario == "grammar":
            assert isinstance(json.loads(reply)["city"], str)
        proposed = stats["timings"].get("draft_n", 0)
        if iteration == 0:
            assert (proposed > 0) == (
                not plain_engine
                and (
                    args.arm == "head"
                    or args.scenario == "plain"
                    or (bool(zoo_source) and not getattr(args, "stack", False))
                )
            ), stats
        record = dict(
            warmup = warm,
            iteration = iteration,
            ttft_s = first,
            wall_s = time.perf_counter() - started,
            peak_bytes = mx.get_peak_memory(),
            reply = reply,
            stats = stats,
        )
        if profiler is not None:
            profiler.disable()
            stream = io.StringIO()
            pstats.Stats(profiler, stream = stream).sort_stats("cumulative").print_stats(55)
            record["profile"] = stream.getvalue()
        controller = getattr(backend._speculative_draft, "controller", None)
        if plain_engine:
            assert controller is not None and controller.tokens > 0 and proposed == 0
        if profile_mode and controller is not None:
            record["controller"] = {
                "warmup": repr(getattr(controller, "_warmup", ())),
                "plain_cost": {
                    str(k): getattr(v, "value", None) for k, v in controller.plain_cost.items()
                },
                "tokens": controller.tokens,
            }
        results.append(record)
    if defer_hidden and args.arm == "head":
        assert hidden_eval_hits, "hidden-state evaluation diagnostic did not engage"
    if prime_handoff and args.arm == "head":
        assert (
            len(handoff_prefetches) == replies
        ), "plain handoff diagnostic did not engage on every reply"
    Path(args.out).write_text(
        json.dumps(
            dict(
                scenario = args.scenario,
                model = args.model,
                backend = "mlx",
                source = str(args.worker),
                native_draft_head = native_head,
                diagnostic_plain_engine = plain_engine,
                diagnostic_defer_hidden_eval = defer_hidden and args.arm == "head",
                hidden_eval_hits = hidden_eval_hits,
                diagnostic_prime_plain_handoff = prime_handoff and args.arm == "head",
                handoff_prefetches = len(handoff_prefetches),
                diagnostic_copy_only_plain = copy_only and args.arm == "head",
                platform = platform.platform(),
                device = str(mx.default_device()),
                metal = True,
                results = results,
                revision = subprocess.check_output(
                    ["git", "-C", args.worker, "rev-parse", "HEAD"], text = True
                ).strip(),
                versions = {
                    name: importlib.metadata.version(name)
                    for name in ("mlx", "mlx-lm", "mlx-vlm", "transformers", "unsloth_zoo")
                },
                zoo_direct_url = importlib.metadata.distribution("unsloth_zoo").read_text(
                    "direct_url.json"
                ),
                zoo_source = zoo_source,
                zoo_runtime_revision = subprocess.check_output(
                    ["git", "-C", zoo_source, "rev-parse", "HEAD"], text = True
                ).strip()
                if zoo_source
                else None,
            ),
            indent = 2,
            default = str,
        ),
        encoding = "utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker")
    parser.add_argument("--scenario", default = "plain")
    parser.add_argument("--arm", choices = ("base", "head"))
    parser.add_argument("--profile", action = "store_true")
    parser.add_argument(
        "--plain-engine",
        action = "store_true",
        help = "diagnostic profile with the head controller restricted to plain decoding",
    )
    parser.add_argument(
        "--defer-hidden-eval",
        action = "store_true",
        help = "diagnostic plain-vs-plain profile; head submits sampled tokens without explicitly evaluating hidden states",
    )
    parser.add_argument(
        "--prime-plain-handoff",
        action = "store_true",
        help = "diagnostic plain-vs-plain profile; head queues its first decode before yielding the prefill token",
    )
    parser.add_argument(
        "--copy-only-plain",
        action = "store_true",
        help = "diagnostic plain-vs-plain profile; head retains the engine but detaches its unused MTP drafter",
    )
    parser.add_argument("--native-draft-head", action = "store_true")
    parser.add_argument("--feature-probe", action = "store_true")
    parser.add_argument(
        "--feature-cases", help = "comma-separated capability probes; requires --feature-probe"
    )
    parser.add_argument("--zoo-revision")
    parser.add_argument("--zoo-source")
    parser.add_argument("--zoo-base")
    parser.add_argument("--zoo-head")
    parser.add_argument("--stack", action = "store_true")
    parser.add_argument("--replies", type = int)
    parser.add_argument("--model", default = "Qwen/Qwen3.5-0.8B")
    parser.add_argument("--base", default = "6401ceda29a54f2a8c3288deede3379f4d55619f")
    parser.add_argument("--head", default = "37836adeeadd836f7c009f804ae890b7d346679a")
    parser.add_argument("--out", required = True)
    args = parser.parse_args()
    if (
        args.plain_engine
        or args.defer_hidden_eval
        or args.prime_plain_handoff
        or args.copy_only_plain
    ) and not args.profile:
        parser.error("decode diagnostics require --profile")
    if sum((args.defer_hidden_eval, args.prime_plain_handoff, args.copy_only_plain)) > 1:
        parser.error("measure decode interventions independently")
    feature_cases = ("kv4", "kv8_batch", "video", "video_batch", "lora")
    if args.feature_cases:
        selected = tuple(args.feature_cases.split(","))
        if (
            not args.feature_probe
            or not selected
            or any(case not in feature_cases for case in selected)
        ):
            parser.error("--feature-cases requires --feature-probe and known capability names")
        feature_cases = selected
    if args.worker:
        if args.feature_probe:
            from mlx_speculative_features import worker as feature_worker
            feature_worker(args)
        else:
            worker(args)
        return
    root = Path.cwd().resolve()
    out = Path(args.out).resolve()
    output_dir = out.parent / (out.stem + "_arms")
    output_dir.mkdir(parents = True, exist_ok = True)
    scratch = Path(os.environ.get("RUNNER_TEMP", root / "temp"))
    scratch.mkdir(parents = True, exist_ok = True)
    work = Path(tempfile.mkdtemp(prefix = "mlx-ab-", dir = str(scratch)))
    subprocess.run(
        ["git", "fetch", "--no-tags", "--depth=1", "origin", args.base, args.head], check = True
    )
    trees = {"base": work / "base", "head": work / "head"}
    created = []
    rows = []
    failures = []
    try:
        zoo_trees = {}
        if args.zoo_base or args.zoo_head:
            assert args.zoo_base and args.zoo_head
            assert (
                args.base == args.head or args.stack
            ), "Zoo-only comparisons must hold Studio fixed"
            for side, ref in (("base", args.zoo_base), ("head", args.zoo_head)):
                source = work / ("zoo_" + side)
                subprocess.run(
                    [
                        "git",
                        "clone",
                        "--no-checkout",
                        "--depth=1",
                        "https://github.com/danielhanchen/unsloth-zoo-staging.git",
                        str(source),
                    ],
                    check = True,
                )
                subprocess.run(
                    ["git", "-C", str(source), "fetch", "--depth=1", "origin", ref], check = True
                )
                subprocess.run(["git", "-C", str(source), "checkout", "--detach", ref], check = True)
                zoo_trees[side] = source
        for side, ref in (("base", args.base), ("head", args.head)):
            subprocess.run(
                ["git", "worktree", "add", "--detach", str(trees[side]), ref], check = True
            )
            created.append(trees[side])
        for scenario in feature_cases if args.feature_probe else ("plain", "tools", "grammar"):
            arms = (
                ("base", "head")
                if args.profile or args.feature_probe
                else (
                    "base",
                    "base",
                    "base",
                    "base",
                    "head",
                    "head",
                    "base",
                    "base",
                    "head",
                    "head",
                    "base",
                )
            )
            for index, side in enumerate(arms):
                tree = trees[side]
                target = output_dir / f"{scenario}_{index}_{side}.json"
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    str(tree),
                    "--scenario",
                    scenario,
                    "--arm",
                    side,
                    "--model",
                    args.model,
                    "--out",
                    str(target),
                ]
                if args.profile:
                    command.append("--profile")
                if args.plain_engine:
                    command.append("--plain-engine")
                if args.defer_hidden_eval:
                    command.append("--defer-hidden-eval")
                if args.prime_plain_handoff:
                    command.append("--prime-plain-handoff")
                if args.copy_only_plain:
                    command.append("--copy-only-plain")
                if args.native_draft_head:
                    command.append("--native-draft-head")
                if args.feature_probe:
                    command.append("--feature-probe")
                if args.replies:
                    command.extend(["--replies", str(args.replies)])
                if args.stack:
                    command.append("--stack")
                if side in zoo_trees:
                    command.extend(["--zoo-source", str(zoo_trees[side])])
                if args.zoo_revision:
                    command.extend(["--zoo-revision", args.zoo_revision])
                env = dict(os.environ, HF_DOWNLOAD_OK = "1", UNSLOTH_DISABLE_AUTO_UPDATES = "1")
                process = subprocess.run(command, cwd = tree, env = env, check = not args.feature_probe)
                if process.returncode:
                    failures.append(
                        dict(side = side, scenario = scenario, returncode = process.returncode)
                    )
                    out.write_text(
                        json.dumps(dict(finished = False, rows = rows, failures = failures), indent = 2),
                        encoding = "utf-8",
                    )
                    continue
                rows.append(
                    dict(side = side, index = index, **json.loads(target.read_text(encoding = "utf-8")))
                )
                out.write_text(
                    json.dumps(dict(finished = False, rows = rows, failures = failures), indent = 2),
                    encoding = "utf-8",
                )
        out.write_text(
            json.dumps(dict(finished = True, rows = rows, failures = failures), indent = 2),
            encoding = "utf-8",
        )
        if failures:
            raise SystemExit(1)
    finally:
        for tree in created:
            subprocess.run(["git", "worktree", "remove", str(tree)], check = False)


if __name__ == "__main__":
    main()
