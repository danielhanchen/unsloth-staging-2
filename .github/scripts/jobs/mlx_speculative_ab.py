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
    paired_native = bool(getattr(args, "paired_native", False))
    paired_scopes = bool(getattr(args, "paired_scopes", False))
    paired_logits = bool(getattr(args, "paired_logits", False))
    paired_forward = bool(getattr(args, "paired_forward", False))
    paired_verifier = paired_forward and getattr(args, "forward_method", None) == "verifier"
    paired_fixed = (
        paired_verifier
        or paired_forward
        and getattr(args, "forward_method", None) == "prefill_lazy_draft"
    )
    plain_engine = (
        defer_hidden
        or prime_handoff
        or copy_only
        or paired_native
        or paired_scopes
        or paired_logits
        or paired_forward
        and not paired_fixed
        or bool(getattr(args, "plain_engine", False) and args.arm == "head")
    )
    if plain_engine:
        assert getattr(args, "profile", False), "plain-engine isolation is diagnostic only"
        from unsloth_zoo.mlx.speculative import DraftController
        backend._speculative_draft.controller = DraftController(max_depth = 0, can_copy = False)
    if paired_fixed:
        from unsloth_zoo.mlx.speculative import DraftController
        backend._speculative_draft.controller = DraftController(
            max_depth = 2, can_copy = False, fixed_depth = True
        )
    engine_draft = backend._speculative_draft
    variant, scope_hits = [False], []
    logit_method, logit_details, logit_hits = ["normalize"], {}, {}
    forward_method, forward_hits = ["position2d"], {}
    forward_only = getattr(args, "forward_method", None)
    forward_methods = (
        ("tokens_only", "clear_cache")
        if forward_only == "loop"
        else ((forward_only,) if forward_only else ("position2d", "retained_rope", "no_hidden"))
    )
    clear_cache_hits = []
    token_ids = []
    engine_records = []
    verify_hits = {}
    prefill_hits = {}
    forward_trace = []
    trace_forwards = bool(getattr(args, "trace_forwards", False))
    if trace_forwards:
        import core.inference.mlx_inference as backend_module

        original_factory = backend_module._recording_class
        original_responses = backend_module._iter_vlm_responses

        def description(value):
            if isinstance(value, mx.array):
                return dict(shape = list(value.shape), dtype = str(value.dtype))
            return (
                value
                if value is None or isinstance(value, (bool, int, float, str))
                else type(value).__qualname__
            )

        def traced_factory(base):
            cls = original_factory(base)
            if not vars(cls).get("_diagnostic_forward_trace"):
                original_call = cls.__call__

                def traced_call(model, *args, **kwargs):
                    inputs = args[0] if args else kwargs.get("inputs")
                    cache = kwargs.get("cache") or []
                    item = dict(
                        inputs = description(inputs),
                        kwargs = {
                            key: description(value)
                            for key, value in kwargs.items()
                            if key != "cache"
                        },
                        cache_before = [
                            dict(
                                type = type(entry).__qualname__,
                                offset = description(getattr(entry, "offset", None)),
                            )
                            for entry in cache
                        ],
                    )
                    result = original_call(model, *args, **kwargs)
                    item["logits"] = description(result.logits)
                    item["hidden"] = [description(value) for value in result.hidden_states or []]
                    item["cache_after"] = [
                        description(getattr(entry, "offset", None)) for entry in cache
                    ]
                    forward_trace.append(item)
                    return result

                cls.__call__ = traced_call
                cls._diagnostic_forward_trace = True
            return cls

        def traced_responses(responses):
            iterator = original_responses(responses)
            try:
                for response in iterator:
                    if response.token is not None:
                        token_ids.append(int(response.token))
                    yield response
            finally:
                iterator.close()

        backend_module._recording_class = traced_factory
        backend_module._iter_vlm_responses = traced_responses
    if paired_forward:
        from unsloth_zoo.mlx.speculative import SpeculativeEngine
        from unsloth_zoo.mlx.speculative import _language_call
        import core.inference.mlx_inference as backend_module

        original_responses = backend_module._iter_vlm_responses

        def recorded_responses(responses):
            iterator = original_responses(responses)
            try:
                for response in iterator:
                    if response.token is not None:
                        token_ids.append(int(response.token))
                    yield response
            finally:
                iterator.close()

        backend_module._iter_vlm_responses = recorded_responses
        original_init = SpeculativeEngine.__init__

        def recorded_init(engine, *args, **kwargs):
            original_init(engine, *args, **kwargs)
            cls = type(engine.lm)
            detail = dict(
                variant = "B" if variant[0] else "A",
                method = forward_method[0],
                model_type = cls.__qualname__,
                forward_module = cls.__call__.__module__,
                exact_before = engine._exact,
                invariant = engine._invariant,
                studio_base = repr(getattr(cls, "_studio_base", None)),
            )
            if paired_verifier:
                base = getattr(cls, "_studio_base", None)
                assert base is not None and base.__call__ is _language_call("qwen3_5"), detail
                assert not engine._exact, detail
                if variant[0]:
                    engine._exact = engine._batch_rounds = True
            detail["exact_after"] = engine._exact
            engine_records.append(detail)

        SpeculativeEngine.__init__ = recorded_init
        if paired_verifier:
            original_verify = SpeculativeEngine._verify

            def recorded_verify(engine, inputs):
                key = "exact" if engine._exact else "generic"
                verify_hits[key] = verify_hits.get(key, 0) + 1
                return original_verify(engine, inputs)

            SpeculativeEngine._verify = recorded_verify

        original_kwargs, original_hidden = SpeculativeEngine._kwargs, SpeculativeEngine._hidden
        original_add = SpeculativeEngine.add

        def selected_add(engine, *args, **kwargs):
            result = original_add(engine, *args, **kwargs)
            if variant[0] and forward_method[0] == "no_hidden":
                for row in engine._rows:
                    row.draft = None
            if variant[0] and forward_method[0] == "clear_cache":
                mx.clear_cache()
                clear_cache_hits.append(1)
            return result

        def selected_kwargs(engine):
            result = original_kwargs(engine)
            if not variant[0]:
                return result
            method = forward_method[0]
            assert len(engine._rows) == 1 and engine._mrope
            forward_hits[method] = forward_hits.get(method, 0) + 1
            if method == "position2d":
                cache = engine.cache[engine.lm.model.fa_idx]
                offset = cache.offset
                if isinstance(offset, mx.array):
                    assert offset.size == 1
                    offset = offset.item()
                result.pop("rope_deltas", None)
                result["position_ids"] = mx.array([[int(offset) + engine._rows[0].rope_delta]])
            elif method == "retained_rope":
                result.pop("rope_deltas", None)
            elif method == "no_hidden":
                result = engine._position_kwargs()
            return result

        def selected_hidden(engine, out):
            return (
                None
                if variant[0] and forward_method[0] == "no_hidden"
                else original_hidden(engine, out)
            )

        SpeculativeEngine._kwargs, SpeculativeEngine._hidden = selected_kwargs, selected_hidden
        SpeculativeEngine.add = selected_add
        if forward_only in ("prefill_lazy", "prefill_lazy_draft"):
            from unsloth_zoo.mlx.speculative import install_speculative_seam

            install_speculative_seam()
            prefill_type = importlib.import_module("mlx_vlm.generate.ar").SpeculativePrefill
            assert getattr(prefill_type, "unsloth_upstream", None) is not None
            original_append = prefill_type.append

            def selected_append(prefill, output):
                if prefill.tail:
                    key = "lazy" if variant[0] else "eager"
                    prefill_hits[key] = prefill_hits.get(key, 0) + 1
                    if variant[0]:
                        prefill.chunks = [
                            mx.concatenate([*prefill.chunks, output.hidden_states[-1]], axis = 1)[
                                :, -prefill.tail :
                            ]
                        ]
                        return
                return original_append(prefill, output)

            prefill_type.append = selected_append
    if paired_logits:
        from unsloth_zoo.mlx.speculative import _Row

        original_normalize = _Row.normalize

        def selected_normalize(row, logits):
            method = logit_method[0]
            if variant[0] and row.sample.temperature == 0 and row.scores is None:
                if method == "normalize":
                    result = logits - mx.logsumexp(logits, axis = -1, keepdims = True)
                elif method == "contiguous":
                    result = mx.contiguous(logits)
                else:
                    result = logits.astype(mx.float32)
                logit_hits[method] = logit_hits.get(method, 0) + 1
            else:
                result = original_normalize(row, logits)
            key = f"{method}:{'B' if variant[0] else 'A'}"
            if key not in logit_details:
                logit_details[key] = dict(
                    input_dtype = str(logits.dtype),
                    output_dtype = str(result.dtype),
                    shape = list(logits.shape),
                )
            return result

        _Row.normalize = selected_normalize
    if paired_scopes:
        original_scopes = backend._generation_scopes

        def selected_scopes(*traits):
            if variant[0] and "speculative" in traits:
                scope_hits.append(1)
                traits = tuple(t for t in traits if t != "speculative")
            return original_scopes(*traits)

        backend._generation_scopes = selected_scopes
    if copy_only and args.arm == "head":
        backend._speculative_draft.drafter = None
    hidden_eval_hits = []
    if defer_hidden and args.arm == "head" or paired_forward and forward_only == "loop":
        evaluate = mx.async_eval

        def token_eval(*arrays):
            active = (
                defer_hidden
                and args.arm == "head"
                or variant[0]
                and forward_method[0] == "tokens_only"
            )
            if (
                active
                and len(arrays) == 2
                and getattr(arrays[0], "ndim", None) == 1
                and getattr(arrays[1], "ndim", None) == 3
            ):
                hidden_eval_hits.append(list(arrays[1].shape))
                return evaluate(arrays[0])
            return evaluate(*arrays)

        mx.async_eval = token_eval
    handoff_prefetches = []
    if prime_handoff and args.arm == "head" or paired_forward and forward_only == "handoff":
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
        prepared_rounds = namespace["rounds"]
        if paired_forward:

            def selected_rounds(self, *args, **kwargs):
                return (prepared_rounds if variant[0] else original)(self, *args, **kwargs)

            SpeculativeDraft.rounds = selected_rounds
        else:
            SpeculativeDraft.rounds = prepared_rounds
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
        variant[0] = iteration % 4 in (1, 2)
        logit_method[0] = ("normalize", "contiguous", "fp32")[(iteration // 4) % 3]
        forward_method[0] = forward_methods[(iteration // 4) % len(forward_methods)]
        token_ids.clear()
        forward_trace.clear()
        native_iteration = paired_native and variant[0]
        backend._speculative_draft = None if native_iteration else engine_draft
        warm = iteration == 0
        backend._release_vlm_snapshots()
        mx.reset_peak_memory()
        profile_points = (
            (0, 9, 48, 49, 52, 53, 56, 57)
            if paired_logits or paired_forward and not forward_only
            else (0, 9)
        )
        profiler = cProfile.Profile() if profile_mode and iteration in profile_points else None
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
        if paired_native or paired_scopes or paired_logits or paired_forward:
            record["diagnostic_variant"] = "B" if variant[0] else "A"
            record["native_iteration"] = native_iteration
        if paired_logits:
            record["diagnostic_logit_method"] = logit_method[0]
        if paired_forward:
            record["diagnostic_forward_method"] = forward_method[0]
            assert token_ids, "raw token recording did not engage"
            record["token_ids"] = list(token_ids)
        if trace_forwards:
            assert (
                forward_trace and token_ids
            ), "forward trace or raw-token recording did not engage"
            record["forward_trace"] = list(forward_trace)
            record["token_ids"] = list(token_ids)
        if profiler is not None:
            profiler.disable()
            stream = io.StringIO()
            pstats.Stats(profiler, stream = stream).sort_stats("cumulative").print_stats(55)
            record["profile"] = stream.getvalue()
        controller = getattr(backend._speculative_draft, "controller", None)
        if plain_engine and not native_iteration:
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
    if paired_scopes:
        assert scope_hits, "speculative-scope diagnostic did not engage"
    if paired_logits:
        assert all(
            logit_hits.get(method, 0) > 0 for method in ("normalize", "contiguous", "fp32")
        ), logit_hits
    if paired_forward:
        assert all(forward_hits.get(method, 0) > 0 for method in forward_methods), forward_hits
        if forward_only == "loop":
            assert hidden_eval_hits and clear_cache_hits, (hidden_eval_hits, clear_cache_hits)
        if forward_only == "handoff":
            assert len(handoff_prefetches) == sum(
                iteration % 4 in (1, 2) for iteration in range(replies)
            ), handoff_prefetches
        if paired_verifier:
            assert (
                verify_hits.get("exact", 0) > 0 and verify_hits.get("generic", 0) > 0
            ), verify_hits
        if forward_only in ("prefill_lazy", "prefill_lazy_draft"):
            assert (
                prefill_hits.get("lazy", 0) > 0 and prefill_hits.get("eager", 0) > 0
            ), prefill_hits
        if forward_only == "prefill_lazy_draft":
            assert all(
                record["stats"]["timings"]["draft_n"] > 0 for record in results
            ), "fixed-depth drafting did not engage on every reply"
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
                diagnostic_paired_native = paired_native,
                diagnostic_paired_scopes = paired_scopes,
                native_scope_hits = len(scope_hits),
                diagnostic_paired_logits = paired_logits,
                logit_details = logit_details,
                logit_hits = logit_hits,
                diagnostic_paired_forward = paired_forward,
                forward_hits = forward_hits,
                engine_records = engine_records,
                verify_hits = verify_hits,
                prefill_hits = prefill_hits,
                clear_cache_hits = len(clear_cache_hits),
                trace_forwards = trace_forwards,
                device_info = mx.device_info()
                if trace_forwards and hasattr(mx, "device_info")
                else None,
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
    parser.add_argument(
        "--paired-native",
        action = "store_true",
        help = "diagnostic within-process ABBA: engine plain A, native AR B",
    )
    parser.add_argument(
        "--trace-forwards",
        action = "store_true",
        help = "record forward-call shapes and cache metadata; requires paired-native profiling and is not timing evidence",
    )
    parser.add_argument(
        "--paired-scopes",
        action = "store_true",
        help = "diagnostic within-process ABBA: engine plain with speculative scopes A, native scopes B",
    )
    parser.add_argument(
        "--paired-logits",
        action = "store_true",
        help = "diagnostic ABBA blocks: raw greedy logits versus normalization, contiguous storage, and fp32",
    )
    parser.add_argument(
        "--paired-forward",
        action = "store_true",
        help = "diagnostic ABBA blocks: explicit 2D positions, retained rope state, and omitted hidden-state capture",
    )
    parser.add_argument(
        "--forward-method",
        choices = (
            "position2d",
            "retained_rope",
            "no_hidden",
            "loop",
            "handoff",
            "verifier",
            "prefill_lazy",
            "prefill_lazy_draft",
        ),
        help = "repeat one intervention, or alternate the two loop interventions",
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
    if args.trace_forwards and not (args.paired_native and args.profile):
        parser.error("--trace-forwards requires --paired-native --profile")
    if args.forward_method and not args.paired_forward:
        parser.error("--forward-method requires --paired-forward")
    if (
        args.plain_engine
        or args.defer_hidden_eval
        or args.prime_plain_handoff
        or args.copy_only_plain
        or args.paired_native
        or args.paired_scopes
        or args.paired_logits
        or args.paired_forward
    ) and not args.profile:
        parser.error("decode diagnostics require --profile")
    if (
        sum(
            (
                args.defer_hidden_eval,
                args.prime_plain_handoff,
                args.copy_only_plain,
                args.paired_native,
                args.paired_scopes,
                args.paired_logits,
                args.paired_forward,
            )
        )
        > 1
    ):
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
                (
                    ("base",)
                    if args.paired_native
                    or args.paired_scopes
                    or args.paired_logits
                    or args.paired_forward
                    else ("base", "head")
                )
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
                if args.paired_native:
                    command.append("--paired-native")
                if args.trace_forwards:
                    command.append("--trace-forwards")
                if args.paired_scopes:
                    command.append("--paired-scopes")
                if args.paired_logits:
                    command.append("--paired-logits")
                if args.paired_forward:
                    command.append("--paired-forward")
                if args.forward_method:
                    command.extend(["--forward-method", args.forward_method])
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
