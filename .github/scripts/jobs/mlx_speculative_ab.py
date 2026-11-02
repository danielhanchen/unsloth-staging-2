# SPDX-License-Identifier: AGPL-3.0-only
"""Paired Studio MLX inference evidence on one Apple Silicon runner."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time


def worker(args):
    sys.path.insert(0, str(Path(args.worker).resolve() / "studio/backend"))
    import mlx.core as mx
    from core.inference.mlx_inference import MLXInferenceBackend

    assert mx.metal.is_available(), "Metal GPU unavailable; no inference timing evidence"
    mx.set_default_device(mx.gpu)
    print("MLX_PROBE_DEVICE", mx.default_device(), flush = True)
    backend = MLXInferenceBackend()
    backend.load_model(args.model, speculative_type = "mtp", max_seq_length = 2048)
    assert backend._speculative_draft is not None and backend._speculative_draft.drafter is not None
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
    for warm in (True, False):
        backend._release_vlm_snapshots()
        mx.reset_peak_memory()
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
        assert (proposed > 0) == (args.arm == "head" or args.scenario == "plain"), stats
        results.append(
            dict(
                warmup = warm,
                ttft_s = first,
                wall_s = time.perf_counter() - started,
                peak_bytes = mx.get_peak_memory(),
                reply = reply,
                stats = stats,
            )
        )
    Path(args.out).write_text(
        json.dumps(
            dict(
                scenario = args.scenario,
                model = args.model,
                backend = "mlx",
                source = str(args.worker),
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
    parser.add_argument("--model", default = "Qwen/Qwen3.5-0.8B")
    parser.add_argument("--base", default = "6401ceda29a54f2a8c3288deede3379f4d55619f")
    parser.add_argument("--head", default = "37836adeeadd836f7c009f804ae890b7d346679a")
    parser.add_argument("--out", required = True)
    args = parser.parse_args()
    if args.worker:
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
    try:
        for side, ref in (("base", args.base), ("head", args.head)):
            subprocess.run(
                ["git", "worktree", "add", "--detach", str(trees[side]), ref], check = True
            )
            created.append(trees[side])
        for scenario in ("plain", "tools", "grammar"):
            for index, side in enumerate(
                (
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
            ):
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
                env = dict(os.environ, HF_DOWNLOAD_OK = "1", UNSLOTH_DISABLE_AUTO_UPDATES = "1")
                subprocess.run(command, cwd = tree, env = env, check = True)
                rows.append(
                    dict(side = side, index = index, **json.loads(target.read_text(encoding = "utf-8")))
                )
                out.write_text(
                    json.dumps(dict(finished = False, rows = rows), indent = 2), encoding = "utf-8"
                )
        out.write_text(json.dumps(dict(finished = True, rows = rows), indent = 2), encoding = "utf-8")
    finally:
        for tree in created:
            subprocess.run(["git", "worktree", "remove", str(tree)], check = False)


if __name__ == "__main__":
    main()
