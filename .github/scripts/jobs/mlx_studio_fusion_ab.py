"""Studio MLX request-scoped Zoo fusions: merge-base Studio vs head Studio, per Zoo ref, on Apple Silicon.

    python jobs/mlx_studio_fusion_ab.py --base-sha SHA [--zoo-refs main,pull/1534,pull/1535] [--out f.json]

Runs from an unsloth checkout (staging branch = upstream main + PR). For each Zoo ref (pip
--no-deps --force-reinstall) and each model / generation path, a fresh worker process loads the
model through Studio's MLXInferenceBackend (base tree = studio/backend with mlx_inference.py
taken at --base-sha, head tree = checkout) and greedy-decodes a fixed prompt. Each Zoo scope
Studio enters is wrapped to record whether Studio entered it and how many modules it engaged.
Checks: replies identical base vs head; head enters every scope its Zoo exports; on a Zoo with
the new scopes, head engages them where base does not. Also runs the PR's refusal-count tests
against both trees (Gate A.2 on real MLX). Last line `JOB_RESULT {...}`; exit 0 / 1.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

SCOPES = (
    "fused_moe_gate_up",
    "fused_decode_conv_silu",
    "fused_residual_norm",
    "fused_moe_router",
    "fused_moe_routed_experts",
    "fused_residual_norm_handoff",
)
NEW = ("fused_moe_routed_experts", "fused_residual_norm_handoff")
PROMPT = "Write three short sentences about the ocean."
TINY_MOE = "tiny_qwen3_next_moe32"


def _ver(p):
    try:
        return md.version(p)
    except md.PackageNotFoundError:
        return None


# ----------------------------------------------------------------------------- worker
def worker(a):
    sys.path.insert(0, a.backend)
    os.chdir(a.backend)
    import unsloth_zoo.mlx.inference as Z
    from contextlib import contextmanager
    import traceback as tb

    rec = {"scopes": {}}

    def _engaged(model, name):
        n = 0
        for _, m in model.named_modules():
            if name == "fused_moe_routed_experts":
                n += "_unsloth_moe_routed" in getattr(m, "__dict__", {})
            elif name == "fused_residual_norm_handoff":
                n += (
                    "_unsloth_handoff_out" in getattr(m, "__dict__", {})
                    or type(m).__name__.startswith("_Prenorm")
                    or hasattr(type(m), "_unsloth_handoff_norm")
                )
            else:
                n += type(m).__name__.startswith("_Fused")
        return n

    for name in SCOPES:
        orig = getattr(Z, name, None)
        if orig is None:
            rec["scopes"][name] = {"exported": False}
            continue
        rec["scopes"][name] = {"exported": True, "studio_entries": 0, "engaged_max": 0}

        def make(orig, name):
            @contextmanager
            def wrapped(model):
                caller = next(
                    (
                        f
                        for f in reversed(tb.extract_stack()[:-1])
                        if not f.filename.endswith(("contextlib.py", os.path.basename(__file__)))
                    ),
                    None,
                )
                if caller is not None and caller.name == "_mlx_optional_fusion":
                    rec["scopes"][name]["studio_entries"] += 1
                with orig(model) as active:
                    try:
                        rec["scopes"][name]["engaged_max"] = max(
                            rec["scopes"][name]["engaged_max"], _engaged(model, name)
                        )
                    except Exception as e:  # noqa: BLE001 - instrumentation only
                        rec["scopes"][name]["engaged_error"] = repr(e)
                    yield active

            return wrapped

        setattr(Z, name, make(orig, name))

    from types import SimpleNamespace
    from core.inference.mlx_inference import MLXInferenceBackend
    import core.inference.mlx_inference as mi

    rec["studio_file"] = mi.__file__
    be = MLXInferenceBackend()
    cfg = SimpleNamespace(identifier = a.model, is_vision = a.path == "vlm", is_lora = False)
    t0 = time.time()
    assert be.load_model(cfg, max_seq_length = 2048) is True
    rec["load_s"] = round(time.time() - t0, 2)
    for s in rec["scopes"].values():
        if s.get("exported"):
            s["studio_entries"] = 0  # count request scopes only, not the load-time gate/up scope
    kw = dict(
        messages = [{"role": "user", "content": PROMPT}],
        temperature = 0.0,
        top_p = 1.0,
        top_k = 0,
        max_new_tokens = a.max_new_tokens,
        seed = 0,
        enable_thinking = False,
    )
    texts, stats = [], []
    for _ in range(a.reps):
        t0 = time.time()
        out = ""
        for snap in be.generate_chat_response(**kw):
            out = snap
        texts.append(out)
        st = be.last_generation_stats
        stats.append(
            {
                "wall_s": round(time.time() - t0, 3),
                **(
                    {
                        k: v
                        for k, v in (st if isinstance(st, dict) else vars(st) if st else {}).items()
                        if isinstance(v, (int, float, str))
                    }
                ),
            }
        )
    rec.update(text = texts[0], texts_consistent = len(set(texts)) == 1, stats = stats)
    Path(a.out).write_text(json.dumps(rec))


# ----------------------------------------------------------------------------- driver helpers
def build_tiny_moe(dst):
    """Random qwen3_next (Qwen sparse MoE + shared expert), 32 experts, 4-bit: the routed-experts kernel
    needs a multiple of 32 experts, which no public tiny checkpoint has."""
    if (dst / "config.json").exists():
        return
    from huggingface_hub import snapshot_download
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    src = Path(
        snapshot_download(
            "hf-internal-testing/tiny-random-Qwen3NextForCausalLM",
            allow_patterns = ["*.json", "*.jinja", "merges.txt", "vocab.json"],
        )
    )
    cfg = json.loads((src / "config.json").read_text())
    cfg.update(
        hidden_size = 512,
        intermediate_size = 512,
        num_hidden_layers = 4,
        full_attention_interval = 4,
        num_experts = 32,
        num_experts_per_tok = 4,
        moe_intermediate_size = 256,
        shared_expert_intermediate_size = 256,
        num_attention_heads = 8,
        num_key_value_heads = 2,
        head_dim = 64,
        linear_num_key_heads = 4,
        linear_num_value_heads = 8,
        linear_key_head_dim = 64,
        linear_value_head_dim = 64,
        decoder_sparse_step = 1,
        mlp_only_layers = [],
        tie_word_embeddings = False,
    )
    cfg.pop("layer_types", None)
    from mlx_lm.utils import _get_classes

    Model, Args = _get_classes(cfg)
    mx.random.seed(3407)
    model = Model(Args.from_dict(cfg))
    params = dict(tree_flatten(model.parameters()))
    model.update(
        __import__("mlx.utils", fromlist = ["tree_unflatten"]).tree_unflatten(
            [
                (k, (mx.random.normal(v.shape) * 0.05).astype(mx.bfloat16))
                if v.ndim >= 2
                else (k, v.astype(mx.bfloat16))
                for k, v in params.items()
            ]
        )
    )
    nn.quantize(
        model,
        group_size = 64,
        bits = 4,
        class_predicate = lambda p, m: hasattr(m, "to_quantized") and m.weight.shape[-1] % 64 == 0,
    )
    cfg["quantization"] = {"group_size": 64, "bits": 4}
    dst.mkdir(parents = True, exist_ok = True)
    mx.save_safetensors(
        str(dst / "model.safetensors"),
        dict(tree_flatten(model.parameters())),
        metadata = {"format": "mlx"},
    )
    (dst / "config.json").write_text(json.dumps(cfg, indent = 1))
    for f in src.iterdir():
        if f.name != "config.json" and f.is_file():
            shutil.copy2(f, dst / f.name)


def base_tree(root, base_sha):
    dst = root / "_base_studio_backend"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree("studio/backend", dst, ignore = shutil.ignore_patterns("__pycache__"))
    rel = "studio/backend/core/inference/mlx_inference.py"
    url = f"https://raw.githubusercontent.com/unslothai/unsloth/{base_sha}/{rel}"
    (dst / "core/inference/mlx_inference.py").write_bytes(
        urllib.request.urlopen(url, timeout = 60).read()
    )
    return dst


def install_zoo(ref):
    spec = "git+https://github.com/unslothai/unsloth-zoo" + (
        "" if ref == "main" else f"@refs/{ref}/head"
    )
    for attempt in range(3):
        r = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "--no-deps",
                "--force-reinstall",
                f"unsloth_zoo @ {spec}",
            ]
        )
        if r.returncode == 0:
            break
        time.sleep(5 * (attempt + 1))
    else:
        raise SystemExit(f"zoo install {ref} failed")
    head = subprocess.run(
        [
            "git",
            "ls-remote",
            "https://github.com/unslothai/unsloth-zoo",
            "refs/heads/main" if ref == "main" else f"refs/{ref}/head",
        ],
        capture_output = True,
        text = True,
    ).stdout.split()[:1]
    return head[0] if head else None


def run_worker(backend, model, path, out, reps, max_new):
    env = {**os.environ, "PYTHONPATH": str(backend)}
    env.pop("UNSLOTH_IS_PRESENT", None)
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--worker",
        "--backend",
        str(backend),
        "--model",
        model,
        "--path",
        path,
        "--wout",
        str(out),
        "--reps",
        str(reps),
        "--max-new-tokens",
        str(max_new),
    ]
    r = subprocess.run(cmd, env = env, capture_output = True, text = True, timeout = 1200)
    if r.returncode != 0 or not out.exists():
        return {"error": (r.stdout[-1500:] + "\n" + r.stderr[-4000:])}
    return json.loads(out.read_text())


def pytest_arm(backend, test_file, k):
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(test_file), "-k", k],
        cwd = backend,
        env = {**os.environ, "PYTHONPATH": str(backend)},
        capture_output = True,
        text = True,
        timeout = 900,
    )
    tail = r.stdout.strip().splitlines()[-1:] or [""]
    return {
        "rc": r.returncode,
        "summary": tail[0],
        "failed": [l for l in r.stdout.splitlines() if l.startswith("FAILED")],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action = "store_true")
    ap.add_argument("--backend")
    ap.add_argument("--model")
    ap.add_argument("--path", default = "text")
    ap.add_argument("--wout")
    ap.add_argument("--reps", type = int, default = 2)
    ap.add_argument("--max-new-tokens", type = int, default = 48)
    ap.add_argument("--base-sha", default = None)
    ap.add_argument("--zoo-refs", default = "main,pull/1534,pull/1535")
    ap.add_argument(
        "--models",
        default = f"{TINY_MOE}:text,mlx-community/Qwen3-0.6B-4bit:text,"
        "mlx-community/Qwen3.5-2B-4bit:text,mlx-community/Qwen3.5-2B-4bit:vlm",
    )
    ap.add_argument("--out", default = "mlx_studio_fusion_ab.json")
    ap.add_argument("--tiny", action = "store_true", help = "ignored (staging_ci)")
    a = ap.parse_args()
    if a.worker:
        a.out = a.wout
        return worker(a)

    root = Path.cwd()
    work = root / "_fusion_ab"
    work.mkdir(exist_ok = True)
    res = {
        "job": "mlx_studio_fusion_ab",
        "base_sha": a.base_sha,
        "checks": {},
        "arms": [],
        "pytest": {},
        "versions": {p: _ver(p) for p in ("mlx", "mlx-lm", "mlx-vlm", "transformers")},
        "passed": False,
    }
    out = Path(a.out)

    def save():
        out.write_text(json.dumps(res, indent = 1))

    def check(name, ok, detail):
        res["checks"][name] = {"ok": bool(ok), "detail": detail}
        print(("PASS " if ok else "FAIL ") + name + ": " + json.dumps(detail)[:600], flush = True)
        save()

    base = base_tree(work, a.base_sha)
    head = root / "studio" / "backend"
    build_tiny_moe(work / TINY_MOE)
    models = [
        (m if m != TINY_MOE else str(work / TINY_MOE), p, m)
        for m, p in (x.rsplit(":", 1) for x in a.models.split(","))
    ]
    test_file = head / "tests" / "test_mlx_inference_backend.py"
    k = "refuses_everywhere or survives_every_fusion_refusing"
    for ref in a.zoo_refs.split(","):
        sha = install_zoo(ref)
        res.setdefault("zoo", {})[ref] = sha
        if ref == "main":
            # Head's tests vs base and head Studio (base must fail the six-scope refusal counts).
            shutil.copy2(test_file, base / "tests" / "test_mlx_inference_backend.py")
            res["pytest"] = {
                "base": pytest_arm(base, "tests/test_mlx_inference_backend.py", k),
                "head": pytest_arm(head, "tests/test_mlx_inference_backend.py", k),
            }
            check(
                "gateA2_refusal_tests_fail_on_base_pass_on_head",
                res["pytest"]["base"]["rc"] == 1 and res["pytest"]["head"]["rc"] == 0,
                res["pytest"],
            )
        for model, path, label in models:
            arms = {}
            order = ["base", "head", "head", "base"] if label == TINY_MOE else ["base", "head"]
            for i, side in enumerate(order):
                wout = (
                    work
                    / f"w_{ref.replace('/', '_')}_{label.replace('/', '_')}_{path}_{i}_{side}.json"
                )
                r = run_worker(
                    base if side == "base" else head, model, path, wout, a.reps, a.max_new_tokens
                )
                r.update(zoo = ref, model = label, path = path, side = side, order = i)
                res["arms"].append(r)
                arms.setdefault(side, []).append(r)
                save()
                print(
                    f"ARM zoo={ref} model={label} path={path} side={side} "
                    f"err={'error' in r} text={r.get('text', '')[:80]!r} "
                    f"scopes={json.dumps(r.get('scopes'))}",
                    flush = True,
                )
            tag = f"{ref}|{label}|{path}"
            errs = [r["error"] for rs in arms.values() for r in rs if "error" in r]
            if errs:
                check(f"runs[{tag}]", False, errs[0][-2500:])
                continue
            texts = {r["text"] for rs in arms.values() for r in rs}
            check(
                f"same_reply_base_vs_head[{tag}]",
                len(texts) == 1 and all(r["texts_consistent"] for rs in arms.values() for r in rs),
                sorted(texts),
            )
            h, b = arms["head"][0]["scopes"], arms["base"][0]["scopes"]
            for name in NEW:
                if not h[name]["exported"]:
                    check(
                        f"absent_export_is_noop[{tag}|{name}]",
                        "studio_entries" not in h[name],
                        h[name],
                    )
                    continue
                check(
                    f"head_enters[{tag}|{name}]",
                    h[name]["studio_entries"] >= 1 and b[name]["studio_entries"] == 0,
                    {"head": h[name], "base": b[name]},
                )
            for name in SCOPES:
                if name not in NEW and h[name].get("exported"):
                    check(
                        f"existing_scope_unchanged[{tag}|{name}]",
                        h[name]["studio_entries"] == b[name]["studio_entries"]
                        and h[name]["engaged_max"] == b[name]["engaged_max"],
                        {"head": h[name], "base": b[name]},
                    )
            res.setdefault("engagement", {})[tag] = {
                n: {"head": h[n].get("engaged_max"), "base": b[n].get("engaged_max")}
                for n in NEW
                if h[n].get("exported")
            }
    eng = res.get("engagement", {})
    for ref, name, models_ in (
        ("pull/1534", "fused_moe_routed_experts", (TINY_MOE,)),
        ("pull/1535", "fused_residual_norm_handoff", None),
    ):
        if ref not in a.zoo_refs.split(","):
            continue
        hits = {
            t: v[name]
            for t, v in eng.items()
            if t.startswith(ref + "|")
            and name in v
            and (models_ is None or t.split("|")[1] in models_)
        }
        check(
            f"new_scope_engages_on_head_only[{ref}|{name}]",
            any((v["head"] or 0) > 0 for v in hits.values()),
            hits,
        )
    res["passed"] = all(c["ok"] for c in res["checks"].values()) and bool(res["checks"])
    save()
    print("ENGAGEMENT " + json.dumps(res.get("engagement")), flush = True)
    print(
        "JOB_RESULT "
        + json.dumps(
            {
                "passed": res["passed"],
                "failed": [k for k, c in res["checks"].items() if not c["ok"]],
            }
        )
    )
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
