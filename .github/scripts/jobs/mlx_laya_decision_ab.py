"""unsloth-zoo MLX Laya loader A/B on Apple Silicon: zoo main vs #1488 vs #1496 (fp32 / fp16 / bf16).

    python jobs/mlx_laya_decision_ab.py [--out mlx_laya_decision_ab.json]

Run from an unsloth checkout on macos-15 (staging_ci --job mlx_laya_decision_ab). Every arm runs in a fresh
process on this runner, against the three real convaiinnovations/laya checkpoints, through Studio's own
_MLXAgent + _predict (load_decision_model is wrapped to pass compute_dtype where the arm asks for it).
Per arm and checkpoint: load ok / error (with the load phase that failed), load seconds, MLX active /
cache / peak memory and RSS after load, request latency, a big request with Studio's MLX chunking on and
off (peak memory), and every answer, compared in the parent with laya's torch fp32 forward on CPU.
Also runs each revision's tests/test_mlx_decision_metal.py and the head test file against the base code.
Prints `JOB_RESULT {json}`; exits 1 only when the harness itself breaks (findings are data).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

ZOO = {
    "main": (
        "https://github.com/unslothai/unsloth-zoo",
        "f980552ad9f9f1d5b82bc2b0303aa062acd840f5",
    ),
    "pr1488": ("https://github.com/Lyxot/unsloth-zoo", "42a11f7d7739907213877f384b956f5dda48f11f"),
    "pr1496": ("https://github.com/Lyxot/unsloth-zoo", "bc5eb9dbf7b9969aa3bfbf176880416c2ff5b253"),
}
# compute: None = Studio decides, a dtype name = forced through the loader, "fp32env" = UNSLOTH_SYSTEMONE_FP32=1,
# "overflow" = Studio decides, and the first forward reports non-finite logits (fp16 overflow recovery).
ARMS = [
    (z, c)
    for z, c in (
        ("main", None),
        ("pr1488", None),
        ("pr1496", None),
        ("pr1496", "float16"),
        ("pr1496", "fp32env"),
        ("pr1496", "overflow"),
    )
    if not os.environ.get("LAYA_AB_ARMS")
    or f"{z}/{c or 'default'}" in os.environ["LAYA_AB_ARMS"].split(",")
]
PHASES = os.environ.get("LAYA_AB_PHASES", "0") == "1"
CHECKPOINTS = ["multilingual", "", "typed-decisions"]
REPO = "convaiinnovations/laya"
STATES = [
    "Everything is down and we have a demo at noon.",
    "I was charged twice for my subscription, please refund one of them.",
    "Mein Konto wurde zweimal belastet, bitte erstatten Sie den Betrag.",
    "Hola, me pueden devolver el dinero? " * 80,
    {"turns": ["hi", "the app crashes when I upload a file", "still broken"]},
]
QUESTIONS = {
    "urgent": {"type": "noul", "instructions": "Does the customer need a reply within the hour?"},
    "team": {
        "type": "choice",
        "instructions": "Which team should handle it?",
        "criteria": {"outage": "service down", "billing": "charges, refunds", "bug": "app defect"},
    },
    "tone": {
        "type": "score",
        "instructions": "How upset is the customer?",
        "criteria": ["calm", "annoyed", "furious"],
    },
}
ARTICLE = (
    "The General Data Protection Regulation is a European Union regulation on information privacy "
    "in the European Union and the European Economic Area. Controllers must implement appropriate "
    "technical measures. "
) * 60
BIG = {
    f"q{i}": {
        "type": "choice",
        "instructions": f"Which clause {i} applies?",
        "criteria": {f"option {j}": None for j in range(4)},
    }
    for i in range(16)
}
WORK = Path(os.environ.get("RUNNER_TEMP", "/tmp")) / "laya_ab"


def _answers_flat(result):
    out = {}
    for name, a in result["answers"].items():
        if a["type"] == "noul":
            out[name] = {"pick": a["noul"] >= 0.5, "p": [a["noul"]]}
        else:
            probs = list(a["probabilities"].values())
            out[name] = {
                "pick": a.get("choice", max(range(len(probs)), key = probs.__getitem__)),
                "p": probs,
            }
    return out


# ------------------------------------------------------------------ child: one arm x one checkpoint
def child(args):
    import psutil

    backend = Path.cwd() / "studio" / "backend"
    sys.path.insert(0, str(backend))
    res = {"arm": args.arm, "compute_dtype": args.compute, "checkpoint": args.sub or "english"}
    proc = psutil.Process()
    try:
        import mlx.core as mx

        import unsloth_zoo.mlx.decision as decision

        res["zoo_file"] = decision.__file__
        res["mlx"] = mx.__version__
        real = decision.load_decision_model
        forced = args.compute not in (None, "fp32env", "overflow")
        if args.compute == "fp32env":
            os.environ["UNSLOTH_SYSTEMONE_FP32"] = "1"

        phase = {"name": "import"}

        import functools

        # wraps keeps the loader's signature visible to Studio's inspect.signature check.
        @functools.wraps(real)
        def load(folder, *a, **k):
            phase["name"] = "load_decision_model"
            res["loader_kwargs"] = {n: str(v) for n, v in k.items()}
            if forced:
                k["compute_dtype"] = getattr(mx, args.compute)
            return real(folder, *a, **k)

        decision.load_decision_model = load
        from core.systemone import laya_runtime

        laya_runtime.load_decision_model = load
        folder = Path(args.folder)
        mx.reset_peak_memory()
        t0 = time.perf_counter()
        import inspect

        if "fp16_checkpoint" in inspect.signature(laya_runtime._MLXAgent).parameters:
            agent = laya_runtime._MLXAgent(folder, laya_runtime._stored_fp16(folder))
        else:
            agent = laya_runtime._MLXAgent(folder)
        mx.synchronize()
        res["load_s"] = round(time.perf_counter() - t0, 2)
        phase["name"] = "after_load"
        time.sleep(1.0)
        res["after_load"] = {
            "active_mib": mx.get_active_memory() / 2**20,
            "cache_mib": mx.get_cache_memory() / 2**20,
            "peak_mib": mx.get_peak_memory() / 2**20,
            "rss_mib": proc.memory_info().rss / 2**20,
        }
        res["agent_dtype"] = str(getattr(agent, "dtype", None))
        params = {}
        from mlx.utils import tree_flatten

        for mod in agent.model.modules():
            kind = type(mod).__name__
            if kind in ("Linear", "_Linear", "Embedding", "LayerNorm"):
                for _, p in tree_flatten(mod.parameters()):
                    params.setdefault(kind, set()).add(str(p.dtype))
        res["param_dtypes"] = {k: sorted(v) for k, v in params.items()}
        if args.compute == "overflow":
            logits = agent.model.logits
            calls = {"n": 0}

            def overflowing(batch):
                calls["n"] += 1
                out = logits(batch)
                return out * float("inf") if calls["n"] == 1 else out

            agent.model.logits = overflowing
        phase["name"] = "requests"
        answers, times = [], []
        for s in STATES:
            laya_runtime._predict(agent, s, QUESTIONS)
        for _ in range(3):
            for i, s in enumerate(STATES):
                t = time.perf_counter()
                out = laya_runtime._predict(agent, s, QUESTIONS)[0]
                times.append(time.perf_counter() - t)
                if len(answers) < len(STATES):
                    answers.append(_answers_flat(out))
        res["answers"] = answers
        if args.compute == "overflow":
            res["overflow"] = {
                "forward_calls": calls["n"],
                "agent_dtype_after": str(getattr(agent, "dtype", None)),
                "param_dtypes_after": sorted(
                    {str(p.dtype) for _, p in tree_flatten(agent.model.parameters())}
                ),
            }
        res["request_ms_median"] = round(statistics.median(times) * 1e3, 2)
        phase["name"] = "big_request"
        for label, budget in (("chunked", None), ("one_forward", 10**9)):
            chunk = getattr(laya_runtime, "_CHUNK_TOKENS", None)
            if isinstance(chunk, dict) and budget is not None:
                chunk["mlx"] = budget
            mx.clear_cache()
            mx.reset_peak_memory()
            t = time.perf_counter()
            out = laya_runtime._predict(agent, ARTICLE, BIG)[0]
            res[f"big_{label}"] = {
                "ms": round((time.perf_counter() - t) * 1e3, 1),
                "peak_mib": round(mx.get_peak_memory() / 2**20),
                "chunk_budget": (chunk or {}).get("mlx") if isinstance(chunk, dict) else None,
            }
            res.setdefault("big_answers", {})[label] = _answers_flat(out)
        res["ok"] = True
    except Exception as exc:  # noqa: BLE001 - the error is the result
        res["ok"] = False
        res["phase"] = locals().get("phase", {}).get("name")
        res["error"] = f"{type(exc).__name__}: {exc}"[:600]
        res["traceback"] = traceback.format_exc()[-2500:]
    Path(args.out).write_text(json.dumps(res, default = str), encoding = "utf-8")


# ------------------------------------------------------------------ child: load phases on zoo main's loader
def phases_child(args):
    """Replays zoo main's load_decision_model step by step with timings; `split` evaluates per module."""
    res = {"checkpoint": args.sub or "english", "mode": args.arm}
    try:
        import mlx.core as mx
        import mlx.nn as nn

        from unsloth_zoo.mlx import decision

        folder = Path(args.folder)
        t = {}
        s = time.perf_counter()
        enc = json.loads((folder / "encoder" / "config.json").read_text())
        agent_cfg = json.loads((folder / "rl_agent_config.json").read_text())
        every = enc.get("global_attn_every_n_layers", 3)
        enc.setdefault(
            "layer_types",
            [
                "full_attention" if i % every == 0 else "sliding_attention"
                for i in range(enc["num_hidden_layers"])
            ],
        )
        model = decision.DecisionModel(enc, agent_cfg.get("head_layers", 2))
        t["construct"] = time.perf_counter() - s
        s = time.perf_counter()
        raw = mx.load(str(folder / "model.safetensors"))
        t["mx_load"] = time.perf_counter() - s
        s = time.perf_counter()
        weights = [
            (decision._checkpoint_name(k), v.astype(mx.float32))
            for k, v in raw.items()
            if not k.startswith("act_head.") and k != "temperature"
        ]
        model.load_weights(weights, strict = True)
        model.eval()
        t["load_weights"] = time.perf_counter() - s
        s = time.perf_counter()
        if args.arm == "split":
            for _, mod in model.named_modules():
                if isinstance(mod, (nn.Linear, nn.Embedding, nn.LayerNorm)):
                    mx.eval(mod.parameters())
        mx.eval(model.parameters())
        t["eval"] = time.perf_counter() - s
        s = time.perf_counter()
        mx.synchronize()
        t["synchronize"] = time.perf_counter() - s
        res["seconds"] = {k: round(v, 3) for k, v in t.items()}
        res["ok"] = True
    except Exception as exc:  # noqa: BLE001
        res["ok"] = False
        res["seconds"] = {k: round(v, 3) for k, v in locals().get("t", {}).items()}
        res["error"] = f"{type(exc).__name__}: {exc}"[:600]
    Path(args.out).write_text(json.dumps(res), encoding = "utf-8")


# ------------------------------------------------------------------ parent
def sh(cmd, **kw):
    p = subprocess.run(cmd, capture_output = True, text = True, **kw)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def install_zoo(name, url, sha):
    target = WORK / f"zoo_{name}"
    src = WORK / f"src_{name}"
    if not (src / ".git").exists():
        sh(["git", "clone", "-q", "--filter=blob:none", url, str(src)])
    rc, out = sh(["git", "-C", str(src), "checkout", "-q", sha])
    if rc:
        sh(["git", "-C", str(src), "fetch", "-q", "origin", sha])
        rc, out = sh(["git", "-C", str(src), "checkout", "-q", sha])
    if rc:
        raise RuntimeError(f"checkout {name} {sha}: {out[-400:]}")
    rc, out = sh(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--no-deps",
            "--target",
            str(target),
            str(src),
        ]
    )
    if rc:
        raise RuntimeError(f"install {name}: {out[-800:]}")
    return target, src


def run_child(
    kind,
    arm,
    zoo_dir,
    compute,
    folder,
    sub,
    timeout = 1800,
):
    out = WORK / f"{kind}_{arm}_{compute}_{sub or 'english'}.json"
    env = {**os.environ, "PYTHONPATH": str(zoo_dir)}
    cmd = [
        sys.executable,
        __file__,
        f"--{kind}",
        "--arm",
        arm,
        "--folder",
        str(folder),
        "--sub",
        sub,
        "--out",
        str(out),
    ]
    if compute:
        cmd += ["--compute", compute]
    try:
        rc, log = sh(cmd, env = env, timeout = timeout, cwd = os.getcwd())
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout"}
    if out.exists():
        return json.loads(out.read_text(encoding = "utf-8"))
    return {"ok": False, "error": f"child rc={rc}", "log": log[-1500:]}


def reference(folder):
    """Laya's own torch fp32 forward on CPU through Studio's _predict (vendored laya)."""
    sys.path.insert(0, str(Path.cwd() / "studio" / "backend"))
    from core.systemone import laya_runtime

    laya = laya_runtime._laya()
    agent = laya.load(str(folder), device = "cpu")
    ans = [_answers_flat(laya_runtime._predict(agent, s, QUESTIONS)[0]) for s in STATES]
    big = _answers_flat(laya_runtime._predict(agent, ARTICLE, BIG)[0])
    del agent
    return ans, big


def compare(got, want):
    worst, flips = 0.0, 0
    for g, w in zip(got, want):
        for name in w:
            worst = max(worst, max(abs(a - b) for a, b in zip(g[name]["p"], w[name]["p"])))
            flips += int(g[name]["pick"] != w[name]["pick"])
    return round(worst, 5), flips


def pytest_ids(src, zoo_dir, test_file):
    # zoo's pyproject sets pytest pythonpath = ["."] at the rootdir, which is the test file's checkout: a head test
    # file must sit inside the base checkout, or it imports the head's unsloth_zoo.
    if not Path(test_file).resolve().is_relative_to(Path(src).resolve()):
        copy = Path(src) / "tests" / f"test_head_{Path(test_file).parent.parent.name}.py"
        copy.write_text(Path(test_file).read_text(encoding = "utf-8"), encoding = "utf-8")
        test_file = copy
    env = {**os.environ, "PYTHONPATH": str(zoo_dir)}
    rc, out = sh(
        [sys.executable, "-m", "pytest", "-q", "-rfEs", "-p", "no:cacheprovider", str(test_file)],
        env = env,
        cwd = str(src),
        timeout = 1800,
    )
    failed = sorted(
        {line.split()[1] for line in out.splitlines() if line.startswith(("FAILED ", "ERROR "))}
    )
    tail = [
        line
        for line in out.splitlines()
        if " passed" in line or " failed" in line or " error" in line
    ][-1:]
    return {"rc": rc, "failed": failed, "summary": tail[0] if tail else out[-300:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default = "mlx_laya_decision_ab.json")
    ap.add_argument("--tiny", action = "store_true", help = "ignored (staging_ci)")
    ap.add_argument("--child", action = "store_true")
    ap.add_argument("--phases", action = "store_true")
    ap.add_argument("--arm")
    ap.add_argument("--compute")
    ap.add_argument("--folder")
    ap.add_argument("--sub", default = "")
    a = ap.parse_args()
    if a.child:
        return child(a)
    if a.phases:
        return phases_child(a)

    res = {
        "machine": platform.machine(),
        "mac": platform.mac_ver()[0],
        "python": sys.version.split()[0],
    }
    try:
        import psutil
        res["ram_gib"] = round(psutil.virtual_memory().total / 2**30, 1)
    except Exception:  # noqa: BLE001
        pass
    harness_error = None
    try:
        WORK.mkdir(parents = True, exist_ok = True)
        from huggingface_hub import snapshot_download

        root = Path(snapshot_download(REPO, local_dir = str(WORK / "laya")))
        zoos = {name: install_zoo(name, *spec) for name, spec in ZOO.items()}
        import mlx.core as mx

        res["mlx"] = mx.__version__
        refs = {}
        for sub in CHECKPOINTS:
            folder = root / sub if sub else root
            refs[sub] = reference(folder)
        res["phases"] = {}
        for sub in CHECKPOINTS:
            folder = root / sub if sub else root
            for mode in ("single", "split"):
                if not PHASES:
                    continue
                r = run_child("phases", mode, zoos["main"][0], None, folder, sub, timeout = 900)
                res["phases"][f"{sub or 'english'}/{mode}"] = r
                print("PHASES", sub or "english", mode, json.dumps(r)[:400], flush = True)
        res["arms"] = {}
        for zoo_name, compute in ARMS:
            for sub in CHECKPOINTS:
                folder = root / sub if sub else root
                r = run_child("child", zoo_name, zoos[zoo_name][0], compute, folder, sub)
                if r.get("ok"):
                    r["vs_fp32"] = compare(r.pop("answers"), refs[sub][0])
                    r["big_vs_fp32"] = {
                        k: compare([v], [refs[sub][1]]) for k, v in r.pop("big_answers").items()
                    }
                key = f"{zoo_name}/{compute or 'default'}/{sub or 'english'}"
                res["arms"][key] = r
                brief = {
                    k: r.get(k)
                    for k in (
                        "ok",
                        "error",
                        "phase",
                        "load_s",
                        "after_load",
                        "request_ms_median",
                        "vs_fp32",
                        "big_chunked",
                        "big_one_forward",
                        "big_vs_fp32",
                        "param_dtypes",
                        "agent_dtype",
                        "loader_kwargs",
                        "overflow",
                    )
                }
                print("ARM", key, json.dumps(brief, default = str)[:1200], flush = True)
        # The PRs' own tests: each revision's file on its own code, and each head's file on the base code.
        res["tests"] = {}
        for name in ZOO:
            target, src = zoos[name]
            res["tests"][f"{name}_own"] = pytest_ids(
                src, target, src / "tests" / "test_mlx_decision_metal.py"
            )
        for head in ("pr1488", "pr1496"):
            base_target, base_src = zoos["pr1488" if head == "pr1496" else "main"]
            res["tests"][f"{head}_tests_on_{'pr1488' if head == 'pr1496' else 'main'}_code"] = (
                pytest_ids(
                    base_src, base_target, zoos[head][1] / "tests" / "test_mlx_decision_metal.py"
                )
            )
        for k, v in res["tests"].items():
            print("TESTS", k, json.dumps(v)[:800], flush = True)
    except Exception as exc:  # noqa: BLE001
        harness_error = f"{type(exc).__name__}: {exc}"
        res["harness_error"] = harness_error
        res["traceback"] = traceback.format_exc()[-3000:]
        print(res["traceback"], flush = True)
    res["ok"] = harness_error is None
    with open(a.out, "w") as f:
        json.dump(res, f, indent = 1, default = str)
    print(
        "JOB_RESULT "
        + json.dumps(
            {
                "ok": res["ok"],
                "harness_error": harness_error,
                "arms_ok": {k: v.get("ok") for k, v in res.get("arms", {}).items()},
            }
        ),
        flush = True,
    )
    sys.exit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()
