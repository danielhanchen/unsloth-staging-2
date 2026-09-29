"""Studio MLX resident vision batch prompt reuse on Apple Silicon: unsloth #12263 + unsloth-zoo #1497 vs base.

    python jobs/mlx_vlm_batch_prefix.py [--out mlx_vlm_batch_prefix.json] [--models a,b]

Run on macos-15 (staging_ci --job mlx_vlm_batch_prefix). Studio base / head and zoo base / head are cloned and
installed into their own dirs, and every arm runs in a fresh process on this runner:
  base   = Studio merge base + zoo merge base, resident batch (width 2)
  head   = Studio #12263 + zoo #1497, resident batch
  mixed  = Studio #12263 + zoo merge base (must behave as base: no prompt_cache_state)
  single = Studio #12263 + zoo #1497, single reply path (parallel slots 1), the reuse reference
One scripted chat per arm, greedy: text turns, a regenerate, an image turn, its regenerate, a text turn.
Per turn: token ids + logprobs (batch arms), text, usage / timings (prompt, cached, prefill ms), wall ms,
MLX peak memory, snapshot store bytes. Also each PR's tests on its head and its test files on the base code.
Prints `JOB_RESULT {json}`; exits 1 only when the harness breaks (findings are data).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

STUDIO = {
    "base": ("https://github.com/unslothai/unsloth", "6c20f0fac752a1ec64537575873fa7f835d348dd"),
    "head": ("https://github.com/Lyxot/unsloth", "36ddd90c03a5a8798768a318a1b55008d9296b41"),
}
ZOO = {
    "base": (
        "https://github.com/unslothai/unsloth-zoo",
        "3b8bd774f0ca96f779f18a6b3fabb0a2f6255155",
    ),
    "head": ("https://github.com/Lyxot/unsloth-zoo", "cea850261eb96e722371bf74c204873ae635035d"),
}
ARMS = {
    "base": ("base", "base", "batch"),
    "head": ("head", "head", "batch"),
    "mixed": ("head", "base", "batch"),
    "single": ("head", "head", "single"),
}
MODELS = [
    "mlx-community/Qwen3.5-2B-4bit",
    "mlx-community/gemma-4-e2b-it-4bit",
    "mlx-community/FastVLM-0.5B-bf16",
]
WORK = Path(os.environ.get("RUNNER_TEMP", "/tmp")) / "vlm_prefix"
MAX_NEW = 24
COLD_REPEATS = 5

# Long enough that every prompt crosses mlx-vlm's 2048-token prefill step, so the grid has a boundary to bank.
SYSTEM = "You are a support assistant for a hardware store. Follow the policy below.\n" + "\n".join(
    f"Policy {i}: items in aisle {i % 17} ship within {i % 5 + 1} days; returns for category {i % 11} need "
    f"receipt number format R-{i:04d} and are refunded to the original card."
    for i in range(120)
)
Q1 = "A customer asks about returning a drill bought last week without a receipt. What do you tell them?"
A1 = "Without a receipt we cannot refund to the original card, but we can offer store credit after checking the item."
Q2 = "They now say they found the receipt, number R-0042. Which policy applies and how long will the refund take?"
A2 = "Policy 42 applies: category 9, refunded to the original card once the receipt is verified."
Q3 = "Here is a photo the customer sent of the item. Describe the colours you see in one sentence."
A3 = "The picture shows a red square next to a blue square on a white background."
Q4 = "Thanks. Summarise the whole case for the next agent in two sentences."


def _image():
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (336, 336), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle((24, 96, 152, 224), fill = "red")
    draw.rectangle((184, 96, 312, 224), fill = "blue")
    return img


def _script(image):
    """(label, messages, image) per turn; a regenerate resubmits the previous turn unchanged."""
    t1 = [{"role": "user", "content": Q1}]
    t2 = t1 + [{"role": "assistant", "content": A1}, {"role": "user", "content": Q2}]
    t4 = t2 + [{"role": "assistant", "content": A2}, {"role": "user", "content": Q3}]
    t6 = t4 + [{"role": "assistant", "content": A3}, {"role": "user", "content": Q4}]
    return [
        ("text", t1, None),
        ("text_followup", t2, None),
        ("text_regenerate", t2, None),
        ("image", t4, image),
        ("image_regenerate", t4, image),
        ("text_after_image", t6, None),
    ]


# ------------------------------------------------------------------ child: one arm x one model
def child(args):
    import mlx.core as mx
    import psutil

    sys.path.insert(0, args.studio)
    res = {"arm": args.arm, "model": args.model, "mode": args.mode, "turns": []}
    try:
        from types import SimpleNamespace

        from core.inference import mlx_inference
        from core.inference.worker import RowRefused
        import unsloth_zoo.mlx.generate as engine

        res["zoo_file"] = engine.__file__
        res["studio_file"] = mlx_inference.__file__
        res["row_prompt_cache_gap"] = getattr(
            mlx_inference, "_row_prompt_cache_gap", lambda: "absent"
        )()
        backend = mlx_inference.MLXInferenceBackend()
        t = time.perf_counter()
        backend.load_model(
            SimpleNamespace(identifier = args.model, is_vision = True), max_seq_length = 8192
        )
        res["load_s"] = round(time.perf_counter() - t, 1)
        res["is_vlm"] = backend._is_vlm
        res["store_enabled"] = backend._vlm_prompt_cache_store() is not None
        res["resident_gap"] = backend.resident_unavailable_reason({})
        sampling = dict(
            temperature = 0.0, top_p = 1.0, top_k = 0, min_p = 0.0, max_new_tokens = MAX_NEW, seed = 0
        )
        session = None
        if args.mode == "batch":
            session = backend.open_resident_batch(width = 2)
            results = {}
            real_step = session.stream.step

            def step():
                for event in real_step():
                    if event.result is not None:
                        results[event.index] = event.result
                    yield event

            session.stream.step = step
        image = _image()
        for n, (label, messages, img) in enumerate(_script(image)):
            turn = {"label": label}
            mx.reset_peak_memory()
            t = time.perf_counter()
            served = args.mode
            if args.mode == "batch":
                handle = f"t{n}"
                request = {"messages": messages, "system_prompt": SYSTEM, "image": img, **sampling}
                try:
                    session.admit(request, handle)
                    row = session._rows[handle].row
                    text = []
                    while session.rows_in_flight:
                        for h, snap in session.step():
                            if h == handle and snap is not None:
                                text.append(snap)
                    stats = session.take_stats(handle)
                    got = results.get(row)
                    turn["token_ids"] = list(got.token_ids) if got else None
                    turn["logprobs"] = [float(x) for x in got.logprobs] if got else None
                    turn["text"] = got.text if got else None
                except RowRefused as refusal:
                    # As Studio's worker does: the reply is served outside the batch.
                    turn["refused"] = str(refusal)[:300]
                    served = "single"
            if served == "single":
                text = "".join(
                    backend.generate_chat_response(
                        messages,
                        system_prompt = SYSTEM,
                        image = img,
                        **{k: v for k, v in sampling.items()},
                    )
                )
                stats = backend.last_generation_stats
                turn.setdefault("text", text)
            turn["served"] = served
            turn["wall_ms"] = round((time.perf_counter() - t) * 1e3, 1)
            turn["peak_mib"] = round(mx.get_peak_memory() / 2**20)
            store = getattr(backend, "_vlm_snapshot_store", None)
            turn["store_bytes"] = getattr(store, "nbytes", None)
            if stats:
                usage, timings = stats.get("usage", {}), stats.get("timings", {})
                turn["prompt_tokens"] = usage.get("prompt_tokens")
                turn["cached_tokens"] = (usage.get("prompt_tokens_details") or {}).get(
                    "cached_tokens"
                )
                turn["prompt_n"] = timings.get("prompt_n")
                turn["prompt_ms"] = round(timings.get("prompt_ms") or 0.0, 1)
                turn["predicted_per_second"] = round(timings.get("predicted_per_second") or 0.0, 1)
            res["turns"].append(turn)
            print(
                "TURN",
                args.arm,
                args.model.split("/")[-1],
                json.dumps(
                    {k: v for k, v in turn.items() if k not in ("token_ids", "logprobs", "text")}
                ),
                flush = True,
            )
        # Cold image prefill, repeated: a fresh image each time, so nothing can be reused on any arm.
        cold = []
        for k in range(COLD_REPEATS):
            from PIL import ImageOps

            img = ImageOps.mirror(image) if k % 2 else image.rotate(90 * (k + 1))
            messages = [{"role": "user", "content": f"{Q3} ({k})"}]
            request = {"messages": messages, "system_prompt": SYSTEM, "image": img, **sampling}
            t = time.perf_counter()
            if args.mode == "batch":
                handle = f"cold{k}"
                try:
                    session.admit(request, handle)
                    while session.rows_in_flight:
                        list(session.step())
                    stats = session.take_stats(handle)
                except RowRefused:
                    list(
                        backend.generate_chat_response(
                            messages, system_prompt = SYSTEM, image = img, **sampling
                        )
                    )
                    stats = backend.last_generation_stats
            else:
                list(
                    backend.generate_chat_response(
                        messages, system_prompt = SYSTEM, image = img, **sampling
                    )
                )
                stats = backend.last_generation_stats
            cold.append(
                {
                    "wall_ms": round((time.perf_counter() - t) * 1e3, 1),
                    "prompt_ms": round(
                        ((stats or {}).get("timings") or {}).get("prompt_ms") or 0.0, 1
                    ),
                }
            )
        res["cold_image"] = cold
        print("COLD", args.arm, args.model.split("/")[-1], json.dumps(cold), flush = True)
        if session is not None:
            session.close()
        res["rss_mib"] = round(psutil.Process().memory_info().rss / 2**20)
        res["ok"] = True
    except Exception as exc:  # noqa: BLE001 - the error is the result
        res["ok"] = False
        res["error"] = f"{type(exc).__name__}: {exc}"[:800]
        res["traceback"] = traceback.format_exc()[-3000:]
    Path(args.out).write_text(json.dumps(res, default = str), encoding = "utf-8")


# ------------------------------------------------------------------ parent
def sh(cmd, **kw):
    p = subprocess.run(cmd, capture_output = True, text = True, **kw)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def checkout(
    name,
    url,
    sha,
    sparse = None,
):
    src = WORK / name
    if not (src / ".git").exists():
        sh(["git", "clone", "-q", "--filter=blob:none", "--no-checkout", url, str(src)])
        if sparse:
            sh(["git", "-C", str(src), "sparse-checkout", "set", *sparse])
    rc, out = sh(["git", "-C", str(src), "checkout", "-q", sha])
    if rc:
        sh(["git", "-C", str(src), "fetch", "-q", "origin", sha])
        rc, out = sh(["git", "-C", str(src), "checkout", "-q", sha])
    if rc:
        raise RuntimeError(f"checkout {name} {sha}: {out[-400:]}")
    return src


def install_zoo(name, url, sha):
    src = checkout(f"zoo_src_{name}", url, sha)
    target = WORK / f"zoo_{name}"
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
        raise RuntimeError(f"install zoo {name}: {out[-800:]}")
    return target, src


def run_child(
    arm,
    model,
    studio,
    zoo_dir,
    mode,
    timeout = 2400,
):
    out = WORK / f"arm_{arm}_{model.split('/')[-1]}.json"
    env = {**os.environ, "PYTHONPATH": str(zoo_dir)}
    cmd = [
        sys.executable,
        __file__,
        "--child",
        "--arm",
        arm,
        "--model",
        model,
        "--studio",
        str(studio),
        "--mode",
        mode,
        "--out",
        str(out),
    ]
    try:
        rc, log = sh(cmd, env = env, timeout = timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout"}
    if out.exists():
        return json.loads(out.read_text(encoding = "utf-8"))
    return {"ok": False, "error": f"child rc={rc}", "log": log[-2000:]}


def pytest_ids(
    cwd,
    pythonpath,
    files,
    extra = (),
):
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(str(p) for p in pythonpath)}
    rc, out = sh(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-rfEs",
            "-p",
            "no:cacheprovider",
            *map(str, files),
            *extra,
        ],
        env = env,
        cwd = str(cwd),
        timeout = 3600,
    )
    failed = sorted(
        {line.split()[1] for line in out.splitlines() if line.startswith(("FAILED ", "ERROR "))}
    )
    messages = [line[:400] for line in out.splitlines() if line.startswith(("FAILED ", "ERROR "))]
    tail = [
        line
        for line in out.splitlines()
        if " passed" in line or " failed" in line or " error" in line
    ][-1:]
    return {
        "rc": rc,
        "failed": failed,
        "messages": messages,
        "summary": tail[0] if tail else out[-400:],
        "log_tail": out[-6000:] if rc else "",
    }


def planted(base_dir, head_file, tag):
    """A head test file copied into the base checkout, so pytest's rootdir (and its pythonpath) is the base."""
    copy = (
        Path(base_dir)
        / Path(head_file).parent.relative_to(Path(head_file).parents[1])
        / f"test_head_{tag}.py"
    )
    copy.write_text(Path(head_file).read_text(encoding = "utf-8"), encoding = "utf-8")
    return copy


def verdicts(arms):
    """Per model: head / mixed vs base bitwise, head reuse vs the single path's."""
    out = {}
    for model in {k.split("|")[1] for k in arms}:
        get = lambda arm: arms.get(f"{arm}|{model}") or {}
        base, head, mixed, single = get("base"), get("head"), get("mixed"), get("single")
        row = {}
        for name, other in (("head_vs_base", head), ("mixed_vs_base", mixed)):
            if not (base.get("ok") and other.get("ok")):
                row[name] = "VOID"
                continue
            diffs = []
            for b, o in zip(base["turns"], other["turns"]):
                if b.get("token_ids") is None or o.get("token_ids") is None:
                    diffs.append(
                        {
                            "turn": b["label"],
                            "compare": "text",
                            "equal": b.get("text") == o.get("text"),
                        }
                    )
                    continue
                same_ids = b["token_ids"] == o["token_ids"]
                lp = max((abs(x - y) for x, y in zip(b["logprobs"], o["logprobs"])), default = 0.0)
                diffs.append({"turn": b["label"], "ids_equal": same_ids, "max_logprob_diff": lp})
            row[name] = diffs
        if head.get("ok") and single.get("ok"):
            row["cached_head_vs_single"] = [
                (h["label"], h.get("cached_tokens"), s.get("cached_tokens"))
                for h, s in zip(head["turns"], single["turns"])
            ]
            row["text_head_vs_single"] = [
                h.get("text") == s.get("text") for h, s in zip(head["turns"], single["turns"])
            ]
        for arm, r in (("base", base), ("head", head), ("mixed", mixed), ("single", single)):
            if r.get("ok"):
                row[f"{arm}_prompt_ms"] = [t.get("prompt_ms") for t in r["turns"]]
                row[f"{arm}_cached"] = [t.get("cached_tokens") for t in r["turns"]]
                row[f"{arm}_peak_mib"] = max(t.get("peak_mib") or 0 for t in r["turns"])
                cold = sorted(c["prompt_ms"] for c in r.get("cold_image") or [])
                row[f"{arm}_cold_image_prompt_ms_median"] = cold[len(cold) // 2] if cold else None
        out[model] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default = "mlx_vlm_batch_prefix.json")
    ap.add_argument("--tiny", action = "store_true", help = "ignored (staging_ci)")
    ap.add_argument("--models", default = os.environ.get("VLM_PREFIX_MODELS") or ",".join(MODELS))
    ap.add_argument("--child", action = "store_true")
    ap.add_argument("--arm")
    ap.add_argument("--model")
    ap.add_argument("--studio")
    ap.add_argument("--mode")
    ap.add_argument(
        "--tests-only", action = "store_true", help = "skip the chat arms, run only the PR tests"
    )
    ap.add_argument("--mlx-vlm", help = "pip spec installed first, e.g. ==0.7.1 (zoo's pin ceiling)")
    a = ap.parse_args()
    if a.child:
        return child(a)
    if a.mlx_vlm:
        rc, out = sh([sys.executable, "-m", "pip", "install", "-q", f"mlx-vlm{a.mlx_vlm}"])
        print("MLX_VLM_INSTALL", a.mlx_vlm, rc, out[-600:], flush = True)
    global WORK
    WORK = WORK.with_name(f"vlm_prefix_{(a.mlx_vlm or 'as_installed').strip('=<>')}")
    a.out = a.out.replace(".json", f"_{(a.mlx_vlm or 'as_installed').strip('=<>')}.json")

    res = {
        "machine": platform.machine(),
        "mac": platform.mac_ver()[0],
        "python": sys.version.split()[0],
    }
    harness_error = None
    try:
        import importlib.metadata as md

        res["versions"] = {p: md.version(p) for p in ("mlx", "mlx-lm", "mlx-vlm", "transformers")}
        WORK.mkdir(parents = True, exist_ok = True)
        studios = {
            n: checkout(f"studio_{n}", *spec, sparse = ["studio/backend"])
            for n, spec in STUDIO.items()
        }
        zoos = {n: install_zoo(n, *spec) for n, spec in ZOO.items()}
        res["arms"] = {}
        for model in [] if a.tests_only else a.models.split(","):
            for arm, (studio, zoo, mode) in ARMS.items():
                r = run_child(
                    arm, model, studios[studio] / "studio" / "backend", zoos[zoo][0], mode
                )
                res["arms"][f"{arm}|{model}"] = r
                brief = {
                    k: r.get(k)
                    for k in (
                        "ok",
                        "error",
                        "load_s",
                        "store_enabled",
                        "resident_gap",
                        "row_prompt_cache_gap",
                        "rss_mib",
                    )
                }
                print("ARM", arm, model, json.dumps(brief, default = str)[:1500], flush = True)
                if not r.get("ok"):
                    print(r.get("traceback") or r.get("log"), flush = True)
        res["verdicts"] = verdicts(res["arms"])
        for model, v in res["verdicts"].items():
            print("VERDICT", model, json.dumps(v, default = str)[:4000], flush = True)

        # The PRs' own tests: head on head, head test files on the base code (planted in the base checkout).
        zb, zh = zoos["base"][1], zoos["head"][1]
        zoo_tests = ["tests/test_mlx_generate.py", "tests/test_mlx_generate_metal.py"]
        res["tests"] = {"zoo_head": pytest_ids(zh, [zoos["head"][0]], zoo_tests)}
        res["tests"]["zoo_head_tests_on_base"] = pytest_ids(
            zb, [zoos["base"][0]], [planted(zb, zh / f, Path(f).stem) for f in zoo_tests]
        )
        sb, sh_ = studios["base"] / "studio" / "backend", studios["head"] / "studio" / "backend"
        studio_tests = ["tests/test_mlx_vlm_prompt_cache.py", "tests/test_mlx_inference_backend.py"]
        res["tests"]["studio_head"] = pytest_ids(sh_, [sh_, zoos["head"][0]], studio_tests)
        res["tests"]["studio_base"] = pytest_ids(sb, [sb, zoos["base"][0]], studio_tests)
        res["tests"]["studio_head_tests_on_base"] = pytest_ids(
            sb, [sb, zoos["base"][0]], [planted(sb, sh_ / f, Path(f).stem) for f in studio_tests]
        )
        for k, v in res["tests"].items():
            print(
                "TESTS",
                k,
                json.dumps({x: y for x, y in v.items() if x != "log_tail"})[:3000],
                flush = True,
            )
            if v.get("log_tail"):
                print(v["log_tail"], flush = True)
    except Exception as exc:  # noqa: BLE001
        harness_error = f"{type(exc).__name__}: {exc}"
        res["harness_error"] = harness_error
        res["traceback"] = traceback.format_exc()[-3000:]
        print(res["traceback"], flush = True)
    res["ok"] = harness_error is None
    with open(a.out, "w") as f:
        json.dump(res, f, default = str)
    print(
        "JOB_RESULT",
        json.dumps(
            {
                "ok": res["ok"],
                "harness_error": harness_error,
                "arms_ok": {k: v.get("ok") for k, v in res.get("arms", {}).items()},
            }
        ),
        flush = True,
    )
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
