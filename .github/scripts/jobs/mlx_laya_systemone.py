"""Decision API (laya) on Apple Silicon with the real laya-multilingual checkpoint.

    python jobs/mlx_laya_systemone.py [--out mlx_laya_systemone.json]

Run from an unsloth checkout (staging_ci --job mlx_laya_systemone on macos-15). Strict: exits 1 on
any failed check. Checks, all through Studio's own laya_runtime:
  1. Device selection: preferred "gpu" on Apple Silicon resolves to "mlx"; "cpu" stays "cpu".
  2. Studio's fast load (skip_init embedding) builds the same tensors as stock laya.load.
     CPU path on arm64 keeps fp32 weights + fp32 compute (the float16 policy is x86/CUDA only), and
     its answers are bit-identical to stock laya Agent.predict on CPU.
  3. MLX path (unsloth_zoo.mlx.decision) loads and matches the fp32 CPU reference within 5e-3, the
     tolerance of test_mlx_answers_match_laya_on_cpu.
  4. decide() end to end on MLX and on CPU.
  5. MLX out-of-memory fallback: a real reload on CPU through _place, still fp32, answers intact.
Prints `JOB_RESULT {json}`.
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
TOL = 5e-3


def _max_diff(a, b):
    worst = 0.0
    for name, x in a["answers"].items():
        y = b["answers"][name]
        if x["type"] == "noul":
            worst = max(worst, abs(x["noul"] - y["noul"]))
        else:
            worst = max(
                worst,
                max(abs(x["probabilities"][k] - y["probabilities"][k]) for k in x["probabilities"]),
            )
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default = "mlx_laya_systemone.json")
    ap.add_argument("--tiny", action = "store_true", help = "ignored (staging_ci)")
    a = ap.parse_args()
    res = {"machine": platform.machine(), "python": sys.version.split()[0], "checks": {}}
    failures = []

    def check(
        name,
        ok,
        detail = None,
    ):
        res["checks"][name] = {"ok": bool(ok), "detail": detail}
        print(
            ("PASS " if ok else "FAIL ") + name + (f": {detail}" if detail is not None else ""),
            flush = True,
        )
        if not ok:
            failures.append(name)

    try:
        for mod in ("structlog", "rich"):
            try:
                __import__(mod)
            except ImportError:
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", mod], check = True)
        backend = os.path.join(os.getcwd(), "studio", "backend")
        sys.path.insert(0, backend)
        import torch
        import mlx.core as mx
        from huggingface_hub import snapshot_download

        res["versions"] = {"torch": torch.__version__, "mlx": mx.__version__}
        import transformers

        res["versions"]["transformers"] = transformers.__version__
        check("apple_silicon", platform.machine() == "arm64", platform.machine())

        from core.systemone import catalog, laya_runtime
        from utils import systemone_settings

        laya = laya_runtime._laya()
        check(
            "vendored_laya",
            os.path.dirname(laya.__file__) == str(laya_runtime._VENDORED_LAYA),
            laya.__file__,
        )

        # 1. Device selection, through the real hardware probe.
        real_pref = systemone_settings.get_device
        systemone_settings.get_device = lambda: "gpu"
        picked_gpu = laya_runtime._device()
        systemone_settings.get_device = lambda: "cpu"
        picked_cpu = laya_runtime._device()
        systemone_settings.get_device = real_pref
        check("device_gpu_is_mlx", picked_gpu == "mlx", picked_gpu)
        check("device_cpu_is_cpu", picked_cpu == "cpu", picked_cpu)
        check(
            "cpu_precision_is_fp32",
            laya_runtime._precision(torch.device("cpu"), True) == (None, None),
            str(laya_runtime._precision(torch.device("cpu"), True)),
        )

        t0 = time.time()
        root = snapshot_download(
            "convaiinnovations/laya",
            allow_patterns = [
                f"multilingual/{p}"
                for p in ("rl_agent_config.json", "model.safetensors", "encoder/*", "tokenizer/*")
            ],
        )
        res["download_s"] = round(time.time() - t0, 1)
        os.environ["UNSLOTH_SYSTEMONE_MODEL"] = root
        os.environ["UNSLOTH_SYSTEMONE_SUBFOLDER"] = "multilingual"
        checkpoint = catalog.default_checkpoint()

        # Stock laya on CPU: the fp32 reference every path is judged against.
        reference = laya.load(root, subfolder = "multilingual", device = "cpu")
        expected = [laya_runtime._predict(reference, s, QUESTIONS)[0] for s in STATES]

        # 2. CPU through Studio's loader.
        laya_runtime._device = lambda: "cpu"
        agent, dev = laya_runtime._load_checkpoint(checkpoint)
        dtypes = sorted({str(p.dtype) for p in agent.model.parameters()})
        check("cpu_loader_device", dev == "cpu", dev)
        check(
            "cpu_weights_fp32",
            dtypes == ["torch.float32"] and agent.dtype == torch.float32,
            f"{dtypes} {agent.dtype}",
        )
        got = [laya_runtime._predict(agent, s, QUESTIONS)[0] for s in STATES]
        check(
            "cpu_bit_identical_to_stock_laya",
            got == expected,
            max(_max_diff(g, e) for g, e in zip(got, expected)),
        )
        # Fast load (skip_init embedding) vs stock laya.load: every parameter and buffer, incl. rotary inv_freq.
        ours = dict(agent.model.named_parameters()) | dict(agent.model.named_buffers())
        theirs = dict(reference.model.named_parameters()) | dict(reference.model.named_buffers())
        differ = [n for n in theirs if n not in ours or not torch.equal(ours[n], theirs[n])]
        check(
            "fast_load_tensors_identical_to_stock",
            ours.keys() == theirs.keys() and not differ,
            f"{len(theirs)} tensors, {len(differ)} differ",
        )

        # 3. MLX through Studio's loader.
        laya_runtime._evict()
        laya_runtime._device = lambda: "mlx"
        mlx_agent, dev = laya_runtime._load_checkpoint(checkpoint)
        check(
            "mlx_loader_device", dev == "mlx" and isinstance(mlx_agent, laya_runtime._MLXAgent), dev
        )
        got = [laya_runtime._predict(mlx_agent, s, QUESTIONS)[0] for s in STATES]
        diff = max(_max_diff(g, e) for g, e in zip(got, expected))
        usage_ok = all(g["usage"] == e["usage"] for g, e in zip(got, expected))
        check(
            "mlx_matches_fp32_cpu",
            diff <= TOL and usage_ok,
            f"max prob diff {diff:.2e}, usage equal {usage_ok}",
        )

        # 4. decide() end to end, MLX then CPU.
        for device in ("mlx", "cpu"):
            laya_runtime.unload()
            laya_runtime._device = lambda device = device: device
            laya_runtime.LOAD_WAIT_S = 600.0
            out = laya_runtime.decide(checkpoint, STATES[0], QUESTIONS)
            status = laya_runtime.status()
            ok = (
                set(out["answers"]) == set(QUESTIONS)
                and status["device"] == device
                and status["error"] is None
            )
            check(
                f"decide_{device}", ok, {"team": out["answers"]["team"]["choice"], "status": status}
            )

        # 5. MLX out of memory -> real CPU reload through _place.
        laya_runtime.unload()
        laya_runtime._device = lambda: "mlx"
        laya_runtime.decide(checkpoint, STATES[1], QUESTIONS)
        live = laya_runtime._agent
        real_logits = live.model.logits

        def oom(batch):
            raise RuntimeError("[malloc] Unable to allocate 2147483648 bytes")

        live.model.logits = oom
        out = laya_runtime.decide(checkpoint, STATES[1], QUESTIONS)
        moved = laya_runtime._agent
        dtypes = sorted({str(p.dtype) for p in moved.model.parameters()})
        diff = _max_diff(laya_runtime._predict(moved, STATES[1], QUESTIONS)[0], expected[1])
        check(
            "mlx_oom_falls_back_to_cpu_fp32",
            laya_runtime.status()["device"] == "cpu"
            and dtypes == ["torch.float32"]
            and diff == 0.0,
            f"device {laya_runtime.status()['device']} dtypes {dtypes} diff {diff}",
        )
        del real_logits
    except Exception as exc:  # noqa: BLE001 - recorded, then the job fails
        res["error"] = f"{type(exc).__name__}: {exc}"
        res["traceback"] = traceback.format_exc()[-3000:]
        print(res["traceback"], flush = True)
        failures.append("exception")
    res["failures"] = failures
    res["ok"] = not failures
    with open(a.out, "w") as f:
        json.dump(res, f, indent = 1, default = str)
    print(
        "JOB_RESULT "
        + json.dumps(
            {
                "ok": res["ok"],
                "failures": failures,
                "checks": {k: v["ok"] for k, v in res["checks"].items()},
            }
        ),
        flush = True,
    )
    sys.exit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()
