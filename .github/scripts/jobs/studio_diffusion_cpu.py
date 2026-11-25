"""Studio diffusion on a CPU runner: install Studio's backend requirements, run the diffusion test files, then render
tiny pipelines end to end through Studio's own DiffusionBackend (begin_load / generate) on the CPU.

Run from a Studio checkout root (the staging branch):
    python jobs/studio_diffusion_cpu.py --tiny --out job.json
    python jobs/studio_diffusion_cpu.py --tiny --skip-install --tests none --out job.json   # render only

Writes one JSON: platform, install rc + seconds, pytest counts + failing / erroring test ids, and per tiny model the
load seconds, render seconds, steps, size and the sha256 of the rendered image's RGB bytes. Exit 1 when the install
or any render failed (test failures are recorded, not fatal: the counts are the deliverable).
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback

TINY = (
    ("hf-internal-testing/tiny-flux-pipe", "flux.1"),
    ("hf-internal-testing/tiny-qwenimage-pipe", "qwen-image"),
)


def _sh(
    cmd,
    timeout = None,
    env = None,
):
    t = time.time()
    try:
        r = subprocess.run(cmd, capture_output = True, text = True, timeout = timeout, env = env)
        return (
            r.returncode,
            round(time.time() - t, 1),
            (r.stdout or "")[-4000:] + (r.stderr or "")[-4000:],
        )
    except subprocess.TimeoutExpired as exc:
        return "timeout", round(time.time() - t, 1), str(exc)[-2000:]


def install(rec):
    py = sys.executable
    req = "studio/backend/requirements"
    steps = [
        [
            py,
            "-m",
            "pip",
            "install",
            "-q",
            "-r",
            f"{req}/studio.txt",
            "psutil",
            "pytest",
            "pytest-asyncio",
            "pytest-timeout",
            "pytest-xdist",
        ],
        [py, "-m", "pip", "install", "-q", "-r", f"{req}/diffusers-pin.txt"],
    ]
    if sys.platform == "win32" and platform.machine().upper() == "ARM64":
        # PyPI has no win_arm64 pyarrow (datasets needs it): Studio's install.ps1 takes it from the Windows-ARM64
        # release wheelhouse, so the job does the same.
        tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
        steps.insert(
            0,
            [
                py,
                "-m",
                "pip",
                "install",
                "-q",
                f"https://github.com/unslothai/unsloth/releases/download/Windows-ARM64/pyarrow-25.0.1-{tag}-{tag}-win_arm64.whl",
            ],
        )
    steps.append(
        [py, "-m", "pip", "install", "-q", "transformers", "accelerate", "safetensors", "pillow"]
    )
    out = []
    for cmd in steps:
        rc, s, log = _sh(cmd, timeout = 1500)
        out.append(
            {"cmd": " ".join(cmd[3:]), "rc": rc, "s": s, "log_tail": log[-1500:] if rc else ""}
        )
        print("INSTALL", rc, s, " ".join(cmd[3:])[:200], flush = True)
        if rc:
            print(log[-3000:], flush = True)
    rec["install"] = out
    vers = _sh(
        [
            py,
            "-c",
            "import torch, diffusers, transformers, platform; print(torch.__version__, diffusers.__version__, "
            "transformers.__version__, platform.machine())",
        ]
    )
    rec["versions"] = vers[2].strip()[-300:]
    return all(o["rc"] == 0 for o in out)


def run_tests(rec, pattern, wall_s, workers):
    """pytest -v streamed to a file and parsed per line, so a wall-capped run still yields every finished test."""
    import re

    files = sorted(glob.glob(pattern))
    rec["tests"] = {"pattern": pattern, "files": len(files)}
    if not files:
        return
    with open("cov_test_files.txt", "w") as f:
        f.write("\n".join(files))
    env = dict(os.environ, PYTHONPATH = "studio/backend")
    env.pop("UNSLOTH_IS_PRESENT", None)
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "@cov_test_files.txt",
        "-v",
        "-p",
        "no:cacheprovider",
        "--timeout",
        "120",
        "-o",
        "console_output_style=classic",
    ]
    if workers:
        cmd += ["-n", str(workers)]
    t = time.time()
    with open("cov_pytest.log", "w") as lf:
        p = subprocess.Popen(cmd, stdout = lf, stderr = subprocess.STDOUT, env = env)
        try:
            rc = p.wait(timeout = wall_s)
        except subprocess.TimeoutExpired:
            p.kill()
            rc = "wall_timeout"
    rec["tests"].update(
        {"rc": rc, "s": round(time.time() - t, 1), "complete": rc != "wall_timeout"}
    )
    pat = re.compile(
        r"(?:^\[gw\d+\]\s+(?:\[\s*\d+%\]\s+)?(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\s+(\S+::\S+))"
        r"|(?:^(\S+::\S+)\s+(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\b)"
    )
    res = {}
    lines = open("cov_pytest.log", errors = "replace").read().splitlines()
    for ln in lines:
        m = pat.match(ln.strip())
        if m:
            st, tid = (m.group(1), m.group(2)) if m.group(1) else (m.group(4), m.group(3))
            if st == "ERROR" or tid not in res or res[tid] == "PASSED":
                res[tid] = st
    counts = {}
    for st in res.values():
        counts[st] = counts.get(st, 0) + 1
    summary = [ln for ln in lines if ln.startswith(("FAILED ", "ERROR "))]
    rec["tests"].update(
        counts = counts,
        failed_ids = sorted(k for k, v in res.items() if v == "FAILED")[:500],
        error_ids = sorted(k for k, v in res.items() if v == "ERROR")[:500],
        summary_lines = [x[:300] for x in summary[:300]],
        tail = "\n".join(lines[-15:])[-2500:],
    )
    print("PYTEST", rc, rec["tests"]["s"], json.dumps(counts), flush = True)
    print("\n".join(lines[-15:])[-2500:], flush = True)


def render(rec, width, height, steps):
    sys.path.insert(0, os.path.abspath("studio/backend"))
    os.environ.pop("UNSLOTH_IS_PRESENT", None)
    import torch

    from core.inference.diffusion import DiffusionBackend

    rec["renders"] = []
    ok = True
    backend = DiffusionBackend()
    for repo, fam in TINY:
        row = {"repo": repo, "family": fam, "width": width, "height": height, "steps": steps}
        rec["renders"].append(row)
        t0 = time.time()
        try:
            from huggingface_hub import snapshot_download

            # Studio loads a full non-unsloth pipeline only from a local path: fetch the tiny repo into the checkout.
            local = snapshot_download(
                repo, local_dir = os.path.abspath(os.path.join("cov_tiny", repo.split("/")[-1]))
            )
            row["fetch_s"] = round(time.time() - t0, 2)
            t0 = time.time()
            backend.begin_load(local, family_override = fam)
            while True:
                p = backend.load_progress() or {}
                if p.get("error"):
                    raise RuntimeError(f"load error: {p['error']}")
                if p.get("phase") == "ready":
                    break
                if time.time() - t0 > 900:
                    raise RuntimeError("load timeout 900 s")
                time.sleep(0.2)
            row["load_s"] = round(time.time() - t0, 2)
            st = backend.status()
            row["device"] = st.get("device")
            row["dtype"] = str(st.get("dtype"))
            for i, seed in enumerate((101, 101)):
                t1 = time.perf_counter()
                res = backend.generate(
                    prompt = "a red fox sitting in fresh snow",
                    width = width,
                    height = height,
                    steps = steps,
                    guidance = 0.0,
                    seed = seed,
                    batch_size = 1,
                    live_preview = False,
                )
                dt = time.perf_counter() - t1
                img = (res.get("images") or [None])[0]
                sha = (
                    hashlib.sha256(img.convert("RGB").tobytes()).hexdigest()
                    if img is not None
                    else None
                )
                row[f"render{i + 1}_s"] = round(dt, 3)
                row[f"sha{i + 1}"] = sha
                if img is not None and i == 0:
                    img.save(f"cov_render_{fam}.png")
            row["repeat_equal"] = row.get("sha1") == row.get("sha2")
            print("RENDER", json.dumps(row), flush = True)
        except Exception as exc:  # noqa: BLE001
            ok = False
            row["error"] = f"{type(exc).__name__}: {exc}"[:3000]
            row["traceback"] = traceback.format_exc()[-4000:]
            print("RENDER_FAIL", repo, row["error"][:1500], flush = True)
            print(row["traceback"], flush = True)
        try:
            backend.unload()
        except Exception:  # noqa: BLE001
            pass
    rec["torch_threads"] = torch.get_num_threads()
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tiny", action = "store_true", help = "accepted for the staging --job contract (always tiny)"
    )
    ap.add_argument("--out", required = True)
    ap.add_argument("--skip-install", action = "store_true")
    ap.add_argument(
        "--tests", default = "studio/backend/tests/test_diffusion*.py", help = "glob, or 'none'"
    )
    ap.add_argument("--tests-wall-s", type = int, default = 1800)
    ap.add_argument("--workers", type = int, default = 0, help = "pytest-xdist workers (0 = serial)")
    ap.add_argument("--width", type = int, default = 256)
    ap.add_argument("--height", type = int, default = 256)
    ap.add_argument("--steps", type = int, default = 4)
    a = ap.parse_args()
    rec = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "cpus": os.cpu_count(),
    }
    os.environ.setdefault("UNSLOTH_STUDIO_HOME", os.path.abspath("cov_studio_home"))

    def dump():
        with open(a.out, "w") as f:
            json.dump(rec, f, indent = 1, default = str)

    ok = True
    if not a.skip_install:
        ok = install(rec)
        dump()
    rok = render(rec, a.width, a.height, a.steps) if ok else False
    dump()
    if ok and a.tests != "none":
        run_tests(rec, a.tests, a.tests_wall_s, a.workers)
        dump()
    rec["ok"] = bool(ok and rok)
    dump()
    print("JOB_DONE", json.dumps({k: rec.get(k) for k in ("machine", "ok")}), flush = True)
    return 0 if rec["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
