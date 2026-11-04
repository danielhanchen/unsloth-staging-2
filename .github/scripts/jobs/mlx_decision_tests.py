"""Runs Unsloth's MLX decision-model tests on Apple Silicon and fails if any of them skips.

    python jobs/mlx_decision_tests.py [--tests tests/test_decision_model_mlx.py] [-k EXPR]

A staging MLX leg with torch installed can leave unsloth._IS_MLX false, which silently skips every
test in the file; this job turns that into a failure. Exit 3 off Apple Silicon.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys


def main(argv = None):
    p = argparse.ArgumentParser()
    p.add_argument("--tests", default = "tests/test_decision_model_mlx.py")
    p.add_argument("-k", default = None)
    p.add_argument("--out", default = None)
    a = p.parse_args(argv)
    if sys.platform != "darwin" or platform.machine() != "arm64":
        print("JOB_RESULT " + json.dumps({"job": "mlx_decision_tests", "skipped": True}))
        return 3
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        a.tests,
        "-rA",
        "-p",
        "no:cacheprovider",
        "-o",
        "addopts=",
    ]
    if a.k:
        cmd += ["-k", a.k]
    # The staging test env forces the CUDA path, which turns the MLX backend off.
    env = {k: v for k, v in os.environ.items() if k != "UNSLOTH_FORCE_GPU_PATH"}
    probe = subprocess.run(
        [sys.executable, "-c", "import unsloth; print('IS_MLX', unsloth._IS_MLX)"],
        capture_output = True,
        text = True,
        env = env,
    )
    print(probe.stdout[-500:], probe.stderr[-2000:], flush = True)
    proc = subprocess.run(cmd, capture_output = True, text = True, env = env)
    out = proc.stdout + proc.stderr
    print(out, flush = True)
    passed = re.findall(r"^PASSED (\S+)", out, re.M)
    skipped = re.findall(r"^SKIPPED .*", out, re.M)
    failed = re.findall(r"^(?:FAILED|ERROR) (\S+)", out, re.M)
    ok = proc.returncode == 0 and passed and not skipped and not failed
    result = {
        "job": "mlx_decision_tests",
        "passed": len(passed),
        "skipped": skipped,
        "failed": failed,
        "ok": bool(ok),
        "is_mlx_probe": probe.stdout.strip()[-80:],
    }
    if a.out:
        with open(a.out, "w", encoding = "utf-8") as f:
            json.dump(result, f)
    print("JOB_RESULT " + json.dumps(result))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
