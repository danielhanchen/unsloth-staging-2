"""End-to-end `unsloth train` on Apple Silicon through the Studio MLX worker, single-process and
under `mlx.launch -n 2` (unsloth#7166). Run from the unsloth checkout root.

    python jobs/mlx_cli_train.py                 # gemma-3-270m-it fp16, 3 steps per arm
    python jobs/mlx_cli_train.py --no-ddp        # single-process arm only

Checks per arm: CLI exit 0, adapter weights saved in --output-dir. DDP arm also: both ranks
reach the end (mlx.launch exits 0 only if every rank does), rank 0 alone writes the output dir.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

DEFAULT_MODEL = "unsloth/gemma-3-270m-it"
# Same pins mlx-ci.yml installs beside studio.txt for the CLI / worker imports.
CLI_DEPS = [
    "typer==0.25.1",
    "pyyaml==6.0.3",
    "jinja2==3.1.6",
    "requests==2.33.1",
    "python-multipart==0.0.27",
    "aiofiles==25.1.0",
    "sqlalchemy==2.0.49",
    "cryptography==48.0.0",
    "httpx==0.28.1",
]


def _ensure_cli_deps(root):
    probe = [sys.executable, "-c", "import unsloth_cli.commands.train"]
    if subprocess.run(probe, cwd = root, capture_output = True).returncode == 0:
        return
    req = root / "studio" / "backend" / "requirements" / "studio.txt"
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", str(req), *CLI_DEPS], check = True
    )
    r = subprocess.run(probe, cwd = root, capture_output = True, text = True)
    if r.returncode != 0:
        raise SystemExit(f"unsloth_cli.commands.train still does not import:\n{r.stderr[-2000:]}")


def _dataset(path):
    rows = [
        {
            "instruction": f"What is {i} plus {i}?",
            "input": "",
            "output": f"{i} plus {i} is {2 * i}.",
        }
        for i in range(16)
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _adapters(out_dir):
    return sorted(str(p.relative_to(out_dir)) for p in out_dir.rglob("*.safetensors"))


def _arm(rec, name, argv, out_dir, root, env, timeout):
    t0 = time.perf_counter()
    log = out_dir.with_suffix(".log")
    with open(log, "w") as fh:
        r = subprocess.run(
            argv, cwd = root, env = env, stdout = fh, stderr = subprocess.STDOUT, timeout = timeout
        )
    wall = round(time.perf_counter() - t0, 1)
    tail = log.read_text(errors = "replace")[-3000:]
    print(f"--- {name} (exit {r.returncode}, {wall}s) ---\n{tail}", flush = True)
    rec.check(f"{name}_exit0", r.returncode == 0, f"exit {r.returncode}; tail: {tail[-600:]}")
    saved = _adapters(out_dir) if out_dir.is_dir() else []
    rec.check(f"{name}_adapter_saved", bool(saved), saved or f"nothing under {out_dir}")
    rec.summary(**{f"{name}_wall_s": wall})
    return tail


def main(argv = None):
    p = C.base_parser("mlx_cli_train", DEFAULT_MODEL, DEFAULT_MODEL, default_steps = 3)
    p.add_argument("--no-ddp", action = "store_true")
    p.add_argument("--timeout", type = int, default = 1500, help = "per arm, seconds")
    a = C.resolve_args(p, argv)
    with C.JobRecorder(a, backend_hint = "unsloth-mlx-cli") as rec:
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            rec.skip("needs Apple Silicon (MLX)")
        root = Path.cwd().resolve()
        if not (root / "unsloth_cli" / "commands" / "train.py").is_file():
            raise SystemExit(f"run from an unsloth checkout root, not {root}")
        _ensure_cli_deps(root)

        work = Path(a.out).resolve().parent / "mlx_cli_train_work"
        shutil.rmtree(work, ignore_errors = True)
        work.mkdir(parents = True)
        data = work / "data.jsonl"
        _dataset(data)
        env = dict(os.environ)
        env["UNSLOTH_STUDIO_HOME"] = str(work / "studio_home")
        env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(root), env.get("PYTHONPATH"))))

        def cli(out_dir):
            return [
                "-m",
                "unsloth_cli",
                "train",
                "--model",
                a.model,
                "--local-dataset",
                str(data),
                "--format-type",
                "alpaca",
                "--max-steps",
                str(a.max_steps),
                "--batch-size",
                "2",
                "--gradient-accumulation-steps",
                "1",
                "--warmup-steps",
                "0",
                "--max-seq-length",
                str(a.max_seq_length),
                "--no-load-in-4bit",
                "--random-seed",
                str(a.seed),
                "--output-dir",
                str(out_dir),
            ]

        single = work / "single"
        _arm(rec, "single", [sys.executable, *cli(single)], single, root, env, a.timeout)

        if not a.no_ddp:
            launcher = Path(sys.executable).with_name("mlx.launch")
            if not launcher.is_file():
                launcher = shutil.which("mlx.launch")
            rec.check("mlx_launch_found", bool(launcher), str(launcher))
            if launcher:
                ddp = work / "ddp"
                tail = _arm(
                    rec,
                    "ddp",
                    [str(launcher), "-n", "2", "--", sys.executable, *cli(ddp)],
                    ddp,
                    root,
                    env,
                    a.timeout,
                )
                # Rank 0 owns the save: one adapter set, not one per rank.
                single_n, ddp_n = len(_adapters(single)), len(_adapters(ddp)) if ddp.is_dir() else 0
                rec.check(
                    "ddp_single_writer",
                    ddp_n > 0 and ddp_n == single_n,
                    f"single saved {single_n} safetensors, ddp saved {ddp_n}",
                )
                rec.check("ddp_no_rank_failure", "Unsloth MLX DDP:" not in tail, tail[-400:])


if __name__ == "__main__":
    main()
