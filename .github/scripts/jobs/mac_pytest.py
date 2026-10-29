"""Apple Silicon: run named pytest files strictly (a skip-only run fails), e.g. MLX tests a PR does not
touch. With --also-base the same files also run against origin/<base-ref> (PYTHONPATH = base tree),
and the check is that head fails no test that base passes (failing IDs compared, not counts).

    python jobs/mac_pytest.py --files tests/python/test_mlx_public_trainer_api.py --also-base
"""

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as c  # noqa: E402

p = c.base_parser("mac_pytest", "n/a", "n/a", default_steps = 0)
p.add_argument("--files", nargs = "+", required = True)
p.add_argument("--also-base", action = "store_true")
p.add_argument("--base-ref", default = "upstream-main")
a = c.resolve_args(p)
c.reject_cfg(a)

_FAILED = re.compile(r"^(?:FAILED|ERROR) (\S+)", re.M)


def run(root, files):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(root), str(root / "studio" / "backend")]
        + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-rfEs", "-p", "no:cacheprovider", *files],
        cwd = str(root),
        env = env,
        capture_output = True,
        text = True,
    )
    out = (r.stdout or "") + (r.stderr or "")
    print(f"===== {root} =====\n" + out[-8000:], flush = True)
    last = out.strip().splitlines()[-1] if out.strip() else ""
    counts = {k: int(v) for v, k in re.findall(r"(\d+) (passed|failed|skipped|errors?)", last)}
    return r.returncode, counts, sorted(set(_FAILED.findall(out)))


with c.JobRecorder(a) as rec:
    rec.set_backend("pytest")
    head = Path.cwd().resolve()
    rc, counts, failed = run(head, a.files)
    rec.summary(head = counts, head_failed = failed)
    rec.check("head_something_passed", counts.get("passed", 0) > 0, counts)
    if not a.also_base:
        rec.check("pytest_exit_zero", rc == 0, f"exit {rc}")
    else:
        base = Path(tempfile.mkdtemp(prefix = "mac_pytest_base_")) / "base"
        subprocess.run(
            ["git", "fetch", "--no-tags", "--depth=1", "origin", a.base_ref], cwd = head, check = True
        )
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(base), "FETCH_HEAD"], cwd = head, check = True
        )
        files = [f for f in a.files if (base / f).exists()]
        brc, bcounts, bfailed = run(base, files)
        new = sorted(set(failed) - set(bfailed))
        rec.summary(base = bcounts, base_failed = bfailed, new_failures = new)
        rec.check("base_something_passed", bcounts.get("passed", 0) > 0, bcounts)
        rec.check("no_new_failures", not new, new)
