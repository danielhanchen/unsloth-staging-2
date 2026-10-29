"""Import / startup A/B on one runner: base (merge base) vs head (this checkout), ABBA-interleaved,
fresh process per sample. Wall time per sample plus -X importtime cumulative us of the target
module (in-process, less sensitive to process-spawn noise). Checks each arm really imported its own
tree (module __file__ under the arm root) and that both arms succeed.

    python jobs/import_ab.py --tiny                          # staging leg: base = origin/upstream-main
    python jobs/import_ab.py --base-dir ../base --rounds 30  # local: explicit base checkout
    python jobs/import_ab.py --target cli=unsloth_cli --target backend=main@studio/backend

Targets NAME=MODULE[@SUBDIR]: SUBDIR (relative to the arm root) is prepended to PYTHONPATH as well.
A target whose import fails in BOTH arms with ModuleNotFoundError for a third-party dep is recorded
as skipped (absent dep), never as a pass.
"""

import os
import random
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as c  # noqa: E402

DEFAULT_TARGETS = (
    "cli=unsloth_cli",
    "backend=main@studio/backend",
    "zoo=unsloth_zoo",
    "unsloth=unsloth",
)

p = c.base_parser("import_ab", "n/a", "n/a", default_steps = 0)
p.add_argument("--base-dir", default = None, help = "base checkout (default: worktree of --base-ref)")
p.add_argument("--base-ref", default = "upstream-main", help = "remote branch fetched for the base arm")
p.add_argument("--head-dir", default = ".", help = "head checkout (default: cwd)")
p.add_argument("--rounds", type = int, default = 24, help = "ABBA rounds (2 samples per arm per round)")
p.add_argument("--warmup", type = int, default = 2, help = "discarded rounds (page cache, pyc writes)")
p.add_argument("--target", action = "append", default = [], metavar = "NAME=MODULE[@SUBDIR]")
a = c.resolve_args(p)
c.reject_cfg(a)

_IMPORTTIME = re.compile(r"import time:\s+(\d+)\s+\|\s+(\d+)\s+\|\s?(\s*)(\S+)")
_PROBE = (
    "import sys, time, json\n"
    "t = time.perf_counter()\n"
    "import {mod} as m\n"
    "dt = time.perf_counter() - t\n"
    "print('IMPORT_AB ' + json.dumps({{'s': dt, 'file': getattr(m, '__file__', None) or '',"
    " 'mods': len(sys.modules)}}))\n"
)


def git(*args, cwd):
    return subprocess.run(
        ["git", *args], cwd = cwd, check = True, capture_output = True, text = True
    ).stdout.strip()


def make_base(head):
    """Worktree of origin/<base-ref> next to the head checkout."""
    dst = Path(tempfile.mkdtemp(prefix = "import_ab_base_"))
    git("fetch", "--no-tags", "--depth=1", "origin", a.base_ref, cwd = head)
    git("worktree", "add", "--detach", str(dst / "base"), "FETCH_HEAD", cwd = head)
    return Path(os.path.realpath(dst / "base"))


def under(path, root):
    """path inside root, comparing real paths (macOS /var -> /private/var, Windows 8.3 short names)."""
    p = os.path.normcase(os.path.realpath(path))
    r = os.path.normcase(os.path.realpath(root))
    return p == r or p.startswith(r.rstrip(os.sep) + os.sep)


def parse_targets(specs):
    out = []
    for spec in specs:
        name, _, rest = spec.partition("=")
        mod, _, sub = rest.partition("@")
        if not name or not mod:
            raise SystemExit(f"--target {spec!r}: expected NAME=MODULE[@SUBDIR]")
        out.append((name, mod, sub))
    return out


def run_once(root, mod, sub):
    env = dict(os.environ)
    paths = [str(root / sub)] if sub else []
    paths.append(str(root))
    env["PYTHONPATH"] = os.pathsep.join(
        paths + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    env.setdefault("UNSLOTH_DISABLE_AUTO_UPDATES", "1")
    env["PYTHONIOENCODING"] = "utf-8"
    t0 = time.perf_counter()
    r = subprocess.run(
        [sys.executable, "-X", "importtime", "-c", _PROBE.format(mod = mod)],
        cwd = str(root),
        env = env,
        capture_output = True,
        text = True,
        encoding = "utf-8",
        errors = "replace",
    )
    wall = time.perf_counter() - t0
    if r.returncode != 0:
        return {"ok": False, "err": (r.stderr or "")[-1500:]}
    line = next((l for l in r.stdout.splitlines() if l.startswith("IMPORT_AB ")), None)
    import json

    info = json.loads(line[len("IMPORT_AB ") :]) if line else {}
    cum = None
    for m in _IMPORTTIME.finditer(r.stderr or ""):
        if m.group(4) == mod and not m.group(3):
            cum = int(m.group(2))
    return {
        "ok": True,
        "wall_s": wall,
        "import_s": info.get("s"),
        "file": info.get("file", ""),
        "mods": info.get("mods"),
        "importtime_us": cum,
    }


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def boot_ci(
    b,
    h,
    n = 2000,
    seed = 0,
):
    """95% bootstrap CI of median(head) / median(base)."""
    b = [x for x in b if x]
    h = [x for x in h if x]
    if len(b) < 3 or len(h) < 3:
        return None
    rng = random.Random(seed)
    rs = sorted(
        statistics.median(rng.choices(h, k = len(h))) / statistics.median(rng.choices(b, k = len(b)))
        for _ in range(n)
    )
    return [round(rs[int(0.025 * n)], 4), round(rs[int(0.975 * n)], 4)]


with c.JobRecorder(a) as rec:
    rec.set_backend("cpu-import")
    head = Path(a.head_dir).resolve()
    base = Path(a.base_dir).resolve() if a.base_dir else make_base(head)
    rec.summary(
        base_dir = str(base),
        head_dir = str(head),
        base_sha = git("rev-parse", "HEAD", cwd = base),
        head_sha = git("rev-parse", "HEAD", cwd = head),
    )
    arms = {"base": base, "head": head}
    for name, mod, sub in parse_targets(a.target or DEFAULT_TARGETS):
        first = {arm: run_once(root, mod, sub) for arm, root in arms.items()}
        bad = {arm: r["err"] for arm, r in first.items() if not r["ok"]}
        if bad:
            missing = all("ModuleNotFoundError" in e for e in bad.values()) and len(bad) == 2
            if missing:
                rec.summary(
                    **{
                        f"{name}_skipped": "dependency absent in both arms: "
                        + bad["head"].strip().splitlines()[-1][:300]
                    }
                )
                continue
            rec.check(f"{name}_imports", False, {k: v[-600:] for k, v in bad.items()})
            continue
        own = {arm: under(r["file"], arms[arm]) for arm, r in first.items()}
        rec.check(
            f"{name}_arm_isolation", all(own.values()), {arm: first[arm]["file"] for arm in arms}
        )
        samples = {"base": [], "head": []}
        failures = []
        for rnd in range(a.warmup + a.rounds):
            order = (
                ("base", "head", "head", "base")
                if rnd % 2 == 0
                else ("head", "base", "base", "head")
            )
            for arm in order:
                r = run_once(arms[arm], mod, sub)
                if not r["ok"]:
                    failures.append((arm, r["err"][-300:]))
                elif rnd >= a.warmup:
                    samples[arm].append(r)
        rec.check(f"{name}_no_failures", not failures, failures[:3])
        row = {"target": name, "module": mod}
        for key in ("wall_s", "import_s", "importtime_us", "mods"):
            bm = med([s[key] for s in samples["base"]])
            hm = med([s[key] for s in samples["head"]])
            row[f"base_{key}"] = bm
            row[f"head_{key}"] = hm
            if bm and hm and key != "mods":
                row[f"ratio_{key}"] = round(hm / bm, 4)
                row[f"ci95_{key}"] = boot_ci(
                    [s[key] for s in samples["base"]], [s[key] for s in samples["head"]]
                )
        row["n_per_arm"] = len(samples["head"])
        rec.step(**row)
        print(f"[import_ab] {row}", flush = True)
    rec.check("measured_something", bool(rec.data["steps"]), "no target produced samples")
