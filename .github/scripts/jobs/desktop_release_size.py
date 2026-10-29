"""Desktop (Tauri) release build: base (origin/<base-ref>) vs head (this checkout) on one runner.
Measures `cargo build --release --locked` wall time and the size of the built binary, each arm in
its own CARGO_TARGET_DIR (so both pay the full dependency compile), base first then head.
Optionally runs `cargo test --locked <filter>` on head.

    python jobs/desktop_release_size.py --tiny                         # staging leg
    python jobs/desktop_release_size.py --cargo-test shell_path         # + head Rust tests
"""

import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as c  # noqa: E402

p = c.base_parser("desktop_release_size", "n/a", "n/a", default_steps = 0)
p.add_argument("--base-ref", default = "upstream-main")
p.add_argument("--head-dir", default = ".")
p.add_argument("--cargo-test", default = None, help = "cargo test filter run on head after the build")
p.add_argument("--skip-build", action = "store_true", help = "only run --cargo-test on head")
a = c.resolve_args(p)
c.reject_cfg(a)

LINUX_DEPS = [
    "libwebkit2gtk-4.1-dev",
    "libappindicator3-dev",
    "librsvg2-dev",
    "libxdo-dev",
    "libssl-dev",
    "patchelf",
]


def run(
    cmd,
    cwd = None,
    env = None,
    check = True,
):
    print("+", " ".join(map(str, cmd)), flush = True)
    return subprocess.run(cmd, cwd = cwd, env = env, check = check)


def git(*args, cwd):
    return subprocess.run(
        ["git", *args], cwd = cwd, check = True, capture_output = True, text = True
    ).stdout.strip()


def make_base(head):
    dst = Path(tempfile.mkdtemp(prefix = "desktop_size_base_"))
    git("fetch", "--no-tags", "--depth=1", "origin", a.base_ref, cwd = head)
    git("worktree", "add", "--detach", str(dst / "base"), "FETCH_HEAD", cwd = head)
    return Path(os.path.realpath(dst / "base"))


def ensure_dist(root):
    """generate_context! embeds frontendDist at compile time; an empty page is enough to link."""
    dist = root / "studio" / "frontend" / "dist"
    if not (dist / "index.html").exists():
        dist.mkdir(parents = True, exist_ok = True)
        (dist / "index.html").write_text("<!doctype html><title>size probe</title>")


def binary(target):
    name = "unsloth-studio.exe" if os.name == "nt" else "unsloth-studio"
    return target / "release" / name


def build(arm, root, target):
    ensure_dist(root)
    env = dict(os.environ, CARGO_TARGET_DIR = str(target), CARGO_TERM_COLOR = "never")
    t0 = time.perf_counter()
    r = run(
        ["cargo", "build", "--release", "--locked"],
        cwd = root / "studio" / "src-tauri",
        env = env,
        check = False,
    )
    secs = time.perf_counter() - t0
    b = binary(target)
    return {
        "arm": arm,
        "ok": r.returncode == 0 and b.exists(),
        "build_s": round(secs, 1),
        "bytes": b.stat().st_size if b.exists() else None,
    }


with c.JobRecorder(a) as rec:
    rec.set_backend("cargo-release")
    if sys.platform.startswith("linux"):
        sudo = [] if os.geteuid() == 0 else ["sudo"]
        run(sudo + ["apt-get", "update", "-qq"])
        run(sudo + ["apt-get", "install", "-y", "-qq", *LINUX_DEPS])
    if not shutil.which("cargo"):
        rec.check("cargo_available", False, "cargo not on PATH")
        raise SystemExit(1)
    head = Path(a.head_dir).resolve()
    rec.summary(
        os = platform.platform(),
        rustc = subprocess.run(["rustc", "-V"], capture_output = True, text = True).stdout.strip(),
        head_sha = git("rev-parse", "HEAD", cwd = head),
    )
    if not a.skip_build:
        base = make_base(head)
        rec.summary(base_sha = git("rev-parse", "HEAD", cwd = base))
        scratch = Path(tempfile.mkdtemp(prefix = "desktop_size_targets_"))
        rows = {}
        for arm, root in (("base", base), ("head", head)):
            rows[arm] = build(arm, root, scratch / arm)
            rec.step(**rows[arm])
            rec.check(f"{arm}_builds", rows[arm]["ok"], rows[arm])
            shutil.rmtree(
                scratch / arm / "release" / "deps", ignore_errors = True
            )  # free runner disk
        if rows["base"]["ok"] and rows["head"]["ok"]:
            rec.summary(
                size_ratio = round(rows["head"]["bytes"] / rows["base"]["bytes"], 4),
                build_time_ratio = round(rows["head"]["build_s"] / rows["base"]["build_s"], 3),
                base_mb = round(rows["base"]["bytes"] / 2**20, 2),
                head_mb = round(rows["head"]["bytes"] / 2**20, 2),
            )
    if a.cargo_test:
        ensure_dist(head)
        env = dict(os.environ, CARGO_TERM_COLOR = "never")
        r = run(
            ["cargo", "test", "--locked", a.cargo_test],
            cwd = head / "studio" / "src-tauri",
            env = env,
            check = False,
        )
        rec.check(
            "head_cargo_test", r.returncode == 0, f"cargo test {a.cargo_test}: exit {r.returncode}"
        )
