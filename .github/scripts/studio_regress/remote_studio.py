"""Studio journeys and external (diffusion_bench) targets on Colab / Kaggle: the same harness, run there.

A Studio target that has waited OFFLOAD_S (20 min) for a local GPU joins a remote BATCH (offload.py):
every such target of one VRAM group that arrives within BATCH_WINDOW_S goes to one machine, so the two
Studio installs (merge base and head, `install.sh --local` each, a few minutes on a warm uv cache) are
paid once per batch, not per target. The remote job (remote_driver.py, shipped with a bundle of
studio_regress + studio_test_kit + diffusion_bench + the top-level modules they import):

  1. clones the PR's repo (blobless), fetches the merge base and head SHAs (and refs/pull/<N>/head);
  2. installs Playwright + Chromium into its venv;
  3. runs `studio_regress.py run --pr N --only <targets> --side both --no-confirm` there, with the SHAs
     pinned (no gh on the VM: STUDIO_REGRESS_PIN_SHAS), the shared cache off and no nested offload.
     On a Kaggle T4x2 the VM's own scheduler places the sides on its two GPUs;
  4. returns that run's root (report.json, before/ after/ evidence, screenshots, logs) as artifacts:
     Kaggle as an output file, Colab as the notebook stream up to ARTIFACT_MAX_MB, above that through
     the private HF artifact dataset (cloud_pool: sealed upload token, 3-day expiry).

Back here, each arm takes its targets' evidence: journey steps' before/<name> and after/<name> trees
are copied into the local root (the local diff then runs as usual), an external target's step comes
from the remote report with its paths rewritten. Every such step's note starts "REMOTE on <machine>".
A batch that never ran (no slot, VM lost) hands its jobs back to the local queue (offload.Requeue);
one whose remote run failed before writing a report makes those steps VOID, never FAIL_HEAD.
Local only: LOCAL_ONLY (desktop apps need a display; full_access_isolated needs host accounts).
"""

from __future__ import annotations

import ast
import io
import json
import os
import shutil
import sys
import tarfile
import threading
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
LOCAL_ONLY = {"desktop", "desktop_update", "full_access_isolated"}
BATCH_WINDOW_S = float(os.environ.get("STUDIO_REGRESS_REMOTE_BATCH_S") or 90)
ARTIFACT_MAX_MB = 200
BIG_GB = 22.5  # above an L4: a batch of its own (A100-HM / G4)
DISK_GB = 40.0  # two Studio homes (~7.4 GB each), two trees, fixture models, Chromium
RAM_GB = 12.0
SETUP_S = 1500  # clone + two installs + Playwright, cold
PACKAGES = ("studio_regress", "studio_test_kit", "diffusion_bench", "pr_ui_scenes")
# Kaggle embeds a kernel's notebooks only up to 0.8 MB compressed: docs, the AMD suites and the
# remote dispatchers (a remote run never offloads again: STUDIO_REGRESS_OFFLOAD=0) stay behind
BUNDLE_SKIP_DIRS = ("diffusion_bench/amd/",)
BUNDLE_SKIP_MODULES = {"cloud_pool", "notebook_cloud_run"}
# studio_regress.PW_PACKAGES plus what the harness imports at the top level
REMOTE_REQS = [
    "playwright",
    "numpy",
    "pillow",
    "httpx",
    "huggingface_hub",
    "pyyaml",
    "pytest",
    "psutil",
    "requests",
]


def local_playwright():
    """The Playwright version this harness runs (each release pins one Chromium build), else None."""
    import importlib.metadata
    try:
        return importlib.metadata.version("playwright")
    except importlib.metadata.PackageNotFoundError:
        return None


def remote_requirements(pw_version = None):
    """REMOTE_REQS with Playwright pinned to the local harness's version, so the VM installs the same
    Chromium build as local runs (both remote sides share that one install); unpinned when unknown."""
    v = pw_version if pw_version is not None else local_playwright()
    return [f"playwright=={v}" if r == "playwright" and v else r for r in REMOTE_REQS]


def group_of(gb):
    return "big" if float(gb or 0) > BIG_GB else "small"


def target_names(job):
    """The switchboard targets a scheduler job runs (journey chain names, or the external target)."""
    names = []
    for a in job.arms:
        if a.kind == "external" and a.target:
            names.append(a.target["name"])
        names.extend(a.names)
    out = []
    for n in names:
        if n not in out:
            out.append(n)
    return out


def _registry():
    try:
        from studio_regress import selection
        return {t["name"]: t for t in selection.load().get("target") or []}
    except Exception:  # noqa: BLE001 - no registry: only the arms' own targets are checked
        return {}


def eligible(job, registry = None):
    """(ok, why not) for a Studio-side job (journey / external). `remote = false` is read from the
    registry by name too: journey arms carry no target dict, only their chain's names."""
    if job.kind not in ("journey", "external"):
        return False, f"{job.kind} units run on local trees"
    names = target_names(job)
    local = sorted(set(names) & LOCAL_ONLY)
    if local:
        return False, f"{', '.join(local)} must run on this host"
    reg = _registry() if registry is None else registry
    if any((reg.get(n) or {}).get("remote") is False for n in names):
        return False, "the target says remote = false"
    for a in job.arms:
        if a.target and a.target.get("remote") is False:
            return False, "the target says remote = false"
        if a.target and a.target.get("compare"):
            return False, "compare targets run both sides locally"
    return True, ""


# --------------------------------------------------------------------------- the bundle
def _imports(path):
    try:
        tree = ast.parse(path.read_text())
    except (OSError, SyntaxError, ValueError):
        return set()
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out.update(a.name.split(".")[0] for a in n.names)
        elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
            out.add(n.module.split(".")[0])
    return out


def bundle_files(scripts = SCRIPTS):
    """studio_regress.py, the shipped packages, and every top-level scripts module they import
    (transitively). Tests and caches stay behind. -> [relative paths]."""
    scripts = Path(scripts)
    files = ["studio_regress.py"]
    for pkg in PACKAGES:
        d = scripts / pkg
        if not d.is_dir():
            continue
        for p in sorted(d.rglob("*")):
            rel = p.relative_to(scripts)
            if (
                p.is_dir()
                or "__pycache__" in rel.parts
                or p.name.startswith("test_")
                or p.suffix in (".pyc", ".log", ".md")
                or str(rel).startswith(BUNDLE_SKIP_DIRS)
            ):
                continue
            files.append(str(rel))
    top = {p.stem for p in scripts.glob("*.py")}
    todo = [scripts / f for f in files if f.endswith(".py")]
    seen = set()
    while todo:
        f = todo.pop()
        for m in (_imports(f) & top) - BUNDLE_SKIP_MODULES:
            rel = f"{m}.py"
            if rel not in seen and rel not in files:
                seen.add(rel)
                files.append(rel)
                todo.append(scripts / rel)
    return files


def build_bundle(dest, scripts = SCRIPTS):
    """tar.xz of bundle_files() at dest (xz: the Kaggle kernel size cap). -> dest."""
    scripts = Path(scripts)
    with tarfile.open(dest, "w:xz") as tar:
        for rel in bundle_files(scripts):
            tar.add(scripts / rel, arcname = rel)
    return Path(dest)


# --------------------------------------------------------------------------- the remote job
def remote_job(
    targets,
    pr,
    gh_repo,
    base_sha,
    head_sha,
    head_ref,
    gb,
    bundle,
    run_id,
    job_id = None,
):
    return {
        "id": job_id or "srs-" + uuid.uuid4().hex[:8],
        "script": str(HERE / "remote_driver.py"),
        "files": [str(bundle)],
        "argv": [
            "--pr",
            str(pr),
            "--gh-repo",
            gh_repo,
            "--base-sha",
            base_sha,
            "--head-sha",
            head_sha,
            "--head-ref",
            head_ref or "",
            "--targets",
            ",".join(targets),
            "--bundle",
            Path(bundle).name,
        ],
        "gb": max(float(gb or 0), 4.0),
        "dtype": "fp16",
        "ram_gb": RAM_GB,
        "disk_gb": DISK_GB,
        "est_s": SETUP_S + 600 * max(1, len(targets)),
        "class": "eval",
        "torch": "none",
        "requirements": remote_requirements(),
        "whole_kernel": True,
        "shared_models": True,
        "artifacts": ["sr_out"],
        "artifact_max": ARTIFACT_MAX_MB * 1024 * 1024,
        "artifact_upload": True,
        "artifact_prefix": f"runs/{time.strftime('%Y%m%dT%H%M', time.gmtime())}-{run_id}",
        "timeout_s": int(SETUP_S * 2 + 900 * max(1, len(targets))),
        "what": f"studio_regress pr{pr} remote Studio: {','.join(targets)}"[:80],
    }


class Batch:
    """Targets of one VRAM group sent to one remote machine. The first arm to need the result
    dispatches it once the joining window has closed; every arm then takes its own evidence."""

    def __init__(
        self,
        bid,
        group,
        machine,
        clock = time.time,
        window_s = None,
    ):
        self.id, self.group, self.machine = bid, group, dict(machine)
        self.clock = clock
        self.window_until = clock() + (BATCH_WINDOW_S if window_s is None else window_s)
        self.targets, self.jobs, self.gb = [], set(), 0.0
        self.lock = threading.Lock()
        self.done = threading.Event()
        self.started = False  # an arm is waiting on it (its window may still be open)
        self.dispatched = False  # the window closed and the remote job went out: nothing joins now
        self.res = None  # cloud_pool result
        self.root = None  # the remote run's root, extracted here
        self.report = None
        self.error = None
        self.requeue = False

    def open(self):
        return not self.dispatched and self.clock() < self.window_until

    def add(self, job_id, names, gb):
        self.jobs.add(job_id)
        self.gb = max(self.gb, float(gb or 0))
        for n in names:
            if n not in self.targets:
                self.targets.append(n)

    def result(
        self,
        run,
        sleep = time.sleep,
    ):
        """Run the batch once (`run(batch)` fills res / root / report / error / requeue); every
        caller blocks until it has finished."""
        with self.lock:
            first = not self.started
            self.started = True
        if first:
            while self.clock() < self.window_until:
                sleep(min(5.0, max(0.0, self.window_until - self.clock())))
            with self.lock:
                self.dispatched = True
            try:
                run(self)
            except Exception as e:  # noqa: BLE001 - one bad batch is VOID steps, never a crash
                self.error = f"{type(e).__name__}: {e}"
            finally:
                self.done.set()
        self.done.wait()
        return self


def dispatch_batch(
    batch,
    pr,
    gh_repo,
    shas,
    root,
    dispatch = None,
    log = print,
):
    """Send `batch` and extract its returned root under <root>/remote/<batch id>/."""
    base_sha, head_sha, head_ref = shas
    work = Path(root).resolve() / "remote" / batch.id
    work.mkdir(parents = True, exist_ok = True)
    bundle = build_bundle(work / "sr_bundle.tar.xz")
    job = remote_job(
        batch.targets,
        pr,
        gh_repo,
        base_sha,
        head_sha,
        head_ref,
        batch.gb,
        bundle,
        batch.id,
        job_id = batch.machine.get("job_id"),
    )
    if dispatch is None:
        import cloud_pool
        dispatch = cloud_pool.dispatch
    log(
        f"SWITCHBOARD: remote Studio batch {batch.id} ({', '.join(batch.targets)}) -> "
        f"{batch.machine.get('backend')} {batch.machine.get('tier')}"
    )
    res = dispatch(job, prefer = batch.machine)
    batch.res = res
    status = res.get("status")
    if status not in ("PASS", "FAIL"):
        batch.requeue = True
        batch.error = f"{status}: {res.get('reason', '')}"[:400]
        return
    if res.get("backend"):
        batch.machine.update(
            {
                k: res.get(k)
                for k in ("backend", "tier", "account", "token_env", "worker")
                if res.get(k)
            }
        )
    found = None
    for p in res.get("artifacts") or []:
        p = Path(p)
        if p.name == "report.json" and p.parent.name == "sr_out":
            found = p.parent
            break
    if found is None:
        tail = ((res.get("result") or {}).get("tail") or "")[-600:]
        batch.error = f"the remote run returned no report (rc {res.get('rc')}): {tail}"
        return
    batch.root = found
    batch.report = json.loads((found / "report.json").read_text())


def describe(batch):
    m = batch.machine
    who = m.get("account") or m.get("token_env") or "?"
    return f"{str(m.get('backend', '?')).upper()} {m.get('tier', '?')} ({who})"


def _rewrite(obj, old, new):
    if isinstance(obj, str):
        return obj.replace(old, new)
    if isinstance(obj, list):
        return [_rewrite(x, old, new) for x in obj]
    if isinstance(obj, dict):
        return {k: _rewrite(v, old, new) for k, v in obj.items()}
    return obj


def take_evidence(batch, arm, root):
    """Copy this arm's evidence from the batch's remote root into the local root. -> the external
    step record (external arms), or {} (journey arms: the local diff reads the copied trees)."""
    root = Path(root).resolve()
    src = batch.root
    sides = ("before", "after") if arm.side == "both" else (arm.side,)
    names = [arm.target["name"]] if arm.kind == "external" else list(arm.names)
    for side in sides:
        for n in names:
            s = src / side / n
            if s.is_dir():
                d = root / side / n
                shutil.rmtree(d, ignore_errors = True)
                shutil.copytree(s, d)
    try:
        remote_root = (src / "remote_root.txt").read_text().strip() or str(src)
    except OSError:
        remote_root = str(src)
    where = describe(batch)
    if arm.kind != "external":
        for side in sides:
            for n in names:
                for f in (root / side / n).rglob("*.json"):
                    try:
                        text = f.read_text()
                    except OSError:
                        continue
                    if remote_root in text:
                        f.write_text(text.replace(remote_root, str(root)))
        return {"_remote": where}
    key = f"{arm.target['name']}/run"
    step = next(
        (st for st in (batch.report or {}).get("steps") or [] if st.get("key") == key), None
    )
    if step is None:
        return None
    step = _rewrite(step, remote_root, str(root))
    step["note"] = f"REMOTE on {where}: {step.get('note') or ''}".rstrip(": ")
    step["remote"] = {
        "where": where,
        "batch": batch.id,
        **{k: batch.machine.get(k) for k in ("backend", "tier", "account", "token_env", "worker")},
    }
    return step


def unpack_tarball(data, dest):
    dest = Path(dest)
    dest.mkdir(parents = True, exist_ok = True)
    with tarfile.open(fileobj = io.BytesIO(data), mode = "r:gz") as tar:
        if sys.version_info >= (3, 12):
            tar.extractall(dest, filter = "data")
        else:
            tar.extractall(dest)
    return dest
