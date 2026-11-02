#!/usr/bin/env python3
"""Before/after Studio evidence for a PR, from two isolated installs.

    python pr_ui_diff.py --pr 8222                    # a registered PR
    python pr_ui_diff.py --pr 8222 --skip-install     # reuse the homes

A PR nobody has registered needs no edit to registry.py to be shot. Name a scene
and what the shot must show, and check it costs nothing first:

    python pr_ui_diff.py --pr 9101 --scene gguf_picker_rows \
        --expect "4 rows -> 18 rows, each at its own true size" \
        --scene-kwargs '{"repo": "org/Model-GGUF"}' --preflight

`--preflight` resolves the PR, imports the scene and stops before building, so a
typo costs seconds rather than two installs. Drop it to shoot. Register the plan
once it works: `verified` is where what was actually observed gets written down.

Installs Studio twice under two distinct UNSLOTH_STUDIO_HOMEs -- one at the PR's
MERGE BASE, one at its head -- drives the same scene against both, and composes a
labelled BEFORE/AFTER image.

Two design points, both load-bearing:

* Two full installs, not one Studio with a `git checkout` between shots.
  `install.sh --local` builds the frontend from the checked-out tree, so a UI
  change only exists in a build made from that tree. Swapping branches under a
  running Studio serves the old bundle and yields two identical screenshots --
  which looks exactly like a successful comparison of a PR that changed nothing.

* BEFORE is the PR's merge base, not current main. `main` moves several times a
  day here; photographing against it shows the PR's change plus everything else
  that landed in between, and credits the lot to the PR.

The scene, its parameters and the difference it is expected to show all live in
`pr_ui_scenes/registry.py`. Read the plan's `expect` before believing any pair.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import fcntl
import hashlib
import importlib
import json
import os
import secrets
import shutil
import signal
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

# A standalone install (identity.json beside this file, e.g. ~/.claude/scripts) has no workspace
# around it: data lives under the cwd, which is the unsloth checkout being worked on.
STANDALONE = Path(__file__).with_name("identity.json").is_file() and not (
    os.environ.get("WORKSPACE") or os.environ.get("UNSLOTH_WORKSPACE")
)
WORKSPACE = (
    Path(
        os.environ.get("WORKSPACE")
        or os.environ.get("UNSLOTH_WORKSPACE")
        or (Path.cwd() if STANDALONE else Path(__file__).resolve().parent.parent)
    )
    .expanduser()
    .resolve()
)
DEFAULT_REPO_DIR = WORKSPACE if STANDALONE else WORKSPACE / "unsloth"
# The driver's OWN directory first, so it imports the `pr_ui_scenes` and
# `studio_test_kit` it ships with. `$WORKSPACE` also holds a deployed copy
# of both, and putting that first meant a checkout's driver silently ran the
# deployed package instead of its own -- the same class of defect the .uidiff_sha
# stamp exists to prevent, and it was hit live as an ImportError only because the
# two copies had drifted far enough to notice. WORKSPACE stays, for data paths.
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pw_env  # noqa: E402

pw_env.ensure(
    pw_env.KIT_MODULES, tag = "pr_ui_diff"
)  # no Playwright here: re-exec under temp/venv_pw
from pr_ui_scenes._common import api_delete, api_get, pick_free_ports, studio_session  # noqa: E402
from pr_ui_scenes.registry import ScenePlan, plan_for  # noqa: E402
from studio_test_kit.compose import hstack_images  # noqa: E402
from studio_test_kit.lifecycle import StudioInstall, launch_studio, stop_studio  # noqa: E402


def _sh(
    cmd: list[str],
    cwd: Optional[Path] = None,
    env: Optional[dict] = None,
    timeout: Optional[int] = None,
    check: bool = True,
) -> str:
    """Run a command, raising with BOTH streams on failure. A 40 minute install
    that dies printing neither is not debuggable after the fact."""
    proc = subprocess.run(
        cmd,
        cwd = cwd,
        env = {**os.environ, **(env or {})},
        text = True,
        capture_output = True,
        timeout = timeout,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"--- stdout ---\n{proc.stdout[-4000:]}\n--- stderr ---\n{proc.stderr[-4000:]}"
        )
    return proc.stdout


# Installs are CPU bound (a vite build and a large dependency resolve), so past a
# point more of them in parallel is slower, not faster, through contention. One PR
# only ever has two sides; the cap is for a batch driving several PRs at once.
MAX_PARALLEL_INSTALLS = 8

# npm script shell that runs the Studio frontend build (`npm run build`: the TypeScript 7
# typecheck + `vite build`) on STUDIO_FRONTEND_CPUS (8) CPUs and every other npm script as is.
# Unpinned, each build sizes its thread pools to all the host's CPUs.
FRONTEND_CPU_CAP_SHELL = Path(__file__).resolve().with_name("npm_build_cpu_cap.sh")


def frontend_cpu_env(env: dict) -> dict:
    """npm_config_script_shell for install.sh, unless the caller set its own script shell (the
    launcher sets this same one) or the wrapper is missing; {} then."""
    if any(k.lower() == "npm_config_script_shell" for k in env):
        return {}
    if not os.access(FRONTEND_CPU_CAP_SHELL, os.X_OK):
        return {}
    return {"npm_config_script_shell": str(FRONTEND_CPU_CAP_SHELL)}


_GH_TOKEN_VARS = ("GH_TOKEN", "GH_TOKEN_shimmyshimmer", "GH_TOKEN_danielhanchen")


def _gh_env() -> dict:
    """gh with every ambient token removed, so it uses the stored login.

    The ambient `GH_TOKEN` lacks scopes and 403s, and `gh` silently prefers it
    over the keyring identity.
    """
    return {k: v for k, v in os.environ.items() if k not in _GH_TOKEN_VARS}


def _gh_envs() -> list[dict]:
    """Every credential worth trying, stored login first.

    One credential is not enough to depend on. `gh pr view --json` goes through
    GraphQL, which has its own budget separate from REST, and an exhausted one
    fails the whole run on its first call with "API rate limit already exceeded"
    even while `gh api rate_limit` reports REST wide open. The accounts do not
    share a budget, so trying the next one costs a second and gets the SHAs.
    """
    envs = [_gh_env()]
    base = _gh_env()
    for var in ("GH_TOKEN_shimmyshimmer", "GH_TOKEN_danielhanchen"):
        token = os.environ.get(var)
        if token:
            envs.append({**base, "GH_TOKEN": token})
    return envs


def password_for(
    root: Path,
    home: Path,
    override: Optional[str] = None,
) -> str:
    """The credential for ONE home, minted once and reused.

    Per home, not per pair, and that is the whole point. `studio_session` sells
    the login as proof that the server answering is the install we built, but a
    password shared by home_before and home_after cannot tell the two apart: if
    the sibling side squats the port -- the likeliest squatter of all, since both
    live under one root and both are launched from the same band -- the login
    succeeds against the WRONG Studio. That costs a mislabelled screenshot when
    the run only photographs, and a wrong deletion once it also resets threads.
    Distinct passwords make the login answer the question it claims to answer.

    Persisted rather than random per run: Studio deletes the bootstrap password
    once rotated, so a fresh one each time would lock us out of our own homes the
    moment --skip-install is used.
    """
    # `--password` wins over everything. Every scene's own entry point takes one,
    # and `studio_session`'s failure hint tells the operator to pass "the same
    # --password", so the driver has to accept it too. Needed whenever the home
    # was rotated by something other than this driver: nothing but a reinstall
    # can change a rotated home's credential, so a minted one would just 403.
    if override:
        per_home = root / f"studio_password_{home.name}.txt"
        per_home.write_text(override)
        return override
    per_home = root / f"studio_password_{home.name}.txt"
    if per_home.exists():
        return per_home.read_text().strip()
    shared = root / "studio_password.txt"
    # Migration: a home rotated by an older run only accepts the shared password,
    # and nothing can change that without reinstalling it. Adopt it for that home
    # and let the next reinstall mint a distinct one. A home holding a bootstrap
    # file has not been rotated yet, so it can take a fresh password now.
    if shared.exists() and not (home / "auth" / ".bootstrap_password").exists():
        password = shared.read_text().strip()
    else:
        password = "UiDiff-" + secrets.token_urlsafe(12).replace("-", "x")
    per_home.write_text(password)
    return password


def _reset_threads(session) -> int:
    """Delete every chat thread on the server we are about to photograph.

    Threads persist in `<home>/studio.db`, and the sidebar renders them under
    Recents, so a home reused via --skip-install shoots whatever a previous run
    left behind. The stamp gate cannot catch this: the code is right and only the
    data differs, so it survives every check that compares SHAs. Observed as 3
    threads against 0, and 4 against 0, on two different pairs.

    Done here, through the session the driver already holds, rather than left to
    the operator: editing studio.db under a running Studio does not take, and a
    step that has to be remembered is a step that gets skipped on the run that
    matters. A freshly installed home has none, so this is a no-op there.
    """
    listing = api_get(session, "/api/chat/threads")
    ids = [t["id"] for t in (listing.get("threads") or [])]
    if ids:
        api_delete(session, "/api/chat/threads", {"ids": ids, "delete_files": True})
        left = api_get(session, "/api/chat/threads").get("threads") or []
        if left:
            raise RuntimeError(
                f"{len(left)} chat threads survived the reset on {session.base_url}; "
                "the two sides would photograph different Recents lists"
            )
    return len(ids)


def _exit_on_term() -> dict:
    """Turn SIGTERM / SIGHUP (a tool timeout, a closed pane) into SystemExit so a
    `finally` runs. A signal already ignored (nohup) stays ignored. Returns the
    handlers to restore; empty off the main thread, where handlers cannot be set."""
    prev: dict = {}
    if threading.current_thread() is not threading.main_thread():
        return prev
    for name in ("SIGTERM", "SIGHUP"):
        n = getattr(signal, name, None)  # no SIGHUP on Windows
        if n is None or signal.getsignal(n) == signal.SIG_IGN:
            continue
        prev[n] = signal.signal(n, lambda s, _f: sys.exit(128 + s))
    return prev


@dataclass
class Side:
    label: str
    sha: str
    worktree: Path
    home: Path
    port: int = 0
    install: Optional[StudioInstall] = None


def resolve_shas(repo: Path, pr: int, gh_repo: str) -> tuple[str, str, str]:
    # A remote switchboard run (studio_regress/remote_driver.py) has the SHAs already and no gh
    # credentials: STUDIO_REGRESS_PIN_SHAS="<base>:<head>:<head ref>" (both already fetched).
    pinned = os.environ.get("STUDIO_REGRESS_PIN_SHAS")
    if pinned:
        base_sha, head_sha, head_ref = (pinned.split(":", 2) + ["", ""])[:3]
        return base_sha, head_sha, head_ref
    cmd = [
        "gh",
        "pr",
        "view",
        str(pr),
        "--repo",
        gh_repo,
        "--json",
        "headRefOid,headRefName,baseRefName,mergeCommit",
    ]
    meta, errors = None, []
    for env in _gh_envs():
        proc = subprocess.run(cmd, text = True, capture_output = True, env = env)
        if proc.returncode == 0:
            meta = json.loads(proc.stdout)
            break
        errors.append(proc.stderr.strip()[:200])
    if meta is None:
        raise RuntimeError(
            "could not read the PR with any available credential. This is the first "
            "call of the run, so it fails before either install starts:\n  " + "\n  ".join(errors)
        )
    head_sha, head_ref, base_ref = meta["headRefOid"], meta["headRefName"], meta["baseRefName"]
    _sh(["git", "fetch", "origin", base_ref, head_sha], cwd = repo, env = _gh_env())
    base_sha = _sh(
        ["git", "merge-base", head_sha, f"origin/{base_ref}"], cwd = repo, env = _gh_env()
    ).strip()
    # Merged with a merge commit (or fast-forwarded): head is already in the base branch, so the
    # merge base IS head. The PR's "before" is then the merge commit's first parent.
    merge_sha = (meta.get("mergeCommit") or {}).get("oid") or ""
    if base_sha == head_sha and merge_sha:
        _sh(["git", "fetch", "origin", merge_sha], cwd = repo, env = _gh_env(), check = False)
        parent = _sh(
            ["git", "rev-parse", f"{merge_sha}^1"], cwd = repo, env = _gh_env(), check = False
        ).strip()
        if parent and parent != head_sha:
            base_sha = parent
    return base_sha, head_sha, head_ref


def make_worktree(repo: Path, sha: str, dest: Path) -> Path:
    """A DETACHED worktree: these are read-only stages for a screenshot, and a
    named branch would collide with whoever is driving the PR itself."""
    if dest.exists():
        _sh(["git", "worktree", "remove", "--force", str(dest)], cwd = repo, check = False)
        shutil.rmtree(dest, ignore_errors = True)
    dest.parent.mkdir(parents = True, exist_ok = True)
    _sh(["git", "worktree", "add", "--detach", str(dest), sha], cwd = repo)
    return dest


@contextlib.contextmanager
def _parked_ancestor_gitignores(starts: list[Path]):
    """Park any ancestor `.gitignore` that is exactly `*`, for the install pass.

    `studio/setup.sh` walks from `<worktree>/studio/frontend` to `/` and renames
    every such file to `.gitignore._twbuild` so the frontend build is not ignored,
    restoring on EXIT. That is fine for one install and a race for two: both walks
    test and move the SAME shared ancestors (this workspace root carries a bare
    `*`), and the first to finish restores the file while the other is still
    building, which is the condition the rename exists to avoid.

    So park them ONCE here, around both installs. Each install's own walk then
    finds nothing to move and there is nothing to race on. Restored in a finally,
    including when an install raises.
    """
    parked: list[Path] = []
    seen: set[Path] = set()
    for start in starts:
        for ancestor in start.resolve().parents:
            if ancestor in seen:
                continue
            seen.add(ancestor)
            gitignore = ancestor / ".gitignore"
            stashed = ancestor / ".gitignore._uidiff"
            try:
                # Self-heal first. `finally` does not run for SIGKILL, and a run
                # killed mid-install would otherwise leave the file parked for
                # good -- silently changing what every later build ignores.
                if stashed.is_file() and not gitignore.exists():
                    print(
                        f"[install] restoring {gitignore} left parked by an " f"earlier run",
                        flush = True,
                    )
                    stashed.rename(gitignore)
                if gitignore.is_file() and gitignore.read_text().splitlines() == ["*"]:
                    gitignore.rename(stashed)
                    parked.append(gitignore)
            except OSError:
                pass
    try:
        yield
    finally:
        for gitignore in parked:
            try:
                (gitignore.parent / ".gitignore._uidiff").rename(gitignore)
            except OSError:
                print(f"WARNING: could not restore {gitignore}", flush = True)


# Host driver CUDA version -> the torch wheel index install.sh should use. install.sh detects it with
# `timeout 10 nvidia-smi` plus an NVML fallback under one 10 s limit; on a congested driver both
# time out and it silently installs cu126 torch, which has no kernels for Blackwell (sm_100).
_TORCH_INDEX = ((13000, "13.0", "cu130"), (12080, "12.8", "cu128"), (12060, "12.6", "cu126"))


def expected_torch_cuda():
    """(cuda 'X.Y', index URL) for this host's driver, or None (no NVIDIA driver / too old).
    Uses libcuda's cuDriverGetVersion only: no cuInit, no NVML, ~1 ms even when the driver is busy."""
    import ctypes

    try:
        cu = ctypes.CDLL("libcuda.so.1")
        v = ctypes.c_int()
        if cu.cuDriverGetVersion(ctypes.byref(v)) != 0:
            return None
    except (OSError, AttributeError):
        return None
    for floor, cuda, tag in _TORCH_INDEX:
        if v.value >= floor:
            return cuda, f"https://download.pytorch.org/whl/{tag}"
    return None


def torch_cuda_of(home: Path):
    """torch.version.cuda of an install home's main venv, read from torch/version.py (no import)."""
    import re

    for f in sorted(Path(home).glob("unsloth_studio/lib/python*/site-packages/torch/version.py")):
        m = re.search(r"^cuda\b[^=]*=\s*['\"]([\d.]+)['\"]", f.read_text(), re.M)
        if m:
            return m.group(1)
    return None


def install_torch_mismatch(home: Path):
    """Reason string when the install's torch was built for an older CUDA than this host's driver
    selects (the silent cu126 fallback), else None. Hosts without an NVIDIA driver never mismatch."""
    exp = expected_torch_cuda()
    got = torch_cuda_of(home)
    if exp is None or got is None:
        return None
    ver = lambda s: tuple(int(x) for x in s.split(".")[:2])
    if ver(got) < ver(exp[0]):
        return f"torch built for CUDA {got}, host driver selects {exp[0]} ({exp[1]})"
    return None


def _star_gitignore_ancestor(start: Path) -> bool:
    """Whether studio/setup.sh's frontend build would hide an ancestor `.gitignore` holding
    exactly `*` (its `grep -qx '\\*'` walk from <worktree>/studio/frontend up to /)."""
    for ancestor in (start / "studio" / "frontend").resolve().parents:
        try:
            if "*" in (ancestor / ".gitignore").read_text().splitlines():
                return True
        except OSError:
            continue
    return False


def install_side(
    side: Side,
    log_dir: Path,
    serialize: Optional[bool] = True,
    uv_cache: Optional[Path] = None,
    env: Optional[dict] = None,
) -> StudioInstall:
    """`install.sh --local` of side.worktree into side.home.

    serialize: True (default) holds the workspace-wide install lock; None holds it only when an
    ancestor `.gitignore` holding `*` makes concurrent installs race (a tree outside the
    workspace, like the studio_regress shared cache, has none); False never.
    uv_cache: defaults to WORKSPACE/temp/uv_cache.
    env: extra environment for install.sh (UNSLOTH_TORCH_INDEX_URL, UNSLOTH_ZOO_REF pins).
    The frontend build runs on 8 CPUs (frontend_cpu_env).
    """
    side.home.mkdir(parents = True, exist_ok = True)
    log_dir.mkdir(parents = True, exist_ok = True)
    log_path = log_dir / f"install_{side.label.lower()}.log"
    # ONE uv cache for every install, and on the same filesystem as the homes.
    # install.sh otherwise points the cache inside UNSLOTH_STUDIO_HOME, so each
    # home re-downloads the whole torch + CUDA stack from cold: several GB per
    # install, twice per PR. It has to be on this filesystem rather than under
    # ~/.cache, because uv hardlinks wheels within a filesystem and COPIES across
    # a boundary, which is the reason the per-home default exists at all.
    # Content-addressed by wheel hash, so it can change where bytes come from but
    # never which bytes: it cannot serve a build from the wrong SHA.
    uv_cache = Path(uv_cache) if uv_cache else WORKSPACE / "temp" / "uv_cache"
    uv_cache.mkdir(parents = True, exist_ok = True)
    # Streamed, not captured. A 45 minute install that dies used to write no log
    # at all, leaving only the tail of each stream on the exception.
    # One install at a time per workspace: studio/setup.sh hides every ancestor
    # .gitignore holding "*" (the workspace root has one) while the frontend
    # builds, via `mv .gitignore .gitignore._twbuild`, so two concurrent installs
    # under the same root race on that one file and the loser fails with
    # "mv: cannot stat". A warm-cache install is about two minutes.
    if serialize is None:
        serialize = _star_gitignore_ancestor(side.worktree)
    lock_path = WORKSPACE / "temp" / "studio_install.lock"
    lock_path.parent.mkdir(parents = True, exist_ok = True)
    env = {
        **os.environ,
        **(env or {}),
        "UNSLOTH_STUDIO_HOME": str(side.home),
        "UV_CACHE_DIR": str(uv_cache),
    }
    env.update(frontend_cpu_env(env))
    exp = expected_torch_cuda()
    if exp and not env.get("UNSLOTH_TORCH_INDEX_URL"):
        env["UNSLOTH_TORCH_INDEX_URL"] = exp[1]  # skip install.sh's timing-sensitive probe
    with (
        lock_path.open("w") if serialize else contextlib.nullcontext() as lock_fh,
        log_path.open("w") as log,
    ):
        if lock_fh is not None:
            fcntl.flock(lock_fh, fcntl.LOCK_EX)
        proc = subprocess.run(
            ["bash", str(side.worktree / "install.sh"), "--local"],
            cwd = side.worktree,
            env = env,
            stdout = log,
            stderr = subprocess.STDOUT,
            text = True,
            timeout = 60 * 45,
        )
    if proc.returncode != 0:
        raise RuntimeError(
            f"install.sh failed ({proc.returncode}) for {side.label}; full log at {log_path}"
        )
    bad = install_torch_mismatch(side.home)
    if bad:
        raise RuntimeError(
            f"install for {side.label} has the wrong torch build: {bad}; log at {log_path}"
        )
    return StudioInstall(home = side.home, repo = side.worktree, branch = side.sha)


def _same_bytes(a: Path, b: Path) -> bool:
    """Whether two shots are the same image.

    Byte equality rather than a perceptual diff on purpose. Both sides render the same
    scene at the same viewport on the same box, so a real UI difference changes bytes;
    anything subtler than that is not going to read in a GitHub comment anyway. A
    perceptual threshold would need tuning per scene and would be one more thing that can
    silently pass.
    """
    try:
        if a.stat().st_size != b.stat().st_size:
            return False
        return hashlib.sha256(a.read_bytes()).digest() == hashlib.sha256(b.read_bytes()).digest()
    except OSError:
        return False


def _facts_diff(before: dict, after: dict) -> dict:
    """Keys whose value differs between the sides, as ``{key: (before, after)}``.

    A key present on one side only counts as changed, since a scene that could not read
    something on one build is itself the finding.

    Keys starting with `_` are the scene's own bookkeeping, not readings off the
    photographed UI: an ephemeral port, a server-minted id, a per-side video path.
    Those differ between the sides BY CONSTRUCTION, so comparing them would make a
    parity plan fail every correct run, and on an ordinary plan they would satisfy
    the "some fact moved" check without any fact about the UI having moved. Kept in
    `meta.json`, excluded from the comparison.
    """
    changed = {}
    for key in sorted(set(before) | set(after)):
        if key.startswith("_"):
            continue
        b_val, a_val = before.get(key), after.get(key)
        if b_val != a_val:
            changed[key] = (b_val, a_val)
    return changed


def _problems(parity: bool, identical: list[str], changed: dict, facts: dict) -> list[str]:
    """What is wrong with this pair, given what the PR claims.

    Both failure modes produce output that looks exactly like a successful run, so
    they have to be errors rather than a printed reminder: a reviewer reading the PR
    comment cannot tell a real "no visible change" from a missed click, and neither
    can whoever pasted the image.

    Which way the checks point depends on the claim, which is why `parity` lives on
    the plan and not on the command line. An ordinary plan promises a difference, so
    two matching halves are the failure. A parity plan promises the opposite, so the
    guard INVERTS rather than switching off: matching halves are expected, a moved
    fact needs explaining, and the scene that rendered NOTHING on both sides -- the
    same picture as a perfect parity result -- is excluded by requiring the facts to
    show content was actually on screen.
    """
    problems: list[str] = []
    if parity:
        empty = [label for label, side_facts in facts.items() if not side_facts]
        if empty:
            problems.append(
                "parity plan, but the scene returned no facts for: "
                + ", ".join(sorted(empty))
                + ". Two blank halves match too; only content read from the live DOM "
                "separates a real parity result from a scene that rendered nothing."
            )
        if changed:
            problems.append(
                "parity plan, but these scene facts MOVED between the two sides: "
                + ", ".join(sorted(changed))
                + ". Either the PR changes something it claims it does not, or the two "
                "sides were not comparable (traps 11 and 12). Explain it with a "
                "measurement before posting; do not caption around it."
            )
    else:
        if identical:
            problems.append("the two sides are BYTE-IDENTICAL for: " + "; ".join(identical))
        if not changed:
            problems.append("no scene fact differs between BEFORE and AFTER")
    return problems


def resolve_plan(args) -> "ScenePlan":
    """The plan for this run, from the registry or built from the command line.

    A PR nobody has registered used to be unrunnable: `plan_for` raised, and
    `--scene` was only ever an override of a plan that already existed. So the
    first thing anyone did with a NEW PR was edit registry.py, which is the one
    step that cannot be scripted and the one most likely to be skipped in a
    hurry. `--scene` plus `--expect` is now enough to shoot it.

    `--expect` stays MANDATORY in that mode, and that is the whole discipline
    rather than paperwork: it is what turns "the run was clean" into a claim
    somebody can check, and writing it afterwards means writing down whatever
    the pair happened to show.

    Registering the plan is still better, because `verified` is where what was
    actually observed gets recorded. Ad hoc mode is for the first shot.
    """
    registered = None
    try:
        registered = plan_for(args.pr)
    except KeyError:
        if not args.scene:
            raise SystemExit(
                f"PR {args.pr} has no registered scene. Either add a ScenePlan to "
                f"pr_ui_scenes/registry.py, or run it ad hoc with --scene <module> "
                f"--expect '<the difference the shot must show>'. Reuse an existing "
                f"scene where the surface matches: another repo or model is a kwarg, "
                f"not a new file."
            ) from None

    overrides = {}
    if args.scene_kwargs:
        try:
            overrides["kwargs"] = json.loads(args.scene_kwargs)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"--scene-kwargs is not valid JSON: {exc}") from None
    if args.expect:
        overrides["expect"] = args.expect
    if args.scene:
        overrides["scene"] = args.scene
    if args.parity:
        overrides["parity"] = True

    if registered is None:
        if not args.expect:
            raise SystemExit(
                "--expect is required when a PR is not in the registry. Name the "
                "difference the shot MUST show, before running: a scene that quietly "
                "photographs the wrong thing is indistinguishable from one that "
                "worked, and `expect` written afterwards just describes whatever "
                "came out. On a --parity plan it must name POSITIVE content both "
                "halves show, never 'no difference'."
            )
        plan = ScenePlan(
            pr = args.pr,
            scene = args.scene,
            what = args.what or f"ad hoc run of {args.scene}",
            expect = args.expect,
            kwargs = overrides.get("kwargs", {}),
            needs_model = args.needs_model,
            parity = args.parity,
        )
        # Only on this path. `plan_for` already does it for a registered plan, and
        # duplicating it here would also reject the stub scenes the driver's own
        # tests use to exercise everything around the scene.
        scene_file = Path(__file__).resolve().parent / "pr_ui_scenes" / f"{plan.scene}.py"
        if not scene_file.exists():
            raise SystemExit(f"no scene module {plan.scene!r} at {scene_file}")
    else:
        merged = dict(overrides)
        if "kwargs" in merged:
            merged["kwargs"] = {**registered.kwargs, **merged["kwargs"]}
        plan = replace(registered, **merged) if merged else registered

    return plan


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pr", type = int, required = True)
    ap.add_argument(
        "--allow-identical",
        action = "store_true",
        help = "report success even when the two sides are byte-identical and no scene fact "
        "moved. Only for a PR whose visible effect is genuinely absent on this host, "
        "and say so in the comment rather than posting the pair as evidence.",
    )
    ap.add_argument(
        "--scene",
        default = None,
        help = "the scene module to drive. With --expect, this is also how an "
        "UNREGISTERED PR is run without editing registry.py first",
    )
    ap.add_argument(
        "--expect",
        default = None,
        help = "the difference the shot must show, written BEFORE the run. "
        "Required for a PR that is not in the registry",
    )
    ap.add_argument("--what", default = None, help = "the UI surface being photographed")
    ap.add_argument(
        "--scene-kwargs",
        default = None,
        help = 'JSON merged over the plan\'s kwargs, e.g. \'{"repo": "org/x"}\'',
    )
    ap.add_argument(
        "--needs-model",
        action = "store_true",
        help = "this scene loads real weights (slow, needs a GPU)",
    )
    ap.add_argument(
        "--parity",
        action = "store_true",
        help = "the PR claims the UI does NOT change: inverts the guards so "
        "matching halves pass and a moved fact fails",
    )
    ap.add_argument(
        "--preflight",
        action = "store_true",
        help = "run every cheap check and stop before the installs",
    )
    ap.add_argument(
        "--password",
        default = None,
        help = "the Studio password to log in with, as every scene's own entry point takes. "
        "Needed for a home rotated by something other than this driver, since nothing "
        "short of a reinstall can change a rotated home's credential. Left unset, each "
        "home gets its OWN minted password, which is what makes the login prove which "
        "install answered; ONE password shared by both sides cannot tell home_before "
        "from home_after apart (trap 12).",
    )
    ap.add_argument("--repo", type = Path, default = DEFAULT_REPO_DIR)
    ap.add_argument("--gh-repo", default = "unslothai/unsloth")
    ap.add_argument("--root", type = Path, default = None)
    ap.add_argument("--skip-install", action = "store_true")
    ap.add_argument(
        "--base-ref",
        default = None,
        help = "override the BEFORE side's commit. For a PR whose visible effect is gated "
        "behind another open PR, the merge base shows nothing and the honest pair is "
        "'that other PR' against 'that other PR plus this one'. Say which refs were "
        "used in the comment: this is no longer the PR's own merge base.",
    )
    ap.add_argument(
        "--head-ref",
        default = None,
        help = "override the AFTER side's commit. See --base-ref.",
    )
    ap.add_argument(
        "--studio-env",
        action = "append",
        default = [],
        metavar = "KEY=VALUE",
        help = "env applied to BOTH launched Studios (not to the installs). Use it to point "
        "a scene at an isolated HF cache: this box's XDG_CACHE_HOME holds a shared "
        "cache of dozens of repos, and Studio scans it IN ADDITION to HF_HOME, so a "
        "scene about cache contents otherwise photographs whatever else is on the box "
        "and its numbers move when another session downloads something.",
    )
    args = ap.parse_args()
    studio_env = dict(kv.split("=", 1) for kv in args.studio_env)

    plan = resolve_plan(args)
    scene_name = args.scene or plan.scene
    # Resolved, always. install.sh is run with cwd inside the worktree and only passes
    # UNSLOTH_STUDIO_HOME through, so a relative --root installs a whole Studio under
    # <worktree>/<root>/home_before and the launcher then cannot find the CLI where this
    # script is looking for it. The install itself succeeds, which is what makes it slow
    # to spot.
    root = (args.root or (WORKSPACE / "outputs" / f"ui_diff_{args.pr}")).resolve()
    root.mkdir(parents = True, exist_ok = True)

    base_sha, head_sha, head_ref = resolve_shas(args.repo, args.pr, args.gh_repo)
    # Overrides are resolved through the SAME repo, so a ref that does not exist locally fails
    # here rather than half way through an install, and both sides print what they actually are.
    overridden = False
    for attr, label in (("base_ref", "base"), ("head_ref", "head")):
        ref = getattr(args, attr)
        if not ref:
            continue
        overridden = True
        sha = _sh(["git", "rev-parse", "--verify", f"{ref}^{{commit}}"], cwd = args.repo).strip()
        if label == "base":
            base_sha = sha
        else:
            head_sha, head_ref = sha, ref
    print(
        f"PR #{args.pr} {head_ref}\n  base {base_sha}\n  head {head_sha}"
        f"\n  scene {scene_name}\n  EXPECT {plan.expect}\n",
        flush = True,
    )
    if overridden:
        print(
            "  NOTE: refs overridden, this is not the PR's own merge base. Say so in the "
            "comment.\n",
            flush = True,
        )

    sides = [
        Side(
            "BEFORE", base_sha, WORKSPACE / "temp" / f"uidiff{args.pr}_before", root / "home_before"
        ),
        Side("AFTER", head_sha, WORKSPACE / "temp" / f"uidiff{args.pr}_after", root / "home_after"),
    ]
    # Ports are chosen per side JUST BEFORE that side launches, not both up front.
    # A bind test proves a port is free at the instant it runs and reserves nothing;
    # picking both here leaves the AFTER port unguarded across the BEFORE install and
    # scene, which is minutes, and anything that starts a Studio meanwhile takes it.
    # (Observed: an unrelated probe of this same workspace grabbed the reserved port
    # and the login identity check was what noticed.)

    # Imported BEFORE the installs. A scene with a syntax error, a bad import or a
    # call the kit cannot take is otherwise discovered after two builds.
    scene = importlib.import_module(f"pr_ui_scenes.{scene_name}")
    if not hasattr(scene, "drive"):
        raise SystemExit(f"scene {scene_name} has no drive() coroutine")

    if args.preflight:
        print(
            "PREFLIGHT OK. Everything checkable without building is checked:\n"
            f"  scene      {scene_name}.drive imports and is callable\n"
            f"  kwargs     {json.dumps(plan.kwargs)}\n"
            f"  parity     {plan.parity}\n"
            f"  base/head  resolved, worktrees not yet made\n"
            f"  homes      {sides[0].home}\n"
            f"             {sides[1].home}\n"
            "Not checked here: that the scene drives the right surface, which only "
            "the composite and `expect` can tell you.\n"
            "Drop --preflight to build and shoot.",
            flush = True,
        )
        return 0

    results: dict[str, list[Path]] = {}
    facts: dict[str, dict] = {}

    # PASS 1: install. Both sides at once, because each installs into its own home
    # from its own worktree and the two do not interact. This is the whole cost of
    # the exercise, so overlapping it roughly halves a cold run.
    #
    # Driving stays SEQUENTIAL below, deliberately. Two Studios photographed at the
    # same time share this box's GPU and memory, and trap 11 is exactly that: a
    # side that measured different hardware is not comparable to the other, and an
    # uncontrolled difference looks precisely like a finding.
    #
    # Nothing here weakens the staleness gate: the stamp is still cleared before an
    # install and written only after that side's install.sh exits 0.
    to_install = []
    for side in sides:
        stamp = side.home / ".uidiff_sha"
        # PER SIDE, not all-or-nothing. A scene that fails after BEFORE installed used to
        # force a full reinstall of BEFORE too, because AFTER had no stamp yet. Reuse
        # whichever side is already built AT THE RIGHT SHA and install only the other.
        # The stamp is what makes reuse safe. A PR under active review moves, and a
        # home built from an older SHA yields a real screenshot of the WRONG code
        # under the right label -- the single most misleading output this tool can
        # make, because nothing about the image looks wrong. So reuse is allowed
        # only on an exact SHA match; a stale home is rebuilt, never photographed.
        built = stamp.read_text().strip() if stamp.exists() else None
        if args.skip_install and built == side.sha:
            # The binary lives under the home, so a reused side needs no worktree.
            print(f"[{side.label}] reusing home built at {side.sha[:9]}", flush = True)
            side.install = StudioInstall(home = side.home, repo = side.worktree, branch = side.sha)
        else:
            if args.skip_install:
                print(
                    f"[{side.label}] --skip-install ignored: home was built from "
                    f"{built or 'an unrecorded SHA'}, PR needs {side.sha[:9]}",
                    flush = True,
                )
            # Serialised: concurrent `git worktree add` races on .git/worktrees, and
            # it takes seconds against an install's tens of minutes.
            make_worktree(args.repo, side.sha, side.worktree)
            # Cleared FIRST: an install that dies half way leaves a home that boots
            # but serves a mix, and a stamp left over from the previous build would
            # bless exactly that on the next --skip-install run.
            stamp.unlink(missing_ok = True)
            to_install.append(side)

    if to_install:
        labels = ", ".join(s.label for s in to_install)
        print(f"[install] {labels} concurrently ...", flush = True)
        with _parked_ancestor_gitignores([s.worktree for s in to_install]):
            with ThreadPoolExecutor(
                max_workers = min(len(to_install), MAX_PARALLEL_INSTALLS)
            ) as pool:
                futures = {pool.submit(install_side, s, root / "logs"): s for s in to_install}
                for future in as_completed(futures):
                    side = futures[future]
                    side.install = future.result()  # re-raises with the log path
                    (side.home / ".uidiff_sha").write_text(side.sha)
                    print(f"[{side.label}] built {side.sha[:9]}", flush = True)

    # Both Studios run in their own sessions and outlive this process, so they are
    # stopped here whatever happens. Without this every run leaked two servers, and
    # a host collected 107 of them (88 GB RAM, 55 GB VRAM).
    launched: list[StudioInstall] = []
    prev = _exit_on_term()
    try:
        for side in sides:
            taken = {s.port for s in sides if s.port}
            side.port = next(p for p in pick_free_ports(3) if p not in taken)
            launched.append(side.install)  # before the launch: it can raise after spawning
            # password_timeout_s=0: `studio_session` reads the bootstrap file itself, so
            # the launcher's poll for it is dead work here, and it runs BEFORE the
            # healthz wait. On a reused home Studio has already deleted that file and
            # never reprints the line, so it burned the full 30s every single launch.
            launch_studio(
                side.install,
                side.port,
                root / f"{side.label.lower()}_studio.log",
                extra_env = studio_env or None,
                password_timeout_s = 0,
            )
            # Proves the server answering is OURS: a stale Studio on this port has a
            # different password and fails here, rather than silently posing for the
            # photograph.
            session = studio_session(
                f"http://127.0.0.1:{side.port}",
                side.home,
                password_for(root, side.home, args.password),
            )
            print(f"[{side.label}] up on :{side.port}, identity verified", flush = True)
            wiped = _reset_threads(session)
            if wiped:
                print(
                    f"[{side.label}] cleared {wiped} leftover chat thread(s) from this home",
                    flush = True,
                )

            out_dir = root / side.label.lower()
            out_dir.mkdir(parents = True, exist_ok = True)
            shots, f = asyncio.run(scene.drive(session, out_dir, side.label, **plan.kwargs))
            results[side.label], facts[side.label] = shots, f
            print(f"[{side.label}] {len(shots)} shots  facts={json.dumps(f)[:300]}", flush = True)
    finally:
        for n in prev:  # a second SIGTERM must not cut the stops short
            signal.signal(n, signal.SIG_IGN)
        try:
            for inst in launched:
                if inst is not None and not stop_studio(inst):
                    print(
                        f"WARNING: Studio on :{inst.port} (session {inst.pid}) is still "
                        f"running after SIGKILL",
                        flush = True,
                    )
        finally:
            for n, handler in prev.items():
                signal.signal(n, signal.SIG_DFL if handler is None else handler)

    before, after = results["BEFORE"], results["AFTER"]
    if len(before) != len(after):
        raise RuntimeError(
            f"scene took {len(before)} shots BEFORE and {len(after)} AFTER; the flows "
            "diverged, so the pairs are not comparable"
        )
    combined = root / "combined"
    combined.mkdir(parents = True, exist_ok = True)
    pairs = []
    identical: list[str] = []
    for i, (b, a) in enumerate(zip(before, after)):
        out = combined / (
            f"pr{args.pr}_before_after.png" if len(before) == 1 else f"pr{args.pr}_pair_{i:02d}.png"
        )
        hstack_images(b, a, out, label_left = "BEFORE", label_right = "AFTER")
        pairs.append(out)
        if _same_bytes(b, a):
            identical.append(f"{b.name} == {a.name}")

    (root / "meta.json").write_text(
        json.dumps(
            {
                "pr": args.pr,
                "head_ref": head_ref,
                "base_sha": base_sha,
                "head_sha": head_sha,
                "scene": scene_name,
                "expect": plan.expect,
                "parity": plan.parity,
                "facts": facts,
                "studio_env": studio_env,
                "pairs": [str(p) for p in pairs],
            },
            indent = 2,
        )
    )

    print("\npairs:")
    for p in pairs:
        print(" ", p)
    print(f"\nEXPECTED: {plan.expect}")
    print("OBSERVED: BEFORE", json.dumps(facts["BEFORE"]))
    print("          AFTER ", json.dumps(facts["AFTER"]))
    print("\nFACTS DIFF (keys whose value moved between the two sides):")
    changed = _facts_diff(facts["BEFORE"], facts["AFTER"])
    if changed:
        for key, (b_val, a_val) in changed.items():
            print(f"  {key}: {json.dumps(b_val)} -> {json.dumps(a_val)}")
    else:
        print("  (none)")

    # Both failure modes below produce output that looks exactly like a successful run,
    # so they have to be errors: a reviewer reading the PR comment cannot tell a real
    # "no visible change" from a missed click, and neither can whoever pasted the image.
    # Which way the checks point depends on what the PR claims, so the plan says, and
    # the operator has nothing to remember at the command line.
    problems = _problems(plan.parity, identical, changed, facts)
    if problems:
        print("\nFAILED, and this is a result rather than a crash:")
        for p in problems:
            print(f"  - {p}")
        if not plan.parity:
            print(
                "\nBefore concluding the PR changes nothing, rule out the traps in\n"
                "pr_ui_evidence_workflow.md: a missed click leaving the panel on its old\n"
                "selection, the wrong dropdown opening, a home built from a stale SHA, or a\n"
                "port answered by someone else's Studio. If the PR really has no visible\n"
                "effect, it did not need a scene. Mark the plan `parity=True` if that is the\n"
                "claim being evidenced, or re-run with --allow-identical to keep the output."
            )
        if not args.allow_identical:
            return 1
        print("\n--allow-identical given, so reporting success regardless.")

    if plan.parity:
        print(
            "\nNow OPEN the pair and confirm BOTH halves show the content `expect` names. "
            "The checks above prove the two sides agree and that the scene read something "
            "off the page; they cannot tell you it read the RIGHT thing. Only the "
            "composite and `expect` do that."
        )
    else:
        print(
            "\nNow OPEN the pair and confirm it shows the expected difference. The checks "
            "above catch two sides that are identical; they cannot tell you that a real "
            "difference is the RIGHT difference. Only the composite and `expect` do that."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
