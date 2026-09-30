"""Run selected journeys on the PR merge base (before) and head (after), then diff.

    python studio_regress.py run --pr 11606                      # switchboard selection
    python studio_regress.py run --pr 11606 --only auth,crawl --tier fast
    python studio_regress.py run --home-before H --home-after H --only auth,settings,crawl
                                                                 # A/A null control, no PR
    python studio_regress.py run --base-url http://127.0.0.1:8888 --password P --only settings
                                                                 # dev: one side, running Studio

Installs: --home-before/--home-after reuse existing install homes; with --pr and no homes,
each side's SHA is installed once per HOST into the shared cache (cache.py:
$STUDIO_REGRESS_SHARED_DIR, default /mnt/disks/unslothai/shared/studio-regress-cache), else per
workspace into temp/studio_regress/installs/<sha> (flock-guarded, `.uidiff_sha` stamp; reused by
every later run / tmux session / user). Base and head install in parallel; with no journeys
selected (external / Core only) the base installs only when an external suite fails on head.
Each side then gets a FRESH state home (engine.state_home) so auth / db / settings start
identical on both sides. Per-phase wall times: `phase ...` log lines, report.json timings.phases.

Order per side: auth (if selected; it needs the bootstrap state) -> other journeys in
switchboard dependency order -> crawl last. One fresh browser context per journey. Journeys whose
module sets HEAD_FIRST (upstream_ci: pass / fail suites, no screenshots) skip the base side; after
the head side, the base runs only the steps head did not pass (a head pass diffs as SAME).

Scheduling: a local run of both sides defaults to --scheduler parallel (scheduler.py): (side x
journey-chain) arms plus one unit per external / Core target, externals and Core overlapping the
Studio sides, a crashed unit VOIDing only its own steps. The default --instance per_unit gives
every arm its own fresh Studio (home state/<root>-iNN, a port of the run's digit count), so both
sides of a chain start together; the home token and port digits are masked glyph-exactly in the
shots and replaced by fixed text in DOM / facts. The after crawl waits for the before crawl (it
replays its control list); HEAD_FIRST chains run head then base. --plan prints the schedule without
installing anything; --jobs N / --gpu-jobs N cap concurrency. --scheduler sequential (--sequential,
also the default for single-side, --base-url and nested isolated runs) is the order above.

Output: <root>/{before,after}/<journey>/<step>.{png,dom.json,facts.json}, manifest.json,
<side>/crawl_manifest/<route>.json, report.json (contract schema), summary.md.

Core targets (kind job / regression, core.py) need no Studio install: they compare the PR merge
base with the head in the Core interpreter (--core-python) and add "<target>/run" steps. When
only Core targets are selected, no Studio is installed. They need --pr and --side both.

    python studio_regress.py run --pr 1373 --gh-repo unslothai/unsloth-zoo   # Core only
    python studio_regress.py run --pr 5123 --only jobs/sft.py,regression_smoke

Confirm (confirm.py): targets with a FAIL_HEAD / VISUAL_DIFF step (or a retryable VOID) rerun once on
both arms under <root>/confirm; a result that does not reproduce becomes FLAKY (both attempts kept).
--no-confirm skips it.

Exit: 0 no regression and no UI change, 3 UI changed (VISUAL/DOM/DIVERGED) without a
functional regression, 1 functional regression (FAIL_HEAD), 2 VOID (install/launch failed,
selected journeys produced no steps, or a Core target proved nothing).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import importlib
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
sys.path.insert(0, str(SCRIPTS))

from studio_regress import cache, confirm, core, coverage, diff, engine, provenance, selection  # noqa: E402
from studio_regress.contract import SUITE_VERSION, Ctx  # noqa: E402

WS = Path(os.environ.get("WORKSPACE") or SCRIPTS.parent.parent.parent)
INSTALLS = WS / "temp" / "studio_regress" / "installs"
EXIT = {"clean": 0, "regression": 1, "void": 2, "ui_changed": 3}


def _log(msg):
    print(f"[studio_regress] {msg}", file = sys.stderr, flush = True)


# ------------------------------------------------------------------ journey registry
def load_journeys():
    """name -> (Journey, module) for every journeys/*.py exposing JOURNEY, plus crawl."""
    out = {}
    for f in sorted((HERE / "journeys").glob("*.py")):
        if f.name.startswith("_"):
            continue
        try:
            mod = importlib.import_module(f"studio_regress.journeys.{f.stem}")
        except Exception as e:  # a broken journey must not take the whole suite down
            _log(f"journey module {f.stem} failed to import: {e}")
            continue
        j = getattr(mod, "JOURNEY", None)
        if j is not None:
            out[j.name] = (j, mod)
    cj = coverage.crawl_journey()
    out[cj.name] = (cj, coverage)
    return out


def order(names, journeys, data):
    """auth first, deps before dependents, crawl last; unknown names dropped (logged)."""
    deps = {t["name"]: t.get("deps") or [] for t in data["target"]}
    names = [n for n in names if n in journeys]
    out, seen = [], set()

    def visit(n):
        if n in seen:
            return
        seen.add(n)
        for d in deps.get(n, []):
            if d in journeys:
                visit(d)
        out.append(n)

    for n in names:
        visit(n)
    out.sort(key = lambda n: (0 if n == "auth" else 2 if n == "crawl" else 1))
    return out


# ------------------------------------------------------------------ installs
def ensure_install(
    repo: Path,
    sha: str,
    log_dir: Path,
    info = None,
) -> Path:
    """Install `sha` once (host-shared cache, else INSTALLS/<sha>) and return its home; the entry
    stays held (gc skips it) until this process exits. `info` receives hit / built / waited."""
    return cache.ensure_install(repo, sha, log_dir, info = info)


def source_dir(install_home):
    """The unsloth source checkout an install home was built from (for upstream tests)."""
    return cache.source_dir(install_home)


class Phases:
    """Wall time per phase: logged as `phase <name> <s>s` and kept for report.json."""

    def __init__(self):
        import threading
        self.t0, self.s = time.time(), {}
        self._lock = threading.Lock()  # the parallel scheduler adds phases from worker threads

    @contextlib.contextmanager
    def __call__(self, name):
        t = time.time()
        try:
            yield
        finally:
            self.add(name, time.time() - t)

    def add(self, name, secs):
        with self._lock:
            self.s[name] = round(self.s.get(name, 0) + secs, 1)
        _log(f"phase {name} {self.s[name]}s")

    def done(self):
        self.s["total"] = round(time.time() - self.t0, 1)
        return self.s


_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def _follow_install(
    log_path,
    side,
    stop,
    poll_s = 5.0,
    heartbeat_s = 120.0,
    log = None,
):
    """Mirror install.sh's step lines into the run log as `install <side> +<s>s: <step>` (the full
    output stays in install_<sha>.log), plus a heartbeat while a step runs long, so a 10+ minute
    install does not look like a hung run."""
    log = log or _log
    t0 = last = time.time()
    pos, seen, buf = 0, set(), ""
    while True:
        done = stop.wait(poll_s)
        try:
            with open(log_path, "rb") as fh:
                fh.seek(pos)
                chunk = fh.read()
                pos += len(chunk)
        except OSError:
            chunk = b""
        buf += chunk.decode(errors = "replace")
        *lines, buf = re.split(r"[\r\n]", buf)
        for ln in lines:
            ln = " ".join(_ANSI.sub("", ln).split())
            if not ln or ln.startswith("/") or ln in seen or len(ln) > 200:
                continue
            seen.add(ln)
            last = time.time()
            log(f"install {side} +{last - t0:.0f}s: {ln}")
        if done:
            return
        if time.time() - last >= heartbeat_s:
            last = time.time()
            log(f"install {side} +{last - t0:.0f}s: still running (tail {log_path})")


def install_sides(
    repo,
    shas,
    root,
    phases,
    meta,
    base_sha = None,
    zoo = None,
):
    """{side: home} for {side: sha}, both at once (one per SHA; the shared cache dedups across
    sessions, and its builds need no workspace install lock). With STUDIO_REGRESS_CLONE_DELTA=1 the
    head may clone the merge-base install, so the base goes first.

    Both sides get the same pins: the host driver's torch index, and one unsloth-zoo commit
    (install.sh --local installs zoo from git main, so installs made hours apart differ): the zoo
    of a cached base install, else main now."""
    from concurrent.futures import ThreadPoolExecutor

    index = cache.expected_index()
    if zoo is None:
        ref = shas.get("before") or base_sha
        hit = cache.lookup(ref, index) if ref else None
        zoo = (hit[2].get("zoo") if hit else None) or cache.resolve_zoo_main()
    meta.update(torch_index = index, zoo = zoo)
    infos = {s: {} for s in shas}

    def one(side):
        import threading

        t = time.time()
        stop = threading.Event()
        tail = threading.Thread(
            target = _follow_install,
            daemon = True,
            args = (root / "logs" / f"install_{shas[side][:9].lower()}.log", side, stop),
        )
        tail.start()
        try:
            donors = (base_sha,) if base_sha and shas[side] != base_sha else ()
            home = cache.ensure_install(
                repo,
                shas[side],
                root / "logs",
                info = infos[side],
                donors = donors,
                zoo = zoo,
                torch_index = index,
            )
        finally:
            stop.set()
            tail.join(timeout = 30)
        phases.add(f"install_{side}", time.time() - t)
        return home

    uniq = {}
    for s, sha in shas.items():  # A/A of one SHA: one install
        uniq.setdefault(sha, s)
    workers = 1 if cache.clone_delta_enabled() else max(1, len(uniq))
    uniq = dict(sorted(uniq.items(), key = lambda kv: kv[1] != "before"))  # base first
    with ThreadPoolExecutor(max_workers = workers) as pool:
        futs = {sha: pool.submit(one, s) for sha, s in uniq.items()}
        homes = {s: str(futs[sha].result()) for s, sha in shas.items()}
    meta.setdefault("installs", {}).update({s: infos[uniq[shas[s]]] for s in shas})
    for s in shas:  # one line per side: which torch / zoo it really got
        m = infos[uniq[shas[s]]].get("meta") or {}
        _log(
            f"install {s}: torch index {m.get('torch_index')} (torch CUDA {m.get('torch_cuda')}), "
            f"zoo {str(m.get('zoo'))[:12]}"
        )
    for s in shas:
        i = infos[uniq[shas[s]]]
        _log(
            f"install {s} {shas[s][:9]}: {i.get('result')} in {i.get('s')}s "
            f"({'shared' if i.get('shared') else 'workspace'} cache {i.get('store')})"
        )
    return homes


class LazyHomes(dict):
    """homes whose missing sides install on first .get() / []: external targets run head first and
    touch the base install only when head fails."""

    def __init__(self, homes, pending):
        super().__init__(homes)
        self._pending = pending  # side -> zero-arg installer returning a home

    def _fill(self, side):
        fn = self._pending.pop(side, None)
        if fn is not None:
            dict.__setitem__(self, side, fn())

    def get(
        self,
        side,
        default = None,
    ):
        self._fill(side)
        return dict.get(self, side, default)

    def __getitem__(self, side):
        self._fill(side)
        return dict.__getitem__(self, side)


def _git_safe_env(dirs):
    """GIT_CONFIG_* adding safe.directory for source trees another user built (shared cache),
    so git (setuptools-scm, suites calling `git rev-parse`) does not refuse them."""
    dirs = [str(d) for d in dirs if d]
    if not dirs:
        return {}
    n = int(os.environ.get("GIT_CONFIG_COUNT") or 0)
    env = {}
    for i, d in enumerate(dirs):
        env[f"GIT_CONFIG_KEY_{n + i}"], env[f"GIT_CONFIG_VALUE_{n + i}"] = "safe.directory", d
    env["GIT_CONFIG_COUNT"] = str(n + len(dirs))
    return env


# ------------------------------------------------------------------ one side
def _models(
    data,
    keys,
    extra = None,
):
    """ctx.models for one journey: switchboard [models] specs by key ({"repo", "variant"} for
    GGUF / training fixtures), a local snapshot dir for `local_snapshot = true` fixtures (tiny
    diffusers pipelines load from a path), plus run-provided `_`-keys (`_studio_home`,
    `_studio_log`)."""
    out = dict(extra or {})
    for k in keys:
        spec = (data.get("models") or {}).get(k)
        if spec is None:
            continue
        if spec.get("local_snapshot"):
            from studio_regress.journeys import _diffusion
            try:
                out[k] = _diffusion.ensure_tiny(spec["repo"])
            except Exception as e:  # the journey falls back to its own ensure_* and says why
                _log(f"fixture {k}: local snapshot of {spec['repo']} failed: {e}")
        else:
            out[k] = dict(spec)
    return out


def isolated_names(names, journeys):
    """Journeys whose module sets ISOLATED = True run in a nested run.py inside
    studio_regress.isolation (Studio, tool children, Chromium and driver in one network
    namespace), never in the side's shared Studio. Inside that nested run there is nothing left
    to isolate."""
    if os.environ.get("STUDIO_REGRESS_ISOLATION"):
        return []
    return [n for n in names if getattr(journeys[n][1], "ISOLATED", False)]


def run_isolated(
    side,
    home,
    root,
    names,
    online = False,
    timeout = None,
    on_proc = None,
):
    """One side of the ISOLATED journeys: `run.py --only N --side S` inside the isolation
    wrapper, writing into the same <root>/<side>/ layout. Returns the wrapper's exit code.
    timeout: stop the wrapper's process group after that many seconds (exit -9); on_proc(Popen):
    the scheduler's cancel hook."""
    tr = WS / "temp" / "studio_regress"
    browsers = Path(os.environ.get("PLAYWRIGHT_BROWSERS_PATH") or WS / "temp" / "pw_browsers")
    cmd = [
        sys.executable,
        str(HERE / "run.py"),
        "--only",
        ",".join(names),
        "--side",
        side,
        f"--home-{side}",
        str(home),
        "--root",
        str(root),
        "--no-prefetch",
    ] + (["--online"] if online else [])
    rw = [root, tr / "state", tr / "home", tr / "hf_home"]
    src = source_dir(home)
    ro = [Path(home), SCRIPTS, tr / "src", Path(sys.prefix), browsers] + ([src] if src else [])
    for d in rw:
        d.mkdir(parents = True, exist_ok = True)
    argv = [
        sys.executable,
        "-m",
        "studio_regress.isolation",
        "--mode",
        os.environ.get("STUDIO_REGRESS_ISOLATION_MODE", "auto"),
    ]
    argv += [f"--rw={d}" for d in rw] + [f"--ro={d}" for d in ro if d.exists()]
    env = {**os.environ, "PLAYWRIGHT_BROWSERS_PATH": str(browsers)}
    with open(root / "logs" / f"isolated_{side}.log", "a") as log:
        if timeout is None and on_proc is None:
            return subprocess.call(
                argv + ["--"] + cmd, cwd = SCRIPTS, env = env, stdout = log, stderr = subprocess.STDOUT
            )
        from studio_regress import plat

        proc = subprocess.Popen(
            argv + ["--"] + cmd,
            cwd = SCRIPTS,
            env = env,
            stdout = log,
            stderr = subprocess.STDOUT,
            stdin = subprocess.DEVNULL,
            **plat.group_kwargs(),
        )
        if on_proc is not None:
            on_proc(proc)
        try:
            return proc.wait(timeout = timeout)
        except subprocess.TimeoutExpired:
            log.write(f"\n[studio_regress] isolated run over {timeout:.0f}s: stopped\n")
            plat.stop_tree(proc.pid, grace_s = 20, proc = proc)
            return -9


async def run_side(
    side,
    install_home,
    root,
    names,
    journeys,
    data,
    password,
    base_url = None,
    extra_env = None,
):
    """The sequential path: launch a fresh-state Studio for `side` (unless base_url), run journeys,
    stop. Same state path and port on both sides (they show in the UI)."""
    from studio_regress.scheduler import Handle, TokenState

    timings, port, home = {}, None, None
    env = dict(extra_env or {})
    studio_log = (
        root
        / "logs"
        / (
            f"studio_{side}_isolated.log"
            if os.environ.get("STUDIO_REGRESS_ISOLATION")
            else f"studio_{side}.log"
        )
    )
    for n in names:
        env.update(getattr(journeys[n][1], "STUDIO_ENV", {}) or {})
    if base_url is None:
        home = engine.state_home(
            install_home, WS / "temp" / "studio_regress" / "state" / root.name
        )  # same path both sides: path text shows in the UI
        port = engine.PORTS.take(seed = root.name)  # same port both sides: it shows in the UI
        t0 = time.time()
        try:
            inst = engine.launch(home, port, studio_log, env)
        except BaseException:
            engine.PORTS.release(port)
            raise
        if inst.port != port:  # launch moves to a fresh port if this one was taken
            engine.PORTS.release(port)
        port = inst.port
        base_url = f"http://127.0.0.1:{port}"
        timings["_launch"] = round(time.time() - t0, 1)
        bootstrap = inst.bootstrap_password
    else:
        bootstrap = None
    handle = Handle(
        base_url = base_url,
        home = str(home) if home else None,
        bootstrap_password = bootstrap,
        studio_log = str(studio_log) if home else None,
        install_home = str(install_home or ""),
        password = password,
        tokens_fn = TokenState(base_url, str(home) if home else None, password, log = _log),
    )
    try:
        await drive_journeys(side, handle, root, names, journeys, data, timings)
    finally:
        if port:
            engine.stop(port)
            engine.PORTS.release(port)
    return timings


async def drive_journeys(
    side,
    handle,
    root,
    names,
    journeys,
    data,
    timings = None,
    browser = None,
):
    """Run `names` in order against the Studio behind `handle` (scheduler.Handle): a fresh browser
    context per journey, closed when the journey ends whatever happens; auth only while the
    bootstrap password is unrotated. browser: the side's Chromium (scheduler.SideBrowser, one per
    run and side); None launches one for this call (the sequential path: one per side)."""
    if browser is None:
        from playwright.async_api import async_playwright
        async with async_playwright() as pw:
            own = await pw.chromium.launch()
            timings = {} if timings is None else timings
            timings["_browser_warm"] = await engine.fixture.warm_browser(
                own
            )  # first frame, outside steps
            try:
                return await drive_journeys(
                    side, handle, root, names, journeys, data, timings, browser = own
                )
            finally:
                with contextlib.suppress(Exception):
                    await own.close()

    out = root / side
    out.mkdir(parents = True, exist_ok = True)
    timings = {} if timings is None else timings
    base_url, home, install_home = handle.base_url, handle.home, handle.install_home
    bootstrap = handle.bootstrap_password
    state_common = {
        "home": str(home) if home else None,
        "install_home": str(install_home or ""),
        "src": str(source_dir(install_home)) if install_home else None,
        "password": handle.password,
        "bootstrap_password": bootstrap,
        "side": side,
        "root": str(root),
    }
    if (
        handle.instance
    ):  # per-unit Studio: captures neutralise its token / port; extra homes hang off it
        state_common.update(_instance = handle.instance, state_base = str(home))
    run_models = {
        "_studio_home": str(home) if home else None,
        "_studio_log": str(handle.studio_log) if home else None,
    }

    async def tokens():
        # login / rotation are blocking HTTP (minutes on a loaded host): off the loop, which other
        # units of this side may share
        return await asyncio.to_thread(handle.tokens)

    for n in names:
        j, _mod = journeys[n]
        t0 = time.time()
        if n == "auth":
            if bootstrap is None:
                _log(f"{side}: auth skipped (no bootstrap state on a reused Studio)")
                continue
            bctx = await engine.fixture.new_context(browser)
            try:
                page = await engine.fixture.prepare_page(await bctx.new_page())
                ctx = Ctx(
                    page = page,
                    base_url = base_url,
                    api = None,
                    side = side,
                    out_dir = str(out),
                    models = _models(data, []),
                    state = dict(state_common),
                )
                await engine.run_journey(j, ctx, out)
            finally:
                with contextlib.suppress(Exception):
                    await bctx.close()
        elif n == "crawl":
            await _run_crawl(browser, base_url, await tokens(), j, side, out, state_common)
        else:
            toks = await tokens()
            tgt = next((t for t in data["target"] if t["name"] == n), {})
            bctx = await engine.authed_context(browser, base_url, toks)
            api = None
            try:
                page = await engine.fixture.prepare_page(await bctx.new_page())
                api = engine.ApiClient(base_url, toks["access_token"])
                # fixture snapshots may download: off the loop, which other units of this side share
                models = await asyncio.to_thread(_models, data, tgt.get("models") or [], run_models)
                ctx = Ctx(
                    page = page,
                    base_url = base_url,
                    api = api,
                    side = side,
                    out_dir = str(out),
                    models = models,
                    state = dict(state_common),
                )
                await engine.run_journey(j, ctx, out)
            finally:
                if api is not None:
                    with contextlib.suppress(Exception):
                        await api.aclose()
                with contextlib.suppress(Exception):
                    await bctx.close()
        timings[n] = round(time.time() - t0, 1)
        _log(f"{side}: {n} {timings[n]}s")
    return timings


def head_first_names(names, journeys):
    """Journeys whose module sets HEAD_FIRST = True (pass / fail suites, no screenshots)."""
    return [n for n in names if getattr(journeys[n][1], "HEAD_FIRST", False)]


def head_first_base_plan(root, names, journeys):
    """After the head side ran: mark every HEAD_FIRST step head passed as not run on the base
    (diff: SAME), and return {name: (Journey with only the steps head did not pass, module)} for
    the base to run."""
    import dataclasses

    rerun = {}
    for n in names:
        j, mod = journeys[n]
        need = []
        for st in j.steps:
            try:
                fa = json.loads((root / "after" / n / f"{st.id}.facts.json").read_text())
            except (OSError, ValueError):
                fa = None
            if fa and fa.get("_status") in ("ok", diff.SKIPPED_NOT_TOUCHED):
                d = root / "before" / n
                d.mkdir(parents = True, exist_ok = True)
                skipped = fa["_status"] == diff.SKIPPED_NOT_TOUCHED  # same diff, same skip
                (d / f"{st.id}.facts.json").write_text(
                    json.dumps(
                        {
                            "_step": st.id,
                            "_status": fa["_status"] if skipped else diff.HEAD_PASSED,
                            **({"_reason": fa.get("_reason")} if skipped else {}),
                        },
                        indent = 1,
                    )
                )
            else:
                need.append(st)
        if need:
            rerun[n] = (dataclasses.replace(j, steps = tuple(need)), mod)
    return rerun


# ------------------------------------------------------------------ base-side reuse
REUSE_FILE = ".reuse_key.json"


def suite_version():
    """sha256 over every file that decides what a side does or records: this package (journeys,
    engine, switchboard.toml) and the helpers it drives (pr_ui_scenes/_common.py, studio_test_kit)."""
    import hashlib

    h = hashlib.sha256()
    files = sorted(
        p
        for p in HERE.rglob("*")
        if p.suffix in (".py", ".toml", ".json")
        and "__pycache__" not in p.parts
        and not p.name.startswith("test_")
    )
    files += [SCRIPTS / "pr_ui_scenes" / "_common.py"] + sorted(
        (SCRIPTS / "studio_test_kit").glob("*.py")
    )
    for f in files:
        try:
            h.update(str(f.relative_to(SCRIPTS)).encode() + b"\0" + f.read_bytes())
        except OSError:
            pass
    return h.hexdigest()


def fixture_revisions(names, data):
    """{fixture key: resolved HF revision (refs/main in the suite HF cache) or the spec}."""
    from studio_regress import prefetch

    out = {}
    for k, spec in prefetch.fixtures_for(names, data).items():
        repo = spec.get("repo")
        ref = (
            engine.SUITE_HF_HOME / "hub" / f"models--{repo.replace('/', '--')}" / "refs" / "main"
            if repo
            else None
        )
        try:
            out[k] = ref.read_text().strip() if ref else json.dumps(spec, sort_keys = True)
        except OSError:
            out[k] = json.dumps(spec, sort_keys = True)
    return out


def base_key(
    sha,
    home,
    names,
    data,
    online,
    mode = "sequential",
):
    """mode: how the base side's Studios were laid out (sequential, or parallel/<instance>): path and
    port text in the shots depend on it, so a base side is only reused under the same one."""
    import platform
    import socket

    return {
        "sha": sha,
        "home": str(home),
        "journeys": sorted(names),
        "suite": suite_version(),
        "fixtures": fixture_revisions(names, data),
        "online": bool(online),
        "mode": mode,
        "platform": [
            socket.gethostname(),
            platform.platform(),
            os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        ],
    }


def base_reusable(root, key, journeys):
    """The previous run in this same output dir recorded the base side under exactly `key` and
    every step there passed. Only the same root: a different PR / workspace has a different
    state-home path and port, which Studio shows, so another session's shots never compare."""
    f = root / "before" / REUSE_FILE
    try:
        if json.loads(f.read_text()) != key:
            return False
        # dates / relative times in the UI drift: never pair shots taken too far apart
        if time.time() - f.stat().st_mtime > 3600 * float(
            os.environ.get("STUDIO_REGRESS_REUSE_BASE_HOURS", "12")
        ):
            return False
    except (OSError, ValueError):
        return False
    for n in key["journeys"]:
        for st in journeys[n][0].steps:
            try:
                if (
                    json.loads((root / "before" / n / f"{st.id}.facts.json").read_text()).get(
                        "_status"
                    )
                    != "ok"
                ):
                    return False
            except (OSError, ValueError):
                return False
    return True


CRAWL_WORKERS = 3
TOKEN_MAX_AGE_S = engine.TOKEN_MAX_AGE_S


def _fresh_tokens(
    tokens,
    tokens_at,
    base_url,
    home,
    password,
    now = None,
):
    """engine.fresh_tokens (the one token path; scheduler.TokenState wraps it per Studio)."""
    return engine.fresh_tokens(tokens, tokens_at, base_url, home, password, now = now, log = _log)


async def _run_crawl(browser, base_url, tokens, j, side, out, state_common):
    """Crawl routes in CRAWL_WORKERS concurrent browser contexts (partitioned by step index,
    so the partition is identical on both sides); each writes its own step files."""
    from studio_regress.contract import Journey

    async def part(steps):
        bctx = await engine.authed_context(browser, base_url, tokens)
        try:
            page = await engine.fixture.prepare_page(await bctx.new_page())
            ctx = Ctx(
                page = page,
                base_url = base_url,
                api = None,
                side = side,
                out_dir = str(out),
                state = dict(state_common),
            )
            await engine.run_journey(Journey(name = j.name, tier = j.tier, steps = steps), ctx, out)
        finally:
            with contextlib.suppress(Exception):
                await bctx.close()

    parts = [tuple(j.steps[i::CRAWL_WORKERS]) for i in range(CRAWL_WORKERS)]
    await asyncio.gather(*(part(p) for p in parts if p))


# ------------------------------------------------------------------ report
def write_report(root, meta, steps, cov, timings):
    provenance.finish(meta)
    func = diff.functional_verdict(steps)
    counts = diff.summarize(steps)
    changed = [s for s in steps if s["verdict"] in ("VISUAL_DIFF", "DOM_ONLY_DIFF", "DIVERGED")]
    report = {
        **meta,
        "suite_version": SUITE_VERSION,
        "coverage": cov,
        "steps": steps,
        "functional": func,
        "counts": counts,
        "timings": timings,
        "provenance": provenance.record(meta),
    }
    (root / "report.json").write_text(json.dumps(report, indent = 1, default = str))
    provenance.update_manifest(root, meta)
    lines = [
        f"# studio_regress {meta.get('pr') or 'A/A'}",
        "",
        *provenance.header_lines(meta),
        "",
        f"functional: **{func}**  counts: {json.dumps(counts)}",
    ]
    if cov:  # Core-only runs drive no browser
        lines.append(
            f"coverage: {cov.get('overall')}% overall ({cov.get('exercised')}/{cov.get('inventory')}), "
            f"safe {cov.get('overall_safe')}%"
        )
    lines.append("")
    listed = changed + [
        s for s in steps if s["verdict"] in ("FAIL_HEAD", "FAIL_BOTH", "FLAKY", "VOID", "PLAN_DIFF")
    ]
    listed += [s for s in steps if s.get("mask_overrun") and s not in listed]
    for s in listed:
        lines.append(f"- {s['verdict']} `{s['key']}` px={s['pixels_changed']} {s.get('note', '')}")
        if s.get("compare"):  # a compare target's moved-cell table (planner_matrix)
            lines += ["", s["compare"], ""]
    for name, cmd in (meta.get("staging_required") or {}).items():
        lines.append(f"- NOT_RUN `{name}` (staging only): `{cmd}`")
    for lp in meta.get("leftover_processes") or []:  # the run did not stop everything it started
        lines.append(
            f"- LEFTOVER pid {lp['pid']} ({'stopped at the end' if lp.get('stopped') else 'STILL ALIVE'}): "
            f"`{lp.get('cmd', '')[:120]}`"
        )
    lines += ["", "timings (s): " + json.dumps(timings)]
    (root / "summary.md").write_text("\n".join(lines) + "\n")
    if func == "REGRESSION":
        return report, EXIT["regression"]
    if not steps or func == "VOID" or any(s["verdict"] == "VOID" for s in steps):
        return report, EXIT["void"]
    return report, EXIT["ui_changed"] if changed else EXIT["clean"]


# ------------------------------------------------------------------ parallel scheduler
def resolve_scheduler(a):
    """--scheduler, else parallel for a local run of both sides; sequential for --sequential,
    --base-url (one running Studio), a single side (staging legs shard one side per job) and the
    nested isolated run.py (its parent already scheduled it)."""
    if a.sequential:
        return "sequential"
    if a.scheduler:
        return a.scheduler
    if a.base_url or a.side != "both" or os.environ.get("STUDIO_REGRESS_ISOLATION"):
        return "sequential"
    return "parallel"


def _plan_for(
    a,
    root,
    names,
    journeys,
    data,
    sides,
    externals,
    cores,
    homes,
    password,
    reused = False,
    iso = None,
    hf = None,
):
    from studio_regress import scheduler

    instance = scheduler.load_instance(
        a.instance,
        root = root,
        homes = homes,
        password = password,
        parallel_sides = a.parallel_sides,
        online = a.online,
    )
    iso = isolated_names(names, journeys) if iso is None else iso
    if hf is None:
        hf = (
            head_first_names([n for n in names if n not in iso], journeys)
            if len(sides) == 2 and not a.record_flaky
            else []
        )
    skip_before = [n for n in names if n not in hf] if reused else []
    plan = scheduler.build_plan(
        names,
        journeys,
        data,
        sides,
        iso = iso,
        hf = hf,
        externals = externals if len(sides) == 2 else [],
        cores = cores,
        skip_before = skip_before,
        instance = instance,
    )
    return instance, plan


def print_plan(a, root, names, journeys, data, externals, cores):
    """--plan: the schedule `--scheduler parallel` would run (no install, no Studio, no GPU lease)."""
    from studio_regress import scheduler

    sides = ["after"] if a.base_url else (["before", "after"] if a.side == "both" else [a.side])
    if cores and not (a.pr and a.side == "both" and not a.base_url):
        cores = []
    _inst, plan = _plan_for(a, root, names, journeys, data, sides, externals, cores, {}, "")
    cap = scheduler.make_cap(a.jobs)
    print(
        scheduler.format_plan(
            plan, cap(), a.instance + (" (parallel sides)" if a.parallel_sides else "")
        )
    )
    print(f"root: {root}\nbase-side reuse is decided at run time (after installs)")
    return 0


def run_parallel(
    a,
    root,
    meta,
    phases,
    data,
    journeys,
    names,
    sides,
    homes,
    externals,
    cores,
    password,
    iso = (),
    hf = (),
    reuse_key = None,
    reused = False,
):
    """The scheduled replacement for the side loop + externals + Core. Returns (steps, timings);
    timings["_failed"] is set when an arm did not finish (its steps are VOID)."""
    from studio_regress import external, plat, scheduler

    instance, plan = _plan_for(
        a,
        root,
        names,
        journeys,
        data,
        sides,
        externals,
        cores,
        homes,
        password,
        reused = reused,
        iso = list(iso),
        hf = list(hf),
    )
    cap = scheduler.make_cap(a.jobs)
    _log("parallel schedule:\n" + scheduler.format_plan(plan, cap(), instance.name))
    # A step that writes nothing this time must not diff against an older run's file in this root
    for arm in plan.arms():
        if arm.side in scheduler.SIDES:
            for n in arm.names:
                shutil.rmtree(root / arm.side / n, ignore_errors = True)
            if "crawl" in arm.names:
                shutil.rmtree(root / arm.side / "crawl_manifest", ignore_errors = True)
    core_kw = {}
    if any(arm.kind == "core" for arm in plan.arms()):
        _targets, core_kw = core.prepare(cores, a.pr, a.gh_repo, python = a.core_python, log = _log)
    book = scheduler.LeaseBook()
    online_env = {"HF_HUB_OFFLINE": "0"} if a.online else {}

    # One Chromium per (run, side) on the side's own loop thread; a fresh context per journey.
    browsers = scheduler.SideBrowsers(log = _log)
    crashed = {}  # arm id -> epoch of the browser crash it was in flight for

    def execute(arm, ctx):
        if arm.kind in ("journey", "head_first"):
            names_, js = list(arm.names), journeys
            if arm.kind == "head_first" and arm.side == "before" and "after" in sides:
                rerun = head_first_base_plan(root, names_, journeys)
                _log(
                    f"head-first {names_}: base "
                    + (
                        f"runs {[(n, [st.id for st in j.steps]) for n, (j, _m) in rerun.items()]}"
                        if rerun
                        else "not needed (head passed)"
                    )
                )
                if not rerun:
                    return "skipped"
                names_, js = list(rerun), {**journeys, **rerun}
            handle = instance.start(
                arm.side, arm, {**ctx.env, **online_env} if ctx.env or online_env else {}
            )
            try:
                return browsers.get_side(arm.side).run(
                    lambda browser: drive_journeys(
                        arm.side, handle, root, names_, js, data, browser = browser
                    ),
                    timeout = ctx.remaining(),
                    on_cancel = ctx.on_cancel,
                )
            except scheduler.BrowserCrashed as e:
                crashed[arm.id] = e.at
                raise
            finally:
                handle.stop()
        if arm.kind == "isolated":
            rc = run_isolated(
                arm.side,
                homes[arm.side],
                root,
                list(arm.names),
                online = a.online,
                timeout = ctx.remaining(),
                on_proc = lambda p: ctx.on_cancel(lambda: plat.stop_tree(p.pid, grace_s = 20, proc = p)),
            )
            if rc != 0:
                raise RuntimeError(
                    f"isolated run exit {rc} ({root / 'logs' / f'isolated_{arm.side}.log'})"
                )
            return rc
        if arm.kind == "external":
            return external.run_target(
                arm.target, homes, root, env = engine.STUDIO_ENV, lease = ctx.lease or book.lease
            )
        if arm.kind == "core":
            return core.run_one(
                arm.target, a.pr, a.gh_repo, root, core_kw, lease = ctx.lease or book.lease, log = _log
            )
        raise ValueError(f"unknown arm kind {arm.kind}")

    t0 = time.time()
    sched = scheduler.Scheduler(
        plan,
        execute,
        instance = instance,
        book = book,
        cap = cap,
        log = _log,
        closers = [browsers.close],
        gpu_jobs = getattr(a, "gpu_jobs", None),
    )
    try:
        results = sched.run()
    except KeyboardInterrupt:
        _log("interrupted: every Studio, browser and suite this run started has been stopped")
        raise
    finally:
        browsers.close()
    t1 = time.time()
    phases.add("schedule", t1 - t0)
    timings, failed = {}, False
    for arm in plan.arms():
        r = results[arm.id]
        _log(
            f"unit {arm.id}: {r.status} {r.secs()}s"
            + (f" gpu {r.gpu}" if r.gpu is not None else "")
            + (f" ({r.error})" if r.error else "")
        )
        if arm.side in scheduler.SIDES:
            if isinstance(r.value, dict):
                timings.setdefault(arm.side, {}).update(r.value)
            elif arm.kind == "isolated":
                timings.setdefault(arm.side, {})["_isolated"] = r.secs()
            if r.status not in ("ok", "skipped"):
                failed = True
                if arm.id in crashed:  # steps the crash broke look "failed": they prove nothing
                    k = scheduler.void_facts_since(
                        root, arm.side, journeys, arm.names, r.error, crashed[arm.id]
                    )
                    _log(f"{arm.id}: {k} steps interrupted by the browser crash marked VOID")
                n = scheduler.void_missing_facts(
                    root, arm.side, journeys, arm.names, f"{r.status}: {r.error}"
                )
                _log(f"{arm.id}: {n} steps without evidence marked VOID")
    if (
        "before" in sides
        and reuse_key
        and not reused
        and all(
            results[x.id].status == "ok"
            for x in plan.arms()
            if x.side == "before" and x.kind in ("journey", "isolated")
        )
    ):
        (root / "before").mkdir(parents = True, exist_ok = True)
        (root / "before" / REUSE_FILE).write_text(json.dumps(reuse_key, indent = 1))
    meta["scheduler"] = {
        **scheduler.summary(plan, results, t0, t1),
        "instance": instance.name,
        "parallel_sides": bool(a.parallel_sides),
        "jobs_hard_cap": cap.hard,
        "max_concurrent_weight": sched.max_weight,
        "gpu_jobs": sched.gpu_jobs,
        "max_concurrent_gpu_weight": sched.max_gpu_weight,
    }
    if failed:
        timings["_failed"] = True
    steps = []
    if len(sides) == 2 and names:
        ext_keys = {f"{t['name']}/run" for t in externals}
        with phases("diff"):
            steps = [
                x
                for x in diff.diff_all(root, flaky = set() if a.record_flaky else None)
                if x["key"] not in ext_keys
            ]
    for arm in plan.arms():  # plan order: externals then Core, as the sequential path
        if arm.kind not in ("external", "core"):
            continue
        r = results[arm.id]
        rec = (
            r.value
            if isinstance(r.value, dict)
            else core._record(arm.target, "VOID", f"{r.status}: {r.error}", {"rc": None})
        )
        steps.append(rec)
        timings.setdefault(arm.kind, {})[arm.target["name"]] = r.secs()
    return steps, timings


STACK_DUMP_EVERY_S = 900


def _arm_stack_dumps(path, every_s = None):
    """Make a hung run diagnosable after the fact: every thread's Python stack goes to `path` on
    `kill -USR1 <pid>` and, unprompted, every STACK_DUMP_EVERY_S seconds
    ($STUDIO_REGRESS_STACK_DUMP_S; 0 turns the periodic dump off).

    Twice a run sat for over half an hour in a futex wait with no child processes, py-spy was not
    installed, and the thread it waited on was never found. A dump costs nothing when nothing is
    stuck, and it is the one thing that names the lock / future a stuck run is waiting on."""
    import faulthandler
    import signal

    try:
        every = float(
            os.environ.get("STUDIO_REGRESS_STACK_DUMP_S", STACK_DUMP_EVERY_S)
            if every_s is None
            else every_s
        )
        fh = open(path, "a", buffering = 1)  # kept open for the life of the process
        fh.write(
            f"# studio_regress pid {os.getpid()} started {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        if hasattr(signal, "SIGUSR1"):
            faulthandler.register(signal.SIGUSR1, file = fh, all_threads = True)
        if every > 0:
            faulthandler.dump_traceback_later(every, repeat = True, file = fh)
        _log(f"stack dumps: kill -USR1 {os.getpid()} (and every {every:.0f}s) -> {path}")
        return fh
    except (OSError, ValueError, RuntimeError) as e:  # diagnostics must never fail a run
        _log(f"stack dumps unavailable: {e}")
        return None


_RUN = {"root": None, "reaped": True}  # the live run's root until its leftovers are reaped


def main(argv = None):
    """_main, plus a reap of the run's leftover processes on every exit path (the early VOID
    returns and exceptions are when a Studio or worker is most likely still up)."""
    _RUN.update(root = None, reaped = True)
    try:
        return _main(argv)
    finally:
        if not _RUN["reaped"] and _RUN["root"] is not None:
            from studio_regress import gc as gc_mod
            _RUN["reaped"] = True
            gc_mod.reap_run(_RUN["root"], log = _log)


def _main(argv = None):
    p = argparse.ArgumentParser(description = "Studio before/after regression run")
    p.add_argument("--pr", type = int)
    p.add_argument("--gh-repo", default = "unslothai/unsloth")
    p.add_argument("--repo", default = str(WS / "unsloth"), help = "local unsloth clone")
    p.add_argument("--root", help = "output dir (default outputs/studio_regress/pr<N> or aa_<ts>)")
    p.add_argument("--home-before")
    p.add_argument("--home-after")
    p.add_argument("--base-url", help = "dev: drive an already-running Studio (single side)")
    p.add_argument("--password", help = "with --base-url: its password")
    p.add_argument("--side", choices = ("before", "after", "both"), default = "both")
    p.add_argument("--only", help = "comma-separated targets / globs")
    p.add_argument("--exclude", help = "comma-separated globs")
    p.add_argument("--tier", action = "append", choices = ("fast", "model", "gpu"))
    p.add_argument("--all", action = "store_true")
    p.add_argument("--list", action = "store_true")
    p.add_argument("--no-prefetch", action = "store_true", help = "skip fixture model prefetch")
    p.add_argument(
        "--online", action = "store_true", help = "let Studio reach the network (default offline)"
    )
    p.add_argument("--record-flaky", action = "store_true", help = "A/A: record non-SAME keys as FLAKY")
    p.add_argument(
        "--fresh-base",
        action = "store_true",
        help = "always rerun the base side (default: reuse this root's previous base side when "
        "base SHA, suite code, fixtures, journeys and host all match and it fully passed)",
    )
    p.add_argument(
        "--core-python",
        default = None,
        help = "interpreter for job / regression targets (default $STUDIO_REGRESS_CORE_PYTHON, "
        "else temp/venv_core, else $VIRTUAL_ENV); needs torch + unsloth deps",
    )
    p.add_argument("--json", action = "store_true")
    p.add_argument(
        "--scheduler",
        choices = ("sequential", "parallel"),
        default = os.environ.get("STUDIO_REGRESS_SCHEDULER") or None,
        help = "parallel (default for a local two-sided run): scheduler.py, one Studio per (side x "
        "journey chain), both sides at once, externals / Core overlapping them; sequential "
        "(default for single-side, --base-url and nested isolated runs): the legacy "
        "before-then-after run on one Studio per side ($STUDIO_REGRESS_SCHEDULER)",
    )
    p.add_argument(
        "--sequential",
        action = "store_true",
        default = os.environ.get("STUDIO_REGRESS_SEQUENTIAL", "") not in ("", "0"),
        help = "same as --scheduler sequential (also STUDIO_REGRESS_SEQUENTIAL=1)",
    )
    p.add_argument(
        "--jobs",
        type = int,
        help = "parallel: hard cap on concurrent arms (default: load / RAM aware, "
        "at most 8; $STUDIO_REGRESS_JOBS)",
    )
    p.add_argument(
        "--gpu-jobs",
        type = int,
        default = int(os.environ.get("STUDIO_REGRESS_GPU_JOBS") or 2),
        help = "parallel: gpu-tier Studio arms at once (default 2, $STUDIO_REGRESS_GPU_JOBS)",
    )
    p.add_argument(
        "--parallel-sides",
        action = "store_true",
        help = "parallel + --instance shared: both sides at once on per-side state paths / ports "
        "(path / port text then differs between sides: A/A timing and plumbing only)",
    )
    p.add_argument(
        "--instance",
        default = os.environ.get("STUDIO_REGRESS_INSTANCE") or "per_unit",
        help = "parallel: per_unit (default: own home + port per side x chain, masked in captures), "
        "shared (one Studio per side) or `module:Factory` implementing "
        "scheduler.IsolatedInstance ($STUDIO_REGRESS_INSTANCE)",
    )
    p.add_argument(
        "--plan",
        action = "store_true",
        help = "print the parallel schedule and exit (no install, no Studio)",
    )
    p.add_argument(
        "--no-confirm",
        action = "store_true",
        help = "do not rerun FAIL_HEAD / VISUAL_DIFF targets once to confirm them (confirm.py)",
    )
    a = p.parse_args(argv)

    phases = Phases()
    data = selection.load()
    journeys = load_journeys()
    if a.list:
        for n, (j, _m) in journeys.items():
            print(f"{n:24} {j.tier:6} {len(j.steps):3} steps")
        for t in data["target"]:
            if t["kind"] == "external":
                print(f"{t['name']:24} {t.get('tier', '?'):6} external: {' '.join(t['cmd'])}")
            elif t["kind"] in selection.CORE_KINDS:
                print(
                    f"{t['name']:24} {t.get('tier', '?'):6} {t['kind']}: {', '.join(selection.target_repos(t))}"
                )
        missing = [
            t["name"]
            for t in data["target"]
            if t["kind"] == "journey" and t["name"] not in journeys
        ]
        if missing:
            print("registered but not implemented:", ", ".join(missing))
        return 0

    meta = provenance.stamp_start({"pr": a.pr, "repo": a.gh_repo})
    if a.pr and not a.only and not a.all:
        with phases("select"):
            files, labels = selection.pr_files_and_labels(a.pr, a.gh_repo)
            sel = selection.select(files, labels, data, tiers = a.tier, repo = a.gh_repo)
        meta["selection"] = sel
    else:
        sel = {"selected": []}
    names = selection.apply_overrides(
        sel,
        only = a.only.split(",") if a.only else None,
        exclude = a.exclude.split(",") if a.exclude else None,
        all_targets = a.all,
        data = data,
    )
    local_skip = {t["name"] for t in data["target"] if t.get("staging_only")}
    if local_skip & set(names):
        _log(f"staging only, skipped locally: {sorted(local_skip & set(names))}")
        staged = selection.staging_commands(
            {
                "selected": [
                    {"name": t["name"], "staging_only": True, "staging": t.get("staging")}
                    for t in data["target"]
                    if t["name"] in local_skip & set(names)
                ]
            },
            a.pr,
            a.gh_repo,
        )
        if staged:
            meta["staging_required"] = staged
            for name, cmd in staged.items():
                _log(f"NOT COVERED HERE, run on staging: {name}: {cmd}")
    by_name = {t["name"]: t for t in data["target"]}
    for_repo = {
        n for n in names if n in by_name and a.gh_repo in selection.target_repos(by_name[n])
    }
    if set(names) - for_repo:
        _log(f"not for {a.gh_repo}, skipped: {sorted(set(names) - for_repo)}")
    externals = [
        by_name[n]
        for n in names
        if by_name.get(n, {}).get("kind") == "external"
        and n in for_repo
        and (not a.tier or by_name[n].get("tier") in a.tier)
    ]
    cores = [
        by_name[n]
        for n in names
        if by_name.get(n, {}).get("kind") in selection.CORE_KINDS
        and n in for_repo
        and n not in local_skip
        and (not a.tier or by_name[n].get("tier") in a.tier)
    ]
    if cores and not (a.pr and a.side == "both" and not a.base_url):
        _log(f"Core targets need --pr and both sides, skipped: {[t['name'] for t in cores]}")
        cores = []
    names = [n for n in names if n in journeys and n not in local_skip and n in for_repo]
    if a.tier:
        names = [n for n in names if journeys[n][0].tier in a.tier]
    names = order(names, journeys, data)
    meta["targets"] = names + [t["name"] for t in externals] + [t["name"] for t in cores]
    if not names and not externals and not cores:
        _log("nothing selected")
        return EXIT["void"]
    # absolute: external targets and the nested isolated run.py use another cwd
    root = Path(
        a.root
        or (
            WS
            / "outputs"
            / "studio_regress"
            / (f"pr{a.pr}" if a.pr else f"aa_{time.strftime('%Y%m%d_%H%M%S')}")
        )
    ).resolve()
    a.scheduler = resolve_scheduler(a)
    if a.plan:
        return print_plan(a, root, names, journeys, data, externals, cores)
    root.mkdir(parents = True, exist_ok = True)
    # A reused root (pr<N>) keeps the last run's report; while it says finished, `gc --procs --kill`
    # would read this live run's processes as leftovers.
    (root / "report.json").unlink(missing_ok = True)
    (root / "logs").mkdir(exist_ok = True)
    _arm_stack_dumps(root / "logs" / "stacks.txt")
    from studio_regress import gc as gc_mod

    os.environ[gc_mod.RUN_ROOT_ENV] = str(
        root
    )  # every process this run starts carries it (reap_run)
    _RUN.update(root = root, reaped = False)
    _log(f"journeys: {names} -> {root}")
    if cores:
        _log(f"core: {[t['name'] for t in cores]}")
    if not names and not externals:  # Core only: no Studio install, no browser
        with phases("core"):
            if a.scheduler == "parallel":
                steps, ptimes = run_parallel(
                    a,
                    root,
                    meta,
                    phases,
                    data,
                    journeys,
                    [],
                    ["before", "after"],
                    {},
                    [],
                    cores,
                    "",
                )
                ctimes = ptimes.get("core", {})
            else:
                steps, ctimes = core.run_all(
                    cores, a.pr, a.gh_repo, root, python = a.core_python, log = _log
                )
        for st in steps:  # the SHAs the Core targets actually compared
            meta.setdefault("base_sha", (st.get("before") or {}).get("sha"))
            meta.setdefault("head_sha", (st.get("after") or {}).get("sha"))
        with phases("confirm"):
            confirm.confirm_steps(a, root, steps, meta, log = _log)
        meta["leftover_processes"] = gc_mod.reap_run(root, log = _log)
        _RUN["reaped"] = True
        report, code = write_report(
            root, meta, steps, {}, {"core": ctimes, "phases": phases.done()}
        )
        print((root / "summary.md").read_text())
        if a.json:
            print(
                json.dumps({"exit": code, "root": str(root), "counts": report["counts"]}, indent = 1)
            )
        return code

    # installs
    homes = {"before": a.home_before, "after": a.home_after}
    if a.base_url:
        sides = [a.side if a.side != "both" else "after"]
    else:
        sides = ["before", "after"] if a.side == "both" else [a.side]
        # Resolve and install only the sides this run drives: a single-side run with its own
        # --home-<side> (the staging legs) needs neither gh nor the other side's install.
        if a.pr and any(not homes[s] for s in sides):
            from pr_ui_diff import resolve_shas

            with phases("resolve_shas"):
                mb, head_sha, head_ref = resolve_shas(Path(a.repo), a.pr, a.gh_repo)
            meta.update(base_sha = mb, head_sha = head_sha, merge_base = mb, head_ref = head_ref)
            # upstream_ci picks the suites this diff can reach (journeys/upstream_ci.py)
            os.environ.update(
                STUDIO_REGRESS_BASE_SHA = mb,
                STUDIO_REGRESS_HEAD_SHA = head_sha,
                STUDIO_REGRESS_SRC_REPO = str(Path(a.repo).resolve()),
            )
            want = {
                s: sha
                for s, sha in (("before", mb), ("after", head_sha))
                if s in sides and not homes[s]
            }
            # No journeys: only external suites use the installs, and they run head first and the
            # base only when head fails, so the base installs on demand.
            lazy = {}
            if not names and "before" in want and "after" in sides:
                lazy["before"] = want.pop("before")
            try:
                with phases("install"):
                    homes.update(
                        install_sides(Path(a.repo), want, root, phases, meta, mb) if want else {}
                    )
            except Exception as e:
                _log(f"install failed: {e}")
                return EXIT["void"]
            if lazy:

                def _lazy_base(sha = lazy["before"]):
                    try:
                        with phases("install_lazy"):
                            home = install_sides(
                                Path(a.repo),
                                {"before": sha},
                                root,
                                phases,
                                meta,
                                mb,
                                zoo = meta.get("zoo"),
                            )["before"]
                        if cache.is_shared(home):  # another user's source tree: let git read it
                            os.environ.update(_git_safe_env([source_dir(home)]))
                        return home
                    except (
                        Exception
                    ) as e:  # external.run_side turns a missing install into a VOID side
                        _log(f"install before failed: {e}")
                        return None

                homes = LazyHomes(homes, {"before": _lazy_base})
                _log("base install deferred: external targets install it only if head fails")
        for s in sides:
            if not dict.get(homes, s) and not getattr(homes, "_pending", {}).get(s):
                p.error(f"--home-{s} (or --pr) required")
    meta["homes"] = homes  # LazyHomes: the dict as it stands when the report is written
    git_env = _git_safe_env(
        [source_dir(h) for h in dict(homes).values() if h and cache.is_shared(h)]
    )
    os.environ.update(git_env)
    keys = [k for n in names for k in journeys[n][0].keys()] + [
        f"{t['name']}/run" for t in externals
    ]
    mpath = root / "manifest.json"
    if mpath.exists() and os.environ.get("STUDIO_REGRESS_ISOLATION"):  # nested isolated run: merge
        old = json.loads(mpath.read_text())
        names_m = old.get("journeys", []) + [n for n in names if n not in old.get("journeys", [])]
        keys = old.get("keys", []) + [k for k in keys if k not in old.get("keys", [])]
    else:
        names_m = names
    mpath.write_text(
        json.dumps(
            {"journeys": names_m, "keys": keys, "provenance": provenance.record(meta)}, indent = 1
        )
    )

    if not a.no_prefetch:
        from studio_regress import prefetch
        with phases("prefetch"):
            for k, spec in prefetch.fixtures_for(
                names + [t["name"] for t in externals], data
            ).items():
                try:
                    prefetch.fetch(spec)
                except Exception as e:
                    _log(f"prefetch {k} failed: {e}")
            for tier in prefetch.regression_tiers([t["name"] for t in cores], data):
                try:
                    prefetch.fetch_regression(tier, a.core_python)
                except Exception as e:
                    _log(f"prefetch regression/{tier} failed: {e}")
    password = a.password or ("Regress-" + secrets.token_urlsafe(10).replace("-", "x"))
    timings = {}
    iso = isolated_names(names, journeys) if not a.base_url else []
    shared = [n for n in names if n not in iso]
    # Pass / fail suites: head first, base only for what head fails. Not in an A/A (--record-flaky).
    hf = head_first_names(shared, journeys) if len(sides) == 2 and not a.record_flaky else []
    reuse_key = None
    if (
        len(sides) == 2
        and names
        and not a.base_url
        and not a.record_flaky
        and meta.get("base_sha")
        and homes.get("before") != homes.get("after")
        and os.environ.get("STUDIO_REGRESS_REUSE_BASE", "1") != "0"
    ):
        reuse_key = base_key(
            meta["base_sha"],
            homes.get("before"),
            [n for n in names if n not in hf],
            data,
            a.online,
            mode = "sequential"
            if a.scheduler != "parallel"
            else f"parallel/{a.instance}" + ("/sides" if a.parallel_sides else ""),
        )
    reused = bool(reuse_key) and not a.fresh_base and base_reusable(root, reuse_key, journeys)
    if reused:
        _log(
            "before: reusing this root's base side (same base SHA, suite, fixtures, host; all steps passed); "
            "--fresh-base reruns it"
        )
        meta["base_reused"] = True
    elif reuse_key:
        (root / "before" / REUSE_FILE).unlink(missing_ok = True)
    if a.scheduler == "parallel" and not a.base_url:
        steps, ptimes = run_parallel(
            a,
            root,
            meta,
            phases,
            data,
            journeys,
            names,
            sides,
            homes,
            externals,
            cores,
            password,
            iso = iso,
            hf = hf,
            reuse_key = reuse_key,
            reused = reused,
        )
        timings.update(ptimes)
        if len(sides) < 2:
            _log(f"single side done: {root / sides[0]}")
            return 0 if not ptimes.get("_failed") else EXIT["void"]
        timings.pop("_failed", None)
    else:
        for s in sides if names else ():
            if s == "before" and reused:
                continue
            run_now = [n for n in shared if not (s == "before" and n in hf)]
            # A step that writes nothing this time must not diff against an older run's file in this root
            for n in run_now + iso + (hf if s == "before" else []):
                shutil.rmtree(root / s / n, ignore_errors = True)
            if "crawl" in run_now:
                shutil.rmtree(root / s / "crawl_manifest", ignore_errors = True)
            try:
                if run_now:
                    with phases(f"side_{s}"):
                        timings[s] = asyncio.run(
                            run_side(
                                s,
                                homes.get(s),
                                root,
                                run_now,
                                journeys,
                                data,
                                password if a.base_url else password + s[:1],
                                base_url = a.base_url,
                                extra_env = {"HF_HUB_OFFLINE": "0"} if a.online else None,
                            )
                        )
            except Exception as e:
                _log(f"{s}: side failed: {type(e).__name__}: {e}")
                return EXIT["void"]
            if iso:
                t0 = time.time()
                rc = run_isolated(s, homes[s], root, iso, online = a.online)
                timings.setdefault(s, {})["_isolated"] = round(time.time() - t0, 1)
                phases.add(f"isolated_{s}", time.time() - t0)
                _log(f"{s}: isolated {iso} exit {rc} ({root / 'logs' / f'isolated_{s}.log'})")
            if s == "before" and reuse_key and not (iso and rc):  # only a base side that completed
                (root / "before").mkdir(parents = True, exist_ok = True)
                (root / "before" / REUSE_FILE).write_text(json.dumps(reuse_key, indent = 1))
        if hf:
            rerun = head_first_base_plan(root, hf, journeys)
            _log(
                f"head-first {hf}: base "
                + (
                    f"runs {[(n, [st.id for st in j.steps]) for n, (j, _m) in rerun.items()]}"
                    if rerun
                    else "not needed (head passed)"
                )
            )
            if rerun:
                try:
                    with phases("side_before_head_first"):
                        t = asyncio.run(
                            run_side(
                                "before",
                                homes.get("before"),
                                root,
                                list(rerun),
                                {**journeys, **rerun},
                                data,
                                password + "b",
                                extra_env = {"HF_HUB_OFFLINE": "0"} if a.online else None,
                            )
                        )
                    timings.setdefault("before", {}).update(t)
                except Exception as e:
                    _log(f"before (head-first rerun): side failed: {type(e).__name__}: {e}")
                    return EXIT["void"]
        if len(sides) < 2:
            if externals:
                _log(f"external targets need both sides, skipped: {[t['name'] for t in externals]}")
            _log(f"single side done: {root / sides[0]}")
            return 0
        ext_keys = {f"{t['name']}/run" for t in externals}
        with phases("diff"):
            steps = [
                x
                for x in diff.diff_all(root, flaky = set() if a.record_flaky else None)
                if x["key"] not in ext_keys
            ]
        for t in externals:  # head first, base only when head fails (external.py)
            from studio_regress import external

            t0 = time.time()
            steps.append(
                external.run_target(t, homes, root, env = engine.STUDIO_ENV)
            )  # git_env: os.environ
            timings.setdefault("external", {})[t["name"]] = round(time.time() - t0, 1)
            phases.add("external", time.time() - t0)
            _log(f"external {t['name']}: {steps[-1]['verdict']} {timings['external'][t['name']]}s")
        if cores:
            with phases("core"):
                csteps, ctimes = core.run_all(
                    cores, a.pr, a.gh_repo, root, python = a.core_python, log = _log
                )
            steps += csteps
            timings["core"] = ctimes
    if a.record_flaky:
        bad = [x for x in steps if x["verdict"] in diff.FLAKY_SOURCES]
        diff.record_flaky((), steps = bad)
        _log(f"recorded {len(bad)} flaky keys")
    with phases("confirm"):
        confirm.confirm_steps(a, root, steps, meta, log = _log)
    meta["leftover_processes"] = gc_mod.reap_run(root, log = _log)
    _RUN["reaped"] = True
    cov = coverage.compute(root / "after")
    meta["homes"] = dict(homes)
    timings["phases"] = phases.done()
    _log(f"phases (s): {json.dumps(timings['phases'])}")
    report, code = write_report(root, meta, steps, cov, timings)
    print((root / "summary.md").read_text())
    if a.json:
        print(
            json.dumps(
                {"exit": code, "root": str(root), "counts": report["counts"], "coverage": cov},
                indent = 1,
            )
        )
    return code


if __name__ == "__main__":
    sys.exit(main())
