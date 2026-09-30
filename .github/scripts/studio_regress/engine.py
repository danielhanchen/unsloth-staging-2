"""Side execution: fresh-state homes, Studio launch, auth, and the per-step loop.

Public helpers (forks B/C journeys and harnesses use these):

  state_home(install_home, dest)      overlay home: heavy install dirs symlinked from the
                                      (shared, read-only) install, mutable state (auth/, *.db,
                                      assets, logs, settings) fresh. Same install on both sides
                                      is then a true A/A control, and N sessions share 1 install.
  launch(home, port, log, env)        `unsloth studio -p PORT` in its own session / process group
  api_login(base_url, user, pw)       -> access_token (raises on failure)
  rotate_bootstrap(base_url, home, new_pw)  bootstrap login + change-password over the API
  authed_context(browser, base_url, token)  fixture context with the SPA's auth keys seeded
  run_journey(journey, ctx, out_dir)  frozen step loop: action, capture, facts (with _status)
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from studio_regress import fixture, plat  # noqa: E402
from studio_regress.contract import (
    SKIPPED_NOT_TOUCHED,
    Ctx,
    StepFailed,
    StepNotTouched,  # noqa: E402
    StepUnreachable,
)

# Install entries shared by symlink; everything else in a state home is created fresh.
SHARED_ENTRIES = (
    "unsloth_studio",
    "bin",
    "llama.cpp",
    "whisper.cpp",
    "share",
    ".unsloth-studio-owned",
    ".uidiff_sha",
    ".llama.cpp.install.lock",
)
SHARED_PREFIXES = (".venv_t5_", ".venv")
# <home>/cache holds hub-state (persisted Hub UI state) and uv / triton scratch: fresh per side.
# Models come from SUITE_HF_HOME, not from here.
SHARED_CACHE = ("cache",)

WS = Path(os.environ.get("WORKSPACE") or SCRIPTS.parent.parent.parent)
# Playwright's browsers for this suite live in the workspace (run.py's isolated path already
# points there). Without this default, a shared side looks under ~/.cache/ms-playwright, which
# holds another playwright version's build, and every side fails at chromium.launch.
if not os.environ.get("PLAYWRIGHT_BROWSERS_PATH") and (WS / "temp" / "pw_browsers").is_dir():
    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(WS / "temp" / "pw_browsers")
# One suite-owned HF cache holding only prefetched fixtures: the Hub "On device" list and every
# model load are then identical on both sides and across runs (the host HF cache changes under
# us as other sessions download). Studio runs offline by default; `--online` / a journey's
# STUDIO_ENV can lift it.
SUITE_HF_HOME = WS / "temp" / "studio_regress" / "hf_home"

# Studio also scans ~/.cache/huggingface/hub and LM Studio dirs under $HOME for "On device"
# models; a suite HOME keeps the host's downloads (which change under us) out of the UI.
SUITE_HOME = WS / "temp" / "studio_regress" / "home"

STUDIO_ENV = {
    "HOME": str(SUITE_HOME),
    "HF_HOME": str(SUITE_HF_HOME),
    "HF_HUB_CACHE": str(SUITE_HF_HOME / "hub"),
    "HF_XET_CACHE": str(SUITE_HF_HOME / "xet"),
    # Legacy / sibling cache vars the host sets: Studio scans them for "On device" models too.
    "HUGGINGFACE_HUB_CACHE": str(SUITE_HF_HOME / "hub"),
    "TRANSFORMERS_CACHE": str(SUITE_HF_HOME / "hub"),
    "HF_DATASETS_CACHE": str(SUITE_HF_HOME / "datasets"),
    "HF_ASSETS_CACHE": str(SUITE_HF_HOME / "assets"),
    "HF_MODULES_CACHE": str(SUITE_HF_HOME / "modules"),
    "XDG_CACHE_HOME": str(SUITE_HOME / ".cache"),
    "HF_HUB_OFFLINE": "1",
    "UNSLOTH_DISABLE_UPDATE_CHECK": "1",
    "UNSLOTH_STUDIO_DISABLE_PUBLIC_CHECK": "1",
    "UNSLOTH_HELPER_MODEL_DISABLE": "1",
    "UNSLOTH_STUDIO_DISABLE_TORCH_WARM": "1",
    # nvidia-smi shim (scripts/nvsmi_cache): serve a GPU reading up to 15 min old while a refresh
    # runs. Studio kills each nvidia-smi after 5-10 s and then falls back to torch, which opens a
    # CUDA context on every GPU (measured 221 s for one utilisation read on a busy host); the
    # readings are masked in every shot, and a real refresh follows each stale answer.
    "NVSMI_CACHE_STALE_DYNAMIC": "900",
    "HF_HUB_DISABLE_TELEMETRY": "1",
}


def state_home(
    install_home: Path,
    dest: Path,
    share_cache: bool = False,
    venv_shim: bool = False,
) -> Path:
    """venv_shim: make <dest>/unsloth_studio a real venv dir (own bin + pyvenv.cfg, shared lib) so
    sys.prefix equals $UNSLOTH_STUDIO_HOME/unsloth_studio. The `unsloth run` / `unsloth studio`
    CLI compares the two unresolved and re-execs forever through a plain symlink."""
    install_home, dest = Path(install_home).resolve(), Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents = True)
    for entry in install_home.iterdir():
        name = entry.name
        if venv_shim and name == "unsloth_studio":
            _venv_shim(entry, dest / name)
        elif (
            name in SHARED_ENTRIES
            or name.startswith(SHARED_PREFIXES)
            or (share_cache and name in SHARED_CACHE)
        ):
            plat.link(dest / name, entry)
    return dest


def _venv_shim(venv: Path, dest: Path):
    dest.mkdir()
    (dest / "bin").mkdir()
    for entry in venv.iterdir():
        if entry.name != "bin":
            plat.link(dest / entry.name, entry)
    old, new = str(venv / "bin"), str(dest / "bin")
    for f in (venv / "bin").iterdir():
        out = dest / "bin" / f.name
        if f.is_symlink():  # python -> /usr/bin/python3.13
            out.symlink_to(os.readlink(f))
            continue
        head = f.read_bytes()[:512]
        if old.encode() in head and not f.name.startswith("activate"):
            out.write_bytes(f.read_bytes().replace(old.encode(), new.encode(), 2))
            out.chmod(f.stat().st_mode)
        else:
            plat.link(out, f)


# A Studio that lost the bind: uvicorn's errno 98 (Linux) / 48 (macOS) text, WinError 10048, and
# Studio's own "Port N is already in use ... will use port M instead" when it moves on.
_LOST_BIND = ("address already in use", "only one usage of each socket address")
_MOVED_PORT = re.compile(r"\bport \d+ is already in use\b")


def _lost_bind(log_path) -> bool:
    try:
        text = Path(log_path).read_text(errors = "replace").lower()
    except OSError:
        return False
    return any(s in text for s in _LOST_BIND) or bool(_MOVED_PORT.search(text))


def studio_bin(home) -> str:
    """The `unsloth` CLI of an install / state home. Windows installs put unsloth.exe in bin/
    (a .cmd when the exe is blocked) and in the venv's Scripts/; POSIX ones a bin/unsloth shim."""
    home = Path(home)
    if plat._is_windows():
        cands = (
            home / "bin" / "unsloth.exe",
            home / "unsloth_studio" / "Scripts" / "unsloth.exe",
            home / "bin" / "unsloth.cmd",
        )
    else:
        cands = (
            home / "bin" / "unsloth",
            home / ".venv_t5_550" / "bin" / "unsloth",
            home / ".venv_t5_530" / "bin" / "unsloth",
            home / "unsloth_studio" / "bin" / "unsloth",
        )
    for c in cands:
        if c.exists():
            return str(c)
    raise FileNotFoundError(f"`unsloth` CLI not found under {home}")


def _pgrep(port) -> list:
    """Pids whose command line is `... studio -p PORT` (candidates only: ownership is _ours)."""
    import shutil
    import subprocess

    if not shutil.which("pgrep"):
        return []
    out = subprocess.run(
        ["pgrep", "-f", f"studio -p {port}"], capture_output = True, text = True
    ).stdout
    return [int(p) for p in out.split() if p.isdigit() and int(p) != os.getpid()]


def _owns_port(
    port: int,
    home,
    log_path: Path,
    pid = None,
) -> bool:
    """The Studio we started is the one answering: the process listening on `port` is the one we
    spawned (`pid`) or one of its descendants (on Windows unsloth.exe runs the backend as a child),
    or, for a Studio this process did not spawn, runs with OUR home (Linux /proc). pick_free_ports
    is check-then-use and this host runs other Studios, so the health check can pass against a
    stranger's server while ours died with errno 98 or moved to the next port."""
    if _lost_bind(log_path):
        return False
    tree = plat.descendants(pid) if pid else set()
    if pid and not tree:
        return False  # ours already exited: whoever answered is a stranger
    cands = plat.listening_pids(port) or set(_pgrep(port))
    if not cands:  # no lsof / pgrep / Get-NetTCPConnection: our live tree is the best proof left
        return bool(tree)
    return any(p in tree or _ours(p, home) for p in cands)


def _spawn(
    home: Path,
    port: int,
    log_path: Path,
    env: dict,
    healthz_timeout_s: int,
    password_timeout_s: int,
):
    """`unsloth studio -p PORT` in its own session / process group, output to log_path. Returns
    (StudioInstall, Popen); the Popen is recorded in _LAUNCHED before any wait, so stop() can
    always reach it. Raises on a health timeout or an early exit."""
    import subprocess
    from studio_test_kit.lifecycle import StudioInstall, _read_bootstrap_password, health_body_ok

    log_path = Path(log_path).resolve()
    log_path.parent.mkdir(parents = True, exist_ok = True)
    inst = StudioInstall(home = Path(home), repo = Path(home), branch = "")
    with open(log_path, "w") as fh:
        proc = subprocess.Popen(
            [studio_bin(home), "studio", "-p", str(port)],
            env = {**os.environ, "UNSLOTH_STUDIO_HOME": str(home), **env},
            stdout = fh,
            stderr = subprocess.STDOUT,
            stdin = subprocess.DEVNULL,
            **plat.group_kwargs(),
        )
    _LAUNCHED[port] = {"home": str(Path(home).resolve()), "proc": proc}
    inst.port, inst.pid = port, proc.pid
    inst.bootstrap_password = _read_bootstrap_password(
        Path(home), log_path, time.time() + password_timeout_s
    )
    urls = [f"http://127.0.0.1:{port}/api/health", f"http://127.0.0.1:{port}/healthz"]
    deadline = time.time() + healthz_timeout_s
    while time.time() < deadline:
        if any(health_body_ok(u) for u in urls):
            return inst, proc
        if proc.poll() is not None and not plat.descendants(proc.pid):
            raise RuntimeError(
                f"Studio exited {proc.returncode} before answering on :{port} (see {log_path})"
            )
        time.sleep(1)
    raise TimeoutError(f"Studio on :{port} did not answer /api/health within {healthz_timeout_s}s")


def _nvsmi_path_env():
    """PATH with the nvidia-smi caching shim (scripts/nvsmi_cache) first. Studio's backend runs
    nvidia-smi for hardware info, VRAM fit checks and llama.cpp placement; on a busy driver each
    call takes 25-100 s, so pages that wait on /api/system/hardware stalled for minutes. The shim
    serves concurrent identical queries from one real call (answers up to a few seconds old).
    UNSLOTH_NVSMI_CACHE=0 leaves PATH alone."""
    shim = SCRIPTS / "nvsmi_cache"
    if os.environ.get("UNSLOTH_NVSMI_CACHE", "1") == "0" or not (shim / "nvidia-smi").exists():
        return {}
    parts = [p for p in os.environ.get("PATH", "").split(os.pathsep) if p and p != str(shim)]
    return {"PATH": os.pathsep.join([str(shim), *parts])}


NVSMI_SHIM = SCRIPTS / "nvsmi_cache"


# The queries Studio's hardware probes run (utils/hardware): answered once into the shim cache
# before Studio asks, since Studio gives up on each call after 5-10 s and falls back to slower paths.
NVSMI_PREWARM = (
    ("--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"),
    (
        "--query-gpu=index,utilization.gpu,temperature.gpu,memory.used,memory.total,power.draw,"
        "power.limit",
        "--format=csv,noheader,nounits",
    ),
)


def prewarm_nvsmi(env: dict):
    """Fire-and-forget the known Studio queries through the shim (never waited on)."""
    import subprocess

    if not env.get("PATH", "").startswith(str(NVSMI_SHIM)):  # shim not in use (opted out / absent)
        return
    for args in NVSMI_PREWARM:
        try:
            import threading
            proc = subprocess.Popen(
                [str(NVSMI_SHIM / "nvidia-smi"), *args],
                env = {**os.environ, **env},
                stdout = subprocess.DEVNULL,
                stderr = subprocess.DEVNULL,
                stdin = subprocess.DEVNULL,
                **plat.group_kwargs(),
            )
            threading.Thread(target = proc.wait, daemon = True).start()  # reaped, never waited on
        except OSError:
            pass


def launch(
    home: Path,
    port: int,
    log_path: Path,
    extra_env: dict | None = None,
    healthz_timeout_s: int = 240,
    attempts: int = 3,
    pick = None,
):
    """Start Studio on `port`; on a lost bind, retry on a fresh port. The returned install's
    `.port` is the port actually serving (callers must use it, not the one they passed).
    `pick`: zero-arg callable for the retry port (default: the process-wide PORTS pool)."""
    pick = pick or PORTS.take
    env = {**STUDIO_ENV, **_nvsmi_path_env(), **(extra_env or {})}
    prewarm_nvsmi(env)
    for i in range(attempts):
        try:
            from studio_regress import timeouts
            inst, proc = _spawn(
                home, port, log_path, env, timeouts.scaled(healthz_timeout_s), timeouts.scaled(60)
            )
        except Exception:
            if i == attempts - 1 or not _lost_bind(log_path):
                # A Studio that started but never answered (health / password timeout) would
                # otherwise outlive the run: callers only stop what launch() returned.
                stop(port, home = home)
                raise
        else:
            if _owns_port(port, home, log_path, proc.pid):
                return inst
        stop(port, home = home)
        with open(log_path, "a") as fh:
            fh.write(f"\n[studio_regress] port {port} was taken by another process; retrying\n")
        port = pick()
    raise RuntimeError(
        f"could not start Studio on a free port after {attempts} attempts (see {log_path})"
    )


# ------------------------------------------------------------------ instances (parallel mode)
# Parallel mode gives every unit (side x journey group) its own Studio. The home path and the
# port reach the UI (the auth 401 reset hint names <home>/unsloth_studio/bin/unsloth, Settings >
# Logs names <home>/logs/..., agent / API panels print window.location.origin), so instances
# are made to differ ONLY in fixed-width runs of characters: homes are state/<root>-iNN (same
# length for every instance of a run) and ports all have the same digit count. Layout is then
# identical across instances by construction; fixture.capture masks exactly the glyphs of the
# token / port digits and the DOM snapshot and facts replace them with fixed text.
INSTANCE_WIDTH = 2
MAX_INSTANCES = 10**INSTANCE_WIDTH


def instance_token(root_name: str, idx: int) -> str:
    if not 0 <= idx < MAX_INSTANCES:
        raise ValueError(f"instance index {idx} out of range (max {MAX_INSTANCES - 1})")
    return f"{root_name}-i{idx:0{INSTANCE_WIDTH}d}"


def instance_home(root_name: str, idx: int) -> Path:
    return WS / "temp" / "studio_regress" / "state" / instance_token(root_name, idx)


def instance_info(root_name: str, idx: int, port: int) -> dict:
    """What fixture / run_journey need to neutralise this instance's literals: `token` is the
    unique text (it appears in every path under the home), `mask_from` the offset where the
    per-instance characters start (only those glyphs are masked), `port` the serving port."""
    tok = instance_token(root_name, idx)
    return {
        "token": tok,
        "mask_from": len(root_name) + 1,
        "port": int(port),
        "canon_token": f"{root_name}-i" + "#" * INSTANCE_WIDTH,
    }


def canon_text(s: str, inst: dict | None) -> str:
    """`s` with this instance's token and port replaced by fixed text (DOM snapshot, facts)."""
    if not inst or not isinstance(s, str):
        return s
    s = s.replace(inst["token"], inst["canon_token"])
    return re.sub(
        r"(?<=[:=])" + str(inst["port"]) + r"(?!\d)|(?<=[Pp]ort )" + str(inst["port"]) + r"(?!\d)",
        "<port>",
        s,
    )


def canon_facts(obj, inst: dict | None):
    if not inst:
        return obj
    if isinstance(obj, str):
        return canon_text(obj, inst)
    if isinstance(obj, dict):
        return {k: canon_facts(v, inst) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [canon_facts(v, inst) for v in obj]
    return obj


class PortPool:
    """The one port allocator for the Studios this process launches (PORTS below): never hands the
    same port to two Studios of this process (a bind probe reserves nothing, and parallel units pick
    within the same second), and every port has the same digit count (the port shows in the UI;
    equal width keeps layout equal). Scans this workspace's band (pick_free_ports' rules) from a
    random point, or, with `seed`, from a point derived from it (the sequential path starts both
    sides on the same port: it shows in the page)."""

    def __init__(
        self,
        width: int | None = None,
        start: int | None = None,
        stop: int | None = None,
    ):
        import threading

        self.lock = threading.Lock()
        self.held: set = set()
        self.width = width
        self.start, self.stop = start, stop

    def _band(self):
        from pr_ui_scenes import _common

        start = self.start
        if start is None:
            env = os.environ.get("UIDIFF_PORT_START")
            start = int(env) if env else _common._workspace_port_base()
        return start, self.stop or start + _common._BAND_SIZE

    def take(self, seed: str | None = None) -> int:
        import hashlib
        import random
        import socket

        start, stop = self._band()
        span = list(range(start, stop))
        k = (
            (int(hashlib.sha256(seed.encode()).hexdigest(), 16) % len(span))
            if seed is not None
            else random.randrange(len(span))
        )
        with self.lock:
            if self.width is None:  # the widest width the band offers most ports of
                w4 = sum(1 for p in span if len(str(p)) == len(str(start)))
                self.width = len(str(start)) if w4 >= len(span) - w4 else len(str(stop - 1))
            for port in span[k:] + span[:k]:
                if port in self.held or len(str(port)) != self.width or port == 8888:
                    continue  # 8888 is Studio's default: the agent panel renders another command
                s = socket.socket()
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    s.bind(("127.0.0.1", port))
                except OSError:
                    continue
                finally:
                    s.close()
                self.held.add(port)
                return port
        raise RuntimeError(
            f"no free {self.width}-digit port in [{start}, {stop}) ({len(self.held)} held)"
        )

    def release(self, port):
        with self.lock:
            self.held.discard(port)


# Every Studio this process starts takes its port here (scheduler instances, the sequential path,
# launch()'s retry after a lost bind); release() when that Studio is stopped.
PORTS = PortPool()


# port -> {"home": state home, "proc": Popen} of the Studio this process launched there.
_LAUNCHED: dict = {}


def _home_matches(val, home) -> bool:
    val = str(Path(val).resolve())
    if home:
        return val == str(Path(home).resolve())
    return val.startswith(str((WS / "temp" / "studio_regress").resolve()))


def _ours(pid: int, home: str | None) -> bool:
    """Only this suite's Studios: `pid` belongs to the process tree of a Studio this process
    spawned with that home, or (Linux, where another process's environment is readable) runs with
    UNSLOTH_STUDIO_HOME set to the home we launched (when unknown, a state home under this
    workspace's temp/studio_regress). The same unix user runs other workspaces' Studios on this
    host, possibly on the same port; never a command-line match."""
    for rec in list(_LAUNCHED.values()):
        if (home is None or _home_matches(rec["home"], home)) and int(pid) in plat.descendants(
            rec["proc"].pid
        ):
            return True
    env = plat.process_env(pid)
    val = (env or {}).get("UNSLOTH_STUDIO_HOME")
    return bool(val) and _home_matches(val, home)


def stop(port: int, home = None):
    """Stop this suite's Studio on `port`: the tree we spawned (own session / process group, so
    llama-server children go too) plus, on Linux, any `studio -p PORT` running with our home."""
    rec = _LAUNCHED.pop(port, None)
    home = home or (rec or {}).get("home")
    proc = (rec or {}).get("proc")
    extra = [p for p in _pgrep(port) if _ours(p, home)]
    if proc is None and not extra:
        return
    plat.stop_tree(proc.pid if proc is not None else None, grace_s = 20, extra = extra, proc = proc)


def _post(
    url,
    payload,
    token = None,
    timeout = 180,
    attempts = 3,
):
    """POST with a patient timeout, retried on a timeout / refused connection: on a loaded host
    (load 300+) Studio's auth endpoints were seen taking over 60 s, and one timeout VOIDed a run."""
    import httpx

    h = {"Authorization": f"Bearer {token}"} if token else {}
    for i in range(attempts):
        try:
            return httpx.post(url, json = payload, headers = h, timeout = timeout)
        except (httpx.TimeoutException, httpx.ConnectError):
            if i == attempts - 1:
                raise
            time.sleep(5 * (i + 1))


def api_login(
    base_url,
    password,
    username = "unsloth",
):
    r = _post(f"{base_url}/api/auth/login", {"username": username, "password": password})
    if r.status_code != 200:
        raise RuntimeError(f"login failed {r.status_code}: {r.text[:200]}")
    return r.json()


# Studio access tokens last 60 min; a side on a loaded host runs well past that, so log in again
# once the token is this old (the first login also rotates the bootstrap password).
TOKEN_MAX_AGE_S = 40 * 60


def fresh_tokens(
    tokens,
    tokens_at,
    base_url,
    home,
    password,
    now = None,
    max_age_s = None,
    log = None,
):
    """(tokens, issued_at): the ONE token path (scheduler.TokenState and run._fresh_tokens call it).
    First call: rotate the bootstrap password (home given) or log in; later calls re-login once
    the token is older than max_age_s (TOKEN_MAX_AGE_S), else return what they were given."""
    now = time.time() if now is None else now
    max_age_s = TOKEN_MAX_AGE_S if max_age_s is None else max_age_s
    if tokens is None:
        return (
            rotate_bootstrap(base_url, home, password) if home else api_login(base_url, password)
        ), now
    if now - tokens_at > max_age_s:
        (log or (lambda m: print(f"[studio_regress] {m}", file = sys.stderr, flush = True)))(
            f"re-login: token is {int((now - tokens_at) // 60)} min old"
        )
        return api_login(base_url, password), now
    return tokens, tokens_at


def rotate_bootstrap(base_url, home, new_password):
    """Bootstrap login + change-password; idempotent when already rotated to new_password."""
    boot = Path(home) / "auth" / ".bootstrap_password"
    if not boot.exists():
        return api_login(base_url, new_password)
    old = boot.read_text().strip()
    tok = api_login(base_url, old)
    r = _post(
        f"{base_url}/api/auth/change-password",
        {"current_password": old, "new_password": new_password},
        token = tok["access_token"],
    )
    if r.status_code != 200:
        try:  # a retried change-password whose first attempt landed after its client timed out
            return api_login(base_url, new_password)
        except RuntimeError:
            pass
        raise RuntimeError(f"change-password failed {r.status_code}: {r.text[:200]}")
    return r.json()


_WARMED: set = set()
WARM_TIMEOUT_S = 600.0
WARM_PATHS = (
    "/api/system",
    "/api/system/hardware?include_details=true",
    "/api/settings/caches",
    "/api/models/list",
    "/api/inference/status",
)


def warm(base_url, token):
    """First /api/system (and /api/system/hardware) on a fresh Studio probes every GPU (~10 s on an 8-GPU host); later calls
    are cached. Pages that render before it answers show "Checking for GPUs..." / "No GPU
    detected", the ones after show the devices, so which one a shot catches is timing. Pay it
    once per Studio before any page opens."""
    key = (base_url, token)  # both sides of a run share a port; the token is per Studio
    if key in _WARMED:
        return
    import httpx
    from concurrent.futures import ThreadPoolExecutor
    from studio_regress import timeouts

    def one(path):
        try:
            # Until the probe is done: a Studio still probing answers every other request late
            # (seen: /api/inference/status 324 s, page loads past 60 s), so starting the steps
            # before it finishes only turns its latency into their false failures.
            httpx.get(
                f"{base_url}{path}",
                headers = {"Authorization": f"Bearer {token}"},
                timeout = timeouts.scaled(WARM_TIMEOUT_S),
            )
        except Exception:
            pass

    # /api/system/hardware (Export's GPU check, the About tab's hardware list) probes cold too,
    # and so do the other first-visit readers a fresh Studio answers slowly: the cache sizes walk
    # (Settings > Data, 451 s once on a busy disk), the model list scan and inference status. All at
    # once, read-only GETs, outside any step: paying them here keeps the first journey's shots
    # from catching "Loading..." / "Unavailable" on one side only.
    with ThreadPoolExecutor(len(WARM_PATHS)) as ex:
        list(ex.map(one, WARM_PATHS))
    _WARMED.add(key)


async def authed_context(browser, base_url, tokens: dict, **kw):
    import asyncio

    await asyncio.to_thread(warm, base_url, tokens["access_token"])
    ctx = await fixture.new_context(browser, **kw)
    seed = {
        "unsloth_auth_token": tokens["access_token"],
        "unsloth_auth_refresh_token": tokens.get("refresh_token", ""),
    }
    await ctx.add_init_script(
        "(() => { try { const s = "
        + json.dumps(seed)
        + "; for (const k in s) if (!localStorage.getItem(k)) localStorage.setItem(k, s[k]); } catch (e) {} })();"
    )
    return ctx


class ApiClient:
    """Tiny async client bound to one Studio + token (ctx.api)."""

    def __init__(self, base_url, token):
        import httpx
        self.base_url = base_url
        self.c = httpx.AsyncClient(
            base_url = base_url, timeout = 120, headers = {"Authorization": f"Bearer {token}"}
        )

    async def get(self, path, **kw):
        r = await self.c.get(path, **kw)
        r.raise_for_status()
        return r.json()

    async def post(
        self,
        path,
        json = None,
        **kw,
    ):
        r = await self.c.post(path, json = json, **kw)
        r.raise_for_status()
        return r.json() if r.content else {}

    async def raw(self, method, path, **kw):
        return await self.c.request(method, path, **kw)

    # The model / GPU journeys (journeys/_b_common.py, parallel.py) speak httpx directly:
    # request() returns the response without raising, stream() is httpx's streaming context.
    async def request(self, method, path, **kw):
        return await self.c.request(method, path, **kw)

    def stream(self, method, path, **kw):
        return self.c.stream(method, path, **kw)

    async def aclose(self):
        await self.c.aclose()


CAPTURE_S = 180.0  # settle (<= ~50 s) + stable screenshots + DOM, idle host
CAPTURE_QUICK_S = 45.0  # evidence of a failed step


async def _run_budgeted(coro, state: dict):
    """Await `coro` until state["_step_deadline"]; raise asyncio.TimeoutError past it (the task is
    cancelled). Unlike asyncio.wait_for the deadline can move: time spent inside
    timeouts.unbudgeted() (a shared prerequisite that is not the step's subject, e.g. Studio's cold
    GPU probe) pushes it back instead of eating the step's budget."""
    task = asyncio.ensure_future(coro)
    try:
        while True:
            left = state["_step_deadline"] - time.monotonic()
            if left <= 0 and not state.get("_paused"):
                task.cancel()
                try:
                    await task
                except BaseException:
                    pass
                raise asyncio.TimeoutError()
            done, _ = await asyncio.wait(
                {task}, timeout = max(0.05, min(left, 1.0)) if left > 0 else 1.0
            )
            if task in done:
                return task.result()
    except asyncio.CancelledError:
        task.cancel()
        raise


def _merge_timings(path: Path, rows: list):
    """<journey>/_timings.json: one row per step. Crawl runs one journey in several concurrent
    partitions (same event loop, no await in here), so rows merge by step id."""
    try:
        old = json.loads(path.read_text()) if path.exists() else []
    except (OSError, ValueError):
        old = []
    mine = {r.get("_step") for r in rows}
    try:
        path.write_text(json.dumps([r for r in old if r.get("_step") not in mine] + rows, indent = 1))
    except OSError:
        pass


async def run_journey(journey, ctx: Ctx, out_dir: Path) -> dict:
    """Run the frozen step list; one facts file per step whatever happens.

    After the first failed / unreachable step the remaining steps are recorded as
    `skipped_after_failure` (both sides see the same outcome shape, and the diff reports
    FAIL_HEAD / DIVERGED on the step that actually broke). A step that runs out of its budget
    (timeouts.step_budget: measured p95 x 3, times the host load factor) stops the journey even
    when it is `independent`: the Studio is wedged or the host starved, and every later step
    would only wait out its own budget too.

    Per-step wall times land in facts (never diffed): `_elapsed_s` (whole step, = `_s`),
    `_action_s`, `_capture_s`, `_budget_s`, `_load_factor`; and in <journey>/_timings.json."""
    from studio_regress import timeouts

    jdir = Path(out_dir) / journey.name
    jdir.mkdir(parents = True, exist_ok = True)
    timings, broken, timed_out = {}, None, False
    f = timeouts.factor()
    if ctx.page is not None:
        # Playwright's own 30 s defaults (goto, click, wait_for without an explicit timeout)
        # stretch with the host like the step budgets: a starved box must not fail a navigation.
        try:
            ctx.page.set_default_timeout(timeouts.PLAYWRIGHT_DEFAULT_MS * f)
            ctx.page.set_default_navigation_timeout(timeouts.PLAYWRIGHT_DEFAULT_MS * f)
        except Exception:
            pass
    rows = []
    inst = (ctx.state or {}).get("_instance")  # per-unit Studio: its home token / port
    for step in journey.steps:
        facts = {"_step": step.id}
        t0 = time.perf_counter()
        if broken and (timed_out or not journey.independent):
            facts.update(_status = "skipped_after_failure", _after = broken)
            if timed_out:
                facts["_reason"] = "timeout"
            (jdir / f"{step.id}.facts.json").write_text(json.dumps(facts, indent = 1, default = str))
            continue
        f = timeouts.factor()
        budget, idle, _ = timeouts.step_budget(journey.name, step, f)
        # Actions size their inner waits against this (timeouts.remaining), so an inner deadline
        # with a useful message, or a subprocess kill, fires before the outer cancel.
        ctx.state["_step_deadline"] = time.monotonic() + budget
        ctx.state["_unbudgeted_s"] = 0.0
        try:
            got = await _run_budgeted(step.action(ctx), ctx.state)
            facts.update(got or {})
            facts["_status"] = "ok"
            timeouts.note_pace(
                f"{journey.name}/{step.id}",
                time.perf_counter() - t0 - ctx.state.get("_unbudgeted_s", 0.0),
            )
        except StepNotTouched as e:
            facts.update(_status = SKIPPED_NOT_TOUCHED, _reason = str(e)[:800])
        except StepUnreachable as e:
            facts.update(_status = "unreachable", _error = str(e)[:500])
            broken = step.id
        except asyncio.TimeoutError as e:
            # wait_for's own timeout (budget spent), or a TimeoutError raised inside the action.
            spent = time.perf_counter() - t0 - ctx.state.get("_unbudgeted_s", 0.0)
            mine = spent >= budget * 0.98
            msg = (
                f"step budget {budget:.0f}s exceeded (idle {idle:.0f}s x load factor {f})"
                if mine
                else f"{type(e).__name__}: {e}"
            )
            facts.update(_status = "failed", _error = msg[:800], _trace = traceback.format_exc()[-1500:])
            if mine:
                facts["_timeout"] = True
                timed_out = True
            broken = step.id
        except (StepFailed, Exception) as e:  # noqa: B014
            facts.update(
                _status = "failed",
                _error = f"{type(e).__name__}: {e}"[:800],
                _trace = traceback.format_exc()[-1500:],
            )
            broken = step.id
        ctx.state.pop("_step_deadline", None)  # teardown and later helpers are not inside this step
        if ctx.state.get("_unbudgeted_s"):
            facts["_unbudgeted_s"] = round(ctx.state["_unbudgeted_s"], 2)
        t1 = time.perf_counter()
        facts["_action_s"] = round(t1 - t0, 2)
        if ctx.page is not None:
            try:
                # A failed step's shot is evidence only (the verdict comes from the status, not the
                # pixels): do not spend the full settle budget on a page that is already wrong.
                # Bounded: a hung renderer never answers page.evaluate, and an unbounded capture
                # after a failed goto was seen holding a side for 23 minutes.
                quick = facts["_status"] != "ok"
                rects, dom = await asyncio.wait_for(
                    fixture.capture(
                        ctx.page,
                        jdir,
                        step.id,
                        masks = step.masks,
                        full_page = step.full_page,
                        shot = step.shot,
                        quick = quick,
                        instance = inst,
                    ),
                    timeout = timeouts.scaled(CAPTURE_QUICK_S if quick else CAPTURE_S),
                )
                facts["_masks"] = rects
                facts["_clicked"] = await fixture.drain_clicks(ctx.page)
                if dom.get("_mask_overrun"):
                    facts["_mask_overrun"] = dom["_mask_overrun"]
                if dom.get("_instance_masked"):
                    facts["_instance_masked"] = dom["_instance_masked"]
                for k in ("_settle_s", "_shot_s"):
                    if k in dom:
                        facts[k] = dom[k]
            except asyncio.TimeoutError:
                facts.setdefault("_capture_error", "capture timed out")
            except Exception as e:
                facts.setdefault("_capture_error", str(e)[:300])
        facts = canon_facts(facts, inst)  # this Studio's home token / port -> fixed text
        t2 = time.perf_counter()
        facts["_capture_s"] = round(t2 - t1, 2)
        facts["_s"] = facts["_elapsed_s"] = round(t2 - t0, 2)
        facts["_budget_s"], facts["_load_factor"] = budget, f
        timings[step.id] = facts["_s"]
        rows.append(
            {
                k: facts.get(k)
                for k in (
                    "_step",
                    "_status",
                    "_elapsed_s",
                    "_action_s",
                    "_capture_s",
                    "_settle_s",
                    "_shot_s",
                    "_budget_s",
                    "_load_factor",
                )
            }
        )
        (jdir / f"{step.id}.facts.json").write_text(json.dumps(facts, indent = 1, default = str))
    if journey.teardown is not None:
        t0 = time.perf_counter()
        try:
            await journey.teardown(ctx)
        except Exception as e:  # teardown must never mask the step results
            print(f"[studio_regress] {journey.name} teardown: {e}", file = sys.stderr)
        rows.append({"_step": "_teardown", "_elapsed_s": round(time.perf_counter() - t0, 2)})
    _merge_timings(jdir / "_timings.json", rows)
    return {"journey": journey.name, "broken": broken, "timings": timings, "timed_out": timed_out}
