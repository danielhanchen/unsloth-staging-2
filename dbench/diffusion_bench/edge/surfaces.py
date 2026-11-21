"""The two surfaces the edge suite drives with the same calls:

  InprocSurface  Studio's DiffusionBackend / VideoBackend in a worker subprocess (edge/inproc_worker.py)
  HttpSurface    a real Studio server (launched from the tree, or attached) through studio_client

Every call either returns or raises one of:
  Refused(status, detail)   a clean refusal: HTTP 4xx, or the in-process exception the route maps to 4xx
  LoadError(detail)         the background load reported phase=error (clean: the UI shows the message)
  Fault(status, detail)     HTTP 5xx / dropped connection, or an exception the route maps to 500
  Hang(detail)              no answer inside the call's timeout
  HarnessHang(detail)       an in-process worker that never came up (no ready line by its deadline): the
                            harness, not the check, failed; carries a stack dump (VOID, not FAIL). A worker
                            that is up but whose setup (the Studio import, i.e. the code under test) never
                            answers raises Hang with the same dump: a PR that deadlocks the import is FAIL.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import setup_studio  # noqa: E402
from studio_client import StudioClient, StudioError, decode_mp4, decode_png  # noqa: E402


class SurfaceError(Exception):
    kind = "error"

    def __init__(self, detail: str = "", status: Optional[int] = None, extra: Optional[dict] = None):
        super().__init__(f"{self.kind}{f' {status}' if status else ''}: {str(detail)[:500]}")
        self.detail, self.status, self.extra = str(detail), status, extra or {}


class Refused(SurfaceError):
    kind = "refused"


class LoadError(SurfaceError):
    kind = "load_error"


class Fault(SurfaceError):
    kind = "fault"


class Hang(SurfaceError):
    kind = "hang"


class HarnessHang(SurfaceError):
    """Not a Hang subclass: the check never reached its subject, so the runner reports VOID and does
    not recover the main surface for it."""
    kind = "harness_hang"


# Startup deadlines of an in-process worker: the ready line (interpreter + protocol) and the setup
# answer (Studio import). Past either, the worker is dumped (faulthandler, py-spy) and killed.
WORKER_READY_S = float(os.environ.get("EDGE_WORKER_READY_S", "180"))
WORKER_SETUP_S = float(os.environ.get("EDGE_WORKER_SETUP_S", "300"))


class Gen:
    """One generate result: ``images`` (list of uint8 HWC arrays) or ``frames`` (uint8 THWC) plus metadata."""

    def __init__(self, images=None, frames=None, meta=None, wall_s=None, fps=None, ids=None):
        self.images, self.frames, self.meta, self.wall_s, self.fps = images or [], frames, meta or {}, wall_s, fps
        self.ids = ids or []

    @property
    def image(self):
        return self.images[0]


KEEP_LOG_LINES = 2000   # server / worker log tail kept under the surface's out dir


def catches_sigusr1(pid: int) -> bool:
    """True when the process installed a SIGUSR1 handler (faulthandler.register). Sending SIGUSR1 to a process
    that did not would terminate it, so every dump checks /proc/<pid>/status SigCgt first."""
    import signal

    if not hasattr(signal, "SIGUSR1"):
        return False
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("SigCgt:"):
                return bool(int(line.split()[1], 16) >> (signal.SIGUSR1 - 1) & 1)
    except (OSError, ValueError):
        pass
    return False


def dump_stacks(pid: Optional[int], log_path=None, wait_s: float = 2.0) -> str:
    """Every thread's stack of a live process: py-spy dump when installed, else faulthandler through SIGUSR1 when
    the process registered it (the dump lands in its stderr log; the new bytes are returned), else a note."""
    import shutil
    import signal

    if not pid or not Path(f"/proc/{pid}").exists():
        return f"pid {pid}: not running, no stack dump"
    spy = shutil.which("py-spy")
    if spy:
        try:
            r = subprocess.run([spy, "dump", "--pid", str(pid)], capture_output = True, text = True, timeout = 60)
            if r.returncode == 0 and r.stdout.strip():
                return f"--- py-spy dump --pid {pid}\n{r.stdout[-12000:]}"
            spy_err = (r.stderr or r.stdout).strip()[-300:]
        except (OSError, subprocess.TimeoutExpired) as e:
            spy_err = str(e)
    else:
        spy_err = "py-spy not installed"
    if catches_sigusr1(pid):
        log = Path(log_path) if log_path else None
        start = log.stat().st_size if log and log.exists() else 0
        try:
            os.kill(pid, signal.SIGUSR1)
            time.sleep(wait_s)
        except OSError as e:
            return f"pid {pid}: SIGUSR1 failed: {e} ({spy_err})"
        if log and log.exists():
            with open(log, "rb") as fh:
                fh.seek(start)
                return f"--- faulthandler (SIGUSR1) pid {pid}, from {log}\n" + fh.read()[-12000:].decode(errors = "replace")
        return f"pid {pid}: SIGUSR1 sent, but its stderr log is unknown ({spy_err})"
    return f"pid {pid}: no stack dump ({spy_err}; the process does not handle SIGUSR1)"


def keep_log_tail(src, dest: Path, header: str = "", lines: int = KEEP_LOG_LINES) -> Optional[Path]:
    """Append the last `lines` lines of `src` to `dest` (one block per server / worker instance)."""
    if not src or not Path(src).exists():
        return None
    tail = Path(src).read_text(errors = "replace").splitlines()[-lines:]
    dest.parent.mkdir(parents = True, exist_ok = True)
    with open(dest, "a") as fh:
        fh.write(f"===== {header or src} (last {len(tail)} lines of {src})\n" + "\n".join(tail) + "\n")
    return dest


class Surface:
    name = "base"
    pid: Optional[int] = None

    def log_file(self) -> Optional[Path]:
        """The model process's stdout + stderr log, when known."""
        return None

    def dump_stacks(self) -> str:
        """Thread stacks of the process that owns the models (a Hang's evidence; framework.Runner)."""
        return dump_stacks(self.pid, self.log_file())

    def mem(self) -> dict:
        """Per-process VRAM (driver view) and host RSS of the process that owns the models."""
        return setup_studio.process_memory(self.pid)

    def trimmed_mem(self) -> dict:
        """Memory after glibc malloc_trim(0) in the model process (in-process worker only; {} elsewhere)."""
        return {}

    def settle_mem(self, wait_s: float = 1.0) -> dict:
        time.sleep(wait_s)
        return self.mem()


# ================================================================================================ in process
class InprocSurface(Surface):
    name = "inproc"

    def __init__(self, python: str, studio_src: str, out: Path, gpu: str, log: Path, studio_home: Optional[str] = None,
                 env: Optional[dict] = None):
        self.python, self.studio_src, self.out, self.gpu, self.log_path = python, studio_src, Path(out), gpu, log
        self.studio_home, self.extra_env = studio_home, dict(env or {})
        self.proc: Optional[subprocess.Popen] = None
        self._pending: dict = {}
        self._lock = threading.Lock()
        self._next = 1

    # -------------------------------------------------------------------------------------- process
    def start(self) -> dict:
        env = setup_studio._clean_env({"CUDA_VISIBLE_DEVICES": str(self.gpu), "HIP_VISIBLE_DEVICES": str(self.gpu),
                                       "PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1", **self.extra_env})
        self.log_path.parent.mkdir(parents = True, exist_ok = True)
        self._logf = open(self.log_path, "ab")
        self.proc = subprocess.Popen([self.python, "-u", str(HERE / "inproc_worker.py")], stdin = subprocess.PIPE,
                                     stdout = subprocess.PIPE, stderr = self._logf, env = env,
                                     cwd = str(self.out), start_new_session = True, text = True, bufsize = 1)
        self.pid = self.proc.pid
        self._reader = threading.Thread(target = self._read, daemon = True)
        self._reader.start()
        ready = self._waiter(0)
        if not ready.wait(WORKER_READY_S):
            raise self._startup_hang(f"worker gave no ready line in {WORKER_READY_S:.0f}s")
        if not (ready.msg or {}).get("ok"):
            raise self._startup_hang(f"worker exited before its ready line (exit {self.proc.poll()})")
        try:
            return self.call("setup", {"studio_src": self.studio_src, "studio_home": self.studio_home},
                             timeout = WORKER_SETUP_S)
        except Hang as e:
            raise self._startup_hang(f"worker setup: {e.detail}", cls = Hang) from None

    def startup_dump(self, wait_s: float = 2.0) -> str:
        """Where a stuck worker is: faulthandler's all-thread dump (SIGUSR1, registered by the worker, lands in
        its stderr log), py-spy when installed, and the log tail."""
        import shutil
        import signal

        parts = [f"pid {self.pid} alive={self.alive()} exit={self.proc.poll() if self.proc else None}"]
        if self.alive() and catches_sigusr1(self.proc.pid):
            try:
                os.kill(self.proc.pid, signal.SIGUSR1)
                time.sleep(wait_s)
            except OSError as e:
                parts.append(f"SIGUSR1 failed: {e}")
        spy = shutil.which("py-spy")
        if spy and self.alive():
            try:
                r = subprocess.run([spy, "dump", "--pid", str(self.proc.pid)], capture_output = True, text = True,
                                   timeout = 60)
                parts += ["--- py-spy dump", (r.stdout or r.stderr)[-6000:]]
            except (OSError, subprocess.TimeoutExpired) as e:
                parts.append(f"py-spy failed: {e}")
        try:
            self._logf.flush()
            parts += [f"--- {self.log_path} (tail)", Path(self.log_path).read_text(errors = "replace")[-8000:]]
        except OSError as e:
            parts.append(f"log unreadable: {e}")
        return "\n".join(parts)

    def _startup_hang(self, what: str, cls = None) -> SurfaceError:
        dump = self.startup_dump()
        path = self.out / "worker_startup_dump.txt"
        try:
            self.out.mkdir(parents = True, exist_ok = True)
            path.write_text(dump)
        except OSError:
            path = None
        self.stop()
        return (cls or HarnessHang)(f"{what}; stack dump in {path}", extra = {"dump": dump[-4000:], "dump_path": str(path)})

    def log_file(self) -> Optional[Path]:
        return Path(self.log_path)   # already under the run's out dir, whole

    def _read(self) -> None:
        for line in self.proc.stdout:
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            slot = self._waiter(msg.get("id"))
            slot.msg = msg
            slot.set()
        for slot in list(self._pending.values()):  # the worker died: wake everyone
            if not slot.is_set():
                slot.msg = {"ok": False, "error": {"cls": "fault", "detail": "worker process exited"}}
                slot.set()

    def _waiter(self, rid) -> threading.Event:
        with self._lock:
            if rid not in self._pending:
                ev = threading.Event()
                ev.msg = None
                self._pending[rid] = ev
            return self._pending[rid]

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self) -> None:
        if self.proc is None:
            return
        try:
            if self.alive():
                self.proc.stdin.write(json.dumps({"id": -1, "op": "exit"}) + "\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout = 30)
        except Exception:  # noqa: BLE001
            pass
        if self.alive():
            import signal

            os.killpg(self.proc.pid, signal.SIGKILL)  # the worker's own session, started above
            self.proc.wait(timeout = 30)
        self.proc = None

    def recover(self) -> None:
        """After a hang: kill the worker (its whole session) and start a fresh one with nothing loaded."""
        self.stop()
        self._pending.clear()
        self.start()

    def call(self, op: str, args: Optional[dict] = None, timeout: float = 600) -> Any:
        with self._lock:
            rid = self._next
            self._next += 1
        slot = self._waiter(rid)
        if not self.alive():
            raise Fault("worker process is not running")
        self.proc.stdin.write(json.dumps({"id": rid, "op": op, "args": args or {}}) + "\n")
        self.proc.stdin.flush()
        if not slot.wait(timeout):
            raise Hang(f"{op} gave no answer in {timeout:.0f}s")
        msg = slot.msg
        self._pending.pop(rid, None)
        if msg.get("ok"):
            return msg["result"]
        err = msg.get("error") or {}
        cls = {"refused": Refused, "load_error": LoadError, "hang": Hang}.get(err.get("cls"), Fault)
        raise cls(err.get("detail", ""), err.get("status"), {"exc": err.get("exc"), "tb": err.get("tb")})

    # -------------------------------------------------------------------------------------- media
    def load(self, kind: str, model: str, timeout: float = 900, **opts) -> dict:
        return self.call("load", {"kind": kind, "model": model, "opts": opts, "timeout": timeout}, timeout + 30)

    def load_async(self, kind: str, model: str, **opts) -> dict:
        return self.call("load_async", {"kind": kind, "model": model, "opts": opts}, timeout = 120)

    def wait_loaded(self, kind: str, timeout: float = 900) -> dict:
        return self.call("wait_loaded", {"kind": kind, "timeout": timeout}, timeout + 30)

    def generate(self, kind: str = "image", timeout: float = 300, stem: Optional[str] = None, **params) -> Gen:
        import numpy as np

        if kind == "image":
            params.setdefault("width", 512)
            params.setdefault("height", 512)
        r = self.call("generate", {"kind": kind, "params": params, "out_dir": str(self.out / "media"),
                                   "stem": stem}, timeout)
        if kind == "video":
            frames = np.load(r["frames"])
            return Gen(frames = frames, meta = r["meta"], wall_s = r["wall_s"], fps = r["meta"].get("fps"))
        return Gen(images = [np.load(p) for p in r["images"]], meta = r["meta"], wall_s = r["wall_s"])

    def cancel(self, kind: str = "image") -> dict:
        return self.call("cancel", {"kind": kind}, timeout = 60)

    def unload(self, kind: str = "image") -> dict:
        return self.call("unload", {"kind": kind}, timeout = 300)

    def status(self, kind: str = "image") -> dict:
        return self.call("status", {"kind": kind}, timeout = 60)

    def progress(self, kind: str = "image") -> dict:
        return self.call("progress", {"kind": kind}, timeout = 60)["generate"]

    def trimmed_mem(self) -> dict:
        try:
            return self.call("trim", timeout = 60)
        except SurfaceError:
            return {}

    def mem(self) -> dict:
        out = setup_studio.process_memory(self.pid)
        try:
            # "mem", not "gc": no empty_cache here, so what the driver shows is what Studio itself left reserved
            out.update({k: v for k, v in self.call("mem", timeout = 60).items() if k.startswith("torch_")})
        except SurfaceError:
            pass
        return out


# ================================================================================================ HTTP
def _http_error(exc: StudioError) -> SurfaceError:
    if exc.status == 0:
        body = exc.body if isinstance(exc.body, dict) else {}
        if body.get("phase") == "error":
            return LoadError(exc.detail)
        if body.get("phase") == "failed":  # a video job's terminal failure, as generate-progress reports it
            if "cancel" in exc.detail.lower():
                return Refused(exc.detail, 409)
            return Fault(exc.detail, 500)
        if "not ready in" in exc.detail or "not finished in" in exc.detail or "timed out" in exc.detail.lower():
            return Hang(exc.detail)
        return Fault(exc.detail, 0)
    if exc.clean_refusal:
        return Refused(exc.detail, exc.status)
    return Fault(exc.detail, exc.status)


class HttpSurface(Surface):
    name = "http"
    progress_timeout = 30.0   # per generate-progress poll; checks raise it from the run's unload times (Ctx.poll_timeout)

    def __init__(self, out: Path, gpu: str, studio_src: Optional[str] = None, python: Optional[str] = None,
                 attach: Optional[dict] = None):
        self.out, self.gpu, self.studio_src, self.python, self.attach = Path(out), gpu, studio_src, python, attach
        self.server: Optional[setup_studio.StudioServer] = None
        self.client: Optional[StudioClient] = None

    def start(self) -> dict:
        from backends.studio_http import connect

        if self.attach:
            self.server = setup_studio.attach(**self.attach)
        else:
            self.server = setup_studio.launch(self.studio_src, python = self.python, gpu = self.gpu,
                                              env = {"HF_HUB_OFFLINE": "1"})
        self.pid = self.server.server_pid
        self.client = connect(self.server)
        return self.server.to_json()

    def log_file(self) -> Optional[Path]:
        return Path(self.server.log) if self.server is not None and self.server.log else None

    def keep_log(self) -> Optional[Path]:
        """The server log lives under $WORKSPACE/logs; keep its tail with this run's evidence (studio_server.log
        under the surface's out dir), one block per server this surface started."""
        if self.server is None:
            return None
        return keep_log_tail(self.server.log, self.out / "studio_server.log",
                             header = f"Studio server pid {self.server.server_pid} port {self.server.port}")

    def stop(self) -> None:
        if self.server is not None and not self.server.attached:
            setup_studio.stop(self.server)
        try:
            self.keep_log()
        except OSError:
            pass
        self.server = None

    def alive(self) -> bool:
        try:
            self.client.health()
            return True
        except Exception:  # noqa: BLE001
            return False

    def recover(self) -> None:
        if self.alive():
            for kind in ("images", "video"):
                try:
                    self.client.post(f"/api/inference/{kind}/generate/cancel", timeout = 30)
                    self.client.post(f"/api/inference/{kind}/unload", timeout = 300)
                except Exception:  # noqa: BLE001
                    pass
            return
        if self.server is not None and self.server.attached:
            raise Fault("attached server stopped answering")
        self.stop()
        self.start()

    def _do(self, fn, *a, **kw):
        try:
            return fn(*a, **kw)
        except StudioError as exc:
            raise _http_error(exc) from None

    def new_client(self) -> StudioClient:
        """A second connection with the same token (concurrency checks)."""
        return StudioClient(self.client.base_url, self.client.username, self.client.password, self.client.token)

    # -------------------------------------------------------------------------------------- media
    def load(self, kind: str, model: str, timeout: float = 900, **opts) -> dict:
        from backends.studio_http import load_body

        body = load_body(opts, kind)
        fn = self.client.video_load if kind == "video" else self.client.images_load
        return self._do(fn, model, timeout = timeout, poll = 0.5, **body)

    def load_async(self, kind: str, model: str, **opts) -> dict:
        from backends.studio_http import load_body

        fn = self.client.video_load if kind == "video" else self.client.images_load
        return self._do(fn, model, wait = False, **load_body(opts, kind))

    def wait_loaded(self, kind: str, timeout: float = 900) -> dict:
        return self._do(self.client._wait_loaded, "video" if kind == "video" else "images", timeout, 0.5)

    def generate(self, kind: str = "image", timeout: float = 300, stem: Optional[str] = None,
                 client: Optional[StudioClient] = None, **params) -> Gen:
        import numpy as np

        c = client or self.client
        params = {k: v for k, v in params.items() if v is not None or k == "seed"}
        prompt = params.pop("prompt")
        t0 = time.perf_counter()
        if kind == "video":
            self._do(c.video_generate, prompt, **{k: v for k, v in params.items() if v is not None})
            prog = self._do(c.video_wait, timeout = timeout, poll = 0.25, progress_timeout = self.progress_timeout)
            vid = prog["video"]
            data = self._do(c.video_file, vid["id"])
            frames, fps = decode_mp4(data)
            if stem:
                (self.out / "media").mkdir(parents = True, exist_ok = True)
                (self.out / "media" / f"{stem}.mp4").write_bytes(data)
            return Gen(frames = frames, meta = {**vid, "decoded_fps": fps}, wall_s = time.perf_counter() - t0,
                       fps = vid.get("fps"), ids = [vid["id"]])
        params.setdefault("width", 512)
        params.setdefault("height", 512)
        res = self._do(c.images_generate, prompt, timeout = timeout, **{k: v for k, v in params.items()
                                                                          if v is not None})
        images = [np.asarray(decode_png(self._do(c.images_file, r["id"])).convert("RGB")) for r in res["images"]]
        return Gen(images = images, meta = {"records": res["images"], "seed": res["images"][0].get("seed"),
                                            "seeds": [r.get("seed") for r in res["images"]]},
                   wall_s = time.perf_counter() - t0, ids = [r["id"] for r in res["images"]])

    def cancel(self, kind: str = "image") -> dict:
        return self._do(self.client.video_cancel if kind == "video" else self.client.images_cancel)

    def unload(self, kind: str = "image") -> dict:
        return self._do(self.client.video_unload if kind == "video" else self.client.images_unload)

    def status(self, kind: str = "image") -> dict:
        return self._do(self.client.video_status if kind == "video" else self.client.images_status)

    def progress(self, kind: str = "image") -> dict:
        return self._do(self.client.video_generate_progress if kind == "video" else
                        self.client.images_generate_progress)
