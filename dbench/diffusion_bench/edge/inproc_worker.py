"""Hosts Studio's DiffusionBackend + VideoBackend in its own process and serves JSON-line requests on stdin, one
thread per request, so the edge runner can overlap calls (generate while loading, unload during a generation,
concurrent generates) and kill this process when one hangs.

Request:  {"id": 7, "op": "generate", "args": {...}}
Response: {"id": 7, "ok": true, "result": {...}}  or  {"id": 7, "ok": false, "error": {"cls", "status", "detail", "exc"}}

Errors are classified the way the HTTP routes map them, so one check can assert against either surface:
  ValueError / FileNotFoundError / VideoShapeError  -> refused 400 (422 for a shape error)
  RuntimeError carrying Studio's not-loaded / cancelled / busy / replaced sentinel -> refused 409
  load(): any RuntimeError -> refused 409 (routes/inference.py load_diffusion_model_gated)
  anything else -> fault (the route answers 500)
Media never crosses the pipe: images are written as .npy (lossless, fast), clips as .npy frames plus the .mp4 bytes.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

import faulthandler
import signal

faulthandler.enable()   # a hard crash leaves its stack in the worker log (stderr)
if hasattr(signal, "SIGUSR1"):   # the runner's startup deadline asks for every thread's stack (surfaces.py)
    faulthandler.register(signal.SIGUSR1, all_threads = True)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

_OUT = os.fdopen(os.dup(1), "w", buffering = 1)
os.dup2(2, 1)  # anything a library prints goes to stderr, never into the protocol stream
sys.stdout = sys.stderr
_LOCK = threading.Lock()


def send(msg: dict) -> None:
    with _LOCK:
        _OUT.write(json.dumps(msg, default = str) + "\n")
        _OUT.flush()


class LoadFailed(RuntimeError):
    """The background load thread reported phase=error (what the HTTP client sees on load-progress)."""


class State:
    image = None
    video = None
    captured: dict = {}
    sentinels: dict = {}


S = State()


def _setup(args: dict) -> dict:
    from backends.studio_inproc import studio_import_path

    tree = studio_import_path({"studio_src": args.get("studio_src"), "studio_home": args.get("studio_home")})
    from core.inference import video as V
    from core.inference.diffusion import DiffusionBackend
    from core.inference import diffusion_families as DF
    from core.inference import video_families as VF

    S.sentinels = {
        409: {getattr(DF, n) for n in dir(DF) if n.startswith("DIFFUSION_") and n.endswith("_MSG")}
        | {getattr(VF, n) for n in dir(VF) if n.startswith("VIDEO_") and n.endswith("_MSG")},
    }
    S.shape_error = getattr(VF, "VideoShapeError", None)
    orig = V.VideoBackend._encode_mp4

    def grab(video_frames, fps, audio, pipe, **k):
        data = orig(video_frames, fps, audio, pipe, **k)
        S.captured[threading.get_ident()] = (video_frames, fps)
        return data

    V.VideoBackend._encode_mp4 = staticmethod(grab)
    S.image = DiffusionBackend()
    S.video = V.VideoBackend()
    return {"tree": tree, "pid": os.getpid()}


def _classify(exc: BaseException, op: str) -> dict:
    msg = str(exc)
    status, cls = None, "fault"
    if isinstance(exc, LoadFailed):
        status, cls = None, "load_error"
    elif S.shape_error is not None and isinstance(exc, S.shape_error):
        status, cls = 422, "refused"
    elif isinstance(exc, (ValueError, FileNotFoundError)):
        status, cls = 400, "refused"
    elif isinstance(exc, RuntimeError) and (msg in S.sentinels.get(409, set()) or op in ("load", "load_async")
                                            or type(exc).__name__ == "DiffusionModelReplacedError"):
        status, cls = 409, "refused"
    elif isinstance(exc, TimeoutError):
        cls = "hang"
    return {"cls": cls, "status": status, "detail": msg[:2000], "exc": type(exc).__name__,
            "tb": traceback.format_exc()[-3000:] if cls == "fault" else None}


def _engine(kind: str):
    return S.video if kind == "video" else S.image


def _wait_ready(engine, timeout: float) -> dict:
    deadline = time.time() + timeout
    null_since = None
    while True:
        p = engine.load_progress() or {}
        if p.get("phase") == "ready":
            return engine.status()
        if p.get("phase") == "error":
            raise LoadFailed(str(p.get("error")))
        if p.get("phase") is None:
            null_since = null_since or time.time()
            if time.time() - null_since > 30:
                raise LoadFailed("load ended without a model or an error")
        if time.time() > deadline:
            raise TimeoutError(f"load not ready in {timeout}s: {p}")
        time.sleep(0.25)


def op_load(a: dict, wait: bool = True) -> dict:
    from backends.studio_inproc import load_kwargs, status_subset

    kind = a.get("kind", "image")
    engine = _engine(kind)
    kw = load_kwargs(a.get("opts") or {}, kind)
    engine.begin_load(a["model"], local_files_only = bool(a.get("local_files_only", True)), **kw)
    if not wait:
        return {"started": True}
    return status_subset(_wait_ready(engine, float(a.get("timeout", 1800))))


def _save_images(images, out_dir: Path, stem: str) -> list:
    import numpy as np

    paths = []
    for i, img in enumerate(images):
        if not hasattr(img, "convert"):
            from common import to_pil

            img = to_pil(img)
        p = out_dir / f"{stem}_{i}.npy"
        np.save(p, np.asarray(img.convert("RGB")))
        paths.append(str(p))
    return paths


def op_generate(a: dict) -> dict:
    import numpy as np

    kind = a.get("kind", "image")
    params = dict(a.get("params") or {})
    out_dir = Path(a["out_dir"])
    out_dir.mkdir(parents = True, exist_ok = True)
    stem = a.get("stem") or f"g{time.time_ns()}"
    t0 = time.perf_counter()
    res = _engine(kind).generate(**params)
    wall = time.perf_counter() - t0
    if kind == "video":
        frames, fps = S.captured.pop(threading.get_ident(), (None, res.get("fps")))
        from common import frames_array

        path = out_dir / f"{stem}.npy"
        np.save(path, frames_array(frames) if frames is not None else np.zeros((0, 1, 1, 3), np.uint8))
        mp4 = out_dir / f"{stem}.mp4"
        mp4.write_bytes(res.get("mp4_bytes") or b"")
        meta = {k: v for k, v in res.items() if k != "mp4_bytes"}
        return {"frames": str(path), "mp4": str(mp4), "meta": meta, "wall_s": wall}
    meta = {k: v for k, v in res.items() if k != "images"}
    return {"images": _save_images(res["images"], out_dir, stem), "meta": meta, "wall_s": wall}


def op_mem(a: dict) -> dict:
    import torch

    from common import host_memory

    out = host_memory()
    if torch.cuda.is_available():
        out["torch_alloc_mib"] = round(torch.cuda.memory_allocated() / 2**20, 1)
        out["torch_reserved_mib"] = round(torch.cuda.memory_reserved() / 2**20, 1)
    return out


def op_gc(a: dict) -> dict:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    return op_mem(a)


def op_trim(a: dict) -> dict:
    """gc + glibc malloc_trim: separates memory the allocator merely kept from memory still referenced."""
    import ctypes

    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass
    return op_mem(a)


def handle(req: dict) -> None:
    rid, op, a = req.get("id"), req.get("op"), req.get("args") or {}
    try:
        if op == "setup":
            result = _setup(a)
        elif op == "load":
            result = op_load(a)
        elif op == "load_async":
            result = op_load(a, wait = False)
        elif op == "wait_loaded":
            from backends.studio_inproc import status_subset

            result = status_subset(_wait_ready(_engine(a.get("kind", "image")), float(a.get("timeout", 1800))))
        elif op == "generate":
            result = op_generate(a)
        elif op == "cancel":
            result = {"cancelled": bool(_engine(a.get("kind", "image")).cancel_generate())}
        elif op == "unload":
            from backends.studio_inproc import status_subset

            result = status_subset(_engine(a.get("kind", "image")).unload())
        elif op == "status":
            from backends.studio_inproc import status_subset

            result = status_subset(_engine(a.get("kind", "image")).status())
        elif op == "progress":
            eng = _engine(a.get("kind", "image"))
            result = {"generate": eng.generate_progress(), "load": eng.load_progress()}
        elif op == "mem":
            result = op_mem(a)
        elif op == "gc":
            result = op_gc(a)
        elif op == "trim":
            result = op_trim(a)
        elif op == "ping":
            result = {"pid": os.getpid()}
        else:
            raise KeyError(f"unknown op {op}")
        send({"id": rid, "ok": True, "result": result})
    except BaseException as exc:  # noqa: BLE001 - every failure is a classified reply
        send({"id": rid, "ok": False, "error": _classify(exc, op)})


def main() -> None:
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        os.environ.pop(key, None)
    send({"id": 0, "ok": True, "result": {"ready": True, "pid": os.getpid()}})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        if req.get("op") == "exit":
            send({"id": req.get("id"), "ok": True, "result": {"bye": True}})
            break
        if req.get("op") == "setup":
            handle(req)  # everything after depends on it
        else:
            threading.Thread(target = handle, args = (req,), daemon = True).start()
    os._exit(0)


if __name__ == "__main__":
    main()
