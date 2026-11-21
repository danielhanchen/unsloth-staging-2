"""A small stdlib-only (urllib) client for the image and video routes of Unsloth Studio / Unsloth Desktop.

Every call returns parsed JSON (or raw bytes for files) and raises ``StudioError`` carrying the HTTP status and the
response body on anything but 2xx, so a caller can tell a clean refusal (400 / 409 / 422) from a server fault (5xx).
Routes and fields follow studio/backend/routes/inference.py, routes/video.py and models/inference.py:

  POST /api/auth/login                       {username, password} -> {access_token, refresh_token, must_change_password}
  POST /api/auth/change-password             {current_password, new_password}
  GET  /api/health
  POST /api/inference/images/download-plan   DiffusionLoadRequest -> {entries, total_bytes, incompatible_reason, ...}
  POST /api/inference/images/load            DiffusionLoadRequest -> status (the load runs in the background)
  GET  /api/inference/images/load-progress   {phase: downloading|finalizing|ready|error|null, error, fraction, ...}
  POST /api/inference/images/generate        DiffusionGenerateRequest -> {images: [GalleryImage]} (synchronous)
  GET  /api/inference/images/generate-progress {active, step, total_steps, fraction, eta_seconds}
  POST /api/inference/images/generate/cancel -> {cancelled}
  GET  /api/inference/images/status | POST /images/unload | GET /images/info
  GET  /api/inference/images/gallery?limit&offset | GET /images/gallery/{id}/file (PNG) | DELETE /images/gallery/{id}
  POST /api/inference/video/{download-plan,load,unload}, GET /video/{load-progress,status,generate-progress}
  POST /api/inference/video/generate         VideoGenerateRequest -> {status: "started"} (a background job)
  POST /api/inference/video/generate/cancel  GET /video/gallery, /video/gallery/{id}/file (MP4)
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Optional

API = "/api/inference"


class StudioError(RuntimeError):
    def __init__(self, method: str, path: str, status: int, body: Any):
        self.method, self.path, self.status, self.body = method, path, status, body
        detail = body.get("detail") if isinstance(body, dict) else body
        super().__init__(f"{method} {path} -> HTTP {status}: {str(detail)[:600]}")

    @property
    def detail(self) -> str:
        return str(self.body.get("detail") if isinstance(self.body, dict) else self.body)

    @property
    def clean_refusal(self) -> bool:
        """4xx: the server understood and declined. 5xx / 0 (no response) is a fault."""
        return 400 <= self.status < 500


class StudioClient:
    def __init__(self, base_url: str, username: str = "unsloth", password: Optional[str] = None,
                 token: Optional[str] = None, timeout: float = 600.0):
        self.base_url = base_url.rstrip("/")
        self.username, self.password, self.token = username, password, token
        self.timeout = timeout
        self.last_elapsed_s: Optional[float] = None
        self.last_headers: dict = {}

    # ------------------------------------------------------------------------------------------ transport
    def request(self, method: str, path: str, body: Any = None, params: Optional[dict] = None, raw: bool = False,
                auth: bool = True, timeout: Optional[float] = None, headers: Optional[dict] = None,
                _retry: bool = True) -> Any:
        url = self.base_url + path
        if params:
            url += "?" + urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        data = None
        hdrs = {"Accept": "application/json", **(headers or {})}
        if body is not None:
            data = json.dumps(body).encode()
            hdrs["Content-Type"] = "application/json"
        if auth and self.token:
            hdrs["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(url, data = data, method = method, headers = hdrs)
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout = timeout or self.timeout) as r:
                payload = r.read()
                self.last_headers = dict(r.headers.items())
        except urllib.error.HTTPError as e:
            self.last_elapsed_s = time.perf_counter() - t0
            text = e.read()
            try:
                parsed = json.loads(text)
            except Exception:  # noqa: BLE001
                parsed = text.decode(errors = "replace")[:4000]
            if e.code == 401 and auth and _retry and self.password and path != "/api/auth/login":
                self.login()  # an expired access token: log in once and replay
                return self.request(method, path, body, params, raw, auth, timeout, headers, _retry = False)
            raise StudioError(method, path, e.code, parsed) from None
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
            self.last_elapsed_s = time.perf_counter() - t0
            raise StudioError(method, path, 0, f"{type(e).__name__}: {e}") from None
        self.last_elapsed_s = time.perf_counter() - t0
        if raw:
            return payload
        if not payload:
            return None
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return payload.decode(errors = "replace")

    def get(self, path: str, **kw) -> Any:
        return self.request("GET", path, **kw)

    def post(self, path: str, body: Any = None, **kw) -> Any:
        return self.request("POST", path, body = {} if body is None else body, **kw)

    # ------------------------------------------------------------------------------------------ auth
    def login(self) -> dict:
        if not self.password:
            if self.token:
                return {"access_token": self.token}
            raise ValueError("login needs a password (or pass token=)")
        out = self.request("POST", "/api/auth/login", {"username": self.username, "password": self.password},
                           auth = False, _retry = False)
        self.token = out["access_token"]
        if out.get("must_change_password"):
            raise StudioError("POST", "/api/auth/login", 403, {"detail": "password change required; "
                                                                "call change_password(new) first"})
        return out

    def change_password(self, new_password: str) -> dict:
        out = self.post("/api/auth/change-password", {"current_password": self.password,
                                                      "new_password": new_password})
        self.password = new_password
        if isinstance(out, dict) and out.get("access_token"):
            self.token = out["access_token"]
        return out

    def health(self) -> Any:
        return self.get("/api/health", auth = False, timeout = 10)

    # ------------------------------------------------------------------------------------------ media, shared
    def _load_payload(self, model_path: str, **fields) -> dict:
        body = {"model_path": model_path}
        body.update({k: v for k, v in fields.items() if v is not None})
        if "loras" in body and body["loras"] and not isinstance(body["loras"][0], dict):
            body["loras"] = [{"id": lid, "weight": w} for lid, w in body["loras"]]
        return body

    def _wait_loaded(self, kind: str, timeout: float, poll: float, expect: Optional[str] = None) -> dict:
        """Poll load-progress until ready (returns status) or error (raises StudioError 0 with the load error).
        A phase of null with nothing loaded for > 20 s means the load thread died without reporting."""
        deadline = time.time() + timeout
        t_null = None
        while True:
            prog = self.get(f"{API}/{kind}/load-progress", timeout = 30)
            phase = (prog or {}).get("phase")
            if phase == "error":
                raise StudioError("GET", f"{API}/{kind}/load-progress", 0, {"detail": prog.get("error"),
                                                                            "phase": "error"})
            if phase in ("ready", None):
                st = self.get(f"{API}/{kind}/status", timeout = 30)
                if st.get("loaded") and (phase == "ready" or not expect or expect in str(st.get("repo_id"))):
                    return st
                if phase is None:
                    t_null = t_null or time.time()
                    if time.time() - t_null > 20:
                        raise StudioError("GET", f"{API}/{kind}/load-progress", 0,
                                          {"detail": "load finished without a loaded model or an error", "status": st})
            if time.time() > deadline:
                raise StudioError("GET", f"{API}/{kind}/load-progress", 0, {"detail": f"load not ready in {timeout}s",
                                                                            "progress": prog})
            time.sleep(poll)

    # ------------------------------------------------------------------------------------------ images
    def images_download_plan(self, model_path: str, **fields) -> dict:
        return self.post(f"{API}/images/download-plan", self._load_payload(model_path, **fields))

    def images_load(self, model_path: str, wait: bool = True, timeout: float = 1800, poll: float = 1.0,
                    **fields) -> dict:
        started = self.post(f"{API}/images/load", self._load_payload(model_path, **fields))
        if not wait:
            return started
        return self._wait_loaded("images", timeout, poll)

    def images_load_progress(self) -> dict:
        return self.get(f"{API}/images/load-progress")

    def images_generate(self, prompt: str, timeout: Optional[float] = None, **fields) -> dict:
        body = {"prompt": prompt, **{k: v for k, v in fields.items() if v is not None}}
        return self.post(f"{API}/images/generate", body, timeout = timeout)

    def images_generate_progress(self) -> dict:
        return self.get(f"{API}/images/generate-progress", timeout = 30)

    def images_cancel(self) -> dict:
        return self.post(f"{API}/images/generate/cancel", timeout = 60)

    def images_status(self) -> dict:
        return self.get(f"{API}/images/status", timeout = 30)

    def images_unload(self) -> dict:
        return self.post(f"{API}/images/unload")

    def images_info(self) -> dict:
        return self.get(f"{API}/images/info")

    def images_gallery(self, limit: int = 50, offset: int = 0, archived: bool = False) -> dict:
        return self.get(f"{API}/images/gallery", params = {"limit": limit, "offset": offset,
                                                             "archived": str(archived).lower()})

    def images_file(self, image_id: str) -> bytes:
        return self.get(f"{API}/images/gallery/{urllib.parse.quote(image_id)}/file", raw = True)

    def images_delete(self, image_id: str) -> Any:
        return self.request("DELETE", f"{API}/images/gallery/{urllib.parse.quote(image_id)}")

    # ------------------------------------------------------------------------------------------ video
    def video_download_plan(self, model_path: str, **fields) -> dict:
        return self.post(f"{API}/video/download-plan", self._load_payload(model_path, **fields))

    def video_load(self, model_path: str, wait: bool = True, timeout: float = 1800, poll: float = 1.0,
                   **fields) -> dict:
        started = self.post(f"{API}/video/load", self._load_payload(model_path, **fields))
        if not wait:
            return started
        return self._wait_loaded("video", timeout, poll)

    def video_load_progress(self) -> dict:
        return self.get(f"{API}/video/load-progress")

    def video_generate(self, prompt: str, **fields) -> dict:
        """Start a clip (returns {"status": "started"}); ``video_wait`` polls it to its terminal state."""
        body = {"prompt": prompt, **{k: v for k, v in fields.items() if v is not None}}
        return self.post(f"{API}/video/generate", body, timeout = 120)

    def video_generate_progress(self, timeout: float = 30) -> dict:
        return self.get(f"{API}/video/generate-progress", timeout = timeout)

    def video_wait(self, timeout: float = 1800, poll: float = 0.5, progress_timeout: float = 30) -> dict:
        """The terminal generate-progress record: phase completed (with ``video``) or raises on failed.
        ``progress_timeout`` bounds each poll (a poll can wait behind a slow unload on a loaded host)."""
        deadline = time.time() + timeout
        while True:
            prog = self.video_generate_progress(timeout = progress_timeout)
            if not prog.get("active"):
                if prog.get("phase") == "completed" and prog.get("video"):
                    return prog
                if prog.get("phase") == "failed" or prog.get("error"):
                    raise StudioError("GET", f"{API}/video/generate-progress", 0, {"detail": prog.get("error"),
                                                                                   "phase": prog.get("phase")})
            if time.time() > deadline:
                raise StudioError("GET", f"{API}/video/generate-progress", 0,
                                  {"detail": f"clip not finished in {timeout}s", "progress": prog})
            time.sleep(poll)

    def video_cancel(self) -> dict:
        return self.post(f"{API}/video/generate/cancel", timeout = 60)

    def video_status(self) -> dict:
        return self.get(f"{API}/video/status", timeout = 30)

    def video_unload(self) -> dict:
        return self.post(f"{API}/video/unload")

    def video_gallery(self, limit: int = 50, offset: int = 0) -> dict:
        return self.get(f"{API}/video/gallery", params = {"limit": limit, "offset": offset})

    def video_file(self, video_id: str) -> bytes:
        return self.get(f"{API}/video/gallery/{urllib.parse.quote(video_id)}/file", raw = True)

    def video_delete(self, video_id: str) -> Any:
        return self.request("DELETE", f"{API}/video/gallery/{urllib.parse.quote(video_id)}")


def decode_png(data: bytes):
    """PIL image from PNG bytes."""
    import io

    from PIL import Image

    img = Image.open(io.BytesIO(data))
    img.load()
    return img


def decode_mp4(data: bytes) -> tuple:
    """(uint8 [T, H, W, 3] frames, fps) from MP4 bytes, via PyAV (Studio's own encoder dependency) or imageio."""
    import io

    import numpy as np

    try:
        import av

        with av.open(io.BytesIO(data)) as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate) if stream.average_rate else None
            frames = [f.to_ndarray(format = "rgb24") for f in container.decode(stream)]
        return np.stack(frames), fps
    except ImportError:
        import imageio.v3 as iio

        arr = iio.imread(data, extension = ".mp4", index = None)
        meta = iio.immeta(data, extension = ".mp4")
        return np.asarray(arr), meta.get("fps")
