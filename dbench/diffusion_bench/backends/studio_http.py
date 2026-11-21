"""Unsloth Studio or Unsloth Desktop over its HTTP API: the path a user's browser (or the Desktop shell) takes,
auth, route validation, the engine router, the GPU arbiter and the gallery included.

Server: attach when DIFFUSION_BENCH_STUDIO_URL (or options.base_url) is set, with _USER / _PASSWORD or _TOKEN;
otherwise a Studio is launched for this cell from options.studio_src (setup_studio.launch, its own
UNSLOTH_STUDIO_HOME under $WORKSPACE/temp, on this process's first visible GPU) and stopped in close().

Cell fields match the in-process backend: options.<load key> / options.load go into the load request body,
options.generate into the generate body. Images are read back from the gallery file route (the PNG the UI shows);
clips from the gallery MP4, decoded to frames. options.keep_gallery=false (the default on an attached server)
deletes each item after it is read, so a bench does not fill a user's gallery.

Timing: run_cell times the whole HTTP round trip, which is what a client sees. Per render the record also carries
``server_s`` (the generate request alone for images; for video the start-to-terminal poll span) and, for video,
the denoise span observed from generate-progress.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

from .base import Backend, Render
from .studio_inproc import (IMAGE_ONLY, LOAD_KEYS, VIDEO_ONLY, ProgressPoller, status_subset, studio_precision,
                            studio_resolved, studio_sampler)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import setup_studio  # noqa: E402
from studio_client import StudioClient, StudioError, decode_mp4, decode_png  # noqa: E402


def load_body(opts: dict, kind: str) -> dict:
    kw = {k: opts[k] for k in LOAD_KEYS if k in opts and opts[k] is not None}
    kw.update(opts.get("load") or {})
    for k in (IMAGE_ONLY if kind == "video" else VIDEO_ONLY):
        kw.pop(k, None)
    return kw


def generate_fields(cell: dict, opts: dict, steps: int, seed: int, kind: str) -> dict:
    """The generate request fields for one render. An image request sends enhance_prompt=false unless the cell asks
    otherwise: Studio's image route rewrites the prompt with an LLM by default for FLUX.3 klein and MiniMax-H3-Image,
    while the baseline-UI arm runs the template with its enhancer switched off (card rule enhancer_off) and the
    in-process backend never rewrites."""
    gen = dict(opts.get("generate") or {})
    out = {"negative_prompt": cell.get("negative_prompt"), "width": cell.get("width"), "height": cell.get("height"),
           "steps": steps, "guidance": cell.get("guidance"), "seed": seed, **gen}
    if kind != "video":
        out.setdefault("enhance_prompt", False)
    return out


def connect(server: "setup_studio.StudioServer") -> StudioClient:
    client = StudioClient(server.base_url, server.username, server.password, server.token)
    if server.password:
        try:
            client.login()
        except StudioError as exc:
            boot = server.extra.get("bootstrap_password")
            if not boot:
                raise
            client.password = boot
            client.login()
            if exc.status:
                client.change_password(server.password)
    return client


class StudioHttpBackend(Backend):
    name = "studio_http"
    in_process = False

    def __init__(self, cell: dict, out: Path):
        super().__init__(cell, out)
        self.kind = cell.get("kind", "image")
        self.server: Optional[setup_studio.StudioServer] = None
        self.client: Optional[StudioClient] = None
        self.keep = self.opts.get("keep_gallery")

    def load(self) -> dict:
        t0 = time.perf_counter()
        # A launched server gets the budget / parity shim (DBENCH_* + DBENCH_PYTHONPATH_PREFIX) and the per-cell
        # Inductor / Triton caches; an attached one is outside our control and the record says so.
        keep = ("DBENCH_", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR")
        shim_env = {k: v for k, v in (self.server_env or {}).items() if k.startswith(keep)}
        self.opts["server_env"] = {**shim_env, **(self.opts.get("server_env") or {})}
        self.server = setup_studio.server_from_env_or_launch(self.opts)
        start_s = round(time.perf_counter() - t0, 2)
        if self.keep is None:
            self.keep = not self.server.attached
        self.client = connect(self.server)
        body = load_body(self.opts, self.kind)
        timeout = float(self.opts.get("load_timeout_s", 3600))
        t1 = time.perf_counter()
        if self.kind == "video":
            st = self.client.video_load(self.cell["model"], timeout = timeout, **body)
        else:
            st = self.client.images_load(self.cell["model"], timeout = timeout, **body)
        return {**status_subset(st), "load_body": body, "server_start_s": start_s,
                "model_load_s": round(time.perf_counter() - t1, 2), "base_url": self.server.base_url,
                "attached": self.server.attached, "studio_rev": self.server.rev,
                "server_mem_after_load": setup_studio.process_memory(self.server.server_pid)}

    def render(self, row: dict, steps: int) -> Render:
        c = self.cell
        common = generate_fields(c, self.opts, steps, row["seed"], self.kind)
        self._steps = steps
        if self.kind == "video":
            t0 = time.perf_counter()
            t0_wall = time.time()
            self.client.video_generate(row["prompt"], num_frames = c.get("frames"), fps = c.get("fps"), **common)
            spans = {}
            deadline = time.time() + float(self.opts.get("render_timeout_s", 3600))
            while True:
                prog = self.client.video_generate_progress()
                now = time.perf_counter() - t0
                if prog.get("active") and int(prog.get("step") or 0) > 0:
                    spans.setdefault("first_step_s", round(now, 3))
                    spans["last_step_seen_s"] = round(now, 3)
                    if prog.get("phase") in (None, "denoise"):
                        self.mark_step(int(prog["step"]), prog.get("total_steps"), t0_wall + now)
                if not prog.get("active") and prog.get("phase") in ("completed", "failed"):
                    break
                if time.time() > deadline:
                    raise TimeoutError(f"clip not finished: {prog}")
                time.sleep(0.05)  # the terminal poll bounds the timing resolution of a clip
            if prog.get("phase") != "completed":
                raise RuntimeError(f"video generation failed: {prog.get('error')}")
            server_s = round(time.perf_counter() - t0, 3)
            vid = prog["video"]
            data = self.client.video_file(vid["id"])
            frames, fps = decode_mp4(data)
            if not self.keep:
                self._delete("video", vid["id"])
            extra = {"server_s": server_s, **spans, "video_id": vid["id"],
                     **{k: vid.get(k) for k in ("width", "height", "num_frames", "fps", "duration_s", "seed",
                                                "offload_policy")}, "decoded_frames": int(frames.shape[0]),
                     "sampling": studio_sampler(vid, None, None, self.opts.get("studio_default_sampler"))}
            self._last_extra = extra
            return Render(frames = frames, fps = vid.get("fps") or (round(fps) if fps else None), extra = extra)
        poller_client = StudioClient(self.client.base_url, self.client.username, self.client.password,
                                     self.client.token)
        poller_client.token = self.client.token
        with ProgressPoller(self, lambda: poller_client.images_generate_progress(), interval = 0.05) as pp:
            res = self.client.images_generate(row["prompt"], **common)
        self.request_facts["studio_phases"] = pp.phases
        server_s = round(self.client.last_elapsed_s or 0, 3)
        rec = res["images"][0]
        image = decode_png(self.client.images_file(rec["id"]))
        if not self.keep:
            for r in res["images"]:
                self._delete("images", r["id"])
        extra = {"server_s": server_s, "image_id": rec["id"], "seed": rec.get("seed"),
                 "applied_flow_shift": rec.get("applied_flow_shift"),
                 "sampling": studio_sampler(rec, None, None, self.opts.get("studio_default_sampler"))}
        if len(res["images"]) > 1:
            extra["batch"] = len(res["images"])
        self._last_extra = extra
        return Render(image = image, extra = extra)

    def _delete(self, kind: str, item_id: str) -> None:
        try:
            (self.client.video_delete if kind == "video" else self.client.images_delete)(item_id)
        except StudioError:
            pass

    def extra_pids(self) -> list:
        pid = getattr(getattr(self, "server", None), "server_pid", None)
        return [pid] if pid else []

    def status(self) -> dict:
        if self.client is None:
            return {}
        st = self.client.video_status() if self.kind == "video" else self.client.images_status()
        return {**status_subset(st), "server_mem": setup_studio.process_memory(self.server.server_pid)}

    def resolved_settings(self) -> dict:
        return studio_resolved(self.status(), self.cell, getattr(self, "_steps", self.cell.get("steps")),
                               getattr(self, "_last_extra", None))

    def precision(self) -> dict:
        return studio_precision(self.status())

    def close(self) -> None:
        try:
            if self.client is not None and self.opts.get("unload_on_close", True):
                try:
                    (self.client.video_unload if self.kind == "video" else self.client.images_unload)()
                except StudioError:
                    pass
        finally:
            if self.server is not None and not self.server.attached:
                setup_studio.stop(self.server)

    def trees(self) -> dict:
        return {"studio": self.server.tree} if self.server is not None and self.server.tree else {}

    # Nothing on the device belongs to this process: the driver-level GpuSampler is the VRAM figure.
    def sync(self) -> None:
        return None

    def reset_peak(self) -> None:
        return None

    def peak_alloc_gib(self) -> Optional[float]:
        return None
