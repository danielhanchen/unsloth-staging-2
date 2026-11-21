"""stable-diffusion.cpp (the unslothai fork Studio ships), GGUF models, as a subprocess.

Two modes:
  server (default when the build has sd-server): one sd-server per cell loads the model once; each render is a
         POST /sdcpp/v1/img_gen + job poll, the path Studio's native tier uses (sd_cpp_server.py).
  cli    one sd-cli per render: the model is re-read from disk every time, which is what the older native tier
         did; wall time includes the load, the per-step figure does not.

Cell options:
  family   (or cell["family"]) picks the text-encoder flags, as Studio's sd_cpp_args._TE_FLAGS_BY_FAMILY:
           z-image / qwen-image-2.1 / flux.2-* -> --llm, qwen-image -> --qwen2vl, flux.1 -> --clip_l + --t5xxl,
           sd / sdxl -> none (single-file -m)
  files    {"dit": gguf (--diffusion-model), "model": full checkpoint (-m), "te": path | [paths], "vae": path,
            "llm_vision": path}
  sdcpp    {"ref", "backend", "method"} -> setup_sdcpp.ensure; DIFFUSION_BENCH_SDCPP_BIN reuses a binary
  mode     server | cli;  sampler (--sampling-method), scheduler, flow_shift, cfg (default: cell guidance)
  fa       true (default) -> --diffusion-fa;  sage -> --sage-attn (needs fa);  offload_to_cpu, vae_tiling
  rng      --rng: cpu draws the initial noise the way BlineUI does (torch CPU generator), cuda (sd.cpp's default)
           the sd-webui way; with rng cpu and the same sampler / scheduler, sd.cpp and BlineUI start from the same
           latent noise for the same seed
  extra_args  list of more flags (last wins);  threads;  startup_timeout_s 900, render_timeout_s 3600
  expect_device  e.g. "CUDA": fail the load when the binary's --list-devices has no such device (a CPU-only build
           on a GPU host is the silent 30x slowdown this catches)

Per-render facts from the process log: step_s (the last N/N progress rate), sampling_s ("sampling completed,
taking"), generate_s ("generate_image completed in"), load lines (CLI mode).
"""

from __future__ import annotations

import base64
import io
import json
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from .base import Backend, Render

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

TE_FLAGS = {
    "z-image": ("--llm",), "qwen-image-2.1": ("--llm",), "flux.2-klein": ("--llm",), "flux.2-dev": ("--llm",),
    "qwen-image": ("--qwen2vl",), "flux.1": ("--clip_l", "--t5xxl"), "sd": (), "sdxl": (), "sdxl-turbo": (),
}
ALIASES = {"z-image-turbo": "z-image", "zimage": "z-image", "qwen21": "qwen-image-2.1", "qwen-image-21": "qwen-image-2.1",
           "flux.1-schnell": "flux.1", "flux.1-dev": "flux.1", "flux": "flux.1", "flux-schnell": "flux.1"}
STEP_RE = re.compile(r"\|\s*(\d+)/(\d+) - ([\d.]+)(it/s|s/it)")
SAMPLING_RE = re.compile(r"sampling completed, taking ([\d.]+)s")
GENERATE_RE = re.compile(r"generate_image completed in ([\d.]+)s")
PARAMS_RE = re.compile(r"total params memory size = ([\d.]+)MB \(VRAM ([\d.]+)MB, RAM ([\d.]+)MB\)")


class SdCppError(RuntimeError):
    pass


def canonical(family: str) -> str:
    fam = (family or "").strip().lower()
    fam = ALIASES.get(fam, fam)
    if fam not in TE_FLAGS:
        raise KeyError(f"sdcpp: unknown family {family!r}; known: {sorted(TE_FLAGS)}")
    return fam


def model_args(family: str, files: dict) -> list:
    """sd-cli / sd-server model flags. Missing files raise FileNotFoundError before a process starts."""
    from common import expand_env

    def path(v: str) -> str:
        p = Path(expand_env(str(v)))
        if not p.exists():
            raise FileNotFoundError(f"sdcpp model file not found: {p}")
        return str(p)

    args: list = []
    if files.get("dit"):
        args += ["--diffusion-model", path(files["dit"])]
    if files.get("model"):
        args += ["-m", path(files["model"])]
    if not (files.get("dit") or files.get("model")):
        raise ValueError("sdcpp needs options.files.dit (a diffusion GGUF) or options.files.model (a full checkpoint)")
    te = files.get("te") or []
    te = [te] if isinstance(te, str) else list(te)
    flags = TE_FLAGS[family]
    if family == "flux.1" and te:
        clip_l = next((t for t in te if "clip_l" in Path(t).name.lower()), te[0])
        t5 = [t for t in te if t != clip_l]
        if not t5:
            raise ValueError("flux.1 needs te = [clip_l, t5xxl]")
        args += ["--clip_l", path(clip_l), "--t5xxl", path(t5[0])]
    elif te:
        if not flags:
            raise ValueError(f"sdcpp {family}: takes no separate text encoder, got {te}")
        if len(te) > len(flags):
            raise ValueError(f"sdcpp {family}: {len(te)} text encoders given, the family takes {len(flags)}")
        for flag, t in zip(flags, te):
            args += [flag, path(t)]
    if files.get("vae"):
        args += ["--vae", path(files["vae"])]
    if files.get("llm_vision"):
        args += ["--llm_vision", path(files["llm_vision"])]
    return args


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def parse_log(text: str) -> dict:
    text = text.replace("\r", "\n")
    out: dict = {}
    rates = [m for m in STEP_RE.findall(text) if m[0] == m[1]]
    if rates:
        v, unit = float(rates[-1][2]), rates[-1][3]
        out["step_s"] = round(1.0 / v if unit == "it/s" else v, 4)
    for key, rx in (("sampling_s", SAMPLING_RE), ("generate_s", GENERATE_RE)):
        m = rx.findall(text)
        if m:
            out[key] = float(m[-1])
    m = PARAMS_RE.findall(text)
    if m:
        out["params_mb"], out["params_vram_mb"], out["params_ram_mb"] = (float(x) for x in m[-1])
    return out


class SdCppBackend(Backend):
    name = "sdcpp"
    in_process = False

    def __init__(self, cell: dict, out: Path):
        super().__init__(cell, out)
        self.work = self.out / "_sdcpp"
        self.srv: Optional[subprocess.Popen] = None
        self.logf = None
        self.renders = 0
        self.info: dict = {}
        self.mode: Optional[str] = None

    # ------------------------------------------------------------------------------------------ helpers
    def _log_text(self, path: Path, start: int = 0) -> str:
        if self.logf:
            self.logf.flush()
        try:
            with open(path, "rb") as f:
                f.seek(start)
                return f.read().decode(errors = "replace")
        except FileNotFoundError:
            return ""

    def _tail(self, path: Path, n: int = 30) -> str:
        lines = [ln for ln in self._log_text(path).replace("\r", "\n").splitlines() if ln.strip()]
        return "\n".join(lines[-n:])

    def _common_flags(self) -> list:
        o = self.opts
        flags = []
        if o.get("fa", True):
            flags.append("--diffusion-fa")
        if o.get("sage"):
            flags.append("--sage-attn")
        if o.get("offload_to_cpu"):
            flags.append("--offload-to-cpu")
        if o.get("vae_tiling"):
            flags.append("--vae-tiling")
        if o.get("threads"):
            flags += ["--threads", str(int(o["threads"]))]
        if o.get("rng"):  # cpu = BlineUI's noise (torch CPU generator), cuda = sd-webui's, std_default
            flags += ["--rng", str(o["rng"])]
        return flags

    def _gen_flags(self, row: dict, steps: int) -> list:
        o, c = self.opts, self.cell
        flags = ["-p", row["prompt"], "--steps", str(int(steps)), "-W", str(int(c["width"])), "-H", str(int(c["height"])),
                 "--seed", str(int(row["seed"])), "--cfg-scale", str(float(o.get("cfg", c.get("guidance") or 1.0)))]
        if c.get("negative_prompt"):
            flags += ["-n", c["negative_prompt"]]
        if o.get("sampler"):
            flags += ["--sampling-method", o["sampler"]]
        if o.get("scheduler"):
            flags += ["--scheduler", o["scheduler"]]
        if o.get("flow_shift") is not None:
            flags += ["--flow-shift", str(float(o["flow_shift"]))]
        return flags

    def _env(self) -> dict:
        import setup_sdcpp

        return {**setup_sdcpp.runtime_env(self.info["bin"])}

    # ------------------------------------------------------------------------------------------ lifecycle
    def load(self) -> dict:
        import setup_sdcpp

        fam = canonical(self.opts.get("family") or self.cell.get("family") or "")
        self.margs = model_args(fam, dict(self.opts.get("files") or {}))
        if self.cell.get("kind") == "video":
            raise ValueError("sdcpp backend renders images only (vid_gen is not wired here)")
        s = dict(self.opts.get("sdcpp") or {})
        self.info = setup_sdcpp.ensure(sdcpp_ref = s.get("ref"), sdcpp_backend = s.get("backend"),
                                       sdcpp_method = s.get("method"))
        binary = self.info.get("bin")
        if not binary or not Path(binary).exists():
            raise SdCppError(f"sd-cli not found: {binary!r} (set DIFFUSION_BENCH_SDCPP_BIN or build with setup_sdcpp.py)")
        devices = setup_sdcpp.binary_backends(binary)
        want = self.opts.get("expect_device")
        # [] means the binary predates --list-devices (it prints its help instead): unknown, not "no device".
        if want and devices and not any(d.upper().startswith(str(want).upper()) for d in devices):
            raise SdCppError(f"sd.cpp binary {binary} has no {want} device (it sees {devices or 'nothing'}): wrong "
                             f"backend build for this host")
        self.mode = self.opts.get("mode") or ("server" if self.info.get("server") else "cli")
        if self.mode == "server" and not self.info.get("server"):
            raise SdCppError(f"mode server asked but this build has no sd-server next to {binary}")
        self.work.mkdir(parents = True, exist_ok = True)
        status = {"backend": "sdcpp", "family": fam, "mode": self.mode, "bin": binary, "devices": devices,
                  "commit": self.info.get("commit"), "ref": self.info.get("ref"), "build_backend": self.info.get("backend"),
                  "method": self.info.get("method"), "flags": self._common_flags() + list(self.opts.get("extra_args") or [])}
        if self.mode == "server":
            status.update(self._start_server())
        return status

    def _start_server(self) -> dict:
        self.port = int(self.opts.get("port") or free_port())
        scratch = self.work / "scratch"
        scratch.mkdir(parents = True, exist_ok = True)
        cmd = [self.info["server"], *self.margs, "--listen-ip", "127.0.0.1", "--listen-port", str(self.port),
               "--lora-model-dir", str(scratch), "--hires-upscalers-dir", str(scratch), "--embd-dir", str(scratch),
               *self._common_flags(), *[str(a) for a in self.opts.get("extra_args") or []]]
        self.server_log = self.work / "server.log"
        self.logf = open(self.server_log, "wb")
        self.srv = subprocess.Popen(cmd, stdout = self.logf, stderr = subprocess.STDOUT, stdin = subprocess.DEVNULL,
                                    env = self._env(), start_new_session = True)
        deadline = time.time() + float(self.opts.get("startup_timeout_s", 900))
        while True:
            if self.srv.poll() is not None:
                raise SdCppError(f"sd-server exited with code {self.srv.returncode} during load:\n"
                                 f"{self._tail(self.server_log)}")
            try:
                self._http("GET", "/v1/models", timeout = 5)
                break
            except Exception:  # noqa: BLE001 - binds only after the model loads
                if time.time() > deadline:
                    raise SdCppError(f"sd-server not ready within the startup timeout:\n{self._tail(self.server_log)}")
                time.sleep(0.5)
        facts = parse_log(self._log_text(self.server_log))
        return {"port": self.port, "cmd": cmd[1:], **{k: v for k, v in facts.items() if k.startswith("params")}}

    def _http(self, method: str, path: str, payload: Optional[dict] = None, timeout: float = 30):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", method = method,
                                     data = None if payload is None else json.dumps(payload).encode(),
                                     headers = {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout = timeout) as r:
                return json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as exc:
            raise SdCppError(f"sd-server {path} -> HTTP {exc.code}: {exc.read().decode(errors = 'replace')[:800]}") from None

    def render(self, row: dict, steps: int) -> Render:
        self.renders += 1
        if self.mode == "server":
            return self._render_server(row, steps)
        return self._render_cli(row, steps)

    def _render_cli(self, row: dict, steps: int) -> Render:
        from PIL import Image

        png = self.work / f"r{self.renders:04d}.png"
        log_path = self.work / f"r{self.renders:04d}.log"
        cmd = [self.info["bin"], "--mode", "img_gen", *self.margs, *self._common_flags(), *self._gen_flags(row, steps),
               "-o", str(png), *[str(a) for a in self.opts.get("extra_args") or []]]
        with open(log_path, "wb") as lf:
            proc = subprocess.Popen(cmd, stdout = lf, stderr = subprocess.STDOUT, stdin = subprocess.DEVNULL,
                                    env = self._env(), start_new_session = True)
            try:
                rc = proc.wait(timeout = float(self.opts.get("render_timeout_s", 3600)))
            except subprocess.TimeoutExpired:
                import common as C

                C.kill_tree(proc, timeout = 5)  # killpg is POSIX-only; this works on Windows too
                raise SdCppError(f"sd-cli timed out; log {log_path}") from None
        if rc != 0 or not png.exists():
            raise SdCppError(f"sd-cli exited {rc}{'' if png.exists() else ' and wrote no image'}:\n{self._tail(log_path)}")
        facts = parse_log(self._log_text(log_path))
        with Image.open(png) as im:
            img = im.convert("RGB")
        if not self.opts.get("keep_raw"):
            png.unlink(missing_ok = True)
        return Render(image = img, step_s = facts.pop("step_s", None), extra = facts)

    def _render_server(self, row: dict, steps: int) -> Render:
        from PIL import Image

        if self.srv is None or self.srv.poll() is not None:
            raise SdCppError(f"sd-server is not running:\n{self._tail(self.server_log)}")
        o, c = self.opts, self.cell
        sample: dict = {"sample_steps": int(steps), "guidance": {"txt_cfg": float(o.get("cfg", c.get("guidance") or 1.0))}}
        if o.get("sampler"):
            sample["sample_method"] = o["sampler"]
        if o.get("scheduler"):
            sample["scheduler"] = o["scheduler"]
        if o.get("flow_shift") is not None:
            sample["flow_shift"] = float(o["flow_shift"])
        body = {"prompt": row["prompt"], "negative_prompt": c.get("negative_prompt") or "", "width": int(c["width"]),
                "height": int(c["height"]), "batch_count": 1, "output_format": "png", "seed": int(row["seed"]),
                "sample_params": sample}
        start = self.server_log.stat().st_size if self.server_log.exists() else 0
        job = self._http("POST", "/sdcpp/v1/img_gen", body)
        jid = job.get("id")
        if not jid:
            raise SdCppError(f"sd-server returned no job id: {job}")
        deadline = time.time() + float(o.get("render_timeout_s", 3600))
        while True:
            jd = self._http("GET", f"/sdcpp/v1/jobs/{jid}")
            if jd.get("status") == "completed":
                break
            if jd.get("status") in ("failed", "cancelled"):
                raise SdCppError(f"sd-server job {jd.get('status')}: {jd.get('error')}\n{self._tail(self.server_log, 15)}")
            if self.srv.poll() is not None:
                raise SdCppError(f"sd-server died mid-render (exit {self.srv.returncode}):\n{self._tail(self.server_log)}")
            if time.time() > deadline:
                raise SdCppError("sd-server render timed out")
            time.sleep(0.05)
        images = ((jd.get("result") or {}).get("images") or [])
        b64 = next((i.get("b64_json") for i in images if isinstance(i, dict) and i.get("b64_json")), None)
        if not b64:
            raise SdCppError(f"sd-server completed with no image: {str(jd)[:500]}")
        img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        facts = parse_log(self._log_text(self.server_log, start))
        facts = {k: v for k, v in facts.items() if not k.startswith("params")}
        return Render(image = img, step_s = facts.pop("step_s", None), extra = facts)

    def status(self) -> dict:
        return {"server_alive": bool(self.srv and self.srv.poll() is None)} if self.mode == "server" else {}

    def close(self) -> None:
        if self.srv is not None and self.srv.poll() is None:
            import common as C

            C.kill_tree(self.srv, timeout = 20)
        if self.logf:
            self.logf.close()
            self.logf = None

    def trees(self) -> dict:
        return {"sdcpp": self.info.get("src")} if self.info.get("src") else {}

    def sync(self) -> None:
        pass

    def reset_peak(self) -> None:
        pass
