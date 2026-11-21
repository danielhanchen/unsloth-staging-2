"""BlineUI, headless, one server per cell, driven through its /prompt API with the family graph from bline_graphs/.

Cell options (all optional unless marked):
  family         (required, or cell["family"]) qwen-image-2.1 | z-image | flux.1 | sdxl-turbo | wan2.2-5b (+ aliases)
  files          (required) {"dit": path, "te": path | [paths], "vae": path, "checkpoint": path, "diffusers": dir};
                 symlinked into <out>/_bline/models and passed with --extra-model-paths-config
  bline          {"commit", "python", "torch", "index", "sage", "kitchen"} -> setup_blineui.ensure; the env vars
                 DIFFUSION_BENCH_BLINE_PYTHON / DIFFUSION_BENCH_BLINE_DIR reuse an existing venv / clone
  weight_dtype   UNETLoader weight_dtype (default | fp8_e4m3fn | fp8_e4m3fn_fast | fp8_e5m2)
  compile        true | "inductor" | "cudagraphs": TorchCompileModel on the model
  highvram       --highvram (keep models resident; BlineUI otherwise offloads between prompts)
  bline_args     extra server flags, a string or a list, e.g. "--fast --use-sage-attention"
  easycache      EasyCache reuse_threshold (e.g. 0.2); easycache_start / easycache_end (0.15 / 0.95)
  sampler / scheduler / shift / cfg / flux_guidance   override the template's (cfg defaults to cell guidance)
  cache_bust     all (default) | sampler | none: see bline_graphs.apply_nonce
  api_nodes      false (default): start with --disable-api-nodes so the server never calls out
  startup_timeout_s 600, render_timeout_s 3600, keep_raw false

Timing facts per render (Render.extra): exec_s (BlineUI's own execution_start -> execution_success), step_s from
the sampler's tqdm line in the server log (the last complete N/N rate), easycache skipped steps, and whether the
sampler came from cache (a cache hit is a bug in the protocol, flagged, never silently timed).

Per-step times come from the server's /ws feed (the outside tester's bench.py method): every ``progress`` event of a
sampler node is a step timestamp, ``executing`` events give node start times (sampler start / end = the denoise
phase), and the executed / cached node counts go into the request (``bline``), where the record validator and the
parity gate require executed nodes > 0 and an uncached sampler on every timed request: the nonce is only trusted
once the execution count proves it (V3-schema nodes included). load() only starts
the server: BlineUI loads model files lazily on the first prompt, so the model load lands in the warmup (cold_s).
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Optional

from .base import Backend, Render

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

IT_RE = re.compile(r"(\d+)/(\d+) \[[^\]]*?,\s*([\d.]+)(it/s|s/it)[,\]]")
EASY_RE = re.compile(r"EasyCache - skipped (\d+)/(\d+) steps")
# Server-log lines that prove an optimisation engaged; a flag BlineUI ignores leaves no line.
ENGAGED = {"sage_attention": "Using sage attention", "pytorch_attention": "Using pytorch attention",
           "flash_attention": "Using Flash Attention", "xformers_attention": "Using xformers attention",
           "fp16_accumulation": "Enabled fp16 accumulation.", "cublas_ops": "Using cublas ops",
           "highvram": "Set vram state to: HIGH_VRAM", "pinned_memory": "Enabled pinned memory",
           "mixed_precision_ops": "Using mixed precision operations"}
FACT_RE = {"device": re.compile(r"Device: (.+)"), "vram_state": re.compile(r"Set vram state to: (\S+)"),
           "blineui_version": re.compile(r"BlineUI version: (\S+)"), "torch": re.compile(r"pytorch version: (\S+)"),
           "bline_kitchen": re.compile(r"bline-kitchen version: (\S+)"),
           "native_ops": re.compile(r"Native ops: (.+)"), "model_type": re.compile(r"model_type (\S+)"),
           # the diffusion model's load line: what the transformer computes in (manual cast wins when set)
           "model_weight_dtype": re.compile(r"model weight dtype ([^,\s]+)"),
           "manual_cast": re.compile(r"manual cast: (\S+)")}
ANSI = re.compile(r"\x1b\[[0-9;]*m")
SAMPLER_CLASSES = ("KSampler", "KSamplerAdvanced", "SamplerCustom", "SamplerCustomAdvanced")
PRECISION_TOKENS = ("nvfp4", "mxfp4", "fp8_e4m3fn_fast", "fp8_e4m3fn", "fp8_e5m2", "fp8", "int8_convrot", "int8", "int4",
                    "q8_0", "q6_k", "q5_k", "q4_k", "q4_0", "bf16", "fp16", "fp32")


def precision_of(name) -> "Optional[str]":
    """The precision a model file name states (first matching token), or None."""
    if isinstance(name, (list, tuple)):
        vals = [precision_of(n) for n in name]
        return "+".join(v or "?" for v in vals) if any(vals) else None
    low = str(name or "").lower()
    return next((t for t in PRECISION_TOKENS if t in low), None)


def custom_nodes(bline_dir) -> "Optional[dict]":
    """{node dir: git commit} for every custom node checkout under <bline_dir>/custom_nodes (None when there are none):
    a GGUF loader or any other node is part of the baseline's revision."""
    out = {}
    root = Path(bline_dir) / "custom_nodes"
    for d in sorted(root.iterdir()) if root.is_dir() else []:
        if (d / ".git").exists():
            try:
                out[d.name] = subprocess.run(["git", "-C", str(d), "rev-parse", "HEAD"], capture_output = True,
                                             text = True, timeout = 10).stdout.strip() or None
            except (OSError, subprocess.SubprocessError):
                out[d.name] = None
    return out or None


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class BlineUIError(RuntimeError):
    pass


class BlineUIBackend(Backend):
    name = "blineui"
    in_process = False

    def __init__(self, cell: dict, out: Path):
        super().__init__(cell, out)
        self.srv: Optional[subprocess.Popen] = None
        self.logf = None
        self.port: Optional[int] = None
        self.work = self.out / "_bline"
        self.log_path = self.work / "server.log"
        self.renders = 0
        self.info: dict = {}
        self.ws = None
        self.ws_error: Optional[str] = None
        self.ws_events: list = []  # (t, type, data) for every JSON message
        self.ws_lock = threading.Lock()
        self.graph: dict = {}

    # ------------------------------------------------------------------------------------------ http
    def api(self, path: str, payload: Optional[dict] = None, timeout: float = 30):
        import extnames as X  # real spellings out (node classes, file names), the toolkit's back (public copy)

        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}",
                                     data = None if payload is None else X.ext(json.dumps(payload)).encode(),
                                     headers = {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout = timeout) as r:
                return json.loads(X.norm(r.read().decode("utf-8", errors = "replace")))
        except urllib.error.HTTPError as exc:
            body = X.norm(exc.read().decode(errors = "replace"))
            raise BlineUIError(f"BlineUI {path} -> HTTP {exc.code}: {self._explain(body)}") from None

    @staticmethod
    def _explain(body: str) -> str:
        """The node errors of a rejected /prompt, flattened to one line a person can act on."""
        try:
            data = json.loads(body)
        except Exception:  # noqa: BLE001
            return body[:1500]
        parts = []
        err = data.get("error") or {}
        if isinstance(err, dict) and err.get("message"):
            parts.append(f"{err.get('message')} {err.get('details') or ''}".strip())
        for node, info in (data.get("node_errors") or {}).items():
            for e in info.get("errors", []):
                parts.append(f"node {node} ({info.get('class_type')}): {e.get('message')} {e.get('details') or ''}".strip())
        return "; ".join(parts)[:2000] or body[:1500]

    def log_text(self, start: int = 0) -> str:
        if self.logf:
            self.logf.flush()
        try:
            with open(self.log_path, "rb") as f:
                f.seek(start)
                import extnames as X

                return X.norm(ANSI.sub("", f.read().decode(errors = "replace")))
        except FileNotFoundError:
            return ""

    def log_size(self) -> int:
        if self.logf:
            self.logf.flush()
        try:
            return self.log_path.stat().st_size
        except FileNotFoundError:
            return 0

    def tail(self, n: int = 40) -> str:
        lines = [ln for ln in self.log_text().replace("\r", "\n").splitlines() if ln.strip()]
        return "\n".join(lines[-n:])

    # ------------------------------------------------------------------------------------------ lifecycle
    def family(self) -> str:
        import bline_graphs

        return bline_graphs.canonical(self.opts.get("family") or self.cell.get("family") or "")

    def load(self) -> dict:
        import bline_graphs
        import setup_blineui

        fam = self.family()
        mod = bline_graphs.module(fam)
        files = dict(self.opts.get("files") or {})
        if not files:
            raise ValueError(f"blineui cell needs options.files ({', '.join(mod.FILES)})")
        unknown = set(files) - set(mod.FILES)
        if unknown:
            raise ValueError(f"blineui {fam}: unknown file roles {sorted(unknown)}; expected {sorted(mod.FILES)}")
        # Symlink the cell's files first: a missing file fails here, before a server starts.
        self.work.mkdir(parents = True, exist_ok = True)
        wiring = {}
        for role, paths in files.items():
            wiring.setdefault(mod.FILES[role], [])
            wiring[mod.FILES[role]] += [paths] if isinstance(paths, str) else list(paths)
        yaml = setup_blineui.wire_models(wiring, self.work / "models")
        self.names = {role: ([Path(p).name for p in v] if isinstance(v, list) else Path(v).name)
                      for role, v in files.items()}

        bline = dict(self.opts.get("bline") or {})
        self.info = setup_blineui.ensure(bline_commit = bline.get("commit"), python = bline.get("python"),
                                         torch = bline.get("torch"), index = bline.get("index"),
                                         kitchen = bline.get("kitchen", True), sage = bline.get("sage", False))
        bline_dir, py = Path(self.info["bline_dir"]), self.info["python"]
        self.port = int(self.opts.get("port") or free_port())
        dirs = {k: self.work / k for k in ("output", "temp", "input", "user")}
        for d in dirs.values():
            d.mkdir(parents = True, exist_ok = True)
        cmd = [py, "-u", str(bline_dir / "main.py"), "--listen", "127.0.0.1", "--port", str(self.port),
               "--disable-auto-launch", "--output-directory", str(dirs["output"]), "--temp-directory", str(dirs["temp"]),
               "--input-directory", str(dirs["input"]), "--user-directory", str(dirs["user"]),
               "--extra-model-paths-config", str(yaml)]
        if not self.opts.get("api_nodes"):
            cmd.append("--disable-api-nodes")
        if self.opts.get("highvram"):
            cmd.append("--highvram")
        extra = self.opts.get("bline_args") or []
        cmd += extra.split() if isinstance(extra, str) else [str(a) for a in extra]
        self.cmd = cmd
        self.logf = open(self.log_path, "wb")
        import extnames as X

        env = X.ext_env({**(self.server_env or os.environ), "PYTHONUNBUFFERED": "1"})
        self.srv = subprocess.Popen(cmd, cwd = str(bline_dir), stdout = self.logf, stderr = subprocess.STDOUT,
                                    stdin = subprocess.DEVNULL, env = env, start_new_session = True)
        deadline = time.time() + float(self.opts.get("startup_timeout_s", 600))
        while True:
            if self.srv.poll() is not None:
                raise BlineUIError(f"BlineUI server exited with code {self.srv.returncode} during startup "
                                   f"(log {self.log_path}):\n{self.tail(25)}")
            try:
                stats = self.api("/system_stats", timeout = 5)
                break
            except Exception:  # noqa: BLE001 - not up yet
                if time.time() > deadline:
                    raise BlineUIError(f"BlineUI did not answer /system_stats within the startup timeout:\n{self.tail(25)}")
                time.sleep(0.5)
        self.check_skipped = None
        self._check_nodes(mod)
        self._start_ws()
        text = self.log_text()
        return {"backend": "blineui", "family": fam, "bline_commit": self.info.get("commit"),
                "bline_tag": self.info.get("tag"), "bline_pin_source": self.info.get("pin_source"),
                "custom_nodes": custom_nodes(bline_dir),
                "ws": self.ws_error or "connected",
                "node_check": f"skipped ({self.check_skipped})" if self.check_skipped else "ok",
                "bline_dir": str(bline_dir), "python": py, "port": self.port, "files": self.names,
                "cmd": cmd[3:], "weight_dtype": self.opts.get("weight_dtype", "default"),
                "compile": self.opts.get("compile") or False, "easycache": self.opts.get("easycache"),
                "cache_bust": self.opts.get("cache_bust", "all"), "models_load_lazily": True,
                "engaged": {k: v in text for k, v in ENGAGED.items()}, "facts": self._facts(text),
                "system": (stats.get("system") or {}), "devices": [d.get("name") for d in stats.get("devices", [])]}

    def _check_nodes(self, mod) -> None:
        """Every node the family graph uses exists in this BlineUI, and every file is one its loader lists: a
        missing node (older commit) or an unlisted file becomes one clear error instead of a node_errors blob."""
        import bline_graphs

        steps = int(((self.cell.get("generate") or {}).get("steps")) or self.cell.get("steps") or 1)
        probe = self.graph_params(prompt = "probe", seed = 0, steps = steps, prefix = "probe", nonce = None)
        try:
            graph = bline_graphs.build(self.family(), probe)
        except Exception as exc:  # noqa: BLE001 - the same error surfaces at render; the record says the check was skipped
            self.check_skipped = f"{type(exc).__name__}: {exc}"[:300]
            return
        info = self.api("/object_info", timeout = 120)
        missing = sorted({n["class_type"] for n in graph.values() if n["class_type"] not in info})
        if missing:
            raise BlineUIError(f"this BlineUI ({self.info.get('commit', '?')[:10]}) has no node(s) {missing}; "
                               f"use a newer bline.commit")
        for nid, node in graph.items():
            spec = (info[node["class_type"]].get("input") or {}).get("required") or {}
            for key, val in node["inputs"].items():
                if not (isinstance(val, str) and key in spec and
                        (key.endswith("_name") or key.startswith("clip_name") or key == "model_path")):
                    continue
                choices = spec[key][0]
                if choices == "COMBO" and len(spec[key]) > 1:
                    choices = (spec[key][1] or {}).get("options")
                if isinstance(choices, list) and val not in choices:
                    raise BlineUIError(f"{node['class_type']}.{key}={val!r} is not in BlineUI's list "
                                       f"({len(choices)} entries, e.g. {choices[:5]})")

    @staticmethod
    def _facts(text: str) -> dict:
        out = {}
        for key, rx in FACT_RE.items():
            m = rx.findall(text)
            if m:
                out[key] = m[-1].strip()
        return out

    # ------------------------------------------------------------------------------------------ progress feed
    def _start_ws(self) -> None:
        from . import _ws

        try:
            self.ws = _ws.connect("127.0.0.1", self.port, f"/ws?clientId={self.client_id}", timeout = 30)
        except Exception as exc:  # noqa: BLE001 - per-step times become unavailable, the render still works
            self.ws, self.ws_error = None, f"{type(exc).__name__}: {exc}"
            return
        self.ws.settimeout(None)

        def loop():
            while self.ws is not None:
                try:
                    msg = self.ws.recv()
                except Exception as exc:  # noqa: BLE001
                    self.ws_error = self.ws_error or f"ws ended: {type(exc).__name__}"
                    return
                t = time.time()
                m = _ws.parse(msg)
                if m and "type" in m:
                    with self.ws_lock:
                        self.ws_events.append((t, m["type"], m.get("data") or {}))

        threading.Thread(target = loop, name = "bline-ws", daemon = True).start()

    def _prompt_events(self, pid: str, wait_s: float = 3.0) -> list:
        """This prompt's ws events, waiting briefly for its terminal event (the history poll can finish first)."""
        deadline = time.time() + wait_s
        while True:
            with self.ws_lock:
                evs = [e for e in self.ws_events if e[2].get("prompt_id") == pid]
            done = any(ty in ("execution_success", "execution_error", "execution_interrupted") or
                       (ty == "executing" and d.get("node") is None) for _, ty, d in evs)
            if done or self.ws is None or time.time() > deadline:
                return evs
            time.sleep(0.02)

    def _apply_events(self, evs: list) -> dict:
        """mark_step / mark_phase from a prompt's events; returns the execution facts for the request."""
        samplers = [k for k, n in self.graph.items() if n.get("class_type") in SAMPLER_CLASSES]
        starts: dict = {}
        order: list = []
        offsets: dict = {}
        last_val: dict = {}
        cached: list = []
        last_step_t = None
        for t, ty, d in evs:
            if ty == "executing" and d.get("node") is not None:
                if d["node"] not in starts:
                    starts[d["node"]] = t
                    order.append(d["node"])
            elif ty == "execution_cached":
                cached += list(d.get("nodes") or [])
            elif ty == "progress" and d.get("node") in samplers:
                node = d["node"]
                if node not in offsets:
                    offsets[node] = sum(last_val.values())
                last_val[node] = max(last_val.get(node, 0), int(d.get("value") or 0))
                self.mark_step(offsets[node] + int(d.get("value") or 0), d.get("max"), t)
                last_step_t = t
        ran = [n for n in samplers if n in starts]
        if ran:
            self.mark_phase("denoise_start", starts[ran[0]])
            after = [starts[n] for n in order if starts[n] > starts[ran[-1]] and n not in samplers]
            end = min(after) if after else last_step_t
            if end is not None:
                self.mark_phase("denoise_end", end)
        return {"executed_nodes": len(starts), "cached_nodes": cached,
                "sampler_cached": any(n in cached for n in samplers),  # ANY stage cached voids the timing
                "sampler_ran": bool(ran), "ws": self.ws_error or "ok"}

    def engaged_lines(self) -> list:
        text = self.log_text()
        return [ln.strip() for ln in text.splitlines() if any(v in ln for v in ENGAGED.values())]

    def resolved_settings(self) -> dict:
        """Steps / cfg / sampler / scheduler / shift / size / frames this render asked for, then the last graph sent
        for keys the params leave open. Params first: multi-stage graphs carry stage-1 latent sizes (ltx), a
        CFGOverride before the guider (ideogram) or ManualSigmas without a steps widget, so the first matching
        widget is not the cell's setting."""
        keymap = {"steps": "steps", "cfg": "cfg", "sampler_name": "sampler", "scheduler": "scheduler",
                  "shift": "shift", "width": "width", "height": "height", "length": "frames", "denoise": "denoise",
                  "guidance": "flux_guidance", "max_shift": "max_shift", "base_shift": "base_shift"}
        p = getattr(self, "_params", None)
        out: dict = {}
        if p is not None:
            for key, attr in (("steps", "steps"), ("cfg", "cfg"), ("sampler", "sampler"), ("scheduler", "scheduler"),
                              ("shift", "shift"), ("width", "width"), ("height", "height"), ("frames", "frames"),
                              ("flux_guidance", "guidance")):
                v = getattr(p, attr, None)
                if v is not None:
                    out[key] = v
        for node in self.graph.values():
            for k, v in (node.get("inputs") or {}).items():
                if k in keymap and not isinstance(v, list):
                    out.setdefault(keymap[k], v)
        out["source"] = "bline params + graph" if p is not None else "bline graph"
        return out

    def precision(self) -> dict:
        names = getattr(self, "names", {}) or {}
        wd = self.opts.get("weight_dtype", "default")
        return {"transformer": wd if wd not in (None, "default") else precision_of(names.get("dit") or
                                                                                    names.get("checkpoint")),
                "text_encoder": precision_of(names.get("te")), "vae": precision_of(names.get("vae"))}

    def graph_params(self, prompt: str, seed: int, steps: int, prefix: str, nonce):
        """The GraphParams a render sends; the load-time node check builds from the same function so it sees the
        graph the cell will run (size, frames, weight dtype, graph_extra, GGUF loader rewrite)."""
        import bline_graphs

        o, c = self.opts, self.cell
        return bline_graphs.GraphParams(
            prompt = prompt, seed = int(seed), steps = int(steps), width = int(c.get("width") or 512),
            height = int(c.get("height") or 512), negative = c.get("negative_prompt"), cfg = o.get("cfg", c.get("guidance")),
            sampler = o.get("sampler"), scheduler = o.get("scheduler"), shift = o.get("shift"),
            guidance = o.get("flux_guidance"), frames = c.get("frames"), files = self.names,
            weight_dtype = o.get("weight_dtype", "default"), compile = o.get("compile"), easycache = o.get("easycache"),
            easycache_start = float(o.get("easycache_start", 0.15)), easycache_end = float(o.get("easycache_end", 0.95)),
            prefix = prefix, nonce = nonce, nonce_nodes = o.get("cache_bust", "all"), extra = dict(o.get("graph_extra") or {}),
            # video graphs take the cell's fps (CreateVideo -> SaveVideo); older GraphParams has no fps field
            **({"fps": c.get("fps")} if "fps" in getattr(bline_graphs.GraphParams, "__dataclass_fields__", {})
               else {}))

    def render(self, row: dict, steps: int) -> Render:
        import bline_graphs

        if self.srv is None or self.srv.poll() is not None:
            raise BlineUIError(f"BlineUI server is not running (exit {self.srv and self.srv.returncode}):\n{self.tail(25)}")
        fam = self.family()
        mod = bline_graphs.module(fam)
        self.renders += 1
        p = self.graph_params(prompt = row["prompt"], seed = int(row["seed"]), steps = int(steps),
                              prefix = f"r{self.renders:04d}_{row.get('id', 'x')}", nonce = self.renders)
        o, c = self.opts, self.cell
        self._params = p
        graph = bline_graphs.build(fam, p)
        self.graph = graph
        start = self.log_size()
        resp = self.api("/prompt", {"prompt": graph, "client_id": self.client_id})
        pid = resp["prompt_id"]
        deadline = time.time() + float(o.get("render_timeout_s", 3600))
        while True:
            h = self.api(f"/history/{pid}")
            if pid in h and (h[pid].get("status") or {}).get("completed") is not None:
                break
            if self.srv.poll() is not None:
                raise BlineUIError(f"BlineUI server died mid-render (exit {self.srv.returncode}):\n{self.tail(30)}")
            if time.time() > deadline:
                self.api("/interrupt", {})
                raise BlineUIError(f"BlineUI render exceeded {o.get('render_timeout_s', 3600)} s")
            time.sleep(0.05)
        st = h[pid]["status"]
        if st.get("status_str") != "success":
            msgs = [m for m in st.get("messages", []) if m[0] == "execution_error"]
            detail = msgs[-1][1] if msgs else st.get("messages")
            if isinstance(detail, dict):
                detail = f"{detail.get('node_type')}: {detail.get('exception_type')}: {detail.get('exception_message')}"
            raise BlineUIError(f"BlineUI job failed: {str(detail)[:1500]}")
        chunk = self.log_text(start)
        rates = [m for m in IT_RE.findall(chunk) if m[0] == m[1]]
        step_s = None
        if rates:
            v, unit = float(rates[-1][2]), rates[-1][3]
            step_s = round(1.0 / v if unit == "it/s" else v, 4)
        msgs = {m[0]: m[1] for m in st.get("messages", [])}
        exec_s = None
        if "execution_success" in msgs and "execution_start" in msgs:
            exec_s = round((msgs["execution_success"]["timestamp"] - msgs["execution_start"]["timestamp"]) / 1000, 3)
        cached = (msgs.get("execution_cached") or {}).get("nodes") or []
        saved = h[pid]["outputs"].get(bline_graphs.SAVE) or {}
        files = [(i.get("subfolder") or "", i["filename"]) for key in ("images", "videos", "gifs")
                 for i in saved.get(key, []) or [] if isinstance(i, dict) and i.get("filename")]
        if not files:
            raise BlineUIError(f"BlineUI finished but the save node wrote nothing (cached nodes: {cached})")
        from PIL import Image

        images = []
        for sub, name in files:
            path = self.work / "output" / sub / name
            if path.suffix.lower() in (".mp4", ".webm", ".mkv", ".mov"):
                # SaveVideo (the templates' export): decode the file BlineUI wrote, i.e. what a user downloads
                from studio_client import decode_mp4

                arr, _ = decode_mp4(path.read_bytes())
                images.extend(Image.fromarray(fr) for fr in arr)
            else:
                with Image.open(path) as im:
                    images.append(im.convert("RGB"))
            if not o.get("keep_raw"):
                path.unlink(missing_ok = True)
        skipped = EASY_RE.findall(chunk)
        facts = self._apply_events(self._prompt_events(pid))
        if not self.ws_events and self.ws is None:
            facts["executed_nodes"] = None  # no feed: unknown, never assumed
        facts["cached_nodes"] = sorted(set(facts["cached_nodes"]) | set(cached))
        samplers = [k for k, n in graph.items() if n.get("class_type") in SAMPLER_CLASSES]
        facts["sampler_cached"] = facts["sampler_cached"] or any(n in cached for n in samplers)
        facts["exec_s"] = exec_s
        self.request_facts["bline"] = facts
        extra = {"exec_s": exec_s, "sampler_cached": facts["sampler_cached"], "cached_nodes": cached}
        if skipped:
            extra["easycache_skipped"] = f"{skipped[-1][0]}/{skipped[-1][1]}"
        if getattr(mod, "VIDEO", False):
            return Render(frames = images, fps = int(c.get("fps") or 24), step_s = step_s, extra = extra)
        return Render(image = images[0], step_s = step_s, extra = extra)

    @property
    def client_id(self) -> str:
        if not hasattr(self, "_cid"):
            self._cid = str(uuid.uuid4())
        return self._cid

    def status(self) -> dict:
        text = self.log_text()
        return {"engaged": {k: v in text for k, v in ENGAGED.items()}, "facts": self._facts(text),
                "server_alive": bool(self.srv and self.srv.poll() is None)}

    def close(self) -> None:
        ws, self.ws = self.ws, None
        if ws is not None:
            ws.close()
        if self.srv is not None and self.srv.poll() is None:
            import common as C

            C.kill_tree(self.srv, timeout = 30)  # psutil tree walk; killpg / taskkill without psutil (Windows safe)
        if self.logf:
            self.logf.close()
            self.logf = None

    def trees(self) -> dict:
        return {"blineui": self.info.get("bline_dir")} if self.info.get("bline_dir") else {}

    def sync(self) -> None:  # render() returns only after BlineUI reports the prompt finished
        pass

    def reset_peak(self) -> None:
        pass
