#!/usr/bin/env python3
"""amd_ci probe: run diffusion_bench cells (and optionally the edge suite) against ONE state's Studio tree and
write what happened as JSON. Observes only; amd/criteria_*.py judge.

Called by amd_ci/lib/differential.py once per state:

  python diffusion_bench/amd/diffusion_probe.py --state head --checkout <states/head> --out obs_head.json \\
      [--spec diffusion_bench/specs/amd_strix_small.json] [--only 'zimg_*' ...] [--budget-min 100]

What one call does, in order:
  1. drop every HF token from the environment (the runner never gets one), keep HF / torch caches under the job's
     work dir, set the hub timeouts the mirror needs;
  2. read the host: vendor, torch / HIP / CUDA versions, device name and arch (gfx1151 on the AMD CI; on NVIDIA the
     same fields, so a dry run shows the detection working);
  3. --install auto: on an AMD host whose interpreter has no ROCm torch, install it from the gfx1151 index
     (repo.amd.com/rocm/whl/gfx1151, what install.sh --local picks on Linux) plus the Studio requirements on
     Windows, NO torchao; `pip install --no-deps lpips` so ROCm torch is never replaced. Never on NVIDIA;
  4. fetch the spec's public models once per job (snapshot_download, token=False, 6 retries) into the work dir
     and point the spec aliases at them through DBENCH_MODEL_<ALIAS>, so download time never lands in load_s;
  5. run each selected cell through matrix.py (one process per cell, per-cell timeout, stop starting new cells
     past --budget-min) with DIFFUSION_BENCH_STUDIO_SRC = this state's checkout, then score.py;
  6. write the per-cell facts (verdict, timings, peaks, engaged levers, LPIPS vs the cell's reference from the
     SAME state, sanity flags, pixel hashes and 32x32 thumbnails for cross-state drift) to --out.

--dry replaces every cell's backend with the fake one and skips installs and downloads: it proves the plumbing
(detection, spec handling, matrix, scoring, JSON) end to end on any host, NVIDIA included. The criteria refuse
a dry observation (INCONCLUSIVE), so a dry run can never read as a result.

Text I/O names utf-8 everywhere: Path.read_text() is cp1252 on the Windows runners.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
TOOLKIT = HERE.parent  # diffusion_bench/
sys.path.insert(0, str(TOOLKIT))
sys.path.insert(0, str(HERE))

GFX1151_INDEX = "https://repo.amd.com/rocm/whl/gfx1151/"
TORCH_PINS = ["torch>=2.11.0,<2.12.0", "torchvision>=0.26.0,<0.27.0"]
# The diffusers pin the Windows gfx1151 runs used (Studio's shipped pin at the time), when the checkout has no
# requirements/diffusers-pin.txt of its own.
DIFFUSERS_FALLBACK = "diffusers @ https://github.com/huggingface/diffusers/archive/80c7ed262aeffbeb43ef13ae04baeb9b84515a69.zip"
WINDOWS_PACKAGES = ["transformers==5.5.0", "accelerate", "safetensors", "gguf>=0.10", "huggingface_hub", "pillow",
                    "numpy", "scipy", "tqdm", "psutil", "imageio", "scikit-image"]
DEFAULT_ONLY = ["tiny_*", "sdxlt_*", "zimg_*", "wan_*"]
TOKEN_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_HUB_TOKEN")
PROBE_VERSION = 1


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def say(msg: str) -> None:
    print(f"[diffusion_probe {time.strftime('%H:%M:%S')}] {msg}", flush = True)


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents = True, exist_ok = True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent = 1, default = str), encoding = "utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(Path(path).read_text(encoding = "utf-8"))
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------------------------- environment
def scrub_environment(work: Path) -> list:
    """No token reaches this process or its children; caches stay inside the job's work dir."""
    dropped = [k for k in TOKEN_VARS if os.environ.pop(k, None) is not None]
    os.environ.setdefault("HF_HOME", str(work / "hf_home"))
    os.environ.setdefault("HF_HUB_CACHE", str(work / "hf_home" / "hub"))
    os.environ.setdefault("TORCH_HOME", str(work / "torch_home"))
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("WORKSPACE", str(work))
    return dropped


HOST_SNIPPET = r"""
import json, platform, sys
out = {"python": sys.version.split()[0], "executable": sys.executable, "system": platform.system(),
       "platform": platform.platform(), "machine": platform.machine(), "node": platform.node()}
try:
    import torch
    out["torch"] = torch.__version__
    out["hip"] = getattr(torch.version, "hip", None)
    out["cuda"] = getattr(torch.version, "cuda", None)
    out["available"] = bool(torch.cuda.is_available())
    if out["available"]:
        p = torch.cuda.get_device_properties(0)
        out["device_count"] = torch.cuda.device_count()
        out["device_name"] = torch.cuda.get_device_name(0)
        out["arch"] = getattr(p, "gcnArchName", None)
        out["capability"] = list(torch.cuda.get_device_capability(0))
        out["total_gib"] = round(p.total_memory / 2**30, 2)
        out["is_integrated"] = getattr(p, "is_integrated", None)
except Exception as e:
    out["torch_error"] = f"{type(e).__name__}: {e}"[:500]
for mod in ("diffusers", "transformers", "accelerate", "torchao", "lpips", "skimage", "huggingface_hub"):
    try:
        m = __import__(mod)
        out.setdefault("packages", {})[mod] = getattr(m, "__version__", "?")
    except Exception:
        out.setdefault("packages", {})[mod] = None
json.dump(out, open(sys.argv[1], "w", encoding="utf-8"))
"""


def read_host(work: Path) -> dict:
    """Host facts from a CHILD interpreter: importing torch here would pin this process's view before an install."""
    import common as C

    # The snippet gets its own directory: a script run from a shared dir puts that dir first on sys.path, and a
    # stray module there once shadowed a real package and broke diffusers imports.
    snip_dir = work / "_host_probe"
    snip_dir.mkdir(parents = True, exist_ok = True)
    path = snip_dir / "host_probe.json"
    snippet = snip_dir / "host_probe.py"
    snippet.write_text(HOST_SNIPPET, encoding = "utf-8")
    proc = subprocess.run([sys.executable, str(snippet), str(path)], capture_output = True, text = True,
                          encoding = "utf-8", errors = "replace", timeout = 600)
    host = read_json(path) or {"error": (proc.stderr or proc.stdout)[-1500:]}
    host["smi_vendor"] = C.gpu_vendor()
    if host.get("hip"):
        host["vendor"] = "amd"
    elif host.get("cuda") and host.get("available"):
        host["vendor"] = "nvidia"
    else:
        host["vendor"] = host["smi_vendor"] if host["smi_vendor"] != "none" else "cpu"
    # gfx1151 reports capability (11, 5), which an NVIDIA ladder would rank as Blackwell; never gate on it.
    host["is_gfx1151"] = "gfx1151" in str(host.get("arch") or "")
    return host


def pip(args: list, log: list, timeout: int = 3600) -> int:
    cmd = [sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check", *args]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output = True, text = True, encoding = "utf-8", errors = "replace",
                          timeout = timeout)
    if proc.returncode and "No module named pip" in (proc.stderr + proc.stdout):
        import shutil

        if shutil.which("uv"):  # uv-built venvs (Studio's) may carry no pip
            cmd = ["uv", "pip", "install", "--python", sys.executable, *args]
            proc = subprocess.run(cmd, capture_output = True, text = True, encoding = "utf-8", errors = "replace",
                                  timeout = timeout)
    entry = {"cmd": " ".join(shlex.quote(str(c)) for c in cmd[1:]), "rc": proc.returncode,
             "s": round(time.time() - t0, 1)}
    if proc.returncode:
        entry["tail"] = (proc.stderr or proc.stdout)[-1500:]
    log.append(entry)
    say(f"pip {entry['cmd'][:160]} -> rc {proc.returncode} in {entry['s']}s")
    return proc.returncode


def ensure_runtime(mode: str, host: dict, checkout: Path, work: Path) -> dict:
    """Make this interpreter able to run Studio cells on gfx1151. Idempotent across states: the first state
    installs, later states find ROCm torch present and only top up what is missing."""
    out: dict = {"mode": mode, "steps": []}
    if mode == "never":
        out["action"] = "none (--install never)"
        return out
    amd_host = host.get("vendor") == "amd" or host.get("smi_vendor") == "amd"
    if not amd_host:
        out["action"] = f"none: not an AMD host (vendor {host.get('vendor')}); ROCm wheels are only installed on AMD"
        return out
    windows = platform.system() == "Windows"
    need_torch = not host.get("hip")
    if need_torch:
        pip(["--index-url", GFX1151_INDEX, *TORCH_PINS], out["steps"])
    if windows and (need_torch or not (host.get("packages") or {}).get("diffusers")):
        pin = checkout / "studio" / "backend" / "requirements" / "diffusers-pin.txt"
        pip(["-r", str(pin)] if pin.is_file() else [DIFFUSERS_FALLBACK], out["steps"])
        pip(WINDOWS_PACKAGES, out["steps"])
        studio_txt = checkout / "studio" / "backend" / "requirements" / "studio.txt"
        if studio_txt.is_file():
            # Past Windows runs installed these too; a failure here is recorded, not fatal (the cells say more).
            pip(["-r", str(studio_txt)], out["steps"])
    pk = host.get("packages") or {}
    if not pk.get("lpips"):
        pip(["--no-deps", "lpips"], out["steps"])  # --no-deps: its torch requirement would replace ROCm torch
    if not pk.get("skimage"):
        pip(["scikit-image"], out["steps"])
    out["action"] = "installed" if out["steps"] else "nothing missing"
    return out


# ---------------------------------------------------------------------------------------------- spec + models
def selected_cells(spec: dict, only: list) -> list:
    return [c for c in spec.get("cells", []) if not c.get("skip")
            and (not only or any(fnmatch.fnmatch(c["tag"], pat) for pat in only))]


def derive_spec(raw: dict, only: list, dry: bool) -> dict:
    spec = json.loads(json.dumps(raw))
    cells = selected_cells(spec, only)
    if dry:
        # Same tags, kinds, references and prompt ids; no model. Non-reference cells get a little noise so the
        # scorer has something non-zero to report, and sizes stay as specified (the fake renders 1/8 scale).
        for c in cells:
            c["backend"] = "fake"
            c.pop("venv", None)
            opts = {"step_delay_s": 0.002}
            if c.get("ref") and c.get("ref") != c["tag"] and not c["tag"].endswith("_repeat"):
                opts["noise"] = 0.02
            c["options"] = opts
            c["steps"] = min(int(c.get("steps") or spec["defaults"].get("steps") or 4), 4)
            if c.get("short_steps"):
                c["short_steps"] = 1
            if c.get("frames"):
                c["frames"] = min(int(c["frames"]), 5)
        spec["defaults"] = {k: v for k, v in spec.get("defaults", {}).items() if k not in ("options", "venv")}
        spec["defaults"].update({"n": 2, "short_n": 1, "warmup": 1})
        for c in cells:
            c.pop("n", None)
    spec["cells"] = cells
    return spec


def fetch_models(spec: dict, cells: list, models_dir: Path, dry: bool, token_blob: Optional[str]) -> dict:
    """Public models once per job; returns {alias: {source, path, seconds, error}}. The alias's env override is
    set here so matrix.py resolves "@alias" to the fetched directory."""
    import common as C

    table = spec.get("models") or {}
    aliases = sorted({c["model"][1:] for c in cells if isinstance(c.get("model"), str) and c["model"].startswith("@")})
    out: dict = {}
    for alias in aliases:
        entry = table.get(alias) or {}
        key = C.model_env_key(alias)
        rec: dict = {"repo": entry.get("repo"), "revision": entry.get("revision")}
        out[alias] = rec
        if os.environ.get(key):
            rec.update(source = "env", path = os.environ[key])
            continue
        local = entry.get("local")
        if local and Path(C.expand_env(local)).exists():
            rec.update(source = "local", path = C.expand_env(local))
            continue
        if dry:
            rec.update(source = "dry", path = None)
            continue
        target = models_dir / alias
        done = target / ".dbench_complete"
        if done.exists():
            rec.update(source = "downloaded", path = str(target), seconds = 0, reused = True)
            os.environ[key] = str(target)
            continue
        token = None
        if entry.get("gated"):
            # Gated weights need an explicit, per-run authorisation and an ENCRYPTED short-lived token (see
            # amd/token_vault.py). Without both, refuse rather than fall back to anything in the environment.
            if not token_blob:
                rec.update(source = "refused", error = "gated model and no --token-blob; public models only")
                continue
            from token_vault import decrypt_token

            token = decrypt_token(token_blob)
            rec["token"] = "encrypted blob (value never logged)"
        from huggingface_hub import snapshot_download

        t0 = time.time()
        last = None
        for attempt in range(6):
            try:
                snapshot_download(repo_id = entry["repo"], revision = entry.get("revision"), local_dir = str(target),
                                  ignore_patterns = (entry.get("download") or {}).get("ignore"),
                                  allow_patterns = (entry.get("download") or {}).get("allow"),
                                  token = token if token else False)
                last = None
                break
            except Exception as exc:  # noqa: BLE001 - mirror 500s happen; retry, then record
                last = f"{type(exc).__name__}: {str(exc)[:400]}"
                say(f"download {alias} attempt {attempt + 1} failed: {last}")
                if "gated" in last.lower() or "401" in last or "403" in last:
                    break
                time.sleep(20)
        token = None
        rec["seconds"] = round(time.time() - t0, 1)
        if last:
            rec.update(source = "failed", error = last)
            continue
        size = sum(p.stat().st_size for p in target.rglob("*") if p.is_file() and ".cache" not in p.parts)
        rec.update(source = "downloaded", path = str(target), gib = round(size / 2**30, 2))
        done.write_text(now(), encoding = "utf-8")
        os.environ[key] = str(target)
    return out


# ---------------------------------------------------------------------------------------------- per-cell facts
STATUS_KEEP = ("engine", "device", "dtype", "offload_policy", "memory_mode", "speed_mode", "speed_optims",
               "transformer_quant", "text_encoder_quant", "attention_backend", "transformer_cache",
               "transformer_cache_stats", "fallback_reason", "vae_tiling", "cfg_batch", "zimage_cfg_split",
               "resolved", "small_host")
# cell.log lines that say which optimization engaged, declined or fell back (kept verbatim, deduplicated).
LOG_NEEDLES = ("cfg_batch", "rocm_fused", "fused RoPE", "rocm_compile", "ROCm compile", "regional", "compiled",
               "cuda_graph", "CUDA graph", "attention", "speed profile", "speed tier", "deferred", "step skip",
               "static_skip", "bf16 small", "small DiT", "kept bf16", "vae_bf16", "VAE", "fallback", "Graph break",
               "Recompiling", "zimage_cfg_split", "unified", "stream_encode", "vae_decode")
LOG_KEEP = 80


def thumb(cell_dir: Path, render: dict) -> dict:
    """Pixel hash of the decoded media and a 32x32 RGB thumbnail (hex): enough for a criteria module with no numpy
    to tell identical / drifted / broken apart across states."""
    try:
        import numpy as np
        from PIL import Image

        path = cell_dir / (render.get("file") or "")
        if not path.is_file():
            return {}
        if path.suffix == ".npz":
            frames = np.load(path)["frames"]
            arr = frames[len(frames) // 2]
            digest = hashlib.sha256(frames.tobytes()).hexdigest()[:16]
        else:
            arr = np.asarray(Image.open(path).convert("RGB"))
            digest = hashlib.sha256(arr.tobytes()).hexdigest()[:16]
        small = np.asarray(Image.fromarray(arr).resize((32, 32), Image.BOX)).astype("uint8")
        return {"sha": digest, "thumb": small.tobytes().hex()}
    except Exception as exc:  # noqa: BLE001
        return {"thumb_error": f"{type(exc).__name__}: {exc}"[:200]}


def cell_facts(run_dir: Path, tag: str, score_row: Optional[dict]) -> dict:
    import common as C

    cell_dir = run_dir / tag
    rec = C.read_record(cell_dir)
    if not rec:
        return {"verdict": "MISSING", "error": "no record.json (the cell process died before writing one)"}
    status = {**(rec.get("status") or {}), **(rec.get("status_after") or {})}
    facts = {
        "verdict": rec.get("verdict"), "error": (rec.get("error") or "")[:600] or None,
        "backend": rec.get("backend"), "kind": (rec.get("cell") or {}).get("kind"),
        "tier": (rec.get("cell") or {}).get("tier"),
        "ref": (rec.get("cell") or {}).get("ref"), "model_source": (rec.get("cell") or {}).get("model_source"),
        "load_s": rec.get("load_s"), "cold_s": rec.get("cold_s"), "new_s": rec.get("wall_s_median"),
        "steady_s": rec.get("steady_s_median"), "step_s": rec.get("step_s_derived"),
        "walls": [r.get("wall_s") for r in rec.get("renders", [])],
        "peak_alloc_gib": rec.get("peak_alloc_gib"), "peak_smi_gib": rec.get("peak_smi_gib"),
        "peak_smi_delta_gib": rec.get("peak_smi_delta_gib"),
        "peak_rss_gib": (rec.get("host_end") or {}).get("peak_rss_gib"),
        "gate": (rec.get("gate") or {}).get("gate"),
        "status": {k: status.get(k) for k in STATUS_KEEP if k in status},
        "renders": [{"id": r.get("id"), "seed": r.get("seed"), "wall_s": r.get("wall_s"),
                     "peak_alloc_gib": r.get("peak_alloc_gib"),
                     "engaged": (r.get("extra") or {}).get("engaged"), **thumb(cell_dir, r)}
                    for r in rec.get("renders", [])],
        "long": rec.get("long"), "short": rec.get("short"), "warmup_s": rec.get("warmup_s"),
        **dict(zip(("log_lines", "log_truncated"), log_lines(cell_dir / "cell.log"))),
    }
    if score_row:
        for k in ("lpips", "lpips_max", "ssim", "psnr", "identical", "flicker_ratio", "score_error"):
            if score_row.get(k) is not None:
                facts[k] = score_row[k]
        facts["flags"] = score_row.get("flags") or {}
    return facts


def log_lines(path: Path) -> tuple:
    """The cell.log lines naming an optimization's engagement or decline, first occurrence of each, in order, and
    whether any matching line was cut or dropped (then the absence of a needle proves nothing)."""
    try:
        text = path.read_text(encoding = "utf-8", errors = "replace")
    except OSError:
        return [], False
    seen, out, truncated = set(), [], False
    for line in text.splitlines():
        if not any(n in line for n in LOG_NEEDLES):
            continue
        if len(out) >= LOG_KEEP:
            truncated = True
            break
        truncated = truncated or len(line) > 400
        key = line[-240:]
        for ch in "0123456789":
            key = key.replace(ch, "")
        if key in seen:
            continue
        seen.add(key)
        out.append(line[-400:])
    return out, truncated


def run_edge(cmd: str, work: Path, env: dict, timeout: int, models: dict) -> dict:
    """Run an edge-suite command line and record rc, time, the log tail and the per-check statuses. Placeholders:
    {out} -> this state's edge output dir, {model:ALIAS} -> the fetched / local path of a spec model alias, e.g.

      python diffusion_bench/edge/run_edge.py --surface inproc --tier fast --out {out}
             --sdxl {model:sdxl_turbo} --zimage {model:zimage_turbo} --wan {model:wan22_5b}

    The suite reads results into <out>/results.json ({meta, counts, results: [{surface, check, status}]}), which
    is folded in as {"surface/check": status}. TODO(verify) on the runner: the edge suite's HTTP surface launches
    a Studio server, which has not been run on gfx1151; --surface inproc is the conservative choice there."""
    import re

    edge_out = work / "edge"
    edge_out.mkdir(parents = True, exist_ok = True)
    log = edge_out / "edge.log"

    def model_path(m):
        rec = models.get(m.group(1)) or {}
        return str(rec.get("path") or m.group(0))

    cmd = re.sub(r"\{model:([A-Za-z0-9_.-]+)\}", model_path, cmd.replace("{out}", str(edge_out)))
    argv = shlex.split(cmd, posix = os.name != "nt")
    if argv and argv[0] in ("python", "python3", "python.exe"):
        argv[0] = sys.executable
    run_env = {**env, "DBENCH_EDGE_OUT": str(edge_out)}
    run_env.setdefault("DIFFUSION_BENCH_STUDIO_PYTHON", sys.executable)  # never build a second Studio venv
    t0 = time.time()
    with open(log, "w", encoding = "utf-8") as fh:
        try:
            rc = subprocess.run(argv, stdout = fh, stderr = subprocess.STDOUT, cwd = str(TOOLKIT.parent),
                                env = run_env, timeout = timeout).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    res = read_json(edge_out / "results.json") or {}
    checks = {f"{r.get('surface')}/{r.get('check')}": r.get("status") for r in res.get("results") or []}
    return {"cmd": cmd, "rc": rc, "s": round(time.time() - t0, 1), "counts": res.get("counts"), "checks": checks,
            "log_tail": log.read_text(encoding = "utf-8", errors = "replace")[-3000:]}


# ---------------------------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path, help = "this state's unsloth checkout")
    ap.add_argument("--out", required = True, type = Path, help = "observation JSON (never stdout)")
    ap.add_argument("--spec", default = str(TOOLKIT / "specs" / "amd_strix_small.json"))
    ap.add_argument("--only", action = "append", default = [],
                    help = f"fnmatch on cell tags, repeatable (default {' '.join(DEFAULT_ONLY)})")
    ap.add_argument("--work", type = Path, default = None,
                    help = "work root (default $AMD_CI_WORK/dbench, else next to --out)")
    ap.add_argument("--models-dir", type = Path, default = None, help = "default <work>/models, shared by states")
    ap.add_argument("--install", choices = ["auto", "never"], default = "auto")
    ap.add_argument("--dry", action = "store_true", help = "fake backend, no install, no download: plumbing only")
    ap.add_argument("--skip-states", default = "merge",
                    help = "comma list of states not to measure (default merge: base and head fill the budget)")
    ap.add_argument("--budget-min", type = float, default = 100.0,
                    help = "per state: no new cell starts after this many minutes")
    ap.add_argument("--cell-timeout", type = float, default = 3600.0)
    ap.add_argument("--no-gate", action = "store_true", help = "skip the matrix GPU-quiet gate")
    ap.add_argument("--no-lpips", action = "store_true", help = "SSIM / PSNR only (default in --dry)")
    ap.add_argument("--set", action = "append", default = [], metavar = "KEY=JSON",
                    help = "passed to matrix.py --set (defaults for every cell)")
    ap.add_argument("--edge", default = None,
                    help = "edge-suite command line to run after the cells; {out} and {model:ALIAS} are filled in")
    ap.add_argument("--edge-timeout", type = int, default = 3600)
    ap.add_argument("--defect", default = None,
                    help = "differential defect spec, recorded for criteria_differential.py, e.g. broken:zimg_fp8")
    ap.add_argument("--token-blob", default = None,
                    help = "encrypted HF token blob for a GATED model (amd/token_vault.py); never plaintext")
    args = ap.parse_args()

    t_start = time.time()
    work_root = args.work or (Path(os.environ["AMD_CI_WORK"]) / "dbench" if os.environ.get("AMD_CI_WORK")
                              else args.out.resolve().parent / "dbench_work")
    work = Path(work_root).resolve()
    work.mkdir(parents = True, exist_ok = True)
    obs: dict = {"probe": "diffusion_bench/amd/diffusion_probe.py", "probe_version": PROBE_VERSION,
                 "state": args.state, "checkout": str(args.checkout), "dry": bool(args.dry), "started": now(),
                 "defect": args.defect, "errors": []}
    if args.state in [s.strip() for s in args.skip_states.split(",") if s.strip()]:
        obs.update(skipped_state = True, reason = f"--skip-states {args.skip_states}")
        write_json(args.out, obs)
        say(f"state {args.state} skipped by request")
        return 0

    obs["tokens_dropped"] = scrub_environment(work)
    import common as C

    obs["checkout_rev"] = C.git_rev(args.checkout)
    try:
        obs["host"] = host = read_host(work)
        say(f"host: vendor {host.get('vendor')} torch {host.get('torch')} hip {host.get('hip')} "
            f"cuda {host.get('cuda')} device {host.get('device_name')} arch {host.get('arch')}")
        if not args.dry:
            obs["runtime"] = ensure_runtime(args.install, host, args.checkout, work)
            if obs["runtime"].get("steps"):
                obs["host"] = host = read_host(work)
        else:
            obs["runtime"] = {"mode": "dry", "action": "none"}

        # As given (relative to the job's checkout), relative to the branch root, or a bare name under specs/.
        cands = [Path(args.spec), TOOLKIT.parent / args.spec, TOOLKIT / "specs" / args.spec,
                 TOOLKIT / "specs" / f"{args.spec}.json"]
        spec_path = next((c for c in cands if c.is_file()), cands[0])
        raw = json.loads(spec_path.read_text(encoding = "utf-8"))
        only = args.only or DEFAULT_ONLY
        spec = derive_spec(raw, only, args.dry)
        obs["spec"] = {"name": raw.get("name"), "path": str(spec_path), "only": only,
                       "selected": [c["tag"] for c in spec["cells"]]}
        if not spec["cells"]:
            raise RuntimeError(f"no cells of {spec_path.name} match {only}")
        obs["models"] = fetch_models(raw, spec["cells"], args.models_dir or work / "models", args.dry,
                                     args.token_blob)
        # the plan's Studio repos (cells load them by repo id with local_files_only) into this job's HF cache: the
        # same fetch_files.prefetch the Linux fetch step and Mac / tester use; Windows has no fetch step at all
        from targets import fetch_files  # noqa: PLC0415

        obs["prefetch"] = fetch_files.prefetch(raw, download = not args.dry)
        state_dir = work / "states" / args.state
        spec_file = state_dir / "spec.json"
        write_json(spec_file, spec)
        run_dir = state_dir / "runs"
        env = {**os.environ}
        if args.dry and "DBENCH_FAKE_GPU" not in env:
            # --dry runs fake cells on a host that may have no GPU: a fake device, so records carry a VRAM metric
            env["DBENCH_FAKE_GPU"] = json.dumps({"name": "AMD Radeon 8060S (fake)", "uuid": "GPU-fake",
                                                 "used_mib": 1000, "own_mib": 800, "sm_mhz": 2000, "temp_c": 50,
                                                 "power_w": 100, "total_mib": 32768})
        env["DIFFUSION_BENCH_STUDIO_SRC"] = str(Path(args.checkout).resolve())
        env["UNSLOTH_STUDIO_HOME"] = env.get("UNSLOTH_STUDIO_HOME") or str(work / "studio_home")
        obs["cells"], obs["skipped_budget"] = {}, []
        for cell in spec["cells"]:
            tag = cell["tag"]
            elapsed_min = (time.time() - t_start) / 60
            if elapsed_min > args.budget_min:
                obs["skipped_budget"].append(tag)
                say(f"budget: {elapsed_min:.0f} min > {args.budget_min}; not starting {tag}")
                continue
            cmd = [sys.executable, "-u", str(TOOLKIT / "matrix.py"), str(spec_file), "--out", str(run_dir),
                   "--only", tag, "--timeout", str(args.cell_timeout), "--force"]
            if args.no_gate or args.dry:
                cmd.append("--no-gate")
            for item in args.set:
                cmd += ["--set", item]
            say(f"cell {tag} ({cell.get('backend') or spec['defaults'].get('backend')}) ...")
            t0 = time.time()
            proc = subprocess.run(cmd, capture_output = True, text = True, encoding = "utf-8", errors = "replace",
                                  env = env, timeout = args.cell_timeout + 900)
            (state_dir / "logs").mkdir(parents = True, exist_ok = True)
            (state_dir / "logs" / f"matrix_{tag}.log").write_text(proc.stdout + proc.stderr, encoding = "utf-8")
            say(f"cell {tag} done in {time.time() - t0:.0f}s (matrix rc {proc.returncode})")

        score_cmd = [sys.executable, str(TOOLKIT / "score.py"), str(run_dir)]
        if args.no_lpips or args.dry:
            score_cmd.append("--no-lpips")
        sp = subprocess.run(score_cmd, capture_output = True, text = True, encoding = "utf-8", errors = "replace",
                            env = env, timeout = 3600)
        obs["score_rc"] = sp.returncode
        if sp.returncode:
            obs["errors"].append(f"score.py rc {sp.returncode}: {(sp.stderr or sp.stdout)[-800:]}")
        scores = read_json(run_dir / "scores.json") or {}
        rows = {r["tag"]: r for r in scores.get("rows", [])}
        for cell in spec["cells"]:
            if cell["tag"] in obs["skipped_budget"]:
                continue
            obs["cells"][cell["tag"]] = cell_facts(run_dir, cell["tag"], rows.get(cell["tag"]))
        obs["scores_md"] = str(run_dir / "scores.md")
        if args.edge:
            obs["edge"] = run_edge(args.edge, state_dir, env, args.edge_timeout, obs.get("models") or {})
    except Exception as exc:  # noqa: BLE001 - a probe failure is an observation, written like any other
        import traceback

        obs["errors"].append(f"{type(exc).__name__}: {str(exc)[:1500]}")
        obs["traceback"] = traceback.format_exc()[-3000:]
        say(f"probe error: {obs['errors'][-1]}")
    obs["elapsed_s"] = round(time.time() - t_start, 1)
    obs["finished"] = now()
    n_ok = sum(1 for c in (obs.get("cells") or {}).values() if c.get("verdict") == "ok")
    obs["summary"] = {"cells": len(obs.get("cells") or {}), "ok": n_ok,
                      "skipped_budget": len(obs.get("skipped_budget") or [])}
    write_json(args.out, obs)
    say(f"state {args.state}: {n_ok}/{len(obs.get('cells') or {})} cells ok in {obs['elapsed_s']}s -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
