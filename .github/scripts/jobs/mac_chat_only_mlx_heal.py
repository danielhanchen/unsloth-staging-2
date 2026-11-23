"""Chat only (--no-torch) Studio on Apple Silicon with a broken MLX stack: does it heal?

Per arm (head = the checkout this runs from, base = --base-sha tarball): `install.sh --local
--no-torch` into a scratch UNSLOTH_STUDIO_HOME, uninstall mlx-vlm from the Studio venv, launch
Studio, force hardware detection via /api/health and wait for the settled verdict, then record
whether mlx-vlm came back, torch stayed out, and /api/train/start refuses.

    python jobs/mac_chat_only_mlx_heal.py --base-sha <merge base>

Head checks gate; base checks only prove the defect reproduces there (else the run is VOID).
Exit 3 off Apple Silicon.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

PASSWORD = "heal-check-pass-1"
NEW_PASSWORD = "heal-check-pass-2"
TRAIN_BODY = {
    "model_name": "unsloth/gemma-3-270m-it",
    "training_type": "LoRA/QLoRA",
    "format_type": "alpaca",
    "hf_dataset": "org/dataset",
}


def _http(
    method,
    url,
    body = None,
    token = None,
    timeout = 60,
):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data = data, method = method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout = timeout) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw or b"null")
        except ValueError:
            return e.code, raw.decode(errors = "replace")


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _probe(py, code):
    r = subprocess.run([str(py), "-c", code], capture_output = True, text = True, timeout = 300)
    return r.returncode == 0, (r.stdout + r.stderr).strip()[-400:]


def _base_tree(sha, scratch):
    dst = scratch / "src_base"
    if dst.exists():
        return dst
    tgz = scratch / "base.tar.gz"
    urllib.request.urlretrieve(f"https://codeload.github.com/unslothai/unsloth/tar.gz/{sha}", tgz)
    with tarfile.open(tgz) as t:
        top = t.getmembers()[0].name.split("/")[0]
        t.extractall(scratch, filter = "data")
    (scratch / top).rename(dst)
    return dst


def run_arm(arm, src, scratch, log):
    home = scratch / f"home_{arm}"
    shutil.rmtree(home, ignore_errors = True)
    env = {
        **os.environ,
        "UNSLOTH_STUDIO_HOME": str(home),
        "UNSLOTH_SKIP_AUTOSTART": "1",
        "UNSLOTH_DISABLE_UPDATE_CHECK": "1",
        "UNSLOTH_STUDIO_DISABLE_PUBLIC_CHECK": "1",
    }
    env.pop("UNSLOTH_DISABLE_MLX_AUTOREPAIR", None)
    out = {"arm": arm, "src": str(src)}
    t = time.time()
    with open(scratch / f"install_{arm}.log", "w") as f:
        rc = subprocess.run(
            ["bash", "./install.sh", "--local", "--no-torch"],
            cwd = src,
            env = env,
            stdout = f,
            stderr = subprocess.STDOUT,
            timeout = 3600,
        ).returncode
    out["install_rc"], out["install_s"] = rc, round(time.time() - t)
    log(f"[{arm}] install rc={rc} in {out['install_s']}s")
    py = home / "unsloth_studio" / "bin" / "python"
    if rc != 0 or not py.exists():
        out["error"] = "install failed"
        return out
    out["torch_before"] = _probe(py, "import torch")[0]
    out["mlx_vlm_before"] = _probe(py, "import mlx_vlm")[0]
    uv = [shutil.which("uv")] if shutil.which("uv") else None
    if uv is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check = True)
        uv = [sys.executable, "-m", "uv"]
    r = subprocess.run(
        [*uv, "pip", "uninstall", "--python", str(py), "mlx-vlm"], capture_output = True, text = True
    )
    out["uninstall"] = (r.stdout + r.stderr).strip()[-300:]
    out["mlx_vlm_broken"] = not _probe(py, "import mlx_vlm")[0]
    log(
        f"[{arm}] mlx-vlm present before={out['mlx_vlm_before']} broken now={out['mlx_vlm_broken']}"
    )

    port = _free_port()
    slog = scratch / f"studio_{arm}.log"
    proc = subprocess.Popen(
        ["bash", "-c", f'"{home}/bin/unsloth" studio -H 127.0.0.1 -p {port} >> "{slog}" 2>&1'],
        env = {**env, "UNSLOTH_STUDIO_PASSWORD": PASSWORD},
        stdin = subprocess.DEVNULL,
        start_new_session = True,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.time() + 600
        while time.time() < deadline:
            try:
                if _http("GET", f"{base}/api/health", timeout = 5)[0] == 200:
                    break
            except OSError:
                pass
            time.sleep(3)
        else:
            out["error"] = "Studio never answered /api/health"
            return out
        token = None
        for pw in (PASSWORD, NEW_PASSWORD):
            st, b = _http("POST", f"{base}/api/auth/login", {"username": "unsloth", "password": pw})
            if st == 200:
                token = b["access_token"]
                if b.get("must_change_password"):
                    st, b = _http(
                        "POST",
                        f"{base}/api/auth/change-password",
                        {"current_password": pw, "new_password": NEW_PASSWORD},
                        token,
                    )
                    token = b["access_token"] if st == 200 else token
                break
        if not token:
            out["error"] = f"login failed: {st} {b}"
            return out
        # The self-heal is a pip install that can take minutes; wait for the verdict to settle.
        t = time.time()
        health = None
        while time.time() - t < 1500:
            st, health = _http(
                "GET", f"{base}/api/health?detect_hardware=true", token = token, timeout = 120
            )
            if st == 200 and not health.get("hardware_detecting") and "chat_only_reason" in health:
                break
            time.sleep(10)
        out["settle_s"] = round(time.time() - t)
        out["health"] = (
            {
                k: health.get(k)
                for k in (
                    "chat_only",
                    "chat_only_reason",
                    "mlx_inference",
                    "chat_only_detail",
                    "chat_only_hint",
                    "device_type",
                    "hardware_detecting",
                )
            }
            if isinstance(health, dict)
            else health
        )
        log(f"[{arm}] verdict after {out['settle_s']}s: {out['health']}")
        st, b = _http("POST", f"{base}/api/train/start", TRAIN_BODY, token)
        out["train_start"] = {
            "status": st,
            "detail": b.get("detail") if isinstance(b, dict) else str(b)[:300],
        }
        log(f"[{arm}] /api/train/start -> {out['train_start']}")
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout = 30)
        except Exception:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass
    out["mlx_vlm_after"] = _probe(py, "import mlx_vlm")[0]
    out["torch_after"] = _probe(py, "import torch")[0]
    text = slog.read_text(errors = "replace") if slog.exists() else ""
    out["heal_log"] = [ln[-300:] for ln in text.splitlines() if "self-heal" in ln.lower()][-8:]
    log(f"[{arm}] mlx-vlm after={out['mlx_vlm_after']} torch after={out['torch_after']}")
    return out


def main():
    p = C.base_parser("mac_chat_only_mlx_heal", "n/a", "n/a", default_steps = 0)
    p.add_argument("--base-sha", default = None)
    p.add_argument("--arms", default = "head,base")
    a = C.resolve_args(p)
    C.reject_cfg(a)
    with C.JobRecorder(a, backend_hint = "studio") as rec:
        if not (sys.platform == "darwin" and platform.machine() == "arm64"):
            rec.skip("needs Apple Silicon")
        scratch = Path(a.out).resolve().parent / "mac_chat_only_mlx_heal_scratch"
        scratch.mkdir(parents = True, exist_ok = True)
        arms = {}
        for arm in a.arms.split(","):
            src = Path.cwd() if arm == "head" else _base_tree(a.base_sha, scratch)
            arms[arm] = run_arm(arm, src, scratch, lambda m: print(m, flush = True))
            rec.summary(arms = arms)
        h = arms.get("head")
        hh = (h or {}).get("health") or {}
        if h is not None:
            rec.check("head_install", h.get("install_rc") == 0, h.get("install_rc"))
            rec.check(
                "head_mlx_vlm_was_broken", h.get("mlx_vlm_broken") is True, h.get("uninstall")
            )
            rec.check(
                "head_heal_restored_mlx_vlm", h.get("mlx_vlm_after") is True, h.get("heal_log")
            )
            rec.check(
                "head_verdict_no_torch_serving_mlx",
                hh.get("chat_only") is True
                and hh.get("chat_only_reason") == "no_torch"
                and hh.get("mlx_inference") is True,
                hh,
            )
            rec.check("head_torch_stays_out", h.get("torch_after") is False, h.get("torch_after"))
            ts = h.get("train_start") or {}
            rec.check(
                "head_train_start_refused",
                ts.get("status") == 400 and "Chat only" in str(ts.get("detail")),
                ts,
            )
        if "base" in arms:
            b = arms["base"]
            bh = b.get("health") or {}
            rec.check(
                "base_reproduces_defect",
                b.get("mlx_vlm_broken") is True
                and (b.get("mlx_vlm_after") is False or bh.get("mlx_inference") is not True),
                {"mlx_vlm_after": b.get("mlx_vlm_after"), "health": bh},
            )


if __name__ == "__main__":
    main()
