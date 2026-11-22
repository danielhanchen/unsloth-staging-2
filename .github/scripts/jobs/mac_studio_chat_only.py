"""Chat only (--no-torch) Studio on Apple Silicon: does an MLX model load and answer?

Installs the tree in the cwd (head) and, with --base-sha, that commit too, each with
`install.sh --local --no-torch` into its own UNSLOTH_STUDIO_HOME, launches it, reads the authed
/api/health, loads --model and sends one chat turn. Stdlib only. Exit 0 checks pass, 1 fail, 3 skip.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROMPT = "What is the capital of France? Answer with one word."


def _req(
    method,
    url,
    body = None,
    token = None,
    timeout = 60,
):
    data = json.dumps(body).encode() if body is not None else None
    h = {"Content-Type": "application/json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    r = urllib.request.Request(url, data = data, headers = h, method = method)
    try:
        with urllib.request.urlopen(r, timeout = timeout) as resp:
            raw = resp.read()
            try:
                return resp.status, json.loads(raw or b"null")
            except ValueError:
                return resp.status, raw.decode(errors = "replace")[:2000]
    except urllib.error.HTTPError as e:
        raw = e.read().decode(errors = "replace")
        try:
            return e.code, json.loads(raw)
        except ValueError:
            return e.code, raw[:2000]


def _port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _arm(name, tree, work, model, install_timeout):
    out = {"arm": name, "tree": str(tree)}
    home = work / f"home_{name}"
    env = {
        **os.environ,
        "UNSLOTH_STUDIO_HOME": str(home),
        "UNSLOTH_DISABLE_UPDATE_CHECK": "1",
        "UNSLOTH_STUDIO_DISABLE_PUBLIC_CHECK": "1",
        "UNSLOTH_SKIP_AUTOSTART": "1",
    }
    for k in ("UNSLOTH_IS_PRESENT", "UNSLOTH_COMPILE_DISABLE", "UNSLOTH_FORCE_GPU_PATH"):
        env.pop(k, None)
    t0 = time.time()
    inst_log = work / f"install_{name}.log"
    with open(inst_log, "w") as f:
        rc = subprocess.run(
            ["bash", "./install.sh", "--local", "--no-torch"],
            cwd = tree,
            env = env,
            stdout = f,
            stderr = subprocess.STDOUT,
            timeout = install_timeout,
        ).returncode
    out["install_rc"], out["install_s"] = rc, round(time.time() - t0, 1)
    tail = inst_log.read_text(errors = "replace").splitlines()
    out["install_mlx_lines"] = [ln for ln in tail if "mlx" in ln.lower()][-12:]
    print(f"[{name}] install rc={rc} in {out['install_s']}s", flush = True)
    for ln in out["install_mlx_lines"]:
        print(f"[{name}]   {ln}", flush = True)
    if rc != 0:
        out["install_tail"] = tail[-40:]
        return out
    py = home / "unsloth_studio" / "bin" / "python"
    out["has_mlx"] = (
        subprocess.run(
            [str(py), "-c", "import mlx.core, mlx_lm, mlx_vlm"], capture_output = True
        ).returncode
        == 0
    )
    out["has_torch"] = (
        subprocess.run([str(py), "-c", "import torch"], capture_output = True).returncode == 0
    )
    port = _port()
    base = f"http://127.0.0.1:{port}"
    slog = work / f"studio_{name}.log"
    p = subprocess.Popen(
        [str(home / "bin" / "unsloth"), "studio", "-H", "127.0.0.1", "-p", str(port)],
        cwd = tree,
        env = env,
        stdout = open(slog, "w"),
        stderr = subprocess.STDOUT,
        start_new_session = True,
    )
    try:
        deadline = time.time() + 300
        while time.time() < deadline:
            try:
                if _req("GET", f"{base}/healthz", timeout = 5)[0] == 200:
                    break
            except OSError:
                pass
            time.sleep(2)
        boot = home / "auth" / ".bootstrap_password"
        for _ in range(60):
            if boot.exists():
                break
            time.sleep(1)
        pw = boot.read_text().strip()
        _, tok = _req("POST", f"{base}/api/auth/login", {"username": "unsloth", "password": pw})
        _, tok = _req(
            "POST",
            f"{base}/api/auth/change-password",
            {"current_password": pw, "new_password": "ci-" + os.urandom(8).hex()},
            token = tok["access_token"],
        )
        token = tok["access_token"]
        _, health = _req("GET", f"{base}/api/health?detect_hardware=true", token = token, timeout = 300)
        out["health"] = (
            {
                k: health.get(k)
                for k in (
                    "device_type",
                    "apple_silicon",
                    "chat_only",
                    "chat_only_reason",
                    "mlx_inference",
                    "cpu_training",
                    "export_supported",
                    "export_unsupported_reason",
                )
            }
            if isinstance(health, dict)
            else health
        )
        print(f"[{name}] health {json.dumps(out['health'])}", flush = True)
        t1 = time.time()
        code, load = _req(
            "POST", f"{base}/api/inference/load", {"model_path": model}, token = token, timeout = 1200
        )
        out["load"] = {
            "status": code,
            "s": round(time.time() - t1, 1),
            "body": load
            if code != 200
            else {
                k: load.get(k)
                for k in ("status", "model", "is_gguf", "inference_backend")
                if isinstance(load, dict)
            },
        }
        print(f"[{name}] load {code}: {json.dumps(out['load']['body'])[:600]}", flush = True)
        if code == 200:
            code, chat = _req(
                "POST",
                f"{base}/v1/chat/completions",
                {
                    "model": model,
                    "messages": [{"role": "user", "content": PROMPT}],
                    "max_tokens": 24,
                    "temperature": 0,
                    "stream": False,
                },
                token = token,
                timeout = 600,
            )
            text = None
            if code == 200 and isinstance(chat, dict):
                text = ((chat.get("choices") or [{}])[0].get("message") or {}).get("content")
            out["chat"] = {"status": code, "text": text, "body": None if text else str(chat)[:1000]}
            print(f"[{name}] chat {code}: {text!r}", flush = True)
        code, tr = _req(
            "POST", f"{base}/api/train/start", {"model_name": model}, token = token, timeout = 120
        )
        out["train_start"] = {"status": code, "detail": str(tr)[:400]}
        print(f"[{name}] train/start {code}: {str(tr)[:300]}", flush = True)
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
        print(f"[{name}] ERROR {out['error']}", flush = True)
    finally:
        try:
            os.killpg(p.pid, signal.SIGTERM)
            p.wait(timeout = 30)
        except Exception:  # noqa: BLE001
            os.killpg(p.pid, signal.SIGKILL)
        out["studio_log_tail"] = [
            ln
            for ln in slog.read_text(errors = "replace").splitlines()
            if "mlx" in ln.lower() or "error" in ln.lower() or "torch" in ln.lower()
        ][-25:]
    return out


def _check(rec, name, ok, detail):
    rec["checks"][name] = {"ok": bool(ok), "detail": detail}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default = "mlx-community/SmolLM-135M-Instruct-4bit")
    ap.add_argument("--base-sha", default = None)
    ap.add_argument("--install-timeout", type = int, default = 1500)
    ap.add_argument("--out", default = "outputs/jobs/mac_studio_chat_only.json")
    a = ap.parse_args()
    rec = {
        "job": "mac_studio_chat_only",
        "model": a.model,
        "backend": "studio",
        "device": "mlx",
        "config": vars(a),
        "checks": {},
        "arms": {},
        "passed": False,
        "error": None,
        "skipped": False,
        "finished": False,
    }
    outp = Path(a.out)
    outp.parent.mkdir(parents = True, exist_ok = True)
    if sys.platform != "darwin" or platform.machine() != "arm64":
        rec.update(skipped = True, finished = True, error = "needs Apple Silicon macOS")
        outp.write_text(json.dumps(rec, indent = 1))
        print("JOB_RESULT " + json.dumps({"skipped": True}))
        return 3
    repo = Path.cwd()
    work = repo.parent / "mlx_chat_only_work"
    work.mkdir(exist_ok = True)
    rec["revisions"] = {
        "head": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output = True, text = True
        ).stdout.strip()
    }
    head = _arm("head", repo, work, a.model, a.install_timeout)
    rec["arms"]["head"] = head
    outp.write_text(json.dumps(rec, indent = 1))
    hh = head.get("health") or {}
    _check(rec, "head_install", head.get("install_rc") == 0, f"rc={head.get('install_rc')}")
    _check(
        rec,
        "head_mlx_installed_without_torch",
        head.get("has_mlx") and not head.get("has_torch"),
        f"mlx={head.get('has_mlx')} torch={head.get('has_torch')}",
    )
    _check(
        rec,
        "head_health_mlx",
        hh.get("mlx_inference") is True and hh.get("chat_only_reason") == "no_torch",
        json.dumps(hh),
    )
    _check(
        rec,
        "head_mlx_loads",
        (head.get("load") or {}).get("status") == 200,
        json.dumps(head.get("load"))[:500],
    )
    _check(
        rec,
        "head_chat_answers",
        bool((head.get("chat") or {}).get("text")),
        json.dumps(head.get("chat"))[:500],
    )
    _check(
        rec,
        "head_train_refused",
        (head.get("train_start") or {}).get("status") in (400, 422),
        json.dumps(head.get("train_start"))[:500],
    )
    if a.base_sha:
        bt = work / "base_tree"
        if not bt.exists():
            subprocess.run(
                [
                    "git",
                    "fetch",
                    "--depth",
                    "1",
                    "https://github.com/unslothai/unsloth",
                    a.base_sha,
                ],
                check = True,
            )
            subprocess.run(["git", "worktree", "add", "--detach", str(bt), a.base_sha], check = True)
        rec["revisions"]["base"] = a.base_sha
        base = _arm("base", bt, work, a.model, a.install_timeout)
        rec["arms"]["base"] = base
        _check(
            rec,
            "base_reproduces_defect",
            (base.get("load") or {}).get("status") != 200 or not base.get("has_mlx"),
            json.dumps({"has_mlx": base.get("has_mlx"), "load": base.get("load")})[:500],
        )
    rec["passed"] = all(c["ok"] for c in rec["checks"].values())
    rec["finished"] = True
    outp.write_text(json.dumps(rec, indent = 1))
    print(
        "JOB_RESULT "
        + json.dumps(
            {"passed": rec["passed"], "checks": {k: v["ok"] for k, v in rec["checks"].items()}}
        )
    )
    return 0 if rec["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
