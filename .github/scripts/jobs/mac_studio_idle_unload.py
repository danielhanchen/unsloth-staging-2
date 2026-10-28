#!/usr/bin/env python3
"""Live Studio idle auto-unload of an MLX chat model, base vs head on one Apple Silicon runner.

Run from the staged checkout (head). Base = upstream main fetched into a worktree. Each arm
launches its own backend (run.py --api-only), turns on the chat idle TTL (60 s floor, API-only
on, keep-KV on), loads an MLX model through /v1 auto-switch (API-loaded), waits for the idle
unload, reloads with a model-less request, then loads through /api/inference/load (UI-loaded)
and checks it survives the TTL. Exit 0 checks pass, 1 failed, 3 not Apple Silicon.
"""

import argparse
import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

MODEL = "mlx-community/SmolLM-135M-Instruct-4bit"
PW = "IdleUnload-CI-pw1!"
TTL = 60
PROMPT = " ".join(f"Fact {i}: the sky over city {i} is blue." for i in range(120))


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class Api:
    def __init__(self, base):
        self.base, self.tok = base, None

    def call(
        self,
        method,
        path,
        body = None,
        tok = None,
        timeout = 600,
    ):
        req = urllib.request.Request(
            self.base + path,
            method = method,
            data = None if body is None else json.dumps(body).encode(),
            headers = {"Content-Type": "application/json"},
        )
        t = tok or self.tok
        if t:
            req.add_header("Authorization", "Bearer " + t)
        try:
            with urllib.request.urlopen(req, timeout = timeout) as r:
                return r.status, json.loads(r.read() or b"null")
        except urllib.error.HTTPError as e:
            return e.code, (e.read() or b"")[:600].decode("utf-8", "replace")
        except (urllib.error.URLError, OSError) as e:
            return 0, str(e)


def status(api):
    code, s = api.call("GET", "/api/inference/status")
    return s if code == 200 and isinstance(s, dict) else {}


def resident(api):
    s = status(api)
    return bool(s.get("active_model") or s.get("loaded"))


def wait_until(
    pred,
    limit,
    step = 5.0,
):
    t0 = time.monotonic()
    while time.monotonic() - t0 < limit:
        if pred():
            return time.monotonic() - t0
        time.sleep(step)
    return None


def run_arm(name, tree, home, log_path):
    port = free_port()
    env = dict(os.environ, UNSLOTH_STUDIO_HOME = str(home), PYTHONUNBUFFERED = "1")
    env.pop("UNSLOTH_MODEL_IDLE_TTL", None)
    backend = tree / "studio" / "backend"
    log = open(log_path, "w")
    proc = subprocess.Popen(
        [
            sys.executable,
            "run.py",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--password",
            PW,
            "--api-only",
        ],
        cwd = backend,
        env = env,
        stdout = log,
        stderr = subprocess.STDOUT,
        start_new_session = True,
    )
    api = Api(f"http://127.0.0.1:{port}")
    r = {"arm": name, "port": port}
    try:
        up = wait_until(
            lambda: proc.poll() is None and api.call("GET", "/api/health", timeout = 5)[0] == 200,
            600,
            2,
        )
        if up is None:
            raise RuntimeError("backend never became healthy")
        code, tok = api.call("POST", "/api/auth/login", {"username": "unsloth", "password": PW})
        if code != 200:
            raise RuntimeError(f"login {code}: {tok}")
        api.tok = tok["access_token"]
        code, key = api.call("POST", "/api/auth/api-keys", {"name": "idle-ci"})
        if code != 200:  # password change forced first
            api.call(
                "POST",
                "/api/auth/change-password",
                {"current_password": PW, "new_password": PW + "x"},
            )
            api.tok = api.call(
                "POST", "/api/auth/login", {"username": "unsloth", "password": PW + "x"}
            )[1]["access_token"]
            code, key = api.call("POST", "/api/auth/api-keys", {"name": "idle-ci"})
        apikey = key["key"] if code == 200 else api.tok
        code, s = api.call(
            "PUT",
            "/api/settings/openai-auto-switch",
            {
                "enabled": True,
                "auto_unload_idle_seconds": TTL,
                "auto_unload_api_only": True,
                "auto_unload_keep_kv": True,
            },
        )
        r["settings"] = s
        chat = {
            "model": MODEL,
            "messages": [{"role": "user", "content": PROMPT + " Say hi."}],
            "max_tokens": 8,
            "temperature": 0,
        }
        t0 = time.monotonic()
        code, out = api.call("POST", "/v1/chat/completions", chat, tok = apikey)
        r["api_load"] = {
            "code": code,
            "s": round(time.monotonic() - t0, 1),
            "usage": out.get("usage") if isinstance(out, dict) else out,
        }
        r["api_loaded_status"] = status(api)
        r["api_loaded"] = code == 200 and resident(api)
        # Loop polls every 15 s; unload is due at TTL + one poll.
        gone = wait_until(lambda: not resident(api), TTL + 75) if r["api_loaded"] else None
        r["api_unloaded_after_s"] = None if gone is None else round(gone, 1)
        if gone is not None:
            nomodel = dict(chat)
            nomodel.pop("model")
            t0 = time.monotonic()
            code, out = api.call("POST", "/v1/chat/completions", nomodel, tok = apikey)
            r["reload"] = {
                "code": code,
                "s": round(time.monotonic() - t0, 1),
                "usage": out.get("usage") if isinstance(out, dict) else out,
            }
            r["reloaded_status"] = status(api)
            r["reloaded"] = code == 200 and resident(api)
        # UI load: pinned by "Only unload models loaded by the API".
        code, out = api.call("POST", "/api/inference/unload", {"model_path": MODEL})
        code, out = api.call("POST", "/api/inference/load", {"model_path": MODEL})
        r["ui_load"] = {"code": code, "detail": out if code != 200 else None}
        if code == 200 and resident(api):
            stayed = wait_until(lambda: not resident(api), TTL + 45)
            r["ui_stayed_resident"] = stayed is None
        r["mlx_lines"] = [
            ln.strip()[:300]
            for ln in open(log_path, errors = "replace")
            if "Idle auto-unload" in ln or "prompt cache" in ln.lower()
        ][-12:]
    except Exception as e:  # noqa: BLE001
        r["error"] = f"{type(e).__name__}: {e}"
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout = 30)
        except Exception:  # noqa: BLE001
            with __import__("contextlib").suppress(Exception):
                os.killpg(proc.pid, signal.SIGKILL)
        log.close()
    r["log_tail"] = open(log_path, errors = "replace").read()[-3000:]
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default = "mac_studio_idle_unload.json")
    ap.add_argument("--base-ref", default = "main")
    a = ap.parse_args()
    res = {
        "job": "mac_studio_idle_unload",
        "model": MODEL,
        "backend": "studio-mlx",
        "device": platform.platform(),
        "checks": {},
        "passed": False,
        "skipped": False,
        "error": None,
    }
    if sys.platform != "darwin" or platform.machine() != "arm64":
        res.update(skipped = True, error = "needs Apple Silicon")
        Path(a.out).write_text(json.dumps(res, indent = 1))
        print("JOB_RESULT " + json.dumps(res))
        sys.exit(3)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "-r",
            "studio/backend/requirements/studio.txt",
        ],
        check = False,
    )
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL)
    head = Path.cwd()
    work = Path(tempfile.mkdtemp(prefix = "idle_ci_"))
    base = work / "base"
    subprocess.run(
        ["git", "fetch", "--depth", "1", "https://github.com/unslothai/unsloth", a.base_ref],
        check = True,
    )
    subprocess.run(["git", "worktree", "add", "--detach", str(base), "FETCH_HEAD"], check = True)
    res["revisions"] = {
        "head": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output = True, text = True
        ).stdout.strip(),
        "base": subprocess.run(
            ["git", "-C", str(base), "rev-parse", "HEAD"], capture_output = True, text = True
        ).stdout.strip(),
    }
    arms = {}
    for name, tree in (("base", base), ("head", head)):
        arms[name] = run_arm(
            name, tree, work / f"home_{name}", Path(f"studio_{name}.log").resolve()
        )
        print(
            name,
            json.dumps({k: v for k, v in arms[name].items() if k != "log_tail"}, default = str),
            flush = True,
        )
        if arms[name].get("error"):
            print(arms[name]["log_tail"], flush = True)
    res["arms"] = arms
    b, h = arms["base"], arms["head"]
    c = res["checks"]
    c["both_arms_loaded_via_api"] = {
        "ok": bool(b.get("api_loaded") and h.get("api_loaded")),
        "detail": [b.get("api_load"), h.get("api_load")],
    }
    c["base_keeps_api_mlx_model_past_ttl"] = {
        "ok": b.get("api_unloaded_after_s") is None,
        "detail": b.get("api_unloaded_after_s"),
    }
    c["head_idle_unloads_api_mlx_model"] = {
        "ok": h.get("api_unloaded_after_s") is not None,
        "detail": h.get("api_unloaded_after_s"),
    }
    c["head_modelless_request_reloads"] = {"ok": bool(h.get("reloaded")), "detail": h.get("reload")}
    c["head_ui_loaded_model_stays"] = {
        "ok": h.get("ui_stayed_resident") is True,
        "detail": h.get("ui_load"),
    }
    c["base_ui_loaded_model_stays"] = {
        "ok": b.get("ui_stayed_resident") is True,
        "detail": b.get("ui_load"),
    }
    res["passed"] = all(v["ok"] for v in c.values())
    Path(a.out).write_text(json.dumps(res, indent = 1, default = str))
    for k, v in c.items():
        print(("PASS " if v["ok"] else "FAIL ") + k, json.dumps(v["detail"], default = str)[:400])
    with __import__("contextlib").suppress(Exception):
        subprocess.run(["git", "worktree", "remove", "--force", str(base)], check = False)
        shutil.rmtree(work, ignore_errors = True)
    print("JOB_RESULT " + json.dumps({k: res[k] for k in ("job", "passed", "skipped", "error")}))
    sys.exit(0 if res["passed"] else 1)


if __name__ == "__main__":
    main()
