"""Installer on a package index that lags PyPI (unslothai/unsloth#14031), base vs head, any OS.

A local PEP 503 index holding only unsloth / unsloth-zoo 2026.10.3 is listed ahead of PyPI
(UV_INDEX, uv first-index), so it shadows PyPI for those two packages exactly as a delayed
corporate mirror does. Desktop's floor (UNSLOTH_DESKTOP_BACKEND_VERSION) is set as the app
sets it. Base = main's installer (raw.githubusercontent), head = the checkout's. Both run
--no-torch in their own studio home. Passes when base fails to resolve and head installs the
mirror's newest release, writes the feed ceiling marker, and the backend reports itself ready.
"""

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

WINDOWS = os.name == "nt"
FEED_VERSION = "2026.10.3"
DESKTOP_VERSION = "2026.10.6"


def build_feed(root):
    feed = root / "feed"
    (feed / "wheels").mkdir(parents = True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "download",
            "--no-deps",
            "-q",
            "-d",
            str(feed / "wheels"),
            f"unsloth=={FEED_VERSION}",
            f"unsloth-zoo=={FEED_VERSION}",
        ],
        check = True,
    )
    for wheel in (feed / "wheels").iterdir():
        project = re.sub(r"[-_.]+", "-", wheel.name.split("-")[0]).lower()
        page = feed / "simple" / project
        page.mkdir(parents = True, exist_ok = True)
        (page / "index.html").write_text(
            f'<html><body><a href="../../wheels/{wheel.name}">{wheel.name}</a></body></html>',
            encoding = "utf-8",
        )
    return (feed / "simple").as_uri()


def run_arm(name, script, root, index_url):
    home = root / name
    home.mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith("UV_")}
    env.update(
        UNSLOTH_STUDIO_HOME = str(home / "studio"),
        UV_INDEX = index_url,
        UNSLOTH_DESKTOP_BACKEND_VERSION = DESKTOP_VERSION,
        UNSLOTH_SKIP_AUTOSTART = "1",
        UNSLOTH_DISABLE_AUTO_UPDATES = "1",
        PYTHONUTF8 = "1",
    )
    if WINDOWS:
        argv = [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            "--no-torch",
            "--no-rollback",
        ]
    else:
        argv = ["sh", str(script), "--no-torch", "--no-rollback"]
    started = time.time()
    proc = subprocess.run(
        argv,
        env = env,
        cwd = str(home),
        stdout = subprocess.PIPE,
        stderr = subprocess.STDOUT,
        text = True,
        encoding = "utf-8",
        errors = "replace",
        timeout = 3000,
    )
    out = proc.stdout
    (root.parent / f"{name}.log").write_text(out, encoding = "utf-8")
    venv = home / "studio" / "unsloth_studio"
    marker = venv / "unsloth_feed_ceiling.json"
    result = {
        "exit": proc.returncode,
        "secs": round(time.time() - started),
        "no_solution": "No solution found" in out,
        "warned": any("does not have unsloth" in line for line in out.splitlines()),
        "marker": json.loads(marker.read_text(encoding = "utf-8")) if marker.exists() else None,
    }
    cli = venv / ("Scripts/unsloth.exe" if WINDOWS else "bin/unsloth")
    if proc.returncode == 0 and cli.exists():
        cap = subprocess.run(
            [str(cli), "studio", "desktop-capabilities", "--json"],
            env = env,
            stdout = subprocess.PIPE,
            stderr = subprocess.DEVNULL,
            text = True,
            timeout = 600,
        )
        try:
            data = json.loads(cap.stdout)
            result["capabilities"] = {
                k: data.get(k) for k in ("version", "studio_install_ok", "llama_runtime_ok")
            }
        except ValueError:
            result["capabilities"] = None
    result["tail"] = out.splitlines()[-8:]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tiny", action = "store_true", help = "accepted for the staging harness; no model is used"
    )
    parser.add_argument("--out", default = "installer_lagging_index.json")
    parser.add_argument("--base-ref", default = "main")
    args = parser.parse_args()

    repo = Path.cwd()
    name = "install.ps1" if WINDOWS else "install.sh"
    root = Path(tempfile.mkdtemp(prefix = "lagidx-", dir = os.environ.get("RUNNER_TEMP"))) / "arms"
    root.mkdir()
    base = root.parent / f"base_{name}"
    url = f"https://raw.githubusercontent.com/unslothai/unsloth/{args.base_ref}/{name}"
    with urllib.request.urlopen(url, timeout = 60) as response:
        base.write_bytes(response.read())
    index_url = build_feed(root.parent)

    arms = {
        "base": run_arm("base", base, root, index_url),
        "head": run_arm("head", repo / name, root, index_url),
    }
    head = arms["head"]
    checks = {
        "base_fails_on_floor": {
            "ok": arms["base"]["exit"] != 0 and arms["base"]["no_solution"],
            "detail": arms["base"]["tail"][-3:],
        },
        "head_installs": {"ok": head["exit"] == 0, "detail": head["tail"][-3:]},
        "head_warns": {"ok": head["warned"], "detail": None},
        "head_marker": {
            "ok": head["marker"] == {"requested": DESKTOP_VERSION, "available": FEED_VERSION},
            "detail": head["marker"],
        },
        "head_backend_ready": {
            "ok": (head.get("capabilities") or {}).get("version") == FEED_VERSION
            and (head.get("capabilities") or {}).get("studio_install_ok") is True,
            "detail": head.get("capabilities"),
        },
    }
    passed = all(c["ok"] for c in checks.values())
    record = {
        "job": "installer_lagging_index",
        "device": platform.platform(),
        "arms": arms,
        "checks": checks,
        "passed": passed,
        "error": None,
        "skipped": False,
    }
    Path(args.out).write_text(json.dumps(record, indent = 1), encoding = "utf-8")
    shutil.rmtree(root, ignore_errors = True)
    print(
        "JOB_RESULT "
        + json.dumps({"passed": passed, "checks": {k: v["ok"] for k, v in checks.items()}})
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
