#!/usr/bin/env python3
"""End-to-end check that the in-app llama.cpp update works on CPU-only hosts
(Windows and macOS GitHub runners, no GPU).

Installs an older prebuilt with the real installer, confirms the update is
detected, applies it the way the Studio button does (get_update_status ->
start_update), and verifies the binary is swapped in place and advances. Both
runners use ggml-org (the CPU/upstream source) because it has an old + new pair.
"""
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
BACKEND = HERE.parents[1]                      # studio/backend
STUDIO = BACKEND.parent                        # studio
INSTALLER = STUDIO / "install_llama_prebuilt.py"
REPO = "ggml-org/llama.cpp"
OLD_TAG = "b9493"
IS_WINDOWS = platform.system() == "Windows"
SERVER = "llama-server.exe" if IS_WINDOWS else "llama-server"

sys.path.insert(0, str(BACKEND))
os.environ["UNSLOTH_LLAMA_INSTALLER"] = str(INSTALLER)


def find_server(root: Path):
    hits = list(root.rglob(SERVER))
    return str(hits[0]) if hits else None


def run(cmd):
    print("RUN:", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, capture_output=True, text=True)


def main() -> int:
    print(f"host: {platform.system()} {platform.machine()} | repo={REPO} | old={OLD_TAG}", flush=True)
    install_dir = (HERE.parent / "_e2e_install").resolve()

    # 1. Install the OLD prebuilt with the real installer.
    r = run([sys.executable, str(INSTALLER), "--install-dir", str(install_dir),
             "--llama-tag", OLD_TAG, "--published-repo", REPO])
    print((r.stdout or "")[-1200:], flush=True)
    if r.returncode != 0:
        print("INSTALL FAILED:\n", (r.stderr or "")[-2000:], flush=True)
        return 1

    binary = find_server(install_dir)
    if not binary:
        print("no llama-server found after install", flush=True)
        return 1
    marker_path = next(install_dir.rglob("UNSLOTH_PREBUILT_INFO.json"))
    old_tag = json.loads(marker_path.read_text()).get("tag")
    v = run([binary, "--version"])
    print("installed tag:", old_tag, "| OLD --version:", (v.stdout + v.stderr).strip().splitlines()[:2], flush=True)

    # Drive the update modules without a full Studio backend.
    import types
    rp = types.ModuleType("routes"); rp.__path__ = []
    ri = types.ModuleType("routes.inference")
    ri.get_llama_cpp_backend = lambda: (_ for _ in ()).throw(RuntimeError("no backend"))
    sys.modules["routes"] = rp
    sys.modules["routes.inference"] = ri

    import utils.llama_cpp_freshness as fr
    import utils.llama_cpp_update as upd
    upd._find_binary = lambda: find_server(install_dir)
    fr.reset_caches()

    # 2. Detection (what the banner reads).
    st = upd.get_update_status(force_refresh=True)
    print("update-status:", json.dumps({k: st[k] for k in
          ("supported", "update_available", "installed_tag", "latest_tag", "published_repo")}), flush=True)
    if not (st["supported"] and st["update_available"]):
        print("DETECTION FAILED: expected an available update", flush=True)
        return 1

    # 3. Apply (what the button does).
    res = upd.start_update()
    print("start_update:", res.get("started"), res.get("reason"), flush=True)
    if not res["started"]:
        return 1
    deadline = time.time() + 1500
    job = None
    while time.time() < deadline:
        job = upd.get_update_status()["job"]
        if job["state"] in ("success", "error"):
            break
        time.sleep(5)
    print("job:", job and job["state"], "to_tag:", job and job.get("to_tag"), "msg:", job and job.get("message"), flush=True)
    if not job or job["state"] != "success":
        print("UPDATE FAILED:", job and job.get("error"), flush=True)
        return 1

    # 4. Verify the swap happened in place.
    new_tag = json.loads(next(install_dir.rglob("UNSLOTH_PREBUILT_INFO.json")).read_text()).get("tag")
    nb = find_server(install_dir)
    v2 = run([nb, "--version"])
    print("new tag:", new_tag, "| NEW --version:", (v2.stdout + v2.stderr).strip().splitlines()[:2], flush=True)
    if new_tag == old_tag:
        print("FAILED: marker tag did not advance", flush=True)
        return 1
    if new_tag != st["latest_tag"]:
        print(f"FAILED: marker {new_tag} != latest {st['latest_tag']}", flush=True)
        return 1
    if v2.returncode != 0:
        print("FAILED: updated binary did not run", flush=True)
        return 1
    print(f"E2E PASS: {old_tag} -> {new_tag} in place, no restart ({platform.system()} CPU)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
