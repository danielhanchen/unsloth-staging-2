#!/usr/bin/env python3
"""End-to-end check that the in-app llama.cpp update works on CPU-only hosts
(Windows and macOS GitHub runners, no GPU).

Installs a prebuilt, makes the install look behind the latest release, applies
the update the way the Studio button does (get_update_status -> start_update),
and verifies the binary is swapped in place and advances.

Per OS source:
- Windows: ggml-org (the CPU/upstream source) has an old + new pair, so install
  an older tag and update to latest for real.
- macOS: real Studio sources macOS from the unsloth fork (it enforces a macOS
  deployment-target floor so the binary runs on the runner; the plain upstream
  macOS bundles target a newer minimum OS). The fork only publishes the latest
  macOS tag, so install latest, mark the marker as an older tag, then update.
"""
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
BACKEND = HERE.parents[1]
STUDIO = BACKEND.parent
INSTALLER = STUDIO / "install_llama_prebuilt.py"

IS_WINDOWS = platform.system() == "Windows"
IS_MAC = platform.system() == "Darwin"
SERVER = "llama-server.exe" if IS_WINDOWS else "llama-server"
FAKE_OLD = "b9000"

if IS_MAC:
    REPO, MODE, OLD_TAG = "unslothai/llama.cpp", "latest-then-mark-old", None
else:  # Windows (and a Linux smoke run) use the upstream CPU source.
    REPO, MODE, OLD_TAG = "ggml-org/llama.cpp", "install-old", "b9493"

sys.path.insert(0, str(BACKEND))
os.environ["UNSLOTH_LLAMA_INSTALLER"] = str(INSTALLER)


def find_server(root: Path):
    hits = list(root.rglob(SERVER))
    return str(hits[0]) if hits else None


def marker_path(root: Path) -> Path:
    return next(root.rglob("UNSLOTH_PREBUILT_INFO.json"))


def run(cmd):
    print("RUN:", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, capture_output=True, text=True)


def install(install_dir: Path, tag: str) -> int:
    r = run([sys.executable, str(INSTALLER), "--install-dir", str(install_dir),
             "--llama-tag", tag, "--published-repo", REPO])
    print((r.stdout or "")[-1200:], flush=True)
    if r.returncode != 0:
        print("INSTALL FAILED:\n", (r.stderr or "")[-2500:], flush=True)
    return r.returncode


def main() -> int:
    print(f"host: {platform.system()} {platform.machine()} | repo={REPO} | mode={MODE}", flush=True)
    install_dir = (HERE.parent / "_e2e_install").resolve()

    # 1. Get a runnable install, then make it look behind the latest release.
    if MODE == "install-old":
        if install(install_dir, OLD_TAG) != 0:
            return 1
        old_tag = json.loads(marker_path(install_dir).read_text()).get("tag")
    else:  # latest-then-mark-old
        if install(install_dir, "latest") != 0:
            return 1
        mp = marker_path(install_dir)
        m = json.loads(mp.read_text())
        m["tag"] = FAKE_OLD
        m["release_tag"] = FAKE_OLD
        m.pop("install_fingerprint", None)  # force a real re-install on update
        mp.write_text(json.dumps(m, indent=2))
        old_tag = FAKE_OLD

    binary = find_server(install_dir)
    if not binary:
        print("no llama-server found after install", flush=True)
        return 1
    v = run([binary, "--version"])
    print("installed tag:", old_tag, "| --version:", (v.stdout + v.stderr).strip().splitlines()[:2], flush=True)

    # Drive the update modules without a full Studio backend (fail-open).
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

    # 4. Verify the swap happened in place and the new binary runs.
    new_tag = json.loads(marker_path(install_dir).read_text()).get("tag")
    nb = find_server(install_dir)
    v2 = run([nb, "--version"])
    print("new tag:", new_tag, "| --version:", (v2.stdout + v2.stderr).strip().splitlines()[:2], flush=True)
    if new_tag == old_tag or new_tag != st["latest_tag"] or v2.returncode != 0:
        print(f"FAILED: old={old_tag} new={new_tag} latest={st['latest_tag']} rc={v2.returncode}", flush=True)
        return 1
    print(f"E2E PASS: {old_tag} -> {new_tag} in place, no restart ({platform.system()} CPU)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
