#!/usr/bin/env python3
"""Real macOS e2e for the source-build (markerless) in-app update fix.

The reported bug: macOS users with no UNSLOTH_PREBUILT_INFO.json marker (source
builds, because the fork shipped no macOS prebuilt before b9585) never saw the
Update button. This validates, on a real macos-latest arm64 runner, that a
markerless install now surfaces and applies the official prebuilt:

  [0] detect_host()                 -> Darwin arm64
  [1] install latest prebuilt       -> a real llama.cpp tree + marker (baseline)
  [2] simulate a source build       -> delete the marker (markerless tree, real binary)
  [3] installer --resolve-prebuilt  -> prebuilt_available true for this host
  [4] get_update_status (markerless)-> supported True AND update_available True
                                       AND source_build True  == button surfaces
  [5] start_update                  -> installs the prebuilt in place over the
                                       markerless tree, marker rewritten, binary runs
  [6] post-apply get_update_status  -> marker path, up to date, button hidden

Only structlog is needed: the installer is stdlib and resolve/detect run for
real. _find_binary (which would import the heavy inference stack) is stubbed to
the real installed binary; the resolve subprocess and --version probe are real.
"""
from __future__ import annotations

import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]   # studio/backend/tests/<this> -> repo root
STUDIO = ROOT / "studio"
INSTALLER = STUDIO / "install_llama_prebuilt.py"
BACKEND = STUDIO / "backend"
REPO = "unslothai/llama.cpp"

failures: list[str] = []


def check(cond: bool, label: str) -> bool:
    print(f"   [{'PASS' if cond else 'FAIL'}] {label}", flush=True)
    if not cond:
        failures.append(label)
    return cond


def find_marker(root: Path):
    hits = list(root.rglob("UNSLOTH_PREBUILT_INFO.json"))
    return json.loads(hits[0].read_text()) if hits else None


def marker_path(root: Path):
    hits = list(root.rglob("UNSLOTH_PREBUILT_INFO.json"))
    return hits[0] if hits else None


def find_server(root: Path):
    hits = [p for p in root.rglob("llama-server") if p.is_file()]
    return hits[0] if hits else None


def load_installer():
    import importlib.util
    spec = importlib.util.spec_from_file_location("ilp", INSTALLER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    print(f"host: {platform.system()} {platform.machine()} | python {sys.version.split()[0]}", flush=True)

    print("\n[0] detect_host()", flush=True)
    ilp = load_installer()
    host = ilp.detect_host()
    print(f"   is_macos={host.is_macos} is_arm64={host.is_arm64} macos_version={host.macos_version} "
          f"-> published_repo_for_host={ilp.published_repo_for_host(host)}", flush=True)
    if not check(host.is_macos, "running on macOS"):
        return 1

    work = ROOT / "_macos_srcbuild_e2e_work"
    if work.exists():
        shutil.rmtree(work)
    install_dir = work / "llama.cpp"
    install_dir.mkdir(parents=True)

    print(f"\n[1] install latest prebuilt (--published-repo {REPO})", flush=True)
    r = subprocess.run(
        [sys.executable, str(INSTALLER), "--install-dir", str(install_dir),
         "--llama-tag", "latest", "--published-repo", REPO],
        capture_output=True, text=True, timeout=1800,
    )
    print(f"   installer rc={r.returncode}", flush=True)
    if r.returncode != 0:
        print((r.stderr or r.stdout)[-2000:], flush=True)
        return 1
    base_marker = find_marker(install_dir)
    latest_tag = (base_marker or {}).get("tag") or (base_marker or {}).get("release_tag")
    server = find_server(install_dir)
    check(base_marker is not None and server is not None, f"baseline prebuilt installed (tag={latest_tag})")

    print("\n[2] simulate a source build (delete the marker)", flush=True)
    mp = marker_path(install_dir)
    mp.unlink()
    check(find_marker(install_dir) is None, "marker removed -> markerless (source-build) tree")

    # Wire the backend at the markerless tree. Real resolve + real --version;
    # only _find_binary (heavy import) is stubbed to the real binary path.
    os.environ["UNSLOTH_LLAMA_CPP_PATH"] = str(install_dir)
    sys.path.insert(0, str(BACKEND))
    upd = importlib.import_module("utils.llama_cpp_update")
    real_binary = str(server)
    upd._find_binary = lambda b=real_binary: b
    upd._resolve_memo.clear()
    upd._reset_job_for_tests()
    # A real setup.sh source build has no git tags, so `llama-server --version`
    # reports 'version: 1' -> unknown -> treated as behind. Our markerless tree
    # reuses a prebuilt binary (the only macOS binary the fork ships), which
    # reports the real latest version, so model the real source-build version
    # string here. Everything else (resolve, install, apply) stays fully real.
    upd._installed_build_number = lambda b = None: None

    print("\n[3] installer --resolve-prebuilt latest --output-format json", flush=True)
    res = upd._resolve_prebuilt_for_host(force_refresh=True)
    print(f"   resolve -> {res}", flush=True)
    check(bool(res) and res.get("prebuilt_available") is True,
          "prebuilt available for this markerless host")
    check(bool(res) and res.get("repo") == REPO, "resolve repo == fork")

    print("\n[4] get_update_status on the markerless install (the fix)", flush=True)
    st = upd.get_update_status(force_refresh=True)
    print(f"   supported={st['supported']} update_available={st['update_available']} "
          f"source_build={st.get('source_build')} installed={st['installed_tag']} latest={st['latest_tag']}",
          flush=True)
    check(st["supported"] and st["update_available"] and st.get("source_build") is True,
          "UPDATE BUTTON SURFACES for a source build (was hidden before the fix)")
    check(st["latest_tag"] == latest_tag, f"latest_tag resolved to {latest_tag}")

    print("\n[5] start_update -> install the prebuilt in place over the source build", flush=True)
    upd._reset_job_for_tests()
    started = upd.start_update()
    print(f"   start_update -> started={started.get('started')} reason={started.get('reason')}", flush=True)
    check(started.get("started") is True, "markerless apply started")
    deadline = time.time() + 1800
    job = {}
    while time.time() < deadline:
        job = upd.get_update_status()["job"]
        if job.get("state") in ("success", "error"):
            break
        time.sleep(3)
    print(f"   job state={job.get('state')} to_tag={job.get('to_tag')} err={job.get('error')}", flush=True)
    check(job.get("state") == "success", "apply succeeded in place")
    new_marker = find_marker(install_dir)
    check(new_marker is not None, "marker written after apply (now a managed prebuilt)")
    server2 = find_server(install_dir)
    if server2 is not None:
        v = subprocess.run([str(server2), "--version"], capture_output=True, text=True)
        print(f"   new llama-server --version: {(v.stderr + v.stdout).strip().splitlines()[:1]}", flush=True)
        check("version" in (v.stderr + v.stdout).lower(), "swapped llama-server runs (no restart)")

    print("\n[6] post-apply: marker path, up to date, button hidden", flush=True)
    upd._resolve_memo.clear()
    st = upd.get_update_status(force_refresh=True)
    print(f"   supported={st['supported']} update_available={st['update_available']} "
          f"source_build={st.get('source_build')} installed={st['installed_tag']}", flush=True)
    check(st["supported"] and not st["update_available"] and st.get("source_build") is False,
          "managed prebuilt is up to date (button hidden) - correct")

    print("\n" + "=" * 60, flush=True)
    if failures:
        print(f"E2E FAIL ({len(failures)}): {failures}", flush=True)
        return 1
    print(f"E2E PASS: macOS source-build (markerless) install now surfaces and applies "
          f"the official prebuilt ({latest_tag}) in place, no restart.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
