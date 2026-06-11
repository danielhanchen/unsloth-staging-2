#!/usr/bin/env python3
"""Real macOS end-to-end check of the in-app "Update llama.cpp" button.

Reproduces, on a GitHub-hosted macOS arm64 runner, exactly what a Mac user's
Studio backend does so we can see WHY the update button does or does not show:

  [0] detect_host()                  -> confirm Darwin / arm64 (+ macOS version)
  [1] install latest fork prebuilt   -> the macOS path setup.sh uses
                                        (--published-repo unslothai/llama.cpp).
                                        ASSERT a marker is written, i.e. the
                                        PREBUILT path was taken, not a source
                                        build (a source build writes no marker,
                                        so the button can never show).
  [2] binary discovery               -> UNSLOTH_LLAMA_CPP_PATH resolves the
                                        llama-server the finder would pick.
  [3] up-to-date control             -> get_update_status: update_available False.
  [4] behind latest (mark old)       -> get_update_status: supported True AND
                                        update_available True == button shows.
  [5] apply (start_update)           -> swaps in place to latest, new binary runs.
  [6] source-build case (no marker)  -> supported False == button correctly
                                        absent (documents the existing-Mac-user
                                        complaint: pre-prebuilt installs are
                                        source builds with no marker).

Only structlog is required: the installer is pure stdlib and the freshness
module needs nothing else. _find_binary (which would import the heavy inference
stack) is stubbed to the real installed binary; its only macOS-specific risk is
the install layout, which [1]/[2] cover directly.
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

# studio/backend/tests/macos_update_e2e.py -> repo root is parents[3].
ROOT = Path(__file__).resolve().parents[3]
STUDIO = ROOT / "studio"
INSTALLER = STUDIO / "install_llama_prebuilt.py"
BACKEND = STUDIO / "backend"
REPO = "unslothai/llama.cpp"
OLD_SENTINEL = "b9000"  # any tag < latest; only the string is compared

failures: list[str] = []


def check(cond: bool, label: str) -> bool:
    print(f"   [{'PASS' if cond else 'FAIL'}] {label}", flush=True)
    if not cond:
        failures.append(label)
    return cond


def find_marker(root: Path) -> dict | None:
    hits = list(root.rglob("UNSLOTH_PREBUILT_INFO.json"))
    return json.loads(hits[0].read_text()) if hits else None


def find_server(root: Path) -> Path | None:
    hits = [p for p in root.rglob("llama-server") if p.is_file()]
    return hits[0] if hits else None


def load_installer():
    import importlib.util
    spec = importlib.util.spec_from_file_location("ilp", INSTALLER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def resolve_like_finder(install_dir: Path) -> Path | None:
    """Mirror _find_llama_server_binary's UNSLOTH_LLAMA_CPP_PATH branch (the
    relevant one for a custom install dir): root, then build/bin."""
    name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
    for cand in (install_dir / name, install_dir / "build" / "bin" / name):
        if cand.is_file():
            return cand
    return None


def main() -> int:
    print(f"host: {platform.system()} {platform.machine()} | python {sys.version.split()[0]}", flush=True)

    # [0] detect_host
    print("\n[0] detect_host()", flush=True)
    ilp = load_installer()
    host = ilp.detect_host()
    print(f"   system={host.system} machine={host.machine} is_macos={host.is_macos} "
          f"is_arm64={host.is_arm64} macos_version={host.macos_version}", flush=True)
    if not check(host.is_macos, "running on macOS"):
        return 1

    work = ROOT / "_macos_update_e2e_work"
    if work.exists():
        shutil.rmtree(work)
    install_dir = work / "llama.cpp"
    install_dir.mkdir(parents=True)

    # [1] install the latest fork prebuilt -> must write a marker (prebuilt path)
    print(f"\n[1] install latest prebuilt: --published-repo {REPO} --llama-tag latest", flush=True)
    r = subprocess.run(
        [sys.executable, str(INSTALLER), "--install-dir", str(install_dir),
         "--llama-tag", "latest", "--published-repo", REPO],
        capture_output=True, text=True, timeout=1800,
    )
    print(f"   installer rc={r.returncode}", flush=True)
    if r.returncode != 0:
        print((r.stderr or r.stdout)[-2000:], flush=True)
    marker = find_marker(install_dir)
    check(marker is not None,
          "prebuilt install wrote UNSLOTH_PREBUILT_INFO.json (not a source build)")
    if marker is None:
        print("   -> no marker: this is the source-build case; button cannot show.", flush=True)
        return 1
    installed_tag = marker.get("tag") or marker.get("release_tag")
    print(f"   marker: tag={installed_tag} published_repo={marker.get('published_repo')} "
          f"asset={marker.get('asset')}", flush=True)
    check(marker.get("published_repo") == REPO, "marker.published_repo == fork")
    server = find_server(install_dir)
    check(server is not None and server.is_file(), "llama-server binary present")

    # [2] discovery via the env var the backend honors
    print("\n[2] binary discovery (UNSLOTH_LLAMA_CPP_PATH)", flush=True)
    os.environ["UNSLOTH_LLAMA_CPP_PATH"] = str(install_dir)
    resolved = resolve_like_finder(install_dir)
    check(resolved is not None, f"finder resolves llama-server -> {resolved}")

    # backend detection module (structlog-only); stub _find_binary to the real path
    sys.path.insert(0, str(BACKEND))
    upd = importlib.import_module("utils.llama_cpp_update")
    real_binary = str(resolved or server)
    upd._find_binary = lambda b=real_binary: b
    upd.reset_caches()
    upd._reset_job_for_tests()

    # [3] up-to-date control: just installed latest, so no update offered
    print("\n[3] up-to-date control", flush=True)
    st = upd.get_update_status(force_refresh=True)
    print(f"   supported={st['supported']} update_available={st['update_available']} "
          f"installed={st['installed_tag']} latest={st['latest_tag']}", flush=True)
    check(st["supported"], "supported True (marker found, repo known)")
    latest = st["latest_tag"]
    check(latest is not None, "latest_tag resolved from GitHub")
    check(st["installed_tag"] == latest and not st["update_available"],
          "fresh install is up to date (button hidden) - correct")

    # [4] behind latest -> the button must surface
    print("\n[4] behind latest (mark installed old)", flush=True)
    marker_path = next(install_dir.rglob("UNSLOTH_PREBUILT_INFO.json"))
    m = json.loads(marker_path.read_text())
    m["tag"] = OLD_SENTINEL
    m["release_tag"] = OLD_SENTINEL
    m.pop("install_fingerprint", None)  # force the apply step to reinstall
    marker_path.write_text(json.dumps(m, indent=2))
    upd.reset_caches()
    st = upd.get_update_status(force_refresh=True)
    print(f"   supported={st['supported']} update_available={st['update_available']} "
          f"installed={st['installed_tag']} latest={st['latest_tag']}", flush=True)
    check(st["supported"] and st["update_available"] and st["latest_tag"] == latest,
          f"UPDATE BUTTON SURFACES on macOS ({OLD_SENTINEL} -> {latest})")

    # [5] apply in place, confirm the new binary runs (no restart)
    print("\n[5] apply (start_update)", flush=True)
    upd._reset_job_for_tests()
    started = upd.start_update()
    print(f"   start_update -> {started.get('started')} reason={started.get('reason')}", flush=True)
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
    new_tag = (new_marker or {}).get("tag") or (new_marker or {}).get("release_tag")
    check(new_tag == latest, f"marker advanced {OLD_SENTINEL} -> {new_tag}")
    server2 = find_server(install_dir)
    if server2 is not None:
        v = subprocess.run([str(server2), "--version"], capture_output=True, text=True)
        ver = (v.stderr + v.stdout).strip().splitlines()[:1]
        print(f"   new llama-server --version: {ver}", flush=True)
        check("version" in (v.stderr + v.stdout).lower(), "swapped llama-server runs")

    # [6] source-build case (no marker) -> button correctly absent
    print("\n[6] source-build case (no marker)", flush=True)
    empty = work / "srcbuild" / "llama.cpp"
    (empty / "build" / "bin").mkdir(parents=True)
    fake = empty / "build" / "bin" / "llama-server"
    fake.write_text("stub")
    upd._find_binary = lambda b=str(fake): b
    upd.reset_caches()
    st = upd.get_update_status(force_refresh=True)
    print(f"   supported={st['supported']} update_available={st['update_available']}", flush=True)
    check(not st["supported"] and not st["update_available"],
          "no marker -> supported False (explains source-build Mac installs)")

    print("\n" + "=" * 60, flush=True)
    if failures:
        print(f"E2E FAIL ({len(failures)}): {failures}", flush=True)
        return 1
    print(f"E2E PASS: macOS update button works for prebuilt installs "
          f"({OLD_SENTINEL} -> {latest} in place, no restart). "
          f"Source builds (no marker) correctly show no button.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
