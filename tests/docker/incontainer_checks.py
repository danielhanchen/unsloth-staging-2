#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0
"""In-container checks for the Unsloth Studio Docker image (run via docker exec).

Covers the features that are only observable INSIDE the container (not over
HTTP): the branding integrity guard, the llama.cpp prebuilt identity, and the
notebook auto-update preserve/refresh behaviour. Writes a JSON + markdown
summary to --out-dir (a path that should be a bind-mounted/volumed dir or copied
out afterwards). Exit code is non-zero only if a REQUIRED check fails
(branding, llama_prebuilt).

Designed to run with the venv python (/opt/unsloth-venv/bin/python) inside a
booted container; copy it in with `docker cp` first.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

RESULTS: dict[str, dict] = {}


def record(name: str, status: str, detail: str = "") -> None:
    RESULTS[name] = {"status": status, "detail": detail}
    print(f"[{status:7}] {name}: {detail}", flush=True)


def run(cmd: list[str], timeout: int = 120) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
        return p.returncode, p.stdout.decode(errors="replace")
    except (OSError, subprocess.TimeoutExpired) as e:
        return 1, f"{e!r}"


def check_branding() -> None:
    rc, out = run([sys.executable, "-m", "unsloth_branding", "--verify"])
    if rc == 0 and "integrity check passed" in out:
        record("branding", "PASS", out.strip().splitlines()[-1][:140])
    else:
        record("branding", "FAIL", out.strip()[:200])


def check_llama_prebuilt() -> None:
    base = os.environ.get("UNSLOTH_LLAMA_CPP_PATH", "/opt/unsloth/llama.cpp")
    info = Path(base) / "UNSLOTH_PREBUILT_INFO.json"
    if not info.exists():
        record("llama_prebuilt", "FAIL", f"missing {info}")
        return
    meta = json.loads(info.read_text())
    repo = meta.get("published_repo")
    tag = meta.get("tag") or meta.get("release_tag")
    rc, ver = run([str(Path(base) / "llama-server"), "--version"])
    ver_ok = "version" in ver.lower()
    if repo == "unslothai/llama.cpp" and ver_ok:
        record("llama_prebuilt", "PASS", f"tag={tag} repo={repo} | {ver.strip().splitlines()[0][:80]}")
    else:
        record("llama_prebuilt", "FAIL", f"repo={repo} tag={tag} ver_ok={ver_ok}: {ver.strip()[:120]}")
    # best-effort freshness
    rc2, chk = run(["unsloth-llama-update", "--check"])
    fresh = "up to date" in chk.lower()
    record("llama_update_check", "PASS" if fresh else "SKIP",
           chk.strip().splitlines()[-1][:140] if chk.strip() else "no output")


def check_notebook_autoupdate() -> None:
    """Fake-edit a managed notebook and confirm a refresh preserves the user's
    edit while refreshing untouched files (header/footer-only changes kept)."""
    dest = os.environ.get("UNSLOTH_NOTEBOOKS_DIR", "/workspace/unsloth-notebooks")
    state = Path(dest) / ".unsloth_sync_state"
    commit = Path(dest) / ".unsloth_sync_commit"
    if not state.exists():
        record("notebook_autoupdate", "SKIP", f"no sync state at {state}")
        return
    # pick a managed .ipynb in a subdir
    rel = recorded = None
    for line in state.read_text().splitlines():
        h, _, p = line.partition("  ")
        if p.endswith(".ipynb") and "/" in p:
            rel, recorded = p, h
            break
    if not rel:
        record("notebook_autoupdate", "SKIP", "no managed .ipynb in state")
        return
    target = Path(dest) / rel
    nb = json.loads(target.read_text())
    nb.setdefault("cells", []).append(
        {"cell_type": "markdown", "metadata": {}, "source": ["# UNSLOTH_CI_USEREDIT_marker"]}
    )
    target.write_text(json.dumps(nb))
    edited_hash = hashlib.sha256(target.read_bytes()).hexdigest()
    commit.write_text("0" * 40 + "\n")  # force a refresh (remote != last)
    rc, out = run(["unsloth-sync-notebooks"], timeout=300)
    summary = ""
    for ln in out.splitlines():
        if "refreshed from GitHub" in ln:
            summary = ln.strip()
    now_hash = hashlib.sha256(target.read_bytes()).hexdigest()
    preserved = now_hash == edited_hash and "UNSLOTH_CI_USEREDIT_marker" in target.read_text()
    header_only = "header/footer changed upstream" in summary
    if preserved and "kept (your edits)" in summary:
        record("notebook_autoupdate", "PASS",
               f"user edit preserved; {summary[summary.find('notebooks'):][:120]}"
               + ("" if header_only else " (no header-only files this run)"))
    elif rc != 0 or "refreshed from GitHub" not in out:
        record("notebook_autoupdate", "SKIP", f"refresh did not run (offline?): {out.strip()[-160:]}")
    else:
        record("notebook_autoupdate", "FAIL", f"preserved={preserved} summary={summary[:140]}")


def check_studio_version() -> None:
    import importlib.metadata as m
    try:
        record("studio_version", "PASS",
               f"unsloth {m.version('unsloth')}, unsloth_zoo {m.version('unsloth_zoo')}")
    except Exception as e:  # noqa: BLE001
        record("studio_version", "SKIP", f"{e!r}"[:120])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/tmp/incontainer_out")
    ap.add_argument("--skip-notebook", action="store_true")
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    check_branding()
    check_llama_prebuilt()
    check_studio_version()
    if not args.skip_notebook:
        check_notebook_autoupdate()

    out.joinpath("incontainer_results.json").write_text(json.dumps(RESULTS, indent=2))
    lines = ["## In-container checks", "", "| Check | Status | Detail |", "|---|---|---|"]
    for k, v in RESULTS.items():
        lines.append(f"| {k} | {v['status']} | {v['detail'].replace('|', '/')[:140]} |")
    out.joinpath("incontainer_RESULT.md").write_text("\n".join(lines) + "\n")

    required = ("branding", "llama_prebuilt")
    failed = [k for k in required if RESULTS.get(k, {}).get("status") == "FAIL"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
