"""Shared pieces of the target adapters: the toolkit bundle at a git SHA, per-VM plan slices, artifact extraction
and record collection."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import shutil
import re
import subprocess
import sys
import tarfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOLKIT = HERE.parent  # scripts/diffusion_bench
if str(TOOLKIT) not in sys.path:
    sys.path.insert(0, str(TOOLKIT))
import common as C  # noqa: E402 - the toolkit's common (unsafe_relpath / contained), not this module
SKIP_PARTS = {"__pycache__", "outputs", ".pytest_cache"}
SKIP_SUFFIX = {".pyc", ".log", ".tgz", ".zip"}
MARK_BEGIN, MARK_END = "ARTIFACT_B64_BEGIN", "ARTIFACT_B64_END"



# A workflow pushed to a public fork / staging repo must not name the baseline UI (names, comments, paths, log
# lines, file names): it is "baseline" there. mac.prepare refuses one, amd/gen_workflow.write the same.
PUBLIC_FORBIDDEN = re.compile(r"bline", re.IGNORECASE)


def public_problems(text: str) -> list:
    """The lines of a generated workflow that name the baseline UI (empty = fine for a public push)."""
    return [f"{i}: {ln.strip()[:120]}" for i, ln in enumerate(text.splitlines(), 1) if PUBLIC_FORBIDDEN.search(ln)]

def _git(*args: str, cwd: Path = TOOLKIT) -> str:
    return subprocess.run(["git", *args], cwd = str(cwd), capture_output = True, text = True,
                          check = True).stdout.strip()


def toolkit_revision() -> dict:
    """{"sha", "dirty", "prefix", "root"} of the git tree holding this toolkit; sha None outside git."""
    try:
        root = _git("rev-parse", "--show-toplevel")
        prefix = _git("rev-parse", "--show-prefix").rstrip("/")
        sha = _git("rev-parse", "HEAD")
        dirty = bool(_git("status", "--porcelain", "--", "."))
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"sha": None, "dirty": True, "prefix": None, "root": None}
    return {"sha": sha, "dirty": dirty, "prefix": prefix, "root": root}


KIT = TOOLKIT.parent / "studio_test_kit"  # setup_studio.KIT: the Studio server launcher, beside diffusion_bench


def needs_kit(cells) -> bool:
    """A cell runs Studio over HTTP (backend studio_http): setup_studio.launch loads studio_test_kit/lifecycle.py,
    so whatever ships the toolkit to that host must ship the kit beside it."""
    return any((c or {}).get("backend") == "studio_http" for c in cells or [])


def kit_files(sha: str | None, dirty: bool) -> dict:
    """{arcname: bytes} for studio_test_kit: at the toolkit SHA when clean, else the tracked and unignored
    working-tree files (never a stray credential or ignored file)."""
    out: dict = {}
    if not KIT.is_dir():
        return out
    if sha and not dirty:
        raw = subprocess.run(["git", "archive", "--format=tar", sha, "--", "."], cwd = str(KIT),
                             capture_output = True, check = True).stdout
        with tarfile.open(fileobj = io.BytesIO(raw), mode = "r:") as tf:
            for m in tf.getmembers():
                if m.isfile() and not any(x in SKIP_PARTS for x in Path(m.name).parts):
                    out["studio_test_kit/" + m.name] = tf.extractfile(m).read()
        return out
    raw = subprocess.run(["git", "ls-files", "-co", "--exclude-standard", "-z", "--", "."], cwd = str(KIT),
                         capture_output = True).stdout
    for rel in sorted(x.decode() for x in raw.split(b"\0") if x):
        if (KIT / rel).is_file() and not any(x in SKIP_PARTS for x in Path(rel).parts):
            out["studio_test_kit/" + rel] = (KIT / rel).read_bytes()
    return out


def write_public_kit(toolkit_copy: Path, cells) -> Path | None:
    """For a public branch dir (Mac staging, AMD fork): the rewritten studio_test_kit beside the rewritten toolkit
    copy at ``toolkit_copy`` when a cell needs it (needs_kit); returns the dir written, or None."""
    if not needs_kit(cells):
        return None
    if not (KIT / "lifecycle.py").is_file():
        raise SystemExit(f"studio_http cells need {KIT}/lifecycle.py and it is missing")
    import public_bundle as PB  # noqa: PLC0415 - private side only

    return PB.write(Path(toolkit_copy).parent / "studio_test_kit", KIT)


def build_bundle(out_dir: Path, allow_dirty: bool = False, extra: dict | None = None, kit: bool = False) -> dict:
    """Tar the toolkit as committed at HEAD (git archive), so the code that runs remotely is exactly a recorded
    SHA. A dirty tree or a copy outside git is refused unless allow_dirty (then the working tree is tarred and the
    revision says "+dirty"). extra = {arcname: bytes} added beside diffusion_bench/ (the plan slice, etc); kit adds
    studio_test_kit/ (needs_kit: the Studio-over-HTTP launcher).
    Writes <out_dir>/bundle_<sha12>.tgz and bundle.json; returns the bundle.json dict plus "path"."""
    rev = toolkit_revision()
    if (rev["sha"] is None or rev["dirty"]) and not allow_dirty:
        raise SystemExit(f"toolkit at {TOOLKIT} is {'not in git' if rev['sha'] is None else 'dirty'}: commit first "
                         "(the bundle must be a recorded SHA), or pass --allow-dirty for an offline dry run")
    buf = io.BytesIO()
    names = []
    with tarfile.open(fileobj = buf, mode = "w:gz") as tf:
        if rev["sha"] and not rev["dirty"]:
            raw = subprocess.run(["git", "archive", "--format=tar", rev["sha"], "--", "."], cwd = str(TOOLKIT),
                                 capture_output = True, check = True).stdout
            with tarfile.open(fileobj = io.BytesIO(raw), mode = "r:") as src:
                for m in src.getmembers():
                    if not m.isfile() or any(p in SKIP_PARTS for p in Path(m.name).parts):
                        continue
                    data = src.extractfile(m).read()
                    _add_bytes(tf, "diffusion_bench/" + m.name, data)
                    names.append(m.name)
        else:
            # a dirty tree: tracked files plus untracked ones git does not ignore (never a stray credential file or
            # anything .gitignore'd); a copy outside git has no such list and bundles nothing it cannot name
            listed = None
            if rev["sha"]:
                raw = subprocess.run(["git", "ls-files", "-co", "--exclude-standard", "-z", "--", "."],
                                     cwd = str(TOOLKIT), capture_output = True, check = True).stdout
                listed = {x.decode() for x in raw.split(b"\0") if x}
            for p in sorted(TOOLKIT.rglob("*")):
                rel = p.relative_to(TOOLKIT)
                if not p.is_file() or any(x in SKIP_PARTS for x in rel.parts) or p.suffix in SKIP_SUFFIX:
                    continue
                if listed is not None and rel.as_posix() not in listed:
                    continue
                _add_bytes(tf, "diffusion_bench/" + rel.as_posix(), p.read_bytes())
                names.append(rel.as_posix())
        if kit:
            kf = kit_files(rev["sha"], rev["dirty"])
            if "studio_test_kit/lifecycle.py" not in kf:
                raise SystemExit(f"studio_http cells need {KIT}/lifecycle.py and it is missing")
            for arc, data in kf.items():
                _add_bytes(tf, arc, data)
                names.append(arc)
        for arc, data in (extra or {}).items():
            _add_bytes(tf, arc, data if isinstance(data, bytes) else str(data).encode())
    data = buf.getvalue()
    sha = rev["sha"] or "nogit"
    label = sha[:12] + ("-dirty" if rev["dirty"] else "")
    out_dir.mkdir(parents = True, exist_ok = True)
    path = out_dir / f"bundle_{label}.tgz"
    path.write_bytes(data)
    meta = {"scripts_sha": sha, "dirty": rev["dirty"], "prefix": rev["prefix"], "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data), "files": len(names), "extra": sorted(extra or {}), "built": time.strftime("%FT%TZ", time.gmtime()),
            "path": str(path)}
    (out_dir / "bundle.json").write_text(json.dumps(meta, indent = 1))
    return meta


def _add_bytes(tf: tarfile.TarFile, name: str, data: bytes) -> None:
    ti = tarfile.TarInfo(name)
    ti.size = len(data)
    ti.mode = 0o755 if name.endswith((".sh", ".py")) else 0o644
    ti.mtime = 0  # reproducible: same SHA -> same tarball bytes
    tf.addfile(ti, io.BytesIO(data))


def vm_slice(plan: dict, vm_id: str) -> dict:
    """The plan restricted to one VM: its schedule plus only the cells it runs (what the VM driver reads)."""
    vm = next((v for v in plan["vms"] if v["vm_id"] == vm_id), None)
    if vm is None:
        raise KeyError(f"vm {vm_id!r} not in plan ({[v['vm_id'] for v in plan['vms']]})")
    tags = {it["tag"] for slot in vm["slots"] for it in slot}
    return {"plan_name": plan["name"], "plan_sha256": plan_digest(plan), "revisions": plan.get("revisions") or {},
            "vm": vm, "cells": {c["tag"]: c for c in plan["cells"] if c["tag"] in tags},
            # downloads, prefetch, the VM's comparisons (parity.gate_plan) and revisions: common.plan_carry
            **{k: v for k, v in C.plan_carry(plan, vm).items() if k != "revisions"}}


def plan_digest(plan: dict) -> str:
    return hashlib.sha256(json.dumps(plan, sort_keys = True).encode()).hexdigest()[:16]


def vms_of(plan: dict, kind: str) -> list:
    return [v for v in plan["vms"] if v["target"].split(":", 1)[0] == kind]


def timed_tags(vm: dict) -> list:
    return [it["tag"] for slot in vm["slots"] for it in slot if it.get("timed")]


B64_LINE = re.compile(r"^[A-Za-z0-9+/=]+$")


def artifact_text(src: Path) -> str:
    """The text the driver printed, from an executed notebook (every cell output's text in order: Colab splits a long
    print over hundreds of stream outputs, and the JSON keys between them are base64-alphabet words) or a log."""
    raw = src.read_text(errors = "replace")
    if src.suffix != ".ipynb":
        return raw
    try:
        nb = json.loads(raw)
    except ValueError:
        return raw
    parts = []
    for cell in nb.get("cells") or []:
        for o in cell.get("outputs") or []:
            t = o.get("text") if o.get("output_type") == "stream" else (o.get("data") or {}).get("text/plain")
            parts.append("".join(t) if isinstance(t, list) else (t or ""))
    return "".join(parts)


def extract_artifact(text: str) -> bytes | None:
    """The base64 tarball the VM driver prints between ARTIFACT_B64_BEGIN / _END: only the lines that are pure
    base64 once a launcher's line prefix ("  | ") is stripped, so interleaved progress lines never corrupt it; None
    when the markers are missing, the block is truncated or it is not a gzip stream."""
    i, j = text.rfind(MARK_BEGIN), text.rfind(MARK_END)
    if i < 0 or j < i:
        return None
    lines = []
    for ln in text[i + len(MARK_BEGIN):j].replace("\r", "\n").split("\n"):
        ln = ln.strip()
        if ln.startswith("|"):
            ln = ln[1:].strip()
        if ln and B64_LINE.match(ln):
            lines.append(ln)
    try:
        data = base64.b64decode("".join(lines), validate = True)
    except Exception:  # noqa: BLE001 - a truncated output is "no artifact", reported by the caller
        return None
    return data if data[:2] == b"\x1f\x8b" else None


def unpack_artifacts(run_dir: Path) -> list:
    """Unpack every artifact tarball a launcher left under run_dir (Kaggle artifact.tgz, or the base64 block in an
    executed notebook / log) into run_dir/unpacked/<source>/. Returns the dirs written."""
    out = []
    srcs = sorted(run_dir.rglob("artifact.tgz")) + sorted(run_dir.rglob("*.ipynb")) + sorted(run_dir.rglob("*.log"))
    for src in srcs:
        if "unpacked" in src.parts:
            continue
        data = src.read_bytes() if src.suffix == ".tgz" else extract_artifact(artifact_text(src))
        if not data:
            continue
        dest = run_dir / "unpacked" / src.stem
        dest.mkdir(parents = True, exist_ok = True)
        try:
            tf = tarfile.open(fileobj = io.BytesIO(data), mode = "r:*")
        except tarfile.TarError as exc:  # one unreadable source never hides the others (log vs notebook copy)
            print(f"unpack {src.name}: not a readable artifact ({exc}); trying the other sources")
            continue
        with tf:
            # regular files / dirs only, relative, no "..", no drive letter, and resolving inside dest even through
            # a symlink already there (C.contained); the "data" filter on top where this Python has it
            safe = [m for m in tf.getmembers() if (m.isfile() or m.isdir()) and C.contained(dest, m.name) is not None]
            dropped = [m.name for m in tf.getmembers() if m not in safe]
            if dropped:
                print(f"unpack {src.name}: refused {len(dropped)} member(s) outside {dest}: {dropped[:5]}")
            tf.extractall(dest, members = safe, **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))
        out.append(dest)
    return out


def stage_cases(run_dir: Path, plan: str | Path | None) -> Path | None:
    """Copy plan.py's cases.json (written beside PLAN.json) into the results dir, where report.py reads its
    expected case set (report.py --expect PLAN_DIR/cases.json is the alternative). ``plan`` = PLAN.json or the
    cases.json itself; None = leave the results dir as it is."""
    if not plan:
        return None
    src = Path(plan)
    if src.name != "cases.json":
        src = src.parent / "cases.json"
    if not src.exists():
        raise SystemExit(f"{src} not found: plan.py --out PLAN.json writes cases.json beside the plan")
    run_dir.mkdir(parents = True, exist_ok = True)
    dst = run_dir / "cases.json"
    if src.resolve() == dst.resolve():
        return dst
    data = json.loads(src.read_text())
    cases = data.get("cases") if isinstance(data, dict) else None
    vm = run_dir.name  # targets name each VM's results dir after its vm_id (runs/<vm_id>)
    if isinstance(cases, list) and any(isinstance(c, dict) and c.get("vm_id") == vm for c in cases):
        # a multi-VM plan: this results dir holds one VM's cells, so it expects only that VM's cases (and the
        # planner refusals for its target); the whole plan's set made report.py refuse a complete VM
        mine = [c for c in cases if isinstance(c, dict) and c.get("vm_id") == vm]
        tgts = {c.get("target") for c in mine}
        data = {**data, "cases": mine, "vm_id": vm,
                "not_run": [r for r in data.get("not_run") or [] if not isinstance(r, dict) or r.get("target") in tgts]}
        dst.write_text(json.dumps(data, indent = 1))
    else:
        shutil.copyfile(src, dst)
    return dst


def collect_records(run_dir: Path, target: str | None = None, plan: str | Path | None = None) -> list:
    """Every record.json under run_dir (after unpacking artifacts), with target / arm copied from the cell's
    options when the record itself lacks them (pre-v2 records), and its path. ``plan`` (PLAN.json): its cases.json
    is copied into run_dir for report.py."""
    run_dir = Path(run_dir)
    stage_cases(run_dir, plan)
    unpack_artifacts(run_dir)
    from . import _remote  # noqa: PLC0415 - the leak scan, over the UNPACKED tree, for every target kind

    _remote.scan_leaks(run_dir)
    recs = []
    for p in sorted(run_dir.rglob("record.json")):
        try:
            text = p.read_text()
            import extnames  # noqa: PLC0415

            if extnames.SENTINEL in text.lower():  # a public copy's record (AMD / Mac): back to the toolkit's names
                import public_bundle  # noqa: PLC0415 - private side only

                text = public_bundle.restore_text(text)
                p.write_text(text)
            r = json.loads(text)
        except (OSError, ValueError) as exc:
            recs.append({"verdict": "BROKEN", "reason": f"unreadable record {p}: {exc}", "_path": str(p)})
            continue
        opts = (r.get("cell") or {}).get("options") or {}
        r.setdefault("target", opts.get("target") or ({"name": target} if target else None))
        r.setdefault("arm", opts.get("arm"))
        r["_path"] = str(p)
        recs.append(r)
    # one entry per run: a payload unpacked from several copies (artifact.tgz + the notebook / log base64 block)
    # must not become extra fresh-process repeats
    recs, dropped = schema_mod().dedup(recs)
    if dropped:
        print(f"collect: dropped {dropped} duplicate record copies (same run_id) under {run_dir}")
    return recs


def schema_mod():
    import schema  # noqa: PLC0415 - toolkit root (on sys.path via this module's import)

    return schema


def missing_cells(plan: dict, records: list, vm_ids: list | None = None) -> list:
    """Planned timed / quality tags of these VMs with no record: the NOT_RUN list a report must show."""
    have = {(r.get("tag") or (r.get("cell") or {}).get("tag")) for r in records}
    out = []
    for vm in plan["vms"]:
        if vm_ids and vm["vm_id"] not in vm_ids:
            continue
        for slot in vm["slots"]:
            for it in slot:
                if it["tag"] not in have:
                    out.append({"tag": it["tag"], "vm_id": vm["vm_id"], "verdict": "NOT_RUN",
                                "reason": "no record came back"})
    return out
