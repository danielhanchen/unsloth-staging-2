# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Cross-platform reproduction harness for the Unsloth Studio GGUF "update available"
loop (issue unslothai/unsloth#7060) and the related false-update/partial family
(PRs #7113, #7033; #7031 already merged).

It downloads a real GGUF quant once, builds both the natural HF cache layout and
the Windows-no-Developer-Mode "no-symlink" layout (real file in snapshots/, empty
blobs/), then drives the REAL Studio backend update check
(hub.services.models.gguf_variants.get_gguf_variants_response) against several code
states and reports, as JSON:

  * (i)  the false "update available" badge and whether it persists across a
         simulated Update click (the #7060 loop),
  * (ii) whether a second hf_hub_download re-fetches a fully cached file (the
         load-triggered re-download #7113 does NOT touch),
  * (iii) whether a complete moved GGUF is mis-flagged partial (expected: no).

The update-check import happens in a child process (`--role check`) so each code
state runs against its own studio/backend source tree via PYTHONPATH, all sharing
one HF cache.

Roles:
  orchestrate (default) -- run the full matrix for one target, emit a JSON report.
  check                 -- import one backend tree, run the update check, print one JSON line.

Everything writes under a job-local HF_HOME; nothing touches the shared cache.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


# --------------------------------------------------------------------------- #
# GGUF metadata editing (self-contained; the `gguf` pip package is NOT a dep). #
# --------------------------------------------------------------------------- #
# Minimal reader that locates a string KV's value byte range so we can rewrite a
# chat template in place. Same-length rewrites keep the file byte-identical in
# size (Case C, the fix's blind spot); different-length rewrites change the size
# (Case B, a legitimately different file).

_GGUF_MAGIC = b"GGUF"
_GGUF_TYPE_STRING = 8
_GGUF_TYPE_ARRAY = 9
# value type -> (struct format, byte size); variable-length types handled inline.
_FIXED = {
    0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2),
    4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1),
    10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8),
}


def _read_kv_string_span(buf: bytes, off: int, str_len_fmt: str) -> tuple[int, int, int]:
    """Return (value_offset, value_len, next_offset) for a gguf string at `off`."""
    n = struct.calcsize(str_len_fmt)
    (length,) = struct.unpack_from(str_len_fmt, buf, off)
    val_off = off + n
    return val_off, length, val_off + length


def find_string_kv_spans(path: Path) -> dict[str, tuple[int, int]]:
    """Map each string-valued KV key -> (value_byte_offset, value_byte_len).

    Only walks the header (magic + KV section); stops before tensor info. Handles
    GGUF v2/v3 (u64 lengths/counts). Memory-maps so a large tokenizer-token array
    does not force a multi-GB read. Returns {} / partial dict if parsing stops.
    """
    import mmap
    spans: dict[str, tuple[int, int]] = {}
    with open(path, "rb") as fh:
        try:
            buf = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        except (ValueError, OSError):
            return {}
        try:
            if buf[:4] != _GGUF_MAGIC:
                return {}
            (version,) = struct.unpack_from("<I", buf, 4)
            if version < 2:
                return {}  # v1 uses u32 lengths; the real Unsloth GGUFs are v3.
            str_len_fmt = "<Q"
            off = 8
            off += 8  # tensor_count (u64), unused
            kv_count = struct.unpack_from("<Q", buf, off)[0]; off += 8
            for _ in range(kv_count):
                koff, klen, off = _read_kv_string_span(buf, off, str_len_fmt)
                key = bytes(buf[koff:koff + klen]).decode("utf-8", "replace")
                (vtype,) = struct.unpack_from("<I", buf, off); off += 4
                if vtype == _GGUF_TYPE_STRING:
                    voff, vlen, off = _read_kv_string_span(buf, off, str_len_fmt)
                    spans[key] = (voff, vlen)
                elif vtype == _GGUF_TYPE_ARRAY:
                    (sub,) = struct.unpack_from("<I", buf, off); off += 4
                    (count,) = struct.unpack_from("<Q", buf, off); off += 8
                    if sub == _GGUF_TYPE_STRING:
                        for _i in range(count):
                            _o, _l, off = _read_kv_string_span(buf, off, str_len_fmt)
                    elif sub in _FIXED:
                        off += _FIXED[sub][1] * count
                    else:
                        break  # unknown subtype; return what we have
                elif vtype in _FIXED:
                    off += _FIXED[vtype][1]
                else:
                    break  # unknown type; stop walking
        except (struct.error, IndexError, ValueError):
            pass  # return whatever KVs we resolved before the truncation
        finally:
            buf.close()
    return spans


def pick_editable_string_key(spans: dict[str, tuple[int, int]]) -> Optional[str]:
    """Prefer the chat template; else the longest string KV (needs room to edit)."""
    for key in ("tokenizer.chat_template", "tokenizer.ggml.chat_template"):
        if key in spans and spans[key][1] >= 8:
            return key
    editable = [(k, v) for k, v in spans.items() if v[1] >= 16 and not k.startswith("split.")]
    if not editable:
        return None
    return max(editable, key=lambda kv: kv[1][1])[0]


def edit_string_kv_same_length(path: Path, key: str) -> bool:
    """Overwrite a string KV's value in place with same-length different bytes.

    Keeps total file size identical (Case C). Returns True on success.
    """
    spans = find_string_kv_spans(path)
    if key not in spans:
        return False
    voff, vlen = spans[key]
    with open(path, "r+b") as f:
        f.seek(voff)
        original = f.read(vlen)
        # Flip to a clearly different, printable, same-length payload.
        replacement = bytes((b ^ 0x20) if 0x21 <= b <= 0x7D else (0x40 if b == 0x20 else b)
                            for b in original)
        if replacement == original:
            replacement = b"X" * vlen
        f.seek(voff)
        f.write(replacement)
    return True


def append_bytes_change_size(path: Path, n: int = 4096) -> bool:
    """Grow the file by n bytes (Case B, size-change fallback when a valid
    rewrite is not available). Only the on-disk size matters to the update
    check; the file is not re-parsed by that path."""
    with open(path, "ab") as f:
        f.write(b"\x00" * n)
    return True


# --------------------------------------------------------------------------- #
# HF cache layout helpers.                                                     #
# --------------------------------------------------------------------------- #

def _pin_hf_cache(hf_home) -> None:
    """Force every HF cache var to this job's home so a stale HF_HUB_CACHE in the
    ambient environment can never redirect the download or the cache scan."""
    hf_home = str(hf_home)
    os.environ["HF_HOME"] = hf_home
    os.environ["HF_HUB_CACHE"] = str(Path(hf_home) / "hub")
    os.environ.pop("HUGGINGFACE_HUB_CACHE", None)
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def _child_env(backend_path: str, hf_home: Path) -> dict:
    env = dict(os.environ)
    env["HF_HOME"] = str(hf_home)
    env["HF_HUB_CACHE"] = str(hf_home / "hub")
    env.pop("HUGGINGFACE_HUB_CACHE", None)
    env["PYTHONPATH"] = backend_path + os.pathsep + env.get("PYTHONPATH", "")
    return env


def repo_cache_dir(hf_home: Path, repo_id: str) -> Path:
    folder = "models--" + repo_id.replace("/", "--")
    return hf_home / "hub" / folder


def symlinks_supported(hf_home: Path) -> bool:
    try:
        from huggingface_hub.file_download import are_symlinks_supported
        return bool(are_symlinks_supported(str(hf_home / "hub")))
    except Exception:
        return False


def describe_layout(repo_dir: Path) -> dict:
    """Report the on-disk layout: is the snapshot a symlink, is blobs/ populated."""
    snaps = repo_dir / "snapshots"
    blobs = repo_dir / "blobs"
    snap_files = list(snaps.rglob("*.gguf")) if snaps.exists() else []
    blob_files = [p for p in blobs.iterdir() if p.is_file()] if blobs.exists() else []
    any_symlink = any(p.is_symlink() for p in snap_files)
    return {
        "snapshot_gguf_count": len(snap_files),
        "blob_file_count": len(blob_files),
        "any_snapshot_symlink": any_symlink,
        "layout": ("symlinked" if any_symlink and blob_files else "no_symlink"),
    }


def force_no_symlink_layout(repo_dir: Path) -> dict:
    """Dereference every snapshot symlink into a real file copy and empty blobs/.

    Reproduces the Windows-no-Developer-Mode cache layout on any OS.
    """
    snaps = repo_dir / "snapshots"
    blobs = repo_dir / "blobs"
    converted = 0
    if snaps.exists():
        for f in snaps.rglob("*"):
            if f.is_symlink():
                target = os.path.realpath(f)
                if os.path.isfile(target):
                    f.unlink()
                    shutil.copy2(target, f)
                    converted += 1
    if blobs.exists():
        for b in blobs.iterdir():
            if b.is_file():
                b.unlink()
    return {"converted_symlinks": converted, **describe_layout(repo_dir)}


def find_snapshot_file(repo_dir: Path, filename: str) -> Optional[Path]:
    snaps = repo_dir / "snapshots"
    if not snaps.exists():
        return None
    for f in snaps.rglob(filename):
        return f
    return None


# --------------------------------------------------------------------------- #
# Download (huggingface_hub only; no Studio import needed).                    #
# --------------------------------------------------------------------------- #

def hf_download(repo_id: str, filename: str, hf_home: Path, force: bool = False) -> tuple[str, float]:
    """Download one file into the job-local cache. Returns (path, seconds)."""
    from huggingface_hub import hf_hub_download
    _pin_hf_cache(hf_home)
    t0 = time.time()
    p = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        force_download=force,
        cache_dir=str(hf_home / "hub"),
        token=os.environ.get("HF_TOKEN"),
    )
    return p, time.time() - t0


def flip_bytes_same_length(path: Path, span: int = 256) -> bool:
    """Same-size content change fallback: flip a run of bytes mid-file (past the
    header, before EOF). Keeps byte size identical (Case C) for any GGUF."""
    size = path.stat().st_size
    if size < span * 4:
        return False
    off = size // 2
    with open(path, "r+b") as f:
        f.seek(off)
        chunk = f.read(span)
        f.seek(off)
        f.write(bytes(b ^ 0xFF for b in chunk))
    return True


# --------------------------------------------------------------------------- #
# Role: check -- import one backend tree and run the real update check.        #
# --------------------------------------------------------------------------- #

def role_resolve(args) -> int:
    """Print the exact files the variant's update check compares (main quant +
    the specific mmproj / MTP drafter companion), so the caller downloads exactly
    that set. Emits one JSON line: {ok, files:[...], error?}."""
    result = {"repo": args.repo, "quant": args.quant}
    try:
        _pin_hf_cache(args.hf_home)
        os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
        os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
        sys.path.insert(0, args.backend_path)
        import importlib
        GV = importlib.import_module("hub.services.models.gguf_variants")
        req = GV.gguf_variant_requirements(args.repo, args.quant, os.environ.get("HF_TOKEN"))
        files: list[str] = []
        if req is not None:
            for e in getattr(req, "expected_files", []):
                path = str(getattr(e, "path", "")).replace("\\", "/")
                keep = (
                    GV.is_main_gguf_variant_path(path, args.quant)
                    or GV._is_mmproj_filename(path)
                    or GV._is_mtp_drafter_path(path)
                )
                if keep and path:
                    files.append(path)
        result["ok"] = True
        result["files"] = sorted(set(files))
    except Exception as e:  # noqa: BLE001
        import traceback
        result["ok"] = False
        result["error"] = f"{type(e).__name__}: {e}"
        result["trace"] = traceback.format_exc()[-1200:]
    print(json.dumps(result))
    return 0


def role_check(args) -> int:
    result = {"backend": args.backend_path, "repo": args.repo, "quant": args.quant}
    try:
        _pin_hf_cache(args.hf_home)
        os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
        os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
        sys.path.insert(0, args.backend_path)

        import asyncio
        import importlib

        gguf_variants = importlib.import_module("hub.services.models.gguf_variants")
        try:
            cache_inventory = importlib.import_module("hub.services.models.cache_inventory")
            result["has_size_identity_fix"] = hasattr(cache_inventory, "local_size_identity")
        except Exception:
            result["has_size_identity_fix"] = None
        result["has_update_check"] = hasattr(
            gguf_variants, "_variant_update_available_from_requirement"
        )

        resp = asyncio.run(
            gguf_variants.get_gguf_variants_response(
                args.repo, hf_token=os.environ.get("HF_TOKEN")
            )
        )
        target = args.quant.lower()
        match = None
        for v in resp.variants:
            q = (getattr(v, "quant", "") or "").lower()
            fn = (getattr(v, "filename", "") or "").lower()
            if q == target or target in fn:
                match = v
                break
        if match is None:
            result["variant_found"] = False
            result["available_quants"] = [getattr(v, "quant", "") for v in resp.variants][:40]
        else:
            result["variant_found"] = True
            result["downloaded"] = bool(getattr(match, "downloaded", False))
            result["update_available"] = bool(getattr(match, "update_available", False))
            result["partial"] = bool(getattr(match, "partial", False))
            result["matched_quant"] = getattr(match, "quant", "")
        result["ok"] = True
    except Exception as e:  # noqa: BLE001 -- report, never crash the matrix cell
        import traceback
        result["ok"] = False
        result["error"] = f"{type(e).__name__}: {e}"
        result["trace"] = traceback.format_exc()[-1500:]
    print(json.dumps(result))
    return 0


# --------------------------------------------------------------------------- #
# Role: orchestrate -- run the full matrix for one target.                    #
# --------------------------------------------------------------------------- #

def resolve_files(code_state: dict, hf_home: Path, repo: str, quant: str) -> list[str]:
    """Run the `resolve` role to list the files this variant's update check compares."""
    cmd = [
        sys.executable, os.path.abspath(__file__), "--role", "resolve",
        "--backend-path", code_state["backend_path"],
        "--hf-home", str(hf_home), "--repo", repo, "--quant", quant,
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                             env=_child_env(code_state["backend_path"], hf_home))
        rec = json.loads(out.stdout.strip().splitlines()[-1])
        return rec.get("files", []) if rec.get("ok") else []
    except Exception:  # noqa: BLE001
        return []


def run_check(code_state: dict, hf_home: Path, repo: str, quant: str) -> dict:
    """Invoke this script in `check` mode against one code state's backend tree."""
    cmd = [
        sys.executable, os.path.abspath(__file__), "--role", "check",
        "--backend-path", code_state["backend_path"],
        "--hf-home", str(hf_home),
        "--repo", repo, "--quant", quant,
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                             env=_child_env(code_state["backend_path"], hf_home))
        line = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ""
        rec = json.loads(line) if line else {"ok": False, "error": "no output"}
        if not rec.get("ok"):
            rec.setdefault("stderr", out.stderr[-800:])
    except Exception as e:  # noqa: BLE001
        rec = {"ok": False, "error": f"subprocess {type(e).__name__}: {e}"}
    rec["code_state"] = code_state["name"]
    return rec


def role_orchestrate(args) -> int:
    hf_home = Path(args.hf_home).resolve()
    hf_home.mkdir(parents=True, exist_ok=True)
    _pin_hf_cache(hf_home)

    code_states = json.loads(args.code_states)  # [{name, backend_path}, ...]

    # Ask the backend which files this variant's update check compares (main
    # quant + the specific mmproj / MTP drafter companion), and download exactly
    # that set. Fall back to --filename + --companions if resolve fails.
    resolved = resolve_files(code_states[0], hf_home, args.repo, args.quant)
    if not resolved:
        resolved = [args.filename] + [c for c in (args.companions or "").split(",") if c]
    if args.filename not in resolved:
        resolved = [args.filename] + resolved
    companions = [f for f in resolved if f != args.filename]

    report: dict = {
        "os": sys.platform,
        "python": sys.version.split()[0],
        "repo": args.repo,
        "quant": args.quant,
        "main_file": args.filename,
        "companions": companions,
        "symlinks_supported": symlinks_supported(hf_home),
        "phases": {},
    }
    repo_dir = repo_cache_dir(hf_home, args.repo)

    def matrix(tag: str) -> None:
        report["phases"][tag] = {
            "layout": describe_layout(repo_dir),
            "checks": [run_check(cs, hf_home, args.repo, args.quant) for cs in code_states],
        }

    # -- Phase 0: download main file + companions (natural layout) -----------
    dl = {}
    path, secs = hf_download(args.repo, args.filename, hf_home)
    dl[args.filename] = {"seconds": round(secs, 1)}
    for c in companions:
        try:
            _p, cs = hf_download(args.repo, c, hf_home)
            dl[c] = {"seconds": round(cs, 1)}
        except Exception as e:  # noqa: BLE001
            dl[c] = {"error": str(e)}
    report["download"] = dl
    report["natural_layout"] = describe_layout(repo_dir)

    # -- Phase 1: natural layout (cross-platform companion/quant behavior) ----
    matrix("natural")

    # -- Phase 2: forced no-symlink layout, identical file (Case A / #7060) ----
    report["force_result"] = force_no_symlink_layout(repo_dir)
    matrix("forced_caseA_identical")

    # -- Phase 3: loop -- simulate "click Update" = force re-download + re-force
    _p, secs = hf_download(args.repo, args.filename, hf_home, force=True)
    report["loop_redownload_seconds"] = round(secs, 1)
    force_no_symlink_layout(repo_dir)
    matrix("forced_loop_after_update")

    # -- Phase 4: load re-download probe (behavior ii; #7113 does NOT fix) -----
    snap_before = find_snapshot_file(repo_dir, args.filename)
    mtime_before = snap_before.stat().st_mtime if snap_before else 0
    _p, secs = hf_download(args.repo, args.filename, hf_home, force=False)
    snap_after = find_snapshot_file(repo_dir, args.filename)
    layout_after = describe_layout(repo_dir)
    report["load_redownload_probe"] = {
        "seconds": round(secs, 2),
        # A real re-fetch takes many seconds and/or repopulates blobs/ or bumps mtime.
        "re_fetched": bool(secs > 5.0 or layout_after["blob_file_count"] > 0
                           or (snap_after and snap_after.stat().st_mtime != mtime_before)),
        "layout_after": layout_after,
    }
    force_no_symlink_layout(repo_dir)  # restore no-symlink for the edit cases

    # -- Phase 5: Case C -- equal-size chat-template edit (fix blind spot) ------
    snap = find_snapshot_file(repo_dir, args.filename)
    caseC = {"attempted": False}
    if snap is not None:
        size_before = snap.stat().st_size
        spans = find_string_kv_spans(snap)
        key = pick_editable_string_key(spans)
        if key:
            ok = edit_string_kv_same_length(snap, key)
            method = f"chat_template_kv:{key}"
        else:
            ok = flip_bytes_same_length(snap)
            method = "raw_midfile_flip"
        caseC = {"attempted": True, "method": method, "ok": ok,
                 "size_preserved": snap.stat().st_size == size_before}
    report["phases"]["forced_caseC_equal_size"] = {
        "edit": caseC,
        "layout": describe_layout(repo_dir),
        "checks": [run_check(cs, hf_home, args.repo, args.quant) for cs in code_states],
    }

    # -- Phase 6: Case B -- size-changing edit (legit difference) --------------
    if snap is not None:
        append_bytes_change_size(snap, 4096)
    report["phases"]["forced_caseB_size_changed"] = {
        "edit": {"method": "append_4096_bytes"},
        "layout": describe_layout(repo_dir),
        "checks": [run_check(cs, hf_home, args.repo, args.quant) for cs in code_states],
    }

    # -- Write + summarize ----------------------------------------------------
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"[repro_7060] wrote {out}")
    _print_summary(report)
    return 0


def _cell(checks: list[dict], name: str) -> dict:
    for c in checks:
        if c.get("code_state") == name:
            return c
    return {}


def _print_summary(report: dict) -> None:
    print("\n==================== SUMMARY ====================")
    print(f"OS={report['os']}  symlinks_supported={report['symlinks_supported']}  "
          f"natural_layout={report['natural_layout']['layout']}")
    print(f"target={report['repo']}  quant={report['quant']}")
    probe = report.get("load_redownload_probe", {})
    print(f"load re-download on no-symlink cache: re_fetched={probe.get('re_fetched')} "
          f"({probe.get('seconds')}s)  <-- #7113 does NOT address this")
    for tag, ph in report["phases"].items():
        parts = []
        for c in ph.get("checks", []):
            if not c.get("ok"):
                token = "ERR"
            else:
                token = f"upd={c.get('update_available')}"
                if not c.get("downloaded"):
                    token += "/notdl"
                if c.get("partial"):
                    token += "/partial"
            parts.append(f"{c.get('code_state')}:{token}")
        print(f"{tag:34} | " + "  ".join(parts))
    print("================================================\n")


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--role", choices=["orchestrate", "check", "resolve"], default="orchestrate")
    # check role
    p.add_argument("--backend-path")
    p.add_argument("--hf-home")
    p.add_argument("--repo")
    p.add_argument("--quant")
    # orchestrate role
    p.add_argument("--filename", help="main GGUF filename to download")
    p.add_argument("--companions", default="", help="comma-separated companion filenames")
    p.add_argument("--code-states", help='JSON: [{"name","backend_path"}, ...]')
    p.add_argument("--out", default="outputs/repro_7060_report.json")
    args = p.parse_args()

    if args.role == "check":
        return role_check(args)
    if args.role == "resolve":
        return role_resolve(args)
    return role_orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
