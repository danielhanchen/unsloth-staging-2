#!/usr/bin/env python3
"""Mirror-staged baseline files: the rule and its verification.

A baseline-UI file the model card has no official source for (``"source": null``: the template names a file only a
mirror publishes) may be staged from that mirror ONLY when it is verified against the official checkpoint it
repackages, and the record says which mirror and how it was verified. A request ``downloads`` entry for a mirror:

  {"repo": "<mirror repo>", "file": "<path>", "sha256": "<the mirror file's sha256: pins the bytes verified>",
   "mirror": true,
   "official": {"repo": "<official repo>", "file": "<path>" | "files": ["<shard>", ...], "sha256": "<published>"},
   "verify": {"method": "sha256" | "tensors", "key_map": {...} | null, "cast": false}}

  method "sha256"   the official repo publishes a file with the SAME sha256 as the mirror file (official.sha256 ==
                    sha256): checked at plan time, and the download's own sha256 check proves the bytes
  method "tensors"  on the VM, after the download: every tensor of the mirror file equals (torch.equal, same dtype and
                    shape) its official tensor, read from the official files; key_map maps a mirror name to an
                    official one, "{i}" a number wildcard: {"a.{i}.w": "b.{i}.w"} renames, {"a.{i}.qkv": {"cat":
                    ["a.{i}.q", "a.{i}.k", "a.{i}.v"], "dim": 0}} is a fused tensor; "cast": true compares after
                    casting the official tensor to the mirror's dtype (a publisher that stored F32 as BF16). A mirror
                    tensor with no official counterpart fails it; official tensors the mirror leaves out are listed.

A quantised derivative (int8 ConvRot, fp8 with requantised scales, GPTQ) is not a repackage of any official
checkpoint: it can be neither hash- nor tensor-verified, so it stays NOT_RUN (a builder from the official weights is
the route for those). Legacy ``"staged_from_mirror": true`` entries without a ``verify`` block are refused.

  python diffusion_bench/mirror.py MIRROR.safetensors OFFICIAL.safetensors [...] [--key-map JSON] [--cast]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

METHODS = ("sha256", "tensors")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def is_mirror(entry) -> bool:
    return isinstance(entry, dict) and bool(entry.get("mirror") or entry.get("staged_from_mirror"))


def official_files(entry: dict) -> list:
    off = (entry or {}).get("official") or {}
    return [f for f in ([off.get("file")] if off.get("file") else []) + list(off.get("files") or []) if f]


def plan_problems(entry, force: bool = False) -> list:
    """Why a mirror entry may not be staged ([] = allowed): no official checkpoint named, no or an unknown
    verification method, a published-hash claim that does not hold, a tensor check without pinned mirror bytes.
    ``force``: treat the entry as a mirror copy even when it does not say so (the planner, for a file the card has
    no official source for)."""
    if not isinstance(entry, dict):
        return ["not a downloads entry"]
    if not (force or is_mirror(entry)):
        return []
    out = []
    off = entry.get("official") or {}
    ver = entry.get("verify") or {}
    method = ver.get("method")
    if not off.get("repo"):
        out.append("names no official checkpoint it repackages (official.repo)")
    if method not in METHODS:
        out.append(f"no verification against the official checkpoint (verify.method {method!r}, want one of "
                   f"{'/'.join(METHODS)})")
    sha = str(entry.get("sha256") or "").lower()
    if not HEX64.match(sha):
        out.append("mirror sha256 not pinned (the verified bytes must be the bytes used)")
    if method == "sha256":
        pub = str(off.get("sha256") or "").lower()
        if not HEX64.match(pub):
            out.append("verify.method sha256 without the official file's published sha256 (official.sha256)")
        elif pub != sha:
            out.append(f"published sha256 {pub[:12]}... of {off.get('repo')} != the mirror file's {sha[:12]}...")
        if not official_files(entry):
            out.append("verify.method sha256 without the official file (official.file)")
    if method == "tensors" and not official_files(entry):
        out.append("verify.method tensors without the official files to compare (official.file / files)")
    return out


def describe(entry: dict, result: Optional[dict] = None) -> dict:
    """The record's account of one staged file: mirror, official checkpoint, method, and the result when run."""
    off = entry.get("official") or {}
    out = {"mirror": {"repo": entry.get("repo"), "file": entry.get("file"), "sha256": entry.get("sha256")},
           "official": {"repo": off.get("repo"), "files": official_files(entry), "sha256": off.get("sha256")},
           "method": (entry.get("verify") or {}).get("method")}
    if result is not None:
        out["result"] = result
    elif out["method"] == "sha256" and not plan_problems(entry):
        out["result"] = {"verified": True, "detail": "mirror sha256 == the official file's published sha256 "
                                                     "(download sha256-checked)"}
    return out


# ---------------------------------------------------------------------------------------------- records
def unsourced(card: Optional[dict], fmt: Optional[str], stack: str = "blineui") -> dict:
    """{file name: its card entry} for the files the card's format has no official source for ({"source": null})."""
    block = ((((card or {}).get("formats") or {}).get(fmt or "") or {}).get("bline" if stack == "blineui" else stack)
             or {})
    return {n: e for n, e in (block.get("sources") or {}).items()
            if isinstance(e, dict) and "source" in e and e["source"] is None}


def unsourced_names(card: Optional[dict], fmt: Optional[str], stack: str = "blineui") -> set:
    """File names the card's format has no official source for ({"source": null})."""
    return set(unsourced(card, fmt, stack))


def record_files(rec: dict) -> set:
    """The bare file names a baseline record ran (settings.files.applied, else the cell's files)."""
    applied = ((rec.get("settings") or {}).get("files") or {}).get("applied")
    vals = applied if isinstance(applied, dict) else (((rec.get("cell") or {}).get("options") or {}).get("files") or {})
    out = set()
    for v in vals.values():
        for x in (v if isinstance(v, list) else [v]):
            if isinstance(x, str):
                out.add(x.replace("\\", "/").rsplit("/", 1)[-1])
    return out


def staged_of(rec: dict) -> dict:
    return rec.get("staged_files") or ((rec.get("cell") or {}).get("options") or {}).get("staged_files") or {}


def record_problems(rec: dict, card: Optional[dict], fmt: Optional[str]) -> list:
    """A baseline record that ran a file the card has no official source for, without a recorded, passing mirror
    verification for it: the pair is VOID (the file may not be the official weights)."""
    stack = (rec.get("arm") or {}).get("stack")
    if stack not in ("blineui", "sdcpp"):
        return []
    staged = staged_of(rec)
    entries = unsourced(card, fmt, stack)
    out = []
    for name in sorted(record_files(rec) & set(entries)):
        res = (staged.get(name) or {}).get("result") or {}
        if not res.get("verified"):
            # the card's own account rides along: why no official source exists and what would unblock the pair,
            # so a VOID line says more than "mirror rule" (MiniMax-H3's NVFP4 AWQ encoder, 3090 recheck Nov 20)
            unblock = entries[name].get("unblock")
            out.append(f"{name}: no official source and no recorded mirror verification against the official "
                       f"checkpoint (mirror rule)" + (f"; unblocks when: {unblock}" if unblock else ""))
    return out


def staged_lines(rec: dict) -> list:
    """'name from mirror repo/file (sha) = official repo, verified by METHOD: detail' per staged file."""
    out = []
    for name, d in sorted(staged_of(rec).items()):
        m, o, r = d.get("mirror") or {}, d.get("official") or {}, d.get("result") or {}
        how = (f"{r.get('equal')}/{r.get('compared')} tensors equal" if d.get("method") == "tensors"
               else r.get("detail") or "")
        out.append(f"{name} from mirror {m.get('repo')}/{m.get('file')} (sha256 {str(m.get('sha256') or '')[:12]}) = "
                   f"official {o.get('repo')} {o.get('files')}, verified by {d.get('method')}"
                   f"{': ' + how if how else ''}{'' if r.get('verified') else ' -- NOT VERIFIED'}")
    return out


# ---------------------------------------------------------------------------------------------- tensor check
def _pattern(name: str) -> re.Pattern:
    return re.compile("^" + re.escape(name).replace(re.escape("{i}"), r"(?P<i>\d+)") + "$")


def map_name(name: str, key_map: Optional[dict]) -> tuple:
    """(official names, op, dim) for a mirror tensor name: identity, a rename, or a "cat" of several."""
    for pat, target in (key_map or {}).items():
        m = _pattern(pat).match(name)
        if not m:
            continue
        sub = (lambda s: s.replace("{i}", m.group("i"))) if "i" in m.groupdict() else (lambda s: s)
        if isinstance(target, dict) and target.get("cat"):
            return [sub(x) for x in target["cat"]], "cat", int(target.get("dim", 0))
        return [sub(str(target))], "same", 0
    return [name], "same", 0


def verify_tensors(mirror_path, official_paths: list, key_map: Optional[dict] = None, cast: bool = False,
                   max_report: int = 8) -> dict:
    """{"verified", "compared", "equal", "mismatches"[:max_report], "unmapped", "official_unused", "seconds"}:
    every mirror tensor against its official tensor(s), exact (torch.equal; dtype and shape must match unless
    ``cast``, which casts the official tensor to the mirror's dtype first)."""
    import torch
    from safetensors import safe_open

    t0 = time.time()
    where: dict = {}
    handles = []
    for p in official_paths:
        h = safe_open(str(p), framework = "pt")
        handles.append(h)
        for k in h.keys():
            where[k] = h
    used: set = set()
    compared = equal = n_bf16 = 0
    bad, unmapped = [], []
    with safe_open(str(mirror_path), framework = "pt") as m:
        for name in m.keys():
            names, op, dim = map_name(name, key_map)
            if any(n not in where for n in names):
                unmapped.append(name)
                continue
            mt = m.get_tensor(name)
            parts = [where[n].get_tensor(n) for n in names]
            ot = parts[0] if op == "same" else torch.cat(parts, dim = dim)
            used.update(names)
            compared += 1
            if cast and ot.dtype != mt.dtype:
                ot = ot.to(mt.dtype)
            if ot.dtype == mt.dtype and ot.shape == mt.shape and torch.equal(ot, mt):
                equal += 1
            else:
                if ot.shape == mt.shape and ot.is_floating_point() and mt.is_floating_point():
                    # diagnosis only, never a pass: equal once both sides are rounded to bf16 (one publisher stored
                    # bf16-rounded values in a wider dtype), and the largest difference
                    n_bf16 += bool(torch.equal(ot.to(torch.bfloat16), mt.to(torch.bfloat16)))
                if len(bad) < max_report:
                    row = {"tensor": name, "official": names, "mirror": [str(mt.dtype), list(mt.shape)],
                           "official_t": [str(ot.dtype), list(ot.shape)]}
                    if ot.shape == mt.shape and ot.is_floating_point() and mt.is_floating_point():
                        row["bf16_equal"] = bool(torch.equal(ot.to(torch.bfloat16), mt.to(torch.bfloat16)))
                        row["max_abs"] = float((ot.double() - mt.double()).abs().max()) if ot.numel() else 0.0
                    bad.append(row)
            del mt, ot, parts
    unused = sorted(set(where) - used)
    ok = compared > 0 and equal == compared and not unmapped
    return {"verified": ok, "compared": compared, "equal": equal, "mismatches": bad, "n_bf16_equal_mismatches": n_bf16,
            "unmapped": unmapped[:max_report], "n_unmapped": len(unmapped), "official_unused": unused[:max_report],
            "n_official_unused": len(unused), "seconds": round(time.time() - t0, 1)}


def verify_entry(entry: dict, mirror_path, official_paths: Optional[list] = None) -> dict:
    """Run the entry's method on the VM (the mirror file already downloaded and sha256-checked). Raises
    RuntimeError when it fails, so the cells that need the file are BROKEN with the reason, never run."""
    probs = plan_problems(entry)
    if probs:
        raise RuntimeError(f"mirror file {entry.get('file')}: " + "; ".join(probs))
    ver = entry.get("verify") or {}
    if ver.get("method") == "sha256":
        return describe(entry)
    res = verify_tensors(mirror_path, official_paths or [], ver.get("key_map"), bool(ver.get("cast")))
    if not res["verified"]:
        raise RuntimeError(f"mirror file {entry.get('file')} from {entry.get('repo')} is not the official "
                           f"{(entry.get('official') or {}).get('repo')} checkpoint: {res['equal']}/{res['compared']} "
                           f"tensors equal, {res['n_unmapped']} unmapped, {res['n_bf16_equal_mismatches']} of the "
                           f"unequal ones equal after rounding both to bf16 ({(res['mismatches'] or res['unmapped'])[:2]})")
    return describe(entry, res)


def fetch_official(entry: dict, token: Optional[str] = None) -> list:
    """The official files of a tensors entry, through the HF cache (a Studio prefetch of the same repo is reused)."""
    from huggingface_hub import hf_hub_download

    import extnames  # noqa: PLC0415 - real repo / file names in a public copy

    off = entry.get("official") or {}
    return [hf_hub_download(extnames.ext(off["repo"]), extnames.ext(f), revision = off.get("revision"), token = token)
            for f in official_files(entry)]


def main() -> int:
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mirror")
    ap.add_argument("official", nargs = "+")
    ap.add_argument("--key-map", default = None, help = "JSON object")
    ap.add_argument("--cast", action = "store_true")
    a = ap.parse_args()
    res = verify_tensors(a.mirror, a.official, json.loads(a.key_map) if a.key_map else None, a.cast)
    print(json.dumps(res, indent = 1))
    return 0 if res["verified"] else 1


if __name__ == "__main__":
    sys.exit(main())
