#!/usr/bin/env python3
"""Build a run plan from models x arms x targets: per-VM ABBA blocks, A/A pairs, trial cells, lanes, estimates.

  python diffusion_bench/plan.py REQUEST.json --out PLAN.json [--dry-run] [--cards-dir DIR]
  python diffusion_bench/plan.py REQUEST.json --dry-run          # list only, writes nothing

REQUEST.json:
  {
    "name": "validation_r1",
    "revisions": {"unsloth_sha": "<40 hex>", "bline_ref": "v0.45.4"},     # required for remote targets
    "models": ["qwen-image-2.1", {"family": "flux3-klein", "formats": ["int8"], "n": 3},
               {"family": "wan2.2-5b", "frames": 41}],                     # frames: a short clip on every arm
    "arms": [{"stack": "studio", "format": "int8", "tier": "default"},
             {"stack": "blineui", "format": "int8"},
             {"stack": "studio", "format": "int8", "budget_gib": 12}],     # budget arms pair with same-budget arms
    "targets": ["colab:L4", {"name": "kaggle:T4x2", "lane_mode": "same_stack"},
                {"name": "local:B200", "purpose": "quality"}, "amd:gfx1151-linux", "mac:macos-15", "cpu"],
    "repeats": 3,               # timed runs per arm, at least 3 (rounded up to whole ABBA blocks)
    "n": 5, "warmup": 1,        # renders per run (each a new prompt + seed), warm-ups outside the timed set
    "short_n": 1,               # same-prompt full / short-step render pairs per run (steady s/image, derived s/step)
    "negative_control": true,   # + one Studio parity probe with a changed step count per comparison: the gate must
                                # VOID it against the baseline probe (proves the gate can fail on real records)
    "account": "michael"        # Colab account (optional; launcher arg otherwise)
  }

Scheduling rules (each one a past mistake):
  - every arm of a comparison (one family x budget on one target) runs on the SAME VM and the same lane (GPU);
    a pair across VMs is never valid, and a pair is valid only with clocks within 3% (the reporter checks);
  - arms run in mirrored blocks: S C C S for two stacks (A B C C B A for three), ceil(repeats / 2) blocks, a fresh
    process per run; inside every block each stack's two runs are its A/A pair (arm.aa_of names the first);
  - Kaggle T4x2: two lanes (one T4 + half the CPUs each). lane_mode "same_stack" (default) runs the two lanes in
    lockstep ONLY when their queues are mirrored (the same stack / format / arm / family sequence: every run's
    neighbour is a run of the same kind), else the VM falls back to serial (one cell at a time), so concurrency
    never differs between the arms of a comparison; "serial" never runs two cells at once. Each run records its
    neighbour (options.target.concurrent_with). A VM's packed work is capped as if serial;
  - a trial cell (n=1, untimed) first on every lane of every VM: if one fails the VM driver stops before any timed
    cell;
  - local:B200 is parity / quality only: a timing arm there is refused;
  - a format the target cannot run (fp8 on T4 / A100, nvfp4 off G4 / B200) is NOT_RUN with the reason. The dense
    "bf16" / "fp16" formats on an fp16-only card (T4, sm75) run at fp16 compute on BOTH stacks (the card's
    formats.<fmt>.planning.on_fp16_only, default "fp16"): options.dense_dtype = "fp16" on every arm, recorded and
    checked by the parity gate, never refused;
  - card hints per target: ``min_vram_gib`` refuses only a format never measured on the target's GPU (a measured
    est_s_per_render means it ran there); ``min_host_ram_gib`` is always held against THIS target's host RAM
    (Colab T4 12.7 GB is not Kaggle T4x2 29 GB);
  - a timed comparison left with one stack (every baseline arm blocked) is NOT_RUN, not timed alone (an arm with
    "solo": true is timed anyway).

Output: PLAN.json, and cases.json beside it (report.py's expected case set: every planned run but the trial
cells, with comparison, arm, block, repeat, prompt ids and seeds). PLAN.json = {"cells": [...run_cell cells with options.target / options.arm...], "vms": [...slots...],
"comparisons", "not_run", "estimate", "revisions"}. matrix.py --plan PLAN.json --vm ID runs one VM's cells in
order (single lane); targets/vm_driver.py runs them on a rented VM (lanes, trial gate, artifact).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mirror  # noqa: E402
from targets import slug, target_info  # noqa: E402

STACK_ORDER = ("studio", "blineui", "sdcpp", "diffusers")
STACK_SHORT = {"studio": "s", "blineui": "c", "sdcpp": "sd", "diffusers": "d"}
PURPOSES = ("timing", "quality", "parity")
# Format -> compute dtype it needs from the card (bf16 weights on a T4 run as fp16 on both stacks: no bf16 there).
FORMAT_DTYPE = {"bf16": "bf16", "fp16": "fp16", "int8": None, "fp8": "fp8", "nvfp4": "nvfp4", "gguf": None}
DENSE_FORMATS = ("bf16", "fp16")
MIN_COMPUTE_CAP = {"fp8": "8.9", "nvfp4": "10.0"}  # NVIDIA capability a format's kernels need (Ada / Blackwell)


def dense_dtype(card: dict, fmt: str, tinfo: dict) -> str | None:
    """The compute dtype a dense format runs at on this target when it differs from the format name: "fp16" for
    "bf16" on an fp16-only card (both stacks run the bf16 files at fp16 there), per the card's
    formats.<fmt>.planning.on_fp16_only (default "fp16"; null = the card refuses it). None = as named."""
    if fmt not in DENSE_FORMATS or "bf16" in (tinfo.get("dtypes") or []) or "fp16" not in (tinfo.get("dtypes") or []):
        return None
    plan = (((card.get("formats") or {}).get(fmt) or {}).get("planning") or {})
    return plan.get("on_fp16_only", "fp16")
# Rough render-time scale against an L4 (estimate only; the trial cell's timing is the real number).
SPEED_VS_L4 = {"colab:T4": 2.6, "colab:L4": 1.0, "colab:A100-40": 0.45, "colab:A100-80": 0.42, "colab:G4": 0.3,
               "kaggle:T4x2": 2.6, "local:B200": 0.15, "amd:gfx1151-linux": 1.6, "amd:gfx1151-windows": 1.8,
               "mac:macos-15": 8.0, "cpu": 60.0}


# ------------------------------------------------------------------------------------------------ cards
def load_cards(families: list, cards_dir: str | None) -> dict:
    """family -> card. --cards-dir reads <dir>/<family>.json directly; otherwise diffusion_bench.models (agent A's
    module) is imported lazily, with a clear error when it is not merged yet."""
    out = {}
    if cards_dir:
        for fam in families:
            p = Path(cards_dir) / f"{fam}.json"
            if not p.exists():
                raise SystemExit(f"no model card {p}")
            out[fam] = json.loads(p.read_text())
        return out
    try:
        from models import load_card  # noqa: PLC0415 - optional until the model cards are merged
    except ImportError as exc:
        raise SystemExit(f"diffusion_bench/models is not importable ({exc}); merge the model cards or pass "
                         "--cards-dir DIR with <family>.json files") from exc
    for fam in families:
        out[fam] = load_card(fam)
    return out


def _files(fmt_entry: dict, stack: str) -> dict:
    block = fmt_entry.get("bline" if stack == "blineui" else stack) or {}
    return dict(block.get("files") or {})


# Card blocks that describe a side but are not load kwargs (models.studio_load strips the same two).
CARD_ONLY_KEYS = ("precision", "sources")
# Target -> the card's est_s_per_render key (cards carry T4 / L4 / A100-40 / G4 measurements only).
EST_KEY = {"colab:T4": "T4", "kaggle:T4x2": "T4", "colab:L4": "L4", "colab:A100-40": "A100-40",
           "colab:A100-80": "A100-40", "colab:G4": "G4"}


def est_key(tname: str) -> str | None:
    return EST_KEY.get(tname)


def file_sources(card: dict, fmt: str, stack: str = "blineui") -> tuple:
    """({file name: download / build entry}, [(file name, reason)]) for a format's baseline-UI (or sd.cpp) files from
    the card's ``formats.<fmt>.bline.sources``: a fetched file -> {"repo": fetch_repo or repo, "file", "sha256",
    "official_repo"}; a built file -> the whole entry (its "build" names the builders/ step); a file with no
    official route ({"source": null, "reason"}) -> listed in the second value."""
    fentry = ((card.get("formats") or {}).get(fmt) or {})
    block = fentry.get("bline" if stack == "blineui" else stack) or {}
    srcs = block.get("sources") or {}
    out, missing = {}, []
    for name in _flat_names(block.get("files") or {}):
        e = srcs.get(name)
        if not isinstance(e, dict):
            continue  # a card without sources: the plan's / request's downloads map, or a pre-staged file
        if "source" in e and e["source"] is None:
            missing.append((name, e.get("reason") or "no official route"))
        elif e.get("build"):
            out[name] = {**e, "official_repo": e.get("repo")}
        elif e.get("repo") and e.get("file"):
            out[name] = {"repo": e.get("fetch_repo") or e["repo"], "file": e["file"], "official_repo": e["repo"],
                         **({"sha256": e["sha256"]} if e.get("sha256") else {})}
    return out, missing


def studio_prefetch(card: dict, fmt: str, tname: str | None = None) -> dict:
    """{repo: {"repo", "file"}} Studio loads for a format (``formats.<fmt>.studio.sources``): fetched into the VM's
    HF cache before the cells, since a Studio cell runs with local_files_only (a timed load never downloads).
    An entry's ``dense_on`` (GPU keys) drops its ``ignore_patterns`` on those GPUs: Studio reads the dense weights
    there (a T4 builds FLUX.3 klein's INT8 from the dense transformer, and the offline load failed on the missing
    shards, val_nov21 T4 trial); the key then carries ":dense" so another target's entry is not overwritten."""
    srcs = (((card.get("formats") or {}).get(fmt) or {}).get("studio") or {}).get("sources") or {}
    gpu = est_key(tname) if tname else None
    out = {}
    for key, e in srcs.items():
        if not isinstance(e, dict) or not e.get("repo") or str(e.get("repo")).startswith("<"):
            continue
        f = e.get("file")
        ent = {"repo": e["repo"], "file": None if (not f or str(f).startswith("<")) else f}
        dense = bool(gpu and gpu in (e.get("dense_on") or []))
        if not ent["file"] and e.get("ignore_patterns") and not dense:
            # weights the format never loads (the dense transformer an int8 format replaces with Studio's hosted
            # file): a whole Qwen-Image-2.1 snapshot was 28 files, tens of GB, for an arm that read none of them
            ent["ignore_patterns"] = list(e["ignore_patterns"])
        out[f"{e['repo']}:{f or '*'}" + (":dense" if dense and e.get("ignore_patterns") else "")] = ent
    return out


def _flat_names(files: dict) -> list:
    out = []
    for v in files.values():
        out += [v] if isinstance(v, str) else [x for x in (v or []) if isinstance(x, str)]
    return out


def card_to_cell(card: dict, arm: dict, n: int, warmup: int) -> dict:
    """One run_cell cell for (card, arm) from the card's effective defaults: the same settings on every stack, the
    stack-specific load / file mapping from the card. Raises KeyError when the card has no such format / stack."""
    d = card.get("defaults") or {}
    fmt = arm["format"]
    fentry = (card.get("formats") or {})[fmt]
    stack = arm["stack"]
    kind = card.get("kind", "image")
    cell = {"backend": stack, "kind": kind, "width": d.get("width"), "height": d.get("height"),
            "steps": d.get("steps"), "guidance": d.get("cfg"), "negative_prompt": d.get("negative") or None,
            "prompts": arm.get("prompts") or ("video_3" if kind == "video" else "image_24"), "n": n, "warmup": warmup,
            "options": {}, "env": dict(arm.get("env") or {})}
    if kind == "video":
        cell.update({"frames": d.get("frames"), "fps": d.get("fps")})
    bline = card.get("bline") or {}
    sched = {k: d.get(k) for k in ("sampler", "scheduler", "shift") if d.get(k) is not None}
    if bline.get("shift_in_graph") is False:
        sched.pop("shift", None)  # the shift is the model's own setting there: no ModelSampling node in the template
    if stack == "studio":
        st = card.get("studio") or {}
        load = dict(st.get("load") or {})
        if not fentry.get("studio"):
            raise KeyError(f"card {card.get('family')}: format {fmt} has no studio mapping")
        fst = {k: v for k, v in fentry["studio"].items() if k not in CARD_ONLY_KEYS}
        model = (fst.pop("model_path", None) or fst.pop("model", None) or load.pop("model_path", None)
                 or load.pop("model", None) or card.get("repo"))
        load.pop("model_path", None)
        load.pop("model", None)
        if not model:
            raise KeyError(f"card {card.get('family')}: no studio model_path")
        cell["model"] = model
        # card studio.gen: the request fields run_cell already passes (steps, guidance, size, frames, fps, negative)
        # become cell fields (Studio's own mapping wins for its cells); only the rest goes through options.generate,
        # so generate() never gets a keyword twice. Studio picks its own sampler / scheduler per family: the parity
        # gate compares what each stack resolved.
        gen = dict(st.get("gen") or {})
        for k_card, k_cell in (("steps", "steps"), ("guidance", "guidance"), ("width", "width"), ("height", "height"),
                               ("num_frames", "frames"), ("fps", "fps"), ("negative_prompt", "negative_prompt")):
            if k_card in gen:
                v = gen.pop(k_card)
                if v is not None and (kind == "video" or k_cell not in ("frames", "fps")):
                    cell[k_cell] = v
        cell["options"] = {**load, **fst, "generate": gen}
        if st.get("default_sampler"):  # what Studio's resident recipe runs when the record says "default"
            cell["options"]["studio_default_sampler"] = st["default_sampler"]
        if st.get("scheduler_map"):  # the baseline's names for this family's resident scheduler classes
            cell["options"]["studio_scheduler_map"] = st["scheduler_map"]
        if arm.get("tier"):
            cell["options"]["speed_mode"] = arm["tier"]
    elif stack in ("blineui", "sdcpp"):
        if not fentry.get("bline" if stack == "blineui" else stack):
            raise KeyError(f"card {card.get('family')}: format {fmt} has no {stack} files")
        cell["options"] = {"family": (card.get(stack) or {}).get("graph") or bline.get("graph") or card["family"],
                           "files": _files(fentry, stack), **sched}
        wd = fentry.get("bline_weight_dtype") or (fentry.get("bline") or {}).get("weight_dtype")
        if stack == "blineui" and wd:
            cell["options"]["weight_dtype"] = wd
        if d.get("guidance") is not None:
            cell["options"]["flux_guidance"] = d["guidance"]  # embedded guidance (FLUX.1-dev), not CFG
        if bline.get("graph_extra"):
            cell["options"]["graph_extra"] = dict(bline["graph_extra"])
        if stack == "blineui":
            cell["options"]["cache_bust"] = "sampler"
            if bline.get("template"):
                cell["options"]["template"] = bline["template"]
            if arm.get("bline_args"):
                cell["options"]["bline_args"] = arm["bline_args"]
    elif stack == "diffusers":
        cell["model"] = (card.get("diffusers") or {}).get("model") or card.get("repo")
        if not cell["model"]:
            raise KeyError(f"card {card.get('family')}: no diffusers model")
    else:
        raise KeyError(f"unknown stack {stack}")
    arm_opts = dict(arm.get("options") or {})
    if isinstance(arm_opts.get("generate"), dict) and isinstance(cell["options"].get("generate"), dict):
        # merged, not replaced: the card's generate keys stay unless the arm sets them
        arm_opts["generate"] = {**cell["options"]["generate"], **arm_opts["generate"]}
    cell["options"].update(arm_opts)
    gen = cell["options"].get("generate")
    if isinstance(gen, dict):
        # request fields run_cell passes itself become cell fields, so generate() never gets a keyword twice
        for k_card, k_cell in (("steps", "steps"), ("guidance", "guidance"), ("width", "width"), ("height", "height"),
                               ("num_frames", "frames"), ("fps", "fps"), ("negative_prompt", "negative_prompt")):
            if k_card in gen:
                v = gen.pop(k_card)
                if v is not None and (kind == "video" or k_cell not in ("frames", "fps")):
                    cell[k_cell] = v
    need_cc = MIN_COMPUTE_CAP.get(FORMAT_DTYPE.get(fmt) or "")
    if need_cc and not cell.get("min_compute_cap"):
        # run_cell / matrix gate on it: an fp8 / nvfp4 cell on a target that only names a label (tester:<label>)
        # is NOT_RUN unsupported on an older card instead of a crash or a silent cast
        cell["min_compute_cap"] = need_cc
    if arm.get("backend"):  # e.g. "fake" (CPU tests) or "studio_http": the backend that measures the stack
        cell["backend"] = arm["backend"]
    cell["variant"] = fmt  # parity / report group = (card family, format, target)
    cell["card"] = card.get("family")
    # matrix.py's venv (local / tester runs); the VM driver uses the VM's own Studio / BlineUI venvs instead
    cell["venv"] = arm.get("venv") or ({"profile": "studio"} if stack == "studio" else "bench")
    return {k: v for k, v in cell.items() if v is not None}


def format_blocker(card: dict, fmt: str, tinfo: dict, arm: dict | None = None, staged: dict | None = None) -> str | None:
    """Why this target cannot run this format (for this arm), else None. Uses the card's planning hints:
    ``min_vram_gib`` against the target's VRAM (or the arm's budget), ``min_host_ram_gib`` against its host RAM
    when the format was never measured on this GPU (a measured est_s_per_render means it ran there),
    ``planning.studio_refused`` for Studio arms, and bline_graphs.GGUF_REFUSED for baseline-UI GGUF arms."""
    arm = arm or {}
    need = FORMAT_DTYPE.get(fmt)
    fentry = (card.get("formats") or {}).get(fmt)
    if fentry is None:
        return f"card has no format {fmt}"
    gen = ((card.get("studio") or {}).get("gen") or {})
    if card.get("kind") == "edit" or any(isinstance(v, str) and v.startswith("<") for v in gen.values()):
        # "<image>" / "<reference width>" placeholders: no reference image is staged on either stack yet, and
        # Studio's generate() has no "reference" keyword
        return "edit benchmark: reference-image staging is not implemented on both stacks (card placeholders)"
    stack = arm.get("stack")
    if stack == "studio" and not fentry.get("studio"):
        return f"card format {fmt} has no Studio mapping"
    if stack == "diffusers" and not ((card.get("diffusers") or {}).get("model") or card.get("repo")):
        return "card has no diffusers model (diffusers.model / repo)"
    if stack in ("blineui", "sdcpp"):
        if stack == "blineui" and card.get("bline") is None:
            return f"no baseline-UI graph for {card.get('family')} ({card.get('bline_reason') or 'bline: null'})"
        if not ((fentry.get("bline" if stack == "blineui" else stack) or {}).get("files")):
            return f"card format {fmt} has no {'baseline-UI' if stack == 'blineui' else stack} files"
        _, missing = file_sources(card, fmt, stack)
        # a file with no official source may come from a mirror only when the request's entry verifies it against
        # the official checkpoint it repackages (mirror.py: a published hash, or tensor by tensor on the VM); an
        # unverified mirror entry (legacy staged_from_mirror, no verify block, a hash that does not match) is refused
        # (a staged entry for a file the card has no official source for IS a mirror copy, flagged or not, unless it
        # is a builder step from official weights)
        unverified = [(n, mirror.plan_problems((staged or {})[n], force = True)) for n, _ in missing
                      if isinstance((staged or {}).get(n), dict) and not (staged or {})[n].get("build")]
        unverified = [(n, p) for n, p in unverified if p]
        if unverified:
            return ("mirror-staged baseline file not verified against the official checkpoint: "
                    + "; ".join(f"{n} ({'; '.join(p)})" for n, p in unverified))
        missing = [(n, r) for n, r in missing if n not in (staged or {})]
        if missing:
            return ("no official source for " + "; ".join(f"{n} ({r})" for n, r in missing)
                    + " (a request \"downloads\" entry can stage a mirror copy verified against the official "
                    "checkpoint: mirror.py)")
    if need and need not in tinfo["dtypes"]:
        if not (fmt in DENSE_FORMATS and dense_dtype(card, fmt, tinfo)):
            return f"{tinfo['name']} has no {need} support ({'/'.join(tinfo['dtypes'])})"
    key = est_key(tinfo["name"])
    measured = key is not None and ((fentry.get("est_s_per_render") or {}).get(key) is not None)
    vram = arm.get("budget_gib") or tinfo.get("vram_gb")
    min_vram = fentry.get("min_vram_gib")
    # a measured time on this GPU means the format ran there (Studio offloads below the hint): not refused for VRAM;
    # a simulated budget below the hint is never waived (the hint is the smallest budget that ran)
    if vram and min_vram and float(vram) < float(min_vram) and (arm.get("budget_gib") or not measured):
        return (f"{fmt} needs {min_vram} GiB VRAM (card min_vram_gib), {tinfo['name']} "
                f"{'budget' if arm.get('budget_gib') else 'has'} {vram}")
    # disk: the card's peak (Studio's fetched files + the baseline's files and builds) against the VM's free disk
    disk, min_disk = tinfo.get("disk_gb"), fentry.get("min_disk_gib")
    if disk and min_disk and float(disk) < float(min_disk):
        return f"{fmt} needs {min_disk} GiB of disk (card min_disk_gib), {tinfo['name']} has about {disk}"
    # host RAM is the target's own (Colab T4 12.7 GB vs Kaggle T4x2 29 GB share the "T4" GPU key): never waived
    ram, min_ram = tinfo.get("ram_gb"), fentry.get("min_host_ram_gib")
    if ram and min_ram and float(ram) < float(min_ram):
        return f"{fmt} needs {min_ram} GiB host RAM (card min_host_ram_gib), {tinfo['name']} has {ram}"
    if arm.get("stack") == "studio":
        for entry in (fentry.get("planning") or {}).get("studio_refused") or []:
            head = str(entry).split("(")[0]
            gpus = [g.strip() for g in head.split("/")]
            hit = (key and any(g == key or (g.startswith("A100") and key.startswith("A100")) for g in gpus)) or \
                (arm.get("budget_gib") and any("offloaded budget" in g for g in gpus))
            if hit:
                return f"Studio refuses {fmt} here (card planning.studio_refused: {entry})"
    if arm.get("stack") == "blineui" and fmt.startswith("gguf"):
        graph = ((card.get("bline") or {}).get("graph") or "")
        try:
            import bline_graphs as G  # noqa: PLC0415

            refused = graph and G.canonical(graph) in G.GGUF_REFUSED
        except Exception:  # noqa: BLE001
            refused = False
        if refused:
            return f"baseline UI cannot load {graph} GGUF (BlineUI-GGUF has no architecture entry; trap gguf_bline_refused)"
    if tinfo["kind"] == "mac" and fmt not in ("gguf", "bf16", "fp16"):
        return f"{fmt} has no MPS path"
    return None


# ------------------------------------------------------------------------------------------------ blocks
def arm_label(arm: dict) -> str:
    parts = [STACK_SHORT.get(arm["stack"], arm["stack"]), arm["format"]]
    if arm.get("tier") and arm["tier"] != "default":
        parts.append(arm["tier"])
    if arm.get("label_suffix"):
        parts.append(arm["label_suffix"])
    if arm.get("budget_gib"):
        parts.append(f"b{arm['budget_gib']:g}")
    return "_".join(str(p) for p in parts)


def mirrored_order(arms: list) -> list:
    """A B ... B A over the arm list (S C C S for two)."""
    return list(arms) + list(reversed(arms))


def comparison_sequence(cmp_id: str, arms: list, repeats: int, once: bool = False) -> list:
    """[(arm, block, pos, aa_of_tag)] in run order: ceil(repeats/2) mirrored blocks, so every arm gets
    >= repeats runs. Within each block an arm's second run is the A/A partner of its first."""
    if once:  # quality / parity: one run per arm, same prompts and seeds, nothing timed
        blocks, order = 1, list(arms)
    elif len(arms) > 1:
        blocks, order = max(2, math.ceil(max(3, repeats) / 2)), mirrored_order(arms)
    else:  # a single arm (quality-only targets): one block of `repeats` back-to-back runs
        blocks, order = 1, list(arms) * max(3, repeats)
    seq = []
    for b in range(1, blocks + 1):
        first = {}
        for pos, arm in enumerate(order):
            tag = f"{cmp_id}__{arm_label(arm)}__k{b}p{pos}"
            aa = first.get(arm_label(arm))
            first.setdefault(arm_label(arm), tag)
            seq.append((arm, b, pos, tag, aa))
    return seq


# ------------------------------------------------------------------------------------------------ estimate
def render_s_estimate(card: dict, fmt: str | None, tname: str) -> tuple:
    """(s per render, basis) from the card's ``formats.<fmt>.est_s_per_render``: the target's own GPU when measured,
    else the L4 figure x a per-target factor, else a 12 s image / 150 s video default x the factor."""
    est = (((card.get("formats") or {}).get(fmt) or {}).get("est_s_per_render") or {}) if fmt else {}
    key = est_key(tname)
    if key and est.get(key) is not None:
        return float(est[key]), f"card {fmt} {key}"
    if est.get("L4") is not None:
        return float(est["L4"]) * SPEED_VS_L4.get(tname, 1.0), f"card {fmt} L4 x {SPEED_VS_L4.get(tname, 1.0)}"
    video = card.get("kind") == "video"
    return (150.0 if video else 12.0) * SPEED_VS_L4.get(tname, 1.0), "default"


def load_s_estimate(card: dict, fmt: str | None, tname: str, stack: str | None) -> tuple:
    """(s of per-run overhead beyond the steady renders, basis) from the card's ``formats.<fmt>.est_load_s``
    ({GPU key: {stack: s}}, measured by card_timings.py from real records: load + the first render's excess over a
    steady one, i.e. compile and warm-up): the target's GPU and the cell's stack, else that stack on any measured GPU
    (load is mostly disk and host work), else a 120 s image / 240 s video default."""
    est = (((card.get("formats") or {}).get(fmt) or {}).get("est_load_s") or {}) if fmt else {}
    key = est_key(tname)
    st = "blineui" if stack in ("blineui", "bline") else stack
    if key and isinstance(est.get(key), dict) and est[key].get(st) is not None:
        return float(est[key][st]), f"card {fmt} {key} {st}"
    other = [float(v[st]) for k, v in sorted(est.items()) if isinstance(v, dict) and v.get(st) is not None]
    if other:
        return max(other), f"card {fmt} {st} (largest measured GPU)"
    return (240.0 if card.get("kind") == "video" else 120.0), "default"


def est_cell_min(card: dict, tname: str, cell: dict) -> float:
    render_s, _ = render_s_estimate(card, cell.get("variant"), tname)
    d_frames = (card.get("defaults") or {}).get("frames")
    if card.get("kind") == "video" and d_frames and cell.get("frames") and int(cell["frames"]) != int(d_frames):
        render_s *= float(cell["frames"]) / float(d_frames)  # a request "frames" override: work ~ frames
    stack = ((cell.get("options") or {}).get("arm") or {}).get("stack") or cell.get("backend")
    load_s, _ = load_s_estimate(card, cell.get("variant"), tname, stack)
    renders = cell.get("warmup", 1) + cell.get("n", 5) + 2 * cell.get("short_n", 0)
    return round((load_s + renders * render_s) / 60.0, 2)


# ------------------------------------------------------------------------------------------------ plan
def norm_target(t) -> dict:
    t = {"name": t} if isinstance(t, str) else dict(t)
    info = target_info(t["name"])
    t.setdefault("purpose", "timing" if info["timing"] else "quality")
    if t["purpose"] not in PURPOSES:
        raise SystemExit(f"target {t['name']}: purpose {t['purpose']!r} not in {PURPOSES}")
    if t["purpose"] == "timing" and not info["timing"]:
        raise SystemExit(f"target {t['name']} refuses timing arms (shared local GPU: parity / quality only); "
                         "set \"purpose\": \"quality\" or \"parity\" for it")
    t.setdefault("lane_mode", "same_stack" if info["lanes"] > 1 else "single")
    if t["lane_mode"] not in ("single", "same_stack", "serial"):
        raise SystemExit(f"target {t['name']}: lane_mode {t['lane_mode']!r}")
    return {**info, **t}


def norm_model(m) -> dict:
    return {"family": m} if isinstance(m, str) else dict(m)


def build_plan(req: dict, cards: dict) -> dict:
    repeats = int(req.get("repeats", 3))
    if repeats < 3:
        raise SystemExit("repeats must be >= 3 (the A/A floor needs three runs per arm)")
    n, warmup = int(req.get("n", 5)), int(req.get("warmup", 1))
    targets = [norm_target(t) for t in req["targets"]]
    names = [t["name"] for t in targets]
    dup = sorted({x for x in names if names.count(x) > 1})
    if dup:
        raise SystemExit(f"target(s) {dup} listed twice: comparison ids and vm_ids are per target name "
                         "(list a target once; repeats / lanes set how often it runs)")
    models = [norm_model(m) for m in req["models"]]
    arms_all = [dict(a) for a in req["arms"]]
    for a in arms_all:
        if a.get("stack") not in STACK_ORDER:
            raise SystemExit(f"arm {a}: stack must be one of {STACK_ORDER}")
        a.setdefault("format", "bf16")
        a.setdefault("tier", "default" if a["stack"] == "studio" else None)
    cells, comparisons, not_run = [], [], []
    downloads: dict = {}
    prefetch: dict = {}
    per_target_cmps: dict = {}
    for t in targets:
        for m in models:
            card = cards[m["family"]]
            fam = m["family"]
            if t.get("families") and fam not in t["families"]:
                continue
            want_fmts = set(m.get("formats") or [a["format"] for a in arms_all])
            arms = [a for a in arms_all if a["format"] in want_fmts and (not a.get("families") or fam in a["families"])]
            if t["kind"] == "mac":
                arms = [a for a in arms if a["stack"] in ("studio", "blineui", "sdcpp")]
            budgets = sorted({a.get("budget_gib") or 0 for a in arms})
            for bud in budgets:
                group = [a for a in arms if (a.get("budget_gib") or 0) == bud]
                ok = []
                for a in group:
                    why = format_blocker(card, a["format"], t, a, req.get("downloads"))
                    if not why and bud and t.get("vram_gb") and bud >= t["vram_gb"]:
                        why = f"budget {bud} GiB is not below {t['name']}'s {t['vram_gb']} GB"
                    if not why and t["kind"] == "local" and (a.get("purpose") or t["purpose"]) == "timing":
                        why = "local:B200 refuses timing arms"
                    if why:
                        not_run.append({"family": fam, "target": t["name"], "arm": arm_label(a), "reason": why})
                    else:
                        ok.append(a)
                ok.sort(key = lambda a: (STACK_ORDER.index(a["stack"]), arm_label(a)))
                if t["purpose"] == "timing" and len({a["stack"] for a in ok}) > 1:
                    # a timed arm with no other stack on its format has nothing to be compared with: it spends VM
                    # time on a number no ratio uses (an arm with "solo": true is timed anyway)
                    pair_fmts = {f for f in {a["format"] for a in ok}
                                 if len({a["stack"] for a in ok if a["format"] == f}) > 1}
                    for a in [a for a in ok if a["format"] not in pair_fmts and not a.get("solo")]:
                        not_run.append({"family": fam, "target": t["name"], "arm": arm_label(a),
                                        "reason": f"no other stack runs {a['format']} here: no pair to time "
                                                  "(\"solo\": true on the arm times it alone)"})
                        ok.remove(a)
                if not ok:
                    continue
                stacks = {a["stack"] for a in ok}
                if len(stacks) < 2 and len({a["stack"] for a in group}) > 1:
                    not_run.append({"family": fam, "target": t["name"], "arm": "comparison",
                                    "reason": f"only {sorted(stacks)} can run here: no cross-stack pair"
                                              + (" (not timed)" if t["purpose"] == "timing" else "")})
                    if t["purpose"] == "timing":
                        # every baseline arm is blocked: three back-to-back runs of one stack are a number no ratio
                        # uses (an arm with "solo": true is timed anyway)
                        ok = [a for a in ok if a.get("solo")]
                        if not ok:
                            continue
                frames_ovr = m.get("frames")
                if frames_ovr is not None and card.get("kind") != "video":
                    raise SystemExit(f"model {fam}: \"frames\" overrides a video card's clip length only")
                cmp_id = (f"{slug(fam)}_{slug(t['name'])}" + (f"_b{bud:g}" if bud else "")
                          + (f"_f{int(frames_ovr)}" if frames_ovr is not None else ""))
                seq = comparison_sequence(cmp_id, ok, repeats, once = t["purpose"] != "timing")
                cmp = {"id": cmp_id, "family": fam, "target": t["name"], "budget_gib": bud or None,
                       "purpose": t["purpose"], "arms": [arm_label(a) for a in ok], "runs": {}, "order": [], "aa_pairs": [],
                       "valid_if": "same vm_id and lane; SM clocks within 3% between the paired runs",
                       "est_min": 0.0}
                pending, failed = [], None
                for arm, block, pos, tag, aa in seq:
                    try:
                        cell = card_to_cell(card, arm, int(m.get("n", n)), warmup)
                        # same-prompt long / short pairs (steady s/image, derived s/step): one pair per run by
                        # default; the polled per-step time is the primary s/step on both stacks, and run_cell's
                        # own default (3 pairs) would add 6 renders to every run that the estimate never counted
                        cell["short_n"] = int(m.get("short_n", req.get("short_n", 1)))
                    except KeyError as exc:
                        failed = (arm_label(arm), str(exc))
                        break
                    if frames_ovr is not None:
                        # a short clip on BOTH stacks (same frames on every arm; a separate comparison id, never
                        # pooled with the template length); the parity gate still checks resolved == rendered
                        cell["frames"] = int(frames_ovr)
                        cell.setdefault("options", {})["frames_override"] = {
                            "frames": int(frames_ovr), "template": (card.get("defaults") or {}).get("frames")}
                    cell["tag"] = tag
                    cell["family"] = fam
                    cell["options"]["target"] = {"name": t["name"], "vm_id": None, "lane": None,
                                                 "account": req.get("account"),
                                                 "simulated_budget_gib": arm.get("budget_gib")}
                    cell["options"]["arm"] = {"stack": arm["stack"], "block": block, "order": "ABBA",
                                              "aa_of": aa, "label": arm_label(arm), "comparison": cmp_id,
                                              "purpose": arm.get("purpose") or t["purpose"], "format": arm["format"],
                                              "tier": arm.get("tier")}
                    dd = dense_dtype(card, arm["format"], t)
                    if dd:  # e.g. bf16 files on a T4: both stacks compute in fp16 (recorded; the gate checks it)
                        cell["options"]["dense_dtype"] = dd
                    if arm.get("budget_gib"):
                        cell["budget_gib"] = arm["budget_gib"]
                    cell["est_min"] = est_cell_min(card, t["name"], cell)
                    pending.append(cell)
                    cmp["runs"].setdefault(arm_label(arm), []).append(tag)
                    cmp["order"].append(tag)
                    cmp["est_min"] += cell["est_min"]
                    if aa:
                        cmp["aa_pairs"].append([aa, tag])
                if not failed:
                    for a in ok:  # baseline-UI files -> download / build entries; Studio repos -> prefetch
                        if a["stack"] in ("blineui", "sdcpp"):
                            downloads.update(file_sources(card, a["format"], a["stack"])[0])
                        elif a["stack"] == "studio":
                            pf = studio_prefetch(card, a["format"], t["name"])
                            prefetch.update(pf)
                            # which comparison needs which prefetch entry: each VM's export keeps only its own
                            # (common.plan_carry), never another VM's family snapshot
                            cmp.setdefault("prefetch", [])
                            cmp["prefetch"] += [k for k in pf if k not in cmp["prefetch"]]
                if failed:
                    not_run.append({"family": fam, "target": t["name"], "arm": failed[0],
                                    "reason": f"card mapping: {failed[1]} (whole comparison dropped)"})
                    continue
                if t["purpose"] == "parity":
                    for c in pending:
                        c["options"]["parity_probe"] = True
                elif req.get("parity", True):
                    pending += parity_cells(cmp, card, ok, pending, fam, t, req, n, warmup)
                cmp["parity_pairs"] = parity_pairs(cmp, ok, pending, t, req, fam)
                cells += pending
                cmp["est_min"] = round(cmp["est_min"], 1)
                comparisons.append(cmp)
                per_target_cmps.setdefault(t["name"], []).append(cmp)
    by_tag = {c["tag"]: c for c in cells}
    vms = []
    for t in targets:
        got, dropped = schedule_target(t, per_target_cmps.get(t["name"], []), by_tag, req)
        vms += got
        for cmp, why in dropped:
            comparisons.remove(cmp)
            gone = set(cmp["order"]) | set(cmp.get("parity_order") or [])
            cells = [c for c in cells if c["tag"] not in gone]
            not_run.append({"family": cmp["family"], "target": t["name"], "arm": "comparison", "reason": why})
    for c in list(by_tag.values()):
        if c["tag"].startswith("trial__"):
            cells.append(c)
    for card in cards.values():  # legacy per-card maps, then the request's own (staged / overridden files) win
        for src in (card.get("downloads"), (card.get("bline") or {}).get("downloads")):
            downloads.update(src or {})
    downloads.update(req.get("downloads") or {})
    plan = {"plan_version": 1, "name": req.get("name", "plan"), "downloads": downloads, "prefetch": prefetch,
            "created": time.strftime("%FT%TZ", time.gmtime()),
            "revisions": dict(req.get("revisions") or {}), "request": req, "cells": cells, "vms": vms,
            "comparisons": comparisons, "not_run": not_run}
    plan["estimate"] = estimate(plan)
    check_plan(plan)
    return plan


def parity_cells(cmp: dict, card: dict, arms: list, pending: list, fam: str, t: dict, req: dict, n: int,
                 warmup: int) -> list:
    """One untimed parity probe (run_cell --parity: sigmas, denoiser evaluations, noise, resolved settings,
    precision) per arm of every format two or more stacks run; they run after the VM's trial and before the
    comparison's timed cells, and parity.gate_plan turns a VOID group's timed cells into NOT_RUN."""
    by_fmt: dict = {}
    for a in arms:
        by_fmt.setdefault(a["format"], []).append(a)
    out = []
    first = {c["options"]["arm"]["label"]: c for c in pending}
    for fmt, fa in by_fmt.items():
        if len({a["stack"] for a in fa}) < 2:
            continue
        for a in fa:
            base = first.get(arm_label(a))
            if base is None:
                continue
            c = copy.deepcopy(base)
            c.update({"tag": f"{cmp['id']}__{arm_label(a)}__parity", "n": 1, "warmup": 1, "short_n": 0})
            c["options"]["parity_probe"] = True
            if a["stack"] == "studio":
                # untimed, so this load may download: Studio fetches what IT resolves for the format (its hosted
                # prequant denoiser / text encoder files, which the card's sources do not list) into the VM's HF
                # cache, and every timed Studio cell after it loads offline (local_files_only) from that cache
                c["options"]["local_files_only"] = False
            c["options"]["arm"] = {**c["options"]["arm"], "order": "parity", "aa_of": None, "block": 0,
                                   "purpose": "parity"}
            c["est_min"] = round(base["est_min"] * 0.5, 2)
            cmp.setdefault("parity_order", []).append(c["tag"])
            cmp["est_min"] = round(cmp["est_min"] + c["est_min"], 2)
            out.append(c)
            if req.get("negative_control") and a["stack"] == "studio" and not any(
                    x["options"].get("negative_control") for x in out):
                # the gate must be able to FAIL on real records: the same Studio probe with one step fewer (more
                # when steps <= 2) is paired with the baseline probe and must come back VOID (parity.gate_plan)
                neg = copy.deepcopy(c)
                steps = int(neg.get("steps") or 2)
                neg["steps"] = steps - 1 if steps > 2 else steps + 1
                neg["tag"] = f"{cmp['id']}__{arm_label(a)}__parity_negctl"
                neg["options"]["negative_control"] = f"steps {steps} -> {neg['steps']}"
                cmp["parity_order"].append(neg["tag"])
                cmp["est_min"] = round(cmp["est_min"] + neg["est_min"], 2)
                out.append(neg)
    return out


def parity_pairs(cmp: dict, arms: list, cells: list, t: dict, req: dict, fam: str) -> list:
    """[{format, a: studio probe tag, b: other probe tag, quality, waive}] for parity.gate_plan."""
    tag_of, neg_of = {}, {}
    for c in cells:
        arm = c["options"]["arm"]
        if c["options"].get("negative_control"):
            neg_of.setdefault(arm["label"], c)
        elif c["options"].get("parity_probe"):
            tag_of.setdefault(arm["label"], c["tag"])
    waive = list((req.get("parity_waive") or {}).get(fam) or [])
    out = []
    for s in [a for a in arms if a["stack"] == "studio"]:
        for o in [a for a in arms if a["stack"] != "studio" and a["format"] == s["format"]]:
            ta, tb = tag_of.get(arm_label(s)), tag_of.get(arm_label(o))
            if ta and tb:
                out.append({"format": s["format"], "a": ta, "b": tb, "quality": t["purpose"] == "quality",
                            "waive": waive})
            neg = neg_of.get(arm_label(s))
            if neg is not None and tb and not any(x.get("expect") == "VOID" for x in out):
                out.append({"format": s["format"], "a": neg["tag"], "b": tb, "quality": t["purpose"] == "quality",
                            "waive": waive, "expect": "VOID", "negative_control": neg["options"]["negative_control"]})
    return out


def schedule_target(t: dict, cmps: list, by_tag: dict, req: dict) -> list:
    """Pack comparisons into VMs (first fit under max_vm_min) and lanes, then build each VM's slots."""
    dropped: list = []
    if not cmps:
        return [], dropped
    lanes = t["lanes"] if t["lane_mode"] != "single" else 1
    cap = float(t.get("max_vm_min") or 1e9) - float(t.get("setup_min") or 0)
    vms: list = []
    for cmp in sorted(cmps, key = lambda c: -c["est_min"]):
        if cmp["est_min"] > cap:
            dropped.append((cmp, f"needs ~{cmp['est_min']:.0f} min, over one {t['name']} VM's {cap:.0f} min: "
                                 "lower n / repeats or split the family's arms"))
            continue
        placed = False
        for vm in vms:
            loads = [sum(c["est_min"] for c in vm["_lanes"][i]) for i in range(lanes)]
            i = min(range(lanes), key = lambda k: loads[k])
            # same_stack may fall back to serial (lanes not mirrored) and serial runs one cell at a time: the VM's
            # wall is then the SUM of its lanes, so that is what the cap holds
            wall = sum(loads) if t["lane_mode"] in ("serial", "same_stack") else max(loads)
            if wall < cap and (sum(loads) if t["lane_mode"] in ("serial", "same_stack") else loads[i]) \
                    + cmp["est_min"] <= cap:
                vm["_lanes"][i].append(cmp)
                placed = True
                break
        if not placed:
            vms.append({"_lanes": [[cmp]] + [[] for _ in range(lanes - 1)]})
    out = []
    for k, vm in enumerate(vms):
        vm_id = f"{slug(t['name'])}-{k + 1}"
        queues = []
        for lane, lane_cmps in enumerate(vm["_lanes"]):
            q = []
            for cmp in lane_cmps:
                cmp["vm_id"], cmp["lane"] = vm_id, lane
                for tag in cmp.get("parity_order") or []:
                    c = by_tag[tag]
                    c["options"]["target"].update({"vm_id": vm_id, "lane": lane if lanes > 1 else None})
                    q.append({"tag": tag, "lane": lane, "stack": c["options"]["arm"]["stack"], "timed": False, "parity": True,
                              "comparison": cmp["id"], "family": cmp["family"], "est_min": c["est_min"]})
                for tag in cmp["order"]:
                    c = by_tag[tag]
                    c["options"]["target"].update({"vm_id": vm_id, "lane": lane if lanes > 1 else None})
                    q.append({"tag": tag, "lane": lane, "stack": c["options"]["arm"]["stack"],
                              "timed": t["purpose"] == "timing", "family": cmp["family"],
                              "comparison": cmp["id"], "est_min": c["est_min"]})
            queues.append(q)
        slots, effective = merge_lanes(queues, t["lane_mode"], req.get("max_parallel_studio"), with_mode = True)
        for slot in slots:  # each run records its neighbour: report / reviewers see who shared the host
            for it in slot:
                by_tag[it["tag"]]["options"]["target"]["concurrent_with"] = \
                    [o["tag"] for o in slot if o is not it] or None
        # one trial per lane (each lane's own GPU / CPU slice): a lane-1 fault surfaces before any timed cell
        trials = []
        for lane, q in enumerate(queues):
            if not q:
                continue
            first = q[0]
            if t["kind"] in ("colab", "kaggle") and first.get("parity"):
                # the lane's first item is already an untimed one-render load of the same stack and model (a parity
                # probe): it IS the trial (the VM driver stops on it the same way), saving a whole Studio cold load
                # + compile per VM (162-585 s per VM in the validation round). Other targets keep a trial__ cell
                # (the AMD / Mac / tester runners key on that tag)
                first["trial"] = True
                by_tag[first["tag"]]["options"]["arm"]["trial"] = True
                continue
            ttag = f"trial__{vm_id}" + (f"__l{lane}" if lane else "")
            trial = copy.deepcopy(by_tag[first["tag"]])
            trial.update({"tag": ttag, "n": 1, "warmup": 0, "short_n": 0, "est_min": 0.0})
            trial["options"].pop("parity_probe", None)
            trial["options"]["arm"] = {**trial["options"]["arm"], "order": "trial", "aa_of": None, "block": 0,
                                       "purpose": "trial"}
            trial["options"]["target"].update({"lane": lane if lanes > 1 else None, "concurrent_with": None})
            trial["est_min"] = round(first["est_min"] * 0.5, 2)
            by_tag[ttag] = trial
            trials.append([{"tag": ttag, "lane": lane, "stack": trial["options"]["arm"]["stack"], "timed": False,
                            "comparison": None, "est_min": trial["est_min"], "trial": True}])
        slots = trials + slots
        wall = sum(max(it["est_min"] for it in s) for s in slots) + float(t.get("setup_min") or 0)
        out.append({"vm_id": vm_id, "target": t["name"], "kind": t["kind"], "gpu": t["gpu"], "lanes": lanes,
                    "lane_mode": t["lane_mode"], "lane_mode_effective": effective, "purpose": t["purpose"],
                    "account": req.get("account"),
                    "comparisons": [c["id"] for lc in vm["_lanes"] for c in lc], "slots": slots,
                    "est_min": round(wall, 1),
                    "est_cu": round(wall / 60.0 * float(t.get("cu_per_h") or 0), 2)})
    return out, dropped


def lane_signature(q: list) -> list:
    """What a lane runs, slot by slot, for the mirrored-lanes test: (stack, family, arm label, timed, parity)."""
    return [(it["stack"], it.get("family"), (it["tag"].split("__")[1] if "__" in it["tag"] else it["tag"]),
             bool(it.get("timed")), bool(it.get("parity"))) for it in q]


def merge_lanes(queues: list, mode: str, max_parallel_studio=None, with_mode: bool = False):
    """Slots (lists of items run at the same time). One lane: one item per slot. serial: one item per slot,
    alternating lanes. same_stack: the lanes run in lockstep only when their queues are MIRRORED (the same stack /
    arm / timed-or-parity sequence, so every run's neighbour is a run of the same kind on the other GPU and the
    arms of a comparison all see the same concurrency); anything else falls back to serial. A Studio cap
    (max_parallel_studio) below the lane count also means serial. With ``with_mode`` returns (slots, mode run)."""
    queues = [list(q) for q in queues if q]
    effective = mode
    if len(queues) <= 1 or mode in ("single",):
        slots = [[it] for q in queues for it in q]
        effective = "single"
    elif mode == "same_stack" and len({len(q) for q in queues}) == 1 and \
            all(lane_signature(q) == lane_signature(queues[0]) for q in queues[1:]) and \
            not (max_parallel_studio is not None and any(it["stack"] == "studio" for it in queues[0])
                 and len(queues) > int(max_parallel_studio)):
        slots = [list(items) for items in zip(*queues)]
        effective = "same_stack (mirrored lanes, lockstep)"
    else:
        slots, i = [], 0
        while any(queues):
            q = queues[i % len(queues)]
            if q:
                slots.append([q.pop(0)])
            i += 1
        effective = "serial" if mode == "serial" else "serial (same_stack fallback: lanes not mirrored)"
    return (slots, effective) if with_mode else slots


def estimate(plan: dict) -> dict:
    per = {}
    for vm in plan["vms"]:
        p = per.setdefault(vm["target"], {"vms": 0, "wall_min": 0.0, "cu": 0.0, "timed_runs": 0, "runs": 0})
        p["vms"] += 1
        p["wall_min"] = round(p["wall_min"] + vm["est_min"], 1)
        p["cu"] = round(p["cu"] + vm["est_cu"], 2)
        p["timed_runs"] += sum(1 for s in vm["slots"] for it in s if it.get("timed"))
        p["runs"] += sum(1 for s in vm["slots"] for it in s if not it.get("trial"))
    return {"per_target": per, "total_cu": round(sum(p["cu"] for p in per.values()), 2),
            "total_vm_min": round(sum(p["wall_min"] for p in per.values()), 1),
            "note": "rough: render time from the card's formats.<fmt>.est_s_per_render for the target's GPU (else its L4 "
                    "figure x a per-target factor, else 12 s image / 150 s video); CU at cloud_pool.TIERS rates; the "
                    "trial cell's timing is the real number"}


def check_plan(plan: dict) -> None:
    """Assert the scheduling rules on the built plan (a bug here must never reach a VM)."""
    tags = [c["tag"] for c in plan["cells"]]
    dup = {t for t in tags if tags.count(t) > 1}
    assert not dup, f"duplicate tags {dup}"
    cell = {c["tag"]: c for c in plan["cells"]}
    vm_ids = [vm["vm_id"] for vm in plan["vms"]]
    assert len(set(vm_ids)) == len(vm_ids), f"duplicate vm_ids {sorted({v for v in vm_ids if vm_ids.count(v) > 1})}"
    for vm in plan["vms"]:
        assert vm["slots"] and vm["slots"][0][0].get("trial"), f"{vm['vm_id']}: trial cell is not first"
        for slot in vm["slots"]:
            timed = [it for it in slot if it.get("timed")]
            assert len({it["stack"] for it in timed}) <= 1, f"{vm['vm_id']}: mixed stacks side by side {slot}"
            assert len({it["lane"] for it in slot}) == len(slot), f"{vm['vm_id']}: two items on one lane {slot}"
            if vm["lane_mode"] == "serial" or str(vm.get("lane_mode_effective", "")).startswith("serial"):
                assert len(slot) <= 1, f"{vm['vm_id']}: two cells at once in a serial VM {slot}"
            if len(slot) > 1:
                assert len({(it["stack"], it.get("family"), it["tag"].split("__")[1] if "__" in it["tag"] else "")
                            for it in slot}) == 1, f"{vm['vm_id']}: side-by-side runs of different kinds {slot}"
    for cmp in plan["comparisons"]:
        vms = {cell[t]["options"]["target"]["vm_id"] for tl in cmp["runs"].values() for t in tl}
        lanes = {cell[t]["options"]["target"]["lane"] for tl in cmp["runs"].values() for t in tl}
        assert len(vms) == 1 and len(lanes) == 1, f"{cmp['id']} split across VMs / lanes {vms} {lanes}"
        if cmp["purpose"] == "timing":
            for arm, tl in cmp["runs"].items():
                assert len(tl) >= 3, f"{cmp['id']} {arm}: {len(tl)} runs < 3"
            assert target_info(cmp["target"])["timing"], f"{cmp['id']}: timing on {cmp['target']}"
    json.dumps(plan, allow_nan = False)


def prompt_rows(cell: dict) -> list:
    """[{id, seed}] run_cell will render as kind "new" (common.load_prompts: the first n rows, or cell ids)."""
    try:
        import common as C  # noqa: PLC0415

        rows = C.load_prompts(cell.get("prompts") or "image_24", cell.get("n"), cell.get("ids"))
        return [{"id": r.get("id"), "seed": r.get("seed")} for r in rows]
    except Exception:  # noqa: BLE001 - an unreadable prompt set leaves the seeds to the record, not a crash
        return []


def cases_doc(plan: dict) -> dict:
    """report.py's expected case set (its expected_cases reads {"cases": [{"tag": ...}]}): every planned run but the
    untimed trial cells, with its comparison, arm, repeat, prompts and seeds; plus each comparison's arms and
    repeats so completeness can be judged per comparison."""
    cell = {c["tag"]: c for c in plan["cells"]}
    cases = []
    for cmp in plan["comparisons"]:
        for tag in cmp["order"]:
            c = cell[tag]
            arm = c["options"]["arm"]
            rows = prompt_rows(c)
            cases.append({"tag": tag, "comparison": cmp["id"], "family": cmp["family"], "target": cmp["target"],
                          "vm_id": c["options"]["target"]["vm_id"], "lane": c["options"]["target"]["lane"],
                          "stack": arm["stack"], "arm": arm.get("label"), "block": arm["block"],
                          "repeat": cmp["runs"][arm["label"]].index(tag) + 1, "aa_of": arm.get("aa_of"),
                          "purpose": arm.get("purpose"), "prompts": c.get("prompts"), "n": c.get("n"),
                          "warmup": c.get("warmup"), "prompt_ids": [r["id"] for r in rows],
                          "seeds": [r["seed"] for r in rows],
                          "simulated_budget_gib": c["options"]["target"].get("simulated_budget_gib")})
    comps = [{"id": c["id"], "family": c["family"], "target": c["target"], "vm_id": c.get("vm_id"),
              "purpose": c["purpose"], "arms": c["arms"], "repeats": {a: len(t) for a, t in c["runs"].items()},
              "budget_gib": c["budget_gib"]} for c in plan["comparisons"]]
    return {"plan": plan["name"], "created": plan["created"], "cases": cases, "comparisons": comps,
            "not_run": plan["not_run"]}


def listing(plan: dict) -> str:
    lines = [f"plan {plan['name']}: {len(plan['cells'])} cells, {len(plan['vms'])} VMs, "
             f"{len(plan['comparisons'])} comparisons, {len(plan['not_run'])} NOT_RUN"]
    for vm in plan["vms"]:
        lines.append(f"\n[{vm['vm_id']}] {vm['target']} lanes={vm['lanes']} ({vm['lane_mode']}) purpose={vm['purpose']} "
                     f"~{vm['est_min']:.0f} min ~{vm['est_cu']:.1f} CU")
        for i, slot in enumerate(vm["slots"]):
            lines.append(f"  {i:3d} " + "  ||  ".join(
                f"L{it['lane']} {'TRIAL ' if it.get('trial') else ''}{'PARITY ' if it.get('parity') else ''}{it['tag']} ({it['stack']}, ~{it['est_min']:.0f}m)"
                for it in slot))
    if plan["not_run"]:
        lines.append("\nNOT_RUN:")
        for r in plan["not_run"]:
            lines.append(f"  {r['family']} @ {r['target']} {r['arm']}: {r['reason']}")
    e = plan["estimate"]
    lines.append(f"\nestimate: {e['total_vm_min']:.0f} VM-min, {e['total_cu']:.1f} CU ({e['note']})")
    for t, p in e["per_target"].items():
        lines.append(f"  {t}: {p['vms']} VM, {p['runs']} runs ({p['timed_runs']} timed), ~{p['wall_min']:.0f} min, ~{p['cu']:.1f} CU")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("request")
    ap.add_argument("--out", help = "PLAN.json to write")
    ap.add_argument("--dry-run", action = "store_true", help = "print the listing; with --out still writes")
    ap.add_argument("--cards-dir", help = "read <family>.json model cards from here instead of diffusion_bench.models")
    a = ap.parse_args(argv)
    req = json.loads(Path(a.request).read_text())
    cards = load_cards(sorted({norm_model(m)["family"] for m in req["models"]}), a.cards_dir)
    plan = build_plan(req, cards)
    if a.dry_run or not a.out:
        print(listing(plan))
    if a.out:
        Path(a.out).parent.mkdir(parents = True, exist_ok = True)
        Path(a.out).write_text(json.dumps(plan, indent = 1, allow_nan = False))
        cases = Path(a.out).parent / "cases.json"
        cases.write_text(json.dumps(cases_doc(plan), indent = 1, allow_nan = False))
        print(f"wrote {a.out} and {cases} (report.py's expected case set)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
