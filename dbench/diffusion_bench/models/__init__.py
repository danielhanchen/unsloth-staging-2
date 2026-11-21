"""Model cards: one JSON per family, the single source of a benchmark's per-model settings.

``diffusion_bench/models/<family>.json`` (format: outputs/dbench_unify/CONTRACT.md). A card holds the baseline UI's
official template and the defaults it RUNS (steps, CFG, sampler, scheduler, shift, size, frames, fps, negative,
enhancer off), the expected denoiser passes, the file set per format and how it is built, the Studio load / generate
mapping, the known traps (each with a machine-checkable ``check`` name from TRAP_CHECKS, or null) and the audit
sources every value came from.

    from diffusion_bench.models import load_card, list_cards, validate_card   # or ``import models`` inside diffusion_bench
    card = load_card("qwen-image-2.1")
    expected_passes(card)                 # 25
    graph_params(card, files = {...})     # a bline_graphs.GraphParams at the card's defaults

BLINE_PIN.json = the baseline UI release tag + commit the cards' templates were read against.

Precision (for the parity gate): every ``formats.<fmt>.studio`` and ``formats.<fmt>.bline`` carries
``precision = {component: {"weights", "compute", "evidence"}}`` for transformer / text_encoder / vae (+ audio_vae).
``weights`` = how the file / load stores it (bf16, fp16, fp32, fp8_e4m3fn, fp8_scaled, int8_convrot, nvfp4,
nvfp4_awq, w4a8_gptq, w6a8, fp4_mixed, fp8mixed, gguf:<quant>, nf4); ``compute`` = the matmul / activation dtype on a
bf16-capable card (sm80+; fp8 compute needs sm89+, nvfp4 sm100+, T4 is the fp16_only_dtype trap), or null when it was
not traced (compare weights only then). Baseline-UI weights come from each file's safetensors header, compute from
its loader (unet_manual_cast over the model's supported_inference_dtypes, per-layer quant configs, VAE
working_dtypes); Studio's from its family tables. A template-default difference between the stacks is a trap with
``accepted_difference: true``, ``components`` and ``formats``: the gate reports it instead of voiding the pair.
``studio_load(card, fmt)`` gives the Studio load kwargs without the ``precision`` / ``sources`` blocks.

Sources (for a remote VM): ``formats.<fmt>.bline.sources`` maps every baseline-UI file name to
``{"repo", "file", "build", ...}`` (fetch ``file`` from the OFFICIAL vendor ``repo``, or ``fetch_repo`` = a
byte-identical mirror with ``sha256``; ``build`` = the converter step when the file is built, null when fetched as
is) or to ``{"source": null, "reason"}`` when no official route exists (never Bline-Org / blineanonymous).
``formats.<fmt>.studio.sources`` names the repo / file Studio loads.

Planning hints per format: ``min_vram_gib``, ``min_host_ram_gib``, ``est_s_per_render`` {T4, L4, A100-40, G4} from
measured rounds (``planning.cite``), null where never measured; ``planning.studio_refused`` lists cards Studio refuses.
"""

from __future__ import annotations

import ast
import json
import math
import operator
from pathlib import Path
from typing import Any, Optional

CARD_DIR = Path(__file__).resolve().parent
KINDS = ("image", "edit", "video")
DEFAULT_KEYS = ("steps", "cfg", "sampler", "scheduler", "shift", "width", "height", "frames", "fps", "negative",
                "enhancer")
REQUIRED = ("family", "kind", "bline", "defaults", "expected_passes", "formats", "studio", "traps", "sources")

# Machine-checkable trap names. The parity gate (parity.py) implements each; a card trap whose check is not here must
# have check = null and is a reviewer-read note only.
TRAP_CHECKS = {
    "passes_equal": "denoiser evaluations equal on both stacks (Studio hook counter vs the baseline's model-call "
                    "count), counted per sample row, so a CFG pair batched as one forward counts 2",
    "sigmas_equal": "sigma / timestep arrays dumped from both stacks equal to 1e-6",
    "settings_equal": "resolved steps / cfg / sampler / scheduler / shift / size / frames / fps / negative equal",
    "precision_equal": "DiT, text encoder and VAE precision resolved per stack and equal for a speed pair",
    "h3_steps_are_evaluations": "Studio's MiniMax-H3 diffusers path runs N evaluations for N steps "
                                "(UNSLOTH_H3_STEP_EVALUATIONS on; older trees ran N-1: 19 for 20)",
    "h3_qkv_regroup": "a baseline-UI H3 DiT built from the release has qkv_proj rows regrouped per head -> thirds "
                      "(models.convert.h3_qkv_regroup), or is a Bline-Org published file",
    "cfg_batch_rows": "evaluation counters count batch rows, not forward calls (Studio may batch cond + uncond)",
    "shared_cpu_noise": "a cross-stack quality score uses the same CPU fp32 initial noise on both stacks; "
                        "otherwise each stack is scored only against its own bf16",
    "png_compress_equal": "edit references and saved outputs use the same PNG bytes / compress level on both stacks "
                          "(Studio saved at level 1, the baseline UI's SaveImage at 4)",
    "new_reference_per_render": "every timed edit render uses a new reference image (a reused one hits Studio's "
                                "reference-latent cache)",
    "gguf_bline_refused": "the baseline UI cannot load this family's GGUF (BlineUI-GGUF has no architecture entry): "
                          "a GGUF cell compares Studio / sd.cpp only",
    "guidance_semantics": "the card's studio.gen.guidance maps to the baseline UI cfg per the card's note "
                          "(e.g. Z-Image g = cfg - 1; FLUX.1-dev embedded guidance is not CFG)",
    "fp16_only_dtype": "on an fp16-only card (T4, sm75) each stack's resolved compute dtype is recorded and equal "
                       "for a speed pair",
    "template_note_vs_sampler": "the template's note disagrees with its sampler; the card records what RUNS",
    "speed_tier_stated": "Studio's speed tier is recorded; the 'max' tier skips steps (static skip / FBCache)",
    "enhancer_off": "the template's prompt enhancer / upsampler is switched off on the baseline UI arm",
    "graph_not_template": "the registered graph is not the template's recipe (see the card's bline.reason)",
    "resolution_equal": "output width / height / frames equal (canvas floors and edit sizing resolved per stack)",
    "cache_defeated": "the baseline UI re-executed every timed request (nonce + execution count > 0)",
    "sampler_differs": "the stacks run a different sampler / schedule with equal denoiser evaluations: with "
                       "``accepted_difference: true``, ``settings`` (sampler / scheduler) and ``purposes`` "
                       "([\"timing\"]) the gate lists the difference on a timing pair instead of voiding it; a "
                       "quality pair stays strict",
    "sigma_rounding": "a published sigma list is rounded on one stack (same length, same work): with "
                      "``accepted_difference: true`` and ``sigma_tol`` (at most 1e-4) the gate accepts and lists a "
                      "sigma difference up to that tolerance",
}
# Settings a sampler_differs trap may accept (a scheduler difference also accepts a same-length sigma difference).
ACCEPTABLE_SETTINGS = ("sampler", "scheduler")


# models.TRAP_CHECKS name -> the parity gate's check name (parity.py TRAP_CHECKS keys: passes, sigmas, noise, size,
# precision, settings, cache). Names not listed here have no gate check and are reported as such.
GATE_CHECK = {"passes_equal": "passes", "h3_steps_are_evaluations": "passes", "cfg_batch_rows": "passes",
              "graph_not_template": "passes", "sigmas_equal": "sigmas", "shared_cpu_noise": "noise",
              "resolution_equal": "size", "precision_equal": "precision", "fp16_only_dtype": "precision",
              "settings_equal": "settings", "guidance_semantics": "settings", "template_note_vs_sampler": "settings",
              "enhancer_off": "settings", "cache_defeated": "cache", "sampler_differs": "settings", "sigma_rounding": "sigmas"}
COMPONENTS = ("transformer", "text_encoder", "vae", "audio_vae")


def _card_path(family: str) -> Path:
    return CARD_DIR / f"{family}.json"


def list_cards() -> list[str]:
    """Family names of every card, sorted."""
    return sorted(p.stem for p in CARD_DIR.glob("*.json") if p.name != "BLINE_PIN.json")


def _aliases() -> dict:
    out = {}
    for fam in list_cards():
        for a in json.loads(_card_path(fam).read_text()).get("aliases", []):
            out[a.lower()] = fam
    return out


def load_card(family: str) -> dict:
    """The card for ``family`` (its file stem or one of its ``aliases``). Raises KeyError for an unknown family."""
    fam = (family or "").strip().lower()
    if not _card_path(fam).exists():
        fam = _aliases().get(fam, fam)
    p = _card_path(fam)
    if not p.exists() or fam == "BLINE_PIN":
        raise KeyError(f"no model card for {family!r}; known: {list_cards()}")
    return json.loads(p.read_text())


def bline_pin() -> dict:
    """{"tag", "sha", ...}: the baseline UI release the cards were read against."""
    return json.loads((CARD_DIR / "BLINE_PIN.json").read_text())


def studio_load(card: dict, fmt: str) -> dict:
    """Studio load kwargs for a format (the card's ``formats.<fmt>.studio`` minus its ``precision`` block)."""
    return {k: v for k, v in card["formats"][fmt]["studio"].items() if k not in ("precision", "sources")}


def precision(card: dict, fmt: str, stack: str) -> Optional[dict]:
    """{component: {weights, compute, evidence}} of a format on ``studio`` or ``bline`` (None: no baseline-UI files)."""
    f = card["formats"][fmt]
    side = f.get("studio") if stack == "studio" else f.get("bline")
    return (side or {}).get("precision")


def accepted_differences(card: dict, fmt: str) -> list:
    """The card's accepted precision differences that apply to ``fmt``: [{components, note, id}]."""
    return [t for t in card.get("traps", []) if t.get("accepted_difference") and fmt in (t.get("formats") or [])]


def accepted_settings(card: dict, purpose: str) -> list:
    """The card's accepted sampler / scheduler differences for a pair of this purpose ("timing" / "quality"):
    [{id, settings, note}]; never any for a quality pair."""
    return [t for t in (card or {}).get("traps", []) if t.get("accepted_difference")
            and t.get("check") == "sampler_differs" and purpose in (t.get("purposes") or [])]


# ---------------------------------------------------------------- expected passes

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.FloorDiv: operator.floordiv,
        ast.Div: operator.truediv, ast.Mod: operator.mod, ast.Gt: operator.gt, ast.GtE: operator.ge,
        ast.Lt: operator.lt, ast.LtE: operator.le, ast.Eq: operator.eq, ast.NotEq: operator.ne,
        ast.USub: operator.neg, ast.Not: operator.not_}


def _eval(node, env):
    if isinstance(node, ast.Expression):
        return _eval(node.body, env)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, bool)):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise ValueError(f"unknown name {node.id!r} in expected_passes")
        return env[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left, env), _eval(node.right, env))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand, env))
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in _OPS:
        return _OPS[type(node.ops[0])](_eval(node.left, env), _eval(node.comparators[0], env))
    if isinstance(node, ast.BoolOp):
        vals = [_eval(v, env) for v in node.values]
        return all(vals) if isinstance(node.op, ast.And) else any(vals)
    if isinstance(node, ast.IfExp):
        return _eval(node.body, env) if _eval(node.test, env) else _eval(node.orelse, env)
    raise ValueError(f"unsupported expression in expected_passes: {ast.dump(node)[:80]}")


def expected_passes(card: dict, **override) -> Optional[int]:
    """Denoiser evaluations per render (per sample row: a CFG pair is 2 even when batched) at the card's defaults,
    with ``override`` (steps=, cfg=, ...) applied. None when the card does not state it."""
    expr = card.get("expected_passes")
    if expr is None:
        return None
    env = {k: v for k, v in (card.get("defaults") or {}).items() if isinstance(v, (int, float, bool)) or v is None}
    env.update({"cfg_batched": False, **(card.get("pass_vars") or {}), **override})
    env = {k: (0 if v is None else v) for k, v in env.items()}
    val = _eval(ast.parse(str(expr), mode = "eval"), env)
    return int(val)


# ---------------------------------------------------------------- validation

def validate_card(card: dict) -> list[str]:
    """Problems with a card (empty = valid). Offline: does not import bline_graphs (see models/test_cards.py)."""
    errs = []
    fam = card.get("family", "?")
    for k in REQUIRED:
        if k not in card:
            errs.append(f"{fam}: missing key {k!r}")
    if errs:
        return errs
    if card["kind"] not in KINDS:
        errs.append(f"{fam}: kind {card['kind']!r} not in {KINDS}")
    c = card["bline"]
    if c is None:
        if not card.get("bline_reason"):
            errs.append(f"{fam}: bline is null without a bline_reason")
    elif not isinstance(c, dict):
        errs.append(f"{fam}: bline must be an object or null")
    else:
        for k in ("graph", "template", "template_sha"):
            if k not in c:
                errs.append(f"{fam}: bline.{k} missing")
        if c.get("graph") is None and not c.get("reason"):
            errs.append(f"{fam}: bline.graph is null without bline.reason")
        if c.get("template_faithful") is False and not c.get("reason"):
            errs.append(f"{fam}: bline.template_faithful false without bline.reason")
    d = card["defaults"]
    if not isinstance(d, dict):
        errs.append(f"{fam}: defaults must be an object")
    else:
        for k in DEFAULT_KEYS:
            if k not in d:
                errs.append(f"{fam}: defaults.{k} missing (null when not applicable)")
        if not isinstance(d.get("steps"), int) or d.get("steps", 0) < 1:
            errs.append(f"{fam}: defaults.steps must be a positive int")
        if card["kind"] == "video" and not d.get("frames"):
            errs.append(f"{fam}: a video card needs defaults.frames")
        if d.get("enhancer") not in (False, None):
            errs.append(f"{fam}: defaults.enhancer must be false (benchmarks run the raw prompt)")
        for k in ("width", "height"):
            if d.get(k) is not None and (not isinstance(d[k], int) or d[k] % 8):
                errs.append(f"{fam}: defaults.{k} must be an int multiple of 8")
    try:
        n = expected_passes(card)
        if n is not None and n < 1:
            errs.append(f"{fam}: expected_passes evaluates to {n}")
    except Exception as e:  # noqa: BLE001
        errs.append(f"{fam}: expected_passes: {e}")
    if card["expected_passes"] is None and not card.get("expected_passes_note"):
        errs.append(f"{fam}: expected_passes is null without expected_passes_note")
    if not isinstance(card["formats"], dict) or not card["formats"]:
        errs.append(f"{fam}: formats must be a non-empty object")
    else:
        for name, f in card["formats"].items():
            if not isinstance(f, dict) or "studio" not in f or "bline" not in f or not f.get("source"):
                errs.append(f"{fam}: formats.{name} needs studio, bline and source")
                continue
            sides = [("studio", (f["studio"] or {}).get("precision"))]
            if (f.get("bline") or {}).get("files"):
                sides.append(("bline", f["bline"].get("precision")))
            if (f.get("bline") or {}).get("files"):
                srcs = f["bline"].get("sources") or {}
                names = []
                for v in f["bline"]["files"].values():
                    names += [v] if isinstance(v, str) else list(v)
                for n in names:
                    e = srcs.get(n)
                    if not isinstance(e, dict):
                        errs.append(f"{fam}: formats.{name}.bline.sources has no entry for {n}")
                    elif "source" in e and e["source"] is None:
                        if not e.get("reason"):
                            errs.append(f"{fam}: formats.{name}.bline.sources[{n}] is null without a reason")
                    elif not e.get("repo") or "build" not in e:
                        errs.append(f"{fam}: formats.{name}.bline.sources[{n}] needs repo, file, build")
                    elif str(e.get("repo", "")).lower().startswith(("bline-org/", "blineanonymous/")) or \
                            str(e.get("fetch_repo", "")).lower().startswith(("bline-org/", "blineanonymous/")):
                        errs.append(f"{fam}: formats.{name}.bline.sources[{n}] points at a Bline-Org repo")
            if not (f.get("studio") or {}).get("sources"):
                errs.append(f"{fam}: formats.{name}.studio.sources missing")
            est = f.get("est_s_per_render")
            if not isinstance(est, dict) or set(est) != {"T4", "L4", "A100-40", "G4"} or \
                    "min_vram_gib" not in f or "min_host_ram_gib" not in f:
                errs.append(f"{fam}: formats.{name} needs min_vram_gib, min_host_ram_gib, est_s_per_render{{T4, L4, "
                            "A100-40, G4}}")
            elif any(v is not None for v in est.values()) and not (f.get("planning") or {}).get("cite"):
                errs.append(f"{fam}: formats.{name} has estimates without planning.cite")
            for side, prec in sides:
                if not isinstance(prec, dict) or "transformer" not in prec:
                    errs.append(f"{fam}: formats.{name}.{side}.precision missing (needs transformer / text_encoder / vae)")
                    continue
                for comp, v in prec.items():
                    if comp not in COMPONENTS or not isinstance(v, dict) or not {"weights", "compute", "evidence"} <= set(v):
                        errs.append(f"{fam}: formats.{name}.{side}.precision.{comp} needs weights, compute, evidence")
    if not isinstance(card["studio"], dict) or "load" not in card["studio"] or "gen" not in card["studio"]:
        errs.append(f"{fam}: studio needs load and gen")
    ids = set()
    for t in card["traps"]:
        if not isinstance(t, dict) or not t.get("id") or "check" not in t or not t.get("note"):
            errs.append(f"{fam}: trap {t!r} needs id, check and note")
            continue
        if t["id"] in ids:
            errs.append(f"{fam}: duplicate trap id {t['id']}")
        ids.add(t["id"])
        if t["check"] is not None and t["check"] not in TRAP_CHECKS:
            errs.append(f"{fam}: trap {t['id']} check {t['check']!r} not in TRAP_CHECKS")
        if t.get("accepted_difference") and t.get("check") == "sampler_differs":
            bad = [k for k in t.get("settings") or [] if k not in ACCEPTABLE_SETTINGS]
            if not t.get("settings") or bad or t.get("purposes") != ["timing"]:
                errs.append(f"{fam}: sampler_differs trap {t['id']} needs settings from {ACCEPTABLE_SETTINGS} "
                            f"(bad {bad}) and purposes [\"timing\"] (quality pairs stay strict)")
        elif t.get("accepted_difference") and t.get("check") == "sigma_rounding":
            tol = t.get("sigma_tol")
            if not isinstance(tol, (int, float)) or isinstance(tol, bool) or not 0 < float(tol) <= 1e-4:
                errs.append(f"{fam}: sigma_rounding trap {t['id']} needs 0 < sigma_tol <= 1e-4 (got {tol!r})")
        elif t.get("accepted_difference"):
            bad = [c for c in t.get("components") or [] if c not in COMPONENTS]
            unknown = [x for x in t.get("formats") or [] if x not in (card.get("formats") or {})]
            if not t.get("components") or bad or not t.get("formats") or unknown:
                errs.append(f"{fam}: accepted_difference trap {t['id']} needs known components and formats "
                            f"(bad {bad}, unknown formats {unknown})")
    if not card["sources"] or not all(isinstance(s, str) and s for s in card["sources"]):
        errs.append(f"{fam}: sources must list at least one audit file")
    try:
        json.dumps(card, allow_nan = False)
    except ValueError as e:
        errs.append(f"{fam}: not strictly finite JSON: {e}")
    for v in _numbers(card):
        if isinstance(v, float) and not math.isfinite(v):
            errs.append(f"{fam}: non-finite number")
            break
    return errs


def _numbers(x: Any):
    if isinstance(x, dict):
        for v in x.values():
            yield from _numbers(v)
    elif isinstance(x, list):
        for v in x:
            yield from _numbers(v)
    elif isinstance(x, (int, float)):
        yield x


# ---------------------------------------------------------------- graph parameters

def graph_params(card: dict, files: Optional[dict] = None, prompt: str = "a red fox in snow", seed: int = 0,
                 **override):
    """A bline_graphs.GraphParams at the card's defaults (override any field). ``files`` defaults to the card's
    template file names (``bline.files``)."""
    try:
        from .. import bline_graphs as G  # package import (diffusion_bench.models)
    except (ImportError, ValueError):
        import bline_graphs as G  # run from inside diffusion_bench
    c = card.get("bline") or {}
    d = card["defaults"]
    # bline.shift_in_graph false: the shift is the model's own sampling setting in the baseline UI (no node in the
    # template); passing it would add a ModelSampling* node the template does not have.
    shift = d.get("shift") if c.get("shift_in_graph", True) else None
    kw = dict(prompt = prompt, seed = seed, steps = d["steps"], width = d.get("width") or 1024,
              height = d.get("height") or 1024, negative = d.get("negative"), cfg = d.get("cfg"),
              sampler = d.get("sampler"), scheduler = d.get("scheduler"), shift = shift,
              guidance = d.get("guidance"), frames = d.get("frames"), fps = d.get("fps"),
              files = dict(files if files is not None else c.get("files") or {}),
              extra = dict(c.get("graph_extra") or {}))
    kw.update(override)
    return G.GraphParams(**kw)
