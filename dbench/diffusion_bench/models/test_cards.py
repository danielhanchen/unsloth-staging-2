"""CPU / offline checks for the model cards and the baseline-UI graph registry.

    python scripts/diffusion_bench/models/test_cards.py      (or: python -m pytest scripts/diffusion_bench/models)

1. every card validates (models.validate_card) and BLINE_PIN.json names a tag + 40-hex sha;
2. every card's bline.graph is a registered bline_graphs family, and the card's template matches the module's
   TEMPLATE family of files (same template, or listed in the card's reason);
3. every registered graph builds an API prompt from its card's defaults (graphs without a card: from the module's
   DEFAULTS and a placeholder file set), every node is {class_type, inputs}, every link points at an existing node,
   the nonce reaches the sampler and never a loader, and the card's file names are what the loaders read;
4. the H3 qkv regroup round-trips and moves the per-head rows to thirds;
5. specs: every blineui cell's family is registered.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DB = HERE.parent
if str(DB) not in sys.path:
    sys.path.insert(0, str(DB))

import bline_graphs as G  # noqa: E402
import models as M  # noqa: E402

# Registered graphs with no card of their own: build them from module DEFAULTS with these placeholder files.
NO_CARD_FILES = {
    "o22-qwen": {"dit": "d.safetensors", "te": "t.safetensors", "vae": "v.safetensors"},
    "o22-h3": {"dit": "d.safetensors", "te": "t.safetensors", "vae": "v.safetensors", "audio_vae": "a.safetensors"},
    "wan2.1-t2v": {"dit": "d.safetensors", "te": "t.safetensors", "vae": "v.safetensors"},
    "o22-qwen-edit": {"dit": "d.safetensors", "te": "t.safetensors", "vae": "v.safetensors"},
    "ltx-2.3-dev": {"checkpoint": "c.safetensors", "te": "t.safetensors"},
}
NO_CARD_SIZE = {"o22-h3": dict(width = 864, height = 480, frames = 124),
                "wan2.1-t2v": dict(width = 832, height = 480, frames = 33),
                "ltx-2.3-dev": dict(width = 768, height = 512, frames = 121, cfg = 4.0)}
LOADER_TYPES = {"UNETLoader", "UnetLoaderGGUF", "CLIPLoader", "DualCLIPLoader", "VAELoader", "CheckpointLoaderSimple",
                "LTXAVTextEncoderLoader", "LatentUpscaleModelLoader", "DiffusersLoader", "LTXVAudioVAELoader",
                "LoraLoaderModelOnly", "LoraLoader", "QuadrupleCLIPLoader"}


def _check_graph(fam: str, g: dict, files: dict) -> list[str]:
    errs = []
    if G.SAVE not in g:
        errs.append(f"{fam}: no '{G.SAVE}' node")
    for nid, node in g.items():
        if set(node) - {"class_type", "inputs", "_meta"} or "class_type" not in node or "inputs" not in node:
            errs.append(f"{fam}: node {nid} malformed: {sorted(node)}")
            continue
        for k, v in node["inputs"].items():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str) and isinstance(v[1], int):
                if v[0] not in g:
                    errs.append(f"{fam}: node {nid}.{k} links to missing node {v[0]!r}")
        if node["class_type"] in LOADER_TYPES and "_bench_nonce" in node["inputs"]:
            errs.append(f"{fam}: loader {nid} got a nonce (would reload weights every render)")
    samplers = [n for n in g.values() if n["class_type"] in ("KSampler", "SamplerCustom", "SamplerCustomAdvanced", "KSamplerAdvanced")]
    if not samplers:
        errs.append(f"{fam}: no sampler node")
    elif not any("_bench_nonce" in n["inputs"] for n in samplers):
        errs.append(f"{fam}: nonce did not reach a sampler")
    try:
        json.dumps(g, allow_nan = False)
    except (TypeError, ValueError) as e:
        errs.append(f"{fam}: graph is not strict JSON: {e}")
    # every file the card names is read by some loader input
    names = set()
    for v in files.values():
        names.update([v] if isinstance(v, str) else list(v))
    read = {str(x) for n in g.values() for x in n["inputs"].values() if isinstance(x, str)}
    missing = sorted(n for n in names if n not in read)
    if missing:
        errs.append(f"{fam}: card files not read by any node: {missing}")
    return errs


def test_cards_validate():
    errs = []
    cards = M.list_cards()
    assert cards, "no model cards"
    for fam in cards:
        c = M.load_card(fam)
        if c["family"] != fam:
            errs.append(f"{fam}.json has family {c['family']!r}")
        errs += M.validate_card(c)
    pin = M.bline_pin()
    if not pin.get("tag") or not re.fullmatch(r"[0-9a-f]{40}", pin.get("sha", "")):
        errs.append(f"BLINE_PIN.json needs tag + 40-hex sha: {pin}")
    assert not errs, "\n".join(errs)


def test_card_graphs_registered():
    errs = []
    for fam in M.list_cards():
        c = M.load_card(fam)
        if c["bline"] is None or c["bline"].get("graph") is None:
            continue
        try:
            G.canonical(c["bline"]["graph"])
        except KeyError as e:
            errs.append(f"{fam}: {e}")
    assert not errs, "\n".join(errs)


def test_graphs_build_from_card_defaults():
    errs = []
    carded = set()
    for fam in M.list_cards():
        c = M.load_card(fam)
        if not c["bline"] or not c["bline"].get("graph"):
            continue
        gfam = G.canonical(c["bline"]["graph"])
        carded.add(gfam)
        p = M.graph_params(c, nonce = 7)
        try:
            g = G.build(gfam, p)
        except Exception as e:  # noqa: BLE001
            errs.append(f"{fam} ({gfam}): build failed: {type(e).__name__}: {e}")
            continue
        errs += _check_graph(f"{fam} ({gfam})", g, c["bline"]["files"])
        # the defaults reach the sampler: steps everywhere a steps input exists
        steps = {n["inputs"]["steps"] for n in g.values() if "steps" in n["inputs"]
                 and n["class_type"] in ("KSampler", "BasicScheduler", "Flux2Scheduler", "LTXVScheduler")}
        if steps and c["defaults"]["steps"] not in steps:
            errs.append(f"{fam}: card steps {c['defaults']['steps']} not in graph steps {steps}")
    for gfam in G.FAMILIES:
        if gfam in carded:
            continue
        if gfam not in NO_CARD_FILES:
            errs.append(f"registered graph {gfam!r} has no card and no NO_CARD_FILES entry")
            continue
        mod = G.module(gfam)
        p = G.GraphParams(prompt = "x", seed = 0, steps = int(mod.DEFAULTS.get("steps", 4)), files = NO_CARD_FILES[gfam],
                          nonce = 7, **NO_CARD_SIZE.get(gfam, {}))
        try:
            errs += _check_graph(gfam, G.build(gfam, p), NO_CARD_FILES[gfam])
        except Exception as e:  # noqa: BLE001
            errs.append(f"{gfam}: build failed: {type(e).__name__}: {e}")
    assert not errs, "\n".join(errs)


def test_graph_guards():
    h3 = {"dit": "h3-Q4_K.gguf", "te": "t", "vae": "v", "audio_vae": "a"}
    try:
        G.build("minimax-h3", G.GraphParams(prompt = "x", seed = 0, steps = 2, files = h3))
        raise AssertionError("minimax-h3 accepted a .gguf DiT (BlineUI-GGUF refuses H3)")
    except ValueError:
        pass
    g = G.build("flux3-klein", G.GraphParams(prompt = "x", seed = 0, steps = 2, files = {"dit": "d.gguf", "te": "t",
                                                                                       "vae": "v"}))
    assert g["unet"]["class_type"] == "UnetLoaderGGUF", g["unet"]
    w = {"dit": "d", "te": "t", "vae": "v"}
    g = G.build("wan2.2-5b", G.GraphParams(prompt = "x", seed = 0, steps = 2, frames = 9, files = w))
    assert g[G.SAVE]["class_type"] == "SaveVideo" and g["video"]["class_type"] == "CreateVideo"
    g = G.build("wan2.2-5b", G.GraphParams(prompt = "x", seed = 0, steps = 2, frames = 9, files = w,
                                           extra = {"video_save": "frames"}))
    assert g[G.SAVE]["class_type"] == "SaveImage" and "video" not in g
    e = G.build("minimax-h3-image-edit", G.GraphParams(prompt = "x", seed = 0, steps = 6, width = 2048, height = 2048,
                                                       files = w, extra = {"input_images": ["a.png", "b.png"]}))
    assert e["cond"]["inputs"]["width"] == ["size", 0] and e["cond"]["inputs"]["images.image_1"] == ["resize_32", 0]
    assert e["cond"]["inputs"]["images.image_2"] == ["ref2", 0]
    e = G.build("minimax-h3-image-edit", G.GraphParams(prompt = "x", seed = 0, steps = 6, width = 1024, height = 1024,
                                                       files = w, extra = {"edit_size": "cell"}))
    assert e["cond"]["inputs"]["width"] == 1024 and "size" not in e


def test_expected_passes():
    assert M.expected_passes(M.load_card("minimax-h3")) == 20  # N steps = N evaluations (not the old 19)
    assert M.expected_passes(M.load_card("flux.3-klein-base-edit")) == 50  # what the template RUNS (cfg 1)
    assert M.expected_passes(M.load_card("flux.3-klein-base-edit"), cfg = 3.5) == 100  # Studio / BFL recipe
    assert M.expected_passes(M.load_card("ltx-2.5")) == 11  # 8 + 3, two stages
    assert M.expected_passes(M.load_card("qwen-image-2.1"), cfg_batched = True) == 25
    assert M.expected_passes(M.load_card("ltx-2.3")) == 11  # template two-stage: 8 + 3
    assert M.expected_passes(M.load_card("ideogram-4")) == 40  # two models per step
    assert M.load_card("flux.3-klein-base-edit")["studio"]["gen"]["guidance"] == 1.0  # Studio matched to what runs


def test_precision_blocks():
    # every format names each component's precision on both stacks; H3's template TE / video VAE differences are
    # accepted (reported, not VOID); a file the header says is fp32 is not reported as bf16
    c = M.load_card("minimax-h3")
    sp, cp = M.precision(c, "int8", "studio"), M.precision(c, "int8", "bline")
    assert cp["text_encoder"]["weights"] == "nvfp4_awq" and sp["text_encoder"]["weights"] == "int8_convrot"
    acc = {x for t in M.accepted_differences(c, "int8") for x in t["components"]}
    assert {"text_encoder", "vae"} <= acc, acc
    assert not any("text_encoder" in t["components"] for t in M.accepted_differences(c, "int8_teint8"))
    assert "precision" not in M.studio_load(c, "int8")
    f = M.load_card("flux.1-schnell")
    assert M.precision(f, "bf16", "bline")["vae"]["weights"] == "fp32"  # ae.safetensors header is F32
    q = M.load_card("qwen-image-2.1")
    assert M.precision(q, "int8", "bline")["transformer"] == {**M.precision(q, "int8", "bline")["transformer"],
                                                               "weights": "int8_convrot", "compute": "int8"}
    assert M.precision(q, "fp8", "bline")["transformer"]["compute"] == "fp8"  # bf16 file + fp8_e4m3fn_fast
    for fam in M.list_cards():
        card = M.load_card(fam)
        for name, fm in card["formats"].items():
            for comp, v in (M.precision(card, name, "studio") or {}).items():
                assert v["evidence"], (fam, name, comp)
    assert set(M.GATE_CHECK) <= set(M.TRAP_CHECKS)


def test_sources_and_planning():
    # every baseline-UI file has a fetch / build source or an explicit null with a reason (validate_card), none from
    # Bline-Org; H3 int8 carries the measured host floor that keeps it off a T4 VM
    n_src = n_null = 0
    for fam in M.list_cards():
        card = M.load_card(fam)
        for name, fm in card["formats"].items():
            for fname, e in ((fm.get("bline") or {}).get("sources") or {}).items():
                if e.get("source", "x") is None:
                    n_null += 1
                    assert e["reason"], (fam, fname)
                else:
                    n_src += 1
                    assert not e["repo"].lower().startswith(("bline-org/", "blineanonymous/")), (fam, fname, e)
    assert n_src > 30 and n_null > 0, (n_src, n_null)
    h3 = M.load_card("minimax-h3")["formats"]["int8"]
    assert h3["min_host_ram_gib"] >= 31 and h3["est_s_per_render"]["T4"] is None and h3["planning"]["cite"]
    q = M.load_card("qwen-image-2.1")["formats"]["int8"]
    # Studio runs INT8 ConvRot on sm75 since #17949 / #18441: no T4 refusal, and the card says why
    assert q["est_s_per_render"]["G4"] and "T4" not in " ".join(q["planning"]["studio_refused"])
    assert any("#18441" in c for c in q["planning"]["cite"])
    # GGUF is a timed pair: the baseline UI loads the same unsloth Q4_K_M (Qwen's with an architecture key added)
    for fam, dit in (("qwen-image-2.1", "qwen-image-2.1-Q4_K_M_arch.gguf"),
                     ("flux.3-klein", "flux-3-image-klein-Q4_K_M.gguf"),
                     ("flux.3-klein-base", "flux-3-image-klein-base-Q4_K_M.gguf")):
        g = M.load_card(fam)["formats"]["gguf"]
        assert g["bline"]["files"]["dit"] == dit and g["bline"]["sources"][dit]["repo"].startswith("unsloth/"), fam
        assert not str(g["studio"].get("gguf_filename", "")).startswith("<"), fam
        # the base the GGUF transformer is assembled with is prefetched (offline load), its dense transformer not
        base = [e for e in g["studio"]["sources"].values() if e.get("file") is None]
        assert base and all("transformer/*.safetensors" in (e.get("ignore_patterns") or []) for e in base), fam
    # a Studio cell loads offline: every format whose Studio encoder is a hosted int8 ConvRot / fp8 file lists that
    # file, so the VM prefetches it (val_nov21 A100 trial: without it Studio kept bf16 and the gate voided the pair)
    for fam in ("flux.3-klein", "flux.3-klein-base", "qwen-image-2.1", "wan2.2-5b"):
        for fmt, f in M.load_card(fam)["formats"].items():
            w = ((f["studio"].get("precision") or {}).get("text_encoder") or {}).get("weights")
            if w in ("int8_convrot", "fp8"):
                files = [e.get("file") or "" for e in f["studio"]["sources"].values()]
                assert any("text_encoder-" in x for x in files), (fam, fmt, files)
    # H3: Studio's audio VAE dtype is traced (no waiver needed), and the card names its standing format and why
    h = M.load_card("minimax-h3")
    assert all(f["studio"]["precision"]["audio_vae"]["weights"] == "fp32" for f in h["formats"].values())
    assert h["standing"]["format"] in h["formats"] and h["standing"]["why"]
    src = M.load_card("minimax-h3")["formats"]["int8"]["bline"]["sources"]
    assert "regroup" in src["minimax_h3_fl2va_pruned_int8_convrot.safetensors"]["build"]
    assert src["qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"]["source"] is None
    assert "sources" not in M.studio_load(M.load_card("minimax-h3"), "int8")


def test_round2_graphs():
    G_ = G
    ltx = G_.build("ltx-2.3", M.graph_params(M.load_card("ltx-2.3"), nonce = 1))
    assert ltx["sig2"]["inputs"]["sigmas"] == "0.85, 0.7250, 0.4219, 0.0" and ltx["noise2"]["inputs"]["noise_seed"] == 42
    assert ltx["vlat"]["inputs"]["width"] == 640 and ltx["vlat"]["inputs"]["height"] == 352
    assert ltx["lora"]["inputs"]["strength_model"] == 0.5 and ltx["guider2"]["inputs"]["positive"] == ["crop", 0]
    assert G_.canonical("ltx-2") == "ltx-2.3-dev"
    q = G_.build("qwen-image-2.1-edit", M.graph_params(M.load_card("qwen-image-2.1-edit"), nonce = 1))
    assert q["sampler"]["inputs"]["latent_image"] == ["cond", 2] and q["cond"]["inputs"]["resolution"] == 0
    assert q["prefix_cache"]["class_type"] == "QwenImage21Cache" and "latent" not in q
    i = G_.build("ideogram-4", M.graph_params(M.load_card("ideogram-4"), nonce = 1))
    assert (i["latent"]["inputs"]["width"], i["latent"]["inputs"]["height"]) == (768, 1376)
    assert i["guider"]["inputs"]["model_negative"] == ["unet_neg", 0] and i["override"]["inputs"]["cfg"] == 3.0
    for fam in ("minimax-h3-image", "minimax-h3-image-edit", "moss-image"):
        try:
            G_.build(fam, G_.GraphParams(prompt = "x", seed = 0, steps = 6, width = 1024, height = 1024,
                                         files = {"dit": "d.gguf", "te": "t", "vae": "v"}))
            raise AssertionError(f"{fam} accepted a .gguf DiT")
        except ValueError:
            pass


def test_h3_qkv_regroup():
    import numpy as np

    from models.convert import h3_qkv_regroup, h3_qkv_ungroup

    heads, hd, cols = 3, 4, 2
    # per-head layout: rows tagged (head, which, i) with which 0 = q, 1 = k, 2 = v
    rows = [(h, w, i) for h in range(heads) for w in range(3) for i in range(hd)]
    x = np.array([[h * 100 + w * 10 + i] * cols for (h, w, i) in rows], dtype = np.float32)
    y = h3_qkv_regroup(x, head_dim = hd)
    want = [(h, w, i) for w in range(3) for h in range(heads) for i in range(hd)]
    assert (y[:, 0] == np.array([h * 100 + w * 10 + i for (h, w, i) in want])).all(), y[:, 0]
    assert (h3_qkv_ungroup(y, head_dim = hd) == x).all()


def test_specs_families_registered():
    errs = []
    for spec in sorted((DB / "specs").glob("*.json")):
        d = json.loads(spec.read_text())
        for cell in d.get("cells") or []:
            if cell.get("backend") != "blineui":
                continue
            fam = (cell.get("options") or {}).get("family") or cell.get("family")
            try:
                G.canonical(fam)
            except KeyError:
                errs.append(f"{spec.name}: cell {cell.get('tag')}: family {fam!r} not registered")
    assert not errs, "\n".join(errs)


def main() -> int:
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok  ", name)
            except Exception as e:  # noqa: BLE001
                fails += 1
                print("FAIL", name, "\n" + str(e))
    print(f"{len(M.list_cards())} cards, {len(G.FAMILIES)} registered graphs:", "FAILED" if fails else "passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
