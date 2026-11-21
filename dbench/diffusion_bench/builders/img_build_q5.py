#!/usr/bin/env python3
"""baseline-UI-side model files for the nov5 round (Qwen-Image-2.1, -Turbo, FLUX.3 klein / klein-base x bf16 / fp8 / int8 /
nvfp4), built on the VM from the OFFICIAL vendor repos (or byte-identical unsloth mirrors, sha256 checked). Nothing is
read from Bline-Org / blineanonymous / any *-bline repo.

Ported from scripts/imgcmp_nov5/build_q5.py (build_bline -> build_bline_files; --tool / --refs default to
$DBENCH_CONVROT_TOOL / builders/ref_headers).
  python diffusion_bench/builders/img_build_q5.py --group q21|q21t|f3|f3b --roles dit_bf16,dit_int8,dit_fp8,dit_nvfp4,dit_nvfp4a,te_int8,vae \
      --out CM --tmp TMP --tool quant_int8_convrot.py --refs REF_DIR

Run with the baseline UI venv python (bline_kitchen + torch CUDA), BLINE_DIR set. A gated / private read uses
HF_TOKEN_ORG (never printed). Layouts:
  dit_bf16   q21/q21t: official transformer shards, img_mlp gate_layer + proj row-concatenated into gate_up (the baseline UI's
             layout, as build_bline.q21_dit). f3/f3b: BFL's own single-file model.safetensors (the baseline UI's bf16 file is
             byte-identical to it: same LFS sha256).
  dit_int8   q21/q21t: bline-model-tools quant_int8_convrot.py on dit_bf16 (the round-3 recipe, header-checked for q21).
             f3b: BFL's int8_convrot file as hosted in unsloth/flux-3-image-klein-base-FP8 (LFS sha256 = BFL's baseline-UI
             file). f3: built here to the same layout: q/k/v fused into qkv, the 172 linears BFL quantizes on the base,
             int8_tensorwise + ConvRot 256 through bline_kitchen, and the per-block {"attention": "bline_kitchen_int8"}
             tensors the base file carries.
  dit_fp8    the dit_int8 linear set as bline_quant float8_e4m3fn (per-tensor weight scale, no input scale: the baseline UI's
             own fp8 activation path), q/k/v fused on FLUX.3.
  dit_nvfp4  q21/q21t: Studio's NVFP4 layer map (unsloth/Qwen-Image-2.1-NVFP4: 48 nvfp4, the rest of its quantized
             set fp8), nvfp4 with a dynamic activation scale (as Studio), fp8 as dit_fp8. f3/f3b: the 198 linears BFL's
             variants/nvfp4 quantizes, re-quantized here with bline_kitchen (the baseline UI's block-scale layout) and BFL's
             calibrated input_scale; q/k/v fused (equal input scales: one input).
  dit_nvfp4a q21/q21t only: every dit_int8 linear nvfp4 (the baseline UI's speed ceiling).
  te_int8    q21: build_bline.q21_te + the tool (header-checked). f3/f3b: the Qwen3-VL-4B text_encoder shards, the same
             key map, the tool with the vision tower left out. q21t / f3b reuse q21 / f3 (Studio uses one encoder file
             for both variants).
  vae        q21: build_bline.q21_vae. f3/f3b: BFL ae.safetensors (= the baseline UI's flux2-vae, same sha256).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
try:  # extnames-aware safetensors (builders/_st.py); as a script or as builders.<module>
    from _st import safe_open
except ImportError:
    from builders._st import safe_open
try:  # extnames-aware safetensors (builders/_st.py); as a script or as builders.<module>
    from _st import save_file
except ImportError:
    from builders._st import save_file

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if os.environ.get("BLINE_DIR"):
    sys.path.insert(0, os.environ["BLINE_DIR"])  # bline.quant_ops for the fp8 / nvfp4 layouts
import build_bline_files as bc  # noqa: E402

TOK = os.environ.get("HF_TOKEN_ORG") or os.environ.get("HF_TOKEN") or None
FORBIDDEN = ("bline-org/", "blineanonymous/")
OUT = {
    "dit_bf16": "{g}_dit_bf16.safetensors", "dit_int8": "{g}_dit_int8_convrot.safetensors",
    "dit_fp8": "{g}_dit_fp8_scaled.safetensors", "dit_nvfp4": "{g}_dit_nvfp4.safetensors",
    "dit_nvfp4a": "{g}_dit_nvfp4_all.safetensors", "te_int8": "{g}_te_int8_convrot.safetensors",
    "dit_int8na": "{g}_dit_int8_convrot_noattn.safetensors",
    "vae": "{g}_vae.safetensors", "gguf": "{g}_dit_Q4_K_M.gguf",
}
GGUF = {"q21": ("unsloth/Qwen-Image-2.1-GGUF", "qwen-image-2.1-Q4_K_M.gguf"),
        "q21t": ("unsloth/Qwen-Image-2.1-Turbo-GGUF", "qwen-image-2.1-turbo-Q4_K_M.gguf"),
        "f3": ("unsloth/flux-3-image-klein-GGUF", "flux-3-image-klein-Q4_K_M.gguf"),
        "f3b": ("unsloth/flux-3-image-klein-base-GGUF", "flux-3-image-klein-base-Q4_K_M.gguf")}
SHARE = {("q21t", "te_int8"): "q21", ("q21t", "vae"): "q21", ("f3b", "te_int8"): "f3", ("f3b", "vae"): "f3"}
F3_REPO = {"f3": "black-forest-labs/flux-3-image-klein", "f3b": "black-forest-labs/flux-3-image-klein-base"}
F3_MIRROR = {"f3": "unsloth/flux-3-image-klein-FP8", "f3b": "unsloth/flux-3-image-klein-base-FP8"}
# BFL's gated repos are mirrored byte-identical, ungated, in unsloth/flux-3-image-klein[-base] (Studio's retired full
# copies): fetched there, accepted only when the sha256 equals BFL's (pinned 2026-11-04, ref_headers/f3_official_sha.json)
F3_FULL = {"f3": "unsloth/flux-3-image-klein", "f3b": "unsloth/flux-3-image-klein-base"}
F3_SHA = json.load(open(HERE / "ref_headers" / "f3_official_sha.json"))


def dl_f3(g, path):
    p = dl(F3_FULL[g], path)
    got, want = bc.sha256(p), F3_SHA[F3_REPO[g]][path]
    if got != want:
        raise SystemExit(f"sha256 mismatch {F3_FULL[g]}/{path}: {got} != BFL {want}")
    say("SHA_OK", F3_FULL[g], path, got[:16])
    return p
Q21T_REPO = "Qwen/Qwen-Image-2.1-Turbo"


def say(*a):
    print(*a, flush = True)


def dl(repo, path, token = None):
    assert not repo.lower().startswith(FORBIDDEN) and not repo.lower().endswith("-bline"), repo
    t = time.time()
    p = os.path.realpath(hf_hub_download(bc.X.ext(repo), bc.X.ext(path), token = token))
    say("FETCH", repo, path, round(os.path.getsize(p) / 1e9, 2), "GB", round(time.time() - t, 1), "s")
    return p


def load_sd(paths):
    sd = {}
    for p in paths:
        with safe_open(p, "pt") as f:
            for k in f.keys():
                sd[k] = f.get_tensor(k)
    return sd


def cq(fmt, **extra):
    conf = {"format": fmt, **extra}
    return torch.tensor(list(json.dumps(conf).encode("utf-8")), dtype = torch.uint8)


# ----------------------------------------------------------------- quantizers (the baseline UI's own layouts)
def _qt():
    qo = bc.X.import_module("bline.quant_ops")  # registers TensorCoreFP8E4M3Layout / TensorCoreNVFP4Layout
    return qo.QuantizedTensor


def q_fp8(base, w, input_scale = None):
    QT = _qt()
    qt = QT.from_float(w.to("cuda", torch.bfloat16).contiguous(), "TensorCoreFP8E4M3Layout", scale = "recalculate")
    out = {k: v.detach().cpu().contiguous() for k, v in qt.state_dict(base + ".weight").items()}
    out[base + ".bline_quant"] = cq("float8_e4m3fn")
    if input_scale is not None:
        out[base + ".input_scale"] = torch.tensor(float(input_scale), dtype = torch.float32)
    return out


def q_nvfp4(base, w, input_scale = None):
    QT = _qt()
    qt = QT.from_float(w.to("cuda", torch.bfloat16).contiguous(), "TensorCoreNVFP4Layout")
    out = {k: v.detach().cpu().contiguous() for k, v in qt.state_dict(base + ".weight").items()}
    out[base + ".bline_quant"] = cq("nvfp4")
    if input_scale is not None:
        out[base + ".input_scale"] = torch.tensor(float(input_scale), dtype = torch.float32)
    return out


def q_int8(base, w):
    QuantizedTensor = bc.X.import_module("bline_kitchen.tensor").QuantizedTensor
    qt = QuantizedTensor.from_float(w.to("cuda", torch.bfloat16).contiguous(), "TensorWiseINT8Layout", is_weight = True,
                                    per_channel = True, convrot = True, convrot_groupsize = 256)
    out = {k: v.detach().cpu().contiguous() for k, v in qt.state_dict(base + ".weight").items()}
    out[base + ".bline_quant"] = cq("int8_tensorwise", convrot = True, convrot_groupsize = 256)
    return out


# ----------------------------------------------------------------- Qwen-Image-2.1 (+ Turbo)
def q21_dit_bf16(g, dst):
    if g == "q21":
        bc.q21_dit(str(dst))
        bc.SOURCES.clear()  # fetched inputs live in the shared HF cache: never deleted here
    else:
        paths = [dl(Q21T_REPO, f) for f in sorted(bc.official_sha(Q21T_REPO, "transformer/")) if f.endswith(".safetensors")]
        sd = load_sd(paths)
        for k in [k for k in sd if k.endswith(".img_mlp.gate_layer.weight")]:
            b = k[: -len("gate_layer.weight")]
            sd[b + "gate_up.weight"] = torch.cat([sd.pop(k), sd.pop(b + "proj.weight")], 0).contiguous()
        save_file(sd, str(dst), metadata = {"format": "pt"})


def q21_int8_layerwise(tool, src, dst):
    import importlib.util
    spec = importlib.util.spec_from_file_location("qtool", tool)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    sd, out, n = load_sd([src]), {}, 0
    for k, v in sd.items():
        base = k[: -len(".weight")] if k.endswith(".weight") else None
        if base and v.dim() == 2 and m.classify(k, tuple(v.shape))[0] and v.shape[1] % 256 == 0:
            out.update(q_int8(base, v))
            n += 1
            torch.cuda.empty_cache()
        else:
            out[k] = v
    save_file(out, str(dst), metadata = {"format": "pt"})
    say("QUANTIZED", Path(dst).name, "int8 layerwise", n)


def quant_set_from(path):
    with safe_open(path, "pt") as f:
        return sorted(k[: -len(".bline_quant")] for k in f.keys() if k.endswith(".bline_quant"))


def q21_variant(src_bf16, int8_file, dst, mode):
    """mode fp8 | nvfp4 (Studio's map) | nvfp4a (all)."""
    qset = quant_set_from(int8_file)
    smap = json.load(open(HERE / "q21_nvfp4_map.json"))  # diffusers names -> nvfp4 | fp8

    def studio(base):
        if base.endswith(".img_mlp.gate_up"):
            a = smap.get(base[: -len("gate_up")] + "gate_layer")
            b = smap.get(base[: -len("gate_up")] + "proj")
            return "nvfp4" if (a == "nvfp4" and b == "nvfp4") else ("fp8" if (a or b) else None)
        return smap.get(base)

    sd = load_sd([src_bf16])
    out, n = {}, {"nvfp4": 0, "fp8": 0, "bf16": 0}
    for k, v in sd.items():
        base = k[: -len(".weight")] if k.endswith(".weight") else None
        fmt = None
        if base is not None and v.dim() == 2:
            if mode == "fp8":
                fmt = "fp8" if base in qset else None
            elif mode == "nvfp4a":
                fmt = "nvfp4" if base in qset else None
            else:
                fmt = studio(base)
        if fmt == "nvfp4":
            out.update(q_nvfp4(base, v))
        elif fmt == "fp8":
            out.update(q_fp8(base, v))
        else:
            out[k] = v
        n[fmt or "bf16"] += 1 if base is not None and v.dim() == 2 else 0
    save_file(out, str(dst), metadata = {"format": "pt"})
    say("QUANTIZED", dst.name, mode, n)


# ----------------------------------------------------------------- FLUX.3 klein / klein-base
QKV = re.compile(r"^(.*)\.(q_proj|k_proj|v_proj)\.weight$")


def fuse_qkv(sd):
    for k in [k for k in sd if k.endswith(".q_proj.weight")]:
        b = k[: -len(".q_proj.weight")]
        sd[b + ".qkv.weight"] = torch.cat([sd.pop(b + ".q_proj.weight"), sd.pop(b + ".k_proj.weight"),
                                          sd.pop(b + ".v_proj.weight")], 0).contiguous()
    return sd


def f3_int8_set(refs):
    d = json.load(open(Path(refs) / "f3b_int8.json"))
    small = d["__small__"]
    qset = sorted(k[: -len(".bline_quant")] for k in small if k.endswith(".bline_quant"))
    attn = {k: v for k, v in small.items() if k.endswith(".bline_attention.config")}
    return qset, attn


def f3_variant(g, src_bf16, dst, mode, refs):
    sd = fuse_qkv(load_sd([src_bf16]))
    qset, attn = f3_int8_set(refs)
    in_scale = {}
    if mode == "nvfp4":
        ref = json.load(open(Path(refs) / f"{F3_REPO[g].split('/')[1]}_nvfp4_set.json"))
        var = dl_f3(g, "variants/nvfp4/model.safetensors")
        with safe_open(var, "pt") as f:
            keys = set(f.keys())
            for b in ref["nvfp4"]:
                if b + ".input_scale" in keys:
                    in_scale[b] = float(f.get_tensor(b + ".input_scale"))
        qset = []
        for b in ref["nvfp4"]:
            m = re.match(r"^(.*)\.(q_proj|k_proj|v_proj)$", b)
            qset.append(m.group(1) + ".qkv" if m else b)
        fused = {}
        for b, s in in_scale.items():
            m = re.match(r"^(.*)\.(q_proj|k_proj|v_proj)$", b)
            if m:
                fused[m.group(1) + ".qkv"] = max(fused.get(m.group(1) + ".qkv", 0.0), s)
        in_scale.update(fused)
        qset = sorted(set(qset))
    qs = set(qset)
    out, n = {}, {}
    for k, v in sd.items():
        base = k[: -len(".weight")] if k.endswith(".weight") else None
        if base in qs and v.dim() == 2:
            if mode in ("int8", "int8na"):
                out.update(q_int8(base, v))
            elif mode == "fp8":
                out.update(q_fp8(base, v))
            else:
                out.update(q_nvfp4(base, v, in_scale.get(base)))
            n[mode] = n.get(mode, 0) + 1
        else:
            out[k] = v
    if mode == "int8na":  # the int8 file without BFL's per-block int8 attention switch (isolates that lever)
        mode = "int8"
        attn = {}
    if mode == "int8":
        for k, txt in attn.items():
            out[k] = torch.tensor(list(txt.encode("utf-8")), dtype = torch.uint8)
    missing = sorted(qs - {k[: -len(".bline_quant")] for k in out if k.endswith(".bline_quant")})
    save_file(out, str(dst), metadata = {"format": "pt", "model": "f3-image-small"})
    say("QUANTIZED", dst.name, mode, n, "input_scales", len([b for b in qs if b in in_scale]), "missing", missing[:6])
    if missing:
        raise SystemExit(f"{dst.name}: {len(missing)} planned layers not quantized")


def f3_te(dst, tool):
    paths = [dl(F3_FULL["f3"], f) for f in ("text_encoder/model-00001-of-00002.safetensors",
                                             "text_encoder/model-00002-of-00002.safetensors")]  # = BFL's shards (LFS sha256)
    sd = {}
    for k, v in bc.iter_tensors(paths):
        sd[k.replace("model.language_model.", "model.", 1) if k.startswith("model.language_model.") else k] = v
    tmp = str(dst) + ".src.safetensors"
    save_file(sd, tmp, metadata = {"format": "pt"})
    del sd
    bc.run_tool(tool, tmp, str(dst), ["--exclude", r"^model\.visual\."])
    os.remove(tmp)


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required = True, choices = ["q21", "q21t", "f3", "f3b"])
    ap.add_argument("--roles", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--tmp", default = None)
    ap.add_argument("--tool", default = os.environ.get("DBENCH_CONVROT_TOOL"),
                    help = "Bline-Org/bline-model-tools quant_int8_convrot.py (GitHub); default $DBENCH_CONVROT_TOOL")
    ap.add_argument("--refs", default = str(HERE / "ref_headers"))
    a = ap.parse_args()
    if any(r in ("dit_int8", "dit_int8na", "te_int8") for r in a.roles.split(",")) and not (a.tool and Path(a.tool).exists()):
        ap.error("an int8 role needs --tool / $DBENCH_CONVROT_TOOL = bline-model-tools quant_int8_convrot.py")
    g = a.group
    out = Path(a.out)
    out.mkdir(parents = True, exist_ok = True)
    tmpd = Path(a.tmp or out / "_tmp")
    tmpd.mkdir(parents = True, exist_ok = True)
    man_p = out / f"manifest_q5_{g}.json"
    man = json.load(open(man_p)) if man_p.exists() else {}
    roles = a.roles.split(",")
    need_bf16 = any(r.startswith("dit_") for r in roles)
    # the listed order when dit_bf16 is listed (drop_bf16 runs where it stands), else dit_bf16 first
    order = roles if "dit_bf16" in roles else (["dit_bf16"] if need_bf16 else []) + roles

    def path(role, grp = None):
        return out / OUT[role].format(g = grp or SHARE.get((g, role), g))

    for role in order:
        if role == "drop_bf16":  # small-disk hosts (Kaggle): the bf16 source is only an input to the quantized files
            bf = path("dit_bf16")
            if bf.exists() or bf.is_symlink():
                if bc.own_remove(bf, [out, tmpd]):  # never a shared-cache blob behind a link
                    say("DROPPED", bf.name)
                elif bf.is_symlink():
                    os.remove(bf)  # only the link in --out; its shared target stays
            continue
        dst = path(role)
        t0 = time.time()
        if dst.exists():
            say("EXISTS", role, dst.name)
            continue
        if (g, role) in SHARE:
            raise SystemExit(f"{role} for {g} comes from group {SHARE[(g, role)]}: build that first")
        if role == "dit_bf16":
            if g in ("q21", "q21t"):
                q21_dit_bf16(g, dst)
            else:
                os.symlink(dl_f3(g, "model.safetensors"), dst)
        elif role == "dit_int8":
            if g in ("q21", "q21t"):
                try:
                    bc.run_tool(a.tool, str(path("dit_bf16")), str(dst), bc.TOOL_ARGS[("q21", "dit")])
                except SystemExit as e:  # T4: the tool's device copies do not fit a 15 GB card on the 16 GB 2.1 DiT
                    say("TOOL_FAILED", e, "-> layer-by-layer bline_kitchen int8 ConvRot with the tool's own layer rule")
                    q21_int8_layerwise(a.tool, str(path("dit_bf16")), dst)
            elif g == "f3b":
                p = dl(F3_MIRROR["f3b"], "flux-3-image-klein-base-INT8.safetensors")
                got = bc.sha256(p)
                want = bc.official_sha(F3_MIRROR["f3b"], "flux-3-image-klein-base-INT8")["flux-3-image-klein-base-INT8.safetensors"]
                say("SHA", got[:16], "mirror", want[:16], "BFL bline int8 8949ecfc1bf3871b")
                os.symlink(p, dst)
            else:
                f3_variant(g, str(path("dit_bf16")), dst, "int8", a.refs)
        elif role == "dit_int8na":
            f3_variant(g, str(path("dit_bf16")), dst, "int8na", a.refs)
        elif role in ("dit_fp8", "dit_nvfp4", "dit_nvfp4a"):
            mode = role[4:]
            if g in ("q21", "q21t"):
                q21_variant(str(path("dit_bf16")), str(path("dit_int8")), dst, mode)
            else:
                if mode == "nvfp4a":
                    continue
                f3_variant(g, str(path("dit_bf16")), dst, mode, a.refs)
        elif role == "te_int8":
            if g == "q21":
                tmp = str(tmpd / "q21_te.src.safetensors")
                bc.q21_te(tmp)
                bc.run_tool(a.tool, tmp, str(dst), bc.TOOL_ARGS[("q21", "te")])
                os.remove(tmp)
                bc.SOURCES.clear()  # shared HF cache inputs: never deleted
            else:
                f3_te(dst, a.tool)
        elif role == "gguf":
            # the same file Studio loads; Qwen-Image-2.1's has no general.architecture, which the GGUF loader node pack needs
            p = dl(*GGUF[g])
            import subprocess
            r = subprocess.run([sys.executable, str(HERE / "gguf_set_arch.py"), p, str(dst), "qwen_image"],
                               capture_output = True, text = True)
            say(r.stdout.strip(), r.stderr[-800:])
            if "ARCH_PRESENT" in r.stdout:
                os.remove(dst)
                os.symlink(p, dst)
        elif role == "vae":
            if g == "q21":
                bc.q21_vae(dst)
            else:
                os.symlink(dl_f3(g, "ae.safetensors"), dst)
        if dst.exists():
            man[role] = {"file": str(dst), "bytes": os.path.getsize(dst), "build_s": round(time.time() - t0, 1)}
            json.dump(man, open(man_p, "w"), indent = 1)
            say("BUILT", g, role, dst.name, round(os.path.getsize(dst) / 1e9, 2), "GB", man[role]["build_s"], "s")
    # names the plan refers to, whatever group built them
    for role in OUT:
        src = path(role)
        alias = out / OUT[role].format(g = g)
        if src != alias and src.exists() and not alias.exists():
            os.symlink(src, alias)
    say("BUILD_Q5_DONE", g, sorted(p.name for p in out.glob(f"{g}_*")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
