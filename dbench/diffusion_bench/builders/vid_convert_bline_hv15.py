#!/usr/bin/env python3
"""Build baseline-UI-loadable HunyuanVideo-1.5 480p t2v files from the OFFICIAL repos (never a Bline-Org repack).

The file set is the one the baseline UI's template video_hunyuan_video_1.5_720p_t2v.json loads (UNETLoader dit, DualCLIPLoader
[qwen_2.5_vl_7b*, byt5_small_glyphxl*] type hunyuan_video_15, VAELoader), with the 480p t2v DiT instead of 720p:
  dit   tencent/HunyuanVideo-1.5 transformer/480p_t2v (fp32, original Tencent layout: separate img/txt q, k, v) ->
        fp16 in the baseline UI's module names (bline.ldm.hunyuan_video.model): q/k/v concatenated into *_attn.qkv,
        *_attn_q_norm -> *_attn.norm.query_norm, *_mod.linear -> *_mod.lin, *_mlp.fc1/fc2 -> *_mlp.0/2,
        MLPEmbedder mlp.0/2 | linear_1/2 -> in_layer/out_layer. These names are fixed points of the baseline UI's
        HunyuanVideo.process_unet_state_dict, so the file loads as is. vision_in.* and cond_type_embedding are kept:
        model detection needs vision_in.proj.0.weight to pick HunyuanVideo15 (vision_in_dim 1152).
  vae   tencent/HunyuanVideo-1.5 vae (fp32) -> fp16; the baseline UI's HV1.5 VAE (bline.ldm.hunyuan_video.vae_refiner, chosen by
        decoder.conv_in.conv.weight with 32 input channels) uses the official key names, so keys are unchanged.
  te    Qwen/Qwen2.5-VL-7B-Instruct (bf16; model.* + visual.* is the baseline UI's Qwen25_7BVLI layout) -> legacy "scaled fp8"
        (per-tensor scale_weight on every decoder-layer projection + a 2-element scaled_fp8 marker = full-precision
        matmul, what bline.utils.convert_old_quants reads), or bf16 with --te bf16. lm_head is dropped (the baseline UI's
        Qwen25_7BVLI has no lm_head; the hidden state of layer -3 is the conditioning).
  byt5  hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v text_encoder_2 (the Glyph-SDXL-v2 ByT5 encoder,
        HF T5EncoderModel layout, fp32) -> fp16, keys unchanged (bline.text_encoders.t5.T5 = HF T5 names; the TE
        routes a state dict holding encoder.block.0.layer.0.SelfAttention.o.weight to the ByT5 model).
--dit-src diffusers / --vae-src diffusers build the SAME files (byte-identical: plans are key-sorted) from
hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v transformer/ (7 fp32 shards) / vae/ (2 shards) by undoing
diffusers' convert_hunyuan_video1_5_to_diffusers.py exactly: renames, plus the token-refiner to_q/to_k/to_v
re-concatenated into self_attn_qkv and final_layer.adaLN_modulation.1 un-swapped (scale/shift). That repo is read through
the shared HF cache (HF_HOME, or --hf-home), an already-cached snapshot is reused, and none of its files are deleted.
Every tensor is converted one at a time from safe_open and written by a streaming safetensors writer (header from the
plan's shapes / dtypes first, then the bytes in order), so peak RSS stays a few GB even for the 33 GB fp32 DiT.
Every output is listed in manifest.json with its source repo, revision, source file LFS sha256 and its own sha256.
Sources are deleted after conversion unless --keep-src (--src cache reads through the HF cache and never deletes).

Ported from scripts/vidcmp_nov5/convert_bline_hv15.py.
Usage: python diffusion_bench/builders/vid_convert_bline_hv15.py --out DIR --src DIR|cache [--only dit,vae,te,byt5] [--keep-src] [--te fp8|bf16]
       [--dit-src tencent|diffusers] [--vae-src tencent|diffusers] [--hf-home DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import re
import shutil
import struct
import time
from dataclasses import dataclass
from pathlib import Path

HV_REPO = "tencent/HunyuanVideo-1.5"
DIT_FILE = "transformer/480p_t2v/diffusion_pytorch_model.safetensors"
VAE_FILE = "vae/diffusion_pytorch_model.safetensors"
QWEN_REPO = "Qwen/Qwen2.5-VL-7B-Instruct"
HVD_REPO = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v"
BYT5_REPO = HVD_REPO
BYT5_FILE = "text_encoder_2/model.safetensors"
OUT = {"dit": "hunyuanvideo1.5_480p_t2v_fp16_official.safetensors",
       "vae": "hunyuanvideo15_vae_fp16_official.safetensors",
       "te_fp8": "qwen_2.5_vl_7b_fp8_scaled_official.safetensors",
       "te_bf16": "qwen_2.5_vl_7b_bf16_official.safetensors",
       "byt5": "byt5_small_glyphxl_fp16_official.safetensors"}
FP8_MAX = 448.0
DTYPE_SIZE = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "F8_E5M2": 1, "I64": 8, "I32": 4, "I16": 2,
              "I8": 1, "U8": 1, "BOOL": 1}




def _ext(name):
    """extnames.ext (real download names in a public copy), loaded from the toolkit root beside builders/."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import extnames  # noqa: PLC0415

    return extnames.ext(name)
def assert_official(repo: str) -> None:
    r = repo.lower()
    if "bline" in r or r.split("/")[0] in ("bline-org", "blineanonymous"):
        raise SystemExit(f"refusing non-official repo {repo}")
    if repo not in (HV_REPO, QWEN_REPO, HVD_REPO):
        raise SystemExit(f"repo {repo} is not an allowed official source")


# ------------------------------------------------------------------ plans (header-only: run on a header json too)
@dataclass
class Entry:
    key: str  # output key
    op: str  # cast | cat | fp8 | fp8_scale | marker
    srcs: tuple  # source keys
    dtype: str  # safetensors dtype string of the output
    shape: tuple


def read_header(path) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def _tensors(header: dict) -> dict:
    return {k: v for k, v in header.items() if k != "__metadata__"}


_DIT_RENAMES = [  # (regex on the official key, replacement) applied in order; first match wins
    (r"^(double_blocks\.\d+\.(?:img|txt))_attn_q_norm\.weight$", r"\1_attn.norm.query_norm.weight"),
    (r"^(double_blocks\.\d+\.(?:img|txt))_attn_k_norm\.weight$", r"\1_attn.norm.key_norm.weight"),
    (r"^(double_blocks\.\d+\.(?:img|txt))_attn_proj\.(weight|bias)$", r"\1_attn.proj.\2"),
    (r"^(double_blocks\.\d+\.(?:img|txt))_mod\.linear\.(weight|bias)$", r"\1_mod.lin.\2"),
    (r"^(double_blocks\.\d+\.(?:img|txt))_mlp\.fc1\.(weight|bias)$", r"\1_mlp.0.\2"),
    (r"^(double_blocks\.\d+\.(?:img|txt))_mlp\.fc2\.(weight|bias)$", r"\1_mlp.2.\2"),
    (r"^(txt_in\.individual_token_refiner\.blocks\.\d+)\.self_attn_qkv\.(weight|bias)$", r"\1.self_attn.qkv.\2"),
    (r"^(txt_in\.individual_token_refiner\.blocks\.\d+)\.self_attn_proj\.(weight|bias)$", r"\1.self_attn.proj.\2"),
    (r"^(txt_in\.individual_token_refiner\.blocks\.\d+)\.mlp\.fc1\.(weight|bias)$", r"\1.mlp.0.\2"),
    (r"^(txt_in\.individual_token_refiner\.blocks\.\d+)\.mlp\.fc2\.(weight|bias)$", r"\1.mlp.2.\2"),
    (r"^(txt_in\.individual_token_refiner\.blocks\.\d+\.(?:norm1|norm2|adaLN_modulation\.1)\.(?:weight|bias))$", r"\1"),
    (r"^(txt_in\.t_embedder|time_in)\.mlp\.0\.(weight|bias)$", r"\1.in_layer.\2"),
    (r"^(txt_in\.t_embedder|time_in)\.mlp\.2\.(weight|bias)$", r"\1.out_layer.\2"),
    (r"^txt_in\.c_embedder\.linear_1\.(weight|bias)$", r"txt_in.c_embedder.in_layer.\1"),
    (r"^txt_in\.c_embedder\.linear_2\.(weight|bias)$", r"txt_in.c_embedder.out_layer.\1"),
    (r"^(txt_in\.input_embedder\.(?:weight|bias))$", r"\1"),
    (r"^(img_in\.proj\.(?:weight|bias))$", r"\1"),
    (r"^(final_layer\.(?:linear|adaLN_modulation\.1)\.(?:weight|bias))$", r"\1"),
    (r"^(byt5_in\.(?:layernorm|fc1|fc2|fc3)\.(?:weight|bias))$", r"\1"),
    (r"^(vision_in\.proj\.[0134]\.(?:weight|bias))$", r"\1"),
    (r"^(cond_type_embedding\.weight)$", r"\1"),
]
_QKV = re.compile(r"^(double_blocks\.\d+\.(?:img|txt))_attn_([qkv])\.(weight|bias)$")


def dit_plan(header: dict, dtype: str = "F16") -> list[Entry]:
    """Official Tencent HV1.5 DiT keys -> baseline-UI hunyuan_video module keys (fp16). Unknown keys are an error."""
    t = _tensors(header)
    out: list[Entry] = []
    seen_qkv = set()
    for k in t:
        m = _QKV.match(k)
        if m:
            base, _, wb = m.groups()
            if (base, wb) in seen_qkv:
                continue
            seen_qkv.add((base, wb))
            srcs = tuple(f"{base}_attn_{p}.{wb}" for p in "qkv")
            shapes = [t[s]["shape"] for s in srcs]
            if any(s != shapes[0] for s in shapes):
                raise ValueError(f"q/k/v shape mismatch at {base} {wb}: {shapes}")
            shape = (shapes[0][0] * 3,) + tuple(shapes[0][1:])
            out.append(Entry(f"{base}_attn.qkv.{wb}", "cat", srcs, dtype, shape))
            continue
        for pat, rep in _DIT_RENAMES:
            if re.match(pat, k):
                out.append(Entry(re.sub(pat, rep, k), "cast", (k,), dtype, tuple(t[k]["shape"])))
                break
        else:
            raise ValueError(f"unmapped HunyuanVideo-1.5 DiT key {k}")
    _check_unique(out)
    return sorted(out, key = lambda e: e.key)  # order-independent: tencent and diffusers sources give the same file


def vae_plan(header: dict, dtype: str = "F16") -> list[Entry]:
    """The baseline UI's HV1.5 VAE (vae_refiner Encoder/Decoder, CarriedConv3d .conv, RMS_norm .gamma) = official names."""
    t = _tensors(header)
    return sorted((Entry(k, "cast", (k,), dtype, tuple(v["shape"])) for k, v in t.items()), key = lambda e: e.key)


_QWEN_FP8 = re.compile(r"^model\.layers\.\d+\.(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)\.weight$")


def te_plan(header: dict, mode: str = "fp8") -> list[Entry]:
    """Qwen2.5-VL-7B-Instruct (model.*, visual.*, lm_head) -> baseline-UI qwen_2.5_vl_7b; fp8 = legacy scaled fp8."""
    t = _tensors(header)
    out: list[Entry] = []
    for k, v in t.items():
        if k == "lm_head.weight":
            continue
        if not (k.startswith("model.") or k.startswith("visual.")):
            raise ValueError(f"unexpected Qwen2.5-VL key {k}")
        if k.startswith("model.language_model."):  # newer transformers save layout
            k2 = "model." + k[len("model.language_model."):]
        elif k.startswith("model.visual."):
            k2 = k[len("model."):]
        else:
            k2 = k
        shape = tuple(v["shape"])
        if mode == "fp8" and _QWEN_FP8.match(k2):
            out.append(Entry(k2, "fp8", (k,), "F8_E4M3", shape))
            out.append(Entry(k2[:-len(".weight")] + ".scale_weight", "fp8_scale", (k,), "F32", ()))
        else:
            out.append(Entry(k2, "cast", (k,), "BF16", shape))
    if mode == "fp8":
        out.append(Entry("scaled_fp8", "marker", (), "F8_E4M3", (2,)))
    _check_unique(out)
    return out


def byt5_plan(header: dict, dtype: str = "F16") -> list[Entry]:
    t = _tensors(header)
    for k in t:
        if not (k.startswith("encoder.") or k == "shared.weight"):
            raise ValueError(f"unexpected ByT5 key {k}")
    return [Entry(k, "cast", (k,), dtype, tuple(v["shape"])) for k, v in t.items()]


# ------------------------------------------------------------------ diffusers source -> official (Tencent) view
# hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v was made by diffusers'
# scripts/convert_hunyuan_video1_5_to_diffusers.py. Its only non-rename steps, undone here exactly:
#   refiner self_attn_qkv was chunked into attn.to_q/to_k/to_v      -> concatenated back (dim 0, q k v)
#   final_layer.adaLN_modulation.1 went through swap_scale_shift    -> swapped back (the swap is its own inverse)
# Everything else (DiT and VAE) is a pure rename with the same block indices. The view maps every OFFICIAL key to
# (op, diffusers source keys), so the official-layout plan (dit_plan / vae_plan) runs unchanged on top of it.
_D2O_DIT = [
    (r"^time_embed\.timestep_embedder\.linear_([12])\.(weight|bias)$", lambda m: f"time_in.mlp.{_l(m[1])}.{m[2]}"),
    (r"^time_embed\.timestep_embedder_r\.linear_([12])\.(weight|bias)$", lambda m: f"time_r_in.mlp.{_l(m[1])}.{m[2]}"),
    (r"^context_embedder\.time_text_embed\.timestep_embedder\.linear_([12])\.(weight|bias)$",
     lambda m: f"txt_in.t_embedder.mlp.{_l(m[1])}.{m[2]}"),
    (r"^context_embedder\.time_text_embed\.text_embedder\.linear_([12])\.(weight|bias)$",
     lambda m: f"txt_in.c_embedder.linear_{m[1]}.{m[2]}"),
    (r"^context_embedder\.proj_in\.(weight|bias)$", lambda m: f"txt_in.input_embedder.{m[1]}"),
    (r"^context_embedder\.token_refiner\.refiner_blocks\.(\d+)\.(norm1|norm2)\.(weight|bias)$",
     lambda m: f"txt_in.individual_token_refiner.blocks.{m[1]}.{m[2]}.{m[3]}"),
    (r"^context_embedder\.token_refiner\.refiner_blocks\.(\d+)\.attn\.to_out\.0\.(weight|bias)$",
     lambda m: f"txt_in.individual_token_refiner.blocks.{m[1]}.self_attn_proj.{m[2]}"),
    (r"^context_embedder\.token_refiner\.refiner_blocks\.(\d+)\.ff\.net\.0\.proj\.(weight|bias)$",
     lambda m: f"txt_in.individual_token_refiner.blocks.{m[1]}.mlp.fc1.{m[2]}"),
    (r"^context_embedder\.token_refiner\.refiner_blocks\.(\d+)\.ff\.net\.2\.(weight|bias)$",
     lambda m: f"txt_in.individual_token_refiner.blocks.{m[1]}.mlp.fc2.{m[2]}"),
    (r"^context_embedder\.token_refiner\.refiner_blocks\.(\d+)\.norm_out\.linear\.(weight|bias)$",
     lambda m: f"txt_in.individual_token_refiner.blocks.{m[1]}.adaLN_modulation.1.{m[2]}"),
    (r"^context_embedder_2\.norm\.(weight|bias)$", lambda m: f"byt5_in.layernorm.{m[1]}"),
    (r"^context_embedder_2\.linear_([123])\.(weight|bias)$", lambda m: f"byt5_in.fc{m[1]}.{m[2]}"),
    (r"^image_embedder\.norm_in\.(weight|bias)$", lambda m: f"vision_in.proj.0.{m[1]}"),
    (r"^image_embedder\.linear_1\.(weight|bias)$", lambda m: f"vision_in.proj.1.{m[1]}"),
    (r"^image_embedder\.linear_2\.(weight|bias)$", lambda m: f"vision_in.proj.3.{m[1]}"),
    (r"^image_embedder\.norm_out\.(weight|bias)$", lambda m: f"vision_in.proj.4.{m[1]}"),
    (r"^x_embedder\.proj\.(weight|bias)$", lambda m: f"img_in.proj.{m[1]}"),
    (r"^cond_type_embed\.weight$", lambda m: "cond_type_embedding.weight"),
    (r"^transformer_blocks\.(\d+)\.norm1\.linear\.(weight|bias)$", lambda m: f"double_blocks.{m[1]}.img_mod.linear.{m[2]}"),
    (r"^transformer_blocks\.(\d+)\.norm1_context\.linear\.(weight|bias)$",
     lambda m: f"double_blocks.{m[1]}.txt_mod.linear.{m[2]}"),
    (r"^transformer_blocks\.(\d+)\.attn\.to_([qkv])\.(weight|bias)$", lambda m: f"double_blocks.{m[1]}.img_attn_{m[2]}.{m[3]}"),
    (r"^transformer_blocks\.(\d+)\.attn\.add_([qkv])_proj\.(weight|bias)$",
     lambda m: f"double_blocks.{m[1]}.txt_attn_{m[2]}.{m[3]}"),
    (r"^transformer_blocks\.(\d+)\.attn\.norm_([qk])\.weight$", lambda m: f"double_blocks.{m[1]}.img_attn_{m[2]}_norm.weight"),
    (r"^transformer_blocks\.(\d+)\.attn\.norm_added_([qk])\.weight$",
     lambda m: f"double_blocks.{m[1]}.txt_attn_{m[2]}_norm.weight"),
    (r"^transformer_blocks\.(\d+)\.attn\.to_out\.0\.(weight|bias)$", lambda m: f"double_blocks.{m[1]}.img_attn_proj.{m[2]}"),
    (r"^transformer_blocks\.(\d+)\.attn\.to_add_out\.(weight|bias)$", lambda m: f"double_blocks.{m[1]}.txt_attn_proj.{m[2]}"),
    (r"^transformer_blocks\.(\d+)\.ff\.net\.0\.proj\.(weight|bias)$", lambda m: f"double_blocks.{m[1]}.img_mlp.fc1.{m[2]}"),
    (r"^transformer_blocks\.(\d+)\.ff\.net\.2\.(weight|bias)$", lambda m: f"double_blocks.{m[1]}.img_mlp.fc2.{m[2]}"),
    (r"^transformer_blocks\.(\d+)\.ff_context\.net\.0\.proj\.(weight|bias)$",
     lambda m: f"double_blocks.{m[1]}.txt_mlp.fc1.{m[2]}"),
    (r"^transformer_blocks\.(\d+)\.ff_context\.net\.2\.(weight|bias)$", lambda m: f"double_blocks.{m[1]}.txt_mlp.fc2.{m[2]}"),
    (r"^proj_out\.(weight|bias)$", lambda m: f"final_layer.linear.{m[1]}"),
]
_D_REFINER_QKV = re.compile(r"^context_embedder\.token_refiner\.refiner_blocks\.(\d+)\.attn\.to_([qkv])\.(weight|bias)$")
_D_NORM_OUT = re.compile(r"^norm_out\.linear\.(weight|bias)$")


def _l(i: str) -> int:  # diffusers linear_1/linear_2 = original mlp.0/mlp.2
    return {"1": 0, "2": 2}[i]


def diffusers_dit_view(header: dict) -> dict:
    """{official key: {"dtype", "shape", "op": id|cat|swap, "srcs": (diffusers keys)}}; unknown keys are an error."""
    t = _tensors(header)
    view: dict = {}

    def put(ok, op, srcs):
        if ok in view:
            raise ValueError(f"two diffusers keys map to {ok}")
        shapes = [t[s]["shape"] for s in srcs]
        dts = {t[s]["dtype"] for s in srcs}
        if len(dts) != 1:
            raise ValueError(f"{ok}: mixed dtypes {dts}")
        shape = list(shapes[0])
        if op == "cat":
            if any(s != shapes[0] for s in shapes):
                raise ValueError(f"{ok}: q/k/v shapes differ {shapes}")
            shape[0] *= len(srcs)
        if op == "swap" and shape[0] % 2:
            raise ValueError(f"{ok}: odd leading dim {shape}")
        view[ok] = {"dtype": dts.pop(), "shape": shape, "op": op, "srcs": tuple(srcs)}

    for k in t:
        m = _D_REFINER_QKV.match(k)
        if m:
            if m[2] == "q":
                pre = k[:k.index(".attn.to_q.")]
                put(f"txt_in.individual_token_refiner.blocks.{m[1]}.self_attn_qkv.{m[3]}", "cat",
                    [f"{pre}.attn.to_{p}.{m[3]}" for p in "qkv"])
            continue
        m = _D_NORM_OUT.match(k)
        if m:
            put(f"final_layer.adaLN_modulation.1.{m[1]}", "swap", [k])
            continue
        for pat, fn in _D2O_DIT:
            m = re.match(pat, k)
            if m:
                put(fn(m), "id", [k])
                break
        else:
            raise ValueError(f"unmapped diffusers HunyuanVideo-1.5 transformer key {k}")
    return view


_D2O_VAE = [
    (r"^(encoder|decoder)\.(conv_in|conv_out)\.conv\.(weight|bias)$", lambda m: f"{m[1]}.{m[2]}.conv.{m[3]}"),
    (r"^(encoder|decoder)\.norm_out\.gamma$", lambda m: f"{m[1]}.norm_out.gamma"),
    (r"^encoder\.down_blocks\.(\d+)\.resnets\.(\d+)\.(.+)$", lambda m: f"encoder.down.{m[1]}.block.{m[2]}.{m[3]}"),
    (r"^encoder\.down_blocks\.(\d+)\.downsamplers\.0\.(.+)$", lambda m: f"encoder.down.{m[1]}.downsample.{m[2]}"),
    (r"^decoder\.up_blocks\.(\d+)\.resnets\.(\d+)\.(.+)$", lambda m: f"decoder.up.{m[1]}.block.{m[2]}.{m[3]}"),
    (r"^decoder\.up_blocks\.(\d+)\.upsamplers\.0\.(.+)$", lambda m: f"decoder.up.{m[1]}.upsample.{m[2]}"),
    (r"^(encoder|decoder)\.mid_block\.resnets\.([01])\.(.+)$", lambda m: f"{m[1]}.mid.block_{int(m[2]) + 1}.{m[3]}"),
    (r"^(encoder|decoder)\.mid_block\.attentions\.0\.to_([qkv])\.(weight|bias)$", lambda m: f"{m[1]}.mid.attn_1.{m[2]}.{m[3]}"),
    (r"^(encoder|decoder)\.mid_block\.attentions\.0\.(norm\.gamma|proj_out\.weight|proj_out\.bias)$",
     lambda m: f"{m[1]}.mid.attn_1.{m[2]}"),
]


def diffusers_vae_view(header: dict) -> dict:
    t = _tensors(header)
    view: dict = {}
    for k, v in t.items():
        for pat, fn in _D2O_VAE:
            m = re.match(pat, k)
            if m:
                ok = fn(m)
                if ok in view:
                    raise ValueError(f"two diffusers VAE keys map to {ok}")
                view[ok] = {"dtype": v["dtype"], "shape": list(v["shape"]), "op": "id", "srcs": (k,)}
                break
        else:
            raise ValueError(f"unmapped diffusers HunyuanVideo-1.5 VAE key {k}")
    return view


def _check_unique(plan: list[Entry]) -> None:
    keys = [e.key for e in plan]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate output keys")


def plan_bytes(plan: list[Entry]) -> int:
    n = 0
    for e in plan:
        c = 1
        for s in e.shape:
            c *= s
        n += c * DTYPE_SIZE[e.dtype]
    return n


# ------------------------------------------------------------------ streaming safetensors writer
class StreamWriter:
    """Writes a safetensors file whose header is computed up front; tensors are appended one at a time in plan order."""

    def __init__(self, path, entries: list[tuple[str, str, tuple]], metadata: dict | None = None,
                 drop_cache_every: int = 2 << 30):
        hdr: dict = {}
        if metadata:
            hdr["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
        off = 0
        self.order = []
        for key, dt, shape in entries:
            c = 1
            for s in shape:
                c *= s
            nb = c * DTYPE_SIZE[dt]
            hdr[key] = {"dtype": dt, "shape": list(shape), "data_offsets": [off, off + nb]}
            self.order.append((key, dt, tuple(shape), nb))
            off += nb
        h = json.dumps(hdr, separators = (",", ":")).encode()
        h += b" " * ((8 - len(h) % 8) % 8)
        self.total = off
        self.path = str(path)
        self.f = open(self.path, "wb")
        self.f.write(struct.pack("<Q", len(h)))
        self.f.write(h)
        self.i = 0
        self.since = 0
        self.every = drop_cache_every

    def write(self, key: str, tensor) -> None:
        import torch

        k, dt, shape, nb = self.order[self.i]
        if key != k:
            raise ValueError(f"write order: got {key}, expected {k}")
        if tuple(tensor.shape) != shape or _st_dtype(tensor.dtype) != dt:
            raise ValueError(f"{key}: got {tuple(tensor.shape)} {tensor.dtype}, planned {shape} {dt}")
        b = tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy()
        if b.nbytes != nb:
            raise ValueError(f"{key}: {b.nbytes} bytes, planned {nb}")
        self.f.write(memoryview(b))
        self.i += 1
        self.since += nb
        if self.since >= self.every:  # keep written pages out of the page cache charged to this process
            self.f.flush()
            os.fdatasync(self.f.fileno())
            os.posix_fadvise(self.f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            self.since = 0

    def close(self) -> None:
        if self.i != len(self.order):
            self.f.close()
            raise ValueError(f"{self.path}: wrote {self.i} of {len(self.order)} tensors")
        self.f.flush()
        os.fsync(self.f.fileno())
        self.f.close()


def _st_dtype(dt) -> str:
    import torch

    return {torch.float32: "F32", torch.float16: "F16", torch.bfloat16: "BF16", torch.float8_e4m3fn: "F8_E4M3",
            torch.float8_e5m2: "F8_E5M2", torch.uint8: "U8", torch.int64: "I64", torch.int32: "I32",
            torch.bool: "BOOL", torch.float64: "F64", torch.int8: "I8", torch.int16: "I16"}[dt]


def _torch_dtype(s: str):
    import torch

    return {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16, "F8_E4M3": torch.float8_e4m3fn,
            "F8_E5M2": torch.float8_e5m2, "U8": torch.uint8, "I64": torch.int64, "I32": torch.int32, "BOOL": torch.bool,
            "F64": torch.float64, "I8": torch.int8, "I16": torch.int16}[s]


def quant_fp8(w):
    import torch

    wf = w.float()
    scale = (wf.abs().amax() / FP8_MAX).clamp(min = 1e-12)
    q = (wf / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q, scale.reshape(()).to(torch.float32)


class Reader:
    """Reads one tensor at a time from safetensors files with pread (no mmap, so read pages can be dropped from the
    page cache and never count toward this process's RSS)."""

    def __init__(self, paths: list):
        self.where: dict = {}
        self.fds: dict = {}
        for p in paths:
            p = str(p)
            with open(p, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(n))
            for k, v in _tensors(hdr).items():
                self.where[k] = (p, 8 + n + v["data_offsets"][0], v["data_offsets"][1] - v["data_offsets"][0],
                                 v["dtype"], tuple(v["shape"]))
            self.fds[p] = os.open(p, os.O_RDONLY)
        self.since = 0

    def get(self, k: str):
        import torch

        p, off, nb, dt, shape = self.where[k]
        buf = bytearray(nb)
        mv = memoryview(buf)
        got = 0
        while got < nb:
            r = os.preadv(self.fds[p], [mv[got:]], off + got)
            if r <= 0:
                raise IOError(f"short read {p} {k}")
            got += r
        self.since += nb
        if self.since >= 2 << 30:
            for fd in self.fds.values():
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            self.since = 0
        if nb == 0:
            return torch.empty(shape, dtype = _torch_dtype(dt))
        return torch.frombuffer(buf, dtype = _torch_dtype(dt)).reshape(shape)

    def close(self) -> None:
        for fd in self.fds.values():
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
        self.fds = {}


class ViewReader:
    """Serves OFFICIAL keys from a diffusers source through a diffusers_*_view (exact inverse of diffusers' converter)."""

    def __init__(self, paths: list, view: dict):
        self.r = Reader(paths)
        self.view = view

    def get(self, k: str):
        import torch

        v = self.view[k]
        if v["op"] == "id":
            return self.r.get(v["srcs"][0])
        if v["op"] == "cat":
            return torch.cat([self.r.get(s) for s in v["srcs"]], 0)
        if v["op"] == "swap":
            a, b = self.r.get(v["srcs"][0]).chunk(2, dim = 0)
            return torch.cat([b, a], 0)
        raise ValueError(v["op"])

    def close(self) -> None:
        self.r.close()


def convert(src_paths: list, plan: list[Entry], out_path, metadata: dict | None = None, view: dict | None = None) -> dict:
    """Stream src safetensors -> out per plan, one tensor in memory at a time. Returns stats.
    view: official key -> diffusers source spec (when the source is the diffusers repo)."""
    import torch

    src = ViewReader(src_paths, view) if view is not None else Reader(src_paths)
    get = src.get
    w = StreamWriter(out_path, [(e.key, e.dtype, e.shape) for e in plan], metadata)
    pending: dict = {}
    nfp8 = 0
    t0 = time.time()
    try:
        for e in plan:
            if e.op == "cast":
                t = get(e.srcs[0])
                t = t.to(_torch_dtype(e.dtype)) if t.is_floating_point() else t
            elif e.op == "cat":
                t = torch.cat([get(s).to(_torch_dtype(e.dtype)) for s in e.srcs], 0)
            elif e.op == "fp8":
                t, pending[e.srcs[0]] = quant_fp8(get(e.srcs[0]))
                nfp8 += 1
            elif e.op == "fp8_scale":
                t = pending.pop(e.srcs[0])
            elif e.op == "marker":
                t = torch.zeros(e.shape, dtype = _torch_dtype(e.dtype))
            else:
                raise ValueError(e.op)
            if e.dtype in ("F16", "BF16") and t.is_floating_point() and not torch.isfinite(t).all():
                raise ValueError(f"{e.key}: non-finite after cast to {e.dtype}")
            w.write(e.key, t)
            del t
        w.close()
    finally:
        src.close()
    hdr = _tensors(read_header(out_path))  # header check: exactly the plan
    if set(hdr) != {e.key for e in plan} or any(tuple(hdr[e.key]["shape"]) != e.shape or hdr[e.key]["dtype"] != e.dtype
                                                for e in plan):
        raise SystemExit(f"{out_path}: written header differs from plan")
    return {"tensors": len(plan), "fp8_linears": nfp8, "seconds": round(time.time() - t0, 1)}


# ------------------------------------------------------------------ fetch + manifest
USE_CACHE = False


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 24)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def repo_files(repo: str, rev: str | None = None) -> tuple[str, dict]:
    from huggingface_hub import HfApi

    assert_official(repo)
    info = HfApi().model_info(repo, revision = rev, files_metadata = True)
    return info.sha, {s.rfilename: (s.lfs.sha256 if s.lfs else None) for s in info.siblings}


def fetch(repo: str, rev: str, name: str, src: Path, shared: bool = False) -> Path:
    """shared: read through the HF cache (HF_HOME / --hf-home) and reuse whatever snapshot is already there (e.g. the
    copy Studio downloaded) before downloading the pinned revision."""
    from huggingface_hub import hf_hub_download

    assert_official(repo)
    t = time.time()
    if shared:
        try:
            p = Path(hf_hub_download(_ext(repo), _ext(name), local_files_only = True))
            print(f"  cached {repo}/{name} {p.stat().st_size / 1e9:.2f} GB ({p})", flush = True)
            return p
        except Exception:  # not in the cache yet
            p = Path(hf_hub_download(_ext(repo), _ext(name), revision = rev))
    elif USE_CACHE:
        p = Path(hf_hub_download(_ext(repo), _ext(name), revision = rev))
    else:
        p = Path(hf_hub_download(_ext(repo), _ext(name), revision = rev, local_dir = str(src / repo.replace("/", "__"))))
    print(f"  fetched {repo}/{name} {p.stat().st_size / 1e9:.2f} GB in {time.time() - t:.0f}s", flush = True)
    return p


def snapshot_rev(p: Path) -> str | None:
    parts = Path(p).parts
    return parts[parts.index("snapshots") + 1] if "snapshots" in parts else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required = True)
    ap.add_argument("--src", required = True, help = "download dir for sources, or 'cache' (HF cache, never deleted)")
    ap.add_argument("--only", default = "dit,vae,te,byt5")
    ap.add_argument("--te", default = "fp8", choices = ["fp8", "bf16"])
    ap.add_argument("--dit-src", default = "tencent", choices = ["tencent", "diffusers"],
                    help = f"diffusers = {HVD_REPO} transformer/ (shared HF cache, never deleted)")
    ap.add_argument("--vae-src", default = "tencent", choices = ["tencent", "diffusers"],
                    help = f"diffusers = {HVD_REPO} vae/ (shared HF cache, never deleted)")
    ap.add_argument("--hf-home", default = None, help = "HF cache root to share (sets HF_HOME, HF_HUB_CACHE, HF_XET_CACHE)")
    ap.add_argument("--keep-src", action = "store_true")
    ap.add_argument("--dit-dtype", default = "fp16", choices = ["fp16", "bf16"],
                    help = "bf16: the DiT as hunyuanvideo1.5_480p_t2v_bf16_official.safetensors (accuracy reference)")
    a = ap.parse_args()
    if a.hf_home:  # before huggingface_hub is imported (it freezes these at import)
        os.environ["HF_HOME"] = a.hf_home
        os.environ["HF_HUB_CACHE"] = os.path.join(a.hf_home, "hub")
        os.environ["HF_XET_CACHE"] = os.path.join(a.hf_home, "xet")
    global USE_CACHE
    USE_CACHE = a.src == "cache"
    if USE_CACHE:
        a.keep_src = True
    only = [x.strip() for x in a.only.split(",") if x.strip()]
    bad = set(only) - {"dit", "vae", "te", "byt5"}
    if bad:
        raise SystemExit(f"--only: unknown {sorted(bad)}")
    # the diffusers repo is Studio's copy: once it is a DiT / VAE source, every file of it is read through the shared
    # HF cache and never deleted (ByT5 text_encoder_2 included)
    hvd_shared = (a.dit_src == "diffusers" and "dit" in only) or (a.vae_src == "diffusers" and "vae" in only)
    out, src = Path(a.out), Path(a.src)
    out.mkdir(parents = True, exist_ok = True)
    if not USE_CACHE:
        src.mkdir(parents = True, exist_ok = True)
    meta = {"format": "pt"}
    jobs = []  # (role, repo, source files | listing prefix, output name, plan fn, format label, view fn)
    if "dit" in only:
        dname = OUT["dit"] if a.dit_dtype == "fp16" else OUT["dit"].replace("_fp16_", "_bf16_")
        dplan = dit_plan if a.dit_dtype == "fp16" else (lambda h: dit_plan(h, "BF16"))
        if a.dit_src == "diffusers":
            jobs.append(("dit", HVD_REPO, "transformer/", dname, dplan, a.dit_dtype, diffusers_dit_view))
        else:
            jobs.append(("dit", HV_REPO, [DIT_FILE], dname, dplan, a.dit_dtype, None))
    if "vae" in only:
        if a.vae_src == "diffusers":
            jobs.append(("vae", HVD_REPO, "vae/", OUT["vae"], vae_plan, "fp16", diffusers_vae_view))
        else:
            jobs.append(("vae", HV_REPO, [VAE_FILE], OUT["vae"], vae_plan, "fp16", None))
    if "te" in only:
        jobs.append(("te", QWEN_REPO, "", OUT["te_" + a.te], lambda h, m = a.te: te_plan(h, m),
                     "scaled fp8 (per-tensor, full-precision matmul)" if a.te == "fp8" else "bf16", None))
    if "byt5" in only:
        jobs.append(("byt5", BYT5_REPO, [BYT5_FILE], OUT["byt5"], byt5_plan, "fp16", None))
    mpath = out / "manifest.json"
    man = {"group": "hunyuanvideo15_480p_t2v", "outputs": []}
    revs: dict = {}
    t0 = time.time()
    for role, repo, names, oname, planfn, fmt, viewfn in jobs:
        o = out / oname
        if o.exists():
            print("exists", o)
            continue
        shared = hvd_shared and repo == HVD_REPO
        if repo not in revs:
            revs[repo] = repo_files(repo)
        rev, files = revs[repo]
        if isinstance(names, str):  # every weight shard under a prefix
            pre = names
            names = sorted(n for n in files if n.startswith(pre) and n.endswith(".safetensors") and "/" not in n[len(pre):])
            if not names:
                raise SystemExit(f"no safetensors under {repo}/{pre}")
        paths = [fetch(repo, rev, n, src, shared = shared) for n in names]
        used = {snapshot_rev(p) for p in paths} - {None}
        if shared and used and used != {rev}:  # Studio's cached snapshot: record what was actually read
            if len(used) > 1:
                raise SystemExit(f"{repo}: shards come from different cached snapshots {used}")
            urev = used.pop()
            revs[repo] = repo_files(repo, urev)
            rev, files = revs[repo]
        header = {}
        for p in paths:
            header.update(_tensors(read_header(p)))
        view = viewfn(header) if viewfn else None
        plan = planfn({k: {"dtype": v["dtype"], "shape": v["shape"]} for k, v in view.items()} if view else header)
        tmp = o.with_name(o.name + ".partial")
        st = convert(paths, plan, tmp, meta, view = view)
        os.replace(tmp, o)
        if not a.keep_src and not shared:
            for p in paths:
                p.unlink(missing_ok = True)
        man["outputs"].append({"file": oname, "role": role, "format": fmt, **st, "source_repo": repo,
                               "source_revision": rev, "source_layout": "diffusers" if view else "original",
                               "source_kept": bool(a.keep_src or shared),
                               "sources": [{"file": n, "lfs_sha256": files.get(n)} for n in names],
                               "sha256": sha256(o), "bytes": o.stat().st_size})
        print(f"built {oname} {o.stat().st_size / 1e9:.2f} GB ({st}) t={time.time() - t0:.0f}s", flush = True)
    old = json.loads(mpath.read_text())["outputs"] if mpath.exists() else []
    have = {x["file"] for x in man["outputs"]}
    man["outputs"] = [x for x in old if x["file"] not in have] + man["outputs"]
    mpath.write_text(json.dumps(man, indent = 1))
    if not a.keep_src:
        for repo in (HV_REPO, QWEN_REPO, HVD_REPO):
            if not (hvd_shared and repo == HVD_REPO):
                shutil.rmtree(src / repo.replace("/", "__"), ignore_errors = True)
    print("MANIFEST", json.dumps(man), flush = True)
    print(f"CONVERT_DONE hv15 {time.time() - t0:.0f}s", flush = True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
