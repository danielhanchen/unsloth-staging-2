#!/usr/bin/env python3
"""Build baseline-UI-loadable Wan files from the OFFICIAL Wan-AI repos (never the Bline-Org repack).

The baseline UI's Wan DiT and VAE are ports of Wan-AI's original code, so the original-format checkpoints already use its key
names; its UMT5 text encoder uses the Hugging Face T5 layout, which is what Wan-AI's diffusers repos ship. So:
  dit  original diffusion_pytorch_model*.safetensors, cast to fp16 (the template's wan2.1_t2v_1.3B_fp16 /
       wan2.2_ti2v_5B_fp16 are fp16) or to scaled fp8 (A14B experts, the template's *_fp8_scaled)
  vae  Wan2.1_VAE.pth / Wan2.2_VAE.pth (torch.load weights_only) -> safetensors, dtype unchanged
  te   Wan-AI/Wan2.2-TI2V-5B-Diffusers text_encoder (UMT5-XXL, bf16, HF layout; the same weights for every Wan base)
       -> legacy "scaled fp8" (per-tensor scale_weight + a scaled_fp8 marker, what bline.utils.convert_old_quants
       reads), the format of the template's umt5_xxl_fp8_e4m3fn_scaled; and/or a single bf16 file
Every output is listed in manifest.json with its source repo, revision, source file LFS sha256 and its own sha256.
Sources are deleted after conversion unless --keep-src.

Usage: python diffusion_bench/builders/vid_convert_bline.py --group wan21_13b|wan22_5b|wan22_a14b --out DIR --src DIR
           [--te fp8|bf16|both|none] [--only dit,vae,te] [--expert high|low] [--dit-dtype template|bf16|fp16]
Ported from scripts/vidcmp_nov5/convert_bline.py. Added for single-file builds (builders.build_file): --only, --expert
(one A14B expert) and --dit-dtype fp16 (the A14B experts as plain fp16, the template's *_fp16 files).
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import time
from pathlib import Path

import torch
from huggingface_hub import HfApi, hf_hub_download
try:  # extnames-aware safetensors (builders/_st.py); as a script or as builders.<module>
    from _st import safe_open
except ImportError:
    from builders._st import safe_open
import extnames as X  # noqa: E402 - real download names in a public copy
try:  # extnames-aware safetensors (builders/_st.py); as a script or as builders.<module>
    from _st import save_file
except ImportError:
    from builders._st import save_file

TE_REPO = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
GROUPS = {
    "wan21_13b": {"repo": "Wan-AI/Wan2.1-T2V-1.3B",
                  "dit": [("diffusion_pytorch_model.safetensors", "wan2.1_t2v_1.3B_fp16_official.safetensors", "fp16")],
                  "vae": ("Wan2.1_VAE.pth", "wan_2.1_vae_official.safetensors")},
    "wan22_5b": {"repo": "Wan-AI/Wan2.2-TI2V-5B",
                 "dit": [("diffusion_pytorch_model-0000{i}-of-00003.safetensors", "wan2.2_ti2v_5B_fp16_official.safetensors",
                          "fp16")],
                 "vae": ("Wan2.2_VAE.pth", "wan2.2_vae_official.safetensors")},
    "wan22_a14b": {"repo": "Wan-AI/Wan2.2-T2V-A14B",
                   "dit": [("high_noise_model/diffusion_pytorch_model-0000{i}-of-00006.safetensors",
                            "wan2.2_t2v_high_noise_14B_fp8_scaled_official.safetensors", "fp8"),
                           ("low_noise_model/diffusion_pytorch_model-0000{i}-of-00006.safetensors",
                            "wan2.2_t2v_low_noise_14B_fp8_scaled_official.safetensors", "fp8")],
                   "vae": ("Wan2.1_VAE.pth", "wan_2.1_vae_official.safetensors")},
}
FP8_MAX = 448.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 24)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def repo_files(api: HfApi, repo: str) -> tuple[str, dict]:
    info = api.model_info(repo, files_metadata = True)
    return info.sha, {s.rfilename: (s.lfs.sha256 if s.lfs else None) for s in info.siblings}


def expand(pattern: str, files: dict) -> list[str]:
    if "{i}" not in pattern:
        return [pattern]
    out = [pattern.format(i = i) for i in range(1, 10) if pattern.format(i = i) in files]
    if not out:
        raise FileNotFoundError(pattern)
    return out


USE_CACHE = False  # --src cache: read through the shared HF cache (local smoke), never delete sources


def fetch(repo: str, rev: str, name: str, src: Path) -> Path:
    t = time.time()
    if USE_CACHE:
        p = Path(hf_hub_download(X.ext(repo), X.ext(name), revision = rev))
    else:
        p = Path(hf_hub_download(X.ext(repo), X.ext(name), revision = rev, local_dir = str(src / repo.replace("/", "__"))))
    print(f"  fetched {repo}/{name} {p.stat().st_size / 1e9:.2f} GB in {time.time() - t:.0f}s", flush = True)
    return p


def quant_fp8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    wf = w.float()
    scale = (wf.abs().amax() / FP8_MAX).clamp(min = 1e-12)
    q = (wf / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q, scale.reshape(()).to(torch.float32)


def is_dit_linear(k: str, t: torch.Tensor) -> bool:
    # the 2-D projection weights inside the transformer blocks (attention q/k/v/o, cross-attention, ffn); embeddings,
    # head, modulation, norms stay in high precision as in the baseline UI's fp8_scaled repacks
    return t.ndim == 2 and k.startswith("blocks.") and k.endswith(".weight") and ".norm" not in k


def is_te_linear(k: str, t: torch.Tensor) -> bool:
    return (t.ndim == 2 and k.startswith("encoder.block.") and k.endswith(".weight")
            and ("SelfAttention." in k or "DenseReluDense." in k) and "relative_attention_bias" not in k)


def convert_dit(paths: list, out: Path, mode: str, delete: bool) -> dict:
    # paths: Paths, or zero-arg callables that fetch one shard: a shard is fetched, converted and deleted before the
    # next one is fetched, so the source never sits on disk whole (the A14B experts are 57 GB fp32 each)
    sd: dict = {}
    nq = 0
    for p in paths:
        if callable(p):
            p = p()
        with safe_open(str(p), "pt") as f:
            for k in f.keys():
                t = f.get_tensor(k)
                if mode == "fp8" and is_dit_linear(k, t):
                    q, s = quant_fp8(t)
                    sd[k] = q
                    sd[k[:-len(".weight")] + ".scale_weight"] = s
                    nq += 1
                elif t.is_floating_point():
                    sd[k] = t.to(torch.float16 if mode == "fp16" else torch.bfloat16).contiguous()
                else:
                    sd[k] = t
                del t
        if delete:
            p.unlink()
        gc.collect()
    if mode == "fp8":
        # nelement 2 -> full_precision_matrix_mult (dequantize, then the compute dtype matmul)
        sd["scaled_fp8"] = torch.zeros(2, dtype = torch.float8_e4m3fn)
    save_file(sd, str(out))
    n = len(sd)
    del sd
    gc.collect()
    return {"tensors": n, "fp8_linears": nq}


def convert_te(paths: list[Path], out_dir: Path, modes: list[str], spiece: Path) -> list[dict]:
    raw: dict = {}
    for p in paths:
        with safe_open(str(p), "pt") as f:
            for k in f.keys():
                raw[k] = f.get_tensor(k)
    # the baseline UI's UMT5 tokenizer reads the sentencepiece model from the checkpoint itself (tokenizer_data["spiece_model"],
    # bline/text_encoders/wan.py); the official tokenizer/spiece.model bytes as a uint8 tensor
    raw["spiece_model"] = torch.frombuffer(bytearray(spiece.read_bytes()), dtype = torch.uint8).clone()
    res = []
    if "bf16" in modes:
        o = out_dir / "umt5_xxl_bf16_official.safetensors"
        save_file({k: (v.to(torch.bfloat16) if v.is_floating_point() else v).contiguous() for k, v in raw.items()}, str(o))
        res.append({"file": o.name, "role": "te", "format": "bf16"})
    if "fp8" in modes:
        sd: dict = {}
        nq = 0
        for k, v in raw.items():
            if is_te_linear(k, v):
                q, s = quant_fp8(v)
                sd[k] = q
                sd[k[:-len(".weight")] + ".scale_weight"] = s
                nq += 1
            else:
                sd[k] = v.to(torch.bfloat16) if v.is_floating_point() else v
        sd["scaled_fp8"] = torch.zeros(2, dtype = torch.float8_e4m3fn)
        o = out_dir / "umt5_xxl_fp8_e4m3fn_scaled_official.safetensors"
        save_file(sd, str(o))
        res.append({"file": o.name, "role": "te", "format": "scaled fp8 (per-tensor, full-precision matmul)",
                    "fp8_linears": nq})
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required = True, choices = sorted(GROUPS))
    ap.add_argument("--out", required = True)
    ap.add_argument("--src", required = True)
    ap.add_argument("--te", default = "fp8", help = "fp8 | bf16 | both | none")
    ap.add_argument("--keep-src", action = "store_true")
    ap.add_argument("--only", default = "dit,vae,te", help = "subset of dit,vae,te")
    ap.add_argument("--expert", default = None, choices = ["high", "low"], help = "A14B: build only this expert")
    ap.add_argument("--dit-dtype", default = "template", choices = ["template", "bf16", "fp16"],
                    help = "bf16: every DiT as a plain bf16 file (*_bf16_official), the accuracy reference")
    ap.add_argument("--skip-vae", action = "store_true")
    a = ap.parse_args()
    global USE_CACHE
    USE_CACHE = a.src == "cache"
    if USE_CACHE:
        a.keep_src = True
    out, src = Path(a.out), Path(a.src)
    out.mkdir(parents = True, exist_ok = True)
    if not USE_CACHE:
        src.mkdir(parents = True, exist_ok = True)
    api = HfApi()
    g = GROUPS[a.group]
    rev, files = repo_files(api, g["repo"])
    man = {"group": a.group, "outputs": []}
    mpath = out / f"manifest_{a.group}.json"
    t0 = time.time()
    only = set(a.only.split(","))
    for pattern, oname, mode in g["dit"] if "dit" in only else []:
        if a.expert and f"{a.expert}_noise" not in oname:
            continue
        if a.dit_dtype == "bf16":
            oname = oname.replace("_fp16_official", "_bf16_official").replace("_fp8_scaled_official", "_bf16_official")
            mode = "bf16"
        elif a.dit_dtype == "fp16":
            oname = oname.replace("_fp8_scaled_official", "_fp16_official")
            mode = "fp16"
        o = out / oname
        if o.exists():
            print("exists", o)
            continue
        names = expand(pattern, files)
        paths = [(lambda n = n: fetch(g["repo"], rev, n, src)) for n in names]
        st = convert_dit(paths, o, mode, delete = not a.keep_src)
        man["outputs"].append({"file": oname, "role": "dit", "format": mode, **st, "source_repo": g["repo"],
                               "source_revision": rev, "sources": [{"file": n, "lfs_sha256": files.get(n)} for n in names],
                               "sha256": sha256(o), "bytes": o.stat().st_size})
        print(f"built {oname} {o.stat().st_size / 1e9:.2f} GB ({st}) t={time.time() - t0:.0f}s", flush = True)
    vname, voname = g["vae"]
    vo = out / voname
    if not vo.exists() and not a.skip_vae and "vae" in only:
        vp = fetch(g["repo"], rev, vname, src)
        sd = torch.load(str(vp), map_location = "cpu", weights_only = True)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        save_file({k: v.contiguous() for k, v in sd.items()}, str(vo))
        if not a.keep_src:
            vp.unlink()
        man["outputs"].append({"file": voname, "role": "vae", "format": str(next(iter(sd.values())).dtype),
                               "source_repo": g["repo"], "source_revision": rev,
                               "sources": [{"file": vname, "lfs_sha256": files.get(vname)}],
                               "sha256": sha256(vo), "bytes": vo.stat().st_size})
    modes = {"both": ["fp8", "bf16"], "none": []}.get(a.te, [a.te]) if "te" in only else []
    modes = [m for m in modes if not (out / {"fp8": "umt5_xxl_fp8_e4m3fn_scaled_official.safetensors",
                                             "bf16": "umt5_xxl_bf16_official.safetensors"}[m]).exists()]
    if modes:
        trev, tfiles = repo_files(api, TE_REPO)
        names = sorted(n for n in tfiles if n.startswith("text_encoder/") and n.endswith(".safetensors"))
        paths = [fetch(TE_REPO, trev, n, src) for n in names]
        spiece = fetch(TE_REPO, trev, "tokenizer/spiece.model", src)
        names = names + ["tokenizer/spiece.model"]
        for r in convert_te(paths, out, modes, spiece):
            o = out / r["file"]
            man["outputs"].append({**r, "source_repo": TE_REPO, "source_revision": trev,
                                   "sources": [{"file": n, "lfs_sha256": tfiles.get(n)} for n in names],
                                   "sha256": sha256(o), "bytes": o.stat().st_size})
        if not a.keep_src:
            for p in paths:
                p.unlink(missing_ok = True)
    old = json.loads(mpath.read_text())["outputs"] if mpath.exists() else []
    have = {o["file"] for o in man["outputs"]}
    man["outputs"] = [o for o in old if o["file"] not in have] + man["outputs"]
    mpath.write_text(json.dumps(man, indent = 1))
    if not a.keep_src:
        shutil.rmtree(src / g["repo"].replace("/", "__"), ignore_errors = True)
        shutil.rmtree(src / TE_REPO.replace("/", "__"), ignore_errors = True)
    print("MANIFEST", json.dumps(man), flush = True)
    print(f"CONVERT_DONE {a.group} {time.time() - t0:.0f}s", flush = True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
