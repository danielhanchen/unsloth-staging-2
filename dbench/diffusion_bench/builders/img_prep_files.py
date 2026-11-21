#!/usr/bin/env python3
"""baseline-UI-loadable files from OFFICIAL repos, built on the VM. Never touches Bline-Org / blineanonymous HF repos.

  python diffusion_bench/builders/img_prep_files.py --groups sdxl,flux --out DIR [--manifest files.json] [--roles te,vae]

Ported from scripts/imgcmp_nov5/prep_files.py. Added: --roles (build only these roles of each group; default all),
--manifest defaults to OUT/prep_manifest.json, and out_name(group, role) = the file a role lands in (LINK_NAME / MERGE).

Each source file is checked against the official repo's LFS sha256 (pinned below, read from the official repo's
metadata 2026-10-24). FLUX.1-schnell is gated on black-forest-labs, so its bytes come from the ungated byte-identical
mirror unsloth/FLUX.1-schnell and are accepted only when the sha256 equals the official one. Text-encoder shards are
merged into one safetensors file here; the merged file's own sha256 is recorded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

# group -> role -> list of (official repo, path, fetch repo, official sha256)
OFFICIAL = {
    "sdxl": {
        "checkpoint": [("stabilityai/stable-diffusion-xl-base-1.0", "sd_xl_base_1.0.safetensors",
                        "stabilityai/stable-diffusion-xl-base-1.0",
                        "31e35c80fc4829d14f90153f4c74cd59c90b779f6afe05a74cd6120b893f7e5b")],
    },
    # SDXL + one public SDXL LoRA (nerijs/pixel-art-xl, kohya format: diffusers and baseline-UI both load it): the
    # LoRA weight-change scenario (#16310)
    "sdxll": {
        "checkpoint": [("stabilityai/stable-diffusion-xl-base-1.0", "sd_xl_base_1.0.safetensors",
                        "stabilityai/stable-diffusion-xl-base-1.0",
                        "31e35c80fc4829d14f90153f4c74cd59c90b779f6afe05a74cd6120b893f7e5b")],
        "lora": [("nerijs/pixel-art-xl", "pixel-art-xl.safetensors", "nerijs/pixel-art-xl",
                  "4234637cb80c998f41e348e6a6cb6bc20d8d038b2b0f256b6129b3b5e353eef7")],
    },
    "flux": {
        "dit": [("black-forest-labs/FLUX.1-schnell", "flux1-schnell.safetensors", "unsloth/FLUX.1-schnell",
                 "9403429e0052277ac2a87ad800adece5481eecefd9ed334e1f348723621d2a0a")],
        "vae": [("black-forest-labs/FLUX.1-schnell", "ae.safetensors", "unsloth/FLUX.1-schnell",
                 "afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38")],
        "te_clip_l": [("black-forest-labs/FLUX.1-schnell", "text_encoder/model.safetensors", "unsloth/FLUX.1-schnell",
                       "893d67a23f4693ed42cdab4cbad7fe3e727cf59609c40da28a46b5470f9ed082")],
        "te_t5": [("black-forest-labs/FLUX.1-schnell", "text_encoder_2/model-00001-of-00002.safetensors",
                   "unsloth/FLUX.1-schnell", "ec87bffd1923e8b2774a6d240c922a41f6143081d52cf83b8fe39e9d838c893e"),
                  ("black-forest-labs/FLUX.1-schnell", "text_encoder_2/model-00002-of-00002.safetensors",
                   "unsloth/FLUX.1-schnell", "a5640855b301fcdbceddfa90ae8066cd9414aff020552a201a255ecf2059da00")],
    },
    "klein": {
        "dit": [("black-forest-labs/FLUX.2-klein-4B", "flux-2-klein-4b.safetensors", "black-forest-labs/FLUX.2-klein-4B",
                 "ec3d4e733a771f61c052fb4856c48b336c55eaf2c65487c2a1faeb9bbda7a343")],
        "vae": [("black-forest-labs/FLUX.2-klein-4B", "vae/diffusion_pytorch_model.safetensors",
                 "black-forest-labs/FLUX.2-klein-4B", "ca70d2202afe6415bdbcb8793ba8cd99fd159cfe6192381504d6c4d3036e0f04")],
        "te": [("black-forest-labs/FLUX.2-klein-4B", "text_encoder/model-00001-of-00002.safetensors",
                "black-forest-labs/FLUX.2-klein-4B", "8c0506e7f4936fa7e26183a4fd8da4e2bdbc5990ba64ae441f965d51228f36ea"),
               ("black-forest-labs/FLUX.2-klein-4B", "text_encoder/model-00002-of-00002.safetensors",
                "black-forest-labs/FLUX.2-klein-4B", "82f2bd839378541b0557bfabaf37c7d3d637071fdcb73302dedd7cf61162ce07")],
    },
    # FLUX.2-klein-9B: gated official repo; bytes from the ungated byte-identical unsloth mirror, sha256 checked. baseline-UI
    # reads BFL's own single-file DiT and the merged Qwen3-8B (its workbox template's flux-2-klein-9b / qwen_3_8b).
    "klein9": {
        "dit": [("black-forest-labs/FLUX.2-klein-9B", "flux-2-klein-9b.safetensors", "unsloth/FLUX.2-klein-9B",
                 "0975d6b77b5f510b99547d6724a208e36527df654e8f6134f59ece3f9f30da58")],
        "vae": [("black-forest-labs/FLUX.2-klein-9B", "vae/diffusion_pytorch_model.safetensors", "unsloth/FLUX.2-klein-9B",
                 "ca70d2202afe6415bdbcb8793ba8cd99fd159cfe6192381504d6c4d3036e0f04")],
        "te": [("black-forest-labs/FLUX.2-klein-9B", f"text_encoder/model-0000{i}-of-00004.safetensors", "unsloth/FLUX.2-klein-9B", h)
               for i, h in ((1, "c0dc64934ae0f730ddc80d99af44968d01a89e8454df07d762096ea1356446bc"),
                            (2, "d58533b468c31caa0222540a8aefddea1d74dc6e1fee928da8556d3d85729d6e"),
                            (3, "74927fec432e050365bf757bb30348f560a44394efac89f492680b7c910b64fd"),
                            (4, "cb73f466fade5716702bda38d4e3b321c9358c39889e46fb9d613fb038bfcb2f"))],
    },
    "q21": {
        "dit": [("Qwen/Qwen-Image-2.1", "transformer/diffusion_pytorch_model-00001-of-00002.safetensors", "unsloth/Qwen-Image-2.1",
                 "9e6bc2d641e67bf277895ea8777141044a38f3edb7101bc469b2961dd7c36b4b"),
                ("Qwen/Qwen-Image-2.1", "transformer/diffusion_pytorch_model-00002-of-00002.safetensors", "unsloth/Qwen-Image-2.1",
                 "3aaf234dcbe128530479735854a346b5e3e66283b7c11db56f836bbd1c13ebaa")],
        "te": [("Qwen/Qwen-Image-2.1", f"text_encoder/model-0000{i}-of-00004.safetensors", "unsloth/Qwen-Image-2.1", h)
               for i, h in ((1, "dde00291b5f7fb92013895310a3da0ddba78674df9f10d505d375243dc01fc6f"),
                            (2, "9047faccc0a6d98496a52d55f27be1c94a9c259d1e283fbea0128d054a948d42"),
                            (3, "8c54187654c0176b73ae73785bf791dc9a14c9df7fb4310083a09d42048cb57e"),
                            (4, "5311532aaaeae3259eb6a7b2c600636be1159adf7ded35f53579f7d0e7d43cdd"))],
        "vae": [("Qwen/Qwen-Image-2.1", "vae/diffusion_pytorch_model.safetensors", "unsloth/Qwen-Image-2.1",
                 "a07a1b7c4ee2966a1b3bdc37de9b4f983d56937e46619f709a80b6e490675417")],
    },
    # gated official repo; bytes from the ungated unsloth mirror, every sha256 checked against the official LFS metadata
    "fluxdev": {
        "dit": [("black-forest-labs/FLUX.1-dev", "flux1-dev.safetensors", "unsloth/FLUX.1-dev",
                 "4610115bb0c89560703c892c59ac2742fa821e60ef5871b33493ba544683abd7")],
        "vae": [("black-forest-labs/FLUX.1-dev", "ae.safetensors", "unsloth/FLUX.1-dev",
                 "afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38")],
        "te_clip_l": [("black-forest-labs/FLUX.1-dev", "text_encoder/model.safetensors", "unsloth/FLUX.1-dev",
                       "893d67a23f4693ed42cdab4cbad7fe3e727cf59609c40da28a46b5470f9ed082")],
        "te_t5": [("black-forest-labs/FLUX.1-dev", "text_encoder_2/model-00001-of-00002.safetensors",
                   "unsloth/FLUX.1-dev", "ec87bffd1923e8b2774a6d240c922a41f6143081d52cf83b8fe39e9d838c893e"),
                  ("black-forest-labs/FLUX.1-dev", "text_encoder_2/model-00002-of-00002.safetensors",
                   "unsloth/FLUX.1-dev", "a5640855b301fcdbceddfa90ae8066cd9414aff020552a201a255ecf2059da00")],
    },
    "q21t": {
        "dit": [("Qwen/Qwen-Image-2.1-Turbo", "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
                 "Qwen/Qwen-Image-2.1-Turbo", "6cccd922767f01694461bdcf1f34ea7b771f3442e4d5e338ea9aa55277431cc0"),
                ("Qwen/Qwen-Image-2.1-Turbo", "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
                 "Qwen/Qwen-Image-2.1-Turbo", "69f53ebb063d2f606bdaff7e22e0e2144f28d978d73dc75e6e4293fc5982eab5")],
        # byte-identical (sha256) to the merge of Qwen-Image-2.1's four text-encoder shards
        "te": [("Qwen/Qwen-Image-2.1-Turbo", "text_encoder/model.safetensors", "Qwen/Qwen-Image-2.1-Turbo",
                "a1eee8233f3063466dd3b06e857b0a2cf13a77c543908374771d51a546beb2e0")],
        "vae": [("Qwen/Qwen-Image-2.1-Turbo", "vae/diffusion_pytorch_model.safetensors", "Qwen/Qwen-Image-2.1-Turbo",
                 "71879ffd5321e6d10c3c87513e2b474b1252efa7f3dec2969214a9bf06a6dd5c")],
    },
    # Z-Image-Turbo: official diffusers repo (fp32 transformer shards, bf16 Qwen3-4B encoder). the baseline UI's UNETLoader
    # converts the diffusers transformer keys itself (bline/model_detection.py convert_diffusers_mmdit, Z-Image branch);
    # the shards are merged and cast to bf16 here, which is what the template's z_image_turbo_bf16 file is. The VAE is
    # the FLUX.1 autoencoder (BFL's ae.safetensors), read from the sha-matching ungated FLUX.1-schnell mirror.
    "zimage": {
        "dit": [("Tongyi-MAI/Z-Image-Turbo", f"transformer/diffusion_pytorch_model-0000{i}-of-00003.safetensors",
                 "Tongyi-MAI/Z-Image-Turbo", h)
                for i, h in ((1, "95facd593e2549e8252acb571c653d57f7ddb7f1060d4e81712f152555a88804"),
                             (2, "a4bbe43ee184a1fb5af4b412d27555f532893bdc3165b1149e304ed82b5d7015"),
                             (3, "aba4e37a590e63210878160a718d916d80398f4e1f78ab6c9b2b2a00d92769fa"))],
        "te": [("Tongyi-MAI/Z-Image-Turbo", f"text_encoder/model-0000{i}-of-00003.safetensors", "Tongyi-MAI/Z-Image-Turbo", h)
               for i, h in ((1, "328a91d3122359d5547f9d79521205bc0a46e1f79a792dfe650e99fc2d651223"),
                            (2, "6cd087b316306a68c562436b5492edbcf6e16c6dba3a1308279caa5a58e21ca5"),
                            (3, "7ca841ee75b9c61267c0c6148fd8d096d3d21b6d3e161256a9b878154f91fc52"))],
        "vae": [("black-forest-labs/FLUX.1-schnell", "ae.safetensors", "unsloth/FLUX.1-schnell",
                 "afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38")],
    },
}
# merged outputs cast to bf16 on the way (the official file is fp32; the baseline UI's template file is the bf16 cast)
MERGE_BF16 = {("zimage", "dit")}
# Qwen-Image-2.1 VAE: diffusers AutoencoderKLWan names -> baseline-UI Wan 2.2 layout (same tensors, renamed), with the
# name map from diffusers' own convert_vae_22 (scripts/convert_wan_to_diffusers.py at this pinned diffusers commit)
DIFFUSERS_CONVERT = ("https://raw.githubusercontent.com/huggingface/diffusers/"
                     "aaf57d03a3edfbd9de457d2872417a92f149a91d/scripts/convert_wan_to_diffusers.py")
# merged outputs: group -> role -> output file name
MERGE = {("flux", "te_t5"): "t5xxl_flux1schnell_merged.safetensors",
         ("q21", "dit"): "qwen_image_2.1_official_dit_merged.safetensors",
         ("q21", "te"): "qwen3vl_8b_q21_official_merged.safetensors",
         ("klein", "te"): "qwen3_4b_klein_merged.safetensors",
         ("klein9", "te"): "qwen3_8b_klein9_merged.safetensors",
         ("q21t", "dit"): "qwen_image_2.1_turbo_official_dit_merged.safetensors",
         ("fluxdev", "te_t5"): "t5xxl_flux1dev_merged.safetensors",
         ("zimage", "dit"): "z_image_turbo_official_dit_bf16.safetensors",
         ("zimage", "te"): "qwen3_4b_zimage_official_merged.safetensors"}
# non-merged roles: the name the blob is linked under (default: the official file's basename)
LINK_NAME = {("klein", "vae"): "flux2_klein_vae_diffusers.safetensors",
             ("klein9", "vae"): "flux2_klein9_vae_diffusers.safetensors",
             ("q21t", "te"): "qwen3vl_8b_q21turbo_official.safetensors",
             ("flux", "te_clip_l"): "clip_l_flux1schnell.safetensors",
             ("fluxdev", "te_clip_l"): "clip_l_flux1dev.safetensors",
             ("zimage", "vae"): "ae_zimage.safetensors",
             ("fluxdev", "vae"): "ae_flux1dev.safetensors",
             ("q21", "vae"): "qwen_image_2.1_vae_official-BlineUI.safetensors",
             ("q21t", "vae"): "qwen_image_2.1_turbo_vae_official-BlineUI.safetensors"}
FORBIDDEN = ("bline-org/", "blineanonymous/")


def out_name(g: str, role: str) -> str:
    """The file name a group's role is written / linked as under --out."""
    if (g, role) in MERGE:
        return MERGE[(g, role)]
    if (g, role) in LINK_NAME:
        return LINK_NAME[(g, role)]
    return Path(OFFICIAL[g][role][0][1]).name


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 24)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def merge(paths, out, bf16 = False):
    import torch
    try:  # extnames-aware safetensors (builders/_st.py); as a script or as builders.<module>
        from _st import safe_open
    except ImportError:
        from builders._st import safe_open
    try:  # extnames-aware safetensors (builders/_st.py); as a script or as builders.<module>
        from _st import save_file
    except ImportError:
        from builders._st import save_file

    tensors = {}
    for p in paths:
        with safe_open(p, "pt") as f:
            for k in f.keys():
                t = f.get_tensor(k)
                tensors[k] = t.to(torch.bfloat16) if bf16 and t.is_floating_point() else t
    save_file(tensors, out, metadata = {"format": "pt"})
    n = len(tensors)
    del tensors
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--manifest", default = None)
    ap.add_argument("--roles", default = None, help = "comma list: build only these roles of each group")
    a = ap.parse_args()
    a.manifest = a.manifest or str(Path(a.out) / "prep_manifest.json")
    only = set(a.roles.split(",")) if a.roles else None
    from huggingface_hub import hf_hub_download

    out = Path(a.out)
    out.mkdir(parents = True, exist_ok = True)
    man = json.loads(Path(a.manifest).read_text()) if Path(a.manifest).exists() else {}
    for g in [x for x in a.groups.split(",") if x]:
        files = {}
        rec = {"files": {}, "sources": []}
        for role, items in OFFICIAL[g].items():
            if only is not None and role not in only:
                continue
            paths = []
            for official, path, fetch, want in items:
                assert not fetch.lower().startswith(FORBIDDEN) and not official.lower().startswith(FORBIDDEN)
                t = time.time()
                sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
                import extnames  # noqa: PLC0415

                p = hf_hub_download(extnames.ext(fetch), extnames.ext(path), token = False)
                dl = time.time() - t
                prev = {(x.get("official_repo"), x.get("path")): x for x in (man.get(g) or {}).get("sources", [])}
                old = prev.get((official, path))
                got = old["sha256"] if old and old.get("match") and old.get("bytes") == os.path.getsize(p) else sha256(p)
                ok = got == want
                rec["sources"].append({"role": role, "official_repo": official, "path": path, "fetched_from": fetch,
                                       "sha256": got, "official_sha256": want, "match": ok,
                                       "bytes": os.path.getsize(p), "download_s": round(dl, 1)})
                print("PREP", g, role, official, path, "via", fetch, "sha256", got[:16], "MATCH" if ok else "MISMATCH",
                      round(os.path.getsize(p) / 1e9, 2), "GB", round(dl, 1), "s", flush = True)
                if not ok:
                    raise SystemExit(f"sha256 mismatch for {official}/{path}")
                paths.append(os.path.realpath(p))
            if (g, role) in MERGE:
                dst = out / MERGE[(g, role)]
                if not dst.exists():
                    t = time.time()
                    n = merge(paths, str(dst), bf16 = (g, role) in MERGE_BF16)
                    print("MERGED", dst, n, "tensors", round(time.time() - t, 1), "s", flush = True)
                oldm = next((x for x in (man.get(g) or {}).get("sources", []) if x.get("merged_file") == dst.name), None)
                msha = oldm["sha256"] if oldm and oldm.get("bytes") == dst.stat().st_size else sha256(str(dst))
                rec["sources"].append({"role": role, "merged_file": dst.name, "from": [p for _, p, _, _ in items],
                                       "sha256": msha, "bytes": dst.stat().st_size})
                files[role] = str(dst)
            else:
                # bline needs a real file name it recognises: link the blob under its official basename
                dst = out / Path(items[0][1]).name if len(paths) == 1 else None
                if g in ("q21", "q21t") and role == "vae":
                    dst = out / out_name(g, role)
                    if not dst.exists():
                        import subprocess
                        import urllib.request
                        conv = out / "convert_wan_to_diffusers.py"
                        urllib.request.urlretrieve(DIFFUSERS_CONVERT, conv)
                        r = subprocess.run([os.environ.get("BLINE_PY", sys.executable), str(Path(__file__).parent / "qwen21_vae_to_bline.py"),
                                            os.environ["BLINE_DIR"], str(conv), paths[0], str(dst)], capture_output = True, text = True)
                        print("VAE_CONVERT", r.returncode, r.stdout[-800:], r.stderr[-1500:], flush = True)
                        if r.returncode:
                            raise SystemExit("q21 vae convert failed")
                    rec["sources"].append({"role": role, "converted_file": dst.name, "from": items[0][1],
                                           "converter": "qwen21_vae_to_bline.py + " + DIFFUSERS_CONVERT,
                                           "sha256": sha256(str(dst)), "bytes": dst.stat().st_size})
                    files[role] = str(dst)
                    continue
                if (g, role) in LINK_NAME:
                    dst = out / LINK_NAME[(g, role)]
                if dst.is_symlink():
                    dst.unlink()
                elif dst.exists():
                    raise SystemExit(f"{dst} exists and is a regular file, not a link: refusing to replace it")
                dst.symlink_to(paths[0])
                files[role] = str(dst)
        old = man.get(g) or {}
        rec["files"] = {**(old.get("files") or {}), **files}
        keep = [x for x in old.get("sources") or [] if x.get("role") not in {s["role"] for s in rec["sources"]}]
        rec["sources"] = keep + rec["sources"]
        man[g] = rec
        Path(a.manifest).write_text(json.dumps(man, indent = 1))
    print("PREP_DONE", json.dumps({g: man[g]["files"] for g in man}), flush = True)


if __name__ == "__main__":
    main()
