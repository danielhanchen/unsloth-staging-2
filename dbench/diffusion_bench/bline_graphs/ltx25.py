"""LTX-2.5 text to video with audio: the two-stage distilled recipe of BlineUI v0.44.0's own template.

Template: video_ltx2_5_t2v.json (blineui-workflow-templates 0.11.101 -> blineui-workflow-templates-json 0.1.127, the
version BlineUI v0.44.0 pins), subgraph "Text to Video (LTX-2.5)", rebuilt in API format minus the optional prompt
enhancer branch (TextGenerateLTX2Prompt on a gemma4 e2b encoder, switched by a boolean; the harness feeds the prompt
text straight to CLIPTextEncode, as the switch's "off" side does):

  UNETLoader(ltx-2.5-22b-distilled-transformer-bline-int8-convrot, default)     [GGUF: BlineUI-GGUF UnetLoaderGGUF]
  CLIPLoader(gemma4-12b-with-proj-ltx-2.5-bline-int8-convrot, ltxv) -> CLIPTextEncode pos / neg -> LTXVConditioning(fps)
  EmptyLTXVLatentVideo(W/2, H/2, F) + LTXVEmptyLatentAudio(F, fps, audio VAE) -> LTXVConcatAVLatent
  stage 1: SamplerCustomAdvanced(RandomNoise, LTXVDualCFGGuider(1, 1), euler_ancestral,
           ManualSigmas "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0")
  -> LTXVSeparateAVLatent -> LTXVLatentUpsampler(x2 spatial upscaler, video VAE) + stage-1 audio -> LTXVConcatAVLatent
  stage 2: SamplerCustomAdvanced(RandomNoise, LTXVDualCFGGuider(1, 1), euler_ancestral, ManualSigmas "0.85, 0.7250,
           0.4219, 0.0") -> LTXVSeparateAVLatent -> VAEDecodeTiled(512, 64, 64, 16) + LTXVAudioVAEDecode
  -> CreateVideo(fps, audio) -> SaveVideo(auto / auto)

Differences from the template, all identical on both stacks of a row: the prompt / seed / size / length come from the
cell (the template: 1280x720 from a ResolutionSelector, 5 s @ 24 -> 121 frames; Studio renders 1280x704, the 64 px
two-stage canvas, so both stacks run 1280x704 here); both stages take the cell's seed (the template randomizes stage 1
and fixes stage 2 at 42). Frames must be 8k+1, width / height multiples of 64.

Files (roles): dit (diffusion_models: a .safetensors via UNETLoader, a .gguf via UnetLoaderGGUF), te (text_encoders),
vae + audio_vae (vae), upscaler (latent_upscale_models: the spatial upscaler, a card file like the others, so the
download / build plan fetches it and the server finds it through extra_model_paths).
graph_extra: steps1 / steps2 sigmas override, sampler (euler_ancestral), tile (512, 64, 64, 16), upscaler (file name).
"""

from . import SAVE, GraphParams, save_video_nodes, wrap_model

TEMPLATE = "video_ltx2_5_t2v.json"
DEFAULTS = {"sampler": "euler_ancestral", "cfg": 1.0, "steps": 8, "fps": 24.0}
FILES = {"dit": "diffusion_models", "te": "text_encoders", "vae": "vae", "audio_vae": "vae", "upscaler": "latent_upscale_models"}
VIDEO = True
DEFAULT_NEGATIVE = "pc game, console game, video game, cartoon, childish, ugly"  # the template's negative
SIGMAS1 = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
SIGMAS2 = "0.85, 0.7250, 0.4219, 0.0"
UPSCALER = "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"


def build(p: GraphParams) -> dict:
    x = p.extra or {}
    frames = int(p.frames or 121)
    if (frames - 1) % 8:
        raise ValueError(f"ltx-2.5: frames must be 8k+1, got {frames}")
    if int(p.width) % 64 or int(p.height) % 64:  # stage 1 runs at half size: the two-stage grid, as ltx23 checks
        raise ValueError(f"ltx-2.5: width / height must be multiples of 64 (two stages), got {p.width}x{p.height}")
    fps = float(p.fps or DEFAULTS["fps"])
    dit = p.files["dit"]
    te = p.files["te"] if isinstance(p.files["te"], str) else p.files["te"][0]
    if str(dit).endswith(".gguf"):
        unet = {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": dit}}
    else:
        unet = {"class_type": "UNETLoader", "inputs": {"unet_name": dit, "weight_dtype": p.weight_dtype or "default"}}
    s1 = x.get("sigmas1", SIGMAS1)
    if int(p.steps) != 8 and "sigmas1" not in x:
        # a shorter stage 1 (profile / smoke renders): the first N distilled sigmas, then 0
        v = [s.strip() for s in SIGMAS1.split(",")][: max(1, int(p.steps))] + ["0.0"]
        s1 = ", ".join(v)
    tile = list(x.get("tile", (512, 64, 64, 16)))
    g = {
        "unet": unet,
        "vvae": {"class_type": "VAELoader", "inputs": {"vae_name": p.files["vae"]}},
        "avae": {"class_type": "VAELoader", "inputs": {"vae_name": p.files["audio_vae"]}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": te, "type": "ltxv", "device": "default"}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": p.prompt}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0],
                                                          "text": DEFAULT_NEGATIVE if p.negative is None else p.negative}},
        "cond": {"class_type": "LTXVConditioning", "inputs": {"positive": ["pos", 0], "negative": ["neg", 0], "frame_rate": fps}},
        "vlat": {"class_type": "EmptyLTXVLatentVideo",
                 "inputs": {"width": int(p.width) // 2, "height": int(p.height) // 2, "length": frames, "batch_size": 1}},
        "alat": {"class_type": "LTXVEmptyLatentAudio",
                 "inputs": {"frames_number": frames, "frame_rate": int(round(fps)), "batch_size": 1, "audio_vae": ["avae", 0]}},
        "av1": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["vlat", 0], "audio_latent": ["alat", 0]}},
        "ksel": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": x.get("sampler", DEFAULTS["sampler"])}},
        "noise1": {"class_type": "RandomNoise", "inputs": {"noise_seed": int(p.seed)}},
        "noise2": {"class_type": "RandomNoise", "inputs": {"noise_seed": int(p.seed)}},
        "sig1": {"class_type": "ManualSigmas", "inputs": {"sigmas": s1}},
        "sig2": {"class_type": "ManualSigmas", "inputs": {"sigmas": x.get("sigmas2", SIGMAS2)}},
        "upm": {"class_type": "LatentUpscaleModelLoader",
                "inputs": {"model_name": x.get("upscaler") or (p.files or {}).get("upscaler") or UPSCALER}},
    }
    model = wrap_model(g, ["unet", 0], p)
    cfg = float(p.cfg if p.cfg is not None else DEFAULTS["cfg"])
    for k in ("guider1", "guider2"):
        g[k] = {"class_type": "LTXVDualCFGGuider", "inputs": {"model": model, "positive": ["cond", 0], "negative": ["cond", 1],
                                                              "video_cfg": cfg, "audio_cfg": cfg}}
    g["s1"] = {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["noise1", 0], "guider": ["guider1", 0],
                                                                 "sampler": ["ksel", 0], "sigmas": ["sig1", 0],
                                                                 "latent_image": ["av1", 0]}}
    g["sep1"] = {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["s1", 0]}}
    g["up"] = {"class_type": "LTXVLatentUpsampler", "inputs": {"samples": ["sep1", 0], "upscale_model": ["upm", 0],
                                                               "vae": ["vvae", 0]}}
    g["av2"] = {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["up", 0], "audio_latent": ["sep1", 1]}}
    g["s2"] = {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["noise2", 0], "guider": ["guider2", 0],
                                                                 "sampler": ["ksel", 0], "sigmas": ["sig2", 0],
                                                                 "latent_image": ["av2", 0]}}
    g["sep2"] = {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["s2", 0]}}
    g["decode"] = {"class_type": "VAEDecodeTiled", "inputs": {"samples": ["sep2", 0], "vae": ["vvae", 0],
                                                              "tile_size": int(tile[0]), "overlap": int(tile[1]),
                                                              "temporal_size": int(tile[2]), "temporal_overlap": int(tile[3])}}
    g["adecode"] = {"class_type": "LTXVAudioVAEDecode", "inputs": {"samples": ["sep2", 1], "audio_vae": ["avae", 0]}}
    save_video_nodes(g, ["decode", 0], fps, p.prefix)
    g["video"]["inputs"]["audio"] = ["adecode", 0]
    assert SAVE in g
    return g
