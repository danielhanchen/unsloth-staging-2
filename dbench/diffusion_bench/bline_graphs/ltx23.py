"""LTX-2.3 text to video with audio, the template's TWO-STAGE distilled recipe (video_ltx2_3_t2v.json, subgraph
"Text to Video (LTX-2.3)", prompt enhancer off, image input bypassed as in t2v):

  CheckpointLoaderSimple(ltx-2.3-22b-dev-fp8) -> LoraLoaderModelOnly(distilled 1.1 LoRA, 0.5) = model
  LTXAVTextEncoderLoader(gemma_3_12B_it_fp4_mixed, ckpt) -> CLIPTextEncode pos / neg -> LTXVConditioning(fps)
  stage 1: EmptyLTXVLatentVideo(W/2, H/2, F) + LTXVEmptyLatentAudio(F, fps, LTXVAudioVAELoader(ckpt)) -> concat ->
           SamplerCustomAdvanced(RandomNoise(seed), CFGGuider(model, cfg 1), euler, ManualSigmas 8 steps)
  upscale: LTXVSeparateAVLatent -> LTXVLatentUpsampler(x2 spatial upscaler, ckpt VAE) -> concat with stage-1 audio
  stage 2: SamplerCustomAdvanced(RandomNoise(42, the template's fixed stage-2 seed), CFGGuider(model,
           LTXVCropGuides(pos, neg, stage-1 video), cfg 1), euler, ManualSigmas "0.85, 0.7250, 0.4219, 0.0" = 3 steps)
  decode:  VAEDecodeTiled(768, 64, 4096, 4) + LTXVAudioVAEDecode -> CreateVideo(fps, audio) -> SaveVideo

= 8 + 3 = 11 denoiser passes at CFG 1. Template size 1280x720, 5 s @ 25 fps (126 frames; the latents floor to 121
frames and 1280x704 output), which is what the card pins (1280x704, 121 frames, fps 25). The abliterated Gemma LoRA
in the template feeds only the prompt enhancer (off), not the conditioning.
Files: checkpoint (ltx-2.3-22b-dev-fp8 or the bf16 dev), te (gemma_3_12B_it_fp4_mixed or gemma_3_12B_it), lora
(the distilled LoRA). graph_extra: sigmas1 / sigmas2, lora_strength (0.5), upscaler (file name, read from BlineUI's
models/latent_upscale_models like ltx25), stage2_seed (42), tile (768, 64, 4096, 4). ``steps`` != 8 keeps the first
N stage-1 sigmas (short smoke renders). Frames 8k+1, width / height multiples of 64.
The single-stage dev graph is "ltx-2.3-dev" (ltx23_dev.py), a non-default extra arm.
"""

from . import SAVE, GraphParams, save_video_nodes, wrap_model

TEMPLATE = "video_ltx2_3_t2v.json"
DEFAULTS = {"sampler": "euler", "cfg": 1.0, "steps": 8, "fps": 25.0}
FILES = {"checkpoint": "checkpoints", "te": "text_encoders", "lora": "loras", "upscaler": "latent_upscale_models"}
VIDEO = True
DEFAULT_NEGATIVE = "pc game, console game, video game, cartoon, childish, ugly"
SIGMAS1 = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
SIGMAS2 = "0.85, 0.7250, 0.4219, 0.0"
LORA = "ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors"
UPSCALER = "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"


def build(p: GraphParams) -> dict:
    x = p.extra or {}
    frames = int(p.frames or 121)
    if (frames - 1) % 8:
        raise ValueError(f"ltx-2.3: frames must be 8k+1, got {frames}")
    if int(p.width) % 64 or int(p.height) % 64:
        raise ValueError(f"ltx-2.3: width / height must be multiples of 64 (two stages), got {p.width}x{p.height}")
    fps = float(p.fps or DEFAULTS["fps"])
    ckpt = p.files["checkpoint"]
    te = p.files["te"] if isinstance(p.files["te"], str) else p.files["te"][0]
    s1 = x.get("sigmas1", SIGMAS1)
    if int(p.steps) != 8 and "sigmas1" not in x:
        s1 = ", ".join([s.strip() for s in SIGMAS1.split(",")][: max(1, int(p.steps))] + ["0.0"])
    cfg = float(p.cfg if p.cfg is not None else DEFAULTS["cfg"])
    tile = list(x.get("tile", (768, 64, 4096, 4)))
    g = {
        "ckpt": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
        "lora": {"class_type": "LoraLoaderModelOnly",
                 "inputs": {"model": ["ckpt", 0], "lora_name": p.files.get("lora") or LORA,
                            "strength_model": float(x.get("lora_strength", 0.5))}},
        "clip": {"class_type": "LTXAVTextEncoderLoader", "inputs": {"text_encoder": te, "ckpt_name": ckpt, "device": "default"}},
        "avae": {"class_type": "LTXVAudioVAELoader", "inputs": {"ckpt_name": ckpt}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": p.prompt}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0],
                                                          "text": DEFAULT_NEGATIVE if p.negative is None else p.negative}},
        "cond": {"class_type": "LTXVConditioning", "inputs": {"positive": ["pos", 0], "negative": ["neg", 0], "frame_rate": fps}},
        "vlat": {"class_type": "EmptyLTXVLatentVideo",
                 "inputs": {"width": int(p.width) // 2, "height": int(p.height) // 2, "length": frames, "batch_size": 1}},
        "alat": {"class_type": "LTXVEmptyLatentAudio",
                 "inputs": {"frames_number": frames, "frame_rate": int(round(fps)), "batch_size": 1, "audio_vae": ["avae", 0]}},
        "av1": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["vlat", 0], "audio_latent": ["alat", 0]}},
        "ksel": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": p.sampler or DEFAULTS["sampler"]}},
        "noise1": {"class_type": "RandomNoise", "inputs": {"noise_seed": int(p.seed)}},
        "noise2": {"class_type": "RandomNoise", "inputs": {"noise_seed": int(x.get("stage2_seed", 42))}},
        "sig1": {"class_type": "ManualSigmas", "inputs": {"sigmas": s1}},
        "sig2": {"class_type": "ManualSigmas", "inputs": {"sigmas": x.get("sigmas2", SIGMAS2)}},
        "upm": {"class_type": "LatentUpscaleModelLoader",
                "inputs": {"model_name": x.get("upscaler") or (p.files or {}).get("upscaler") or UPSCALER}},
    }
    model = wrap_model(g, ["lora", 0], p)
    g["guider1"] = {"class_type": "CFGGuider", "inputs": {"model": model, "positive": ["cond", 0],
                                                          "negative": ["cond", 1], "cfg": cfg}}
    g["s1"] = {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["noise1", 0], "guider": ["guider1", 0],
                                                                 "sampler": ["ksel", 0], "sigmas": ["sig1", 0],
                                                                 "latent_image": ["av1", 0]}}
    g["sep1"] = {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["s1", 0]}}
    g["up"] = {"class_type": "LTXVLatentUpsampler", "inputs": {"samples": ["sep1", 0], "upscale_model": ["upm", 0],
                                                               "vae": ["ckpt", 2]}}
    g["av2"] = {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["up", 0], "audio_latent": ["sep1", 1]}}
    g["crop"] = {"class_type": "LTXVCropGuides", "inputs": {"positive": ["cond", 0], "negative": ["cond", 1],
                                                            "latent": ["sep1", 0]}}
    g["guider2"] = {"class_type": "CFGGuider", "inputs": {"model": model, "positive": ["crop", 0],
                                                          "negative": ["crop", 1], "cfg": cfg}}
    g["s2"] = {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["noise2", 0], "guider": ["guider2", 0],
                                                                 "sampler": ["ksel", 0], "sigmas": ["sig2", 0],
                                                                 "latent_image": ["av2", 0]}}
    g["sep2"] = {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["s2", 0]}}
    g["decode"] = {"class_type": "VAEDecodeTiled", "inputs": {"samples": ["sep2", 0], "vae": ["ckpt", 2],
                                                              "tile_size": int(tile[0]), "overlap": int(tile[1]),
                                                              "temporal_size": int(tile[2]), "temporal_overlap": int(tile[3])}}
    g["adecode"] = {"class_type": "LTXVAudioVAEDecode", "inputs": {"samples": ["sep2", 1], "audio_vae": ["avae", 0]}}
    save_video_nodes(g, ["decode", 0], fps, p.prefix)
    g["video"]["inputs"]["audio"] = ["adecode", 0]
    assert SAVE in g
    return g
