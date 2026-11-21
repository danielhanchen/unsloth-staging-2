"""LTX-2.3 (and LTX-2) text to video with audio, SINGLE STAGE: registered as "ltx-2.3-dev", a NON-default extra arm
(Studio's ltx-2 dev route). The template-faithful default is "ltx-2.3" (ltx23.py, two-stage distilled, 11 passes).
Original summary: SINGLE STAGE at the requested size: the dev checkpoint, plain CFG.

Template: video_ltx2_3_t2v.json (blineui-workflow-templates 0.11.70, BlineUI a716932248). The template is NOT what
this graph runs by default: it is a two-stage distilled pipeline (half-res stage 1 with the distilled LoRA at 0.5,
8 manual sigmas, CFG 1, euler; x2 latent upsample; 3-sigma refine; VAEDecodeTiled 768/64/4096/4) at 1280x720,
5 s @ 25 fps (length 126 -> 8k+1 lattice), fp8 dev checkpoint, gemma_3_12B_it_fp4_mixed text encoder and an
optional Gemma prompt enhancer (TextGenerateLTX2Prompt + abliterated LoRA). The non-distilled LTX-2 template
(video_ltx2_t2v.json) runs stage 1 as LTXVScheduler(20, max_shift 2.05, base_shift 0.95, stretch, terminal 0.1),
CFGGuider 4, euler_ancestral, then the same distilled refine.

This graph = the same nodes minus the distilled / upscale / enhancer branch, so it matches Studio's LTX-2 family
default (diffusers LTX2Pipeline: 40 steps, CFG 4, 768x512x121 @ 24 fps, FlowMatchEuler with exponential dynamic
shift base 0.95 / max 2.05 over 1024..4096 tokens and shift_terminal 0.1, which is exactly LTXVScheduler's math):
CheckpointLoaderSimple(ckpt) -> [TorchCompile / EasyCache] -> CFGGuider(cfg) ; LTXAVTextEncoderLoader(te, ckpt) ->
CLIPTextEncode pos / neg -> LTXVConditioning(frame_rate) ; EmptyLTXVLatentVideo + LTXVEmptyLatentAudio(audio VAE)
-> LTXVConcatAVLatent -> LTXVScheduler(steps, on the AV latent) -> SamplerCustomAdvanced(euler) ->
LTXVSeparateAVLatent -> VAEDecodeTiled (template tile values; ``graph_extra.vae_decode = "full"`` for VAEDecode)
and LTXVAudioVAEDecode -> PreviewAudio (so the audio decode runs, as the template's CreateVideo makes it).

Files: checkpoint = Lightricks/LTX-2.3 ltx-2.3-22b-dev.safetensors (bf16, model + video VAE + audio VAE +
connectors) or Lightricks/LTX-2.3-fp8 ltx-2.3-22b-dev-fp8.safetensors (the template's); te =
Bline-Org/ltx-2 split_files/text_encoders/gemma_3_12B_it.safetensors (bf16) or gemma_3_12B_it_fp4_mixed (template).
LTX-2: ltx-2-19b-dev(.fp8).safetensors from Lightricks/LTX-2 with the same graph. fp8 comes from the fp8
checkpoint file (CheckpointLoaderSimple has no weight_dtype). Frames must be 8k+1; width / height multiples of 32.
graph_extra: sampler (euler), max_shift, base_shift, stretch, terminal, vae_decode (tiled | full),
tile (768, 64, 4096, 4), audio (true).
"""

from . import SAVE, GraphParams, pick, save_node, wrap_model

TEMPLATE = "video_ltx2_3_t2v.json"
DEFAULTS = {"sampler": "euler", "cfg": 4.0, "steps": 40, "fps": 24.0}
FILES = {"checkpoint": "checkpoints", "te": "text_encoders"}
VIDEO = True
DEFAULT_NEGATIVE = "pc game, console game, video game, cartoon, childish, ugly"  # the LTX-2.3 template's negative


def build(p: GraphParams) -> dict:
    import sys

    mod = sys.modules[__name__]
    x = p.extra or {}
    frames = int(p.frames or 121)
    if (frames - 1) % 8:
        raise ValueError(f"ltx-2.3: frames must be 8k+1, got {frames}")
    fps = float(p.fps or DEFAULTS["fps"])
    ckpt = p.files["checkpoint"]
    te = p.files["te"] if isinstance(p.files["te"], str) else p.files["te"][0]
    g = {
        "ckpt": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
        "clip": {"class_type": "LTXAVTextEncoderLoader", "inputs": {"text_encoder": te, "ckpt_name": ckpt, "device": "default"}},
        "audio_vae": {"class_type": "LTXVAudioVAELoader", "inputs": {"ckpt_name": ckpt}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": p.prompt}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0],
                                                          "text": DEFAULT_NEGATIVE if p.negative is None else p.negative}},
        "cond": {"class_type": "LTXVConditioning", "inputs": {"positive": ["pos", 0], "negative": ["neg", 0], "frame_rate": fps}},
        "vlat": {"class_type": "EmptyLTXVLatentVideo",
                 "inputs": {"width": p.width, "height": p.height, "length": frames, "batch_size": 1}},
        "alat": {"class_type": "LTXVEmptyLatentAudio",
                 "inputs": {"frames_number": frames, "frame_rate": int(round(fps)), "batch_size": 1, "audio_vae": ["audio_vae", 0]}},
        "avlat": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["vlat", 0], "audio_latent": ["alat", 0]}},
        "sigmas": {"class_type": "LTXVScheduler",
                   "inputs": {"steps": int(p.steps), "max_shift": float(x.get("max_shift", 2.05)),
                              "base_shift": float(x.get("base_shift", 0.95)), "stretch": bool(x.get("stretch", True)),
                              "terminal": float(x.get("terminal", 0.1)), "latent": ["avlat", 0]}},
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": int(p.seed)}},
        "ksampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": pick(p, mod, "sampler")}},
    }
    vals = x.get("sigmas")
    if isinstance(vals, str):
        vals = [float(v) for v in vals.replace(",", " ").split()]
    if vals and int(p.steps) == len(vals) - 1:
        # distilled checkpoints at their trained step count: the fixed curve incl. the terminal 0 (ManualSigmas, as
        # the template's distilled stage); any other step count (the harness's short renders) keeps LTXVScheduler
        g["sigmas"] = {"class_type": "ManualSigmas",
                       "inputs": {"sigmas": ", ".join(str(float(v)) for v in vals)}}
    model = wrap_model(g, ["ckpt", 0], p)
    g["guider"] = {"class_type": "CFGGuider", "inputs": {"model": model, "positive": ["cond", 0], "negative": ["cond", 1],
                                                         "cfg": float(pick(p, mod, "cfg"))}}
    g["sampler"] = {"class_type": "SamplerCustomAdvanced",
                    "inputs": {"noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["ksampler", 0],
                               "sigmas": ["sigmas", 0], "latent_image": ["avlat", 0]}}
    g["sep"] = {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["sampler", 0]}}
    if x.get("vae_decode", "tiled") == "full":
        g["decode"] = {"class_type": "VAEDecode", "inputs": {"samples": ["sep", 0], "vae": ["ckpt", 2]}}
    else:
        t = list(x.get("tile", (768, 64, 4096, 4)))
        g["decode"] = {"class_type": "VAEDecodeTiled",
                       "inputs": {"samples": ["sep", 0], "vae": ["ckpt", 2], "tile_size": int(t[0]), "overlap": int(t[1]),
                                  "temporal_size": int(t[2]), "temporal_overlap": int(t[3])}}
    if x.get("audio", True):
        g["adecode"] = {"class_type": "LTXVAudioVAEDecode", "inputs": {"samples": ["sep", 1], "audio_vae": ["audio_vae", 0]}}
        g["apreview"] = {"class_type": "PreviewAudio", "inputs": {"audio": ["adecode", 0]}}
    g[SAVE] = save_node(["decode", 0], p.prefix)
    return g
