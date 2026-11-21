"""HunyuanVideo-1.5 text to video (480p t2v DiT by default).

Template: video_hunyuan_video_1.5_720p_t2v.json (blineui-workflow-templates 0.11.70; there is no 480p t2v template):
UNETLoader(hunyuanvideo1.5_720p_t2v_fp16) -> [EasyCache, bypassed] -> ModelSamplingSD3(shift 7) -> CFGGuider(6);
DualCLIPLoader(qwen_2.5_vl_7b_fp8_scaled, byt5_small_glyphxl_fp16, type hunyuan_video_15) -> CLIPTextEncode pos /
neg ("") ; EmptyHunyuanVideo15Latent(1280, 720, 121) ; BasicScheduler(simple, 20 steps) ; KSamplerSelect(euler) ;
RandomNoise -> SamplerCustomAdvanced -> VAEDecode(hunyuanvideo15_vae_fp16) -> CreateVideo(24) (the 1080p SR
branch is bypassed). In the template BasicScheduler reads the UNSHIFTED model, so its sigmas use the model's own
default shift (7.0, bline supported_models HunyuanVideo15) and the ModelSamplingSD3 node does not change them.

Here BasicScheduler reads the shifted model so ``shift`` controls the schedule (at the default shift 7 this equals
the template's unshifted read). Defaults = the 720p template: 20 steps, CFG 6, shift 7, 1280x720x121 @ 24 fps (the
cell sets size / frames); Studio's family default moved to the same 20 / 6 / shift 7.
Files: dit = Bline-Org/HunyuanVideo_1.5_repackaged split_files/diffusion_models/hunyuanvideo1.5_480p_t2v_fp16
(no non-distilled 480p t2v fp8 is shipped: fp8 = weight_dtype fp8_e4m3fn on the fp16 file); te = [qwen_2.5_vl_7b
(bf16) or qwen_2.5_vl_7b_fp8_scaled (template), byt5_small_glyphxl_fp16] in that order; vae = hunyuanvideo15_vae_fp16.
Frames 4k+1; width / height multiples of 16.
"""

from . import GraphParams, pick, save_video_nodes, wrap_model

TEMPLATE = "video_hunyuan_video_1.5_720p_t2v.json"
# Template values (Studio adopted them: bline_defaults_audit.md, DONE 7a8cfee55c / 4a96b1595c): 20 steps, CFG 6, shift 7.
DEFAULTS = {"sampler": "euler", "scheduler": "simple", "cfg": 6.0, "shift": 7.0, "steps": 20}
FILES = {"dit": "diffusion_models", "te": "text_encoders", "vae": "vae"}
VIDEO = True


def build(p: GraphParams) -> dict:
    import sys

    mod = sys.modules[__name__]
    frames = int(p.frames or 121)
    if (frames - 1) % 4:
        raise ValueError(f"hunyuanvideo-1.5: frames must be 4k+1, got {frames}")
    te = p.files["te"]
    if isinstance(te, str) or len(te) != 2:
        raise ValueError("hunyuanvideo-1.5: files.te must be [qwen_2.5_vl_7b*, byt5_small_glyphxl*]")
    g = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": p.files["dit"], "weight_dtype": p.weight_dtype}},
        "shift": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["unet", 0], "shift": float(pick(p, mod, "shift"))}},
        "clip": {"class_type": "DualCLIPLoader",
                 "inputs": {"clip_name1": te[0], "clip_name2": te[1], "type": "hunyuan_video_15", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": p.files["vae"]}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": p.prompt}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": p.negative or ""}},
        "latent": {"class_type": "EmptyHunyuanVideo15Latent",
                   "inputs": {"width": p.width, "height": p.height, "length": frames, "batch_size": 1}},
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": int(p.seed)}},
        "ksampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": pick(p, mod, "sampler")}},
    }
    model = wrap_model(g, ["shift", 0], p)
    g["sigmas"] = {"class_type": "BasicScheduler",
                   "inputs": {"model": model, "scheduler": pick(p, mod, "scheduler"), "steps": int(p.steps), "denoise": 1.0}}
    g["guider"] = {"class_type": "CFGGuider", "inputs": {"model": model, "positive": ["pos", 0], "negative": ["neg", 0],
                                                         "cfg": float(pick(p, mod, "cfg"))}}
    g["sampler"] = {"class_type": "SamplerCustomAdvanced",
                    "inputs": {"noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["ksampler", 0],
                               "sigmas": ["sigmas", 0], "latent_image": ["latent", 0]}}
    g["decode"] = {"class_type": "VAEDecode", "inputs": {"samples": ["sampler", 0], "vae": ["vae", 0]}}
    save_video_nodes(g, ["decode", 0], p.fps or 24, p.prefix)
    return g
