"""Wan2.2-TI2V-5B text to video: Bline-Org's video_wan2_2_5B_ti2v template. UNETLoader(wan2.2_ti2v_5B_fp16) ->
ModelSamplingSD3(shift 8) -> KSampler(uni_pc / simple, CFG 5, 20 steps); CLIPLoader(umt5_xxl, type wan) ->
CLIPTextEncode pos / neg; Wan22ImageToVideoLatent(vae, width, height, length); VAEDecode(wan2.2_vae). The template
saves through CreateVideo + SaveVideo (mp4); here the decoded frames go to SaveImage instead, one lossless PNG per
frame, which is what the scorer compares (run_cell writes frames.npz + mp4 itself).
Files (Bline-Org/Wan_2.2_BlineUI_Repackaged split_files): dit wan2.2_ti2v_5B_fp16.safetensors,
te umt5_xxl_fp8_e4m3fn_scaled.safetensors (or umt5_xxl_fp16), vae wan2.2_vae.safetensors.
Length must be 4k+1 frames; width / height multiples of 32.
"""

from . import GraphParams, pick, save_video_nodes, wrap_model

TEMPLATE = "video_wan2_2_5B_ti2v.json"
DEFAULTS = {"sampler": "uni_pc", "scheduler": "simple", "cfg": 5.0, "shift": 8.0, "steps": 20}
FILES = {"dit": "diffusion_models", "te": "text_encoders", "vae": "vae"}
VIDEO = True
DEFAULT_NEGATIVE = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，"
                    "多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")


def build(p: GraphParams) -> dict:
    import sys

    mod = sys.modules[__name__]
    frames = int(p.frames or 121)
    if (frames - 1) % 4:
        raise ValueError(f"wan2.2-5b: frames must be 4k+1, got {frames}")
    g = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": p.files["dit"], "weight_dtype": p.weight_dtype}},
        "shift": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["unet", 0], "shift": float(pick(p, mod, "shift"))}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": p.files["te"], "type": "wan", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": p.files["vae"]}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": p.prompt}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0],
                                                          "text": DEFAULT_NEGATIVE if p.negative is None else p.negative}},
        "latent": {"class_type": "Wan22ImageToVideoLatent",
                   "inputs": {"vae": ["vae", 0], "width": p.width, "height": p.height, "length": frames, "batch_size": 1}},
    }
    model = wrap_model(g, ["shift", 0], p)
    g["sampler"] = {"class_type": "KSampler",
                    "inputs": {"model": model, "seed": int(p.seed), "steps": int(p.steps), "cfg": float(pick(p, mod, "cfg")),
                               "sampler_name": pick(p, mod, "sampler"), "scheduler": pick(p, mod, "scheduler"),
                               "positive": ["pos", 0], "negative": ["neg", 0], "latent_image": ["latent", 0],
                               "denoise": 1.0}}
    g["decode"] = {"class_type": "VAEDecode", "inputs": {"samples": ["sampler", 0], "vae": ["vae", 0]}}
    save_video_nodes(g, ["decode", 0], p.fps or 24, p.prefix)
    return g
