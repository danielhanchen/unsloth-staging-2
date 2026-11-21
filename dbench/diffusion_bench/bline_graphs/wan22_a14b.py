"""Wan2.2-T2V-A14B text to video, two experts (high noise then low noise) via two KSamplerAdvanced.

Template: video_wan2_2_14B_t2v.json (blineui-workflow-templates 0.11.70, BlineUI a716932248), Lightning LoRA switch
off: UNETLoader(wan2.2_t2v_high_noise_14B_fp8_scaled) and UNETLoader(..low_noise..fp8_scaled), each ->
ModelSamplingSD3(shift 5); CLIPLoader(umt5_xxl_fp8_e4m3fn_scaled, wan) -> CLIPTextEncode pos / neg (Wan's Chinese
default negative); EmptyHunyuanLatentVideo(640, 640, 81 = 5 s x 16 fps + 1); KSamplerAdvanced high (add_noise
enable, 20 steps, CFG 3.5, euler / simple, steps 0-10, return_with_leftover_noise enable) -> KSamplerAdvanced low
(add_noise disable, steps 10-20) -> VAEDecode(wan_2.1_vae) -> CreateVideo(16).

Studio's family default differs: diffusers Wan-AI/Wan2.2-T2V-A14B-Diffusers, 50 steps, CFG 5 (both experts),
1280x720x81 @ 16 fps (832x480 preset), UniPC flow_shift 3.0, expert switch at boundary_ratio 0.875 (the high-noise
expert runs while timestep >= 875). ``graph_extra.boundary`` (e.g. 0.875) computes the switch step from this graph's
own shifted ``simple`` sigmas (count of steps with sigma >= boundary); ``graph_extra.switch_step`` sets it directly;
neither = steps // 2 (the template's 10 of 20). ``graph_extra.cfg2`` sets the low-noise expert's CFG (default = cfg).
Files: dit / dit2 = Bline-Org/Wan_2.2_BlineUI_Repackaged wan2.2_t2v_{high,low}_noise_14B_fp16 (bf16 ref) or
_fp8_scaled (template); te = umt5_xxl_fp16 or umt5_xxl_fp8_e4m3fn_scaled; vae = wan_2.1_vae. ``weight_dtype`` applies
to both UNETLoaders. Frames 4k+1; width / height multiples of 16.
"""

from . import GraphParams, pick, save_video_nodes, wrap_model
from .wan22_5b import DEFAULT_NEGATIVE

TEMPLATE = "video_wan2_2_14B_t2v.json"
DEFAULTS = {"sampler": "euler", "scheduler": "simple", "cfg": 3.5, "shift": 5.0, "steps": 20}
FILES = {"dit": "diffusion_models", "dit2": "diffusion_models", "te": "text_encoders", "vae": "vae"}
VIDEO = True


def boundary_switch_step(steps: int, shift: float, boundary: float) -> int:
    """Number of leading steps whose sigma is >= ``boundary`` under BlineUI's ModelSamplingSD3(shift) + ``simple``
    scheduler (sigma table t = k/1000, k = 1..1000, shifted s*t / (1 + (s-1)*t); simple picks
    table[-(1 + int(x * 1000 / steps))] for step x)."""
    n = 0
    for x in range(int(steps)):
        t = (1000 - int(x * 1000 / steps)) / 1000.0
        sigma = shift * t / (1 + (shift - 1) * t)
        if sigma >= boundary:
            n += 1
    return n


def build(p: GraphParams) -> dict:
    import sys

    mod = sys.modules[__name__]
    x = p.extra or {}
    frames = int(p.frames or 81)
    if (frames - 1) % 4:
        raise ValueError(f"wan2.2-a14b: frames must be 4k+1, got {frames}")
    steps = int(p.steps)
    shift = float(pick(p, mod, "shift"))
    if x.get("switch_step") is not None:
        switch = int(x["switch_step"])
    elif x.get("boundary") is not None:
        switch = boundary_switch_step(steps, shift, float(x["boundary"]))
    else:
        switch = steps // 2
    switch = max(0, min(steps, switch))
    cfg = float(pick(p, mod, "cfg"))
    cfg2 = float(x.get("cfg2", cfg))
    te = p.files["te"] if isinstance(p.files["te"], str) else p.files["te"][0]
    g = {
        "unet_hi": {"class_type": "UNETLoader", "inputs": {"unet_name": p.files["dit"], "weight_dtype": p.weight_dtype}},
        "unet_lo": {"class_type": "UNETLoader", "inputs": {"unet_name": p.files["dit2"], "weight_dtype": p.weight_dtype}},
        "shift_hi": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["unet_hi", 0], "shift": shift}},
        "shift_lo": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["unet_lo", 0], "shift": shift}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": te, "type": "wan", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": p.files["vae"]}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": p.prompt}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0],
                                                          "text": DEFAULT_NEGATIVE if p.negative is None else p.negative}},
        "latent": {"class_type": "EmptyHunyuanLatentVideo",
                   "inputs": {"width": p.width, "height": p.height, "length": frames, "batch_size": 1}},
    }
    hi = wrap_model(g, ["shift_hi", 0], p, suffix = "_hi")
    lo = wrap_model(g, ["shift_lo", 0], p, suffix = "_lo")
    common = {"sampler_name": pick(p, mod, "sampler"), "scheduler": pick(p, mod, "scheduler"), "steps": steps,
              "positive": ["pos", 0], "negative": ["neg", 0]}
    g["sampler_hi"] = {"class_type": "KSamplerAdvanced",
                       "inputs": {**common, "model": hi, "add_noise": "enable", "noise_seed": int(p.seed), "cfg": cfg,
                                  "latent_image": ["latent", 0], "start_at_step": 0, "end_at_step": switch,
                                  "return_with_leftover_noise": "enable"}}
    # "sampler" is the node id the backend's cache check reads; it is the last sampler, as in the other graphs.
    g["sampler"] = {"class_type": "KSamplerAdvanced",
                    "inputs": {**common, "model": lo, "add_noise": "disable", "noise_seed": 0, "cfg": cfg2,
                               "latent_image": ["sampler_hi", 0], "start_at_step": switch, "end_at_step": 10000,
                               "return_with_leftover_noise": "disable"}}
    g["decode"] = {"class_type": "VAEDecode", "inputs": {"samples": ["sampler", 0], "vae": ["vae", 0]}}
    save_video_nodes(g, ["decode", 0], p.fps or 16, p.prefix)
    return g
