"""MiniMax-H3 text to video with audio (BlineUI native minimax_h3 model, MiniMaxH3ImageToVideo with no frames).

Template: video_minimax_h3_t2v.json (blineui-workflow-templates 0.11.70, BlineUI a716932248), turbo switch off:
UNETLoader(minimax_h3_fl2va_pruned_int8_convrot) -> BasicGuider (no CFG); CLIPLoader(qwen3vl_32b_minimax_h3_nvfp4_awq,
type minimax); VAELoader(minimax_h3_video_vae_int8_convrot), VAELoader(minimax_h3_audio_vae_fp32);
MiniMaxH3ImageToVideo(prompt, 1344x768 by the ResolutionSelector's 16:9 / 0.4 MP -> 864x480 on the top level, 5 s ->
length 124) -> SamplerCustomAdvanced(RandomNoise, res_multistep, BasicScheduler simple 20 steps) -> VAEDecode +
VAEDecodeAudio -> CreateVideo(24). Model sampling shift 12 / audio shift 3 come from the model config (bline
supported_models MiniMaxH3), the same values Studio uses (default_flow_shift 12, audio 3). Turbo on = the lightx2v
8-step LoRA with 6 steps.

Studio's family default: 30 steps, CFG 1 (no CFG), 1344x768 (960x544 "faster" preset), 124 frames @ 24 fps, the
FULL (not pruned) fl2va denoiser (MiniMaxAI/MiniMax-H3; hosted INT8-ConvRot under auto) and a bf16 Qwen3-VL-32B
text encoder. For like-for-like use dit = minimax_h3_fl2va_bf16 (bf16 ref) or minimax_h3_fl2va_int8_convrot; the
_pruned_ files are a smaller, different model (template default), quality not comparable to Studio's.
Files (Bline-Org/MiniMax-H3): dit diffusion_models/*.safetensors; te text_encoders/qwen3vl_32b_minimax_h3_{bf16,
int8_convrot,nvfp4_awq}; vae vae/minimax_h3_video_vae_{fp16,int8_convrot}; audio_vae vae/minimax_h3_audio_vae_fp32.
Frames 17k+5 (124, 141, ...); width / height multiples of 32. graph_extra: audio (true), audio_shift (3.0).
``shift`` (option) inserts MiniMaxH3SigmaShift(shift_video = shift, shift_audio = graph_extra.audio_shift), the
model's own shift node: ModelSamplingSD3 would replace its ModelSamplingAV and drop the separate audio shift.
"""

from . import SAVE, GraphParams, pick, save_node, wrap_model

TEMPLATE = "video_minimax_h3_t2v.json"
DEFAULTS = {"sampler": "res_multistep", "scheduler": "simple", "steps": 30, "shift": None}
FILES = {"dit": "diffusion_models", "te": "text_encoders", "vae": "vae", "audio_vae": "vae"}
VIDEO = True


def build(p: GraphParams) -> dict:
    import sys

    mod = sys.modules[__name__]
    x = p.extra or {}
    frames = int(p.frames or 124)
    if frames < 5 or (frames - 5) % 17:
        raise ValueError(f"minimax-h3: frames must be 17k+5, got {frames}")
    te = p.files["te"] if isinstance(p.files["te"], str) else p.files["te"][0]
    g = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": p.files["dit"], "weight_dtype": p.weight_dtype}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": te, "type": "minimax", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": p.files["vae"]}},
        "audio_vae": {"class_type": "VAELoader", "inputs": {"vae_name": p.files["audio_vae"]}},
        "cond": {"class_type": "MiniMaxH3ImageToVideo",
                 "inputs": {"clip": ["clip", 0], "vae": ["vae", 0], "prompt": p.prompt, "width": p.width,
                            "height": p.height, "length": frames}},
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": int(p.seed)}},
        "ksampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": pick(p, mod, "sampler")}},
    }
    link = ["unet", 0]
    if p.shift is not None:
        g["shift"] = {"class_type": "MiniMaxH3SigmaShift",
                      "inputs": {"model": link, "shift_video": float(p.shift),
                                 "shift_audio": float(x.get("audio_shift", 3.0))}}
        link = ["shift", 0]
    model = wrap_model(g, link, p)
    g["sigmas"] = {"class_type": "BasicScheduler",
                   "inputs": {"model": model, "scheduler": pick(p, mod, "scheduler"), "steps": int(p.steps), "denoise": 1.0}}
    g["guider"] = {"class_type": "BasicGuider", "inputs": {"model": model, "conditioning": ["cond", 0]}}
    g["sampler"] = {"class_type": "SamplerCustomAdvanced",
                    "inputs": {"noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["ksampler", 0],
                               "sigmas": ["sigmas", 0], "latent_image": ["cond", 1]}}
    g["decode"] = {"class_type": "VAEDecode", "inputs": {"samples": ["sampler", 0], "vae": ["vae", 0]}}
    if x.get("audio", True):
        g["adecode"] = {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["sampler", 0], "vae": ["audio_vae", 0]}}
        g["apreview"] = {"class_type": "PreviewAudio", "inputs": {"audio": ["adecode", 0]}}
    g[SAVE] = save_node(["decode", 0], p.prefix)
    return g
