"""Ideogram 4 text to image: Bline-Org's image_ideogram4_t2i (fp8) / image_ideogram4_t2i_int8 templates, subgraph
"Text to Image (Ideogram v4)" at its 'Default' preset (CustomCombo -> JsonExtractString: V4_DEFAULT_20 = 20 steps,
mu 0.0, std 1.75; the Ideogram4Scheduler widgets [20, 1024, 1024, 0.5, 1.75] are stale, the linked preset runs):

  UNETLoader(ideogram4_fp8_scaled) -> CFGOverride(cfg 3, start 0.7, end 1.0) = model
  UNETLoader(ideogram4_unconditional_fp8_scaled) = model_negative
  CLIPLoader(qwen3vl_8b_fp8_scaled, type ideogram4) -> CLIPTextEncode(prompt) -> positive; ConditioningZeroOut -> negative
  DualModelGuider(model, model_negative, positive, negative, cfg 7) ; Ideogram4Scheduler(steps, W, H, mu, std)
  RandomNoise ; KSamplerSelect(euler) ; EmptyFlux2LatentImage(W, H) -> SamplerCustomAdvanced -> VAEDecode(flux2-vae)

Two models per step (conditional + unconditional), so 2 passes per step; CFG 7, overridden to 3 for the last 30%
(sigma <= 0.3). Size: the fp8 template's ResolutionSelector is 9:16 at 1 MP, multiple 8 = 768x1368, then the subgraph
rounds each side up to 16 (min 256) = 768x1376; the int8 template is 1:1 = 1024x1024. Width / height multiples of 16.
The template ships a JSON-structured caption as the prompt and a caption-template helper that feeds nothing; the
benchmark prompt goes to CLIPTextEncode as given on both stacks.
Files: dit, dit2 (unconditional) in diffusion_models; te; vae. graph_extra: mu (0.0), std (1.75),
cfg_override {cfg, start, end} (3, 0.7, 1.0).
"""

from . import SAVE, GraphParams, pick, save_node, wrap_model

TEMPLATE = "image_ideogram4_t2i.json"
DEFAULTS = {"sampler": "euler", "cfg": 7.0, "steps": 20}
FILES = {"dit": "diffusion_models", "dit2": "diffusion_models", "te": "text_encoders", "vae": "vae"}


def build(p: GraphParams) -> dict:
    import sys

    mod = sys.modules[__name__]
    x = p.extra or {}
    if int(p.width) % 16 or int(p.height) % 16:
        raise ValueError(f"ideogram-4: width / height must be multiples of 16, got {p.width}x{p.height}")
    ov = {"cfg": 3.0, "start": 0.7, "end": 1.0, **(x.get("cfg_override") or {})}
    g = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": p.files["dit"], "weight_dtype": p.weight_dtype}},
        "unet_neg": {"class_type": "UNETLoader", "inputs": {"unet_name": p.files["dit2"], "weight_dtype": p.weight_dtype}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": p.files["te"], "type": "ideogram4", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": p.files["vae"]}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": p.prompt}},
        "neg": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["pos", 0]}},
        "latent": {"class_type": "EmptyFlux2LatentImage",
                   "inputs": {"width": int(p.width), "height": int(p.height), "batch_size": 1}},
        "sigmas": {"class_type": "Ideogram4Scheduler",
                   "inputs": {"steps": int(p.steps), "width": int(p.width), "height": int(p.height),
                              "mu": float(x.get("mu", 0.0)), "std": float(x.get("std", 1.75))}},
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": int(p.seed)}},
        "ksel": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": pick(p, mod, "sampler")}},
        "override": {"class_type": "CFGOverride",
                     "inputs": {"model": ["unet", 0], "cfg": float(ov["cfg"]), "start_percent": float(ov["start"]),
                                "end_percent": float(ov["end"])}},
    }
    model = wrap_model(g, ["override", 0], p)
    g["guider"] = {"class_type": "DualModelGuider",
                   "inputs": {"model": model, "model_negative": ["unet_neg", 0], "positive": ["pos", 0],
                              "negative": ["neg", 0], "cfg": float(pick(p, mod, "cfg"))}}
    g["sampler"] = {"class_type": "SamplerCustomAdvanced",
                    "inputs": {"noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["ksel", 0],
                               "sigmas": ["sigmas", 0], "latent_image": ["latent", 0]}}
    g["decode"] = {"class_type": "VAEDecode", "inputs": {"samples": ["sampler", 0], "vae": ["vae", 0]}}
    g[SAVE] = save_node(["decode", 0], p.prefix)
    return g
