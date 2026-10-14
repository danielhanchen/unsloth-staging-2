"""One arm: Studio's MPS target (applies the arm's cap), then a real diffusers load + 1-step render of sd-turbo at the
target dtype. Writes the image as .npy plus peak driver memory. Usage: mps_normal_load.py <arm_root> <out_prefix>"""
import json, os, sys, time

root, prefix = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.join(root, "studio", "backend"))
import numpy as np
import torch
from core.inference.diffusion_device import _mps_or_cpu_target

tgt = _mps_or_cpu_target(torch)
assert tgt.device == "mps"
from diffusers import AutoPipelineForText2Image

t0 = time.time()
pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo", torch_dtype = tgt.dtype, variant = "fp16").to("mps")
load_s = time.time() - t0
g = torch.Generator("cpu").manual_seed(42)
t0 = time.time()
img = pipe("a photo of a red fox in the snow", num_inference_steps = 1, guidance_scale = 0.0, height = 512, width = 512,
           generator = g, output_type = "np").images[0]
gen_s = time.time() - t0
np.save(prefix + ".npy", img)
res = {"dtype": str(tgt.dtype), "load_s": round(load_s, 2), "gen_s": round(gen_s, 2),
       "driver_gib": torch.mps.driver_allocated_memory() / 1024**3, "mean": float(img.mean()),
       "finite": bool(np.isfinite(img).all())}
print(json.dumps(res))
json.dump(res, open(prefix + ".json", "w"))
