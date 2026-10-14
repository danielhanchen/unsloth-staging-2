"""sd.cpp --mmap A/B on this Mac with the pinned prebuilt. Usage: sdcpp_mmap.py <sd-cli> <gguf> <base_root> <head_root> <out_dir>"""
import json, os, re, subprocess, sys

binary, gguf, base_root, head_root, out_dir = sys.argv[1:6]
os.makedirs(out_dir, exist_ok = True)
COMMON = ["-p", "a photo of a red fox in the snow", "--steps", "1", "--cfg-scale", "1.0", "-W", "512", "-H", "512",
          "--seed", "42", "--sampling-method", "euler_a"]

BUILD = r'''
import json, sys
sys.path.insert(0, sys.argv[1] + "/studio/backend")
from core.inference.sd_cpp_args import SdCppGenParams, SdCppModelFiles, build_sd_cpp_command
cmd = build_sd_cpp_command(sys.argv[2], SdCppModelFiles(diffusion_model = sys.argv[3]),
    SdCppGenParams(prompt = "a photo of a red fox in the snow", width = 512, height = 512, steps = 1, cfg_scale = 1.0,
                   seed = 42, sampling_method = "euler_a"), output_path = sys.argv[4])
print(json.dumps(cmd))
'''


def studio_argv(root, out_png):
    p = subprocess.run([sys.executable, "-c", BUILD, root, binary, gguf, out_png], capture_output = True, text = True)
    if p.returncode:
        raise SystemExit(f"argv build failed for {root}: {p.stderr[-2000:]}")
    argv = json.loads(p.stdout.strip().splitlines()[-1])
    # sd-turbo is a whole checkpoint: same argv, loaded as -m (Studio's placement / speed / mmap flags untouched).
    return ["-m" if a == "--diffusion-model" else a for a in argv]


def timed(name, argv):
    log = os.path.join(out_dir, name + ".log")
    p = subprocess.run(["/usr/bin/time", "-l", *argv], capture_output = True, text = True, timeout = 1500)
    open(log, "w").write(p.stdout + "\n----stderr----\n" + p.stderr)
    t = p.stderr
    rss = re.search(r"(\d+)\s+maximum resident set size", t)
    fp = re.search(r"(\d+)\s+peak memory footprint", t)
    real = re.search(r"([\d.]+) real", t)
    return {"rc": p.returncode, "max_rss_mib": int(rss.group(1)) / 2**20 if rss else None,
            "peak_footprint_mib": int(fp.group(1)) / 2**20 if fp else None,
            "real_s": float(real.group(1)) if real else None, "mmap_flag": "--mmap" in argv,
            "backend_cpu": "--backend" in argv, "argv_tail": argv[argv.index("--output") + 2:] if "--output" in argv else None}


res = {}
for name, extra in [("cpu_nommap", ["--backend", "cpu"]), ("cpu_mmap", ["--backend", "cpu", "--mmap"]),
                    ("metal_nommap", []), ("metal_mmap", ["--mmap"])]:
    png = os.path.join(out_dir, name + ".png")
    res[name] = timed(name, [binary, "-m", gguf, *COMMON, *extra, "-o", png])
    res[name]["png"] = png
for arm, root in (("studio_base", base_root), ("studio_head", head_root)):
    png = os.path.join(out_dir, arm + ".png")
    argv = studio_argv(root, png)
    res[arm] = timed(arm, argv)
    res[arm]["png"] = png
    res[arm]["argv"] = argv

import numpy as np
from PIL import Image


def px(name):
    p = res[name]["png"]
    return np.asarray(Image.open(p).convert("RGB")).astype(np.int16) if res[name]["rc"] == 0 and os.path.exists(p) else None


def diff(a, b):
    x, y = px(a), px(b)
    return None if x is None or y is None else int(np.abs(x - y).max())


res["maxdiff_cpu_mmap_vs_nommap"] = diff("cpu_mmap", "cpu_nommap")
res["maxdiff_metal_mmap_vs_nommap"] = diff("metal_mmap", "metal_nommap")
res["maxdiff_studio_head_vs_base"] = diff("studio_head", "studio_base")
print(json.dumps(res, indent = 1))
json.dump(res, open(os.path.join(out_dir, "sdcpp_mmap.json"), "w"), indent = 1)
