"""sd.cpp --mmap A/B on this Mac with the pinned prebuilt. Usage: sdcpp_mmap.py <sd-cli> <gguf> <base_root> <head_root> <out_dir>"""
import json, os, re, subprocess, sys

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


def studio_argv(binary, gguf, root, out_png):
    p = subprocess.run([sys.executable, "-c", BUILD, root, binary, gguf, out_png], capture_output = True, text = True)
    if p.returncode:
        raise SystemExit(f"argv build failed for {root}: {p.stderr[-2000:]}")
    argv = json.loads(p.stdout.strip().splitlines()[-1])
    # sd-turbo is a whole checkpoint: same argv, loaded as -m (Studio's placement / speed / mmap flags untouched).
    return ["-m" if a == "--diffusion-model" else a for a in argv]


def timed(out_dir, name, argv):
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


