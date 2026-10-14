"""Repeated, interleaved sd.cpp --mmap A/B (A/A noise first). Usage: sdcpp_aa.py <sd-cli> <gguf> <base_root> <head_root> <out_dir> <repeats>"""
import json, os, statistics, subprocess, sys

sys.argv = sys.argv[:6] + sys.argv[7:] if False else sys.argv
binary, gguf, base_root, head_root, out_dir, reps = sys.argv[1:7]
reps = int(reps)
sys.argv = sys.argv[:6]
import importlib.util
spec = importlib.util.spec_from_file_location("m", os.path.join(os.path.dirname(__file__), "sdcpp_mmap_lib.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
os.makedirs(out_dir, exist_ok = True)
arms = {
    "cpu_nommap": [binary, "-m", gguf, *m.COMMON, "--backend", "cpu"],
    "cpu_mmap": [binary, "-m", gguf, *m.COMMON, "--backend", "cpu", "--mmap"],
}
runs = {k: [] for k in ("cpu_nommap", "cpu_mmap", "studio_base", "studio_head")}
for r in range(reps):
    for name in runs:
        png = os.path.join(out_dir, f"{name}_{r}.png")
        if name.startswith("studio"):
            argv = m.studio_argv(binary, gguf, base_root if name == "studio_base" else head_root, png)
        else:
            argv = arms[name] + ["-o", png]
        res = m.timed(out_dir, f"{name}_{r}", argv)
        res["png"] = png
        runs[name].append(res)
        print(name, r, {k: res[k] for k in ("rc", "peak_footprint_mib", "max_rss_mib", "real_s", "mmap_flag")}, flush = True)
summary = {}
for name, rs in runs.items():
    fp = [x["peak_footprint_mib"] for x in rs]
    summary[name] = {"rc": [x["rc"] for x in rs], "footprint_mib": fp, "footprint_median": statistics.median(fp),
                     "footprint_spread": max(fp) - min(fp), "rss_median": statistics.median([x["max_rss_mib"] for x in rs]),
                     "real_median": statistics.median([x["real_s"] for x in rs]), "mmap_flag": rs[0]["mmap_flag"]}
import numpy as np
from PIL import Image
pngs = [x["png"] for rs in runs.values() for x in rs]
ref = np.asarray(Image.open(pngs[0]).convert("RGB")).astype(np.int16)
summary["max_pixel_diff_all_runs"] = max(int(np.abs(np.asarray(Image.open(p).convert("RGB")).astype(np.int16) - ref).max()) for p in pngs)
print(json.dumps(summary, indent = 1))
json.dump({"summary": summary, "runs": runs}, open(os.path.join(out_dir, "sdcpp_aa.json"), "w"), indent = 1)
fails = []
for name in runs:
    if any(rc != 0 for rc in summary[name]["rc"]): fails.append(f"{name} nonzero rc")
if summary["max_pixel_diff_all_runs"] != 0: fails.append("pixels differ")
for a, b in (("cpu_nommap", "cpu_mmap"), ("studio_base", "studio_head")):
    if not max(summary[b]["footprint_mib"]) < min(summary[a]["footprint_mib"]): fails.append(f"{b} footprint not below every {a} run")
if not (summary["studio_head"]["mmap_flag"] and not summary["studio_base"]["mmap_flag"]): fails.append("argv")
print("FAILS:", fails)
sys.exit(1 if fails else 0)
