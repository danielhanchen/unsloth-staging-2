"""Hard gates over the job's JSON. Exit 1 on any failure; prints a verdict table."""
import json, sys
import numpy as np

fails = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


b = json.load(open("out/cap_base.json"))
h = json.load(open("out/cap_head.json"))
print(json.dumps({"base": b, "head": h}, indent = 1))
check(h["error"] is not None and h["studio_classifies_oom"], "head: overrun raises an OOM Studio classifies as OOM")
check(h["allocated_gib"] < h["memsize_gib"], f"head: stops below RAM ({h['allocated_gib']:.2f} < {h['memsize_gib']:.2f} GiB)")
check(h["post_matmul_finite"], "head: process still computes after the OOM")
check(b["post_matmul_finite"], "base: process alive after fill (recorded)")
print(f"INFO base reached_limit={b['reached_limit']} allocated={b['allocated_gib']:.2f} GiB swap {b['swap_before_mib']} -> {b['swap_after_fill_mib']} MiB error={b['error']}")

nb = json.load(open("out/normal_base.json"))
nh = json.load(open("out/normal_head.json"))
ib, ih = np.load("out/normal_base.npy"), np.load("out/normal_head.npy")
md = float(np.abs(ib.astype(np.float64) - ih.astype(np.float64)).max())
print("normal", nb, nh, "maxdiff", md)
check(nh["finite"] and nb["finite"], "normal sd-turbo load renders on both arms")
check(md == 0.0, f"normal render identical base vs head (max abs diff {md})")

s = json.load(open("out/sdcpp/sdcpp_mmap.json"))
for k in ("cpu_nommap", "cpu_mmap", "studio_base", "studio_head"):
    check(s[k]["rc"] == 0, f"sd.cpp {k} rc 0")
check(s["maxdiff_cpu_mmap_vs_nommap"] == 0, f"sd.cpp CPU backend --mmap pixel-identical ({s['maxdiff_cpu_mmap_vs_nommap']})")
check(s["maxdiff_studio_head_vs_base"] == 0, f"Studio argv head vs base pixel-identical ({s['maxdiff_studio_head_vs_base']})")
check(s["studio_head"]["mmap_flag"] and not s["studio_base"]["mmap_flag"], "Studio argv: head passes --mmap, base does not")
check(s["cpu_mmap"]["peak_footprint_mib"] < s["cpu_nommap"]["peak_footprint_mib"],
      f"sd.cpp CPU peak footprint drops with --mmap ({s['cpu_nommap']['peak_footprint_mib']:.0f} -> {s['cpu_mmap']['peak_footprint_mib']:.0f} MiB)")
check(s["studio_head"]["peak_footprint_mib"] < s["studio_base"]["peak_footprint_mib"],
      f"Studio sd.cpp peak footprint head < base ({s['studio_base']['peak_footprint_mib']:.0f} -> {s['studio_head']['peak_footprint_mib']:.0f} MiB)")
for k in ("metal_nommap", "metal_mmap"):
    print(f"INFO {k}: rc={s[k]['rc']} footprint={s[k]['peak_footprint_mib']} rss={s[k]['max_rss_mib']}")
print(f"INFO metal maxdiff {s['maxdiff_metal_mmap_vs_nommap']}")
sys.exit(1 if fails else 0)
