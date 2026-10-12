set -x
mkdir -p out
sysctl -n hw.model machdep.cpu.brand_string hw.memsize; system_profiler SPDisplaysDataType | head -20
python bench/probe.py out/probe_${LEG}.json 2>&1 | grep -E "cpu|mps_avail|recommended|gemm_mps|weight_int8pack|sdpa_512"
python -c "import torch, diffusers; print(torch.__version__, diffusers.__version__, torch.backends.mps.is_available())"
python bench/studio_int8_ab.py fetch 2>&1 | tail -3
(cd bench/head_backend && python -m pytest -q -p no:cacheprovider -rs tests/test_diffusion_native_quant_mps.py -k "not image_gate" 2>&1 | tail -15) | tee out/pytest_${LEG}.txt
for side in $SIDES; do
  for round in ${ROUNDS:-1 2 3}; do
    case $round in 1) arms="dense base_native head_native";; 2) arms="head_native dense base_native";; 3) arms="base_native head_native dense";; esac
    for a in $arms; do python bench/studio_int8_ab.py run $a $side out/ab_${LEG}_${side}_${a}_r${round}.json 2>&1 | tail -1; done
  done
done
python - <<'PY'
import json, glob, statistics, torch, os
leg = os.environ["LEG"]
rows = {}
for f in sorted(glob.glob(f"out/ab_{leg}_*_r*.json")):
    r = json.load(open(f)); rows.setdefault((r["side"], r["arm"]), []).append(r)
summ = {}
for (side, arm), rs in sorted(rows.items()):
    meds = [r["med_ms"] for r in rs]
    summ[f"{side}_{arm}"] = {"round_medians_ms": meds, "median_ms": statistics.median(meds), "alloc_gib": rs[0].get("alloc_gib"),
                            "swapped": rs[0].get("swapped"), "int8pack_calls_per_step": rs[0].get("int8pack_calls_per_step")}
for side in {k[0] for k in rows}:
    d = torch.load(f"out/ab_{leg}_{side}_dense_r1.pt")
    for arm in ("base_native", "head_native"):
        o = torch.load(f"out/ab_{leg}_{side}_{arm}_r1.pt")
        summ[f"{side}_{arm}"]["rel_l2_vs_dense"] = float((o - d).norm() / d.norm())
    hb = torch.load(f"out/ab_{leg}_{side}_head_native_r1.pt"); bb = torch.load(f"out/ab_{leg}_{side}_base_native_r1.pt")
    summ[f"{side}_head_vs_base_native_rel_l2"] = float((hb - bb).norm() / bb.norm())
json.dump(summ, open(f"out/summary_{leg}.json", "w"), indent = 1)
print(json.dumps(summ, indent = 1))
PY
rm -f out/*.pt
