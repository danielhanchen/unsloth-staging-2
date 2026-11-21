#!/usr/bin/env python3
"""Map the outside tester's own bench.py result JSONs (rounds up to r5, Nov 17, and the Nov 18 package: baseline
v0.45.4, Studio 96bf6de124d) into schema v2 records, marked legacy.

  python diffusion_bench/import_tester.py RESULTS_DIR --out DIR [--gpu-labels '0=Ada 48GB,1=RTX 3090']
      [--bline-tag 2aa4d535=v0.45.4] [--studio-rev 96bf6de124d] [--schema-dir DIR]

One record per tester JSON at DIR/<name>/record.json plus DIR/import_summary.json. Validated with
schema.validate when diffusion_bench/schema.py is importable (or --schema-dir points at one); otherwise a local
skeleton with the same top-level keys is written and the summary says validation was skipped.

What maps (tester field -> v2):
  stack studio|bline                    -> cell.backend studio|blineui, arm.stack
  outcome ok|error, error               -> verdict ok|BROKEN, reason
  wl {model, w, h, steps, frames, ...}  -> cell.model / width / height / steps / frames / guidance
  cfg (configs.json entry)              -> cell.options.tester_config (+ the config name)
  gpu index                             -> target "tester:<label>" and gpu.name (labels from --gpu-labels)
  sim_total_mib / free_mib_target       -> memory.budget {gib, simulated: true} and target.simulated_budget_gib
  requests kind cold|warmup|new_prompt|same_prompt -> cold|warmup|new|same; total_s, seed, idx -> prompt_id;
           pre_sampler_s -> phases.te_s (submit to first step: text encode + queue), post_sampler_s -> phases.vae_s,
           sampler_s or the rest -> phases.denoise_s; s_per_step -> step_s_polled (mean of the polled steps;
           per-step lists were not kept, so step_s stays []); executed_nodes -> bline.executed_nodes
  summary / requests                    -> timing.load_s, cold_s, new_s_median, steady_s_median (new prompts),
                                           step_s_polled_median
  cell_peak_vram_mib / _over_base_mib   -> memory.peak_vram_device_gib / peak_vram_over_base_gib;
  max request peak_rss_mib              -> memory.peak_rss_tree_gib
  request clocks                        -> telemetry sm_mhz_mean (mean), temp_c_max (max), power_w_mean (mean) over
                                           the new-prompt requests; per request sm_mhz_mean / power_w_mean
  code.studio_rev / bline_rev           -> revisions.studio / blineui_sha (+ blineui_tag from --bline-tag, else
                                           code.bline_describe when it is a release tag)
  studio_packages (Studio cells)        -> revisions.zoo / torch / triton
  cell_peak_vram_over_base_mib + base   -> memory.peak_vram_device_gib when cell_peak_vram_mib is absent
  cell_peak_host_mib                    -> legacy.host_used_peak_gib only: it is the host's used RAM, not the
                                           server's RSS, so it never fills memory.peak_rss_tree_gib
Not available in the tester's harness (left null, never invented): GPU UUID / driver / PCIe, torch / CUDA / triton
versions, scripts SHA, parity evidence, engagement levers, per-step lists.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402 - path containment for tester-supplied names

HERE = Path(__file__).resolve().parent
KIND = {"cold": "cold", "warmup": "warmup", "warm": "warmup", "new_prompt": "new", "new": "new",
        "same_prompt": "same", "same": "same"}
STACK = {"studio": "studio", "bline": "blineui", "blineui": "blineui"}
DEFAULT_LABELS = {"0": "Ada 48GB", "1": "RTX 3090"}  # the r5 tester's box (HANDOFF.md tables)


def load_schema(schema_dir: str | None):
    for d in ([schema_dir] if schema_dir else []) + [str(HERE)]:
        if d not in sys.path:
            sys.path.insert(0, d)
    try:
        import schema  # noqa: PLC0415

        return schema
    except ImportError:
        return None


def skeleton(cell: dict) -> dict:
    keys = ("schema_version", "tag", "verdict", "reason", "cell", "target", "arm", "revisions", "gpu", "requests",
            "timing", "memory", "telemetry", "parity", "engagement", "media")
    rec = {k: None for k in keys}
    rec.update({"schema_version": 2, "tag": cell.get("tag"), "cell": cell, "requests": [], "parity": {}, "media": [],
                "revisions": {}, "gpu": {}, "timing": {}, "memory": {"budget": {"gib": None, "simulated": False}},
                "telemetry": {}, "engagement": {"levers": {}, "status_after": None, "bline_log_lines": []}})
    return rec


def fin(x):
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else None


def gib(mib):
    v = fin(mib)
    return round(v / 1024.0, 4) if v is not None else None


def med(vals):
    vals = [v for v in vals if fin(v) is not None]
    return round(statistics.median(vals), 4) if vals else None


def mean(vals):
    vals = [v for v in vals if fin(v) is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def convert(d: dict, src: Path, labels: dict, bline_tags: dict, studio_rev: str | None, schema) -> dict:
    stack = STACK.get(d.get("stack"), d.get("stack"))
    wl = d.get("wl") or {}
    label = labels.get(str(d.get("gpu")), f"gpu{d.get('gpu')}")
    sim = fin(d.get("sim_total_mib")) or None
    budget_gib = round(sim / 1024.0, 3) if sim else None
    video = bool(wl.get("frames"))
    config = str(d.get("config") or "default")
    # one comparison per workload x GPU x budget, the variant = the tester config: report.py groups on both, so
    # 8 / 12 / 16 / 24 GiB and full-card runs, and different configs, never share a ratio
    comparison = f"{d.get('workload') or wl.get('model') or 'workload'}_{label}_" + (f"b{budget_gib:g}" if budget_gib
                                                                                      else "full")
    cell = {"tag": d["name"], "backend": stack, "model": wl.get("model"), "kind": "video" if video else "image",
            "variant": config, "budget_gib": budget_gib,
            "width": wl.get("w"), "height": wl.get("h"), "steps": wl.get("steps"), "frames": wl.get("frames"),
            "guidance": wl.get("guidance"),
            "options": {"tester_config": d.get("config"), "tester_cfg": d.get("cfg"), "workload": d.get("workload"),
                        "target": {"name": f"tester:{label}", "vm_id": f"tester-{label}", "lane": None,
                                   "account": None, "simulated_budget_gib": budget_gib},
                        "arm": {"stack": stack, "block": None, "order": "legacy", "aa_of": None,
                                "comparison": comparison, "format": config}}}
    rec = schema.new_record(cell) if schema else skeleton(cell)
    rec["tag"] = d["name"]
    rec["target"] = dict(cell["options"]["target"])
    rec["arm"] = dict(cell["options"]["arm"])
    outcome = d.get("outcome")
    rec["verdict"] = "ok" if outcome == "ok" else "BROKEN"
    rec["reason"] = None if outcome == "ok" else str(d.get("error") or f"tester outcome {outcome!r}")[:2000]
    rec["legacy"] = {"source": "tester bench.py", "file": str(src), "round": d["name"].rsplit("_", 1)[-1],
                     "started": d.get("started"), "finished": d.get("finished"), "mem_max": d.get("mem_max"),
                     "cooldown": d.get("cooldown"), "start_temp_c": d.get("start_temp_c"),
                     "free_mib_target": d.get("free_mib_target"), "free_at_start_mib": d.get("free_at_start_mib"),
                     "server_ready_s": d.get("server_ready_s"),
                     "note": "phases approximate: te_s = submit to first step (incl. queue), vae_s = after last step"}
    code = d.get("code") or {}
    csha = code.get("bline_rev") or None
    # the Nov 18 package also writes `git describe --tags` (v0.45.4) and both trees' dirty bits
    desc = code.get("bline_describe")
    tag = next((t for p, t in bline_tags.items() if csha and csha.startswith(p)), None) or \
        (desc if isinstance(desc, str) and desc.startswith("v") and "-g" not in desc else None)
    pk = d.get("studio_packages") if isinstance(d.get("studio_packages"), dict) else {}
    rec["revisions"].update({"studio": code.get("studio_rev") or studio_rev, "blineui_sha": csha, "blineui_tag": tag})
    if stack == "studio":  # the Studio venv's versions (studio_packages); the baseline venv's are not recorded
        rec["revisions"].update({"zoo": pk.get("unsloth_zoo"), "torch": pk.get("torch"), "triton": pk.get("triton")})
    rec["legacy"].update({"studio_dirty": code.get("studio_dirty"), "bline_dirty": code.get("bline_dirty"),
                          "bline_describe": desc, "studio_packages": pk or None, "gpu_name": d.get("gpu_name"),
                          "driver": d.get("driver"), "base_used_mib": d.get("base_used_mib"),
                          "idle_used_mib": d.get("idle_used_mib"),
                          # psutil.virtual_memory().used: the whole host's used RAM, NOT the server's RSS, so it is
                          # kept here and never mapped to memory.peak_rss_tree_gib
                          "host_used_peak_gib": gib(d.get("cell_peak_host_mib"))})
    rec["gpu"].update({"name": label})
    reqs = []
    for r in d.get("requests") or []:
        kind = KIND.get(r.get("kind"))
        if kind is None:
            continue
        total = fin(r.get("total_s"))
        pre, post = fin(r.get("pre_sampler_s")), fin(r.get("post_sampler_s"))
        den = fin(r.get("sampler_s"))
        if den is None and None not in (total, pre, post):
            den = round(total - pre - post, 4)
        q = {"kind": kind, "seed": r.get("seed"), "prompt_id": r.get("idx"), "total_s": total, "steps": wl.get("steps"),
             "step_s": [], "step_s_polled": fin(r.get("s_per_step")),
             "phases": {"te_s": pre, "denoise_s": den, "vae_s": post},
             "peak_vram_over_base_gib": gib(r.get("peak_vram_over_base_mib")), "peak_rss_gib": gib(r.get("peak_rss_mib")),
             "sm_mhz_mean": fin((r.get("clocks") or {}).get("sm_mhz_mean")),
             "power_w_mean": fin((r.get("clocks") or {}).get("power_mean"))}
        if stack == "blineui":
            q["bline"] = {"executed_nodes": r.get("executed_nodes"), "sampler_cached": None}
        if r.get("image_file"):
            rec["media"].append(r["image_file"])
        reqs.append(q)
    rec["requests"] = reqs
    new = [q for q in reqs if q["kind"] == "new"]
    summ = d.get("summary") or {}
    load = d.get("load") or {}
    rec["timing"].update({
        "load_s": fin(summ.get("load_s")) if fin(summ.get("load_s")) is not None else fin(load.get("load_s")),
        "cold_s": fin(summ.get("cold_total_s")),
        "new_s_median": med([q["total_s"] for q in new]),
        "steady_s_median": med([q["total_s"] for q in new]),
        "step_s_polled_median": med([q["step_s_polled"] for q in new]),
        # the tester harness measured no long / short pair: derived s/step is simply absent (UNAVAILABLE), not
        # flagged as a step cache the data does not show
        "step_s_derived": None, "step_s_derived_flag": None})
    rss = [fin(r.get("peak_rss_mib")) for r in d.get("requests") or []] + [fin(load.get("peak_rss_mib"))]
    rss = [x for x in rss if x is not None]
    dev = fin(d.get("cell_peak_vram_mib"))
    if dev is None and None not in (fin(d.get("cell_peak_vram_over_base_mib")), fin(d.get("base_used_mib"))):
        dev = d["cell_peak_vram_over_base_mib"] + d["base_used_mib"]  # the Nov 18 package writes over-base + base
    rec["memory"].update({"peak_vram_device_gib": gib(dev),
                          "peak_vram_over_base_gib": gib(d.get("cell_peak_vram_over_base_mib")),
                          "peak_rss_tree_gib": gib(max(rss)) if rss else None,
                          "budget": {"gib": budget_gib, "simulated": bool(sim)}})
    # requests[].clocks: NVML means over the request (None when no sample above 500 MHz fell inside it); the
    # telemetry means are over the new-prompt requests, the timed ones, else every request
    timed = [r for r in d.get("requests") or [] if KIND.get(r.get("kind")) == "new" and r.get("clocks")]
    clocks = [r.get("clocks") or {} for r in (timed or d.get("requests") or [])]
    temps = [c.get("temp_max") for c in clocks if fin(c.get("temp_max")) is not None]
    rec["telemetry"].update({"sm_mhz_mean": mean([c.get("sm_mhz_mean") for c in clocks]),
                             "temp_c_max": max(temps) if temps else None,
                             "power_w_mean": mean([c.get("power_mean") for c in clocks]),
                             "samples": None, "source": "tester bench.py per-request NVML means"})
    if stack == "studio":
        final = [((r.get("resp_meta") or {}).get("final") or {}) for r in d.get("requests") or []]
        meta = next((f.get("video") or f.get("image") for f in reversed(final) if f.get("video") or f.get("image")), None)
        if isinstance(meta, dict):
            keep = ("model", "model_kind", "transformer_quant", "text_encoder_quant", "memory_mode", "offload_policy",
                    "gguf_filename", "steps", "guidance", "flow_shift", "width", "height", "num_frames", "fps")
            rec["engagement"]["status_after"] = {k: meta.get(k) for k in keep if k in meta}
    return rec


def parse_map(s: str | None, default: dict) -> dict:
    if not s:
        return dict(default)
    return {k.strip(): v.strip() for k, _, v in (p.partition("=") for p in s.split(",")) if k.strip()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results")
    ap.add_argument("--out", required = True)
    ap.add_argument("--gpu-labels", default = None, help = "GPU index -> card label, e.g. '0=Ada 48GB,1=RTX 3090'")
    ap.add_argument("--bline-tag", default = None, help = "BlineUI sha prefix -> release tag, e.g. 2aa4d535=v0.45.4")
    ap.add_argument("--studio-rev", default = None, help = "Studio commit when the JSONs do not carry one")
    ap.add_argument("--schema-dir", default = None, help = "dir holding schema.py (default: this toolkit)")
    a = ap.parse_args(argv)
    schema = load_schema(a.schema_dir)
    labels = parse_map(a.gpu_labels, DEFAULT_LABELS)
    tags = parse_map(a.bline_tag, {})
    out = Path(a.out)
    out.mkdir(parents = True, exist_ok = True)
    rows, ok, bad = [], 0, 0
    for p in sorted(Path(a.results).glob("*.json")):
        try:
            d = json.loads(p.read_text())
            rec = convert(d, p, labels, tags, a.studio_rev, schema)
        except Exception as exc:  # noqa: BLE001 - one unreadable file is a row in the summary, not a crash
            rows.append({"file": p.name, "status": "failed", "why": f"{type(exc).__name__}: {exc}"})
            bad += 1
            continue
        d_out = C.contained(out, rec.get("tag"))
        if d_out is None or d_out == out.resolve() or "/" in str(rec.get("tag")) or "\\" in str(rec.get("tag")):
            # a tester-supplied name ("../x", "/abs", "C:/x", "a/b") never becomes a path outside one --out/<tag>
            rows.append({"file": p.name, "status": "failed",
                         "why": f"unsafe result name {rec.get('tag')!r}: {C.unsafe_relpath(rec.get('tag')) or 'nested'}"})
            bad += 1
            continue
        problems = schema.validate(rec) if schema else None
        d_out.mkdir(parents = True, exist_ok = True)
        text = schema.dumps(rec, indent = 1) if schema and hasattr(schema, "dumps") else json.dumps(rec, indent = 1,
                                                                                                     allow_nan = False)
        (d_out / "record.json").write_text(text)
        status = "ok" if not problems else "invalid"
        ok += status == "ok"
        bad += status != "ok"
        rows.append({"file": p.name, "tag": rec["tag"], "verdict": rec["verdict"], "status": status,
                     "problems": problems, "reason": rec.get("reason")})
    summary = {"results_dir": a.results, "count": len(rows), "valid": ok, "failed_or_invalid": bad,
               "validated_with": getattr(schema, "__file__", None) if schema else "SKIPPED: schema.py not importable",
               "verdicts": {v: sum(1 for r in rows if r.get("verdict") == v) for v in ("ok", "BROKEN")}, "rows": rows}
    (out / "import_summary.json").write_text(json.dumps(summary, indent = 1))
    print(f"{len(rows)} files: {ok} valid, {bad} failed / invalid; verdicts {summary['verdicts']}; "
          f"schema {summary['validated_with']}")
    for r in rows:
        if r["status"] != "ok":
            print(f"  {r['file']}: {r['status']} {r.get('why') or r.get('problems')}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
