#!/usr/bin/env python3
"""Score a matrix output: quality against a reference cell, speed and memory from each record, sanity flags.

  python diffusion_bench/score.py outputs/dbench/cmp            # reference map from the spec
  python diffusion_bench/score.py DIR --ref s_bf16 --pair s_int8,c_int8 --plot

Quality is paired per prompt (same id, same seed) against the reference cell of the SAME backend by default
(spec "reference": {"studio": "s_bf16", "blineui": "c_bf16"}; a cell's own "ref" field wins over the map, for
specs with several models; --ref overrides for every cell). Two
frameworks draw their initial noise differently, so a cross-backend LPIPS is not a quality number: it is
printed as the cross-framework floor only, and any cross-framework delta smaller than it is noise.

Image metrics: LPIPS (alex, the number every past comparison used), SSIM, PSNR, mean / max absolute pixel
difference. Video: the same per frame on the saved frames, averaged per clip, plus a temporal flicker figure
(mean absolute difference between consecutive frames, relative to the reference's own).
Sanity flags on every render regardless of reference: black, constant, NaN-looking (uniform) or a frozen clip.

--pair A,B prints a paired bootstrap 95% CI of LPIPS(A) - LPIPS(B); an interval spanning 0 is a tie.
Writes scores.json and scores.md in the directory (one per pass subdirectory when passes exist).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402
import schema  # noqa: E402


# ---------------------------------------------------------------------------------------------- loading
def load_media(cell_dir: Path, render: dict):
    import numpy as np

    f = render.get("file")
    if not f:
        return None
    path = cell_dir / f
    if not path.exists():
        return None
    if path.suffix == ".npz":
        return np.load(path)["frames"]
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"))[None]  # [1, H, W, 3]


def sanity(arr) -> list:
    """Flags for media that no user would accept, whatever the reference says."""
    import numpy as np

    flags = []
    a = arr.astype(np.float32)
    if a.mean() < 4 and a.std() < 4:
        flags.append("black")
    elif a.std() < 3:
        flags.append("constant")
    per_frame_std = a.reshape(a.shape[0], -1).std(axis = 1)
    if (per_frame_std < 3).any() and not flags:
        flags.append("blank_frame")
    if a.shape[0] > 2:
        motion = np.abs(np.diff(a, axis = 0)).mean()
        if motion < 0.2:
            flags.append("frozen")
    return flags


class Metrics:
    def __init__(self, use_lpips: bool = True):
        self.net = None
        self.device = "cpu"
        if use_lpips:
            try:
                import lpips
                import torch

                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.net = lpips.LPIPS(net = "alex", verbose = False).to(self.device).eval()
            except Exception as exc:  # noqa: BLE001
                C.log(f"LPIPS unavailable ({exc}); SSIM / PSNR only")

    def lpips(self, a, b) -> Optional[float]:
        if self.net is None:
            return None
        import torch

        def t(x):
            return torch.from_numpy(x).permute(0, 3, 1, 2).float().div(127.5).sub(1).to(self.device)

        vals = []
        with torch.no_grad():
            for i in range(0, a.shape[0], 8):
                vals.append(self.net(t(a[i:i + 8]), t(b[i:i + 8])).flatten().cpu())
        return float(torch.cat(vals).mean())

    @staticmethod
    def ssim(a, b) -> Optional[float]:
        try:
            from skimage.metrics import structural_similarity
        except Exception:  # noqa: BLE001
            return None
        return float(statistics.mean(structural_similarity(x, y, channel_axis = 2, data_range = 255) for x, y in zip(a, b)))

    @staticmethod
    def psnr(a, b) -> float:
        import numpy as np

        mse = float(((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean())
        return 99.0 if mse == 0 else round(10 * np.log10(255.0**2 / mse), 3)

    def pair(self, a, b) -> dict:
        import numpy as np

        if a.shape[1:] != b.shape[1:]:
            return {"error": f"size mismatch {list(a.shape[1:3])} vs {list(b.shape[1:3])}"}
        n = min(a.shape[0], b.shape[0])
        a, b = a[:n], b[:n]
        diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
        out = {"lpips": self.lpips(a, b), "ssim": self.ssim(a, b), "psnr": self.psnr(a, b),
               "mean_abs": round(float(diff.mean()), 4), "max_abs": int(diff.max()), "identical": bool(diff.max() == 0)}
        if n > 2:
            fa = float(np.abs(np.diff(a.astype(np.float32), axis = 0)).mean())
            fb = float(np.abs(np.diff(b.astype(np.float32), axis = 0)).mean())
            out["flicker_ratio"] = round(fa / fb, 4) if fb else None
        return out


# ---------------------------------------------------------------------------------------------- scoring
# LPIPS between two renders of the same prompt and seed on the same backend; above this they are different images.
FAR_FROM_REFERENCE = 0.6

def cells_in(root: Path) -> dict:
    """{tag: (cell_dir, record)} for the records directly under ``root``; v1 and v2 both read (strictly finite JSON:
    a record with NaN / Infinity is skipped with a warning, never scored)."""
    import schema

    out = {}
    for rec_path in sorted(root.glob(f"*/{C.RECORD}")):
        try:
            rec = schema.loads(rec_path.read_text())
        except ValueError as exc:
            C.log(f"skip {rec_path}: {exc}")
            continue
        out[rec.get("tag") or rec_path.parent.name] = (rec_path.parent, rec)
    return out


def quality_vs_reference(cell_dir: Path, rec: dict, ref_dir: Path, ref: dict, metrics: "Metrics") -> dict:
    """Mean LPIPS / SSIM / PSNR of ``rec``'s renders against ``ref``'s, paired by prompt id AND seed. Used by
    report.py for "quality against the stack's own bf16". Every render is accounted for: ``n_expected`` renders,
    ``n_paired`` scored, ``dropped`` [{id, reason}] for the rest (no reference render, a different seed,
    unreadable media, a size mismatch); ``size_mismatch`` lists the ids rendered on a different canvas (a quality
    failure, never a skip). ``n`` = n_paired (0 = nothing could be paired)."""
    ref_by = {r["id"]: r for r in ref.get("renders") or []}
    rows, dropped, mismatch = [], [], []
    renders = rec.get("renders") or []
    for r in renders:
        rid = r.get("id")
        if rid not in ref_by:
            dropped.append({"id": rid, "reason": "no reference render with this prompt id"})
            continue
        if r.get("seed") is not None and ref_by[rid].get("seed") is not None and r["seed"] != ref_by[rid]["seed"]:
            dropped.append({"id": rid, "reason": f"seed {r['seed']} != reference seed {ref_by[rid]['seed']}"})
            continue
        a, b = load_media(cell_dir, r), load_media(ref_dir, ref_by[rid])
        if a is None or b is None:
            dropped.append({"id": rid, "reason": "unreadable media" + (" (reference)" if a is not None else "")})
            continue
        m = metrics.pair(a, b)
        if "error" in m:
            dropped.append({"id": rid, "reason": m["error"]})
            if str(m["error"]).startswith("size mismatch"):
                mismatch.append(rid)
            continue
        rows.append(m)
    out: dict = {"n": len(rows), "n_expected": len(renders), "n_paired": len(rows), "dropped": dropped,
                 "size_mismatch": mismatch}
    for key in ("lpips", "ssim", "psnr"):
        vals = [x[key] for x in rows if x.get(key) is not None]
        out[key] = round(statistics.mean(vals), 4) if vals else None
    out["identical"] = f"{sum(x['identical'] for x in rows)}/{len(rows)}" if rows else None
    return out


def bootstrap_diff(a: list, b: list, iters: int = 10000, seed: int = 0) -> dict:
    import random

    rng = random.Random(seed)
    d = [x - y for x, y in zip(a, b)]
    if not d:
        return {}
    means = sorted(statistics.mean(rng.choice(d) for _ in d) for _ in range(iters))
    lo, hi = means[int(0.025 * iters)], means[int(0.975 * iters) - 1]
    return {"mean_diff": round(statistics.mean(d), 5), "ci95": [round(lo, 5), round(hi, 5)], "n": len(d),
            "verdict": "tie" if lo <= 0 <= hi else ("first_worse" if lo > 0 else "first_better")}


def score_dir(root: Path, spec: dict, ref_override: Optional[str], metrics: Metrics, pairs: list) -> dict:
    cells = cells_in(root)
    ref_map = dict(spec.get("reference") or {})
    rows, per_prompt = [], {}
    for tag, (cell_dir, rec) in cells.items():
        backend = rec.get("backend")
        # A cell's own "ref" wins over the per-backend map: a spec with several models needs one reference per
        # model, and scoring Z-Image against a FLUX reference would be a size mismatch at best.
        ref_tag = ref_override or (rec.get("cell") or {}).get("ref") or ref_map.get(backend) or ref_map.get("*")
        row = {"tag": tag, "backend": backend, "verdict": rec.get("verdict"), "ref": ref_tag,
               "new_s": rec.get("wall_s_median"), "steady_s": rec.get("steady_s_median"),
               # a flagged derived s/step (step cache on, non-positive) is never a fallback, as in report.py
               "step_s": ((rec.get("timing") or {}).get("step_s_polled_median")
                          or (None if (rec.get("timing") or {}).get("step_s_derived_flag") else rec.get("step_s_derived"))
                          or rec.get("step_s_reported_median")), "cold_s": rec.get("cold_s"),
               "load_s": rec.get("load_s"), "peak_smi_gib": rec.get("peak_smi_gib"),
               "peak_smi_delta_gib": rec.get("peak_smi_delta_gib"), "peak_own_gib": rec.get("peak_own_gib"),
               "peak_alloc_gib": rec.get("peak_alloc_gib"), "peak_rss_gib": (rec.get("host_end") or {}).get("peak_rss_gib"),
               "gate": (rec.get("gate") or {}).get("gate"), "error": (rec.get("error") or "")[:200], "flags": {},
               "unpaired": []}
        scores = []
        if rec.get("verdict") == "ok":
            ref = cells.get(ref_tag) if ref_tag and ref_tag != tag else None
            ref_renders = {r["id"]: r for r in (ref[1].get("renders", []) if ref else [])}
            for r in rec.get("renders", []):
                arr = load_media(cell_dir, r)
                if arr is None:
                    row["flags"][r["id"]] = ["missing_media"]
                    continue
                flags = sanity(arr)
                if flags:
                    row["flags"][r["id"]] = flags
                if not ref:
                    continue
                if r["id"] not in ref_renders:
                    row["unpaired"].append({"id": r["id"], "reason": "no reference render with this prompt id"})
                    continue
                rs = ref_renders[r["id"]].get("seed")
                if r.get("seed") is not None and rs is not None and r["seed"] != rs:
                    row["unpaired"].append({"id": r["id"], "reason": f"seed {r['seed']} != reference seed {rs}"})
                    continue
                ref_arr = load_media(ref[0], ref_renders[r["id"]])
                if ref_arr is None:
                    row["unpaired"].append({"id": r["id"], "reason": "reference media unreadable"})
                    continue
                m = metrics.pair(arr, ref_arr)
                m["id"] = r["id"]
                if "error" in m:
                    row["unpaired"].append({"id": r["id"], "reason": m["error"]})
                scores.append(m)
        per_prompt[tag] = scores
        good = [s for s in scores if "error" not in s]
        for key in ("lpips", "ssim", "psnr", "mean_abs"):
            vals = [s[key] for s in good if s.get(key) is not None]
            row[key] = round(statistics.mean(vals), 4) if vals else None
        vals = [s["lpips"] for s in good if s.get("lpips") is not None]
        row["lpips_max"] = round(max(vals), 4) if vals else None
        # Garbage that passes the pixel-statistics sanity check (e.g. a video family below its working size) is
        # still far from its own reference; same-seed, same-backend LPIPS above this is not a quality delta.
        far = [s["id"] for s in good if (s.get("lpips") or 0) > FAR_FROM_REFERENCE]
        if far:
            for rid in far:
                row["flags"].setdefault(rid, []).append("far_from_reference")
        row["identical"] = f"{sum(s['identical'] for s in good)}/{len(good)}" if good else None
        flick = [s["flicker_ratio"] for s in good if s.get("flicker_ratio")]
        row["flicker_ratio"] = round(statistics.mean(flick), 4) if flick else None
        errs = sorted({s["error"] for s in scores if "error" in s})
        if errs:
            row["score_error"] = errs[0]
        rows.append(row)
    # Cross-framework floor: every pair of reference cells from different backends.
    floor = {}
    refs = {b: t for b, t in ref_map.items() if t in cells}
    names = sorted(refs)
    for i, b1 in enumerate(names):
        for b2 in names[i + 1:]:
            (d1, r1), (d2, r2) = cells[refs[b1]], cells[refs[b2]]
            by_id = {r["id"]: r for r in r2.get("renders", [])}
            vals = []
            for r in r1.get("renders", []):
                if r["id"] in by_id:
                    a, b = load_media(d1, r), load_media(d2, by_id[r["id"]])
                    if a is not None and b is not None:
                        m = metrics.pair(a, b)
                        if m.get("lpips") is not None:
                            vals.append(m["lpips"])
            if vals:
                floor[f"{b1}:{refs[b1]} vs {b2}:{refs[b2]}"] = round(statistics.mean(vals), 4)
    paired = {}
    for spec_pair in pairs:
        a_tag, b_tag = spec_pair.split(",")
        a = {s["id"]: s.get("lpips") for s in per_prompt.get(a_tag, [])}
        b = {s["id"]: s.get("lpips") for s in per_prompt.get(b_tag, [])}
        common_ids = [i for i in a if i in b and a[i] is not None and b[i] is not None]
        paired[spec_pair] = bootstrap_diff([a[i] for i in common_ids], [b[i] for i in common_ids])
    return {"rows": rows, "per_prompt": per_prompt, "cross_framework_floor": floor, "paired": paired}


def markdown(result: dict, title: str) -> str:
    cols = [("tag", "cell"), ("backend", "backend"), ("verdict", "ok"), ("lpips", "LPIPS"), ("lpips_max", "max"),
            ("ssim", "SSIM"), ("psnr", "PSNR"), ("identical", "identical"), ("new_s", "s/img new"),
            ("steady_s", "s/img steady"), ("step_s", "s/step"), ("cold_s", "cold s"), ("load_s", "load s"),
            ("peak_own_gib", "peak GiB (own procs)"), ("peak_smi_delta_gib", "peak GiB (device delta)"), ("peak_alloc_gib", "peak GiB (torch)"), ("peak_rss_gib", "peak RSS GiB")]
    lines = [f"## {title}", "", "| " + " | ".join(c[1] for c in cols) + " | flags |",
             "|" + "---|" * (len(cols) + 1)]
    for r in result["rows"]:
        flags = "; ".join(f"{k}: {','.join(v)}" for k, v in r["flags"].items())
        if r.get("error"):
            flags = (flags + " " if flags else "") + f"error: {r['error'][:80]}"
        if r.get("gate") == "timeout":
            flags += " gate timeout (GPU busy, timing suspect)"
        if r.get("unpaired"):
            flags += (f" UNPAIRED {len(r['unpaired'])} (mean over the rest only): "
                      + ", ".join(f"{u['id']} ({u['reason'][:40]})" for u in r["unpaired"][:5]))
        lines.append("| " + " | ".join("" if r.get(k) is None else str(r.get(k)) for k, _ in cols) + f" | {flags} |")
    if result["cross_framework_floor"]:
        lines += ["", "Cross-framework LPIPS floor (different noise, not a quality number):"]
        lines += [f"- {k}: {v}" for k, v in result["cross_framework_floor"].items()]
    if result["paired"]:
        lines += ["", "Paired bootstrap, LPIPS(first) - LPIPS(second):"]
        lines += [f"- {k}: {v}" for k, v in result["paired"].items()]
    return "\n".join(lines) + "\n"


def plot(result: dict, path: Path, x: str = "new_s", y: str = "lpips") -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        C.log(f"no matplotlib ({exc}); skipping the plot")
        return
    fig, ax = plt.subplots(figsize = (8, 5))
    for r in result["rows"]:
        if r.get(x) is not None and r.get(y) is not None:
            ax.scatter(r[x], r[y])
            ax.annotate(r["tag"], (r[x], r[y]), fontsize = 7)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.grid(alpha = 0.3)
    fig.tight_layout()
    fig.savefig(path, dpi = 130)


def main() -> int:
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--ref", default = None, help = "reference cell tag for every cell (overrides the spec map)")
    ap.add_argument("--pair", action = "append", default = [], help = "A,B: paired bootstrap CI on LPIPS")
    ap.add_argument("--no-lpips", action = "store_true")
    ap.add_argument("--plot", action = "store_true", help = "LPIPS vs s/image scatter per pass")
    args = ap.parse_args()
    root = Path(C.expand_env(args.dir))
    spec = json.loads((root / "spec.json").read_text()) if (root / "spec.json").exists() else {}
    metrics = Metrics(use_lpips = not args.no_lpips)
    passes = [p["name"] for p in spec.get("passes") or [] if (root / p["name"]).is_dir()] or [""]
    md = []
    for p in passes:
        d = root / p if p else root
        result = score_dir(d, spec, args.ref, metrics, args.pair)
        clean, replaced = schema.sanitize(result)  # strict finite JSON: a NaN LPIPS (blank image) becomes null
        if replaced:
            C.log(f"scores.json: non-finite values written as null at {replaced[:10]}")
        (d / "scores.json").write_text(schema.dumps(clean))
        md.append(markdown(result, f"{spec.get('name', root.name)} {p}".strip()))
        if args.plot:
            plot(result, d / "pareto_lpips_vs_speed.png")
    text = "\n".join(md)
    (root / "scores.md").write_text(text, encoding = "utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
