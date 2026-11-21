"""Shared reading of diffusion_probe.py observations for the two criteria modules. Pure Python (no numpy): the
criteria run inside differential.py, whichever interpreter the job gave it.

Tolerances come from the environment so a workflow can tighten them without editing the module; defaults are
set from past gfx1151 runs (bf16 repeat LPIPS 0, warm-render spread about 20% with one 524 s outlier on Windows).
"""

from __future__ import annotations

import math
import os
import statistics
from typing import Optional

LPIPS_TOL = float(os.environ.get("DBENCH_AMD_LPIPS_TOL", "0.03"))  # head LPIPS may exceed base's by this much
MAX_SLOWDOWN = float(os.environ.get("DBENCH_AMD_MAX_SLOWDOWN", "1.25"))  # head/base s/image before it is a regression
REPEAT_LPIPS_MAX = float(os.environ.get("DBENCH_AMD_REPEAT_LPIPS_MAX", "0.02"))  # noise-floor gate on *_repeat cells
BAD_FLAGS = {"black", "constant", "blank_frame", "frozen", "missing_media"}

# Every capability a diffusion change can touch. The report's "Not tested here" section is NEEDS minus what the
# host had, so this is deliberately the whole list, not what the runner has.
NEEDS = ["rocm", "gpu", "nvidia", "discrete_gpu", "multi_gpu", "windows", "linux", "windows_rocm_wddm", "mlx", "xpu"]


def states(obs: dict) -> dict:
    return {k: v for k, v in obs.items() if not k.startswith("_") and isinstance(v, dict)}


def measured(o: dict) -> bool:
    return bool(o) and not o.get("skipped_state")


def cells(o: dict) -> dict:
    return (o or {}).get("cells") or {}


def ok(c: Optional[dict]) -> bool:
    return bool(c) and c.get("verdict") == "ok"


def bad_flags(c: Optional[dict]) -> set:
    out = set()
    for flags in ((c or {}).get("flags") or {}).values():
        out |= set(flags) & BAD_FLAGS
    return out


def fmt(v, nd: int = 2) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def thumb_psnr(a: Optional[dict], b: Optional[dict]) -> Optional[float]:
    """Mean PSNR of the 32x32 thumbnails of renders with the same prompt id in two cells (99 = identical)."""
    if not a or not b:
        return None
    by_id = {r.get("id"): r for r in b.get("renders") or []}
    vals = []
    for r in a.get("renders") or []:
        o = by_id.get(r.get("id"))
        if not o or not r.get("thumb") or not o.get("thumb"):
            continue
        if r.get("sha") and r.get("sha") == o.get("sha"):
            vals.append(99.0)
            continue
        x, y = bytes.fromhex(r["thumb"]), bytes.fromhex(o["thumb"])
        if len(x) != len(y) or not x:
            continue
        mse = sum((p - q) ** 2 for p, q in zip(x, y)) / len(x)
        vals.append(99.0 if mse == 0 else 10 * math.log10(255.0 ** 2 / mse))
    return round(statistics.mean(vals), 2) if vals else None


def host_gates(obs: dict, names: tuple = ("base", "head")) -> list:
    """Non-vacuity gates every diffusion verdict needs before any comparison is shown."""
    out = []
    for name in names:
        o = obs.get(name) or {}
        if not measured(o):
            out.append((f"{name} measured", False, "state skipped or missing"))
            continue
        rc = o.get("_probe_rc")
        problems = [p for p in (o.get("_parse_error"), "no output" if o.get("_missing_output") else None) if p]
        out.append((f"{name} probe produced an observation", rc == 0 and not problems,
                    f"rc={rc}" + (f"; {'; '.join(problems)}" if problems else "")
                    + (f"; errors: {str(o.get('errors'))[:200]}" if o.get("errors") else "")))
        host = o.get("host") or {}
        out.append((f"{name} was a real run (not --dry)", not o.get("dry"),
                    "dry run: fake backend, plumbing only" if o.get("dry") else "real cells"))
        out.append((f"{name} ran on ROCm gfx1151", bool(host.get("hip")) and bool(host.get("is_gfx1151")),
                    f"vendor {host.get('vendor')}, torch {host.get('torch')}, hip {host.get('hip')}, "
                    f"arch {host.get('arch')}, device {host.get('device_name')}"))
        models = o.get("models") or {}
        bad = {a: m.get("source") for a, m in models.items() if m.get("source") in ("failed", "refused")}
        out.append((f"{name} models fetched without a token", not bad,
                    ", ".join(f"{a}: {m.get('source')}" for a, m in models.items()) or "no models"))
    b, h = obs.get("base") or {}, obs.get("head") or {}
    if measured(b) and measured(h):
        sb = (b.get("spec") or {}).get("selected") or []
        sh = (h.get("spec") or {}).get("selected") or []
        out.append(("same cells selected at base and head", sb == sh, f"base {len(sb)}, head {len(sh)}"))
        n_ok = sum(1 for c in cells(b).values() if ok(c))
        out.append(("base rendered at least one cell", n_ok > 0,
                    f"{n_ok}/{len(cells(b))} ok at base; with none, there is nothing to regress from"))
        for tag, c in cells(b).items():
            if tag.endswith("_repeat") and ok(c) and c.get("lpips") is not None:
                out.append((f"noise floor {tag} (base) LPIPS <= {REPEAT_LPIPS_MAX}", c["lpips"] <= REPEAT_LPIPS_MAX,
                            f"LPIPS {c['lpips']} vs {c.get('ref')}; above it, differences below it are noise"))
    return out


def side_by_side(obs: dict, extra_cols: bool = True) -> str:
    b, h = obs.get("base") or {}, obs.get("head") or {}
    tags = list(dict.fromkeys(list(cells(b)) + list(cells(h))))
    rows = ["| cell | base | head | base s/img | head s/img | head/base | base s/step | head s/step | base LPIPS | "
            "head LPIPS | base-head thumb PSNR | peak GiB b/h | head engaged | flags |",
            "|" + "---|" * 14]
    for tag in tags:
        cb, ch = cells(b).get(tag) or {}, cells(h).get(tag) or {}
        ratio = (ch.get("new_s") / cb.get("new_s")) if cb.get("new_s") and ch.get("new_s") else None
        st = ch.get("status") or {}
        engaged = ", ".join(f"{k}={st.get(k)}" for k in ("transformer_quant", "transformer_cache", "offload_policy",
                                                        "speed_mode") if st.get(k) not in (None, "", [], {}))
        flags = sorted(bad_flags(cb) | bad_flags(ch))
        err = ch.get("error") or cb.get("error")
        rows.append(f"| {tag} | {cb.get('verdict', '-')} | {ch.get('verdict', '-')} | {fmt(cb.get('new_s'))} | "
                    f"{fmt(ch.get('new_s'))} | {fmt(ratio)} | {fmt(cb.get('step_s'), 3)} | {fmt(ch.get('step_s'), 3)} | "
                    f"{fmt(cb.get('lpips'), 4)} | {fmt(ch.get('lpips'), 4)} | {fmt(thumb_psnr(cb, ch))} | "
                    f"{fmt(cb.get('peak_alloc_gib'))}/{fmt(ch.get('peak_alloc_gib'))} | {engaged or '-'} | "
                    f"{', '.join(flags) or ('error: ' + str(err)[:60] if err else '')} |")
    notes = []
    for name, o in (("base", b), ("head", h)):
        if o.get("skipped_budget"):
            notes.append(f"{name}: not started (budget) {', '.join(o['skipped_budget'])}")
    for name, o in states(obs).items():
        if name not in ("base", "head") and o.get("skipped_state"):
            notes.append(f"{name}: not measured ({o.get('reason')})")
    host = (h.get("host") or b.get("host") or {})
    notes.append(f"host: {host.get('device_name')} {host.get('arch')}, torch {host.get('torch')} hip {host.get('hip')}, "
                 f"{host.get('platform')}")
    notes.append("LPIPS is alex vs the cell's reference from the SAME state; thumb PSNR compares base and head "
                 "renders of the same cell (99 = pixel-identical). s/img is the median of new-prompt renders.")
    return "\n".join(rows) + "\n\n" + "\n".join(f"- {n}" for n in notes)
