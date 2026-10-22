"""Before/after comparison of step evidence: pixels, normalised DOM, facts.

    python studio_regress.py diff --root outputs/studio_regress/pr11606 [--json]

Per step key (`journey/step`), from <root>/{before,after}/<journey>/<step>.{png,dom.json,facts.json}:

  pixels  decoded RGB(A) equality first; otherwise a pixelmatch-style count of pixels whose
          max channel delta / 255 exceeds `threshold` (0.1), ignoring masked rectangles
          recorded in facts["_masks"] ([x, y, w, h] in page pixels). Size mismatch is a diff.
          Changed > min(max_pixels, max_frac * unmasked area) is VISUAL_DIFF.
  dom     normalised a11y/text snapshot equality (volatile text already normalised by fixture).
  facts   keys not starting with "_" compared exactly.

Verdict per step (contract.VERDICTS):
  FAIL_BOTH    step failed on both sides          FAIL_HEAD  passed before, failed after
  DIVERGED     unreachable / skipped after an earlier failure on exactly one side (or evidence
               missing on one side)
  VOID         evidence missing on both sides (the step never ran: proves nothing), or one side's
               scheduler unit crashed / timed out (facts `_status: unit_void`, scheduler.py), or
               the two sides ran different Chromium builds (<side>/<journey>/_browser.json, run.py)
  FLAKY        key listed in data/studio_regress/flaky.json (differs between identical runs) and
               the change is within the envelope those A/A runs recorded (else reported as a diff)
  VISUAL_DIFF  pixels differ over budget           DOM_ONLY_DIFF  DOM/facts differ, pixels same
  SAME         otherwise
A step that passed after but failed before counts as SAME-or-diff by evidence (fix, not regression).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

THRESHOLD = 0.1
MAX_PIXELS = 100
MAX_FRAC = 0.0001  # 0.01%

WS = Path(os.environ.get("WORKSPACE") or Path(__file__).resolve().parents[3])
FLAKY_PATH = WS / "data" / "studio_regress" / "flaky.json"


def _load_png(path):
    import numpy as np
    from PIL import Image

    im = Image.open(path).convert("RGBA")
    return np.asarray(im, dtype = np.int16)


def pixel_diff(
    a_path,
    b_path,
    masks = (),
    threshold = THRESHOLD,
):
    """(changed_pixels, unmasked_area, size_mismatch)."""
    import numpy as np

    a, b = _load_png(a_path), _load_png(b_path)
    if a.shape != b.shape:
        return max(a.shape[0] * a.shape[1], b.shape[0] * b.shape[1]), a.shape[0] * a.shape[1], True
    if np.array_equal(a, b):
        return 0, a.shape[0] * a.shape[1], False
    keep = np.ones(a.shape[:2], dtype = bool)
    for m in masks or ():
        x, y, w, h = (int(round(v)) for v in m)
        keep[max(y, 0) : max(y + h, 0), max(x, 0) : max(x + w, 0)] = False
    delta = np.abs(a - b).max(axis = 2) / 255.0
    changed = int(((delta > threshold) & keep).sum())
    return changed, int(keep.sum()), False


def pixel_budget(
    area,
    max_pixels = MAX_PIXELS,
    max_frac = MAX_FRAC,
):
    return min(max_pixels, int(max_frac * area))


def _read_json(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def facts_delta(before, after):
    before, after = before or {}, after or {}
    out = {}
    for k in sorted(set(before) | set(after)):
        if k.startswith("_"):
            continue
        if before.get(k) != after.get(k):
            out[k] = {"before": before.get(k), "after": after.get(k)}
    return out


def dom_delta(
    before,
    after,
    limit = 12,
):
    """Line-level diff of normalised text lines (a11y tree flattened by fixture)."""
    if before == after:
        return None
    bl = (before or {}).get("lines") or []
    al = (after or {}).get("lines") or []
    bs, as_ = set(bl), set(al)
    return {
        "removed": [l for l in bl if l not in as_][:limit],
        "added": [l for l in al if l not in bs][:limit],
    }


def load_flaky(path = FLAKY_PATH):
    d = _read_json(path) or {}
    return set(d.get("keys") or [])


def load_flaky_map(path = FLAKY_PATH):
    """{key: envelope or None}. The envelope is the largest change the A/A runs saw on that step;
    None (a key recorded before envelopes existed) keeps the old rule: any change is FLAKY."""
    d = _read_json(path) or {}
    env = d.get("envelope") or {}
    return {k: env.get(k) for k in d.get("keys") or []}


# Verdicts an A/A run may mark FLAKY. FAIL_HEAD / DIVERGED / FAIL_BOTH are decided before the
# flaky check, so marking them would only let a later visual change on that step hide.
FLAKY_SOURCES = ("VISUAL_DIFF", "DOM_ONLY_DIFF")


def _envelope_of(step):
    return {
        "pixels": int(step.get("pixels_changed") or 0),
        "dom": bool(step.get("dom_delta")),
        "facts": sorted(step.get("facts_delta") or {}),
        "size": "size" in (step.get("note") or ""),
    }


def record_flaky(
    keys,
    path = FLAKY_PATH,
    steps = None,
):
    """Add `keys` (or, given `steps`, their FLAKY_SOURCES records with the change they showed)."""
    path.parent.mkdir(parents = True, exist_ok = True)
    d = _read_json(path) or {}
    cur = set(d.get("keys") or []) | set(keys or [])
    env = dict(d.get("envelope") or {})
    for s in steps or ():
        if s.get("verdict") not in FLAKY_SOURCES:
            continue
        k, e = s["key"], _envelope_of(s)
        cur.add(k)
        old = env.get(k)
        if old:
            e = {
                "pixels": max(old["pixels"], e["pixels"]),
                "dom": old["dom"] or e["dom"],
                "facts": sorted(set(old["facts"]) | set(e["facts"])),
                "size": old["size"] or e["size"],
            }
        env[k] = e
    path.write_text(json.dumps({"keys": sorted(cur), "envelope": env}, indent = 1))
    return cur


def within_envelope(rec, env, mismatch):
    """True when this change is no bigger than, and of the same kind as, the recorded A/A noise."""
    if env is None:
        return True
    if mismatch and not env.get("size"):
        return False
    if rec["dom_delta"] and not env.get("dom"):
        return False
    if set(rec["facts_delta"]) - set(env.get("facts") or ()):
        return False
    return rec["pixels_changed"] <= 2 * int(env.get("pixels") or 0) + MAX_PIXELS


HEAD_PASSED = "not_run_head_passed"
SKIPPED_NOT_TOUCHED = "skipped_not_touched"  # contract.SKIPPED_NOT_TOUCHED
# scheduler.void_missing_facts: the unit (instance / arm) running this step on that side crashed or
# timed out before the step wrote evidence. Proves nothing about the PR, so VOID, never DIVERGED.
UNIT_VOID = "unit_void"
BROWSER_FILE = (
    "_browser.json"  # run.write_browser_info: {chromium, playwright} the side's journey ran
)


def browser_mismatch(root, journey):
    """ "before Chromium X (Playwright p), after Y (q)" when both sides recorded a different Chromium,
    else "" (same, or either side unrecorded: older evidence is not checked)."""
    b = _read_json(Path(root) / "before" / journey / BROWSER_FILE) or {}
    a = _read_json(Path(root) / "after" / journey / BROWSER_FILE) or {}
    if not (b.get("chromium") and a.get("chromium")) or b["chromium"] == a["chromium"]:
        return ""
    return (
        f"before Chromium {b['chromium']} (Playwright {b.get('playwright')}), "
        f"after {a['chromium']} (Playwright {a.get('playwright')})"
    )


def compare_step(
    root,
    journey,
    step,
    flaky = frozenset(),
    threshold = THRESHOLD,
):
    root = Path(root)
    key = f"{journey}/{step}"
    rec = {
        "key": key,
        "journey": journey,
        "step": step,
        "pixels_changed": 0,
        "dom_delta": None,
        "facts_delta": {},
        "png_before": None,
        "png_after": None,
        "note": "",
    }
    fb = _read_json(root / "before" / journey / f"{step}.facts.json")
    fa = _read_json(root / "after" / journey / f"{step}.facts.json")
    sb = (fb or {}).get("_status", "missing")
    sa = (fa or {}).get("_status", "missing")
    rec["status_before"], rec["status_after"] = sb, sa
    if UNIT_VOID in (sb, sa):
        side = "before" if sb == UNIT_VOID else "after"
        rec["verdict"] = "VOID"
        rec["note"] = (
            f"{side} unit did not finish: "
            + ((fb if side == "before" else fa) or {}).get("_error", "")[:240]
        )
        return rec
    if SKIPPED_NOT_TOUCHED in (sb, sa):
        if sb == sa:
            rec["verdict"] = "SKIPPED_NOT_TOUCHED"
            rec["note"] = ((fa or {}).get("_reason") or (fb or {}).get("_reason") or "")[:300]
        else:  # both sides select from the same diff, so this is a harness fault, not a result
            rec["verdict"] = "DIVERGED"
            rec["note"] = f"before={sb} after={sa}"
        return rec
    other = browser_mismatch(root, journey)
    if other:  # a browser change can move pixels and DOM on its own: proves nothing about the PR
        rec["verdict"] = "VOID"
        rec["note"] = "sides ran different browsers: " + other
        return rec
    if sb == HEAD_PASSED:  # a HEAD_FIRST journey step (run.py): base runs only when head fails
        rec["verdict"] = "SAME" if sa == "ok" else "VOID"
        rec["note"] = "head passed; base not run" if sa == "ok" else f"base skipped but head {sa}"
        return rec
    if sb == "failed" and sa == "failed":
        rec["verdict"] = "FAIL_BOTH"
        return rec
    if sb in ("ok",) and sa == "failed":
        rec["verdict"] = "FAIL_HEAD"
        rec["note"] = (fa or {}).get("_error", "")[:300]
        return rec
    if sb == "failed" and sa == "ok":
        rec["note"] = "fails before, passes after (fix)"
    if sb == sa == "missing":
        # Neither side produced evidence (the side or its isolated run died first): proves nothing.
        rec["verdict"] = "VOID"
        rec["note"] = "no evidence on either side"
        return rec
    if any((sb == x) != (sa == x) for x in ("unreachable", "missing", "skipped_after_failure")):
        # skipped vs failed: head fails a step the base never reached (it broke earlier)
        rec["verdict"] = "DIVERGED"
        rec["note"] = f"before={sb} after={sa}"
        return rec
    rec["facts_delta"] = facts_delta(fb, fa)
    rec["dom_delta"] = dom_delta(
        _read_json(root / "before" / journey / f"{step}.dom.json"),
        _read_json(root / "after" / journey / f"{step}.dom.json"),
    )
    pb = root / "before" / journey / f"{step}.png"
    pa = root / "after" / journey / f"{step}.png"
    visual = mismatch = False
    if pb.exists() and pa.exists():
        masks = list((fb or {}).get("_masks") or []) + list((fa or {}).get("_masks") or [])
        changed, area, mismatch = pixel_diff(pb, pa, masks, threshold)
        rec["pixels_changed"] = changed
        rec["png_before"], rec["png_after"] = str(pb), str(pa)
        visual = mismatch or changed > pixel_budget(area)
        if mismatch:
            rec["note"] = "screenshot size differs"
    elif pb.exists() != pa.exists():
        rec["verdict"] = "DIVERGED"
        rec["note"] = "screenshot on one side only"
        return rec
    # Extra evidence PNGs (crawl overlays, upstream-suite shots), paired by relative name.
    extras = sorted(
        set((fb or {}).get("_extra_png") or []) | set((fa or {}).get("_extra_png") or [])
    )
    changed_extras = []
    for rel in extras:
        xb, xa = root / "before" / rel, root / "after" / rel
        if xb.exists() and xa.exists():
            ch, ar, mm = pixel_diff(xb, xa, (), threshold)
            if mm or ch > pixel_budget(ar):
                changed_extras.append(
                    {
                        "png": rel,
                        "pixels_changed": ch,
                        "size_mismatch": mm,
                        "png_before": str(xb),
                        "png_after": str(xa),
                    }
                )
        else:
            changed_extras.append(
                {"png": rel, "one_side_only": "before" if xb.exists() else "after"}
            )
    rec["main_visual_diff"] = visual  # publish shows the differing overlay when this is False
    if changed_extras:
        rec["extras"] = changed_extras
        # An overlay shot on one side only is click timing (did the popover open in time), not
        # a UI change: listed, but only a paired shot that differs makes the step VISUAL_DIFF.
        visual = visual or any("one_side_only" not in x for x in changed_extras)
    overrun = max((fb or {}).get("_mask_overrun") or 0, (fa or {}).get("_mask_overrun") or 0)
    if overrun:
        # Masks over the 5% budget can hide the step's subject: never silent.
        rec["mask_overrun"] = overrun
        rec["note"] = (
            rec["note"] + "; " if rec["note"] else ""
        ) + f"masks cover {overrun:.0%} of the shot"
    flaky = flaky if isinstance(flaky, dict) else {k: None for k in flaky}
    changed_any = visual or rec["dom_delta"] or rec["facts_delta"]
    if changed_any and key in flaky and not within_envelope(rec, flaky[key], mismatch):
        rec["note"] = (
            rec["note"] + "; " if rec["note"] else ""
        ) + "larger than the A/A noise recorded for this step, so not treated as FLAKY"
    if changed_any and key in flaky and within_envelope(rec, flaky[key], mismatch):
        rec["verdict"] = "FLAKY"
    elif visual:
        rec["verdict"] = "VISUAL_DIFF"
    elif rec["dom_delta"] or rec["facts_delta"]:
        rec["verdict"] = "DOM_ONLY_DIFF"
    else:
        rec["verdict"] = "SAME"
    return rec


def step_keys(root):
    """Union of (journey, step) seen on either side, ordered by manifest when present."""
    root = Path(root)
    man = _read_json(root / "manifest.json") or {}
    keys = [tuple(k.split("/", 1)) for k in man.get("keys") or []]
    seen = set(keys)
    extra = set()
    for side in ("before", "after"):  # shards merged from staging may not carry one manifest
        for f in (root / side).glob("*/*.facts.json"):
            k = (f.parent.name, f.name[: -len(".facts.json")])
            if k not in seen:
                extra.add(k)
    return keys + sorted(extra)


def diff_all(
    root,
    flaky = None,
    threshold = THRESHOLD,
):
    flaky = load_flaky_map() if flaky is None else flaky
    return [compare_step(root, j, s, flaky, threshold) for j, s in step_keys(root)]


def functional_verdict(steps):
    # A step skipped because the diff cannot reach it proves nothing either way.
    steps = [s for s in steps if s["verdict"] != "SKIPPED_NOT_TOUCHED"]
    if any(s["verdict"] == "FAIL_HEAD" for s in steps):
        return "REGRESSION"
    if not steps or all(s["verdict"] == "VOID" for s in steps):
        return "VOID"
    return "NO_REGRESSION"


def summarize(steps):
    out = {}
    for s in steps:
        out[s["verdict"]] = out.get(s["verdict"], 0) + 1
    return out


def receipt_steps(paths):
    """VOID records for staging side jobs that did not finish: `rc=N` receipts (run.py / desktop.py).
    Without this a side that died before writing evidence shows up as DIVERGED ("missing" on one
    side), i.e. as a UI change, not as a run that proved nothing. 0 is done; 1 is a failed step for
    desktop.py (its evidence says which), an uncaught crash for run.py (a single side never exits 1)."""
    out = []
    for p in paths or ():
        p = Path(p)
        try:
            rc = int(p.read_text().strip().rsplit("rc=", 1)[1].split()[0])
        except (OSError, IndexError, ValueError):
            rc = None
        if rc == 0 or (rc == 1 and p.name.startswith("out-desktop-")):
            continue
        name = p.name[len("out-") :] if p.name.startswith("out-") else p.name
        name = name[: -len(".rc")] if name.endswith(".rc") else name
        out.append(
            {
                "key": f"{name}/side",
                "journey": name,
                "step": "side",
                "kind": "side",
                "pixels_changed": 0,
                "dom_delta": None,
                "facts_delta": {},
                "png_before": None,
                "png_after": None,
                "status_before": "not_run",
                "status_after": "not_run",
                "verdict": "VOID",
                "note": f"side job exit {rc if rc is not None else 'unknown'} ({p.name})",
            }
        )
    return out


def main(argv = None):
    import argparse

    p = argparse.ArgumentParser(description = "Compare before/after studio_regress evidence")
    p.add_argument("--root", required = True)
    p.add_argument("--json", action = "store_true")
    p.add_argument("--threshold", type = float, default = THRESHOLD)
    p.add_argument(
        "--record-flaky", action = "store_true", help = "A/A run: record every non-SAME key as FLAKY"
    )
    p.add_argument(
        "--report",
        action = "store_true",
        help = "also write <root>/report.json + summary.md (sides run in separate jobs, e.g. staging)",
    )
    p.add_argument("--pr", type = int)
    p.add_argument("--base-sha")
    p.add_argument("--head-sha")
    p.add_argument(
        "--side-rc",
        nargs = "*",
        default = [],
        help = "staging `rc=N` receipts of the side jobs; one that did not finish is a VOID step",
    )
    a = p.parse_args(argv)
    steps = diff_all(a.root, flaky = set() if a.record_flaky else None, threshold = a.threshold)
    steps += receipt_steps(a.side_rc)
    if a.report:
        from studio_regress import coverage, run  # lazy: run imports this module

        after = Path(a.root) / "after"
        cov = coverage.compute(after) if after.is_dir() else {}
        run.write_report(
            Path(a.root),
            {"pr": a.pr, "base_sha": a.base_sha, "head_sha": a.head_sha},
            steps,
            cov,
            {},
        )
    if a.record_flaky:
        bad = [s for s in steps if s["verdict"] in FLAKY_SOURCES]
        record_flaky((), steps = bad)
        print(f"recorded {len(bad)} flaky keys")
    if a.json:
        print(json.dumps(steps, indent = 1))
    else:
        for s in steps:
            if s["verdict"] != "SAME":
                print(f"{s['verdict']:14} {s['key']}  px={s['pixels_changed']} {s['note']}")
        print("SUMMARY", json.dumps(summarize(steps)), functional_verdict(steps))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
