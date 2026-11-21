#!/usr/bin/env python3
"""Build a card format's baseline-UI files from the official repos (the card's ``bline.sources``), or list the steps.

  python diffusion_bench/builders/run.py --card FAM --format FMT --out DIR [--file NAME] [--dry-run]
      [--tool quant_int8_convrot.py] [--python BLINE_PY] [--bline-dir BLINE_DIR] [--tmp DIR] [--fetch]
  python diffusion_bench/builders/run.py --list

Without --fetch only the files with a build step are made; --fetch also downloads the fetched-as-is files, so DIR
holds the whole format. A file whose source is null (no official route) is reported and skipped (exit 1 if any).
HF_TOKEN from the environment is passed to the steps (gated vendor repos).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import builders as BL  # noqa: E402


def main(argv = None) -> int:
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action = "store_true", help = "every distinct build step and the card files using it")
    ap.add_argument("--card")
    ap.add_argument("--format")
    ap.add_argument("--file", help = "one file name of the format (default: all)")
    ap.add_argument("--out")
    ap.add_argument("--tmp")
    ap.add_argument("--dry-run", action = "store_true")
    ap.add_argument("--fetch", action = "store_true", help = "also download the fetched-as-is files")
    ap.add_argument("--tool", default = None)
    ap.add_argument("--python", default = None)
    ap.add_argument("--bline-dir", default = None)
    a = ap.parse_args(argv)
    if a.list:
        for step, uses in sorted(BL.card_steps().items()):
            mod = BL.parse_build(uses[0][3]["build"])["module_step"]
            print(f"{step}\n    -> {mod}")
            for fam, fmt, name, _ in uses:
                print(f"    {fam} {fmt} {name}")
        return 0
    if not (a.card and a.format and a.out):
        ap.error("--card, --format and --out are required (or --list)")
    from models import load_card

    card = load_card(a.card)
    srcs = ((card["formats"].get(a.format) or {}).get("bline") or {}).get("sources") or {}
    if not srcs:
        ap.error(f"{card['family']} {a.format}: no baseline-UI sources")
    names = [a.file] if a.file else sorted(srcs)
    missing = 0
    for n in names:
        e = srcs.get(n)
        if e is None:
            ap.error(f"{n} is not a file of {card['family']} {a.format}: {sorted(srcs)}")
        if "source" in e and e["source"] is None:
            print(f"SKIP {n}: no official source ({e.get('reason')})", flush = True)
            missing += 1
            continue
        if not e.get("build") and not (a.fetch or a.file):
            continue
        p = BL.build_file(e, n, a.out, tmp = a.tmp, token = os.environ.get("HF_TOKEN"), dry_run = a.dry_run,
                          tool = a.tool, python = a.python, bline_dir = a.bline_dir)
        print(f"{'PLANNED' if a.dry_run else 'READY'} {p}", flush = True)
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
