#!/usr/bin/env python3
"""Detect runaway POST /api/inference/load entries in a Studio log.

Exit codes:
  0  no spam (count <= threshold)
  1  spam detected (count > threshold)
  2  log file missing / unreadable

Usage:
  log_spam_detector.py --log PATH [--threshold N] [--window SECONDS]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TARGET_PATH = "/api/inference/load"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, type=Path,
                    help="path to studio.log (jsonl request_completed lines)")
    ap.add_argument("--threshold", type=int, default=3,
                    help="max allowed /api/inference/load POSTs (default 3)")
    ap.add_argument("--method", default="POST")
    return ap.parse_args()


def count_target(log_path: Path, method: str) -> tuple[int, list[str]]:
    if not log_path.is_file():
        raise FileNotFoundError(log_path)

    count = 0
    samples: list[str] = []
    with log_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or TARGET_PATH not in line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # Best-effort: substring match counts; surface the raw line.
                count += 1
                if len(samples) < 5:
                    samples.append(line)
                continue
            if obj.get("path") == TARGET_PATH and obj.get("method", method) == method:
                count += 1
                if len(samples) < 5:
                    samples.append(line)
    return count, samples


def main() -> int:
    args = parse_args()
    try:
        count, samples = count_target(args.log, args.method)
    except FileNotFoundError:
        print(f"FAIL: log file not found: {args.log}", file=sys.stderr)
        return 2
    print(f"{TARGET_PATH} {args.method} count = {count}  threshold = {args.threshold}")
    for s in samples:
        print(f"  sample: {s}")
    return 0 if count <= args.threshold else 1


if __name__ == "__main__":
    raise SystemExit(main())
