#!/usr/bin/env python3
"""Compare DiffusionGemma eval logit dumps against a reference dump.

Usage: compare_logits.py N_VOCAB REF.bin OTHER.bin [OTHER.bin ...] -> one JSON line per OTHER.
"""
import json
import sys

import numpy as np


def load(path, n_vocab):
    a = np.fromfile(path, dtype=np.float32)
    return a.reshape(-1, n_vocab)


def entropy(x):
    x = x.astype(np.float64)
    x = x - x.max(axis=1, keepdims=True)
    p = np.exp(x)
    p /= p.sum(axis=1, keepdims=True)
    return float((-(p * np.log(np.clip(p, 1e-300, None))).sum(axis=1)).mean())


def main():
    n_vocab = int(sys.argv[1])
    ref = load(sys.argv[2], n_vocab)
    for path in sys.argv[3:]:
        try:
            o = load(path, n_vocab)
        except Exception as e:  # noqa: BLE001
            print(json.dumps({"file": path, "error": str(e)}))
            continue
        if o.shape != ref.shape:
            print(json.dumps({"file": path, "error": f"shape {o.shape} vs {ref.shape}"}))
            continue
        finite = bool(np.isfinite(o).all())
        d = np.abs(o.astype(np.float64) - ref)
        print(json.dumps({
            "file": path,
            "finite": finite,
            "max_abs": float(np.nanmax(d)),
            "mean_abs": float(np.nanmean(d)),
            "top1_agree": float((o.argmax(1) == ref.argmax(1)).mean()),
            "entropy_ref": entropy(ref),
            "entropy": entropy(o) if finite else None,
            "ref_absmax": float(np.abs(ref).max()),
        }))


if __name__ == "__main__":
    main()
