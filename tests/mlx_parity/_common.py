"""Shared constants + helpers for MLX parity probes.

The probes deliberately share NOTHING with `unsloth_zoo.mlx.trainer` —
each probe re-derives the quantity from first principles so we can tell
where the trainer's wiring differs from the textbook HF/PyTorch recipe.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np


MODEL_NAME = "unsloth/gemma-3-270m-it"
TRAIN_TEXT = "<<HELLO!!>> My name is Unsloth!"
PROMPT = "<<HELLO!!>> My name is "
SEED = 3407
MAX_SEQ_LEN = 64

OUT_DIR = Path(__file__).resolve().parent / ".out"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    try:
        import mlx.core as mx
        mx.random.seed(seed)
    except Exception:
        pass


def banner(title: str) -> None:
    print()
    print("=" * 72)
    print(f"=== {title}")
    print("=" * 72, flush=True)


def section(title: str) -> None:
    print(f"\n--- {title} ---", flush=True)


def report(name: str, value) -> None:
    print(f"  {name}: {value}", flush=True)
