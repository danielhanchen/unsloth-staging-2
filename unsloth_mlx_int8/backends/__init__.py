# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""W8A8 backends.

`portable` implements the algorithm in plain MLX ops and runs anywhere, which is what
makes the arithmetic testable without an M5. `metal_mpp` is the real one.
"""

import logging
import os

logger = logging.getLogger(__name__)

_backend = None


def select(name=None):
    """Return the backend module. `name` defaults to UNSLOTH_MLX_INT8_BACKEND, then to
    metal_mpp with a fallback to portable if Metal kernels cannot be imported."""
    global _backend
    if name is None and _backend is not None:
        return _backend

    name = name or os.environ.get("UNSLOTH_MLX_INT8_BACKEND", "metal_mpp")
    if name == "portable":
        from . import portable as mod
    elif name == "metal_mpp":
        from . import metal_mpp as mod
    else:
        raise ValueError(f"unknown backend {name!r}")

    _backend = mod
    return mod


def reset():
    global _backend
    _backend = None
