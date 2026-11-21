"""safetensors for the builders, through extnames: tensor keys and metadata are written under their real spelling
(the server reads them) and read back under the toolkit's (identity outside a public copy; see extnames.py)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import extnames as X  # noqa: E402
from safetensors import safe_open as _safe_open  # noqa: E402
from safetensors.torch import save_file as _save_file  # noqa: E402


def save_file(tensors: dict, filename, metadata = None):
    meta = None if metadata is None else {X.ext(k): X.ext(v) for k, v in metadata.items()}
    return _save_file({X.ext(k): v for k, v in tensors.items()}, filename, metadata = meta)


class _Reader:
    def __init__(self, f):
        self._f = f

    def keys(self):
        return [X.norm(k) for k in self._f.keys()]

    def get_tensor(self, k):
        return self._f.get_tensor(X.ext(k))

    def get_slice(self, k):
        return self._f.get_slice(X.ext(k))

    def metadata(self):
        m = self._f.metadata()
        return None if m is None else {X.norm(k): X.norm(v) for k, v in m.items()}

    def __enter__(self):
        self._f.__enter__()
        return self

    def __exit__(self, *exc):
        return self._f.__exit__(*exc)


def safe_open(path, framework = "pt", device = "cpu"):
    return _Reader(_safe_open(path, framework, device = device))
