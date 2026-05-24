# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""PR #5727 cross-platform smoke gate.

The PR adds save_method="merged_16bit" to two MLX save_pretrained_merged
calls in studio/backend/core/export/export.py:export_base_model. Those
two lines only execute on Apple Silicon (gated on _IS_MLX), so on
Linux/Windows runners the test surface is:

  (a) The patched file parses + py_compiles.
  (b) The patched kwarg is actually present in the source (lock in the
      regression so a future revert is caught).
  (c) ExportBackend imports cleanly with the Studio backend dep set.
      tests/conftest.py CUDA-spoofs the device_type chain so this works
      on a runner with no CUDA / XPU / MPS.
"""

from __future__ import annotations

import ast
import importlib
import platform
import py_compile
import sys
from pathlib import Path

import pytest

# The actual module-import test requires the full unsloth dep stack
# (bitsandbytes on Linux, triton on Windows). On non-Mac platforms the
# patched lines never execute (gated on _IS_MLX), so AST + py_compile +
# kwarg-presence are the only meaningful signal there; the real import
# test is reserved for macOS Apple Silicon where it actually means
# something. The macos-14 workflow runs run_pr5727_non_peft_base_export.py
# which already loads ExportBackend end-to-end on real MLX.
_IS_APPLE_ARM = (
    platform.system() == "Darwin" and platform.machine() == "arm64"
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPORT_PY = REPO_ROOT / "studio" / "backend" / "core" / "export" / "export.py"


def test_export_py_ast_parses():
    """Surface a syntax regression immediately."""
    ast.parse(EXPORT_PY.read_text())


def test_export_py_compiles(tmp_path):
    py_compile.compile(str(EXPORT_PY), cfile = str(tmp_path / "export.pyc"), doraise = True)


def test_pr5727_kwarg_present():
    """Lock in the PR fix: both MLX save_pretrained_merged calls in
    export_base_model must pass save_method='merged_16bit'. Counted at
    source level so a future commit that drops either one is caught."""
    src = EXPORT_PY.read_text()
    occurrences = src.count('save_method = "merged_16bit"')
    assert occurrences >= 2, (
        f"expected >=2 occurrences of save_method = \"merged_16bit\" "
        f"in studio/backend/core/export/export.py, got {occurrences}. "
        f"PR #5727 added two such occurrences (lines 478 and 514 in the "
        f"fix-mlx-studio-export branch); a drop suggests a regression."
    )


def test_pr5727_kwargs_inside_export_base_model_mlx_branch():
    """AST-level: both occurrences must live inside
    ExportBackend.export_base_model, specifically under an `if _IS_MLX:`
    branch, on save_pretrained_merged calls. This is stronger than
    a string count -- it prevents the kwarg from drifting elsewhere
    while the fix gets accidentally reverted in export_base_model."""
    tree = ast.parse(EXPORT_PY.read_text())

    target_fn: ast.FunctionDef | None = None
    for cls in ast.walk(tree):
        if isinstance(cls, ast.ClassDef) and cls.name == "ExportBackend":
            for item in cls.body:
                if isinstance(item, ast.FunctionDef) and item.name == "export_base_model":
                    target_fn = item
                    break
            break
    assert target_fn is not None, "missing ExportBackend.export_base_model"

    def _has_save_method_merged_16bit(call: ast.Call) -> bool:
        for kw in call.keywords:
            if (
                kw.arg == "save_method"
                and isinstance(kw.value, ast.Constant)
                and kw.value.value == "merged_16bit"
            ):
                return True
        return False

    def _is_save_pretrained_merged(call: ast.Call) -> bool:
        return (
            isinstance(call.func, ast.Attribute)
            and call.func.attr == "save_pretrained_merged"
        )

    def _walk_under_is_mlx(node: ast.AST) -> list[ast.Call]:
        """Collect Call nodes that live inside an `if _IS_MLX:` (or
        nested under one) within node."""
        results: list[ast.Call] = []
        # Stack of (node, under_is_mlx)
        stack: list[tuple[ast.AST, bool]] = [(node, False)]
        while stack:
            cur, under = stack.pop()
            if isinstance(cur, ast.If):
                this_under = under
                if (
                    isinstance(cur.test, ast.Name) and cur.test.id == "_IS_MLX"
                ):
                    this_under = True
                for child in cur.body:
                    stack.append((child, this_under))
                for child in cur.orelse:
                    stack.append((child, under))
                continue
            if isinstance(cur, ast.Call) and under:
                results.append(cur)
            for child in ast.iter_child_nodes(cur):
                stack.append((child, under))
        return results

    mlx_calls = _walk_under_is_mlx(target_fn)
    matched = [
        c
        for c in mlx_calls
        if _is_save_pretrained_merged(c) and _has_save_method_merged_16bit(c)
    ]
    assert len(matched) >= 2, (
        f"expected >=2 save_pretrained_merged calls with "
        f"save_method='merged_16bit' inside the `if _IS_MLX:` branches of "
        f"ExportBackend.export_base_model, found {len(matched)}. "
        f"PR #5727 added these to fix the non-PEFT MLX base-export bug."
    )


@pytest.mark.skipif(
    not _IS_APPLE_ARM,
    reason = (
        "ExportBackend full import requires the unsloth dep stack "
        "(bitsandbytes on Linux, triton on Windows) which we do not "
        "install on the cross-platform smoke; the change is _IS_MLX-gated "
        "anyway, so AST + py_compile + kwarg-presence already cover non-Mac "
        "platforms. The macos-14 workflow runs the real ExportBackend "
        "end-to-end."
    ),
)
def test_export_backend_imports():
    """ExportBackend imports with the Studio backend dep set. On a
    runner with no real CUDA/XPU/MPS, tests/conftest.py:140 has
    already pre-loaded unsloth_zoo.device_type under a mocked
    torch.cuda.is_available()=True so the import chain survives."""
    studio_dir = REPO_ROOT / "studio"
    if str(studio_dir) not in sys.path:
        sys.path.insert(0, str(studio_dir))
    mod = importlib.import_module("backend.core.export.export")
    assert hasattr(mod, "ExportBackend"), "ExportBackend missing from imported module"
    backend_cls = mod.ExportBackend
    assert hasattr(backend_cls, "export_base_model"), "ExportBackend.export_base_model missing"
    assert hasattr(backend_cls, "export_merged_model"), "ExportBackend.export_merged_model missing"
