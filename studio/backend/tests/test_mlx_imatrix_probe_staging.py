# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Staging-only: prove PR 8603's thesis against the REAL MLX binding on macOS.

The PR's own suites drive fake savers, which can only show that the probe's *logic* is
right. They cannot show the thing the PR is actually about: that `_supports_kwarg` returns
True for `unsloth_zoo.mlx.loader._mlx_save_pretrained_gguf` even when that binding drops
`imatrix_file` on the floor. This file asserts that against the installed unsloth_zoo, so
the answer changes with the unsloth_zoo version pinned by the workflow -- which is the
version-skew matrix the fix exists for.

Not proposed for the PR: it is pinned to unsloth_zoo internals and only means anything on
a runner that can install mlx.
"""

import ast
import importlib.util
import inspect
import json
import os
import platform
import sys
from pathlib import Path

import pytest


IS_DARWIN = sys.platform == "darwin"
IS_ARM_MAC = IS_DARWIN and platform.machine() == "arm64"


# -- the two probes, lifted from the PR under review -------------------------------------


def _export_py():
    """studio/backend/core/export/export.py, found from this file's repo root."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "studio" / "backend" / "core" / "export" / "export.py"
        if candidate.is_file():
            return candidate
    pytest.fail(f"could not locate studio/backend/core/export/export.py from {here}")


def _probes():
    """`_supports_kwarg` + `_imatrix_export_supported` exec'd from source.

    Importing export.py needs the whole Studio backend and its stub harness, which replaces
    `unsloth_zoo` with a fake -- exactly the module whose real contents decide the answer
    here. Lifting the two functions keeps the real one installed.
    """
    source = _export_py().read_text(encoding = "utf-8")
    tree = ast.parse(source)
    wanted = {"_supports_kwarg", "_imatrix_export_supported"}
    namespace: dict = {}
    found = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            exec(compile(ast.Module(body = [node], type_ignores = []), "<export.py>", "exec"), namespace)
            found[node.name] = namespace[node.name]
    missing = wanted - set(found)
    if missing:
        pytest.fail(f"export.py is missing {sorted(missing)} -- is this the PR head?")
    return found["_supports_kwarg"], found["_imatrix_export_supported"]


# -- the real MLX binding ----------------------------------------------------------------


def _mlx_loader_path():
    spec = importlib.util.find_spec("unsloth_zoo")
    if spec is None or not spec.submodule_search_locations:
        return None
    path = Path(list(spec.submodule_search_locations)[0]) / "mlx" / "loader.py"
    return path if path.is_file() else None


def _binding_from_source():
    """`_mlx_save_pretrained_gguf` compiled from the installed source, no package import.

    `unsloth_zoo.mlx.loader` drags in torch/torchao at import, which is not the thing under
    test and fails for unrelated reasons on some runners. `inspect.signature` on a plain
    function is decided by the def, so the exec'd copy answers the probe identically -- and
    the real object is checked too, whenever it imports.
    """
    path = _mlx_loader_path()
    if path is None:
        return None, None
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    namespace: dict = {}
    out = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in (
            "_mlx_save_pretrained_gguf", "_mlx_supported_kwargs",
        ):
            exec(compile(ast.Module(body = [node], type_ignores = []), str(path), "exec"), namespace)
            out[node.name] = namespace[node.name]
    return out.get("_mlx_save_pretrained_gguf"), out.get("_mlx_supported_kwargs")


def _resolver_present():
    """Exactly the import `_imatrix_export_supported` makes, plus why it failed."""
    try:
        from unsloth_zoo.llama_cpp import resolve_imatrix_file  # noqa: F401
        return True, None
    except Exception as exception:
        return False, f"{type(exception).__name__}: {exception}"


@pytest.fixture(scope = "module")
def binding():
    save_fn, filter_fn = _binding_from_source()
    if save_fn is None:
        if IS_ARM_MAC:
            pytest.fail("unsloth_zoo/mlx/loader.py is not installed on an arm64 mac runner")
        pytest.skip("unsloth_zoo mlx loader not installed on this runner")
    return save_fn, filter_fn


# -- the assertions ----------------------------------------------------------------------


def test_the_old_guard_is_a_false_positive_on_the_real_mlx_binding(binding):
    """Why the bug was invisible: the merge base's guard says yes to a binding that drops it.

    This is the merge-base row of the skew matrix and does not depend on the zoo version.
    """
    save_fn, _filter = binding
    supports_kwarg, _probe = _probes()

    assert supports_kwarg(save_fn, "imatrix_file") is True, (
        "_supports_kwarg no longer accepts the MLX binding -- the premise of PR 8603 has "
        "changed and the rest of this file is meaningless"
    )
    params = inspect.signature(save_fn).parameters
    assert "imatrix_file" not in params
    assert any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


def test_the_binding_actually_discards_the_keyword(binding):
    """The mechanism: the allow-list, not the signature, decides what survives."""
    _save_fn, filter_fn = binding
    if filter_fn is None:
        pytest.skip("_mlx_supported_kwargs not found in the installed loader")

    kept = filter_fn({"imatrix_file": "/tmp/imatrix.gguf", "first_conversion": "bf16"},
                     ("first_conversion",))
    assert "imatrix_file" not in kept, (
        "the MLX allow-list now keeps imatrix_file; if unsloth_zoo has been fixed, this "
        "file's expectations need revisiting"
    )
    assert kept == {"first_conversion": "bf16"}


def test_the_new_probe_tracks_the_installed_unsloth_zoo(binding, record_property):
    """The row this run contributes: probe answer == whether the zoo can really apply it."""
    save_fn, _filter = binding
    _supports_kwarg, imatrix_export_supported = _probes()

    present, reason = _resolver_present()
    verdict = imatrix_export_supported(save_fn)

    try:
        import unsloth_zoo
        zoo_version = getattr(unsloth_zoo, "__version__", "unknown")
    except Exception:
        zoo_version = "unimportable"

    row = {
        "platform": f"{sys.platform}/{platform.machine()}",
        "unsloth_zoo": zoo_version,
        "unsloth_zoo_pin": os.environ.get("ZOO_PIN", "unset"),
        "resolve_imatrix_file_importable": present,
        "resolve_import_error": reason,
        "old_supports_kwarg_says": _supports_kwarg(save_fn, "imatrix_file"),
        "new_probe_says": verdict,
    }
    sys.stderr.write("MATRIX_ROW " + json.dumps(row) + "\n")
    sys.stderr.flush()
    record_property("matrix_row", json.dumps(row))

    assert verdict is present, (
        f"the probe disagrees with the installed unsloth_zoo: probe={verdict}, "
        f"resolve_imatrix_file importable={present} ({reason})"
    )

    # Agreement alone is satisfied by both pins, so the workflow states which row it is
    # and this turns a green tick into the actual claim.
    expected = os.environ.get("EXPECT_RESOLVER")
    if expected is not None:
        assert present is (expected == "1"), (
            f"pin {row['unsloth_zoo_pin']} expected resolve_imatrix_file "
            f"importable={expected == '1'}, got {present} ({reason})"
        )


def test_the_real_module_agrees_with_the_source_copy():
    """Belt and braces: when the package imports, the live object answers the same."""
    try:
        from unsloth_zoo.mlx.loader import _mlx_save_pretrained_gguf as live
    except Exception as exception:
        pytest.skip(f"unsloth_zoo.mlx.loader does not import here: {type(exception).__name__}")

    source_fn, _filter = _binding_from_source()
    assert str(inspect.signature(live)) == str(inspect.signature(source_fn))

    _supports_kwarg, imatrix_export_supported = _probes()
    present, _reason = _resolver_present()
    assert _supports_kwarg(live, "imatrix_file") is True
    assert imatrix_export_supported(live) is present
