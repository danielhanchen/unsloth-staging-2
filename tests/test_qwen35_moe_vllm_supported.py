# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""`qwen3_5_moe` has to be in `VLLM_SUPPORTED_VLM`, and the gate is exact membership.

Qwen3.5 / 3.6 MoE checkpoints ship as `Qwen3_5MoeForConditionalGeneration` with a
`vision_config`, so `fast_inference = True` reaches the VLM gate in
`unsloth/models/vision.py`. That gate is `any(arch in VLLM_SUPPORTED_VLM for arch in
model_types)`, a plain list membership test, so the already-listed `"qwen3_5"` does
not admit the `"qwen3_5_moe"` model_type: without its own entry the load raises
"Fast inference is only supported for ..." before vLLM is ever started.

Read by AST rather than by import, deliberately. `import unsloth` runs the package
`__init__`, which on Linux and Windows pulls in triton and probes the accelerator,
and on macOS arm64 takes the MLX branch instead; none of that is available or
relevant on a bare CI runner, and none of it is what this assertion is about. The
parse reads the very source file that ships, so it is a real assertion about the
shipped constant, not a restatement of a fixture. The cost is that a `VLLM_SUPPORTED_VLM`
rewritten into something an AST literal cannot evaluate (a comprehension, a runtime
`+=`) fails this suite loudly rather than silently passing, which is the correct
direction to fail in.

Runs on CPU with no GPU, no vLLM, no transformers, no torch: stdlib plus pytest.
"""

from __future__ import annotations

import ast
import functools
import importlib.util
import os
import pathlib

import pytest

_RELATIVE = pathlib.Path("unsloth") / "models" / "vision.py"


def _find_vision_py() -> pathlib.Path:
    """Locate the shipped `unsloth/models/vision.py`.

    Order: an explicit `UNSLOTH_REPO_ROOT`, then a walk up from this file (the
    normal in-repo `tests/` layout), then the installed package located through
    the import system WITHOUT executing `unsloth/__init__.py`.
    """
    root = os.environ.get("UNSLOTH_REPO_ROOT")
    if root:
        candidate = pathlib.Path(root).expanduser().resolve() / _RELATIVE
        if candidate.is_file():
            return candidate

    for parent in pathlib.Path(__file__).resolve().parents:
        candidate = parent / _RELATIVE
        if candidate.is_file():
            return candidate

    # find_spec on the top-level package locates it without running its __init__.
    spec = importlib.util.find_spec("unsloth")
    locations = list(getattr(spec, "submodule_search_locations", None) or [])
    for location in locations:
        candidate = pathlib.Path(location) / "models" / "vision.py"
        if candidate.is_file():
            return candidate

    pytest.fail(
        "could not locate unsloth/models/vision.py; set UNSLOTH_REPO_ROOT to the "
        "repository root"
    )


@functools.lru_cache(maxsize=1)
def _vision_module_ast() -> tuple[pathlib.Path, ast.Module]:
    path = _find_vision_py()
    return path, ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


@functools.lru_cache(maxsize=1)
def _supported_vlm() -> list:
    _, tree = _vision_module_ast()
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if "VLLM_SUPPORTED_VLM" not in targets:
            continue
        value = ast.literal_eval(node.value)
        assert isinstance(value, list), (
            f"VLLM_SUPPORTED_VLM is a {type(value).__name__}, not a list"
        )
        return value
    pytest.fail("no module-level VLLM_SUPPORTED_VLM assignment in unsloth/models/vision.py")


# Every entry the constant carried before the Qwen3.5 MoE change, from
# `git show origin/main:unsloth/models/vision.py`. Frozen as data so a shallow CI
# checkout with no history still gets a real before/after.
BASE_SUPPORTED = (
    "qwen2_5_vl",
    "gemma3",
    "mistral3",
    "qwen3_vl",
    "qwen3_vl_moe",
    "qwen3_5",
)


def test_qwen3_5_moe_is_listed():
    supported = _supported_vlm()
    assert "qwen3_5_moe" in supported, (
        "Qwen3.5 / 3.6 MoE checkpoints report model_type qwen3_5_moe; without this "
        "entry fast_inference = True is refused at the VLM gate"
    )


def test_previously_supported_models_are_untouched():
    """Additive only: nothing that already worked may have been dropped."""
    supported = _supported_vlm()
    missing = [name for name in BASE_SUPPORTED if name not in supported]
    assert not missing, f"VLLM_SUPPORTED_VLM lost entries: {missing}"
    assert len(supported) == len(set(supported)), f"duplicate entries: {supported}"
    assert all(isinstance(name, str) for name in supported)


def test_the_gate_is_exact_membership_not_a_prefix_match():
    """`qwen3_5` cannot stand in for `qwen3_5_moe`, which is why the entry is needed.

    Stated as a property of the list, not of the string: with the new entry removed,
    the exact-membership predicate the gate uses rejects the MoE model_type.
    """
    supported = _supported_vlm()

    def gate(model_types):
        # Mirrors unsloth/models/vision.py's `any(arch in VLLM_SUPPORTED_VLM for arch in model_types)`.
        return any(arch in supported for arch in model_types)

    assert gate(["qwen3_5_moe"])

    without_moe = [name for name in supported if name != "qwen3_5_moe"]
    assert "qwen3_5" in without_moe
    assert not any(arch in without_moe for arch in ["qwen3_5_moe"]), (
        "the prefix qwen3_5 must NOT admit qwen3_5_moe; if it does, the gate stopped "
        "being an exact membership test and this entry's justification changed"
    )

    # A genuinely unsupported architecture is still refused.
    assert not gate(["some_unsupported_vlm"])


def test_gate_source_really_uses_list_membership():
    """Check the claim above against the source, rather than assuming it.

    Every use of `VLLM_SUPPORTED_VLM` in the module must be the right-hand side of
    an `in` comparison. A `startswith` / substring / regex rewrite would change what
    the added entry means, and would slip past a test that only inspected the list.
    """
    path, tree = _vision_module_ast()

    membership_uses = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            for op, comparator in zip(node.ops, node.comparators):
                if (
                    isinstance(op, (ast.In, ast.NotIn))
                    and isinstance(comparator, ast.Name)
                    and comparator.id == "VLLM_SUPPORTED_VLM"
                ):
                    membership_uses += 1

    total_loads = sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and node.id == "VLLM_SUPPORTED_VLM"
        and isinstance(node.ctx, ast.Load)
    )

    assert membership_uses >= 1, f"no `arch in VLLM_SUPPORTED_VLM` comparison in {path}"
    assert membership_uses == total_loads, (
        f"{total_loads - membership_uses} use(s) of VLLM_SUPPORTED_VLM in {path} are not "
        "exact list membership; the gate's semantics changed"
    )


def test_entry_is_reachable_from_a_real_qwen3_5_moe_model_type_list():
    """`model_types` comes from the config chain, MoE type first, `siglip` possible.

    The gate scans every entry, so the ordering the loader produces still admits the
    model. This is the shape the failing load actually had.
    """
    supported = _supported_vlm()
    for model_types in (
        ["qwen3_5_moe"],
        ["qwen3_5_moe", "qwen3_5"],
        ["siglip", "qwen3_5_moe"],
    ):
        assert any(arch in supported for arch in model_types), model_types
