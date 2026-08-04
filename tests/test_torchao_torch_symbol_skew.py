"""torchao 0.18 must not be able to kill `import unsloth`.

torchao 0.17 guarded the import:

    if torch_version_at_least("2.10.0"):
        from torch.nn.functional import ScalingType, SwizzleType

0.18.0 moved a neighbouring guard to "2.12.0.dev0" and left THIS import
unguarded at module level. Verified against the released sources of both
tags. On any torch below 2.10 it raises

    ImportError: cannot import name 'ScalingType' from 'torch.nn.functional'

and because it surfaces while importing transformers, unsloth_zoo's import
guard catches it, finds no "Unpack" in the text, and re-raises a bare
Exception. `import unsloth` is dead and the message names neither torchao
nor torch. Seen on Colab in Gemma3_(4B)-Vision-GRPO, Qwen3_5_(4B)_Vision and
Qwen3_8B_FP8_GRPO -- three notebooks, one cause, spreading as 0.18 rolls out.

The placeholder deliberately refuses to be used. 0.17 left these names
undefined on old torch, so anything wanting them already raised NameError; a
stub that quietly impersonated a real enum could hand a float8 path a
meaningless value, which is worse than the crash. Import works, use raises
with the version skew spelled out.

Checked against the 0.18.0 source: neither symbol is referenced at module
level, in a class body, or in a default/annotation, so nothing evaluates
them at import time and the strict placeholder is safe.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from unsloth.import_fixes import (          # noqa: E402
    _TORCHAO_TORCH_SYMBOLS,
    _make_torch_symbol_placeholder,
    fix_torchao_torch_symbol_skew,
)

GPU_INIT = ROOT / "unsloth" / "_gpu_init.py"


# ---- the placeholder ------------------------------------------------------

def test_it_can_be_imported():
    """The whole point: satisfy `from torch.nn.functional import X`."""
    ph = _make_torch_symbol_placeholder("ScalingType", "detail here")
    assert ph is not None
    assert ph.__name__ == "ScalingType"


def test_using_it_raises_with_an_actionable_message():
    ph = _make_torch_symbol_placeholder("ScalingType", "torchao 0.18 vs torch 2.9")
    with pytest.raises(RuntimeError) as e:
        ph.DYNAMIC
    msg = str(e.value)
    assert "torchao<0.18" in msg, "the user needs something to actually do"
    assert "ScalingType" in msg


def test_instantiating_it_raises_too():
    ph = _make_torch_symbol_placeholder("SwizzleType", "d")
    with pytest.raises(RuntimeError):
        ph()


def test_it_never_pretends_to_be_a_real_value():
    """A stub returning None or 0 would flow into a float8 config and produce
    silently wrong behaviour instead of an error."""
    ph = _make_torch_symbol_placeholder("ScalingType", "d")
    for attr in ("DYNAMIC", "STATIC", "value", "name"):
        with pytest.raises(RuntimeError):
            getattr(ph, attr)


def test_repr_is_honest():
    ph = _make_torch_symbol_placeholder("ScalingType", "d")
    assert "placeholder" in repr(ph)


def test_it_is_marked_as_ours():
    ph = _make_torch_symbol_placeholder("ScalingType", "d")
    assert getattr(ph, "__unsloth_placeholder__", False) is True


# ---- the gating -----------------------------------------------------------

def test_it_is_a_no_op_when_torch_already_has_the_symbols():
    """On torch >= 2.10 there is nothing to fix, and overwriting the real
    enum with a placeholder that raises would BREAK float8 rather than fix
    anything."""
    import torch.nn.functional as F
    added = [n for n in _TORCHAO_TORCH_SYMBOLS if not hasattr(F, n)]
    if not added:
        assert fix_torchao_torch_symbol_skew() is False


def test_a_healthy_torchao_is_left_alone():
    """0.17 and earlier guard their own import. Patching there would put a
    placeholder into torch for no reason at all."""
    if importlib.util.find_spec("torchao") is None:
        pytest.skip("torchao not installed")
    from importlib.metadata import version
    from packaging.version import Version
    if Version(version("torchao")) < Version("0.18.0"):
        import torch.nn.functional as F
        before = {n: hasattr(F, n) for n in _TORCHAO_TORCH_SYMBOLS}
        assert fix_torchao_torch_symbol_skew() is False
        after = {n: hasattr(F, n) for n in _TORCHAO_TORCH_SYMBOLS}
        assert before == after, "nothing may be added for a healthy torchao"


def test_no_torchao_means_nothing_to_do():
    if importlib.util.find_spec("torchao") is not None:
        pytest.skip("torchao is installed here")
    assert fix_torchao_torch_symbol_skew() is False


def test_it_never_raises():
    """It runs during `import unsloth`. Anything it raises replaces the
    problem it exists to prevent."""
    assert fix_torchao_torch_symbol_skew() in (True, False)


def test_calling_it_twice_is_stable():
    first = fix_torchao_torch_symbol_skew()
    second = fix_torchao_torch_symbol_skew()
    assert second is False or first == second


# ---- the wiring -----------------------------------------------------------

def test_it_runs_before_unsloth_zoo_is_imported():
    """unsloth_zoo is what pulls in transformers and therefore torchao. Called
    after that import, the fix would be pointless."""
    lines = GPU_INIT.read_text(encoding="utf-8").splitlines()
    call = next(i for i, l in enumerate(lines)
                if l.strip() == "fix_torchao_torch_symbol_skew()")
    zoo = next(i for i, l in enumerate(lines)
               if l.strip() == "import unsloth_zoo")
    assert call < zoo, f"called at line {call+1}, too late for line {zoo+1}"


def test_it_is_imported_and_cleaned_up():
    src = GPU_INIT.read_text(encoding="utf-8")
    assert "fix_torchao_torch_symbol_skew," in src, "not imported"
    assert "del fix_torchao_torch_symbol_skew" in src, (
        "every other fix is deleted after use; this one must be too")


def test_the_symbol_list_matches_what_torchao_imports():
    """Both names come from one `from torch.nn.functional import` line in
    torchao 0.18's mx_tensor.py. Shimming only one leaves the import broken."""
    assert set(_TORCHAO_TORCH_SYMBOLS) == {"ScalingType", "SwizzleType"}


# ---- the fix actually unblocks the import --------------------------------

def test_the_real_torchao_018_import_line_is_unblocked(monkeypatch):
    """The decisive test: reproduce torchao 0.18 mx_tensor.py line 39 exactly,
    on this torch, and show it goes from raising to succeeding.

    Everything above tests the gating; this tests that the fix works.
    """
    import torch.nn.functional as F
    import unsloth.import_fixes as IF

    if any(hasattr(F, n) for n in _TORCHAO_TORCH_SYMBOLS):
        pytest.skip("this torch already provides the symbols")

    # the line as torchao 0.18 ships it
    line = "from torch.nn.functional import ScalingType, SwizzleType"

    with pytest.raises(ImportError):
        exec(line, {})

    monkeypatch.setattr(IF, "importlib_version",
                        lambda name: "0.18.0" if name == "torchao" else "0")
    try:
        assert IF.fix_torchao_torch_symbol_skew() is True
        exec(line, {})                      # must not raise now
        from torch.nn.functional import ScalingType
        with pytest.raises(RuntimeError):
            ScalingType.DYNAMIC             # still refuses to be used
    finally:
        for n in _TORCHAO_TORCH_SYMBOLS:
            if getattr(getattr(F, n, None), "__unsloth_placeholder__", False):
                delattr(F, n)


def test_the_cleanup_in_the_test_above_is_real():
    """Guards the fixture, not the product: if the delattr above failed, every
    later test in this session would see a patched torch and pass vacuously."""
    import torch.nn.functional as F
    for n in _TORCHAO_TORCH_SYMBOLS:
        obj = getattr(F, n, None)
        assert not getattr(obj, "__unsloth_placeholder__", False), (
            f"a placeholder for {n} leaked out of a test")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
