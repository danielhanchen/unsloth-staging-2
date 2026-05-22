"""Sim test: scan all changed files for bidi/zero-width Unicode."""
import os
import pytest

# In CI we install unsloth_zoo from the PR head; derive the on-disk source
# location via the module's __file__.
import unsloth_zoo
ROOT = os.path.dirname(os.path.dirname(unsloth_zoo.__file__))
CHANGED = [
    "unsloth_zoo/empty_model.py",
    "unsloth_zoo/hf_utils.py",
    "unsloth_zoo/vllm_utils.py",
]
# tests/test_vllm_to_hf_conversion.py only exists in the source repo, not the
# installed wheel — skip if absent.

BIDI = {
    "LRM (U+200E)": "‎",
    "RLM (U+200F)": "‏",
    "LRE (U+202A)": "‪",
    "RLE (U+202B)": "‫",
    "PDF (U+202C)": "‬",
    "LRO (U+202D)": "‭",
    "RLO (U+202E)": "‮",
    "LRI (U+2066)": "⁦",
    "RLI (U+2067)": "⁧",
    "FSI (U+2068)": "⁨",
    "PDI (U+2069)": "⁩",
    "ZWSP (U+200B)": "​",
    "ZWNJ (U+200C)": "‌",
    "ZWJ (U+200D)": "‍",
}


@pytest.mark.parametrize("path", CHANGED)
def test_no_bidi_or_zerowidth(path):
    full = os.path.join(ROOT, path)
    if not os.path.exists(full):
        pytest.skip(f"{full} not present in this install")
    with open(full, encoding="utf-8") as f:
        src = f.read()
    found = {name: src.count(ch) for name, ch in BIDI.items() if src.count(ch) > 0}
    assert not found, f"hidden unicode in {path}: {found}"
