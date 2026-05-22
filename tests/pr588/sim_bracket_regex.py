"""Sim test: the new bracket regex in convert_vllm_to_huggingface handles
trailing-digit segments as well as the older middle-digit pattern."""
import re


# The new regex from the PR (line 1469 of vllm_utils.py after the diff)
NEW_REGEX = r"\.([\d]+)(?=\.|$)"
OLD_REGEX = r"\.([\d]{1,})\."


def _convert(s, pattern):
    return re.sub(pattern, lambda x: f"[{x.group(1)}]", s)


def test_middle_index_works_both():
    """Both regexes should bracket middle-index digits identically."""
    s = "model.layers.5.self_attn.q_proj"
    assert _convert(s, NEW_REGEX) == "model.layers[5].self_attn.q_proj"
    # Old regex outputs the same here
    assert re.sub(OLD_REGEX, r"[\1].", s) == "model.layers[5].self_attn.q_proj"


def test_trailing_index_only_new_handles():
    """A name ending in `.{N}` is new-regex-friendly but old-regex misses it."""
    s = "model.visual.merger.linear_fc.1"
    assert _convert(s, NEW_REGEX) == "model.visual.merger.linear_fc[1]"
    # Old regex required a trailing `.` so this would be unchanged
    assert re.sub(OLD_REGEX, r"[\1].", s) == "model.visual.merger.linear_fc.1"


def test_multi_digit_indices():
    s = "model.layers.123.self_attn.q_proj"
    assert _convert(s, NEW_REGEX) == "model.layers[123].self_attn.q_proj"


def test_back_to_back_indices():
    s = "model.blocks.0.layers.5.weight"
    assert _convert(s, NEW_REGEX) == "model.blocks[0].layers[5].weight"


def test_does_not_eat_alpha_segments():
    s = "model.layers0.foo"  # NOT a digit segment after dot - regex needs the dot prefix
    out = _convert(s, NEW_REGEX)
    assert out == s, "alpha segment got mangled"
