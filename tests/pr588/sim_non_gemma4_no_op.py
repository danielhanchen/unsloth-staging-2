"""Sim test: confirm non-Gemma4/non-Qwen3.5 paths are no-ops.

Covers Llama, Qwen2, Qwen3, Gemma3, Mistral3, mllama, Qwen2.5-VL, Qwen3-VL, phi3.
"""
import sys, types
# sys.path: unsloth_zoo is pip-installed in CI

from unsloth_zoo.empty_model import _is_gemma4_config


def _cfg(model_type, text_model_type=None):
    cfg = types.SimpleNamespace(model_type=model_type)
    if text_model_type is not None:
        cfg.text_config = types.SimpleNamespace(model_type=text_model_type)
    else:
        cfg.text_config = cfg
    return cfg


def test_is_gemma4_config_false_for_non_gemma4():
    cases = [
        _cfg("llama"),
        _cfg("qwen2"),
        _cfg("qwen3"),
        _cfg("gemma3"),
        _cfg("mistral3"),
        _cfg("mllama"),
        _cfg("qwen2_5_vl"),
        _cfg("qwen3_vl"),
        _cfg("phi3"),
        _cfg("qwen3_5"),         # Qwen 3.5 itself is NOT Gemma4
        _cfg("vlm", text_model_type="llama"),
    ]
    for c in cases:
        assert _is_gemma4_config(c) is False, f"_is_gemma4_config({c.model_type}) should be False"


def test_is_gemma4_config_true_only_for_gemma4_variants():
    assert _is_gemma4_config(_cfg("gemma4")) is True
    assert _is_gemma4_config(_cfg("vlm", text_model_type="gemma4")) is True
    assert _is_gemma4_config(_cfg("vlm", text_model_type="gemma4_text")) is True


def test_is_gemma4_config_none_returns_false():
    assert _is_gemma4_config(None) is False


def test_is_gemma4_config_no_attrs_returns_false():
    bare = object()
    assert _is_gemma4_config(bare) is False


def test_get_gemma4_bnb_skip_module_aliases_non_gemma4():
    from unsloth_zoo.vllm_utils import _get_gemma4_bnb_skip_module_aliases
    assert _get_gemma4_bnb_skip_module_aliases(None) is None
    assert _get_gemma4_bnb_skip_module_aliases({}) is None
    # Existing Llama config: skip modules without language_model prefix -> no aliases needed -> None
    assert _get_gemma4_bnb_skip_module_aliases({"llm_int8_skip_modules": ["model.layers.0.mlp"]}) is None
    # Empty list of skip modules
    assert _get_gemma4_bnb_skip_module_aliases({"llm_int8_skip_modules": []}) is None


def test_get_gemma4_bnb_skip_module_aliases_gemma4():
    from unsloth_zoo.vllm_utils import _get_gemma4_bnb_skip_module_aliases
    aliased = _get_gemma4_bnb_skip_module_aliases(
        {"llm_int8_skip_modules": ["model.language_model.layers.0.mlp"]}
    )
    assert aliased is not None
    new_skip = aliased["llm_int8_skip_modules"]
    assert "model.language_model.layers.0.mlp" in new_skip
    assert "model.layers.0.mlp" in new_skip
    assert "language_model.model.layers.0.mlp" in new_skip
