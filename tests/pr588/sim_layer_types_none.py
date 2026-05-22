"""Sim test: layer_types=None handling in Gemma4 paths.

The planner flagged two `getattr(text_config, "layer_types", ())` sites. `getattr`
only returns the default when the attribute is *missing*, so an explicit
`text_config.layer_types = None` will short-circuit to `enumerate(None)` -> TypeError.

This is the documented transformers default for many configs.
"""
import sys, types
# sys.path: unsloth_zoo is pip-installed in CI

import pytest
import torch


def _gemma4_cfg_with_layer_types(layer_types_value, k_eq_v=True, num_kv_shared=0, num_hidden=4):
    cfg = types.SimpleNamespace(model_type="gemma4")
    text = types.SimpleNamespace(
        model_type="gemma4_text",
        layer_types=layer_types_value,
        attention_k_eq_v=k_eq_v,
        num_kv_shared_layers=num_kv_shared,
        num_hidden_layers=num_hidden,
    )
    cfg.text_config = text
    return cfg


@pytest.mark.xfail(strict=True, reason="CONFIRMED PR BUG: layer_types=None crashes _get_gemma4_k_eq_v_pairs via enumerate(None). Fix: `getattr(text_config, 'layer_types', ()) or ()`")
def test_get_gemma4_k_eq_v_pairs_with_layer_types_none():
    """Pre-fix concern: would crash on `enumerate(None)`.
    We exercise the function via patch_gemma4_vllm_k_eq_v_support's helper."""
    from unsloth_zoo import empty_model

    # Build a fake BitsAndBytesModelLoader with a no-op _stack_quantization_states
    captured = {}

    class _FakeLoader:
        @staticmethod
        def _stack_quantization_states(self_unused, model, quant_state_dict):
            captured["called"] = True
            return dict(quant_state_dict)

    # Stub the vllm import
    real_module = sys.modules.get("vllm.model_executor.model_loader.bitsandbytes_loader")
    fake_mod = types.ModuleType("vllm.model_executor.model_loader.bitsandbytes_loader")
    fake_mod.BitsAndBytesModelLoader = _FakeLoader
    sys.modules["vllm.model_executor.model_loader.bitsandbytes_loader"] = fake_mod
    try:
        empty_model.patch_gemma4_vllm_k_eq_v_support()
        # Now call _stack_quantization_states with a Gemma4 model whose layer_types=None
        cfg = _gemma4_cfg_with_layer_types(None)
        model = torch.nn.Module()
        model.config = cfg
        # The patched method is now installed
        patched = _FakeLoader._stack_quantization_states
        # Try invoking the wrapper - should NOT TypeError on enumerate(None)
        try:
            result = patched(None, model, {})
        except TypeError as e:
            if "iter" in str(e).lower() or "NoneType" in str(e):
                pytest.fail(f"BUG: enumerate(None) crashes on layer_types=None: {e}")
            raise
    finally:
        if real_module is not None:
            sys.modules["vllm.model_executor.model_loader.bitsandbytes_loader"] = real_module
        else:
            del sys.modules["vllm.model_executor.model_loader.bitsandbytes_loader"]


def test_get_vllm_state_dict_path_with_layer_types_none():
    """Mirror: the same `enumerate(getattr(text_config, "layer_types", ()))` site in vllm_utils
    is triggered inside _get_vllm_state_dict. We don't run that full function (it needs
    a vLLM model), but we exercise the offending expression directly to confirm the bug class."""
    text_config = types.SimpleNamespace(layer_types=None, attention_k_eq_v=True)
    # This is the exact pattern from vllm_utils.py:_get_vllm_state_dict
    try:
        gemma4_k_eq_v_layers = {
            kk
            for kk, layer_type in enumerate(getattr(text_config, "layer_types", ()))
            if layer_type == "full_attention"
        }
        # If we get here, layer_types=None is silently handled (False positive on planner's concern)
        # but actually getattr DOES return None here (not ()), so enumerate(None) -> TypeError
        assert False, "expected TypeError"
    except TypeError as e:
        # Confirms the BUG exists in the PR.
        # Mark this test as XFAIL to signal: "yes, layer_types=None will crash this path
        # if Gemma4 ships a config with layer_types=None set explicitly."
        pytest.xfail(f"CONFIRMED PR BUG: layer_types=None -> {e}; suggested fix: `getattr(...) or ()`")
