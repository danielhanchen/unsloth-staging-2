"""Sim test: set_dtype_in_config produces JSON-safe values and routes correctly."""
import sys, json
# sys.path: unsloth_zoo is pip-installed in CI

import torch
from transformers import PretrainedConfig
from unsloth_zoo.hf_utils import set_dtype_in_config, dtype_from_config, HAS_TORCH_DTYPE


def test_bfloat16_roundtrip_json():
    cfg = PretrainedConfig()
    set_dtype_in_config(cfg, torch.bfloat16)
    got = dtype_from_config(cfg)
    assert got == "bfloat16"
    json.dumps(cfg.to_dict())  # must not raise TypeError


def test_float16_roundtrip_json():
    cfg = PretrainedConfig()
    set_dtype_in_config(cfg, torch.float16)
    assert dtype_from_config(cfg) == "float16"
    json.dumps(cfg.to_dict())


def test_float32_roundtrip_json():
    cfg = PretrainedConfig()
    set_dtype_in_config(cfg, torch.float32)
    assert dtype_from_config(cfg) == "float32"
    json.dumps(cfg.to_dict())


def test_string_input_passthrough():
    cfg = PretrainedConfig()
    set_dtype_in_config(cfg, "bfloat16")
    assert dtype_from_config(cfg) == "bfloat16"


def test_bare_object_writes_correct_field():
    class _Bare:
        pass
    obj = _Bare()
    set_dtype_in_config(obj, torch.float16)
    expected = "torch_dtype" if HAS_TORCH_DTYPE else "dtype"
    other = "dtype" if HAS_TORCH_DTYPE else "torch_dtype"
    assert getattr(obj, expected) == "float16"
    assert getattr(obj, other, None) is None


def test_no_deprecation_warning():
    import warnings
    cfg = PretrainedConfig()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        set_dtype_in_config(cfg, torch.bfloat16)
    bad = [w for w in caught if "torch_dtype" in str(w.message) and "deprecated" in str(w.message).lower()]
    assert not bad, f"unexpected deprecation: {[str(w.message) for w in bad]}"
