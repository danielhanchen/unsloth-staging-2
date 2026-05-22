"""Sim test: finalize_huggingface_model should be a safe refactor for non-Gemma4 models."""
import sys, types
# sys.path: unsloth_zoo is pip-installed in CI

import torch
from unsloth_zoo.empty_model import finalize_huggingface_model


class _RotaryCfg:
    pass


class _FakeRotary(torch.nn.Module):
    def __init__(self, config=None, device=None):
        super().__init__()
        self.config = config if config is not None else _RotaryCfg()
        self.register_buffer("inv_freq", torch.arange(8, dtype=torch.float32))
        self.register_buffer("original_inv_freq", torch.arange(8, dtype=torch.float32))


class _Attn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_idx = -1
        self.rotary_emb = _FakeRotary()


class _MLP(torch.nn.Module):
    pass


class _Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_idx = -1
        self.self_attn = _Attn()
        self.mlp = _MLP()


class _LlamaModel(torch.nn.Module):
    def __init__(self, n=4):
        super().__init__()
        class _Inner(torch.nn.Module):
            def __init__(self, n):
                super().__init__()
                self.layers = torch.nn.ModuleList([_Layer() for _ in range(n)])
        self.model = _Inner(n)


def _cfg(model_type="llama"):
    cfg = types.SimpleNamespace(model_type=model_type)
    cfg.text_config = cfg
    return cfg


def test_finalize_llama_sets_layer_idx_on_all_layers_and_submodules():
    model = _LlamaModel(n=4)
    finalize_huggingface_model(
        model, None, _cfg("llama"), torch.float16,
        quantization_config=None, bnb_config=None,
    )
    for i, layer in enumerate(model.model.layers):
        assert layer.layer_idx == i, f"layer {i}.layer_idx = {layer.layer_idx}"
        assert layer.self_attn.layer_idx == i, f"layer {i}.self_attn.layer_idx = {layer.self_attn.layer_idx}"


def test_finalize_llama_rotary_buffers_remain_float32():
    model = _LlamaModel()
    finalize_huggingface_model(
        model, None, _cfg("llama"), torch.bfloat16,
        quantization_config=None, bnb_config=None,
    )
    for layer in model.model.layers:
        assert layer.self_attn.rotary_emb.inv_freq.dtype == torch.float32
        assert layer.self_attn.rotary_emb.original_inv_freq.dtype == torch.float32


def test_finalize_does_not_modify_unrelated_submodule_config():
    class _SubModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = types.SimpleNamespace(dtype="float32")  # arbitrary unrelated config

    class _Wrap(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.sub = _SubModule()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList()

    w = _Wrap()
    finalize_huggingface_model(w, None, _cfg("llama"), torch.bfloat16, quantization_config={}, bnb_config=None)
    # Unrelated submodule config must NOT be overwritten.
    assert w.sub.config.dtype == "float32"


def test_finalize_text_only_model_with_rotary_pos_emb_no_crash():
    # Pre-PR there was a hard assert; should now be a no-op or graceful skip.
    class _Rotary(torch.nn.Module):
        pass

    class _Layer2(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_idx = -1
            self.rotary_pos_emb = _Rotary()

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([_Layer2()])

    # vision_config is None (text-only)
    finalize_huggingface_model(_Model(), None, _cfg("llama"), torch.float16,
                                quantization_config={"x": 1}, bnb_config=None)


def test_finalize_handles_empty_layers():
    # ModuleList() is empty; should not crash.
    class _Empty(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList()
    finalize_huggingface_model(_Empty(), None, _cfg("llama"), torch.float16,
                                quantization_config=None, bnb_config=None)
