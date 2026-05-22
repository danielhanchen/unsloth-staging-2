"""Sim test: set_additional_modules preserves legacy paths and handles new wrappers."""
import sys
# sys.path: unsloth_zoo is pip-installed in CI

import types
import torch
from unsloth_zoo.empty_model import set_additional_modules


def _llama_like_model():
    class _LM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(4, 2)
            self.norm = torch.nn.LayerNorm(2)

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _LM()
            self.lm_head = torch.nn.Linear(2, 4, bias=False)
    return _Model()


def _llama_qsd():
    return {
        "model.embed_tokens.weight": torch.zeros(4, 2),
        "model.norm.weight": torch.ones(2),
        "lm_head.weight": torch.randn(4, 2),
    }


def test_llama_like_model_works():
    model = _llama_like_model()
    cfg = types.SimpleNamespace(pad_token_id=0, tie_word_embeddings=False)
    set_additional_modules(model, _llama_qsd(), cfg)
    # Embedding got replaced
    assert torch.allclose(model.model.embed_tokens.weight, torch.zeros(4, 2))
    # lm_head reassigned with correct shape
    assert model.lm_head.weight.shape == (4, 2)


def test_vlm_like_model_with_language_model_attribute():
    """Tests the new branch: hasattr(new_model.model, "language_model")."""
    class _LM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(4, 2)
            self.norm = torch.nn.LayerNorm(2)

    class _Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = _LM()

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _Inner()
            self.lm_head = torch.nn.Linear(2, 4, bias=False)

    model = _Model()
    qsd = {
        "model.language_model.embed_tokens.weight": torch.zeros(4, 2),
        "model.language_model.norm.weight": torch.ones(2),
        "lm_head.weight": torch.randn(4, 2),
    }
    cfg = types.SimpleNamespace(pad_token_id=0, tie_word_embeddings=False)
    set_additional_modules(model, qsd, cfg)
    assert torch.allclose(model.model.language_model.embed_tokens.weight, torch.zeros(4, 2))


def test_tied_embeddings_calls_tie_weights():
    """tie_word_embeddings=True should call model.tie_weights() (a method
    transformers' PreTrainedModel exposes)."""
    model = _llama_like_model()
    cfg = types.SimpleNamespace(pad_token_id=0, tie_word_embeddings=True)

    called = {"flag": False}
    def _tied():
        called["flag"] = True
    # Attach a method by setting on the object (sidestepping nn.Module __setattr__ quirks)
    object.__setattr__(model, "tie_weights", _tied)

    set_additional_modules(model, _llama_qsd(), cfg)
    assert called["flag"], "tie_weights was not invoked despite tie_word_embeddings=True"


def test_pad_token_id_zero_valid():
    """Pre-fix concern: pad_token_id=0 was treated as falsy; the post-PR check uses
    explicit `is None`."""
    model = _llama_like_model()
    cfg = types.SimpleNamespace(pad_token_id=0, tie_word_embeddings=False)
    set_additional_modules(model, _llama_qsd(), cfg)
    assert model.model.embed_tokens.padding_idx == 0


def test_pad_token_id_none_no_assert():
    """If pad_token_id is None, no assertion should fire."""
    model = _llama_like_model()
    cfg = types.SimpleNamespace(pad_token_id=None, tie_word_embeddings=False)
    set_additional_modules(model, _llama_qsd(), cfg)
    assert model.model.embed_tokens.padding_idx is None


def test_unwrap_modelweightparameter():
    """Simulate vLLM's ModelWeightParameter (a torch.Tensor subclass exposing .data
    that returns a plain tensor). set_additional_modules should _unwrap_tensor."""
    class _ModelWeightParameter(torch.Tensor):
        pass

    raw = torch.zeros(4, 2).as_subclass(_ModelWeightParameter)

    model = _llama_like_model()
    cfg = types.SimpleNamespace(pad_token_id=0, tie_word_embeddings=False)
    qsd = {
        "model.embed_tokens.weight": raw,
        "model.norm.weight": torch.ones(2),
        "lm_head.weight": torch.randn(4, 2),
    }
    set_additional_modules(model, qsd, cfg)
    # The tensor should be assigned successfully without dtype mismatch
    assert model.model.embed_tokens.weight.shape == (4, 2)
