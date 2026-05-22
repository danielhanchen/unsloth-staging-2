"""Sim test: verify or refute the chatgpt-codex P1 about prequant 4-bit Qwen3-VL
merger.linear_fc1/fc2 being reconstructed as plain nn.Linear instead of Linear4bit.

Pre-PR: model.visual.merger.linear_fc{kk} was in standard_layers; kk=1,2 happened to
        land on correct names, so the convert_vllm_to_huggingface main loop saw their
        weight.quant_state and built Linear4bit.
Post-PR: They moved to non_layered_components and are processed by set_additional_modules
         which does `torch.nn.Parameter(val, requires_grad=False); exec(f"new_model.{key} = val")`
         WITHOUT consulting `*.weight.quant_state`. For prequant 4-bit, this silently
         assigns uint8 packed weights to a plain nn.Linear module.
"""
import sys, inspect, types
# sys.path: unsloth_zoo is pip-installed in CI

import torch
from unsloth_zoo import empty_model


def test_set_additional_modules_does_not_consult_weight_quant_state():
    """Source-level: set_additional_modules has no branch reading `.weight.quant_state`."""
    src = inspect.getsource(empty_model.set_additional_modules)
    # No `Linear4bit` or `quant_state` branch -- it's a generic Parameter assignment
    assert "Linear4bit" not in src, "set_additional_modules SHOULD NOT (and doesn't) construct Linear4bit"
    assert ".weight.quant_state" not in src, "set_additional_modules SHOULD NOT (and doesn't) reference quant_state"


def test_set_additional_modules_assigns_parameter_to_existing_linear():
    """Behavioral: set_additional_modules assigns weight directly to the existing module.
    If the module is nn.Linear (from empty_model) and we feed a uint8 packed weight,
    the assignment succeeds but the resulting module is still nn.Linear, not Linear4bit."""

    class _Merger(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # empty_model puts a tiny placeholder Linear here pre-conversion
            self.linear_fc1 = torch.nn.Linear(1, 1, bias=False)
            self.linear_fc2 = torch.nn.Linear(1, 1, bias=False)

    class _Visual(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.merger = _Merger()

    class _LM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(2, 1)
            self.norm = torch.nn.LayerNorm(1)

    class _Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = _LM()
            self.visual = _Visual()

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _Inner()
            self.lm_head = torch.nn.Linear(1, 2, bias=False)

    model = _Model()
    # Simulate a prequant-4bit-shaped weight: uint8 packed (the actual encoding bnb uses)
    fc1_packed = torch.zeros(4, 1, dtype=torch.uint8)  # 4-bit packed = uint8 with half-sized first dim
    fc2_packed = torch.zeros(4, 1, dtype=torch.uint8)

    # In a real prequant load, the corresponding quant_state objects are stored separately
    class _FakeQuantState:
        def __init__(self): self.shape = (1, 1)

    qsd = {
        "model.language_model.embed_tokens.weight": torch.zeros(2, 1),
        "model.language_model.norm.weight": torch.ones(1),
        "lm_head.weight": torch.zeros(2, 1),
        "model.visual.merger.linear_fc1.weight": fc1_packed,
        "model.visual.merger.linear_fc2.weight": fc2_packed,
        # These quant_state sidecars would have built Linear4bit pre-PR
        "model.visual.merger.linear_fc1.weight.quant_state": _FakeQuantState(),
        "model.visual.merger.linear_fc2.weight.quant_state": _FakeQuantState(),
    }
    cfg = types.SimpleNamespace(
        pad_token_id=0,
        text_config=types.SimpleNamespace(tie_word_embeddings=False),
    )
    empty_model.set_additional_modules(model, qsd, cfg)

    # The module type is STILL nn.Linear (not Linear4bit), and the assigned weight is uint8.
    # This is the bug: a Linear module with uint8 weights is non-functional.
    assert isinstance(model.model.visual.merger.linear_fc1, torch.nn.Linear)
    assert model.model.visual.merger.linear_fc1.weight.dtype == torch.uint8, (
        f"Expected uint8 packed weight got dtype={model.model.visual.merger.linear_fc1.weight.dtype}"
    )
    # That would crash on a real forward; this confirms the chatgpt-codex P1 in code.
    # We don't actually call forward (would crash), the type/dtype state proves the bug.
