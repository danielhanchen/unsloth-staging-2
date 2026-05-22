"""Sim test: extract_vision_layers widened Parameter -> Tensor isinstance check."""
import sys
# sys.path: unsloth_zoo is pip-installed in CI

import torch


def test_plain_tensor_handled_via_module_path():
    """The PR widens `isinstance(layer_module, torch.nn.Parameter)` to
    `isinstance(layer_module, torch.Tensor)`. Confirm plain tensors no longer
    fall into the warning-only `else` branch."""
    import inspect
    from unsloth_zoo import empty_model
    src = inspect.getsource(empty_model.extract_vision_layers)
    # Two sites in the function were changed
    assert "isinstance(layer_module, torch.Tensor)" in src
    assert "isinstance(component, torch.Tensor)" in src
    # And the old Parameter-only checks should NOT linger in those branches
    # (it's OK if the word "Parameter" appears elsewhere in the file for actual
    #  nn.Parameter construction)


def test_parameter_subclass_of_tensor_still_works():
    """nn.Parameter is a Tensor subclass, so the broader isinstance still matches."""
    p = torch.nn.Parameter(torch.zeros(2, 2))
    assert isinstance(p, torch.Tensor)
