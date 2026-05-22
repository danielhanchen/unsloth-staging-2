"""Sim test: copy_attributes excludes 'layer_scalar' to avoid Gemma4-specific clobber."""
import sys
# sys.path: unsloth_zoo is pip-installed in CI

import torch


def test_copy_attributes_skips_layer_scalar_in_source():
    """A buffer named 'layer_scalar' on the original should NOT be copied to new_model."""
    import inspect
    from unsloth_zoo import empty_model
    src = inspect.getsource(empty_model.copy_attributes)
    # The fix is `attr in buffer_names and attr != "layer_scalar"`
    assert 'attr != "layer_scalar"' in src or "attr != 'layer_scalar'" in src


def test_other_buffers_still_copied():
    """Sanity: copy_attributes should still copy non-layer_scalar buffers."""
    import inspect
    from unsloth_zoo import empty_model
    src = inspect.getsource(empty_model.copy_attributes)
    # The body should still iterate buffer_names
    assert "buffer_names" in src
    assert "setattr(module, attr, original_val.to(new_model.device))" in src
