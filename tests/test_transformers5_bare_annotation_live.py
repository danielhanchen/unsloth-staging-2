"""The transformers-5 config fix, demonstrated against a real transformers 5.

transformers 5.x turns `PretrainedConfig` subclasses into dataclasses. vLLM's
`transformers_utils/configs/deepseek_vl2.py` declares

    vision_config: VisionEncoderConfig

with no default, and a dataclass will not accept a non-default field after an
inherited default one:

    TypeError: non-default argument 'vision_config' follows default argument
               'problem_type'

That fires while *importing* `vllm.transformers_utils.configs`, which takes
down `import vllm` and, because unsloth imports vLLM, `import unsloth` too.
It was seen in the wild as `unsloth: "ABSENT: TypeError"`.

The other tests for this fix assert on source text. This one reproduces the
exact failing shape and checks the outcome, so it would catch the fix
silently ceasing to work -- which is the failure mode that matters, and which
source-level assertions cannot see.

No vLLM install is needed: the config class above IS the reproduction, and
building it by hand avoids pulling several GB to test four lines of
behaviour. It skips on transformers 4.x, where subclasses are not converted
to dataclasses and there is nothing to fix.
"""

import pytest

transformers = pytest.importorskip("transformers")

from packaging.version import Version  # noqa: E402

pytestmark = pytest.mark.skipif(
    Version(transformers.__version__) < Version("5.0.0"),
    reason="transformers 4.x does not convert config subclasses to dataclasses",
)


def _build(tag):
    """A vLLM-shaped config pair: a bare annotation with no default."""
    from transformers.configuration_utils import PretrainedConfig

    class VisionEncoderConfig(PretrainedConfig):
        model_type = f"vision_{tag}"

    class DeepseekVL2Config(PretrainedConfig):
        model_type = f"deepseek_vl_v2_{tag}"
        vision_config: VisionEncoderConfig       # no default: the trigger

    return DeepseekVL2Config


@pytest.fixture
def unpatched():
    """Remove the patch so the failure can be observed, then restore it."""
    from transformers.configuration_utils import PretrainedConfig
    saved = PretrainedConfig.__dict__.get("__init_subclass__")
    flag = getattr(PretrainedConfig, "_unsloth_patched_init_subclass", False)
    inner = getattr(saved, "__func__", saved)
    original = getattr(inner, "__wrapped__", None)
    if flag and original is not None:
        PretrainedConfig.__init_subclass__ = classmethod(original)
        PretrainedConfig._unsloth_patched_init_subclass = False
    yield
    if saved is not None:
        PretrainedConfig.__init_subclass__ = saved
    PretrainedConfig._unsloth_patched_init_subclass = flag


def test_the_failure_is_real_without_the_fix(unpatched):
    """Guards the premise. If this ever stops raising, transformers has
    changed and the fix below is no longer testing anything."""
    from unsloth.import_fixes import fix_transformers5_bare_annotation_configs
    from transformers.configuration_utils import PretrainedConfig

    if getattr(PretrainedConfig, "_unsloth_patched_init_subclass", False):
        pytest.skip("could not unpatch; the wrapped original was not reachable")
    with pytest.raises(TypeError, match="non-default argument"):
        _build("unpatched")


def test_the_fix_lets_it_import():
    from unsloth.import_fixes import fix_transformers5_bare_annotation_configs
    fix_transformers5_bare_annotation_configs()
    cls = _build("patched")
    assert cls.__name__ == "DeepseekVL2Config"


def test_applying_twice_is_a_no_op():
    from unsloth.import_fixes import fix_transformers5_bare_annotation_configs
    from transformers.configuration_utils import PretrainedConfig
    fix_transformers5_bare_annotation_configs()
    first = PretrainedConfig.__dict__.get("__init_subclass__")
    fix_transformers5_bare_annotation_configs()
    assert PretrainedConfig.__dict__.get("__init_subclass__") is first


def test_ordinary_configs_are_unaffected():
    """The patch runs for EVERY config subclass, so it must not disturb the
    overwhelming majority that were always fine."""
    from unsloth.import_fixes import fix_transformers5_bare_annotation_configs
    from transformers.configuration_utils import PretrainedConfig
    fix_transformers5_bare_annotation_configs()

    class Ordinary(PretrainedConfig):
        model_type = "ordinary_probe"

        def __init__(self, hidden_size = 16, **kwargs):
            self.hidden_size = hidden_size
            super().__init__(**kwargs)

    cfg = Ordinary(hidden_size = 32)
    assert cfg.hidden_size == 32
    assert cfg.model_type == "ordinary_probe"


def test_a_real_model_config_still_loads():
    from unsloth.import_fixes import fix_transformers5_bare_annotation_configs
    fix_transformers5_bare_annotation_configs()
    from transformers import LlamaConfig
    cfg = LlamaConfig(hidden_size = 64, num_hidden_layers = 2)
    assert cfg.hidden_size == 64


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
