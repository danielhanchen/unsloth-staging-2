"""Adversarial verification of Component A.

Proves capability detection never executes a model repo's auto_map Python,
and that detection is functionally correct (lossless) across model classes.
"""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.models.model_config import (  # noqa: E402
    is_vision_model,
    _vision_detection_cache,
    _VLM_MODEL_TYPES,
    _AUDIO_ONLY_MODEL_TYPES,
)


def _write_model_dir(tmp_path, cfg: dict, with_evil_module: bool = False) -> str:
    d = tmp_path
    (d / "config.json").write_text(json.dumps(cfg))
    if with_evil_module:
        # Imports of this module write a sentinel; if detection ever execs the
        # repo's auto_map code, the sentinel file appears.
        sentinel = d / "PWNED_SENTINEL"
        (d / "modeling_evil.py").write_text(
            "import os\n"
            f"open({str(sentinel)!r}, 'w').write('pwned')\n"
            "class EvilConfig: pass\n"
            "class EvilModel: pass\n"
        )
    return str(d)


@pytest.fixture(autouse=True)
def _clear_cache():
    _vision_detection_cache.clear()
    yield
    _vision_detection_cache.clear()


def test_no_code_execution_on_detection(tmp_path):
    """A malicious local model with auto_map -> modeling_evil must NOT execute."""
    cfg = {
        "model_type": "deepseek_vl_v2",
        "architectures": ["DeepseekOCRForCausalLM"],
        "auto_map": {
            "AutoConfig": "modeling_evil.EvilConfig",
            "AutoModel": "modeling_evil.EvilModel",
        },
        "vision_config": {"image_size": 1024},
        "max_position_embeddings": 4096,
    }
    path = _write_model_dir(tmp_path, cfg, with_evil_module=True)
    sentinel = tmp_path / "PWNED_SENTINEL"

    # Detection path
    result = is_vision_model(path)

    # The two metadata probes
    from utils.hardware.hardware import _load_config_for_gpu_estimate
    from utils.transformers_version import _load_config_json

    ns = _load_config_for_gpu_estimate(path)
    raw = _load_config_json(path)

    assert not sentinel.exists(), "SECURITY FAILURE: auto_map code was executed during detection!"
    assert result is True, "DeepSeek-OCR-style model must still be detected as vision via raw vision_config"
    assert ns is not None and getattr(ns, "max_position_embeddings", None) == 4096
    assert raw is not None and raw.get("model_type") == "deepseek_vl_v2"


@pytest.mark.parametrize(
    "cfg, expected",
    [
        # repo-code VLMs (auto_map) detected via declarative vision_config
        ({"model_type": "deepseek_vl_v2", "architectures": ["DeepseekOCRForCausalLM"],
          "auto_map": {"AutoConfig": "x.Y"}, "vision_config": {}}, True),
        ({"model_type": "kimi_k25", "architectures": ["KimiK25ForConditionalGeneration"],
          "auto_map": {"AutoConfig": "x.Y"}, "vision_config": {}}, True),
        # newer-native vision (v5) via vision_config
        ({"model_type": "gemma4_unified", "architectures": ["Gemma4UnifiedForConditionalGeneration"],
          "vision_config": {}, "image_token_id": 7}, True),
        # text MoE -> NOT vision (the model that motivated the subprocess)
        ({"model_type": "glm4_moe_lite", "architectures": ["Glm4MoeLiteForCausalLM"]}, False),
        # suffix-bug regression: T5/Bart text seq2seq must NOT be vision
        ({"model_type": "t5", "architectures": ["T5ForConditionalGeneration"]}, False),
        ({"model_type": "bart", "architectures": ["BartForConditionalGeneration"]}, False),
        # audio-only -> NOT vision (share the ForConditionalGeneration suffix)
        ({"model_type": "whisper", "architectures": ["WhisperForConditionalGeneration"]}, False),
        ({"model_type": "csm", "architectures": ["CsmForConditionalGeneration"]}, False),
        # registry-native VLMs via model_type
        ({"model_type": "qwen2_vl", "architectures": ["Qwen2VLForConditionalGeneration"]}, True),
        ({"model_type": "llava", "architectures": ["LlavaForConditionalGeneration"]}, True),
    ],
)
def test_functional_detection(tmp_path, cfg, expected):
    path = _write_model_dir(tmp_path, cfg)
    assert is_vision_model(path) is expected, f"{cfg['model_type']} expected vision={expected}"


def test_registry_derivation():
    # Registry-derived sets are large and include the curated repo-code VLMs.
    assert len(_VLM_MODEL_TYPES) >= 50, f"_VLM_MODEL_TYPES too small: {len(_VLM_MODEL_TYPES)}"
    assert len(_AUDIO_ONLY_MODEL_TYPES) >= 20, f"_AUDIO_ONLY_MODEL_TYPES too small: {len(_AUDIO_ONLY_MODEL_TYPES)}"
    for repo_vlm in ("deepseek_vl_v2", "kimi_k25", "phi3_v", "cogvlm2", "minicpmv"):
        assert repo_vlm in _VLM_MODEL_TYPES, f"curated repo-code VLM {repo_vlm} missing"
    for native in ("llava", "qwen2_vl"):
        assert native in _VLM_MODEL_TYPES, f"registry-native VLM {native} missing"
    for audio in ("whisper", "csm"):
        assert audio in _AUDIO_ONLY_MODEL_TYPES, f"audio type {audio} missing"
