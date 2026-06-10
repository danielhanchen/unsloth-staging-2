# Real Apple Silicon probes for the unsloth-zoo MLX save / GGUF export
# surface merged in unslothai/unsloth-zoo#697. Runs against REAL mlx /
# mlx-lm / mlx-vlm wheels (no simulation shim). Asserts hard; any failure
# exits non-zero.

import inspect
import json
import platform
import sys
import tempfile
import types
from pathlib import Path

assert platform.system() == "Darwin" and platform.machine() == "arm64", (
    platform.system(), platform.machine(),
)

import mlx.core as mx

assert "mlx_simulation" not in (getattr(mx, "__file__", "") or ""), mx.__file__
print(f"OK: real mlx {getattr(mx, '__version__', 'unknown')} at {mx.__file__}")

from unsloth_zoo.mlx.runtime import is_mlx_available

assert is_mlx_available() is True
print("OK: is_mlx_available() on real hardware")

import unsloth_zoo.mlx.utils as mutils
import unsloth_zoo.mlx.loader as loader


def probe_arrays_match():
    for shape in [(4,), (2, 3), (2, 3, 4), (2, 3, 4, 5)]:
        size = 1
        for dim in shape:
            size *= dim
        base = mx.arange(size, dtype=mx.float32).reshape(shape)
        clone = mx.array(base)
        assert mutils._mlx_arrays_match(base, clone) is True, shape
        perturbed = clone + 1
        assert mutils._mlx_arrays_match(base, perturbed) is False, shape
    assert mutils._mlx_arrays_match(1.5, 1.5) is True
    assert mutils._mlx_arrays_match(torch_like := mx.zeros((2, 3)), torch_like) is True
    print("OK: _mlx_arrays_match value-checks real mx arrays across ranks")


def probe_copy_weights_isinstance():
    # The isinstance(value, mx.array) check flagged in review must hold on
    # real wheels, and copies must be detached.
    weights = {"a": mx.ones((2, 2)), "b": "not an array"}
    copies = mutils._copy_mlx_vlm_sanitize_weights(weights)
    assert isinstance(copies["a"], mx.array)
    assert copies["b"] == "not an array"
    copies["a"] = copies["a"] + 5
    assert float(weights["a"][0, 0]) == 1.0
    print("OK: _copy_mlx_vlm_sanitize_weights isinstance + isolation on real mlx")


class _ConvAndRenameSanitizer:
    @staticmethod
    def sanitize(weights):
        out = {}
        for name, tensor in weights.items():
            new_name = name.replace("visual.", "vision_tower.")
            if hasattr(tensor, "shape") and len(tensor.shape) == 4:
                tensor = mx.transpose(tensor, (0, 2, 3, 1))
            out[new_name] = tensor
        return out


def probe_rewrite_roundtrip():
    mlx_tensor = mx.arange(2 * 3 * 4 * 5, dtype=mx.float32).reshape(2, 3, 4, 5)
    new_name, new_tensor, changed = mutils._rewrite_mlx_vlm_tensor_for_gguf(
        "vision_tower.patch_embed.weight",
        mlx_tensor,
        [(_ConvAndRenameSanitizer, None)],
    )
    assert changed is True
    assert new_name == "visual.patch_embed.weight"
    assert tuple(new_tensor.shape) == (2, 5, 3, 4)
    roundtrip = mx.transpose(new_tensor, (0, 2, 3, 1))
    assert bool(mx.all(roundtrip == mlx_tensor).item())
    print("OK: _rewrite_mlx_vlm_tensor_for_gguf 4D roundtrip on real mx ops")


def probe_sanitize_proxy():
    seen = {}

    class InstanceStyle:
        def sanitize(self, weights):
            seen["config"] = self.config
            return {k: v * self.config["scale"] for k, v in weights.items()}

    out = mutils._call_mlx_vlm_sanitize(
        InstanceStyle, {"scale": 2}, {"w": mx.ones((2,))}
    )
    assert seen["config"] == {"scale": 2}
    assert float(out["w"][0]) == 2.0
    print("OK: _MlxVlmSanitizeProxy two-arg sanitize on real arrays")


def probe_submodule_only_pipelines():
    class Tower:
        def sanitize(self, weights):
            return weights

    tower = Tower()
    wrapper = types.SimpleNamespace(vision_tower=tower)
    pipelines = mutils._get_mlx_vlm_model_sanitize_pipelines(wrapper)
    assert pipelines == [[(tower, None)]]
    print("OK: submodule-only sanitizer pipelines")


def probe_prepare_export_directory():
    original = mutils._build_mlx_vlm_sanitize_pipelines
    mutils._build_mlx_vlm_sanitize_pipelines = (
        lambda config, model=None: [[(_ConvAndRenameSanitizer, None)]]
    )
    try:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "staging"
            out.mkdir()
            (out / "config.json").write_text(
                json.dumps({"model_type": "probe_vlm"}), encoding="utf-8"
            )
            conv = mx.arange(2 * 3 * 4 * 5, dtype=mx.float32).reshape(2, 3, 4, 5)
            mx.save_safetensors(
                str(out / "model-00001-of-00001.safetensors"),
                {"vision_tower.patch_embed.weight": conv},
                metadata={"format": "mlx"},
            )
            (out / "model.safetensors.index.json").write_text(
                json.dumps({
                    "metadata": {},
                    "weight_map": {
                        "vision_tower.patch_embed.weight":
                            "model-00001-of-00001.safetensors",
                    },
                }),
                encoding="utf-8",
            )
            rewritten = mutils._prepare_vlm_gguf_export_directory(out)
            assert rewritten == 1, rewritten
            tensors = mx.load(str(out / "model-00001-of-00001.safetensors"))
            assert list(tensors) == ["visual.patch_embed.weight"]
            assert tuple(tensors["visual.patch_embed.weight"].shape) == (2, 5, 3, 4)
            index = json.loads(
                (out / "model.safetensors.index.json").read_text(encoding="utf-8")
            )
            assert list(index["weight_map"]) == ["visual.patch_embed.weight"]
    finally:
        mutils._build_mlx_vlm_sanitize_pipelines = original
    print("OK: _prepare_vlm_gguf_export_directory end-to-end with real mx I/O")


def probe_save_config_real_backends():
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "config.json"
        mutils._save_mlx_config(
            {"model_type": "llama", "quantization": {"bits": 4}},
            target,
            is_vlm=False,
        )
        saved = json.loads(target.read_text(encoding="utf-8"))
        assert saved["model_type"] == "llama"
        assert saved["quantization"] == {"bits": 4}
        # Real mlx_lm.save_config mirrors quantization into
        # quantization_config itself; if present it must agree.
        if "quantization_config" in saved:
            assert saved["quantization_config"] == {"bits": 4}
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "config.json"
        mutils._save_mlx_config(
            {
                "model_type": "probe_vlm",
                "vision_config": {"hidden_size": 8},
                "quantization": {"bits": 4},
            },
            target,
            is_vlm=True,
        )
        saved = json.loads(target.read_text(encoding="utf-8"))
        assert saved["quantization_config"] == {"bits": 4}
    print("OK: _save_mlx_config via real mlx_lm / mlx_vlm save_config")


def probe_push_signature():
    params = list(inspect.signature(mutils.push_to_hub_gguf).parameters)
    assert params == [
        "model", "tokenizer", "save_directory", "repo_id",
        "quantization_method", "token", "private", "first_conversion",
    ], params
    print("OK: push_to_hub_gguf keeps pre-existing positional order")


def probe_read_json_file():
    with tempfile.TemporaryDirectory() as tmp:
        sidecar = Path(tmp) / "processor_config.json"
        sidecar.write_text("[1, 2]", encoding="utf-8")
        assert loader._read_json_file(sidecar) == {}
    print("OK: _read_json_file coerces non-object JSON")


probe_arrays_match()
probe_copy_weights_isinstance()
probe_rewrite_roundtrip()
probe_sanitize_proxy()
probe_submodule_only_pipelines()
probe_prepare_export_directory()
probe_save_config_real_backends()
probe_push_signature()
probe_read_json_file()
print("ALL PROBES PASSED")
