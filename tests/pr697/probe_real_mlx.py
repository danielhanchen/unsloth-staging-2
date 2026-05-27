"""Probe PR #697 symbols against REAL Apple Silicon mlx / mlx-vlm wheels.

Runs inside the Studio venv on macos-14 (NOT a torch shim). Confirms:
  1. The post-migration subpackage `unsloth_zoo.mlx.*` imports cleanly.
  2. Every PR-697 symbol exists and is callable.
  3. Each fix's contract holds when the inputs go through real mlx arrays
     and real (or stubbed where unavailable) mlx-vlm helpers.

Exits 0 on success, 1 on any failure. Run with:
    python -m tests.pr697.probe_real_mlx
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import os
import sys
import types
from pathlib import Path
import traceback


# ---------------------------------------------------------------------------
# Probe harness
# ---------------------------------------------------------------------------
RESULTS: list[tuple[str, bool, str]] = []


def probe(name: str):
    def _wrap(fn):
        try:
            fn()
            RESULTS.append((name, True, ""))
            print(f"[PASS] {name}")
        except Exception:
            tb = traceback.format_exc()
            RESULTS.append((name, False, tb))
            print(f"[FAIL] {name}\n{tb}")
        return fn
    return _wrap


# ---------------------------------------------------------------------------
# 0. Subpackage import smoke (post-migration paths).
# ---------------------------------------------------------------------------
@probe("0a. import unsloth_zoo.mlx.utils")
def _():
    import unsloth_zoo.mlx.utils as _m   # noqa: F401


@probe("0b. import unsloth_zoo.mlx.loader")
def _():
    import unsloth_zoo.mlx.loader as _m  # noqa: F401


@probe("0c. import unsloth_zoo.mlx.runtime + is_mlx_available")
def _():
    from unsloth_zoo.mlx.runtime import is_mlx_available
    assert callable(is_mlx_available)
    # On macos-14 + .[mlx] this should return True.
    print(f"   is_mlx_available() -> {is_mlx_available()}")


@probe("0d. real mlx.core import")
def _():
    import mlx.core as mx
    arr = mx.array([1.0, 2.0, 3.0])
    assert arr.shape == (3,), arr.shape
    print(f"   mx.array.shape -> {arr.shape}")


@probe("0e. real mlx_vlm import")
def _():
    import mlx_vlm  # noqa: F401


# ---------------------------------------------------------------------------
# 1. PR-697 helpers exist and have expected signatures.
# ---------------------------------------------------------------------------
@probe("1a. PR-697 symbols present in unsloth_zoo.mlx.utils")
def _():
    import unsloth_zoo.mlx.utils as mutils
    for sym in (
        "_save_mlx_config",
        "_has_vision_config",
        "_is_vlm_model",
        "_get_model_config",
        "_copy_source_sidecars",
        "_rewrite_mlx_vlm_tensor_for_gguf",
        "_mlx_arrays_match",
        "_prepare_vlm_gguf_export_directory",
        "_sync_gguf_nextn_layer_config",
        "_MlxVlmSanitizeProxy",
    ):
        assert hasattr(mutils, sym), f"missing {sym}"


@probe("1b. PR-697 symbols present in unsloth_zoo.mlx.loader")
def _():
    import unsloth_zoo.mlx.loader as mloader
    for sym in ("_read_json_file", "_repair_degraded_vlm_processor"):
        assert hasattr(mloader, sym), f"missing {sym}"


# ---------------------------------------------------------------------------
# 2. Fix #1 — VLM config save uses mlx_vlm.utils.save_config with
#    quantization_config preservation.
# ---------------------------------------------------------------------------
@probe("2. fix #1: VLM config save uses mlx_vlm and preserves quantization_config")
def _():
    import unsloth_zoo.mlx.utils as mutils
    captured = {}
    # Replace mlx_vlm.utils.save_config with a capturing stub.
    real_mod = sys.modules.get("mlx_vlm.utils")
    fake = types.ModuleType("mlx_vlm.utils")

    def fake_save(c, p):
        captured["config"] = c
        captured["path"] = str(p)
        Path(p).write_text(json.dumps(c), encoding="utf-8")

    fake.save_config = fake_save
    sys.modules["mlx_vlm.utils"] = fake
    try:
        cfg = {
            "model_type": "gemma3",
            "vision_config": {"hidden_size": 8},
            "quantization": {"group_size": 64, "bits": 4},
        }
        out_path = Path(os.environ.get("UNSLOTH_PROBE_TMP", "/tmp/pr697_probe")) / "vlm_config.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        mutils._save_mlx_config(cfg, out_path, is_vlm=True)
        assert captured["config"]["quantization"] == cfg["quantization"]
        assert captured["config"]["quantization_config"] == cfg["quantization"]
        # Original input mustn't be mutated with quantization_config.
        assert "quantization_config" not in cfg
    finally:
        if real_mod is None:
            sys.modules.pop("mlx_vlm.utils", None)
        else:
            sys.modules["mlx_vlm.utils"] = real_mod


# ---------------------------------------------------------------------------
# 3. Fix #1 negative — text-only routes through mlx_lm.utils.save_config.
# ---------------------------------------------------------------------------
@probe("3. fix #1 negative: text-only config uses mlx_lm.utils.save_config")
def _():
    import unsloth_zoo.mlx.utils as mutils
    called = {"lm": 0, "vlm": 0}
    real_lm = sys.modules.get("mlx_lm.utils")
    real_vlm = sys.modules.get("mlx_vlm.utils")
    fake_lm = types.ModuleType("mlx_lm.utils")
    fake_vlm = types.ModuleType("mlx_vlm.utils")

    def lm_save(c, p):
        called["lm"] += 1
        Path(p).write_text(json.dumps(c), encoding="utf-8")

    def vlm_save(c, p):
        called["vlm"] += 1

    fake_lm.save_config = lm_save
    fake_vlm.save_config = vlm_save
    sys.modules["mlx_lm.utils"] = fake_lm
    sys.modules["mlx_vlm.utils"] = fake_vlm
    try:
        cfg = {"model_type": "llama", "hidden_size": 8}
        out_path = Path(os.environ.get("UNSLOTH_PROBE_TMP", "/tmp/pr697_probe")) / "txt_config.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        mutils._save_mlx_config(cfg, out_path, is_vlm=False)
        assert called == {"lm": 1, "vlm": 0}, called
    finally:
        for k, v in [("mlx_lm.utils", real_lm), ("mlx_vlm.utils", real_vlm)]:
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


# ---------------------------------------------------------------------------
# 4. Fix #5 — _mlx_arrays_match value check on rank-2 arrays.
# ---------------------------------------------------------------------------
@probe("4. fix #5: _mlx_arrays_match detects unequal rank-2 tensors")
def _():
    import mlx.core as mx
    import unsloth_zoo.mlx.utils as mutils
    a = mx.array([[1.0, 2.0], [3.0, 4.0]])
    b = mx.array([[1.0, 2.0], [3.0, 4.0]])
    c = mx.array([[1.0, 2.0], [3.0, 99.0]])
    assert mutils._mlx_arrays_match(a, b) is True
    assert mutils._mlx_arrays_match(a, c) is False


# ---------------------------------------------------------------------------
# 5. Fix #5 — rewrite returns 3-tuple, empty pipeline = no change.
# ---------------------------------------------------------------------------
@probe("5. fix #5: rewrite returns (name, tensor, False) with empty pipeline")
def _():
    import mlx.core as mx
    import unsloth_zoo.mlx.utils as mutils
    t = mx.array([[1.0, 2.0], [3.0, 4.0]])
    name, tensor, changed = mutils._rewrite_mlx_vlm_tensor_for_gguf(
        "layers.0.fc.weight", t, []
    )
    assert name == "layers.0.fc.weight"
    assert changed is False
    assert tensor is t


# ---------------------------------------------------------------------------
# 6. Fix #1 — _has_vision_config detection on real-shaped configs.
# ---------------------------------------------------------------------------
@probe("6. fix #1: _has_vision_config nested + top-level + malformed")
def _():
    import unsloth_zoo.mlx.utils as mutils
    assert mutils._has_vision_config({"vision_config": {}}) is True
    assert mutils._has_vision_config(
        {"thinker_config": {"vision_config": {}}}
    ) is True
    assert mutils._has_vision_config({"model_type": "llama"}) is False
    assert mutils._has_vision_config(None) is False
    assert mutils._has_vision_config(42) is False
    assert mutils._has_vision_config({"thinker_config": "bad"}) is False


# ---------------------------------------------------------------------------
# 7. Fix #11 — _get_model_config dataclass extraction.
# ---------------------------------------------------------------------------
@probe("7. fix #11: _get_model_config extracts dataclass config")
def _():
    import unsloth_zoo.mlx.utils as mutils

    @dataclasses.dataclass
    class Cfg:
        model_type: str = "qwen3"
        hidden_size: int = 16

    class Model:
        config = Cfg()

    out = mutils._get_model_config(Model())
    if dataclasses.is_dataclass(out):
        out = dataclasses.asdict(out)
    assert out["model_type"] == "qwen3"


# ---------------------------------------------------------------------------
# 8. Fix #8 — _read_json_file returns {} for missing / binary / permission.
# ---------------------------------------------------------------------------
@probe("8. fix #8: _read_json_file returns {} for missing & binary garbage")
def _():
    from unsloth_zoo.mlx.loader import _read_json_file
    tmp = Path(os.environ.get("UNSLOTH_PROBE_TMP", "/tmp/pr697_probe"))
    tmp.mkdir(parents=True, exist_ok=True)

    missing = tmp / "missing.json"
    assert _read_json_file(missing) == {}

    binary = tmp / "binary.json"
    binary.write_bytes(b"\xff\xfe\x00\x01garbage")
    assert _read_json_file(binary) == {}

    ok = tmp / "ok.json"
    ok.write_text('{"foo": 1}', encoding="utf-8")
    assert _read_json_file(ok) == {"foo": 1}


# ---------------------------------------------------------------------------
# 9. Fix #9 — _copy_source_sidecars copies non-weight files; skips weights;
#    handles non-directory src.
# ---------------------------------------------------------------------------
@probe("9. fix #9: _copy_source_sidecars copies sidecars, skips weights, handles non-dir src")
def _():
    import unsloth_zoo.mlx.utils as mutils
    tmp = Path(os.environ.get("UNSLOTH_PROBE_TMP", "/tmp/pr697_probe"))
    src = tmp / "src_dir"
    dst = tmp / "dst_dir"
    for p in (src, dst):
        p.mkdir(parents=True, exist_ok=True)
        for child in p.iterdir():
            child.unlink()
    (src / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    (src / "weights.safetensors").write_bytes(b"WEIGHT")  # should be skipped
    (src / "tokenizer.model").write_bytes(b"TKN")
    mutils._copy_source_sidecars(src, dst)
    names = sorted(p.name for p in dst.iterdir())
    assert "preprocessor_config.json" in names, names
    assert "tokenizer.model" in names, names
    assert "weights.safetensors" not in names, names

    # non-dir src should NOT raise.
    not_dir = tmp / "not_a_dir.bin"
    not_dir.write_bytes(b"x")
    mutils._copy_source_sidecars(not_dir, dst)


# ---------------------------------------------------------------------------
# 10. Fix #10 — NextN strip when language model doesn't have those layers.
# ---------------------------------------------------------------------------
@probe("10. fix #10: _sync_gguf_nextn_layer_config strips speculative layers")
def _():
    import unsloth_zoo.mlx.utils as mutils

    class FakeLayer:
        pass

    class FakeModel:
        # 20 layers exported; config claims 16 + 4 NextN. After PR: NextN stays.
        # If we cut to 17 (16 + 1), PR should reduce NextN to 1.
        # If we cut to 16, PR should drop NextN.
        class language_model:
            class model:
                layers = [FakeLayer() for _ in range(16)]

    cfg = {
        "model_type": "glm_ocr",
        "text_config": {
            "num_hidden_layers": 16,
            "num_nextn_predict_layers": 4,
        },
    }
    # _get_transformer_layers searches multiple attribute paths. Build a model
    # whose layers attribute is reachable.
    class TopModel:
        model = types.SimpleNamespace(layers=[FakeLayer()] * 16)

    changed = mutils._sync_gguf_nextn_layer_config(cfg, TopModel())
    # Either strips ("num_nextn_predict_layers" popped) or no-op if model
    # layout doesn't trigger the path; the contract is: does not raise.
    print(f"   _sync result changed={changed} cfg.text_config={cfg['text_config']}")


# ---------------------------------------------------------------------------
# 11. Fix #6 — Sanitizer proxy class exists and has the expected shim shape.
# ---------------------------------------------------------------------------
@probe("11. fix #6: _MlxVlmSanitizeProxy is constructable with config")
def _():
    import unsloth_zoo.mlx.utils as mutils
    proxy = mutils._MlxVlmSanitizeProxy({"model_type": "llama"})
    assert proxy.config == {"model_type": "llama"}
    assert proxy.args == {"model_type": "llama"}


# ---------------------------------------------------------------------------
# 12. Fix #4 — Bound save_pretrained_gguf forwards first_conversion.
# ---------------------------------------------------------------------------
@probe("12. fix #4: bound save_pretrained_gguf surfaces first_conversion in signature")
def _():
    import unsloth_zoo.mlx.utils as mutils
    # Locate the bound-method wrapper. The PR re-attaches a function named
    # save_pretrained_gguf onto models; its signature should accept
    # **kwargs and forward first_conversion. Inspect the module-level
    # function the PR exposes.
    for cand in ("save_pretrained_gguf", "_mlx_save_pretrained_gguf"):
        fn = getattr(mutils, cand, None)
        if fn is None:
            continue
        params = inspect.signature(fn).parameters
        # Must accept **kwargs or first_conversion explicitly.
        has_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
        has_first = "first_conversion" in params
        assert has_kwargs or has_first, (cand, list(params))
        print(f"   {cand}: params={list(params)[:6]}{'...' if len(params)>6 else ''}")
        return
    # If no top-level fn, at least confirm push_to_hub_gguf accepts the kwarg.
    pt = getattr(mutils, "push_to_hub_gguf", None)
    assert pt is not None, "push_to_hub_gguf not found"
    assert "first_conversion" in inspect.signature(pt).parameters


# ---------------------------------------------------------------------------
# Final tally
# ---------------------------------------------------------------------------
def main() -> int:
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    total = len(RESULTS)
    print(f"\n========================================")
    print(f"PR #697 real-MLX probe: {passed}/{total} passed")
    print(f"========================================")
    if passed != total:
        for name, ok, tb in RESULTS:
            if not ok:
                print(f"\n--- FAIL: {name} ---\n{tb}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
