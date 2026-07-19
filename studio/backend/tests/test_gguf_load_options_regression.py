# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Route-level regression tests for the GGUF load options (issue #7164, Codex #7239):

- an explicit draft (--model-draft) is sized into the training coexistence guard
  before the load, so a load the guard permits cannot OOM active training;
- an explicit local draft path is validated (404 -> 400) instead of silently
  loading without the requested drafter;
- /validate resolves GGUF gpu_ids the same way /load does, so a preflight of the
  intended payload is not rejected for a selection /load accepts.
"""

import asyncio
import importlib.util
from contextlib import ExitStack, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from models.inference import LoadRequest, ValidateModelRequest

_BACKEND_ROOT = Path(__file__).resolve().parent.parent


async def _inline_to_thread(func, /, *args, **kwargs):
    return func(*args, **kwargs)


def _load_route_module(name: str):
    spec = importlib.util.spec_from_file_location(name, _BACKEND_ROOT / "routes" / "inference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fastapi_request():
    return SimpleNamespace(
        app = SimpleNamespace(state = SimpleNamespace(llama_parallel_slots = 1)),
        scope = {},
    )


def _local_gguf_config(**overrides):
    base = dict(
        is_gguf = True,
        is_lora = False,
        is_vision = False,
        is_audio = False,
        audio_type = None,
        has_audio_input = False,
        gguf_hf_repo = None,
        gguf_file = "/tmp/model.gguf",
        gguf_mmproj_file = None,
        gguf_mtp_file = None,  # no auto-detected sibling
        gguf_variant = None,
        identifier = "/local/model",
        display_name = "model",
        path = "/local/model",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_load_sizes_explicit_draft_before_training_guard():
    """The training coexistence guard must see the explicit drafter on the config so
    its VRAM estimate includes the extra --model-draft; otherwise a load the guard
    permits can OOM the active training run (Codex #7239)."""
    route = _load_route_module("gguf_opts_regression_draft_guard")
    request = LoadRequest(
        model_path = "/local/model",
        draft_model_path = "/tmp/explicit-draft.gguf",
    )
    config = _local_gguf_config()

    captured = {}

    def _capture_guard(cfg, **kwargs):
        captured["mtp"] = getattr(cfg, "gguf_mtp_file", None)
        # Short-circuit before the real GGUF load (no llama-server in the test).
        raise HTTPException(status_code = 409, detail = "stop-after-guard")

    with (
        patch.object(route, "ModelConfig", SimpleNamespace(from_identifier = lambda **_: config)),
        patch.object(route, "_guard_chat_load_against_training", _capture_guard),
        patch.object(route.asyncio, "to_thread", new = _inline_to_thread),
        patch.object(route, "_hf_offline_if_dns_dead", nullcontext),
    ):
        with pytest.raises(HTTPException) as exc:
            asyncio.run(route._load_model_impl(request, _fastapi_request(), current_subject = "u"))

    assert exc.value.status_code == 409
    assert captured["mtp"] == "/tmp/explicit-draft.gguf"


def test_load_rejects_missing_explicit_local_draft():
    """A non-native local load with an explicit draft_model_path that does not exist
    must return 400 rather than silently loading without the requested drafter (the
    optional-drafter drop path would otherwise swallow a typo) (Codex #7239)."""
    route = _load_route_module("gguf_opts_regression_draft_validate")
    request = LoadRequest(
        model_path = "/local/model",
        draft_model_path = "/tmp/this-draft-does-not-exist-7239.gguf",
    )
    config = _local_gguf_config()

    with (
        patch.object(route, "ModelConfig", SimpleNamespace(from_identifier = lambda **_: config)),
        patch.object(route, "_guard_chat_load_against_training", return_value = None),
        patch.object(route.asyncio, "to_thread", new = _inline_to_thread),
        patch.object(route, "_hf_offline_if_dns_dead", nullcontext),
    ):
        with pytest.raises(HTTPException) as exc:
            asyncio.run(route._load_model_impl(request, _fastapi_request(), current_subject = "u"))

    assert exc.value.status_code == 400
    assert "Draft model path" in exc.value.detail


def test_load_accepts_existing_explicit_local_draft(tmp_path):
    """A valid explicit local drafter passes the new existence check and reaches the
    real load (which then fails only because there is no llama-server in the test),
    proving the check does not reject legitimate paths (Codex #7239)."""
    route = _load_route_module("gguf_opts_regression_draft_ok")
    draft = tmp_path / "draft.gguf"
    draft.write_bytes(b"gguf-stub")
    request = LoadRequest(model_path = "/local/model", draft_model_path = str(draft))
    config = _local_gguf_config()

    captured = {}

    def _fail_load(**kwargs):
        captured["mtp_draft_path"] = kwargs.get("mtp_draft_path")
        raise RuntimeError("no llama-server in test")

    with (
        patch.object(route, "ModelConfig", SimpleNamespace(from_identifier = lambda **_: config)),
        patch.object(route, "_guard_chat_load_against_training", return_value = None),
        patch.object(route.asyncio, "to_thread", new = _inline_to_thread),
        patch.object(route, "_hf_offline_if_dns_dead", nullcontext),
        patch.object(
            route.get_llama_cpp_backend(), "load_model", side_effect = _fail_load, create = True
        ),
    ):
        with pytest.raises(HTTPException) as exc:
            asyncio.run(route._load_model_impl(request, _fastapi_request(), current_subject = "u"))

    # Not the 400 existence error; the drafter path was passed through to the load.
    assert "Draft model path" not in (exc.value.detail or "")
    assert captured.get("mtp_draft_path") == str(draft)


def _loaded_gguf_backend(**overrides):
    """A live llama.cpp backend already serving a GGUF, with every attribute the
    already-loaded dedupe LoadResponse reads."""
    base = dict(
        is_loaded = True,
        hf_variant = None,
        model_identifier = "/tmp/model.gguf",
        _audio_probed = True,
        _is_vision = False,
        is_diffusion = False,
        _is_audio = False,
        _audio_type = None,
        _has_audio_input = False,
        context_length = 4096,
        max_context_length = 8192,
        native_context_length = 8192,
        supports_reasoning = False,
        reasoning_style = "enable_thinking",
        reasoning_effort_levels = [],
        reasoning_always_on = False,
        supports_preserve_thinking = False,
        supports_tools = False,
        chat_template = None,
        requested_spec_mode = None,
        spec_draft_n_max = None,
        tensor_parallel = False,
        # GPU-memory knobs the already-loaded echo reads back (merged from main):
        # a keep-in-VRAM load leaves these at their auto defaults.
        gpu_memory_mode = "auto",
        gpu_layers = -1,
        n_cpu_moe = 0,
        tensor_split = None,
        n_layers = None,
        n_moe_layers = 0,
        # The three fields under test: launched-with residency/mlock/GPU pin.
        keep_resident = True,
        mlock = True,
        gpu_ids = [0],
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_already_loaded_gguf_echoes_active_load_options():
    """An identical /load that dedupes to the live server must echo its actual
    keep_model_in_vram / mlock / gpu_ids, not Pydantic defaults (Codex #7239)."""
    route = _load_route_module("gguf_opts_regression_dedupe_echo")
    request = LoadRequest(
        model_path = "/tmp/model.gguf",
        keep_model_in_vram = True,
        mlock = True,
        gpu_ids = [0],
    )
    llama = _loaded_gguf_backend()

    with (
        patch.object(route, "get_llama_cpp_backend", return_value = llama),
        patch.object(
            route, "get_inference_backend", return_value = SimpleNamespace(active_model_name = None)
        ),
        patch.object(route, "load_inference_config", return_value = {}),
        patch.object(route, "resolve_effective_chat_template_override", return_value = None),
        patch.object(route, "_request_matches_loaded_settings", return_value = True),
    ):
        resp = asyncio.run(route._load_model_impl(request, _fastapi_request(), current_subject = "u"))

    assert resp.status == "already_loaded"
    assert resp.keep_model_in_vram is True
    assert resp.mlock is True
    assert resp.gpu_ids == [0]


def test_validate_accepts_gguf_gpu_ids_like_load():
    """/validate must route GGUF gpu_ids through the same resolution /load uses, not a
    blanket "not supported" reject, so a preflight of the intended payload agrees with
    the follow-up load (Codex #7239)."""
    route = _load_route_module("gguf_opts_regression_validate_ids")
    request = ValidateModelRequest(model_path = "unsloth/test", gpu_ids = [0, 1])
    config = _local_gguf_config(identifier = "unsloth/test", gguf_file = None)

    seen = {}

    def _fake_resolve(ids, is_vulkan = False):
        seen["ids"] = list(ids)
        return list(ids)

    with (
        patch.object(route, "ModelConfig", SimpleNamespace(from_identifier = lambda **_: config)),
        patch("utils.hardware.resolve_requested_gpu_ids", _fake_resolve),
        patch.object(route, "_guard_chat_load_against_training", return_value = None),
        patch.object(route.asyncio, "to_thread", new = _inline_to_thread),
    ):
        resp = asyncio.run(route.validate_model(request, current_subject = "u"))

    assert resp.valid is True
    assert seen["ids"] == [0, 1]


def test_validate_surfaces_invalid_gguf_gpu_ids_from_resolution():
    """An invalid GGUF gpu_ids selection now surfaces the resolver's actionable 400
    instead of the old blanket "not supported for GGUF" message (Codex #7239)."""
    route = _load_route_module("gguf_opts_regression_validate_bad_ids")
    request = ValidateModelRequest(model_path = "unsloth/test", gpu_ids = [99])
    config = _local_gguf_config(identifier = "unsloth/test", gguf_file = None)

    def _fake_resolve(ids, is_vulkan = False):
        raise ValueError("SENTINEL requested GPUs are outside the parent-visible set")

    with (
        patch.object(route, "ModelConfig", SimpleNamespace(from_identifier = lambda **_: config)),
        patch("utils.hardware.resolve_requested_gpu_ids", _fake_resolve),
        patch.object(route, "_guard_chat_load_against_training", return_value = None),
        patch.object(route.asyncio, "to_thread", new = _inline_to_thread),
    ):
        with pytest.raises(HTTPException) as exc:
            asyncio.run(route.validate_model(request, current_subject = "u"))

    assert exc.value.status_code == 400
    assert "SENTINEL" in exc.value.detail
    assert "not supported for GGUF" not in exc.value.detail


def test_classify_diffusion_gguf_ignores_bare_diffusion_name(tmp_path):
    """A GGUF whose name/path merely contains "diffusion" (but not the DiffusionGemma
    runner token) with a LOCAL header that decodes an ordinary architecture must NOT be
    classified as diffusion: the local header is authoritative, so the Vulkan gpu_ids
    reject cannot falsely fire on a normal llama-server GGUF (Codex #7239). The name-only
    hint stays scoped to the "diffusiongemma" family for a remote/uncached header."""
    route = _load_route_module("gguf_opts_regression_classify_diffusion")

    # LOCAL header authoritative: a real file whose probe decodes a non-diffusion arch.
    local_gguf = tmp_path / "stable-diffusion-prompt-writer.gguf"
    local_gguf.write_bytes(b"gguf-stub")
    local_config = _local_gguf_config(
        identifier = "/models/stable-diffusion/prompt-writer",
        gguf_file = str(local_gguf),
        gguf_hf_repo = None,
        gguf_variant = None,
    )

    def _fake_read_meta(self, path):
        # Same effect as the real probe on an ordinary GGUF: no diffusion flag, a
        # successfully decoded architecture (which proves it is a normal model).
        self._is_diffusion = False
        self._architecture = "llama"

    with patch.object(route.LlamaCppBackend, "_read_gguf_metadata", _fake_read_meta):
        assert route._classify_diffusion_gguf(local_config) is False

    # REMOTE/uncached + DiffusionGemma name token -> True (pre-download hint).
    remote_dg = _local_gguf_config(
        identifier = "unsloth/DiffusionGemma-4B-GGUF",
        gguf_file = None,
        gguf_hf_repo = None,
        gguf_variant = None,
    )
    assert route._classify_diffusion_gguf(remote_dg) is True

    # REMOTE/uncached + bare "diffusion" (not the runner family) -> None (stays guarded,
    # NOT misclassified as normal, but also not falsely rejected as DiffusionGemma).
    remote_bare = _local_gguf_config(
        identifier = "org/stable-diffusion-prompts-GGUF",
        gguf_file = None,
        gguf_hf_repo = None,
        gguf_variant = None,
    )
    assert route._classify_diffusion_gguf(remote_bare) is None


def test_validate_rejects_diffusion_gguf_gpu_ids_on_vulkan():
    """/validate must mirror /load's diffusion+Vulkan rejection: an explicit gpu_ids for a
    diffusion GGUF on a Vulkan build is rejected (400) BEFORE the gpu-id resolution, so a
    preflight cannot pass and let the frontend unload the working model before /load
    returns the same 400 (Codex #7239)."""
    route = _load_route_module("gguf_opts_regression_validate_diffusion_vulkan")
    request = ValidateModelRequest(model_path = "unsloth/test", gpu_ids = [0, 1])
    config = _local_gguf_config(identifier = "unsloth/test", gguf_file = None)

    seen = {}

    def _fake_resolve(ids, is_vulkan = False):
        seen["reached"] = True
        return list(ids)

    with (
        patch.object(route, "ModelConfig", SimpleNamespace(from_identifier = lambda **_: config)),
        patch.object(route.LlamaCppBackend, "_is_vulkan_backend", lambda *a, **k: True),
        patch.object(route, "_classify_diffusion_gguf", lambda config: True),
        patch("utils.hardware.resolve_requested_gpu_ids", _fake_resolve),
        patch.object(route, "_guard_chat_load_against_training", return_value = None),
        patch.object(route.asyncio, "to_thread", new = _inline_to_thread),
    ):
        with pytest.raises(HTTPException) as exc:
            asyncio.run(route.validate_model(request, current_subject = "u"))

    assert exc.value.status_code == 400
    assert "Vulkan" in exc.value.detail
    assert "DiffusionGemma" in exc.value.detail
    # The reject short-circuits before gpu-id resolution (as /load does).
    assert "reached" not in seen


def test_validate_rejects_absent_vulkan_ordinal_like_load():
    """/validate must mirror /load's Vulkan-ordinal existence check: a requested ordinal
    absent from the ggml-probed set is rejected (400 "not present") BEFORE the frontend
    unloads the working model, and a present ordinal is NOT falsely rejected (Codex
    #7239)."""
    route = _load_route_module("gguf_opts_regression_validate_vulkan_ordinal")
    config = _local_gguf_config(identifier = "unsloth/test", gguf_file = None)

    # Only Vulkan ordinal 0 is present (free/total bytes are irrelevant to the check).
    def _one_gpu(binary = None):
        return [(0, 8 * 1024**3, 16 * 1024**3)]

    def _common_patches():
        return [
            patch.object(
                route, "ModelConfig", SimpleNamespace(from_identifier = lambda **_: config)
            ),
            patch.object(route.LlamaCppBackend, "_is_vulkan_backend", lambda *a, **k: True),
            patch.object(route, "_classify_diffusion_gguf", lambda config: False),
            patch.object(
                route.LlamaCppBackend,
                "_find_llama_server_binary",
                lambda *a, **k: "/fake/llama-server",
            ),
            patch.object(route.LlamaCppBackend, "_get_gpu_memory", _one_gpu),
            patch.object(route, "_guard_chat_load_against_training", return_value = None),
            patch.object(route.asyncio, "to_thread", new = _inline_to_thread),
        ]

    # Absent ordinal [99] -> 400 "not present".
    def _resolve_99(ids, is_vulkan = False):
        return [99]

    with ExitStack() as stack:
        for cm in _common_patches():
            stack.enter_context(cm)
        stack.enter_context(patch("utils.hardware.resolve_requested_gpu_ids", _resolve_99))
        with pytest.raises(HTTPException) as exc:
            asyncio.run(
                route.validate_model(
                    ValidateModelRequest(model_path = "unsloth/test", gpu_ids = [99]),
                    current_subject = "u",
                )
            )
    assert exc.value.status_code == 400
    assert "not present" in exc.value.detail

    # Present ordinal [0] -> the ordinal check does NOT raise (a valid Vulkan pin is
    # accepted, proving the guard does not falsely reject).
    def _resolve_0(ids, is_vulkan = False):
        return [0]

    with ExitStack() as stack:
        for cm in _common_patches():
            stack.enter_context(cm)
        stack.enter_context(patch("utils.hardware.resolve_requested_gpu_ids", _resolve_0))
        resp = asyncio.run(
            route.validate_model(
                ValidateModelRequest(model_path = "unsloth/test", gpu_ids = [0]),
                current_subject = "u",
            )
        )
    assert resp.valid is True
