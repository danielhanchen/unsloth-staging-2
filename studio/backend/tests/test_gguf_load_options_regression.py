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
            patch.object(route, "ModelConfig", SimpleNamespace(from_identifier = lambda **_: config)),
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


# ── FIX 5: an unused draft path (off / user --spec-type) must not 400 a load ──


def test_load_disabled_spec_accepts_missing_draft_path():
    """speculative_type="off" never emits --model-draft, so a stale/absent
    draft_model_path must NOT 400 the load: the drafter would never launch. The
    load reaches the real GGUF load (drafter still threaded through) instead of
    the existence 400 (Codex #7239)."""
    route = _load_route_module("gguf_opts_regression_off_missing_draft")
    request = LoadRequest(
        model_path = "/local/model",
        draft_model_path = "/tmp/does-not-exist-off-7239.gguf",
        speculative_type = "off",
    )
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

    # No existence 400; the load proceeded (only the missing test server stops it).
    assert "Draft model path" not in (exc.value.detail or "")
    assert captured.get("mtp_draft_path") == "/tmp/does-not-exist-off-7239.gguf"


def test_load_mtp_still_rejects_missing_draft_path():
    """An explicit drafter under an effective-MTP mode (mtp) is still hard-400'd
    when the path is absent, proving the Fix 5 gate only relaxes the modes that
    never launch a drafter (Codex #7239)."""
    route = _load_route_module("gguf_opts_regression_mtp_missing_draft")
    request = LoadRequest(
        model_path = "/local/model",
        draft_model_path = "/tmp/does-not-exist-mtp-7239.gguf",
        speculative_type = "mtp",
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


# ── FIX 2: an omitted draft path inherits the live server's explicit drafter ──


def test_omitted_draft_inherits_loaded_custom_drafter(tmp_path):
    """A local GGUF serving an EXPLICIT custom drafter, reloaded by a request that
    OMITS draft_model_path, must keep the drafter rather than drop it back to the
    auto sibling. _resolve_inherited_draft_path returns the live drafter for the
    omitted-field reload of the same local GGUF (Codex #7239)."""
    route = _load_route_module("gguf_opts_regression_inherit_draft")
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"gguf-stub")
    custom_draft = "/custom/x.gguf"

    llama = SimpleNamespace(gguf_path = str(gguf), mtp_draft_path = custom_draft)
    config = _local_gguf_config(gguf_file = str(gguf))

    # Omitted field -> inherit the live drafter.
    request_omitted = LoadRequest(model_path = "/local/model")
    with patch.object(route, "get_llama_cpp_backend", return_value = llama):
        assert (
            route._resolve_inherited_draft_path(request_omitted, config, "/local/model")
            == custom_draft
        )

    # Explicit clear (field present, empty) -> do NOT inherit (the clear owns it).
    request_cleared = LoadRequest(model_path = "/local/model", draft_model_path = "")
    with patch.object(route, "get_llama_cpp_backend", return_value = llama):
        assert route._resolve_inherited_draft_path(request_cleared, config, "/local/model") is None

    # A different loaded GGUF -> never leak the drafter across models.
    other = _local_gguf_config(gguf_file = str(tmp_path / "other.gguf"))
    (tmp_path / "other.gguf").write_bytes(b"gguf-stub")
    with patch.object(route, "get_llama_cpp_backend", return_value = llama):
        assert route._resolve_inherited_draft_path(request_omitted, other, "/local/model") is None


def test_omitted_draft_reload_threads_inherited_drafter(tmp_path):
    """End to end: the omitted-field reload sizes the inherited drafter into the
    config the training guard sees, so the reload preserves it instead of dropping
    it (the old reload only set gguf_mtp_file for a truthy request field) (#7239)."""
    route = _load_route_module("gguf_opts_regression_inherit_draft_e2e")
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"gguf-stub")
    custom_draft = "/custom/x.gguf"
    config = _local_gguf_config(gguf_file = str(gguf))
    request = LoadRequest(model_path = "/local/model")  # draft_model_path omitted

    llama = SimpleNamespace(gguf_path = str(gguf), mtp_draft_path = custom_draft, extra_args = [])
    captured = {}

    def _capture_guard(cfg, **kwargs):
        captured["mtp"] = getattr(cfg, "gguf_mtp_file", None)
        raise HTTPException(status_code = 409, detail = "stop-after-guard")

    with (
        patch.object(route, "ModelConfig", SimpleNamespace(from_identifier = lambda **_: config)),
        patch.object(route, "get_llama_cpp_backend", return_value = llama),
        patch.object(
            route, "get_inference_backend", return_value = SimpleNamespace(active_model_name = None)
        ),
        patch.object(route, "_guard_chat_load_against_training", _capture_guard),
        patch.object(route.asyncio, "to_thread", new = _inline_to_thread),
        patch.object(route, "_hf_offline_if_dns_dead", nullcontext),
    ):
        with pytest.raises(HTTPException) as exc:
            asyncio.run(route._load_model_impl(request, _fastapi_request(), current_subject = "u"))

    assert exc.value.status_code == 409
    assert captured["mtp"] == custom_draft


# ── FIX 4: a remote root mtp- companion is only counted when it will launch ──


def test_remote_companion_bytes_gate_mtp():
    """_remote_gguf_companion_bytes excludes the repo-root mtp- drafter when
    include_mtp is False (the drafter will not be fetched under off/ngram/user
    --spec-type) and includes it when True (Codex #7239)."""
    route = _load_route_module("gguf_opts_regression_remote_companion")

    info = SimpleNamespace(
        siblings = [
            SimpleNamespace(rfilename = "model-Q4_K_M.gguf", size = 1000),
            SimpleNamespace(rfilename = "mtp-model-Q4_K_M.gguf", size = 200),
            SimpleNamespace(rfilename = "MTP/mtp-model.gguf", size = 500),  # subdir: ignored
        ]
    )

    with patch("huggingface_hub.model_info", return_value = info):
        with_mtp = route._remote_gguf_companion_bytes(
            "org/repo", hf_token = None, include_mmproj = False, include_mtp = True
        )
        without_mtp = route._remote_gguf_companion_bytes(
            "org/repo", hf_token = None, include_mmproj = False, include_mtp = False
        )

    assert with_mtp == 200  # only the repo-root mtp- sibling
    assert without_mtp == 0


def test_estimate_required_gb_forwards_include_mtp():
    """_estimate_gguf_required_gb forwards include_mtp_draft to the remote companion
    sizer, so a remote drafter picked with the mode off does not inflate the VRAM
    estimate and 409 a load that fits without it (Codex #7239)."""
    route = _load_route_module("gguf_opts_regression_estimate_forward")
    config = _local_gguf_config(gguf_file = None, gguf_hf_repo = "org/repo", gguf_variant = "Q4_K_M")

    seen = {}

    def _fake_variants(repo, hf_token = None):
        return ([SimpleNamespace(quant = "Q4_K_M", size_bytes = 1_000_000_000)], False)

    def _fake_companions(
        repo,
        *,
        hf_token,
        include_mmproj,
        include_mtp = True,
    ):
        seen["include_mtp"] = include_mtp
        return 200_000_000 if include_mtp else 0

    with (
        patch("utils.models.model_config.list_gguf_variants", _fake_variants),
        patch.object(route, "_remote_gguf_companion_bytes", _fake_companions),
    ):
        gb_off = route._estimate_gguf_required_gb(config, include_mtp_draft = False)
        assert seen["include_mtp"] is False
        gb_on = route._estimate_gguf_required_gb(config, include_mtp_draft = True)
        assert seen["include_mtp"] is True

    assert gb_on > gb_off  # the drafter's bytes are only counted when it will launch


# ── FIX 1: /validate mirrors the follow-up load's draft/spec choices ──────────


def test_validate_off_mode_threads_drafter_and_skips_existence_check(tmp_path):
    """/validate now carries speculative_type + draft_model_path, so the training
    guard sizes the drafter exactly as /load does and an unused drafter (mode off)
    is not existence-checked. The guard sees the mode and the threaded drafter, and
    an absent path under "off" does NOT 400 (matching /load) (Codex #7239)."""
    route = _load_route_module("gguf_opts_regression_validate_off")
    draft = tmp_path / "draft.gguf"
    draft.write_bytes(b"gguf-stub")
    request = ValidateModelRequest(
        model_path = "unsloth/test",
        draft_model_path = str(draft),
        speculative_type = "off",
    )
    config = _local_gguf_config(identifier = "unsloth/test", gguf_file = str(tmp_path / "m.gguf"))

    captured = {}

    def _capture_guard(cfg, **kwargs):
        captured["spec"] = kwargs.get("speculative_type")
        captured["mtp"] = getattr(cfg, "gguf_mtp_file", None)
        return None

    with (
        patch.object(route, "ModelConfig", SimpleNamespace(from_identifier = lambda **_: config)),
        patch.object(
            route,
            "get_llama_cpp_backend",
            return_value = SimpleNamespace(extra_args = [], mtp_draft_path = None, gguf_path = None),
        ),
        patch.object(route, "_guard_chat_load_against_training", _capture_guard),
        patch.object(route.asyncio, "to_thread", new = _inline_to_thread),
    ):
        resp = asyncio.run(route.validate_model(request, current_subject = "u"))

    assert resp.valid is True
    assert captured["spec"] == "off"  # mode forwarded to the guard
    assert captured["mtp"] == str(draft)  # drafter threaded into the estimate


def test_validate_rejects_missing_draft_under_mtp():
    """A typo'd explicit draft_model_path under an effective-MTP mode must 400 in
    /validate (not pass here then 400 in /load after the frontend unloaded the
    working model) (Codex #7239)."""
    route = _load_route_module("gguf_opts_regression_validate_mtp_bad_draft")
    request = ValidateModelRequest(
        model_path = "unsloth/test",
        draft_model_path = "/tmp/does-not-exist-validate-7239.gguf",
        speculative_type = "mtp",
    )
    config = _local_gguf_config(identifier = "unsloth/test", gguf_file = "/tmp/m.gguf")

    with (
        patch.object(route, "ModelConfig", SimpleNamespace(from_identifier = lambda **_: config)),
        patch.object(
            route,
            "get_llama_cpp_backend",
            return_value = SimpleNamespace(extra_args = [], mtp_draft_path = None, gguf_path = None),
        ),
        patch.object(route, "_guard_chat_load_against_training", return_value = None),
        patch.object(route.asyncio, "to_thread", new = _inline_to_thread),
    ):
        with pytest.raises(HTTPException) as exc:
            asyncio.run(route.validate_model(request, current_subject = "u"))

    assert exc.value.status_code == 400
    assert "Draft model path" in exc.value.detail


# ── BUG B: /validate inherits the active drafter when draft_model_path is omitted ──


def test_validate_omitted_draft_inherits_active_drafter(tmp_path):
    """/validate now mirrors /load's omitted-field inheritance: when a custom drafter
    is live on the SAME local GGUF and the ValidateModelRequest OMITS draft_model_path,
    the training guard must see the inherited drafter on the config so its estimate
    matches the follow-up /load (which inherits the same drafter). Otherwise validate
    sizes the smaller no-draft estimate, approves, the frontend unloads the model, and
    /load then 409s after the model is already gone (Codex #7239)."""
    route = _load_route_module("gguf_opts_regression_validate_inherit_omitted")
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"gguf-stub")
    custom_draft = "/custom/x.gguf"
    request = ValidateModelRequest(model_path = "unsloth/test")  # draft_model_path omitted
    config = _local_gguf_config(identifier = "unsloth/test", gguf_file = str(gguf))

    captured = {}

    def _capture_guard(cfg, **kwargs):
        captured["mtp"] = getattr(cfg, "gguf_mtp_file", None)
        return None

    with (
        patch.object(route, "ModelConfig", SimpleNamespace(from_identifier = lambda **_: config)),
        patch.object(
            route,
            "get_llama_cpp_backend",
            return_value = SimpleNamespace(
                extra_args = [], mtp_draft_path = custom_draft, gguf_path = str(gguf)
            ),
        ),
        patch.object(route, "_guard_chat_load_against_training", _capture_guard),
        patch.object(route.asyncio, "to_thread", new = _inline_to_thread),
    ):
        resp = asyncio.run(route.validate_model(request, current_subject = "u"))

    assert resp.valid is True
    assert captured["mtp"] == custom_draft  # inherited drafter sized into the guard


def test_validate_explicit_draft_unchanged_by_inherit(tmp_path):
    """The explicit-path FIX-1 branch is unchanged: an explicit draft_model_path still
    sizes exactly that path (never the live drafter) into the guard (Codex #7239)."""
    route = _load_route_module("gguf_opts_regression_validate_explicit_still")
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"gguf-stub")
    draft = tmp_path / "draft.gguf"
    draft.write_bytes(b"gguf-stub")
    request = ValidateModelRequest(
        model_path = "unsloth/test",
        draft_model_path = str(draft),
        speculative_type = "mtp",
    )
    config = _local_gguf_config(identifier = "unsloth/test", gguf_file = str(gguf))

    captured = {}

    def _capture_guard(cfg, **kwargs):
        captured["mtp"] = getattr(cfg, "gguf_mtp_file", None)
        return None

    with (
        patch.object(route, "ModelConfig", SimpleNamespace(from_identifier = lambda **_: config)),
        patch.object(
            route,
            "get_llama_cpp_backend",
            # A different live drafter must NOT override the explicit request path.
            return_value = SimpleNamespace(
                extra_args = [], mtp_draft_path = "/custom/other.gguf", gguf_path = str(gguf)
            ),
        ),
        patch.object(route, "_guard_chat_load_against_training", _capture_guard),
        patch.object(route.asyncio, "to_thread", new = _inline_to_thread),
    ):
        resp = asyncio.run(route.validate_model(request, current_subject = "u"))

    assert resp.valid is True
    assert captured["mtp"] == str(draft)  # explicit path wins, no inheritance


def test_validate_omitted_draft_not_inherited_across_models(tmp_path):
    """A drafter live on a DIFFERENT local GGUF must not be inherited into a validate
    of another GGUF: the guard sees no drafter (no cross-model leak) (Codex #7239)."""
    route = _load_route_module("gguf_opts_regression_validate_inherit_crossmodel")
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"gguf-stub")
    other = tmp_path / "other.gguf"
    other.write_bytes(b"gguf-stub")
    request = ValidateModelRequest(model_path = "unsloth/test")  # draft_model_path omitted
    config = _local_gguf_config(identifier = "unsloth/test", gguf_file = str(gguf))

    captured = {}

    def _capture_guard(cfg, **kwargs):
        captured["mtp"] = getattr(cfg, "gguf_mtp_file", None)
        return None

    with (
        patch.object(route, "ModelConfig", SimpleNamespace(from_identifier = lambda **_: config)),
        patch.object(
            route,
            "get_llama_cpp_backend",
            # Live drafter, but on a different GGUF than the one being validated.
            return_value = SimpleNamespace(
                extra_args = [], mtp_draft_path = "/custom/x.gguf", gguf_path = str(other)
            ),
        ),
        patch.object(route, "_guard_chat_load_against_training", _capture_guard),
        patch.object(route.asyncio, "to_thread", new = _inline_to_thread),
    ):
        resp = asyncio.run(route.validate_model(request, current_subject = "u"))

    assert resp.valid is True
    assert captured["mtp"] is None  # no drafter inherited across models


# ── BUG A: dedupe compares the RAW requested GPU pin, not the fit-narrowed one ──


def _pinned_backend(
    requested_gpu_ids,
    effective_gpu_ids,
    *,
    requested_keep_resident = False,
    keep_resident = False,
    requested_mlock = False,
    mlock = False,
    extra_args = (),
):
    """A live GGUF backend whose fit narrowed an explicit pin: requested_gpu_ids
    keeps the RAW request; gpu_ids echoes the EFFECTIVE (narrowed) pin for /status.
    requested_keep_resident / requested_mlock likewise keep the RAW residency request
    while keep_resident / mlock echo the EFFECTIVE state (a last-wins extra --mmap /
    --no-mlock flips them). Every non-selected field matches a default LoadRequest so
    only the pin under test decides the dedupe (Codex #7239)."""
    return SimpleNamespace(
        is_diffusion = False,
        gpu_ids = effective_gpu_ids,
        requested_gpu_ids = requested_gpu_ids,
        keep_resident = keep_resident,
        mlock = mlock,
        requested_keep_resident = requested_keep_resident,
        requested_mlock = requested_mlock,
        mtp_draft_path = None,
        requested_n_ctx = 0,
        cache_type_kv = None,
        extra_args = list(extra_args),
        tensor_parallel = False,
        gpu_memory_mode = "auto",
        gpu_layers = -1,
        n_cpu_moe = 0,
        tensor_split = None,
        layer_preserves_tensor_intent = False,
        requested_spec_mode = None,
        hf_repo = None,
        spec_fallback_reason = None,
        spec_draft_n_max = None,
        chat_template_override = None,
        gguf_path = None,
    )


def test_dedupe_matches_narrowed_pin_resent_raw():
    """The fitter narrowed an explicit [0, 1] to [0] and recorded gpu_ids=[0] for
    /status, but the RAW request [0, 1] is kept as requested_gpu_ids. The frontend
    re-sends [0, 1] on the next Apply; the dedupe must compare RAW-vs-RAW ([0, 1] ==
    [0, 1]) and MATCH so the healthy server is not needlessly killed and reloaded
    (correctness point 3) (Codex #7239)."""
    route = _load_route_module("gguf_opts_regression_dedupe_narrowed")
    backend = _pinned_backend(requested_gpu_ids = [0, 1], effective_gpu_ids = [0])
    request = LoadRequest(model_path = "/tmp/model.gguf", gpu_ids = [0, 1])

    assert route._request_matches_loaded_settings(request, backend) is True
    # /status still echoes the EFFECTIVE (narrowed) pin, not the raw request.
    assert backend.gpu_ids == [0]


def test_dedupe_reloads_on_explicit_pin_change():
    """An explicit CHANGE from [0, 1] to [0] must still reload: raw [0] != raw
    [0, 1], so the user's new explicit pin is honored (correctness point 4)."""
    route = _load_route_module("gguf_opts_regression_dedupe_change")
    backend = _pinned_backend(requested_gpu_ids = [0, 1], effective_gpu_ids = [0])
    request = LoadRequest(model_path = "/tmp/model.gguf", gpu_ids = [0])

    assert route._request_matches_loaded_settings(request, backend) is False


def test_dedupe_matches_non_narrowed_explicit_pin():
    """A non-narrowed explicit [0, 1] -> [0, 1] still matches (correctness point 2)."""
    route = _load_route_module("gguf_opts_regression_dedupe_nonnarrowed")
    backend = _pinned_backend(requested_gpu_ids = [0, 1], effective_gpu_ids = [0, 1])
    request = LoadRequest(model_path = "/tmp/model.gguf", gpu_ids = [0, 1])

    assert route._request_matches_loaded_settings(request, backend) is True


def test_dedupe_matches_auto_none_vs_none():
    """An auto load (gpu_ids omitted) still dedupes None-vs-None (point 1)."""
    route = _load_route_module("gguf_opts_regression_dedupe_auto")
    backend = _pinned_backend(requested_gpu_ids = None, effective_gpu_ids = None)
    request = LoadRequest(model_path = "/tmp/model.gguf")  # gpu_ids omitted

    assert route._request_matches_loaded_settings(request, backend) is True


# ── dedupe compares the RAW requested residency, not the effective state a ──────
# ── last-wins extra --mmap / --no-mlock flipped (same class as the GPU pin) ─────


def test_dedupe_matches_residency_negated_by_extra_resent():
    """keep_model_in_vram=true launched with a last-wins extra --mmap records the
    EFFECTIVE keep_resident=False for /status while the RAW request is kept as
    requested_keep_resident=True. The frontend re-sends the identical request; the
    dedupe must compare RAW-vs-RAW (true == true) and MATCH so a healthy keep-in-VRAM
    server is not needlessly torn down and reloaded, mirroring the GPU-pin case (#7239)."""
    route = _load_route_module("gguf_opts_regression_dedupe_residency_match")
    backend = _pinned_backend(
        requested_gpu_ids = None,
        effective_gpu_ids = None,
        requested_keep_resident = True,
        keep_resident = False,
        extra_args = ["--mmap"],
    )
    request = LoadRequest(
        model_path = "/tmp/model.gguf",
        keep_model_in_vram = True,
        llama_extra_args = ["--mmap"],
    )

    assert route._request_matches_loaded_settings(request, backend) is True
    # /status still echoes the EFFECTIVE (extra-negated) residency, not the raw request.
    assert backend.keep_resident is False


def test_dedupe_reloads_on_residency_request_change():
    """A real change to the residency request (keep_model_in_vram true -> false) must
    still reload: raw false != raw true, so the user's new intent is honored even
    though the effective keep_resident was already False from the prior --mmap."""
    route = _load_route_module("gguf_opts_regression_dedupe_residency_change")
    backend = _pinned_backend(
        requested_gpu_ids = None,
        effective_gpu_ids = None,
        requested_keep_resident = True,
        keep_resident = False,
        extra_args = ["--mmap"],
    )
    request = LoadRequest(
        model_path = "/tmp/model.gguf",
        keep_model_in_vram = False,
        llama_extra_args = ["--mmap"],
    )

    assert route._request_matches_loaded_settings(request, backend) is False


def test_dedupe_matches_mlock_negated_by_extra_resent():
    """mlock=true launched with a last-wins extra --no-mlock records the EFFECTIVE
    mlock=False while keeping requested_mlock=True; an identical re-request dedupes
    RAW-vs-RAW rather than reloading, the mlock counterpart of the above (#7239)."""
    route = _load_route_module("gguf_opts_regression_dedupe_mlock_match")
    backend = _pinned_backend(
        requested_gpu_ids = None,
        effective_gpu_ids = None,
        requested_mlock = True,
        mlock = False,
        extra_args = ["--no-mlock"],
    )
    request = LoadRequest(
        model_path = "/tmp/model.gguf",
        mlock = True,
        llama_extra_args = ["--no-mlock"],
    )

    assert route._request_matches_loaded_settings(request, backend) is True
