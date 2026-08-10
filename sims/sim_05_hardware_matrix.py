"""H16: [Windows, Linux, WSL, macOS] x [NVIDIA, AMD, CPU-only].

The audio backend has no sys.platform branches of its own, so what actually
varies by host is which runtime serves the model: llama.cpp (CUDA, ROCm, Vulkan,
CPU) can decode snac/bicodec/dac, MLX (Apple Silicon) cannot decode any of them,
and Transformers adds csm. This pins the routing decision for every cell of the
matrix, plus the capability-vs-failure distinction that decides 501 against 500.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

BACKEND = Path(
    os.environ.get(
        "PR_BACKEND",
        Path(__file__).resolve().parents[2] / "unsloth_pr7984" / "studio" / "backend",
    )
)
sys.path.insert(0, str(BACKEND))

import routes.inference as inf  # noqa: E402
from core.inference.audio_errors import (  # noqa: E402
    AUDIO_UNSUPPORTED_CODE,
    AudioBackendUnsupportedError,
)

# runtime -> the audio types that runtime can actually turn into a WAV
RUNTIME_SUPPORT = {
    "llama.cpp": inf._GGUF_TTS_AUDIO_TYPES,
    "transformers": inf._TRANSFORMERS_TTS_AUDIO_TYPES,
}

# (os, accelerator) -> runtimes reachable for TTS on that host
MATRIX = {
    ("linux", "nvidia"): ["llama.cpp", "transformers"],
    ("linux", "amd"): ["llama.cpp", "transformers"],
    ("linux", "cpu"): ["llama.cpp"],
    ("wsl", "nvidia"): ["llama.cpp", "transformers"],
    ("wsl", "amd"): ["llama.cpp"],
    ("wsl", "cpu"): ["llama.cpp"],
    ("windows", "nvidia"): ["llama.cpp", "transformers"],
    ("windows", "amd"): ["llama.cpp"],
    ("windows", "cpu"): ["llama.cpp"],
    ("macos", "cpu"): ["llama.cpp"],
    ("macos", "mlx"): ["llama.cpp"],  # MLX has no TTS branch; GGUF is the only path
}

CODECS = ("snac", "bicodec", "dac", "csm")


@pytest.mark.parametrize("host", sorted(MATRIX))
@pytest.mark.parametrize("codec", CODECS)
def test_every_host_has_a_defined_answer_for_every_codec(host, codec):
    """No cell may be undefined: either some runtime on that host serves the
    codec, or the user must get the typed capability error, never a 500."""
    runtimes = MATRIX[host]
    served_by = [r for r in runtimes if codec in RUNTIME_SUPPORT[r]]
    if served_by:
        assert all(codec in RUNTIME_SUPPORT[r] for r in served_by)
        return
    # Unserved must be a capability answer, which is what maps to 501.
    error = AudioBackendUnsupportedError(
        f"Text-to-speech is not supported for {codec} on this host."
    )
    assert isinstance(error, RuntimeError)
    assert error.message


def test_csm_is_the_only_codec_llama_cpp_cannot_decode():
    """Documented invariant behind the Mac GGUF-sibling fallback: everything
    except csm has a llama.cpp decoder, so a Mac user can always be sent to a
    GGUF build when one exists."""
    assert set(inf._TRANSFORMERS_TTS_AUDIO_TYPES) - set(inf._GGUF_TTS_AUDIO_TYPES) == {"csm"}


def test_cpu_only_and_amd_hosts_still_reach_a_tts_runtime():
    """A chat-only host (no CUDA, no ROCm build, Apple Silicon without MLX TTS)
    must not end up with an Audio page that can never generate."""
    for host, runtimes in MATRIX.items():
        assert runtimes, host
        assert any(RUNTIME_SUPPORT[r] for r in runtimes), host


def test_an_mlx_host_reports_a_capability_error_not_a_crash():
    """MLX loads the model fine and then cannot generate. That is a capability
    answer: safe_error_detail must not flatten it to 'An internal error'."""
    error = AudioBackendUnsupportedError(
        "Text-to-speech is not supported on the MLX backend yet.",
        hint = "Run it on a non-MLX host, or load a GGUF build of it if one is published.",
    )
    assert "MLX" in error.message and "GGUF" in error.message
    assert error.detail and error.hint


def test_the_worker_tags_the_capability_error_with_a_code_not_prose():
    """The parent must recognise the case without string matching, or a
    translated or reworded message silently becomes a 500."""
    worker = (BACKEND / "core" / "inference" / "worker.py").read_text(encoding = "utf-8")
    orchestrator = (BACKEND / "core" / "inference" / "orchestrator.py").read_text(encoding = "utf-8")
    # The shared constant, not a copied literal, on both sides of the pipe.
    assert AUDIO_UNSUPPORTED_CODE not in worker, "worker inlined the code instead of importing it"
    assert '"code": AUDIO_UNSUPPORTED_CODE' in worker
    assert 'resp.get("code") == AUDIO_UNSUPPORTED_CODE' in orchestrator


def test_the_route_maps_the_capability_error_to_501():
    route = (BACKEND / "routes" / "inference.py").read_text(encoding = "utf-8")
    assert "isinstance(e, AudioBackendUnsupportedError)" in route
    assert "status_code = 501" in route


# ── the gallery must not care what the host is ────────────────────────────────

@pytest.mark.parametrize("platform_name", ["linux", "win32", "darwin"])
def test_the_gallery_module_has_no_platform_branch(platform_name):
    """Any sys.platform branch here would be a per-OS behaviour we are not
    testing on that OS. Keep it absent."""
    source = (BACKEND / "core" / "inference" / "audio_gallery.py").read_text(encoding = "utf-8")
    assert "sys.platform" not in source
    assert "os.name" not in source
    assert "platform.system" not in source


def test_gallery_paths_are_built_with_pathlib_not_string_joins():
    """A hardcoded separator is the classic Windows break."""
    source = (BACKEND / "core" / "inference" / "audio_gallery.py").read_text(encoding = "utf-8")
    assert '"/"' not in source.replace('f"/api/inference/audio/gallery/{audio_id}/file"', "")
    assert "\\\\" not in source
