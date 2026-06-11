# Simulation suite 1: capability probing, flag gating, /props readback, and
# pydantic contracts. Designed to run on minimal venvs (pytest + pydantic +
# httpx) and on Linux/macOS/Windows: fake llama-server binaries are .bat on
# Windows and sh scripts elsewhere.

from __future__ import annotations

import http.server
import json
import os
import socket
import stat
import sys
import threading
import types as _types
from pathlib import Path

import pytest

_THIS = Path(__file__).resolve()
_BACKEND_DIR = os.environ.get("STUDIO_BACKEND_DIR")
if not _BACKEND_DIR:
    for cand in [_THIS.parents[2] / "unsloth_main" / "studio" / "backend",
                 _THIS.parents[1] / "studio" / "backend",
                 _THIS.parents[2] / "studio" / "backend"]:
        if cand.is_dir():
            _BACKEND_DIR = str(cand)
            break
assert _BACKEND_DIR, "set STUDIO_BACKEND_DIR"
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# Stub only what the venv lacks (mirrors the in-repo test pattern).
try:
    import loggers  # noqa: F401
except ImportError:
    _l = _types.ModuleType("loggers")
    _l.get_logger = lambda name: __import__("logging").getLogger(name)
    sys.modules.setdefault("loggers", _l)
try:
    import structlog  # noqa: F401
except ImportError:
    sys.modules.setdefault("structlog", _types.ModuleType("structlog"))

import httpx  # noqa: F401  (real httpx required for the /props sim)

from core.inference.llama_cpp import LlamaCppBackend


# ---------------------------------------------------------------------------
# Fake llama-server binaries (OS-portable)
# ---------------------------------------------------------------------------

_HELP_COMMON = """\
-h, --help                              print usage and exit
-m, --model FNAME                       model path
-c, --ctx-size N                        size of the prompt context (default: 4096)
-np, --parallel N                       number of parallel sequences to decode (default: 1)
--no-context-shift                      disables context shift
--jinja                                 use jinja template for chat
"""

_HELP_UNSLOTH_LATEST = _HELP_COMMON + """\
-kvu,  --kv-unified, -no-kvu, --no-kv-unified
                                        use single unified KV buffer shared across all sequences (default:
                                        enabled if number of slots is auto)
                                        (env: LLAMA_ARG_KV_UNIFIED)
-fit,  --fit [on|off]                   whether to adjust unset arguments to fit in device memory
-fitc, --fit-ctx N                      minimum ctx size that can be set by --fit option, default: 4096
--spec-type none,draft-simple,draft-mtp,ngram-mod,ngram-cache
                                        comma-separated list of types of speculative decoding
--spec-draft-n-max N                    max drafted tokens
--spec-ngram-mod-n-match N              ngram match
--spec-ngram-mod-n-min N                ngram min
--spec-ngram-mod-n-max N                ngram max
"""

_HELP_GGML_OLD = _HELP_COMMON + """\
--spec-type none,draft-simple
                                        speculative decoding types
--draft-max N                           max drafted tokens
--draft-min N                           min drafted tokens
--spec-ngram-size-n N                   ngram size
"""

_HELP_ROCM_MIN = _HELP_COMMON

_HELP_STUB_REMOVED = _HELP_COMMON + """\
--kv-unified                            argument has been removed, use slots auto
--fit-ctx N                             argument has been removed
"""

_VARIANTS = {
    "unsloth_latest": dict(help=_HELP_UNSLOTH_LATEST, kvu=True, fitc=True),
    "ggml_old": dict(help=_HELP_GGML_OLD, kvu=False, fitc=False),
    "rocm_min": dict(help=_HELP_ROCM_MIN, kvu=False, fitc=False),
    "stub_removed": dict(help=_HELP_STUB_REMOVED, kvu=False, fitc=False),
}


def _write_fake_binary(dirpath: Path, name: str, help_text: str | None,
                       exit_code: int = 0, hang: bool = False) -> str:
    dirpath.mkdir(parents=True, exist_ok=True)
    if help_text is not None:
        (dirpath / f"{name}_help.txt").write_text(help_text, encoding="utf-8")
    if os.name == "nt":
        p = dirpath / f"{name}.bat"
        lines = ["@echo off"]
        if hang:
            lines.append("ping -n 31 127.0.0.1 >nul")
        if help_text is not None:
            lines.append(f'type "%~dp0{name}_help.txt"')
        lines.append(f"exit /b {exit_code}")
        p.write_text("\r\n".join(lines), encoding="utf-8")
    else:
        p = dirpath / name
        lines = ["#!/bin/sh"]
        if hang:
            lines.append("sleep 30")
        if help_text is not None:
            lines.append(f'cat "$(dirname "$0")/{name}_help.txt"')
        lines.append(f"exit {exit_code}")
        p.write_text("\n".join(lines), encoding="utf-8")
        p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return str(p)


@pytest.fixture(scope="module")
def fake_bins(tmp_path_factory):
    d = tmp_path_factory.mktemp("fake_bins")
    bins = {n: _write_fake_binary(d, n, v["help"]) for n, v in _VARIANTS.items()}
    bins["broken"] = _write_fake_binary(d, "broken", None, exit_code=1)
    bins["hangs"] = _write_fake_binary(d, "hangs", _HELP_UNSLOTH_LATEST, hang=True)
    return bins


# ---------------------------------------------------------------------------
# Probe matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", list(_VARIANTS))
def test_probe_detects_flags_per_build(fake_bins, variant):
    info = LlamaCppBackend.probe_server_capabilities(binary=fake_bins[variant])
    exp = _VARIANTS[variant]
    assert info["found"] is True
    assert info["supports_kv_unified"] is exp["kvu"], variant
    assert info["supports_fit_ctx"] is exp["fitc"], variant


def test_probe_broken_binary_yields_no_flags(fake_bins):
    info = LlamaCppBackend.probe_server_capabilities(binary=fake_bins["broken"])
    assert info["supports_kv_unified"] is False
    assert info["supports_fit_ctx"] is False


def test_probe_hanging_binary_times_out_safely(fake_bins):
    info = LlamaCppBackend.probe_server_capabilities(binary=fake_bins["hangs"])
    assert info["found"] is True
    assert info["supports_kv_unified"] is False  # timed out, conservative


def test_probe_missing_binary():
    info = LlamaCppBackend.probe_server_capabilities(binary="/nope/llama-server")
    assert info["found"] is False
    assert info["supports_kv_unified"] is False
    assert info["supports_fit_ctx"] is False


# ---------------------------------------------------------------------------
# Flag gating cartesian: builds x parallel x fit x requested ctx
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", list(_VARIANTS) + ["broken"])
@pytest.mark.parametrize("n_parallel", [1, 2, 4])
@pytest.mark.parametrize("use_fit", [False, True])
@pytest.mark.parametrize("requested", [0, 98304])
def test_flag_gating_cartesian(fake_bins, variant, n_parallel, use_fit, requested):
    caps = LlamaCppBackend.probe_server_capabilities(binary=fake_bins[variant])
    effective = requested if requested > 0 else 262144
    flags = LlamaCppBackend._ctx_integrity_flags(
        n_parallel, use_fit, requested, effective, caps
    )
    kvu_expected = n_parallel > 1 and caps["supports_kv_unified"]
    fitc_expected = use_fit and requested > 0 and caps["supports_fit_ctx"]
    assert ("--kv-unified" in flags) == kvu_expected
    assert ("--fit-ctx" in flags) == fitc_expected
    if fitc_expected:
        assert flags[flags.index("--fit-ctx") + 1] == str(effective)
    # Never emit anything else.
    leftover = [f for f in flags if f not in ("--kv-unified", "--fit-ctx", str(effective))]
    assert leftover == []


# ---------------------------------------------------------------------------
# /props readback against a real local HTTP server
# ---------------------------------------------------------------------------


class _PropsHandler(http.server.BaseHTTPRequestHandler):
    payload: bytes = b"{}"
    status: int = 200

    def do_GET(self):
        if self.path == "/props":
            self.send_response(self.status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(self.payload)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *a):
        pass


def _serve(payload: bytes, status: int = 200):
    _PropsHandler.payload = payload
    _PropsHandler.status = status
    srv = http.server.HTTPServer(("127.0.0.1", 0), _PropsHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv


def _backend_on(port: int, effective=98304):
    inst = LlamaCppBackend.__new__(LlamaCppBackend)
    inst._port = port
    inst._effective_context_length = effective
    inst._context_length = 262144
    return inst


def test_readback_real_http_shrunk():
    srv = _serve(json.dumps({"default_generation_settings": {"n_ctx": 24576}}).encode())
    try:
        b = _backend_on(srv.server_address[1])
        b._reconcile_effective_ctx_with_server()
        assert b._effective_context_length == 24576
    finally:
        srv.shutdown()


def test_readback_real_http_matching():
    srv = _serve(json.dumps({"default_generation_settings": {"n_ctx": 98304}}).encode())
    try:
        b = _backend_on(srv.server_address[1])
        b._reconcile_effective_ctx_with_server()
        assert b._effective_context_length == 98304
    finally:
        srv.shutdown()


def test_readback_real_http_malformed_json_is_safe():
    srv = _serve(b"this is not json{{{")
    try:
        b = _backend_on(srv.server_address[1])
        b._reconcile_effective_ctx_with_server()
        assert b._effective_context_length == 98304
    finally:
        srv.shutdown()


def test_readback_real_http_404_is_safe():
    srv = _serve(b"{}", status=404)
    try:
        b = _backend_on(srv.server_address[1])
        b._reconcile_effective_ctx_with_server()
        assert b._effective_context_length == 98304
    finally:
        srv.shutdown()


def test_readback_no_server_is_safe():
    # Grab a free port and close it so nothing listens there.
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    b = _backend_on(port)
    b._reconcile_effective_ctx_with_server()
    assert b._effective_context_length == 98304


# ---------------------------------------------------------------------------
# Pydantic contracts: ChatMessage and context_overflow field
# ---------------------------------------------------------------------------

from models.inference import ChatMessage, ChatCompletionRequest  # noqa: E402


@pytest.mark.parametrize("content,expected", [("", ""), (None, ""), ([], ""), ("ok", "ok")])
def test_tool_message_content_matrix(content, expected):
    msg = ChatMessage(role="tool", content=content, tool_call_id="c1")
    assert msg.content == expected


def test_user_and_system_still_require_content():
    with pytest.raises(ValueError):
        ChatMessage(role="user", content=None)
    with pytest.raises(ValueError):
        ChatMessage(role="system", content=None)


def test_assistant_sentinel_collapse_unchanged():
    assert ChatMessage(role="assistant", content="").content is None
    assert ChatMessage(role="assistant", content=[]).content is None


@pytest.mark.parametrize("value", [None, "error", "truncate_middle"])
def test_context_overflow_accepts_valid(value):
    req = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hi"}], context_overflow=value
    )
    assert req.context_overflow == value


def test_context_overflow_rejects_invalid():
    with pytest.raises(ValueError):
        ChatCompletionRequest(
            messages=[{"role": "user", "content": "hi"}], context_overflow="bogus"
        )


def test_tool_history_round_trip_with_empty_results():
    """An OpenCode-style history with empty tool outputs must validate."""
    req = ChatCompletionRequest(
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "c1", "type": "function",
                "function": {"name": "bash", "arguments": "{\"command\":\"mkdir x\"}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": ""},
            {"role": "assistant", "content": "done"},
        ],
        tools=[{"type": "function", "function": {"name": "bash", "parameters": {}}}],
    )
    assert req.messages[3].content == ""
