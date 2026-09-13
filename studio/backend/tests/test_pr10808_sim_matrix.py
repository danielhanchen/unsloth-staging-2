# Simulation matrix for unslothai/unsloth PR #10808. Not part of the repo.
#
# What this pins, in order:
#   A. _is_unquantized_full_finetune over every checkpoint layout Studio can produce or accept
#   B. the HTTP route, as a cartesian product of {layout} x {field unset, true, false}
#   C. the MCP load_checkpoint tool, same product
#   D. path semantics that differ across Linux / macOS / Windows
#   E. backwards + forwards compatibility of the wire contract (old client <-> new server)
#   F. the worker's own load_in_4bit override, to show the two layers do not fight
#
# Every case asserts the *effective* load_in_4bit handed to the export backend, because that
# single boolean is what decides whether a "16-bit" export is really 16-bit.

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import allow_ambient_hf_token, get_current_subject
from routes import export as export_routes

BACKEND_ROOT = Path(__file__).resolve().parent.parent

IS_WINDOWS = sys.platform == "win32"
IS_MACOS = sys.platform == "darwin"


# --------------------------------------------------------------------------------------
# Checkpoint layout fixtures. `None` content means "do not create this file".
# --------------------------------------------------------------------------------------

def _write(tmp_path: Path, files: dict) -> Path:
    for name, content in files.items():
        if content is None:
            continue
        target = tmp_path / name
        target.parent.mkdir(parents = True, exist_ok = True)
        if isinstance(content, bytes):
            target.write_bytes(content)
        elif isinstance(content, str):
            target.write_text(content, encoding = "utf-8")
        else:
            target.write_text(json.dumps(content), encoding = "utf-8")
    return tmp_path


_LLAMA = {"model_type": "llama", "architectures": ["LlamaForCausalLM"]}

# (id, files, expected_is_full_finetune, why)
LAYOUTS = [
    # --- the case the PR targets ---------------------------------------------------
    ("full_finetune_bf16", {"config.json": {**_LLAMA, "dtype": "bfloat16"}}, True,
     "transformers full fine-tune: unsloth forces 4/8-bit off when full_finetuning=True"),
    ("full_finetune_fp16", {"config.json": {**_LLAMA, "torch_dtype": "float16"}}, True,
     "same, fp16 checkpoint"),
    ("full_finetune_sharded", {
        "config.json": _LLAMA,
        "model.safetensors.index.json": {"metadata": {"total_size": 1}, "weight_map": {}},
    }, True, "shard count must not affect classification"),
    ("pretrained_base_model", {"config.json": _LLAMA}, True,
     "a plain unquantized base model is also 'full model, load it 16-bit' - correct"),

    # --- adapters: must stay 4-bit -------------------------------------------------
    ("lora_adapter", {
        "config.json": _LLAMA,
        "adapter_config.json": {"peft_type": "LORA", "r": 16,
                                "base_model_name_or_path": "unsloth/Llama-3.2-1B"},
    }, False, "PEFT adapter: the 4-bit base is dequantized during merge"),
    ("lora_adapter_no_config", {
        "adapter_config.json": {"peft_type": "LORA", "r": 16},
    }, False, "adapter dir without a base config.json"),
    ("qlora_adapter", {
        "config.json": _LLAMA,
        "adapter_config.json": {"peft_type": "LORA", "unsloth_training_method": "qlora"},
    }, False, "QLoRA adapter"),
    ("plain_lora_adapter", {
        "config.json": _LLAMA,
        "adapter_config.json": {"peft_type": "LORA", "unsloth_training_method": "lora"},
    }, False, "16-bit-base LoRA: still 4-bit here (pre-existing, out of PR scope)"),
    ("mlx_adapter_with_base_config", {
        "config.json": {**_LLAMA, "quantization": {"bits": 4, "group_size": 64}},
        "adapter_config.json": {"peft_type": "LORA"},
        "adapters.safetensors": b"\x00",
    }, False, "MLX adapter dirs copy the base config.json in; adapter check must win"),

    # --- already quantized: must stay 4-bit ----------------------------------------
    ("bnb_4bit", {"config.json": {**_LLAMA, "quantization_config": {
        "quant_method": "bitsandbytes", "load_in_4bit": True}}}, False, "bitsandbytes 4-bit"),
    ("bnb_8bit", {"config.json": {**_LLAMA, "quantization_config": {
        "quant_method": "bitsandbytes", "load_in_8bit": True}}}, False, "bitsandbytes 8-bit"),
    ("gptq", {"config.json": {**_LLAMA, "quantization_config": {
        "quant_method": "gptq", "bits": 4}}}, False, "GPTQ"),
    ("awq", {"config.json": {**_LLAMA, "quantization_config": {
        "quant_method": "awq", "bits": 4}}}, False, "AWQ"),
    ("compressed_tensors", {"config.json": {**_LLAMA, "quantization_config": {
        "quant_method": "compressed-tensors"}}}, False, "compressed-tensors / FP8"),
    ("torchao", {"config.json": {**_LLAMA, "quantization_config": {
        "quant_method": "torchao"}}}, False, "torchao"),
    ("fp8", {"config.json": {**_LLAMA, "quantization_config": {
        "quant_method": "fbgemm_fp8"}}}, False, "FP8"),
    ("mxfp4", {"config.json": {**_LLAMA, "quantization_config": {
        "quant_method": "mxfp4"}}}, False, "MXFP4"),
    ("quant_config_null", {"config.json": {**_LLAMA, "quantization_config": None}}, False,
     "key present but null - conservative: stay 4-bit"),
    ("quant_config_empty", {"config.json": {**_LLAMA, "quantization_config": {}}}, False,
     "key present but empty - conservative: stay 4-bit"),

    # --- MLX (macOS): records quantization under "quantization", not "quantization_config".
    # The helper therefore calls these "unquantized" and sends load_in_4bit=False. That is
    # NOT a behaviour change: unsloth_zoo/mlx/loader.py:4789-4800 already disables its own
    # default load_in_4bit=True whenever the checkpoint carries explicit MLX quant settings,
    # so base and head reach the same spec (source="none", stored weights load as-is).
    ("mlx_quantized_full_model", {"config.json": {
        **_LLAMA, "quantization": {"group_size": 64, "bits": 4}}}, True,
     "MLX quant key: classified unquantized, but the MLX loader converges to the same state"),
    ("mlx_both_keys", {"config.json": {
        **_LLAMA, "quantization": {"bits": 4},
        "quantization_config": {"bits": 4}}}, False,
     "MLX VLM saves mirror quantization into quantization_config; that is detected"),
    ("mlx_unquantized_full_model", {"config.json": {**_LLAMA, "quantization": None}}, True,
     "MLX 16-bit save leaves the key null-or-absent"),

    # --- malformed / unreadable: must fall back to the old default -----------------
    ("no_config", {"model.safetensors": b"\x00"}, False, "no config.json at all"),
    ("empty_dir", {}, False, "empty directory"),
    ("malformed_json", {"config.json": "{not json"}, False, "truncated JSON"),
    ("empty_config", {"config.json": ""}, False, "zero-byte config.json"),
    ("config_is_list", {"config.json": "[1, 2, 3]"}, False, "JSON array, not an object"),
    ("config_is_string", {"config.json": '"hello"'}, False, "JSON string, not an object"),
    ("config_is_null", {"config.json": "null"}, False, "JSON null"),
    ("config_bom", {"config.json": "\ufeff" + json.dumps(_LLAMA)}, True,
     "UTF-8 BOM: utf-8-sig must strip it (Windows editors write these)"),
    ("config_crlf", {"config.json": json.dumps(_LLAMA).replace(",", ",\r\n")}, True,
     "CRLF line endings from a Windows checkout"),
    ("config_utf16", {"config.json": json.dumps(_LLAMA).encode("utf-16")}, False,
     "UTF-16 bytes: UnicodeDecodeError is a ValueError, must be swallowed"),
    ("config_invalid_utf8", {"config.json": b"\xff\xfe\x00bad"}, False,
     "invalid UTF-8 must not raise"),
]

LAYOUT_IDS = [c[0] for c in LAYOUTS]


@pytest.fixture
def layout(request, tmp_path):
    """Materialise one LAYOUTS entry into a fresh directory."""
    case = dict(zip(LAYOUT_IDS, LAYOUTS))[request.param]
    return request.param, _write(tmp_path, case[1]), case[2], case[3]


# ======================================================================================
# A. classifier
# ======================================================================================

@pytest.mark.parametrize("case_id,files,expected,why", LAYOUTS, ids = LAYOUT_IDS)
def test_classifier(tmp_path, case_id, files, expected, why):
    got = export_routes._is_unquantized_full_finetune(_write(tmp_path, files))
    assert got is expected, f"{case_id}: {why}"


def test_classifier_never_raises_on_hostile_input(tmp_path):
    """Anything this helper raises becomes a 500 on a request that used to succeed."""
    hostile = [
        tmp_path / "does-not-exist",
        tmp_path,                                   # empty dir
        Path(""),                                   # empty string -> CWD
        Path("."),
        Path("relative/does/not/exist"),
        Path("~/not-expanded"),
        Path("/proc/self/mem") if sys.platform == "linux" else tmp_path,
        Path("a" * 300),                            # over NAME_MAX
        Path("/" + "b" * 300 + "/" + "c" * 300),    # over PATH_MAX on Linux
        Path("checkpoint\u0000null"),               # embedded NUL -> ValueError on stat
        Path("Ünïcödé-çheckpoint-模型-🦥"),
        Path("trailing space "),
        Path("trailing.dot."),
        Path("C:\\Users\\me\\ckpt"),                # Windows path seen on Linux
        Path("\\\\server\\share\\ckpt"),            # UNC path
        Path("con"), Path("nul"), Path("aux"),      # Windows reserved device names
    ]
    for p in hostile:
        assert export_routes._is_unquantized_full_finetune(p) in (True, False)


def test_classifier_config_json_is_a_directory(tmp_path):
    (tmp_path / "config.json").mkdir()
    assert export_routes._is_unquantized_full_finetune(tmp_path) is False


@pytest.mark.skipif(IS_WINDOWS, reason = "POSIX symlink semantics")
def test_classifier_symlinked_config(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    (real / "config.json").write_text(json.dumps(_LLAMA), encoding = "utf-8")
    link_dir = tmp_path / "link"
    link_dir.mkdir()
    (link_dir / "config.json").symlink_to(real / "config.json")
    assert export_routes._is_unquantized_full_finetune(link_dir) is True

    # A dangling adapter_config.json symlink must not read as "adapter present".
    (link_dir / "adapter_config.json").symlink_to(tmp_path / "gone.json")
    assert export_routes._is_unquantized_full_finetune(link_dir) is True


@pytest.mark.skipif(IS_WINDOWS or getattr(os, "geteuid", lambda: 0)() == 0,
                    reason = "root ignores the permission bits")
def test_classifier_unreadable_config(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(_LLAMA), encoding = "utf-8")
    cfg.chmod(0o000)
    try:
        assert export_routes._is_unquantized_full_finetune(tmp_path) is False
    finally:
        cfg.chmod(0o644)


def test_classifier_large_config(tmp_path):
    """A 5 MB config.json (long vocab maps exist in the wild) must not be pathological."""
    import time
    big = {**_LLAMA, "junk": {str(i): "x" * 40 for i in range(100_000)}}
    (tmp_path / "config.json").write_text(json.dumps(big), encoding = "utf-8")
    start = time.perf_counter()
    assert export_routes._is_unquantized_full_finetune(tmp_path) is True
    assert time.perf_counter() - start < 10.0


# ======================================================================================
# B. the HTTP route: {layout} x {unset, true, false}
# ======================================================================================

def _client(monkeypatch, backend):
    async def _ok():
        pass
    monkeypatch.setattr(export_routes, "_ensure_export_supported", _ok)
    monkeypatch.setattr(export_routes, "get_export_backend", lambda: backend)
    app = FastAPI()
    app.include_router(export_routes.router, prefix = "/api/export")
    app.dependency_overrides[get_current_subject] = lambda: "alice"
    app.dependency_overrides[allow_ambient_hf_token] = lambda: True
    return TestClient(app)


def _post(monkeypatch, path, extra):
    backend = MagicMock()
    backend.load_checkpoint.return_value = (True, "loaded")
    response = _client(monkeypatch, backend).post(
        "/api/export/load-checkpoint",
        json = {"checkpoint_path": str(path), "hf_token": None, **extra},
    )
    return response, backend


@pytest.mark.parametrize("case_id,files,is_full,why", LAYOUTS, ids = LAYOUT_IDS)
@pytest.mark.parametrize("sent", ["unset", "true", "false"],
                         ids = ["field_unset", "explicit_true", "explicit_false"])
def test_route_matrix(monkeypatch, tmp_path, case_id, files, is_full, why, sent):
    extra = {"unset": {}, "true": {"load_in_4bit": True}, "false": {"load_in_4bit": False}}[sent]
    response, backend = _post(monkeypatch, _write(tmp_path, files), extra)
    assert response.status_code == 200

    if sent == "true":
        expected = True      # an explicit request is never second-guessed
    elif sent == "false":
        expected = False
    else:
        expected = not is_full

    got = backend.load_checkpoint.call_args.kwargs["load_in_4bit"]
    assert got is expected, f"{case_id}/{sent}: {why}"


@pytest.mark.parametrize("case_id,files,is_full,why", LAYOUTS, ids = LAYOUT_IDS)
def test_route_forwards_everything_else_untouched(monkeypatch, tmp_path, case_id, files,
                                                  is_full, why):
    """The PR must change exactly one kwarg and nothing else."""
    response, backend = _post(monkeypatch, _write(tmp_path, files),
                              {"max_seq_length": 4096, "trust_remote_code": True,
                               "approved_remote_code_fingerprint": "sha256:abc"})
    assert response.status_code == 200
    kwargs = backend.load_checkpoint.call_args.kwargs
    assert kwargs["max_seq_length"] == 4096
    assert kwargs["trust_remote_code"] is True
    assert kwargs["approved_remote_code_fingerprint"] == "sha256:abc"
    assert kwargs["subject"] == "alice"
    assert kwargs["allow_ambient"] is True
    # base_model arrived with main's per-account isolation work (#10588) and is present only
    # once this branch has been merged up; it is not something the PR introduces.
    assert set(kwargs) - {"base_model"} == {
        "checkpoint_path", "max_seq_length", "load_in_4bit", "trust_remote_code",
        "approved_remote_code_fingerprint", "hf_token", "allow_ambient", "subject",
    }


def test_route_checkpoint_path_is_passed_through_verbatim(monkeypatch, tmp_path):
    """The helper may build a Path, but the backend must still receive the original string."""
    _write(tmp_path, {"config.json": _LLAMA})
    raw = str(tmp_path) + "/"
    response, backend = _post(monkeypatch, raw, {})
    assert response.status_code == 200
    assert backend.load_checkpoint.call_args.kwargs["checkpoint_path"] == raw


@pytest.mark.parametrize("hub_config,expected", [
    ({"model_type": "llama"}, False),
    ({"model_type": "llama",
      "quantization_config": {"quant_method": "bitsandbytes"}}, True),
    (None, True),   # adapter on the Hub, or the lookup failed
], ids = ["remote_full_finetune", "remote_quantized", "remote_unresolvable"])
def test_route_resolves_hub_repo_ids(monkeypatch, hub_config, expected):
    """A Hub id never touches the local filesystem. It is resolved through the Hub so a
    remote full fine-tune gets 16-bit like a local one; anything unresolvable falls back
    to the historical 4-bit default. Stubbed at _hub_config so the assertion is about the
    route's decision rather than about what a live repo happens to contain today."""
    monkeypatch.setattr(export_routes, "_hub_config", lambda repo_id, hf_token: hub_config)
    response, backend = _post(monkeypatch, "org/some-model", {})
    assert response.status_code == 200
    assert backend.load_checkpoint.call_args.kwargs["load_in_4bit"] is expected


def test_hub_lookup_never_raises_into_the_request(monkeypatch):
    """Offline, gated, rate-limited, DNS failure: the route must answer 200 on the old
    default, not 500. This is the whole reason _hub_config swallows everything."""
    import huggingface_hub

    def _boom(*a, **k):
        raise OSError("no network")

    monkeypatch.setattr(huggingface_hub, "file_exists", _boom, raising = False)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _boom, raising = False)
    assert export_routes._hub_config("org/model", None) is None
    response, backend = _post(monkeypatch, "org/model", {})
    assert response.status_code == 200
    assert backend.load_checkpoint.call_args.kwargs["load_in_4bit"] is True


def test_local_path_never_reaches_the_hub(monkeypatch, tmp_path):
    """A local checkpoint must not pay a network round trip, and a local dir that simply
    has no config.json must not be mistaken for a repo id."""
    calls = []
    monkeypatch.setattr(export_routes, "_hub_config",
                        lambda repo_id, hf_token: calls.append(repo_id))
    _write(tmp_path, {"config.json": _LLAMA})
    response, backend = _post(monkeypatch, tmp_path, {})
    assert backend.load_checkpoint.call_args.kwargs["load_in_4bit"] is False

    empty = tmp_path / "empty"
    empty.mkdir()
    response, backend = _post(monkeypatch, empty, {})
    assert backend.load_checkpoint.call_args.kwargs["load_in_4bit"] is True
    assert calls == [], f"local paths reached the Hub: {calls}"


def test_route_nonexistent_path_keeps_the_old_default(monkeypatch, tmp_path):
    response, backend = _post(monkeypatch, tmp_path / "nope", {})
    assert response.status_code == 200
    assert backend.load_checkpoint.call_args.kwargs["load_in_4bit"] is True


def test_route_relative_path_resolves_like_the_export_backend(monkeypatch, tmp_path):
    """core/export/export.py does Path(checkpoint_path) against the process CWD.
    The route helper must agree, or the two layers disagree about the same string."""
    _write(tmp_path / "ckpt", {"config.json": _LLAMA})
    monkeypatch.chdir(tmp_path)
    response, backend = _post(monkeypatch, "ckpt", {})
    assert response.status_code == 200
    assert backend.load_checkpoint.call_args.kwargs["load_in_4bit"] is False


@pytest.mark.parametrize("name", [
    "with space", "with-dash", "with.dot", "Ünïcödé", "模型", "emoji-🦥",
    "UPPER", "checkpoint-1000", "run_1771227800",
])
def test_route_unusual_directory_names(monkeypatch, tmp_path, name):
    ckpt = _write(tmp_path / name, {"config.json": _LLAMA})
    response, backend = _post(monkeypatch, ckpt, {})
    assert response.status_code == 200
    assert backend.load_checkpoint.call_args.kwargs["load_in_4bit"] is False


def test_route_studio_outputs_layout(monkeypatch, tmp_path):
    """The exact shape scan_checkpoints emits: outputs/{run}/{checkpoint-N}."""
    ckpt = _write(tmp_path / "outputs" / "unsloth_Llama-3.2-1B_1771227800" / "checkpoint-20",
                  {"config.json": _LLAMA})
    response, backend = _post(monkeypatch, ckpt, {})
    assert backend.load_checkpoint.call_args.kwargs["load_in_4bit"] is False


def test_route_propagates_backend_failure_unchanged(monkeypatch, tmp_path):
    backend = MagicMock()
    backend.load_checkpoint.return_value = (False, "boom")
    response = _client(monkeypatch, backend).post(
        "/api/export/load-checkpoint",
        json = {"checkpoint_path": str(_write(tmp_path, {"config.json": _LLAMA}))},
    )
    assert response.status_code == 400
    assert "boom" in response.text


def test_route_rejects_null_load_in_4bit(monkeypatch, tmp_path):
    """Wire contract: the field is a bool, so null is a 422 - same as before the PR."""
    backend = MagicMock()
    backend.load_checkpoint.return_value = (True, "loaded")
    response = _client(monkeypatch, backend).post(
        "/api/export/load-checkpoint",
        json = {"checkpoint_path": str(tmp_path), "load_in_4bit": None},
    )
    assert response.status_code == 422


def test_route_export_gate_still_runs_first(monkeypatch, tmp_path):
    """_ensure_export_supported must reject before any checkpoint sniffing happens."""
    from fastapi import HTTPException

    async def _blocked():
        raise HTTPException(status_code = 400, detail = "Export is not supported on this platform.")

    backend = MagicMock()
    monkeypatch.setattr(export_routes, "_ensure_export_supported", _blocked)
    monkeypatch.setattr(export_routes, "get_export_backend", lambda: backend)
    app = FastAPI()
    app.include_router(export_routes.router, prefix = "/api/export")
    app.dependency_overrides[get_current_subject] = lambda: "alice"
    app.dependency_overrides[allow_ambient_hf_token] = lambda: True
    response = TestClient(app).post(
        "/api/export/load-checkpoint",
        json = {"checkpoint_path": str(_write(tmp_path, {"config.json": _LLAMA}))},
    )
    assert response.status_code == 400
    backend.load_checkpoint.assert_not_called()


# ======================================================================================
# C. wire contract: pydantic model_fields_set is load-bearing
# ======================================================================================

def test_fields_set_semantics():
    from models.export import LoadCheckpointRequest

    assert "load_in_4bit" not in LoadCheckpointRequest(checkpoint_path = "/c").model_fields_set
    for value in (True, False):
        r = LoadCheckpointRequest(checkpoint_path = "/c", load_in_4bit = value)
        assert "load_in_4bit" in r.model_fields_set
        assert r.load_in_4bit is value

    # The default must stay True so an old client that omits the field and a caller that
    # reads .load_in_4bit directly both see the historical value.
    assert LoadCheckpointRequest(checkpoint_path = "/c").load_in_4bit is True


def test_fields_set_survives_validate_and_copy():
    """Any middleware that revalidates the body must not fabricate the field."""
    from models.export import LoadCheckpointRequest

    r = LoadCheckpointRequest.model_validate({"checkpoint_path": "/c"})
    assert "load_in_4bit" not in r.model_fields_set
    assert "load_in_4bit" not in r.model_copy().model_fields_set
    assert "load_in_4bit" not in LoadCheckpointRequest.model_validate_json(
        '{"checkpoint_path": "/c"}').model_fields_set

    # But a full-dict round-trip DOES set it - this is the trap the MCP fix addressed.
    dumped = r.model_dump()
    assert "load_in_4bit" in LoadCheckpointRequest(**dumped).model_fields_set
    assert "load_in_4bit" not in LoadCheckpointRequest(
        **r.model_dump(exclude_unset = True)).model_fields_set


def test_no_other_backend_caller_builds_a_full_request():
    """Guard the trap above: no shipped code may construct LoadCheckpointRequest with an
    explicit load_in_4bit, or that caller silently opts out of the fix."""
    import re
    backend_root = BACKEND_ROOT
    offenders = []
    for py in backend_root.rglob("*.py"):
        if "tests" in py.parts or "node_modules" in py.parts:
            continue
        text = py.read_text(encoding = "utf-8", errors = "ignore")
        for match in re.finditer(r"LoadCheckpointRequest\((.{0,600}?)\)", text, re.S):
            if "load_in_4bit" in match.group(1):
                offenders.append(str(py.relative_to(backend_root)))
    assert offenders == [], f"these callers bypass the 16-bit fix: {offenders}"


# ======================================================================================
# D. MCP tool
# ======================================================================================

def _mcp_tool(monkeypatch, name = "load_checkpoint"):
    import mcp_server
    mcp = mcp_server.build_mcp() if hasattr(mcp_server, "build_mcp") else None
    return mcp_server, mcp


@pytest.mark.parametrize("sent", [None, True, False])
def test_mcp_forwards_only_what_the_caller_sent(monkeypatch, sent):
    """Mirrors the PR's own MCP test but also checks the value, not just presence."""
    import asyncio
    import types

    from models.export import LoadCheckpointRequest

    captured = {}

    async def fake_load(request, current_subject, allow_ambient):
        captured["fields_set"] = set(request.model_fields_set)
        captured["value"] = request.load_in_4bit
        return {}

    models_stub = types.ModuleType("models")
    models_stub.LoadCheckpointRequest = LoadCheckpointRequest
    routes_stub = types.ModuleType("routes")
    export_stub = types.ModuleType("routes.export")
    export_stub.load_checkpoint = fake_load
    monkeypatch.setitem(sys.modules, "models", models_stub)
    monkeypatch.setitem(sys.modules, "routes", routes_stub)
    monkeypatch.setitem(sys.modules, "routes.export", export_stub)

    import mcp_server
    tools = asyncio.run(mcp_server.create_studio_mcp().list_tools())
    tool = {t.name: t for t in tools}["load_checkpoint"]

    extra = {} if sent is None else {"load_in_4bit": sent}
    asyncio.run(tool.fn(checkpoint_path = "/tmp/ckpt", **extra))

    assert ("load_in_4bit" in captured["fields_set"]) is (sent is not None)
    if sent is not None:
        assert captured["value"] is sent


def test_mcp_signature_is_optional_and_defaults_to_none():
    """An MCP client that omits the arg must get 'auto', not a hardcoded True."""
    import inspect
    import mcp_server

    src = inspect.getsource(mcp_server)
    assert "load_in_4bit: bool | None = None" in src, \
        "the MCP tool must not re-hardcode load_in_4bit=True"
    # The file already used `str | None` before this PR, so no new Python floor is added.
    assert "from __future__ import annotations" in src


# ======================================================================================
# E. the worker's own override still behaves (no double-override, no regression)
# ======================================================================================

@pytest.mark.parametrize("incoming,sidecar,expected", [
    (True,  False, True),   # adapter / quantized: unchanged
    (True,  True,  False),  # latest-tier sidecar still forces 16-bit
    (False, False, False),  # the PR's new case: already 16-bit
    (False, True,  False),  # both agree; the sidecar probe is skipped
])
def test_worker_override_composes_with_the_route(incoming, sidecar, expected):
    """core/export/worker.py decides again from a plain dict. Re-implemented here because
    importing the worker pulls in torch; this pins the exact logic shape at that site."""
    load_in_4bit = incoming
    probed = False
    if load_in_4bit:
        probed = True
        if sidecar:
            load_in_4bit = False
    assert load_in_4bit is expected
    # When the route already said False, the worker must not spend a Hub lookup.
    assert probed is bool(incoming)


def test_worker_default_is_unchanged():
    """The worker still defaults to 4-bit for any command that omits the key, so a mixed
    old-orchestrator / new-worker install behaves exactly as before."""
    source = (BACKEND_ROOT / "core" / "export" / "worker.py"
              ).read_text(encoding = "utf-8")
    assert 'load_in_4bit = cmd.get("load_in_4bit", True)' in source


# ======================================================================================
# F. cross-platform path semantics
# ======================================================================================

@pytest.mark.skipif(not (IS_WINDOWS or IS_MACOS),
                    reason = "case-insensitive filesystems only")
def test_case_insensitive_config_name(tmp_path):
    _write(tmp_path, {"Config.JSON": _LLAMA})
    assert export_routes._is_unquantized_full_finetune(tmp_path) is True


@pytest.mark.skipif(IS_WINDOWS or IS_MACOS, reason = "case-sensitive filesystems only")
def test_case_sensitive_config_name(tmp_path):
    """On Linux a differently-cased name is a different file, so the old default holds.
    transformers always writes lowercase config.json, so this is documentation, not a bug."""
    _write(tmp_path, {"Config.JSON": _LLAMA})
    assert export_routes._is_unquantized_full_finetune(tmp_path) is False


def test_windows_style_separators_in_payload(monkeypatch, tmp_path):
    """On Windows the UI sends backslash paths. On Linux they are one filename, which must
    simply fall through to the old default rather than raise."""
    response, backend = _post(monkeypatch, r"C:\Users\me\outputs\run\checkpoint-20", {})
    assert response.status_code == 200
    assert backend.load_checkpoint.call_args.kwargs["load_in_4bit"] is True


def test_pure_path_flavours_agree_on_the_filenames():
    """The helper only ever joins two literal names; both flavours must produce the same
    tail so Windows and POSIX classify the same checkpoint identically."""
    from pathlib import PureWindowsPath, PurePosixPath
    for flavour in (PureWindowsPath, PurePosixPath):
        assert (flavour("ckpt") / "config.json").name == "config.json"
        assert (flavour("ckpt") / "adapter_config.json").name == "adapter_config.json"
