# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression: tokenizer_config.json is read as UTF-8, not as the operator's locale.

`_detect_audio_from_tokenizer` reads a cached tokenizer_config.json to look for
audio tokens. It used to call `read_text()` with no encoding, which resolves to
`locale.getencoding()`: UTF-8 on the Linux and macOS runners, cp1252 on a stock
Windows install. DeepSeek and Qwen ship chat templates carrying U+FF5C and
U+2581, so on Windows that read raised UnicodeDecodeError -- and because the call
sits inside a broad `except Exception: logger.debug(...)`, detection silently
returned None instead of the real audio type, with nothing in the logs above
debug level.

`locale.getpreferredencoding` cannot be monkeypatched to test this: CPython 3.11+
resolves the default inside `io.open` via `_Py_GetLocaleEncoding`, never
consulting the Python-level `locale` module. Shimming `io.open` (which
`pathlib.Path.open` looks up by module attribute) is what actually reproduces a
Windows default.
"""

from __future__ import annotations

import builtins
import io
import json

import pytest

from utils.models import model_config


# U+FF5C FULLWIDTH VERTICAL LINE and U+2581 LOWER ONE EIGHTH BLOCK: the two
# characters in a DeepSeek chat template that cp1252 cannot decode. U+2581
# encodes to e2 96 81, and 0x81 is undefined in cp1252.
DEEPSEEK_TEMPLATE = "{% if x %}｜User｜{% endif %}▁begin▁of▁sentence"


@pytest.fixture
def cp1252_default(monkeypatch):
    """Make every unencoded text read decode as cp1252, as on Windows."""
    real_open = io.open

    def shim(file, mode = "r", buffering = -1, encoding = None, *args, **kwargs):
        if "b" not in mode and encoding in (None, "locale"):
            encoding = "cp1252"
        return real_open(file, mode, buffering, encoding, *args, **kwargs)

    monkeypatch.setattr(io, "open", shim)
    monkeypatch.setattr(builtins, "open", shim)
    return shim


def _write_cache(root, tokens: list[str]):
    """Lay out an HF cache dir holding one snapshot with a non-ASCII template."""
    snapshot = root / "snapshots" / "deadbeef"
    snapshot.mkdir(parents = True)
    (snapshot / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "chat_template": DEEPSEEK_TEMPLATE,
                "added_tokens_decoder": {
                    str(i): {"content": t} for i, t in enumerate(tokens)
                },
            },
            ensure_ascii = False,
        ),
        encoding = "utf-8",
    )
    return root


def test_cp1252_fixture_actually_forces_a_windows_default(cp1252_default, tmp_path):
    """Guard the guard: if the shim stops biting, the test below passes vacuously."""
    target = tmp_path / "template.json"
    target.write_text(DEEPSEEK_TEMPLATE, encoding = "utf-8")
    with pytest.raises(UnicodeDecodeError):
        target.read_text()


def test_detects_audio_tokens_under_a_cp1252_default(cp1252_default, tmp_path, monkeypatch):
    """The whole point: a non-ASCII chat template must not hide the audio tokens."""
    cache = _write_cache(tmp_path / "cache", ["<|startoftranscript|>", "<|notimestamps|>"])
    monkeypatch.setattr(model_config, "get_cache_path", lambda name: cache)

    audio_type, definitive = model_config._detect_audio_from_tokenizer(
        "unsloth/whisper-large-v3", local_files_only = True
    )

    assert audio_type == "whisper"
    assert definitive is True


def test_non_audio_model_is_definitively_none_under_cp1252(
    cp1252_default, tmp_path, monkeypatch
):
    """A clean read with no audio tokens is a definitive None, not a read failure.

    Before the fix the UnicodeDecodeError was swallowed and `read_any` stayed
    False, so this returned the same (None, False) as a network error and the
    caller kept retrying.
    """
    cache = _write_cache(tmp_path / "cache", ["<bos>", "<eos>"])
    monkeypatch.setattr(model_config, "get_cache_path", lambda name: cache)

    audio_type, definitive = model_config._detect_audio_from_tokenizer(
        "unsloth/gemma-3-4b-it", local_files_only = True
    )

    assert audio_type is None
    assert definitive is True
