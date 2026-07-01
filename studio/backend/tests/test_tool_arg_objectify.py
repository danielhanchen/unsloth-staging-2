# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tool-call argument objectification in chat_template_helpers.

When a template renders assistant ``tool_calls`` arguments as a mapping (Gemma:
``call:NAME{key:value}``), the OpenAI-spec JSON *string* form makes it emit the
raw JSON verbatim inside the open brace (``call:NAME{{"k":"v"}}``) -- malformed,
so the model mis-consumes tool results on the transformers/MLX path while GGUF
(llama-server ``func_args_not_string``) is fine. The helper probes the template
and converts string args to dicts only when that is demonstrably the fix, leaving
OpenAI-spec templates that print ``arguments`` as a string untouched.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# Load the dependency-light helper directly (no core.inference side effects).
_H_PATH = Path(_BACKEND_DIR) / "core" / "inference" / "chat_template_helpers.py"
_spec = importlib.util.spec_from_file_location("_cth_test", _H_PATH)
cth = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cth)


# Mapping branch + string branch, like the real gemma-4 template (lines 246-257).
GEMMA_LIKE = (
    "{%- for m in messages -%}"
    "{%- if m['tool_calls'] -%}"
    "{%- for tc in m['tool_calls'] -%}"
    "{{ '<|tool_call>call:' + tc['function']['name'] + '{' }}"
    "{%- if tc['function']['arguments'] is mapping -%}"
    "{%- for k, v in tc['function']['arguments'] | dictsort -%}"
    "{{ k }}:<|\"|>{{ v }}<|\"|>{%- if not loop.last %},{% endif -%}"
    "{%- endfor -%}"
    "{%- elif tc['function']['arguments'] is string -%}"
    "{{ tc['function']['arguments'] }}"
    "{%- endif -%}"
    "{{ '}<tool_call|>' }}"
    "{%- endfor -%}{%- endif -%}{%- endfor -%}"
)

# OpenAI-spec: prints arguments as a scalar string (must stay a string).
OPENAI_LIKE = (
    "{%- for m in messages -%}{%- if m['tool_calls'] -%}{%- for tc in m['tool_calls'] -%}"
    "{{ tc['function']['name'] }}({{ tc['function']['arguments'] }})"
    "{%- endfor -%}{%- endif -%}{%- endfor -%}"
)

# tojson: re-serializes arguments; conservative gate must NOT touch it.
TOJSON_LIKE = (
    "{%- for m in messages -%}{%- if m['tool_calls'] -%}{%- for tc in m['tool_calls'] -%}"
    "{{ tc['function']['name'] }}({{ tc['function']['arguments'] | tojson }})"
    "{%- endfor -%}{%- endif -%}{%- endfor -%}"
)


class FakeTokenizer:
    """Minimal stand-in: renders the given jinja template the way transformers
    does (ImmutableSandboxedEnvironment, ascii off)."""

    def __init__(self, template: str):
        self.chat_template = template

    def apply_chat_template(self, messages, tokenize = False, add_generation_prompt = True, **kw):
        from jinja2.sandbox import ImmutableSandboxedEnvironment

        env = ImmutableSandboxedEnvironment(trim_blocks = True, lstrip_blocks = True)
        env.policies["json.dumps_kwargs"]["ensure_ascii"] = False
        return env.from_string(self.chat_template).render(
            messages = messages,
            add_generation_prompt = add_generation_prompt,
            bos_token = "<bos>",
            **kw,
        )


def _convo(arguments):
    return [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "search", "arguments": arguments}}
            ],
        },
        {"role": "tool", "name": "search", "content": "RESULT", "tool_call_id": "c1"},
    ]


STR_ARGS = '{"query":"hello world"}'
DICT_ARGS = {"query": "hello world"}


def setup_function(_):
    pytest.importorskip("jinja2")


# ── the probe ───────────────────────────────────────────────────────────


def test_probe_true_for_mapping_template():
    assert cth._template_wants_object_tool_args(FakeTokenizer(GEMMA_LIKE)) is True


def test_probe_false_for_openai_string_template():
    # bare {{ arguments }} -> dict would render as python repr; must stay string.
    assert cth._template_wants_object_tool_args(FakeTokenizer(OPENAI_LIKE)) is False


def test_probe_false_for_tojson_template():
    # conservative: re-serializing templates are left alone (no regression).
    assert cth._template_wants_object_tool_args(FakeTokenizer(TOJSON_LIKE)) is False


def test_probe_is_cached_on_tokenizer():
    tok = FakeTokenizer(GEMMA_LIKE)
    assert cth._template_wants_object_tool_args(tok) is True
    cached = getattr(tok, cth._TOKENIZER_OBJECTIFY_ATTR)
    assert cached == (GEMMA_LIKE, True)  # keyed by template content


def test_probe_reprobes_when_template_swapped():
    # The tool path swaps an override template for the native one on the same
    # tokenizer; the cache must not return a stale result after the swap.
    tok = FakeTokenizer(OPENAI_LIKE)
    assert cth._template_wants_object_tool_args(tok) is False
    tok.chat_template = GEMMA_LIKE  # swap to the tool-capable native template
    assert cth._template_wants_object_tool_args(tok) is True


# ── the message transform ───────────────────────────────────────────────


def test_objectify_parses_json_object_string():
    out = cth._objectify_tool_call_arguments(_convo(STR_ARGS))
    assert out[1]["tool_calls"][0]["function"]["arguments"] == DICT_ARGS


def test_objectify_leaves_non_json_string_untouched():
    out = cth._objectify_tool_call_arguments(_convo("not json"))
    assert out[1]["tool_calls"][0]["function"]["arguments"] == "not json"


def test_objectify_leaves_non_object_json_untouched():
    # a JSON array is not an object; func_args_not_string only objectifies dicts.
    out = cth._objectify_tool_call_arguments(_convo("[1, 2, 3]"))
    assert out[1]["tool_calls"][0]["function"]["arguments"] == "[1, 2, 3]"


def test_objectify_does_not_mutate_input():
    convo = _convo(STR_ARGS)
    cth._objectify_tool_call_arguments(convo)
    assert convo[1]["tool_calls"][0]["function"]["arguments"] == STR_ARGS


def test_has_string_tool_args_detection():
    assert cth._messages_have_string_tool_args(_convo(STR_ARGS)) is True
    assert cth._messages_have_string_tool_args(_convo(DICT_ARGS)) is False
    assert cth._messages_have_string_tool_args([{"role": "user", "content": "hi"}]) is False


# ── end to end through apply_chat_template_for_generation ────────────────


def test_gemma_render_no_double_brace_after_fix():
    prompt = cth.apply_chat_template_for_generation(FakeTokenizer(GEMMA_LIKE), _convo(STR_ARGS))
    assert "call:search{query:<|\"|>hello world<|\"|>}" in prompt
    assert "{{" not in prompt  # the double-brace corruption is gone


def test_gemma_render_double_brace_without_fix_baseline():
    # Sanity: the same template with the raw string arg DOES corrupt (the bug).
    tok = FakeTokenizer(GEMMA_LIKE)
    raw = tok.apply_chat_template(_convo(STR_ARGS))
    assert 'call:search{{"query":"hello world"}}' in raw


def test_openai_render_keeps_string_arguments():
    # Must not objectify: the scalar-print template needs the JSON string intact.
    prompt = cth.apply_chat_template_for_generation(FakeTokenizer(OPENAI_LIKE), _convo(STR_ARGS))
    assert 'search({"query":"hello world"})' in prompt


def test_no_tool_calls_is_passthrough():
    msgs = [{"role": "user", "content": "hi"}]
    prompt = cth.apply_chat_template_for_generation(FakeTokenizer(GEMMA_LIKE), msgs)
    assert prompt == ""  # template emits nothing for a plain user turn
