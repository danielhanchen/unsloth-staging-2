# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Dependency-light wrapper around tokenizer.apply_chat_template with a kwarg
fallback for templates that reject reasoning/tools args.

Also ports llama.cpp's ``func_args_not_string`` normalization: when a template
renders assistant ``tool_calls`` arguments by iterating them as a mapping (e.g.
Gemma's ``call:NAME{key:value}``), feeding the OpenAI-spec JSON *string* form
makes the template emit the raw JSON verbatim inside the already-open brace
(``call:NAME{{"key":"value"}}``), a malformed shape the model never saw in
training. llama-server avoids this server-side (caps.cpp ``supports_object_
arguments`` -> ``func_args_not_string``); the transformers/MLX path renders the
embedded jinja directly and had no equivalent, so the same model consumed tool
results via GGUF but not via safetensors. We replicate the gate with a render
probe so OpenAI-spec templates that print arguments as a string are untouched.
"""

import json
from typing import Optional


# Sentinel key/value used by the object-arguments probe. The JSON form uses the
# same compact separators the tool loop emits so the "raw JSON leaked verbatim"
# check matches real assistant tool_call arguments.
_PROBE_KEY = "unsloth_objarg_probe_key"
_PROBE_VALUE = "unsloth_objarg_probe_value"
_PROBE_JSON = json.dumps(
    {_PROBE_KEY: _PROBE_VALUE}, ensure_ascii = False, separators = (",", ":")
)
_TOKENIZER_OBJECTIFY_ATTR = "_unsloth_objectify_tool_args"


def _messages_have_string_tool_args(messages: list) -> bool:
    """True if any assistant message carries tool_call arguments as a string."""
    for message in messages:
        if not isinstance(message, dict):
            continue
        for tool_call in message.get("tool_calls") or []:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if isinstance(function, dict) and isinstance(function.get("arguments"), str):
                return True
    return False


def _raw_render(tokenizer, messages) -> Optional[str]:
    """Render a probe conversation, swallowing any template error (-> None)."""
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize = False, add_generation_prompt = True
        )
    except Exception:
        return None


def _template_wants_object_tool_args(tokenizer) -> bool:
    """Decide whether tool_call string arguments should be parsed to dicts.

    Mirrors llama.cpp's ``supports_object_arguments`` gate via a differential
    render: convert only when the *string* form leaks the raw JSON verbatim into
    the prompt (the ``{{...}}`` double-brace) while the *mapping* form renders
    cleanly and structurally. Conservative -- any ambiguity returns False, so
    templates that print ``arguments`` as a scalar string keep getting a string.
    Cached on the tokenizer keyed by chat_template, so a swap between an Unsloth
    override template and the model's native template (which the tool path does)
    re-probes instead of reusing a stale result.
    """
    template = getattr(tokenizer, "chat_template", None)
    cached = getattr(tokenizer, _TOKENIZER_OBJECTIFY_ATTR, None)
    if isinstance(cached, tuple) and cached[0] == template:
        return cached[1]

    def probe(arguments) -> Optional[str]:
        return _raw_render(
            tokenizer,
            [
                {"role": "user", "content": "x"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_probe",
                            "type": "function",
                            "function": {"name": "probe_fn", "arguments": arguments},
                        }
                    ],
                },
            ],
        )

    out_string = probe(_PROBE_JSON)
    out_dict = probe({_PROBE_KEY: _PROBE_VALUE})

    result = False
    if out_string is not None and out_dict is not None:
        leaks_in_string = _PROBE_JSON in out_string  # raw JSON dumped verbatim
        clean_in_dict = _PROBE_JSON not in out_dict  # mapping form not re-serialized
        not_python_repr = "{'" + _PROBE_KEY not in out_dict  # not str(dict) single-quotes
        value_present = _PROBE_VALUE in out_dict  # value survived the render
        result = leaks_in_string and clean_in_dict and not_python_repr and value_present

    try:
        setattr(tokenizer, _TOKENIZER_OBJECTIFY_ATTR, (template, result))
    except Exception:
        pass
    return result


def _objectify_tool_call_arguments(messages: list) -> list:
    """Copy ``messages`` with assistant tool_call JSON-object string arguments
    parsed into dicts (mirrors llama.cpp ``func_args_not_string``). Non-object
    or non-JSON strings are left untouched."""
    new_messages: list = []
    for message in messages:
        if not (isinstance(message, dict) and message.get("tool_calls")):
            new_messages.append(message)
            continue
        new_calls = []
        changed = False
        for tool_call in message["tool_calls"]:
            function = tool_call.get("function") if isinstance(tool_call, dict) else None
            if isinstance(function, dict) and isinstance(function.get("arguments"), str):
                try:
                    parsed = json.loads(function["arguments"])
                except (json.JSONDecodeError, ValueError):
                    parsed = None
                if isinstance(parsed, dict):
                    new_calls.append(
                        {**tool_call, "function": {**function, "arguments": parsed}}
                    )
                    changed = True
                    continue
            new_calls.append(tool_call)
        if changed:
            new_message = dict(message)
            new_message["tool_calls"] = new_calls
            new_messages.append(new_message)
        else:
            new_messages.append(message)
    return new_messages


def apply_chat_template_for_generation(
    tokenizer,
    messages: list,
    *,
    tools: Optional[list] = None,
    enable_thinking: Optional[bool] = None,
    reasoning_effort: Optional[str] = None,
    preserve_thinking: Optional[bool] = None,
) -> str:
    """Render the chat prompt. Try richest kwargs first; drop one
    group at a time on TypeError. Jinja / missing-variable errors
    propagate."""
    # Normalize tool_call string arguments to dicts for templates that render
    # them as a mapping (Gemma etc.), matching llama-server's GGUF behavior.
    if _messages_have_string_tool_args(messages) and _template_wants_object_tool_args(
        tokenizer
    ):
        messages = _objectify_tool_call_arguments(messages)

    reasoning_kwargs: dict = {}
    if enable_thinking is not None:
        reasoning_kwargs["enable_thinking"] = enable_thinking
    if reasoning_effort is not None:
        reasoning_kwargs["reasoning_effort"] = reasoning_effort
    if preserve_thinking is not None:
        reasoning_kwargs["preserve_thinking"] = preserve_thinking

    attempts: list[dict] = []
    if tools and reasoning_kwargs:
        attempts.append({"tools": tools, **reasoning_kwargs})
    if tools:
        attempts.append({"tools": tools})
    if reasoning_kwargs:
        attempts.append(dict(reasoning_kwargs))
    attempts.append({})

    last_exc: Optional[Exception] = None
    for kwargs in attempts:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize = False,
                add_generation_prompt = True,
                **kwargs,
            )
        except TypeError as e:
            last_exc = e
            continue
        except Exception as e:
            last_exc = e
            break
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("apply_chat_template_for_generation: no attempt produced a result")
