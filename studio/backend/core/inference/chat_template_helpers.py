# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Dependency-light wrapper around tokenizer.apply_chat_template with a kwarg
fallback for templates that reject reasoning/tools args, plus the shared
native-chat-template fallback used by the transformers and MLX backends.

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

import copy
import json
import logging
from typing import Optional


logger = logging.getLogger(__name__)


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
    # Normalize OpenAI-spec string tool_call arguments to dicts when the template
    # renders them as a mapping (Gemma etc.), matching llama-server's GGUF behavior.
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


def render_native_template(
    *,
    model_info: dict,
    active_model_name: Optional[str],
    messages: list,
    tools: list,
    enable_thinking: Optional[bool] = None,
    reasoning_effort: Optional[str] = None,
    preserve_thinking: Optional[bool] = None,
    apply_fn = None,
    hf_token: Optional[str] = None,
) -> Optional[str]:
    """Render ``messages`` + ``tools`` with the model's NATIVE chat template.

    Some Unsloth override templates (e.g. ``mistral``, ``gemma-4``) do not emit
    the ``tools`` schema, so a tool-calling turn silently stops advertising tools.
    The native template ships in the model repo and carries the family's
    tool-calling syntax. It is loaded straight from the repo (bypassing any
    override on the live tokenizer) and cached on ``model_info``. Returns the
    rendered prompt only if the native template actually emits the tools (render
    differs with vs without tools); otherwise ``None``.

    ``hf_token`` is the token the model was loaded with -- passed to the repo load
    so a gated/private model's native template can still be fetched (otherwise the
    fallback fails silently and keeps the override prompt that dropped tools).
    """
    # ``apply_fn`` lets a backend inject its own render (e.g. one that peels
    # template-rejected kwargs); defaults to the dependency-light module helper.
    if apply_fn is None:
        apply_fn = apply_chat_template_for_generation
    native_tpl = model_info.get("native_chat_template")
    if native_tpl is None:
        # For a LoRA adapter the native chat template lives on the base model;
        # active_model_name is the adapter id/path and often ships no template.
        template_source = model_info.get("base_model") or active_model_name
        try:
            from transformers import AutoTokenizer
            nt = AutoTokenizer.from_pretrained(
                template_source,
                token = hf_token if hf_token and hf_token.strip() else None,
            )
            native_tpl = nt.chat_template or False
        except Exception as exc:
            logger.warning(
                "Could not load native chat template for '%s': %s",
                template_source,
                exc,
            )
            native_tpl = False
        model_info["native_chat_template"] = native_tpl
    if not native_tpl:
        return None

    tokenizer = model_info.get("tokenizer") or model_info.get("processor")
    if tokenizer is None:
        return None
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    # Render on a shallow copy carrying the native template. Mutating the shared
    # tokenizer.chat_template here races concurrent requests (this runs outside the
    # generation lock), which could make another request render with the native
    # template or restore over its saved value.
    try:
        render_tokenizer = copy.copy(tokenizer)
        render_tokenizer.chat_template = native_tpl
    except Exception as exc:
        logger.warning(
            "Could not clone tokenizer for native-template render of '%s': %s",
            active_model_name,
            exc,
        )
        return None
    try:
        with_tools = apply_fn(
            render_tokenizer,
            messages,
            tools = tools,
            enable_thinking = enable_thinking,
            reasoning_effort = reasoning_effort,
            preserve_thinking = preserve_thinking,
        )
        no_tools = apply_fn(
            render_tokenizer,
            messages,
            tools = None,
            enable_thinking = enable_thinking,
            reasoning_effort = reasoning_effort,
            preserve_thinking = preserve_thinking,
        )
    except Exception as exc:
        logger.warning(
            "Native-template tool render failed for '%s': %s",
            active_model_name,
            exc,
        )
        return None
    return with_tools if with_tools != no_tools else None


def render_with_native_template_fallback(
    *,
    formatted_prompt: str,
    tokenizer,
    model_info: dict,
    active_model_name: Optional[str],
    messages: list,
    tools: Optional[list],
    enable_thinking: Optional[bool] = None,
    reasoning_effort: Optional[str] = None,
    preserve_thinking: Optional[bool] = None,
    apply_fn = None,
    hf_token: Optional[str] = None,
) -> str:
    """Return ``formatted_prompt``, swapping in a native-template render when an
    override template dropped the ``tools`` schema.

    If ``tools`` were requested but the live render is identical with and without
    them (detected by comparison, robust against tool names in the system prompt),
    re-render with the model's native template. Shared by the transformers and MLX
    backends so both advertise tools consistently. ``hf_token`` is forwarded so a
    gated/private model's native template can still be fetched."""
    if not tools:
        return formatted_prompt
    if apply_fn is None:
        apply_fn = apply_chat_template_for_generation
    # The no-tools probe re-renders the live template with ``tools=None`` to detect
    # whether it dropped the schema. A template that *requires* tools can raise here;
    # that is not a reason to discard the already-valid tools prompt, so on any error
    # keep ``formatted_prompt`` (the transformers path would otherwise fall back to
    # manual formatting and lose the schema, and the MLX path would let it escape).
    try:
        probe_no_tools = apply_fn(
            tokenizer,
            messages,
            tools = None,
            enable_thinking = enable_thinking,
            reasoning_effort = reasoning_effort,
            preserve_thinking = preserve_thinking,
        )
    except Exception as exc:
        logger.warning(
            "No-tools probe failed for '%s'; keeping the existing tools prompt: %s",
            active_model_name,
            exc,
        )
        return formatted_prompt
    if formatted_prompt != probe_no_tools:
        return formatted_prompt  # template already emits the tools schema
    native_prompt = render_native_template(
        model_info = model_info,
        active_model_name = active_model_name,
        messages = messages,
        tools = tools,
        enable_thinking = enable_thinking,
        reasoning_effort = reasoning_effort,
        preserve_thinking = preserve_thinking,
        apply_fn = apply_fn,
        hf_token = hf_token,
    )
    if native_prompt:
        logger.info(
            "Override template for '%s' dropped tool schemas; using the model's "
            "native template for this tool-calling turn.",
            active_model_name,
        )
        return native_prompt
    return formatted_prompt
