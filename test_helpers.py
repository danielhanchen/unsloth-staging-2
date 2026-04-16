"""Shared helpers for PR #5049 tests. Extracts functions from source
without triggering unsloth's full import chain."""
import ast as pyast
import logging
import os
import re

_SRC_PATH = os.path.join(os.path.dirname(__file__), "unsloth", "tokenizer_utils.py")
_CACHE = {}


def _source():
    if "src" not in _CACHE:
        with open(_SRC_PATH) as f:
            _CACHE["src"] = f.read()
    return _CACHE["src"]


def _exec_functions(*names):
    src = _source()
    tree = pyast.parse(src)
    _logger = logging.getLogger("unsloth.test")
    _logger.warning_once = _logger.warning
    ns = {"re": re, "os": os, "logger": _logger, "__builtins__": __builtins__}
    for node in tree.body:
        if isinstance(node, pyast.Assign):
            for t in node.targets:
                if isinstance(t, pyast.Name) and (t.id.startswith("_RE_") or t.id.startswith("_RENDER_DIFF")):
                    exec(pyast.get_source_segment(src, node), ns)
    deps = set(names)
    deps.update([
        "_if_body_emits_content", "_has_add_generation_prompt_block",
        "_find_end_position", "_template_ends_with_toplevel_for",
        "_derive_assistant_prefix_by_render", "_name_is_local_path",
        "_is_strict_chat_template_mode", "_format_chat_template_message",
        "_fix_chat_template", "_validate_patched_template",
        "_repair_string_template", "_fix_chat_template_for_tokenizer",
    ])
    for node in pyast.walk(tree):
        if isinstance(node, (pyast.FunctionDef, pyast.ClassDef)):
            if node.name in deps:
                exec(pyast.get_source_segment(src, node), ns)
    return ns


def get_fn(name):
    return _exec_functions(name)[name]


def get_fns(*names):
    ns = _exec_functions(*names)
    return {n: ns[n] for n in names if n in ns}


def get_all():
    return _exec_functions()


import jinja2


class StubTokenizer:
    def __init__(self, template, name_or_path="test-model", read_only=False):
        self._template = template
        self.name_or_path = name_or_path
        self._read_only = read_only

    @property
    def chat_template(self):
        return self._template

    @chat_template.setter
    def chat_template(self, value):
        if self._read_only:
            raise AttributeError("read-only")
        self._template = value

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False):
        env = jinja2.Environment(autoescape=False, keep_trailing_newline=True)
        return env.from_string(self._template).render(
            messages=messages, add_generation_prompt=add_generation_prompt,
        )
