"""Full pipeline end-to-end: fix_chat_template with real template shapes."""
import ast as pyast
import logging
import os
import re
import jinja2
from test_helpers import _source, StubTokenizer

# Build full fix_chat_template
src = _source()
tree = pyast.parse(src)
_logger = logging.getLogger("test_e2e")
_logger.warning_once = _logger.warning
ns = {"re": re, "os": os, "logger": _logger, "__builtins__": __builtins__}
for node in tree.body:
    if isinstance(node, pyast.Assign):
        for t in node.targets:
            if isinstance(t, pyast.Name) and (t.id.startswith("_RE_") or t.id.startswith("_RENDER")):
                exec(pyast.get_source_segment(src, node), ns)
for node in tree.body:
    if isinstance(node, (pyast.FunctionDef, pyast.ClassDef)):
        if node.name.startswith("_") or node.name == "fix_chat_template":
            try:
                exec(pyast.get_source_segment(src, node), ns)
            except Exception:
                pass

fix_chat_template = ns["fix_chat_template"]


def _render(template, msgs, agp):
    env = jinja2.Environment(autoescape=False, keep_trailing_newline=True)
    return env.from_string(template).render(messages=msgs, add_generation_prompt=agp)


CHATML_NO_AGP = (
    "{% for message in messages %}"
    "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
    "{% endfor %}"
)


def test_string_chatml_full_pipeline():
    tok = StubTokenizer(CHATML_NO_AGP)
    result = fix_chat_template(tok)
    assert isinstance(result, str)
    assert "add_generation_prompt" in result
    msgs = [{"role": "user", "content": "Hi"}]
    yes = _render(result, msgs, True)
    no = _render(result, msgs, False)
    assert yes != no
    assert yes.startswith(no)


def test_dict_chatml_full_pipeline():
    tok = StubTokenizer(CHATML_NO_AGP)
    tok._template = {"default": CHATML_NO_AGP}
    result = fix_chat_template(tok)
    assert isinstance(result, dict)
    assert "add_generation_prompt" in result["default"]


def test_list_chatml_full_pipeline():
    tok = StubTokenizer(CHATML_NO_AGP)
    tok._template = [{"name": "default", "template": CHATML_NO_AGP}]
    result = fix_chat_template(tok)
    assert isinstance(result, list)
    assert "add_generation_prompt" in result[0]["template"]


def test_already_correct_unchanged():
    good = CHATML_NO_AGP + "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    tok = StubTokenizer(good)
    result = fix_chat_template(tok)
    assert result == good


def test_outer_if_wrapper_full_pipeline():
    t = (
        "{% if messages %}"
        "{% for message in messages %}"
        "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
        "{% endfor %}"
        "{% endif %}"
    )
    tok = StubTokenizer(t)
    result = fix_chat_template(tok)
    assert "add_generation_prompt" in result
    msgs = [{"role": "user", "content": "Hi"}]
    yes = _render(result, msgs, True)
    no = _render(result, msgs, False)
    assert yes != no


def test_both_dash_template_full_pipeline():
    t = (
        "{%- for message in messages -%}"
        "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
        "{%- endfor -%}"
    )
    tok = StubTokenizer(t)
    result = fix_chat_template(tok)
    assert "add_generation_prompt" in result
