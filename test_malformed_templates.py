"""Malformed/adversarial template inputs: broken syntax, empty, unicode."""
from test_helpers import get_fn, StubTokenizer

_fix = get_fn("_fix_chat_template")
_fix_tok = get_fn("_fix_chat_template_for_tokenizer")
_find = get_fn("_find_end_position")
_has_agp = get_fn("_has_add_generation_prompt_block")
_toplevel = get_fn("_template_ends_with_toplevel_for")


def test_empty_string():
    assert _fix("") == ""
    assert _find("") is None
    assert _has_agp("") is False
    assert _toplevel("") is False


def test_none_like_content():
    assert _fix("None") == "None"
    assert _find("None") is None


def test_unclosed_jinja_tag():
    t = "{% for m in messages %}{{ m }}"
    result = _fix(t)
    assert isinstance(result, str)


def test_mismatched_braces():
    t = "{% for m in messages %}{{ m }}{% endfor %}}}"
    result = _fix(t)
    assert isinstance(result, str)


def test_unicode_content():
    t = (
        "{% for message in messages %}"
        "\u2603{{ message['role'] }}\n{{ message['content'] }}\u2603\n"
        "{% endfor %}"
    )
    result = _fix(t, is_sharegpt=False)
    assert isinstance(result, str)


def test_very_long_template():
    body = "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
    t = "{% for message in messages %}" + body * 100 + "{% endfor %}"
    result = _fix(t, is_sharegpt=False)
    assert "add_generation_prompt" in result


def test_template_with_raw_block():
    t = "{% raw %}{% endfor %}{% endraw %}{% for m in messages %}{{ m }}{% endfor %}"
    r = _find(t)
    assert r is not None


def test_only_whitespace():
    assert _fix("   \n\t  ") == "   \n\t  "


def test_tokenizer_with_broken_template():
    tok = StubTokenizer("{% invalid syntax")
    result = _fix_tok(tok, "{% invalid syntax")
    assert result == "{% invalid syntax"
