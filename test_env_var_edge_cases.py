"""Environment variable edge cases for UNSLOTH_STRICT_CHAT_TEMPLATE."""
import os
from test_helpers import get_fn

fn = get_fn("_is_strict_chat_template_mode")


def _with_env(val):
    old = os.environ.get("UNSLOTH_STRICT_CHAT_TEMPLATE")
    if val is None:
        os.environ.pop("UNSLOTH_STRICT_CHAT_TEMPLATE", None)
    else:
        os.environ["UNSLOTH_STRICT_CHAT_TEMPLATE"] = val
    try:
        return fn()
    finally:
        if old is None:
            os.environ.pop("UNSLOTH_STRICT_CHAT_TEMPLATE", None)
        else:
            os.environ["UNSLOTH_STRICT_CHAT_TEMPLATE"] = old


def test_unset():
    assert _with_env(None) is False


def test_zero():
    assert _with_env("0") is False


def test_one():
    assert _with_env("1") is True


def test_true_lowercase():
    assert _with_env("true") is True


def test_true_uppercase():
    assert _with_env("TRUE") is True


def test_true_mixed():
    assert _with_env("True") is True


def test_yes():
    assert _with_env("yes") is True


def test_on():
    assert _with_env("on") is True


def test_false_string():
    assert _with_env("false") is False


def test_no():
    assert _with_env("no") is False


def test_off():
    assert _with_env("off") is False


def test_whitespace_padded():
    assert _with_env("  1  ") is True


def test_empty_string():
    assert _with_env("") is False


def test_random_text():
    assert _with_env("enabled") is False
