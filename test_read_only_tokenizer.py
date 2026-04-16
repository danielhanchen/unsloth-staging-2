"""Tests for read-only tokenizer behavior across repair pipeline."""
from test_helpers import get_fn, StubTokenizer

_validate = get_fn("_validate_patched_template")
_repair = get_fn("_repair_string_template")
_fix_tok = get_fn("_fix_chat_template_for_tokenizer")

CHATML_NO_AGP = (
    "{% for message in messages %}"
    "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
    "{% endfor %}"
)

CHATML_WITH_AGP = (
    CHATML_NO_AGP
    + "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def test_validate_read_only_returns_false():
    tok = StubTokenizer(CHATML_NO_AGP, read_only=True)
    assert _validate(tok, CHATML_WITH_AGP, is_sharegpt=False) is False


def test_repair_read_only_returns_none():
    tok = StubTokenizer(CHATML_NO_AGP, read_only=True)
    result = _repair(tok, CHATML_NO_AGP, is_sharegpt=False)
    assert result is None


def test_fix_for_tokenizer_read_only_warns_not_crashes():
    tok = StubTokenizer(CHATML_NO_AGP, read_only=True)
    result = _fix_tok(tok, CHATML_NO_AGP)
    # Should return something (original or repaired), never raise
    assert isinstance(result, str)


def test_validate_restores_on_success():
    original = CHATML_NO_AGP
    tok = StubTokenizer(original)
    _validate(tok, CHATML_WITH_AGP, is_sharegpt=False)
    assert tok.chat_template == original


def test_validate_restores_on_failure():
    original = CHATML_NO_AGP
    tok = StubTokenizer(original)
    _validate(tok, "bad template {{ undefined.thing }}", is_sharegpt=False)
    assert tok.chat_template == original
