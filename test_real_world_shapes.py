"""Real-world template shapes: Phi-4 im_sep, deep nesting, multi-role."""
import jinja2
from test_helpers import get_fn, StubTokenizer

_fix = get_fn("_fix_chat_template")
_fix_tok = get_fn("_fix_chat_template_for_tokenizer")
_derive = get_fn("_derive_assistant_prefix_by_render")


def _render(template, msgs, agp):
    env = jinja2.Environment(autoescape=False, keep_trailing_newline=True)
    return env.from_string(template).render(messages=msgs, add_generation_prompt=agp)


PHI4 = (
    "{% for message in messages %}"
    "<|im_start|>{{ message['role'] }}<|im_sep|>{{ message['content'] }}<|im_end|>\n"
    "{% endfor %}"
)

LLAMA3 = (
    "<|begin_of_text|>"
    "{% for message in messages %}"
    "<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n\n"
    "{{ message['content'] }}<|eot_id|>"
    "{% endfor %}"
)

GEMMA_LIKE = (
    "{% for message in messages %}"
    "<start_of_turn>{{ message['role'] }}\n{{ message['content'] }}<end_of_turn>\n"
    "{% endfor %}"
)


def test_phi4_im_sep_prefix():
    prefix = _derive(PHI4, is_sharegpt=False)
    assert prefix is not None
    assert "<|im_start|>assistant<|im_sep|>" in prefix


def test_phi4_repair():
    tok = StubTokenizer(PHI4)
    result = _fix_tok(tok, PHI4)
    assert "add_generation_prompt" in result
    msgs = [{"role": "user", "content": "Hi"}]
    out = _render(result, msgs, True)
    assert "<|im_start|>assistant<|im_sep|>" in out


def test_llama3_prefix():
    prefix = _derive(LLAMA3, is_sharegpt=False)
    assert prefix is not None
    assert "<|start_header_id|>assistant<|end_header_id|>" in prefix


def test_llama3_repair():
    tok = StubTokenizer(LLAMA3)
    result = _fix_tok(tok, LLAMA3)
    assert "add_generation_prompt" in result
    msgs = [{"role": "user", "content": "Hi"}]
    out = _render(result, msgs, True)
    assert "<|start_header_id|>assistant<|end_header_id|>" in out


def test_gemma_like_prefix():
    prefix = _derive(GEMMA_LIKE, is_sharegpt=False)
    assert prefix is not None
    assert "<start_of_turn>assistant" in prefix


def test_gemma_like_repair():
    tok = StubTokenizer(GEMMA_LIKE)
    result = _fix_tok(tok, GEMMA_LIKE)
    assert "add_generation_prompt" in result


def test_system_message_template():
    t = (
        "{% if messages[0]['role'] == 'system' %}"
        "SYSTEM: {{ messages[0]['content'] }}\n"
        "{% set messages = messages[1:] %}"
        "{% endif %}"
        "{% for message in messages %}"
        "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
        "{% endfor %}"
    )
    tok = StubTokenizer(t)
    result = _fix_tok(tok, t)
    assert "add_generation_prompt" in result
