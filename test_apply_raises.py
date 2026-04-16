"""Tests for _fix_chat_template_for_tokenizer when apply_chat_template raises."""
import jinja2
from test_helpers import get_fn

_fix_tok = get_fn("_fix_chat_template_for_tokenizer")


class RaisingTokenizer:
    """Tokenizer whose apply_chat_template always raises."""
    def __init__(self, template):
        self._t = template
        self.name_or_path = "raise-model"

    @property
    def chat_template(self):
        return self._t

    @chat_template.setter
    def chat_template(self, v):
        self._t = v

    def apply_chat_template(self, messages, **kwargs):
        raise RuntimeError("rendering failed")


class FirstCallRaises:
    """Tokenizer that raises on first HF probe but succeeds on ShareGPT."""
    def __init__(self, template):
        self._t = template
        self.name_or_path = "mixed-model"
        self._call_count = 0

    @property
    def chat_template(self):
        return self._t

    @chat_template.setter
    def chat_template(self, v):
        self._t = v

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False):
        self._call_count += 1
        if any("role" in m for m in messages):
            raise TypeError("HF format not supported")
        env = jinja2.Environment(autoescape=False, keep_trailing_newline=True)
        return env.from_string(self._t).render(
            messages=messages, add_generation_prompt=add_generation_prompt,
        )


def test_all_probes_raise_returns_original():
    t = "{% for m in messages %}{{ m }}{% endfor %}"
    tok = RaisingTokenizer(t)
    result = _fix_tok(tok, t)
    assert result == t


def test_hf_probe_raises_sharegpt_succeeds():
    t = (
        "{% for message in messages %}"
        "<|im_start|>{{ message['from'] }}\n{{ message['value'] }}<|im_end|>\n"
        "{% endfor %}"
    )
    tok = FirstCallRaises(t)
    result = _fix_tok(tok, t)
    assert "add_generation_prompt" in result


def test_no_yes_render_raises_returns_original():
    """Tokenizer where the is_sharegpt probe succeeds but no/yes render raises."""

    class PartialRaiser:
        def __init__(self, t):
            self._t = t
            self.name_or_path = "partial"
            self._probe_done = False

        @property
        def chat_template(self):
            return self._t

        @chat_template.setter
        def chat_template(self, v):
            self._t = v

        def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False):
            if not self._probe_done:
                self._probe_done = True
                return "ok"
            raise RuntimeError("second call fails")

    t = "{% for m in messages %}{{ m }}{% endfor %}"
    tok = PartialRaiser(t)
    result = _fix_tok(tok, t)
    assert result == t
