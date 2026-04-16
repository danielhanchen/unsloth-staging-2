"""Verify all 4 iteration-1 fixes are present and working."""
import ast as pyast
import os
import re
import tempfile
from test_helpers import _source, get_fn, StubTokenizer


def test_fix1_sandbox_in_proxy():
    """Proxy fallback must use SandboxedEnvironment, not plain Environment."""
    src = _source()
    tree = pyast.parse(src)
    for node in pyast.walk(tree):
        if isinstance(node, pyast.ClassDef) and node.name == "_VariantTokenizerProxy":
            method_src = pyast.get_source_segment(src, node)
            assert "SandboxedEnvironment" in method_src
            assert "jinja2.Environment(" not in method_src.split("SandboxedEnvironment")[0].split("apply_chat_template")[-1]


def test_fix2_benign_outer_if_unwrapped():
    fn = get_fn("_template_ends_with_toplevel_for")
    t = (
        "{% if messages %}"
        "{% for m in messages %}<|im_start|>{{ m.role }}{% endfor %}"
        "{% endif %}"
    )
    assert fn(t) is True


def test_fix2_qwen3_guard_still_rejected():
    fn = get_fn("_template_ends_with_toplevel_for")
    t = (
        "{% if messages[0].role == 'system' %}"
        "{% for m in messages %}{{ m }}{% endfor %}"
        "{% else %}DEFAULT{% endif %}"
    )
    assert fn(t) is False


def test_fix3_source_path_preserved():
    src = _source()
    tree = pyast.parse(src)
    ns = {"re": re, "os": os, "__builtins__": __builtins__}
    for node in pyast.walk(tree):
        if isinstance(node, pyast.ClassDef) and node.name == "_VariantTokenizerProxy":
            exec(pyast.get_source_segment(src, node), ns)
    Proxy = ns["_VariantTokenizerProxy"]
    tok = StubTokenizer("t", name_or_path="/local/dir")
    proxy = Proxy(tok, "t", variant_label="variant='default'")
    assert proxy._source_path == "/local/dir"
    assert "(variant=" in proxy.name_or_path


def test_fix4_strict_message_no_contradiction():
    fn = get_fn("_format_chat_template_message")
    msg = fn("m", repaired=False, strict=True)
    assert "will still load" not in msg
    assert "Set UNSLOTH_STRICT" not in msg


def test_fix4_warn_message_has_guidance():
    fn = get_fn("_format_chat_template_message")
    msg = fn("m", repaired=False, strict=False)
    assert "will still load" in msg
    assert "Set UNSLOTH_STRICT" in msg


def test_fix3_local_path_source_works():
    fn = get_fn("_format_chat_template_message")
    with tempfile.TemporaryDirectory() as d:
        msg = fn(f"{d} (variant='default')", repaired=False, local_path_source=d)
        assert "local path" in msg
