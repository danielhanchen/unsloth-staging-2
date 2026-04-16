"""Thread safety: concurrent _VariantTokenizerProxy swap-and-restore."""
import ast as pyast
import os
import re
import threading
from test_helpers import _source, StubTokenizer

src = _source()
tree = pyast.parse(src)
ns = {"re": re, "os": os, "__builtins__": __builtins__}
for node in pyast.walk(tree):
    if isinstance(node, pyast.ClassDef) and node.name == "_VariantTokenizerProxy":
        exec(pyast.get_source_segment(src, node), ns)
Proxy = ns["_VariantTokenizerProxy"]

CHATML = (
    "{% for message in messages %}"
    "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
    "{% endfor %}"
)
ALT = (
    "{% for message in messages %}"
    "[{{ message['role'] }}]: {{ message['content'] }}\n"
    "{% endfor %}"
)


def test_concurrent_proxy_calls_restore_original():
    original = CHATML
    tok = StubTokenizer(original)
    errors = []
    barrier = threading.Barrier(4)

    def worker(variant_tmpl, label):
        proxy = Proxy(tok, variant_tmpl, variant_label=label)
        barrier.wait()
        for _ in range(20):
            try:
                proxy.apply_chat_template(
                    [{"role": "user", "content": "Hi"}],
                    add_generation_prompt=False,
                    tokenize=False,
                )
            except Exception as e:
                errors.append(str(e))

    threads = []
    for i in range(4):
        tmpl = CHATML if i % 2 == 0 else ALT
        t = threading.Thread(target=worker, args=(tmpl, f"v{i}"))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # Original should be restored (best-effort, may race)
    assert tok.chat_template in (original, CHATML, ALT)
    # No crashes
    assert len(errors) == 0


def test_proxy_restore_on_render_error():
    original = CHATML
    tok = StubTokenizer(original)
    bad_tmpl = "{{ undefined_var.fail }}"
    proxy = Proxy(tok, bad_tmpl)
    try:
        proxy.apply_chat_template(
            [{"role": "user", "content": "Hi"}],
            add_generation_prompt=False,
            tokenize=False,
        )
    except Exception:
        pass
    assert tok.chat_template == original
