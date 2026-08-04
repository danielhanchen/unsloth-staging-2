"""A trainer that fails to build must say so, not fall back in silence.

`_patch_trl_rl_trainers` swallowed every exception into `logger.info`. The
swallow itself is deliberate -- TRL 1.x moves and renames trainer classes, and
a missing trainer is not an error -- but that benign case never reaches the
handler: `_patch_trl_rl_trainers_impl` RETURNS early when
`trl.trainer.<name>` cannot be imported. Anything that does reach the handler
means the module imported and generation failed, and the consequence is severe:
the run continues on trl's own trainer, losing Unsloth's compute_loss, its
bf16/fp16 fixup and its dataset handling in one go.

That is not hypothetical. `_maybe_compile` was added to a function whose source
the compiler copies verbatim into UnslothSFTTrainer.py; the name did not resolve
there; the NameError was logged at info level and never surfaced. Six notebooks
failed with six unrelated-looking errors -- unmatched dataset columns, a vision
collator rejection, empty logits, two 0-d tensor crashes and a bf16 refusal on a
Turing T4 -- and the only signal connecting them was that `UnslothSFTTrainer`
appeared 3 times in every artifact that trained and 0 times in every one that
did not.
"""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = (ROOT / "unsloth" / "models" / "rl.py").read_text(encoding="utf-8")


def _handler():
    """The except block of `_patch_trl_rl_trainers`, by AST rather than by
    line offsets, which drift."""
    for node in ast.walk(ast.parse(SRC)):
        if isinstance(node, ast.FunctionDef) and node.name == "_patch_trl_rl_trainers":
            return ast.get_source_segment(SRC, node)
    raise AssertionError("_patch_trl_rl_trainers not found")


def test_generation_failure_warns_rather_than_whispers():
    body = _handler()
    assert "logger.warning_once(" in body
    assert "logger.info(" not in body, "info level is how this hid for a day"


def test_the_message_says_what_the_user_loses():
    """'Could not patch X' reads like a no-op. The user needs to know training
    silently changed."""
    body = _handler()
    assert "trl's own trainer" in body


def test_the_message_carries_the_exception():
    body = _handler()
    assert "type(e).__name__" in body and "{e}" in body


def test_it_still_swallows(monkeypatch):
    """Raising would break every TRL version that renames a trainer, which is
    the reason the swallow exists. The fix is visibility, not propagation."""
    body = _handler()
    assert "except Exception" in body
    assert "raise" not in body


def test_the_benign_case_never_reaches_the_handler():
    """`_patch_trl_rl_trainers_impl` returns early when the trainer module is
    absent, so a missing trainer cannot trigger the new warning and spam it for
    every trainer this TRL does not ship."""
    for node in ast.walk(ast.parse(SRC)):
        if isinstance(node, ast.FunctionDef) and node.name == "_patch_trl_rl_trainers_impl":
            body = ast.get_source_segment(SRC, node)
            break
    else:
        raise AssertionError("impl not found")
    head = body[:body.index("Patch for vLLM") + 2000] if "Patch for vLLM" in body else body[:2000]
    assert "Could not import trl.trainer" in head
    i = head.index("Could not import trl.trainer")
    assert "return" in head[i:i + 200], "the benign path must return, not raise"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
