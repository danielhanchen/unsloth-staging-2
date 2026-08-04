"""One training step may straddle a torch.compile fallback. Retry it.

When Dynamo exhausts its recompilation cache for a `fullgraph = True` forward,
unsloth_zoo latches that function to eager for the rest of the run. Every step
after the latch is self-consistent, and every step before it was too. Exactly
one step is not: the one whose forward packed activations while compiled and
whose backward recomputes them eagerly. Non-reentrant activation checkpointing
compares the two and aborts:

    AssertionError: Something went unexpectedly wrong in activation
    checkpoint. Please report this bug by filing an issue to PyTorch.

That message names PyTorch for something PyTorch did not do -- it is our
compiler fallback landing mid-step -- and it kills the run at the first
backward. Gemma4_(E2B)-Vision fails exactly this way (rerun16, live L4; the
traceback ends at `checkpoint.py:906`, `holder.handles[gid] in
self.recomputed[gid]`).

So the wrapper under test catches that one assertion, asks unsloth_zoo to
confirm a fallback really happened and to latch everything else too, throws
away the partial gradients, and runs the step again.

The three ways this could go wrong, all pinned below:

  * swallowing assertions that are not this one,
  * retrying when no fallback happened, which would paper over a real bug,
  * looping, which turns a crash into a hang.
"""

import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MSG = ("Something went unexpectedly wrong in activation checkpoint. "
       "Please report this bug by filing an issue to PyTorch.")


@pytest.fixture
def mod(monkeypatch):
    """Load just the two functions under test, without importing unsloth.

    `unsloth/models/_utils.py` imports torch, transformers, triton and a good
    deal else at module scope; none of it is involved here, and pulling it in
    would make a unit test depend on the whole stack being importable.
    """
    import ast
    import functools
    import logging

    src = (ROOT / "unsloth" / "models" / "_utils.py").read_text()
    tree = ast.parse(src)
    wanted = {"_CHECKPOINT_MISMATCH_TEXT", "_compile_mode_flipped_under_us",
              "_retry_step_after_compile_mode_flip"}
    keep = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            keep.append(node)
        elif isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in wanted for t in node.targets):
            keep.append(node)
    assert len(keep) == 3, f"expected 3 definitions, found {len(keep)}"

    ns = types.ModuleType("_utils_slice")
    ns.functools = functools
    ns.logger = logging.getLogger("test")
    exec(compile(ast.Module(body=keep, type_ignores=[]), "<slice>", "exec"),
         ns.__dict__)
    return ns


class Trainer:
    """Just enough of a trainer for the wrapper to zero gradients on."""

    def __init__(self):
        self.optimizer = types.SimpleNamespace(
            zero_grad=lambda set_to_none=True: self.zeroed.append("optimizer"))
        self.zeroed = []


class Model:
    def __init__(self, trainer):
        self._t = trainer

    def zero_grad(self, set_to_none=True):
        self._t.zeroed.append("model")


def _step(failures, record=None):
    """A training step that raises `failures` assertions, then returns 7.0."""
    state = {"n": 0}

    def step(self, model, inputs=None, num_items_in_batch=None):
        state["n"] += 1
        if record is not None:
            record.append(state["n"])
        if state["n"] <= len(failures):
            raise failures[state["n"] - 1]
        return 7.0

    step.calls = state
    return step


def _fallback(n):
    """Stand-in for unsloth_zoo.force_eager_fallback returning n."""
    return lambda: n


# ---- the happy path is untouched -----------------------------------------

def test_a_step_that_works_is_passed_straight_through(mod):
    t = Trainer()
    step = _step([])
    wrapped = mod._retry_step_after_compile_mode_flip(step)
    assert wrapped(t, Model(t)) == 7.0
    assert step.calls["n"] == 1
    assert t.zeroed == []


def test_the_wrapper_keeps_the_wrapped_name(mod):
    def _unsloth_training_step(self, model):
        return 1

    w = mod._retry_step_after_compile_mode_flip(_unsloth_training_step)
    assert w.__name__ == "_unsloth_training_step"


def test_wrapping_twice_is_a_no_op(mod):
    """patch_gradient_accumulation_fix can run more than once per process, and
    a second wrapper would allow two retries of the same step."""
    def step(self, model):
        return 1

    once = mod._retry_step_after_compile_mode_flip(step)
    assert mod._retry_step_after_compile_mode_flip(once) is once


# ---- the retry ------------------------------------------------------------

def test_the_step_is_retried_after_a_confirmed_mode_flip(mod, monkeypatch):
    monkeypatch.setattr(mod, "_compile_mode_flipped_under_us", _fallback(3))
    t = Trainer()
    step = _step([AssertionError(MSG)])
    wrapped = mod._retry_step_after_compile_mode_flip(step)
    assert wrapped(t, Model(t)) == 7.0
    assert step.calls["n"] == 2


def test_partial_gradients_are_discarded_before_the_retry(mod, monkeypatch):
    """The failed backward left gradients on the parameters. Retrying without
    clearing them would accumulate this step's data twice."""
    monkeypatch.setattr(mod, "_compile_mode_flipped_under_us", _fallback(1))
    t = Trainer()
    wrapped = mod._retry_step_after_compile_mode_flip(
        _step([AssertionError(MSG)]))
    wrapped(t, Model(t))
    assert t.zeroed == ["model", "optimizer"]


def test_the_model_is_found_as_a_keyword_too(mod, monkeypatch):
    monkeypatch.setattr(mod, "_compile_mode_flipped_under_us", _fallback(1))
    t = Trainer()
    wrapped = mod._retry_step_after_compile_mode_flip(
        _step([AssertionError(MSG)]))
    assert wrapped(t, model=Model(t)) == 7.0
    assert "model" in t.zeroed


def test_a_trainer_without_an_optimizer_yet_does_not_crash(mod, monkeypatch):
    """training_step can run before the optimizer exists on some paths, and
    failing to clear gradients must not replace the error we are recovering
    from with a different one."""
    monkeypatch.setattr(mod, "_compile_mode_flipped_under_us", _fallback(1))

    class Bare:
        pass

    wrapped = mod._retry_step_after_compile_mode_flip(
        _step([AssertionError(MSG)]))
    assert wrapped(Bare(), None) == 7.0


def test_the_retry_is_announced(mod, monkeypatch, caplog):
    monkeypatch.setattr(mod, "_compile_mode_flipped_under_us", _fallback(4))
    t = Trainer()
    wrapped = mod._retry_step_after_compile_mode_flip(
        _step([AssertionError(MSG)]))
    with caplog.at_level("WARNING", logger="test"):
        wrapped(t, Model(t))
    assert any("retried" in r.message and "4" in r.message
               for r in caplog.records), caplog.records


# ---- what must NOT be retried --------------------------------------------

def test_an_unrelated_assertion_propagates(mod, monkeypatch):
    monkeypatch.setattr(mod, "_compile_mode_flipped_under_us", _fallback(9))
    t = Trainer()
    step = _step([AssertionError("shapes do not match")])
    wrapped = mod._retry_step_after_compile_mode_flip(step)
    with pytest.raises(AssertionError, match="shapes do not match"):
        wrapped(t, Model(t))
    assert step.calls["n"] == 1, "must not have retried"


def test_a_non_assertion_propagates(mod, monkeypatch):
    monkeypatch.setattr(mod, "_compile_mode_flipped_under_us", _fallback(9))
    t = Trainer()
    wrapped = mod._retry_step_after_compile_mode_flip(
        _step([RuntimeError("CUDA out of memory")]))
    with pytest.raises(RuntimeError):
        wrapped(t, Model(t))


def test_the_checkpoint_assertion_propagates_when_nothing_fell_back(mod,
                                                                    monkeypatch):
    """The important negative. If no compiled forward ever fell back then the
    mismatch is a genuine bug in the checkpointed code or in torch, and
    hiding it behind a retry would be much worse than the crash."""
    monkeypatch.setattr(mod, "_compile_mode_flipped_under_us", _fallback(0))
    t = Trainer()
    step = _step([AssertionError(MSG)])
    wrapped = mod._retry_step_after_compile_mode_flip(step)
    with pytest.raises(AssertionError, match="activation checkpoint"):
        wrapped(t, Model(t))
    assert step.calls["n"] == 1
    assert t.zeroed == [], "no recovery attempted, so nothing to clear"


def test_it_retries_once_and_only_once(mod, monkeypatch):
    """A retry loop would convert a hard failure into a hang, which is worse:
    the crash at least tells you what happened."""
    monkeypatch.setattr(mod, "_compile_mode_flipped_under_us", _fallback(2))
    t = Trainer()
    step = _step([AssertionError(MSG), AssertionError(MSG)])
    wrapped = mod._retry_step_after_compile_mode_flip(step)
    with pytest.raises(AssertionError, match="activation checkpoint"):
        wrapped(t, Model(t))
    assert step.calls["n"] == 2


# ---- the bridge to unsloth_zoo -------------------------------------------

def test_an_older_zoo_without_the_helper_is_tolerated(mod, monkeypatch):
    """unsloth is routinely installed alongside an older unsloth_zoo. The
    import failing means there is no retry available, not that anything is
    broken."""
    import builtins
    real = builtins.__import__

    def no_helper(name, *a, **k):
        if name == "unsloth_zoo.temporary_patches.utils":
            raise ImportError("no such helper")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_helper)
    assert mod._compile_mode_flipped_under_us() == 0


def test_a_helper_that_raises_is_tolerated(mod, monkeypatch):
    import builtins
    real = builtins.__import__

    def boom_helper(name, *a, **k):
        if name == "unsloth_zoo.temporary_patches.utils":
            m = types.ModuleType(name)
            def _boom():
                raise RuntimeError("nope")
            m.force_eager_fallback = _boom
            return m
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", boom_helper)
    assert mod._compile_mode_flipped_under_us() == 0


def _zoo_helper():
    try:
        from unsloth_zoo.temporary_patches.utils import force_eager_fallback
    except Exception:
        return None
    return force_eager_fallback


@pytest.mark.skipif(_zoo_helper() is None,
                    reason="paired unsloth_zoo change not installed")
def test_the_real_zoo_helper_is_reachable(mod):
    """The two repos have to agree on the name. A typo is invisible without
    this: the import guard turns it into "no fallback happened" and the retry
    quietly never fires again.

    Skipped rather than failed against an older unsloth_zoo, because tolerating
    exactly that is the documented behaviour of
    `_compile_mode_flipped_under_us` -- but not skipped silently: the reason
    says which half is missing.
    """
    force_eager_fallback = _zoo_helper()
    assert force_eager_fallback() == 0, "nothing compiled in this process"


@pytest.mark.skipif(_zoo_helper() is None,
                    reason="paired unsloth_zoo change not installed")
def test_the_unsloth_side_calls_the_name_the_zoo_exports(mod):
    """Pins the string, not just the import: `_compile_mode_flipped_under_us`
    swallows ImportError, so a renamed helper would degrade to a permanent
    no-retry with nothing logged."""
    import ast
    src = (ROOT / "unsloth" / "models" / "_utils.py").read_text()
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef)
              and n.name == "_compile_mode_flipped_under_us")
    imports = [(n.module, a.name) for n in ast.walk(fn)
               if isinstance(n, ast.ImportFrom) for a in n.names]
    assert ("unsloth_zoo.temporary_patches.utils",
            "force_eager_fallback") in imports, imports


# ---- it is actually installed --------------------------------------------

def test_patch_gradient_accumulation_fix_installs_the_wrapper():
    """Structural: the wrapper is useless if nothing applies it, and this is
    a single line in a 200-line function that is easy to lose in a rebase."""
    import ast

    src = (ROOT / "unsloth" / "models" / "_utils.py").read_text()
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef)
              and n.name == "patch_gradient_accumulation_fix")
    calls = [n.func.id for n in ast.walk(fn)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    assert "_retry_step_after_compile_mode_flip" in calls


def test_the_wrapper_is_applied_after_the_source_rewrite():
    """Order matters. `_unsloth_training_step` is built by rewriting the
    transformers source and assigning it; wrapping before that assignment
    would be overwritten by it."""
    src = (ROOT / "unsloth" / "models" / "_utils.py").read_text()
    assign = src.index("Trainer.training_step = _unsloth_training_step")
    wrap = src.index("Trainer.training_step = _retry_step_after_compile_mode_flip")
    assert assign < wrap


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
