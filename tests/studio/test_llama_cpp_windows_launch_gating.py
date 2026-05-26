"""Regression locks for PR #5749 review fixes #2 and #3.

The Windows GGUF cancel/spinlock PR added three Windows-specific
mitigations to ``LlamaCppBackend.load_model``:

  * ``--threads 2`` (correctly gated on the full-offload local)
  * ``--cache-ram 0 --ctx-checkpoints 0 --no-cache-prompt --checkpoint-every-n-tokens -1``
  * ``env["OMP_WAIT_POLICY"] = "PASSIVE"`` + ``OMP_NUM_THREADS=2``

The code review found that the second and third blocks were applied to
*every* Windows launch, not just the full-GPU-offload case the comments
described, regressing CPU-only and partial-offload Windows runs. These
tests lock in the full-offload gate on all three blocks. The fully-
offloaded flag is matched by name suffix so renames between
``_fully_gpu_offloaded`` and ``fully_gpu_offloaded`` do not break the
lock.

Test strategy: AST-extract the three relevant ``if`` statements from the
source file (no full module import — the function lives 200+ lines deep
inside ``load_model`` with hundreds of dependencies) and assert each
conditional includes both ``sys.platform == "win32"`` AND
``_fully_gpu_offloaded``. Behavior is then validated by ``exec``-ing the
extracted snippets under a controlled namespace covering all three
relevant input combinations.

Runs identically on ubuntu-latest, macos-14, and windows-latest because
it operates on the source text, not on a real subprocess.
"""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import pytest

_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "studio"
    / "backend"
    / "core"
    / "inference"
    / "llama_cpp.py"
)

_SRC = _SOURCE.read_text(encoding="utf-8")
_TREE = ast.parse(_SRC)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _has_name(node, name: str) -> bool:
    return any(
        isinstance(n, ast.Name) and n.id == name for n in ast.walk(node)
    )


_FULL_OFFLOAD_NAMES = {"fully_gpu_offloaded", "_fully_gpu_offloaded"}
_THREADS_LOCAL_NAMES = {"threads_arg", "_t"}


def _has_full_offload_name(node) -> bool:
    return any(
        isinstance(n, ast.Name) and n.id in _FULL_OFFLOAD_NAMES
        for n in ast.walk(node)
    )


def _has_threads_local_name(node) -> bool:
    return any(
        isinstance(n, ast.Name) and n.id in _THREADS_LOCAL_NAMES
        for n in ast.walk(node)
    )


def _is_win32_check(test: ast.expr) -> bool:
    """``_sys.platform == 'win32'`` or ``sys.platform == 'win32'``."""

    if not isinstance(test, ast.Compare):
        return False
    if not (
        isinstance(test.left, ast.Attribute)
        and test.left.attr == "platform"
        and isinstance(test.left.value, ast.Name)
        and test.left.value.id in {"sys", "_sys"}
    ):
        return False
    if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
        return False
    comp = test.comparators[0]
    return isinstance(comp, ast.Constant) and comp.value == "win32"


def _find_threads_gate() -> ast.If:
    """``if sys.platform == 'win32' and fully_gpu_offloaded:`` for --threads.

    Matches both the underscore-prefixed legacy names and the cleaned-up
    names introduced by the M2 follow-up."""

    for node in ast.walk(_TREE):
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.BoolOp)
            and isinstance(node.test.op, ast.And)
            and any(_is_win32_check(v) for v in node.test.values)
            and _has_full_offload_name(node.test)
            and _has_threads_local_name(node)
            and not any(
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and child.func.attr == "extend"
                and any(
                    isinstance(arg, (ast.List, ast.Tuple))
                    and any(
                        isinstance(e, ast.Constant) and e.value == "--cache-ram"
                        for e in arg.elts
                    )
                    for arg in child.args
                )
                for child in ast.walk(node)
            )
        ):
            return node
    raise AssertionError("could not locate the --threads gate")


def _find_cache_flags_block() -> ast.If:
    """``if ...win32... and _fully_gpu_offloaded:`` that emits --cache-ram."""

    for node in ast.walk(_TREE):
        if not isinstance(node, ast.If):
            continue
        emits_cache_ram = False
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and child.func.attr == "extend"
                and any(
                    isinstance(arg, (ast.List, ast.Tuple))
                    and any(
                        isinstance(e, ast.Constant) and e.value == "--cache-ram"
                        for e in arg.elts
                    )
                    for arg in child.args
                )
            ):
                emits_cache_ram = True
                break
        if emits_cache_ram:
            return node
    raise AssertionError("could not locate the --cache-ram block")


def _find_omp_env_block() -> ast.If:
    """The innermost ``if`` whose direct body sets OMP_NUM_THREADS via
    ``env.setdefault(...)``. Excludes outer ``if sys.platform == 'win32':``
    blocks that merely contain the inner gate transitively."""

    def _stmt_sets_omp(stmt: ast.stmt) -> bool:
        if not isinstance(stmt, ast.Expr):
            return False
        c = stmt.value
        return (
            isinstance(c, ast.Call)
            and isinstance(c.func, ast.Attribute)
            and c.func.attr == "setdefault"
            and len(c.args) >= 1
            and isinstance(c.args[0], ast.Constant)
            and c.args[0].value == "OMP_NUM_THREADS"
        )

    for node in ast.walk(_TREE):
        if not isinstance(node, ast.If):
            continue
        if any(_stmt_sets_omp(s) for s in node.body):
            return node
    raise AssertionError("could not locate the OMP_NUM_THREADS setdefault block")


# ---------------------------------------------------------------------------
# Structural tests
# ---------------------------------------------------------------------------


def test_threads_gate_requires_full_offload():
    """``--threads 2`` is the original (correct) Windows gate; ensure it
    keeps gating on the full-offload local after future refactors."""

    node = _find_threads_gate()
    assert _has_full_offload_name(node.test), (
        "--threads 2 gate must keep the fully-offloaded local in its condition."
    )


def test_cache_flags_block_is_gated_on_full_offload():
    """Regression lock for PR 5749 review fix #2: the
    --cache-ram/--no-cache-prompt block must only fire on Windows AND
    when the model is fully GPU-offloaded."""

    node = _find_cache_flags_block()
    test_src = ast.unparse(node.test)
    assert any(_is_win32_check(v) for v in (
        node.test.values if isinstance(node.test, ast.BoolOp) else [node.test]
    )), f"Cache-flags block must check sys.platform == 'win32'; got: {test_src}"
    assert _has_full_offload_name(node.test), (
        "Cache-flags block must gate on the fully-offloaded local; "
        "otherwise Windows CPU-only / partial-offload runs lose prompt-"
        "cache reuse across turns. Found condition: "
        f"{test_src}"
    )


def _smallest_enclosing_if(tree: ast.AST, target: ast.AST) -> ast.If | None:
    """Walk ``tree`` and return the smallest-line-span ``If`` strictly
    containing ``target``."""

    best: ast.If | None = None
    best_span: int | None = None
    target_start = getattr(target, "lineno", None)
    target_end = getattr(target, "end_lineno", None) or target_start
    if target_start is None:
        return None
    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or node is target:
            continue
        ns = getattr(node, "lineno", None)
        ne = getattr(node, "end_lineno", None) or ns
        if ns is None or ne is None:
            continue
        if ns < target_start and ne >= target_end:
            span = ne - ns
            if best_span is None or span < best_span:
                best = node
                best_span = span
    return best


def test_omp_env_block_is_gated_on_full_offload():
    """Regression lock for PR 5749 review fix #3: OMP_NUM_THREADS=2 +
    OMP_WAIT_POLICY=PASSIVE must only fire on Windows AND when fully
    GPU-offloaded. Otherwise Windows CPU/hybrid users get throttled to
    2 OpenMP threads."""

    node = _find_omp_env_block()
    assert _has_full_offload_name(node.test), (
        "OMP env block must gate on the fully-offloaded local; got: "
        f"{ast.unparse(node.test)}"
    )

    parent = _smallest_enclosing_if(_TREE, node)
    assert parent is not None, "OMP env block must be nested under a parent if"
    parent_values = (
        parent.test.values if isinstance(parent.test, ast.BoolOp) else [parent.test]
    )
    assert any(_is_win32_check(v) for v in parent_values), (
        "OMP env block's enclosing if must check sys.platform == 'win32'; "
        f"got: {ast.unparse(parent.test)}"
    )


# ---------------------------------------------------------------------------
# Behavioral tests: exec each block under all 4 combinations of
# (platform, _fully_gpu_offloaded) and assert the right flags / env keys
# appear (or don't).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "platform,fully_offloaded,expect_cache_flags",
    [
        ("win32", True, True),
        ("win32", False, False),
        ("linux", True, False),
        ("darwin", True, False),
    ],
)
def test_cache_flags_block_behavior(platform, fully_offloaded, expect_cache_flags):
    block = _find_cache_flags_block()
    block_src = ast.get_source_segment(_SRC, block, padded=True)
    assert block_src is not None
    block_src = textwrap.dedent(block_src)

    class _Sys:
        pass

    sys_mod = _Sys()
    sys_mod.platform = platform

    cmd: list[str] = []
    # Inject both legacy underscore-prefixed names and the cleaned-up
    # names so this exec works regardless of which set the source uses.
    ns = {
        "_sys": sys_mod,
        "sys": sys_mod,
        "_fully_gpu_offloaded": fully_offloaded,
        "fully_gpu_offloaded": fully_offloaded,
        "cmd": cmd,
    }
    exec(block_src, ns)  # noqa: S102 - controlled exec on first-party source

    if expect_cache_flags:
        assert "--cache-ram" in cmd
        assert "--no-cache-prompt" in cmd
        assert "--ctx-checkpoints" in cmd
        assert "--checkpoint-every-n-tokens" in cmd
    else:
        assert "--cache-ram" not in cmd, (
            f"--cache-ram leaked into cmd on platform={platform!r} "
            f"fully_offloaded={fully_offloaded}: {cmd}"
        )
        assert "--no-cache-prompt" not in cmd


@pytest.mark.parametrize(
    "fully_offloaded,expect_omp",
    [
        (True, True),
        (False, False),
    ],
)
def test_omp_env_block_behavior(fully_offloaded, expect_omp):
    block = _find_omp_env_block()
    block_src = ast.get_source_segment(_SRC, block, padded=True)
    assert block_src is not None
    block_src = textwrap.dedent(block_src)

    env: dict = {}
    ns = {
        "_fully_gpu_offloaded": fully_offloaded,
        "fully_gpu_offloaded": fully_offloaded,
        "env": env,
    }
    exec(block_src, ns)  # noqa: S102

    if expect_omp:
        assert env.get("OMP_WAIT_POLICY") == "PASSIVE"
        assert env.get("OMP_NUM_THREADS") == "2"
    else:
        assert "OMP_WAIT_POLICY" not in env, (
            f"OMP_WAIT_POLICY leaked into env when not fully GPU-offloaded: {env}"
        )
        assert "OMP_NUM_THREADS" not in env


@pytest.mark.parametrize(
    "platform,fully_offloaded,expect_threads_2",
    [
        ("win32", True, True),
        ("win32", False, False),
        ("linux", True, False),
        ("darwin", True, False),
    ],
)
def test_threads_gate_behavior(platform, fully_offloaded, expect_threads_2):
    block = _find_threads_gate()
    block_src = ast.get_source_segment(_SRC, block, padded=True)
    assert block_src is not None
    block_src = textwrap.dedent(block_src)

    class _Sys:
        pass

    sys_mod = _Sys()
    sys_mod.platform = platform

    ns = {
        "_sys": sys_mod,
        "sys": sys_mod,
        "_fully_gpu_offloaded": fully_offloaded,
        "fully_gpu_offloaded": fully_offloaded,
        "n_threads": None,
        "_t": None,
        "threads_arg": None,
    }
    exec(block_src, ns)  # noqa: S102
    result = ns.get("threads_arg")
    if result is None:
        result = ns.get("_t")
    if expect_threads_2:
        assert result == 2
    else:
        assert result != 2, (
            f"--threads 2 was forced on platform={platform!r} "
            f"fully_offloaded={fully_offloaded}: threads={result!r}"
        )
