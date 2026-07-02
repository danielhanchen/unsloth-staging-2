# SPDX-License-Identifier: AGPL-3.0-only
"""Real Apple Silicon validation of the MLX presence-penalty logits processor.

This runs on a macos-14 runner where mlx actually installs, so the on-device
scatter semantics and the mlx_lm.stream_generate calling convention that the
safetensors dev host can only reason about are exercised for real. Everywhere
without mlx the whole module self-skips via importorskip.
"""
import os
import sys

import pytest

mx = pytest.importorskip("mlx.core", reason="MLX only ships on arm64 macOS")

# studio/backend on the path so `core.inference...` resolves from tests/.
_BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from core.inference.mlx_inference import _make_mlx_presence_penalty_processor


def _run(proc, seq_ids, vocab=32):
    """Feed a full running sequence; first call latches the prompt length."""
    return proc(mx.array(seq_ids), mx.zeros((1, vocab)))


def test_prompt_only_call_is_noop():
    proc = _make_mlx_presence_penalty_processor(1.5)
    out = _run(proc, [10, 11])  # first call = prompt only
    assert float(out[0, 10]) == pytest.approx(0.0)
    assert float(out[0, 11]) == pytest.approx(0.0)


def test_single_completion_token_penalized_once():
    proc = _make_mlx_presence_penalty_processor(1.5)
    _run(proc, [10, 11])  # latch prompt_len = 2
    out = _run(proc, [10, 11, 5])  # completion = [5]
    assert float(out[0, 5]) == pytest.approx(-1.5)
    assert float(out[0, 10]) == pytest.approx(0.0)  # prompt excluded


def test_duplicate_completion_tokens_penalized_once():
    # Core on-device assumption: scatter-ASSIGN is idempotent for duplicate ids,
    # so a token generated three times is still penalized once (presence, not
    # frequency). This is exactly what cannot be checked without real mlx.
    proc = _make_mlx_presence_penalty_processor(1.5)
    _run(proc, [10, 11])
    out = _run(proc, [10, 11, 7, 7, 7])
    assert float(out[0, 7]) == pytest.approx(-1.5)  # once, not -4.5


def test_multiple_distinct_tokens_each_penalized_once():
    proc = _make_mlx_presence_penalty_processor(2.0)
    _run(proc, [1, 2, 3])
    out = _run(proc, [1, 2, 3, 4, 5, 6])
    for t in (4, 5, 6):
        assert float(out[0, t]) == pytest.approx(-2.0)
    for t in (1, 2, 3):
        assert float(out[0, t]) == pytest.approx(0.0)


def test_negative_penalty_raises_seen_logits():
    proc = _make_mlx_presence_penalty_processor(-1.0)
    _run(proc, [0, 1])
    out = _run(proc, [0, 1, 9])
    assert float(out[0, 9]) == pytest.approx(1.0)


def test_dtype_preserved():
    proc = _make_mlx_presence_penalty_processor(1.5)
    _run(proc, [3, 4])
    logits = mx.zeros((1, 16), dtype=mx.float32)
    out = proc(mx.array([3, 4, 8]), logits)
    assert out.dtype == mx.float32


@pytest.mark.skipif(
    os.environ.get("MLX_CI_SKIP_MODEL") == "1",
    reason="tiny-model integration disabled",
)
def test_stream_generate_threads_processor_and_changes_output():
    """End-to-end on a tiny real MLX model: mlx_lm.stream_generate calls our
    processor as fn(tokens, logits) with tokens 1-D and logits [1, vocab], and a
    strong presence penalty changes the greedy continuation vs no penalty."""
    mlx_lm = pytest.importorskip("mlx_lm")
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    # The mlx_lm tokenizer wrapper returns token ids by default (tokenize=True).
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Repeat the word banana ten times."}],
        add_generation_prompt=True,
    )
    greedy = make_sampler(temp=0.0)

    seen = {"calls": 0, "tokens_1d": True, "logits_2d": True}

    def probe(tokens, logits):
        seen["calls"] += 1
        seen["tokens_1d"] &= len(tokens.shape) == 1
        seen["logits_2d"] &= logits.shape[0] == 1
        return logits

    def gen(processors):
        toks = []
        for r in stream_generate(
            model, tokenizer, prompt=prompt, max_tokens=24,
            sampler=greedy, logits_processors=processors,
        ):
            toks.append(r.token)
        return toks

    gen([probe])  # confirm the calling convention
    assert seen["calls"] > 0
    assert seen["tokens_1d"] and seen["logits_2d"]

    base = gen(None)
    penalized = gen([_make_mlx_presence_penalty_processor(3.0)])
    assert base != penalized, "presence penalty did not change greedy output"
