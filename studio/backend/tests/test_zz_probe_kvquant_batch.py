"""Real-Metal probe for unsloth#12343: KV-quantized text loads, single path on both arms, resident batch on head."""

import json
import platform
import sys
from types import SimpleNamespace

import pytest

pytestmark = [
    pytest.mark.skipif(
        sys.platform != "darwin" or platform.machine() != "arm64", reason = "Apple Silicon only"
    ),
    pytest.mark.allow_network,
]

MODEL = "mlx-community/Qwen3-0.6B-4bit"
RULES = " ".join(f"Rule {i}: answer in one short sentence." for i in range(120))
QUESTIONS = ["Name a colour.", "What is the capital of France?"]


def _messages(question, rules = True):
    return [{"role": "user", "content": f"{RULES}\n\n{question}" if rules else question}]


def _load(kv_quant):
    from core.inference.mlx_inference import MLXInferenceBackend

    backend = MLXInferenceBackend()
    config = SimpleNamespace(identifier = MODEL, is_vision = False, is_lora = False)
    assert backend.load_model(config, max_seq_length = 4096, kv_quant = kv_quant)
    return backend


def _single(
    backend,
    question,
    rules = True,
):
    out = list(
        backend.generate_chat_response(
            _messages(question, rules),
            temperature = 0.0,
            top_p = 1.0,
            top_k = 0,
            max_new_tokens = 24,
            enable_thinking = False,
        )
    )
    usage = (backend.last_generation_stats or {}).get("usage") or {}
    return out[-1] if out else None, usage.get("prompt_tokens_details", {}).get("cached_tokens")


@pytest.mark.parametrize("kv_quant", [None, "4", "8", "tq-4"])
def test_single_path_text(kv_quant):
    backend = _load(kv_quant)
    rows = {
        "kv_quant": kv_quant,
        "is_vlm": backend._is_vlm,
        "kv_bits": backend._kv_quant_bits(),
        "resident_reason": backend.resident_unavailable_reason({}),
        "short": [_single(backend, q, rules = False)[0] for q in QUESTIONS],
        "long": [_single(backend, q)[0] for q in QUESTIONS],
    }
    print("PROBE_SINGLE", json.dumps(rows, default = str))
    assert all(rows["short"]) and all(rows["long"])


def _batch_one(backend, question):
    session = backend.open_resident_batch(width = 1)
    text, stats = None, None
    try:
        session.admit(
            {
                "messages": _messages(question),
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 0,
                "min_p": 0.0,
                "max_new_tokens": 24,
                "repetition_penalty": 1.0,
                "enable_thinking": False,
            },
            0,
        )
        while stats is None:
            for _handle, snapshot in session.step():
                if snapshot is None:
                    stats = session.take_stats(0)
                else:
                    text = snapshot
    finally:
        session.close()
    return text, stats["usage"]["prompt_tokens_details"]["cached_tokens"]


@pytest.mark.parametrize("kv_quant", ["4", "tq-4"])
def test_resident_rows_resume_warm_equal_cold(kv_quant):
    backend = _load(kv_quant)
    reason = backend.resident_unavailable_reason({})
    print("PROBE_RESIDENT_REASON", kv_quant, reason)
    if reason is not None:
        pytest.skip(f"no resident batch here: {reason}")
    cold = _batch_one(backend, QUESTIONS[0])
    warm = _batch_one(backend, QUESTIONS[0])
    single = _single(backend, QUESTIONS[0])
    print(
        "PROBE_RESIDENT",
        json.dumps({"kv_quant": kv_quant, "cold": cold, "warm": warm, "single_after": single}),
    )
    assert warm[0] == cold[0], "warm != cold"
    assert warm[1] > 0 and single[1] and single[1] > 0
