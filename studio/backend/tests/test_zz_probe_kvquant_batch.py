"""Real-Metal probe for unsloth#12343: a KV-quantized text load batched in Studio's resident batch."""

import json
import platform
import sys
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine() != "arm64", reason = "Apple Silicon only"
)

MODEL = "mlx-community/Qwen3-0.6B-4bit"
RULES = " ".join(f"Rule {i}: answer in one short sentence." for i in range(120))


def _request(question):
    return {
        "messages": [{"role": "user", "content": f"{RULES}\n\n{question}"}],
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        "max_new_tokens": 24,
        "repetition_penalty": 1.0,
        "enable_thinking": False,
    }


def _run_batch(backend, questions):
    session = backend.open_resident_batch(width = len(questions))
    texts, stats = {}, {}
    try:
        for handle, question in enumerate(questions):
            session.admit(_request(question), handle)
        while len(stats) < len(questions):
            for handle, snapshot in session.step():
                if snapshot is None:
                    stats[handle] = session.take_stats(handle)
                else:
                    texts[handle] = snapshot
    finally:
        session.close()
    return [(texts.get(h), stats[h]) for h in range(len(questions))]


@pytest.mark.allow_network
@pytest.mark.parametrize("kv_quant", ["4", "tq-4"])
def test_kv_quantized_text_load_batches_and_reuses_snapshots(kv_quant):
    from core.inference.mlx_inference import MLXInferenceBackend

    backend = MLXInferenceBackend()
    config = SimpleNamespace(identifier = MODEL, is_vision = False, is_lora = False)
    assert backend.load_model(config, max_seq_length = 4096, kv_quant = kv_quant)
    reason = backend.resident_unavailable_reason({})
    report = {
        "kv_quant": kv_quant,
        "is_vlm": backend._is_vlm,
        "kv_bits": backend._kv_quant_bits(),
        "resident_reason": reason,
    }
    print("PROBE_LOAD", json.dumps(report, default = str))
    assert reason is None, reason
    assert backend._is_vlm and backend._kv_quant_bits() is not None

    cold = _run_batch(backend, ["Name a colour.", "Name a fruit."])
    warm = _run_batch(backend, ["Name a colour.", "Name a fruit."])
    rows = [
        {
            "text": text,
            "cached": s["usage"]["prompt_tokens_details"]["cached_tokens"],
            "prompt": s["usage"]["prompt_tokens"],
        }
        for text, s in cold + warm
    ]
    print("PROBE_BATCH", json.dumps(rows, default = str))
    assert all(r["text"] for r in rows)
    assert [r["text"] for r in rows[2:]] == [r["text"] for r in rows[:2]], "warm != cold"
    assert all(r["cached"] > 0 for r in rows[2:]), "warm rows reused nothing"

    # The single path resumes from what the batch banked.
    out = list(
        backend.generate_chat_response(
            [{"role": "user", "content": f"{RULES}\n\nName an animal."}],
            temperature = 0.0,
            top_p = 1.0,
            top_k = 0,
            max_new_tokens = 16,
            enable_thinking = False,
        )
    )
    single = backend.last_generation_stats or {}
    cached = (single.get("usage") or {}).get("prompt_tokens_details", {}).get("cached_tokens")
    print(
        "PROBE_SINGLE",
        json.dumps({"text": out[-1] if out else None, "cached": cached}, default = str),
    )
    assert out and cached and cached > 0, "single path reused nothing from the batch"
