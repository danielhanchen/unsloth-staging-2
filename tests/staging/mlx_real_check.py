"""Real mlx-vlm on real Apple Silicon: does Studio's prompt cache change an answer?

Staging-only harness for unslothai/unsloth#10778. It drives the SAME wiring
``MLXInferenceBackend._generate_vlm`` uses (``prompt_cache`` / ``prompt_cache_state``
/ ``prefill_step_size`` into ``mlx_vlm.stream_generate``), against a real model, and
compares three arms token for token:

  native  the store switched off entirely, which is plain mlx-vlm
  cold    the same request through the store with nothing to reuse
  warm    the next turn of the same chat, resuming the stored prefix

The claims: cold == native (the cache must not change an unreused answer) and
warm == cold (a reused turn must answer as an unreused one). The second request
is deliberately run in the SAME process as the first, because the defect this PR
fixes is state one request leaves behind for the next.

Usage:  python tests/staging/mlx_real_check.py --model mlx-community/... [--image]
Exit 0 only if every arm matches.
"""

import argparse
import json
import os
import sys
import time

BACKEND = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "studio", "backend")
sys.path.insert(0, BACKEND)


def log(message):
    print(message, flush = True)


def load_backend_module():
    from core.inference import mlx_inference
    return mlx_inference


def prompt_for(processor, config, text, image):
    from mlx_vlm.prompt_utils import apply_chat_template
    return apply_chat_template(processor, config, text, num_images = 1 if image else 0)


def media_ids(mi, config):
    from core.inference.mlx_inference import MLXInferenceBackend
    return MLXInferenceBackend._vlm_media_token_ids(config)


def run_once(mi, model, processor, config, text, images, store, step, label):
    """One request. ``store`` None means the store is off, which is plain mlx-vlm."""
    from mlx_vlm import stream_generate as vlm_stream
    from mlx_vlm.models.cache import make_prompt_cache

    rendered = prompt_for(processor, config, text, bool(images))
    kwargs = dict(max_tokens = 32, temperature = 0.0, verbose = False)
    session = None
    if store is not None:
        language_model = getattr(model, "language_model", model)
        session = mi.VLMPromptCacheSession(
            store,
            "staging-model",
            language_model,
            lambda: make_prompt_cache(language_model, max_kv_size = None),
            media_token_ids = media_ids(mi, config),
            releases_unserved = bool(images),
            **({"step": step} if _takes_step(mi) else {}),
        )
        kwargs["prompt_cache"] = session.cache
        kwargs["prompt_cache_state"] = session
        kwargs["prefill_step_size"] = step

    tokens = []
    started = time.perf_counter()
    scope = session if session is not None else _null()
    with scope:
        final = None
        for response in vlm_stream(model, processor, rendered, images, **kwargs):
            final = response
            token = getattr(response, "token", None)
            if token is not None:
                tokens.append(int(token))
        if session is not None:
            session.finish()
    elapsed = time.perf_counter() - started
    cached = int(getattr(final, "cached_tokens", 0) or 0)
    prompt_tokens = int(getattr(final, "prompt_tokens", 0) or 0)
    log(f"  [{label}] prompt={prompt_tokens} cached={cached} tokens={len(tokens)} {elapsed:.1f}s")
    return {"tokens": tokens, "cached": cached, "prompt_tokens": prompt_tokens, "seconds": elapsed}


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def _takes_step(mi):
    return "step" in mi.VLMPromptCacheSession.__init__.__code__.co_varnames


def module_step(mi):
    if hasattr(mi, "vlm_prefill_step"):
        return mi.vlm_prefill_step()
    return mi.VLM_PROMPT_CACHE_PREFILL_STEP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required = True)
    parser.add_argument("--image", action = "store_true")
    parser.add_argument("--turns", type = int, default = 3)
    parser.add_argument("--filler", type = int, default = 900,
                        help = "words of context per turn, to push the prompt past the grid")
    args = parser.parse_args()

    mi = load_backend_module()
    step = module_step(mi)
    log(f"module step = {step}   session takes a step = {_takes_step(mi)}")

    from mlx_vlm import load
    from mlx_vlm.utils import load_config

    model, processor = load(args.model)
    config = load_config(args.model)

    images = []
    if args.image:
        from PIL import Image, ImageDraw
        picture = Image.new("RGB", (224, 224), (30, 90, 200))
        ImageDraw.Draw(picture).rectangle((40, 40, 180, 180), fill = (240, 200, 20))
        images = [picture]

    filler = " ".join(f"note{index}" for index in range(args.filler))
    questions = [
        f"{filler} Reply with one short sentence about the number {index}."
        for index in range(args.turns)
    ]

    report = {"model": args.model, "step": step, "image": args.image, "turns": []}
    failures = []

    # A text request first, in this same process, so anything it leaves behind
    # (mRoPE position state, per-layer rows) is there for the requests that follow.
    log("priming request (text, short) ...")
    prime_store = mi.VLMPromptSnapshotStore(4 * 1024**3)
    run_once(mi, model, processor, config, "Say hi.", [], prime_store, step, "prime/store")

    store = mi.VLMPromptSnapshotStore(4 * 1024**3)
    for index, question in enumerate(questions):
        log(f"turn {index}")
        native = run_once(mi, model, processor, config, question, images, None, step, "native")
        cold = run_once(mi, model, processor, config, question, images, store, step, "store")
        warm = run_once(mi, model, processor, config, question, images, store, step, "store again")

        turn = {
            "turn": index,
            "native_tokens": native["tokens"][:12],
            "cold_equals_native": cold["tokens"] == native["tokens"],
            "warm_equals_native": warm["tokens"] == native["tokens"],
            "cached_cold": cold["cached"],
            "cached_warm": warm["cached"],
            "prompt_tokens": native["prompt_tokens"],
            "native_seconds": native["seconds"],
            "warm_seconds": warm["seconds"],
        }
        report["turns"].append(turn)
        if not turn["cold_equals_native"]:
            failures.append(f"turn {index}: an unreused request through the store changed the answer")
        if not turn["warm_equals_native"]:
            failures.append(f"turn {index}: a reused request answered differently")

    report["failures"] = failures
    log(json.dumps(report, indent = 2))
    if failures:
        log("FAIL")
        return 1
    log("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
