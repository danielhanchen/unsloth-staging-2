"""Inference smoke job: load, greedy generate, time it.

    python jobs/inference_smoke.py                          # unsloth/Qwen3-0.6B, HF generate
    python jobs/inference_smoke.py --fast-inference         # vLLM fast_generate (Linux CUDA)
    python jobs/inference_smoke.py --lora outputs/my_lora   # with a saved adapter
    python jobs/inference_smoke.py --tiny                   # tiny model, CPU / CI smoke

Reports cold and warm first-token latency (a 1-token generate), decode tokens/s and peak memory.
Checks: every prompt yields 1..max_new_tokens tokens, prompt loss is finite, greedy is repeatable.
"""

import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as c  # noqa: E402

p = c.base_parser("inference_smoke", "unsloth/Qwen3-0.6B", "trl-internal-testing/tiny-Qwen3ForCausalLM", default_steps=0)
p.add_argument("--fast-inference", action="store_true")
p.add_argument("--gpu-memory-utilization", type=float, default=0.3)
p.add_argument("--load-in-4bit", action="store_true")
p.add_argument("--lora", default=None, help="saved LoRA adapter dir to apply")
p.add_argument("--max-new-tokens", type=int, default=32)
a = c.resolve_args(p)

PROMPTS = ["The capital of France is", "Write one sentence about the ocean.",
           "List three prime numbers:", "def fibonacci(n):"]


with c.JobRecorder(a) as rec:
    c.seed_everything(a.seed)
    if a.backend != "hf":
        import unsloth  # noqa: F401
    elif a.fast_inference:
        raise SystemExit("--fast-inference needs the Unsloth backend")
    import torch
    import _compat as k

    rec.set_backend(k.backend_name(a, c.detect_device()) + ("+vllm" if a.fast_inference else ""))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if a.backend != "hf":
        from unsloth import FastLanguageModel
        extra = dict(fast_inference=True, gpu_memory_utilization=a.gpu_memory_utilization,
                     enforce_eager=True) if a.fast_inference else {}
        if a.fast_inference and a.lora:
            extra["max_lora_rank"] = 64
        model, tok = FastLanguageModel.from_pretrained(a.model, max_seq_length=a.max_seq_length,
                                                       load_in_4bit=a.load_in_4bit, **extra)
        if a.lora and not a.fast_inference:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, a.lora)
        FastLanguageModel.for_inference(model)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.model)
        model = AutoModelForCausalLM.from_pretrained(
            a.model, torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32).to(dev).eval()
        if a.lora:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, a.lora)
    k.ensure_pad(tok)
    t = getattr(tok, "tokenizer", tok)
    t.padding_side = "left"
    c.peak_memory_gb(reset=True)

    def hf_generate(n):
        enc = t(PROMPTS, return_tensors="pt", padding=True).to(model.device)
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=n, do_sample=False, pad_token_id=t.pad_token_id)
        if dev == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        new = out[:, enc["input_ids"].shape[1]:]
        counts = [int((row != t.pad_token_id).sum()) for row in new]
        return dt, counts, new.tolist()

    def vllm_generate(n):
        from vllm import SamplingParams
        sp = SamplingParams(temperature=0.0, max_tokens=n, seed=a.seed)
        lora = model.load_lora(a.lora) if a.lora else None
        t0 = time.perf_counter()
        outs = model.fast_generate(PROMPTS, sampling_params=sp, lora_request=lora, use_tqdm=False)
        dt = time.perf_counter() - t0
        ids = [list(o.outputs[0].token_ids) for o in outs]
        return dt, [len(x) for x in ids], ids

    gen = vllm_generate if a.fast_inference else hf_generate
    ttft_cold, _, _ = gen(1)
    ttft_warm, _, _ = gen(1)
    dt, counts, ids = gen(a.max_new_tokens)
    dt2, counts2, ids2 = gen(a.max_new_tokens)
    decode_tokens = sum(max(x - 1, 0) for x in counts2)
    decode_s = max(dt2 - ttft_warm, 1e-9)

    loss = None
    if not a.fast_inference:
        enc = t(PROMPTS, return_tensors="pt", padding=True).to(model.device)
        labels = enc["input_ids"].masked_fill(enc["attention_mask"] == 0, -100)
        with torch.no_grad():
            loss = float(model(**enc, labels=labels).loss)

    rec.summary(first_token_s_cold=round(ttft_cold, 4), first_token_s_warm=round(ttft_warm, 4),
                generate_s=round(dt2, 4), decode_tokens_per_s=round(decode_tokens / decode_s, 2),
                generated_tokens=counts2, prompt_loss=loss, peak_mem_gb=c.peak_memory_gb(),
                sample=t.decode(ids2[0], skip_special_tokens=True)[:200])

    rec.backend_check()
    rec.check("tokens_generated", all(0 < x <= a.max_new_tokens for x in counts2), counts2)
    rec.check("greedy_repeatable", ids == ids2, "two greedy runs gave identical token ids" if ids == ids2 else "outputs differ")
    if loss is not None:
        rec.check("prompt_loss_finite", math.isfinite(loss), loss)
