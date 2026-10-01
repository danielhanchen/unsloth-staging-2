"""GRPO.

    python jobs/grpo.py --max-steps 3                     # notebook preset, HF generation
    python jobs/grpo.py --max-steps 3 --fast-inference    # notebook preset, vLLM
    python jobs/grpo.py --preset smoke --fast-inference   # test_fast_inference.py shape
    python jobs/grpo.py --tiny                            # smoke preset, CPU / CI
    python jobs/grpo.py --preset gpt_oss_tiny             # gpt-oss-(20B)-GRPO recipe on tiny gpt-oss (MoE)

notebook: Qwen3_(4B)-GRPO (chat template, capped SFT priming on OpenMathReasoning-mini,
DAPO-Math-17k, its 4 rewards). smoke: fixed prompts + tie-broken length reward (reward_std > 0);
the only preset --tiny can use, since a random model never earns math rewards.
gpt_oss_tiny: the gpt-oss notebook's GRPOConfig on trl-internal-testing/tiny-GptOssForCausalLM (MoE experts ->
torch._grouped_mm, Unsloth's compiled gpt-oss paths) with the smoke reward; counts grouped_mm calls.
"""

import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as c  # noqa: E402
import _presets as P  # noqa: E402

p = c.base_parser("grpo", "unsloth/Qwen3-0.6B-Base", "trl-internal-testing/tiny-Qwen3ForCausalLM", default_steps=3)
p.add_argument("--preset", choices=("notebook", "smoke", *P.GRPO), default=None,
               help="default: smoke with --tiny, else notebook")
p.add_argument("--fast-inference", action="store_true", help="vLLM generation (fast_inference=True)")
p.add_argument("--gpu-memory-utilization", type=float, default=0.3)
p.add_argument("--lora-r", type=int, default=32)
p.add_argument("--num-generations", type=int, default=4)
p.add_argument("--max-completion-length", type=int, default=256)
p.add_argument("--prime-steps", type=int, default=5, help="notebook SFT priming steps (0 = skip)")
p.add_argument("--rows", type=int, default=64)
p.set_defaults(lr=5e-6, max_seq_length=2048)  # notebook; completions capped
a = c.resolve_args(p)
a.preset = a.preset or ("smoke" if a.tiny else "notebook")
if a.preset == "smoke":
    a.num_generations, a.max_completion_length, a.lora_r = 2, 16, 8
GPT_OSS = a.preset in P.GRPO
if GPT_OSS:
    pre, a.model = P.resolve(P.GRPO, a.preset, a.tiny, a.model if P.model_given(sys.argv) else None)
    a.num_generations, a.max_completion_length, a.lora_r = pre["num_generations"], pre["max_completion_length"], pre["lora_r"]
    if not any(x == "--lr" or x.startswith("--lr=") for x in sys.argv):
        a.lr = pre["lr"]
    if not any(x.startswith("--max-seq-length") for x in sys.argv):
        a.max_seq_length = pre["max_seq_length"]
    a.prime_steps = 0

REASONING_START, REASONING_END = "<start_working_out>", "<end_working_out>"
SOLUTION_START, SOLUTION_END = "<SOLUTION>", "</SOLUTION>"
SYSTEM_PROMPT = (f"You are given a problem.\nThink about the problem and provide your working out.\n"
                 f"Place it between {REASONING_START} and {REASONING_END}.\n"
                 f"Then, provide your solution between {SOLUTION_START}{SOLUTION_END}")


def notebook_chat_template():
    t = ("{% if messages[0]['role'] == 'system' %}{{ messages[0]['content'] + eos_token }}"
         "{% set loop_messages = messages[1:] %}{% else %}{{ 'SYSTEM' + eos_token }}"
         "{% set loop_messages = messages %}{% endif %}{% for message in loop_messages %}"
         "{% if message['role'] == 'user' %}{{ message['content'] }}"
         "{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token }}{% endif %}"
         "{% endfor %}{% if add_generation_prompt %}{{ 'RSTART' }}{% endif %}")
    return t.replace("'SYSTEM'", repr(SYSTEM_PROMPT)).replace("'RSTART'", repr(REASONING_START))


def notebook_rewards(eos):
    end_re = r"</SOLUTION>[\s]{0,}" + "(?:" + re.escape(eos) + ")?"
    match_format = re.compile(rf"{REASONING_END}.*?{SOLUTION_START}(.+?){end_re}[\s]{{0,}}$",
                              flags=re.MULTILINE | re.DOTALL)
    match_numbers = re.compile(SOLUTION_START + r".*?[\s]{0,}([-]?[\d\.\,]{1,})", flags=re.MULTILINE | re.DOTALL)

    def match_format_exactly(completions, **kw):
        return [3.0 if match_format.search(x[0]["content"]) else 0.0 for x in completions]

    def match_format_approximately(completions, **kw):
        out = []
        for x in completions:
            r = x[0]["content"]
            out.append(sum(0.5 if r.count(t) == 1 else -1.0 for t in (REASONING_END, SOLUTION_START, SOLUTION_END)))
        return out

    def check_answer(prompts, completions, answer, **kw):
        out = []
        for x, true in zip(completions, answer):
            g = match_format.search(x[0]["content"])
            if g is None:
                out.append(-2.0)
                continue
            guess = g.group(1)
            if guess == true:
                out.append(5.0)
            elif guess.strip() == true.strip():
                out.append(3.5)
            else:
                try:
                    r = float(guess) / float(true)
                    out.append(2.0 if 0.9 <= r <= 1.1 else 1.5 if 0.8 <= r <= 1.2 else -2.5)
                except Exception:
                    out.append(-4.5)
        return out

    def check_numbers(prompts, completions, answer, **kw):
        out = []
        for x, true in zip(completions, answer):
            g = match_numbers.search(x[0]["content"])
            if g is None:
                out.append(-2.5)
                continue
            try:
                out.append(3.5 if float(g.group(1).strip().replace(",", "")) == float(true.strip()) else -1.5)
            except Exception:
                out.append(0.0)
        return out

    return [match_format_exactly, match_format_approximately, check_answer, check_numbers]


def _finite_kl(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def smoke_reward(completions, **kw):
    # Length + positional tie-breaker: nonzero advantages every step (test_fast_inference.py).
    out = []
    for i, x in enumerate(completions):
        text = x[0]["content"] if isinstance(x, list) else x
        out.append(len(text) / 10.0 + 1e-3 * i)
    return out


with c.JobRecorder(a) as rec:
    c.seed_everything(a.seed)
    if a.backend != "hf":
        import unsloth  # noqa: F401
    elif a.fast_inference:
        raise SystemExit("--fast-inference needs the Unsloth backend")
    import _compat as k
    from datasets import Dataset, load_dataset
    from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer

    rec.set_backend(k.backend_name(a, c.detect_device()) + ("+vllm" if a.fast_inference else ""))
    gmm = k.GroupedMMCounter() if GPT_OSS else None
    extra = dict(fast_inference=True, max_lora_rank=a.lora_r, gpu_memory_utilization=a.gpu_memory_utilization,
                 enforce_eager=True) if a.fast_inference else {}
    model, tok = k.load_text_model(a, lora_r=a.lora_r, lora_alpha=2 * a.lora_r if a.preset in ("notebook", *P.GRPO) else a.lora_r, **extra)
    k.ensure_pad(tok)
    t = getattr(tok, "tokenizer", tok)
    out_dir = os.path.join(os.path.dirname(a.out) or ".", "grpo_run")

    if a.preset == "notebook":
        t.chat_template = notebook_chat_template()
        if a.prime_steps > 0:
            import pandas as pd
            prime = load_dataset("unsloth/OpenMathReasoning-mini", split="cot").to_pandas()
            prime = prime[pd.to_numeric(prime["expected_answer"], errors="coerce").notnull()]
            rows = []
            for _, r in prime.iterrows():
                thoughts = r["generated_solution"].replace("<think>", "").replace("</think>", "").strip()
                msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": r["problem"]},
                        {"role": "assistant", "content": REASONING_START + thoughts + REASONING_END
                         + SOLUTION_START + r["expected_answer"] + SOLUTION_END}]
                text = t.apply_chat_template(msgs, tokenize=False)
                if len(t(text)["input_ids"]) <= a.max_seq_length // 2:
                    rows.append({"text": text})
                if len(rows) >= a.rows:
                    break
            if not rows:
                raise SystemExit(f"no priming rows fit max_seq_length/2={a.max_seq_length // 2}; raise --max-seq-length or --prime-steps 0")
            sa = k.common_train_kwargs(a, out_dir + "_prime")
            # notebook priming SFTConfig; warmup 0 (its 5 would span the capped run)
            sa.update(max_steps=a.prime_steps, per_device_train_batch_size=1, learning_rate=2e-4,
                      weight_decay=0.001, lr_scheduler_type="linear",
                      optim="adamw_8bit" if a.backend != "hf" else "adamw_torch")
            SFTTrainer(model=model, train_dataset=Dataset.from_list(rows),
                       args=k.make(SFTConfig, dataset_text_field="text", max_length=a.max_seq_length, **sa),
                       **k.tokenizer_kwarg(SFTTrainer, tok)).train()
            rec.summary(prime_rows=len(rows), prime_steps=a.prime_steps)
        raw = load_dataset("open-r1/DAPO-Math-17k-Processed", "en", split=f"train[:{a.rows}]")
        rows = [{"prompt": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": r["prompt"]}],
                 "answer": r["solution"]} for r in raw]
        lens = sorted(len(t.apply_chat_template(r["prompt"], add_generation_prompt=True, tokenize=True)) for r in rows)
        cutoff = lens[int(0.9 * (len(lens) - 1))]
        rows = [r for r in rows if len(t.apply_chat_template(r["prompt"], add_generation_prompt=True, tokenize=True)) <= cutoff]
        max_prompt = cutoff + 1
        rewards = notebook_rewards(t.eos_token)
    elif GPT_OSS:
        # gpt-oss notebook prompt shape (write a fast pure-Python kernel), 8 fixed variants.
        ops = ["matrix multiplication", "matrix transpose", "vector dot product", "prefix sum", "softmax",
               "layer norm", "2D convolution", "argmax"]
        rows = [{"prompt": [{"role": "user", "content": f"Create a new fast {op} function using only native "
                                                        f"Python code. Output the new function in backticks."}]}
                for op in ops]
        max_prompt = 96
        rewards = [smoke_reward]
    else:
        qs = ["Write a short poem about the sea.", "Explain gravity in one line.", "Name three colors.",
              "Say hello politely.", "Describe a cat.", "What is rain?", "Count to five.", "Greet the reader."]
        rows = [{"prompt": [{"role": "user", "content": q}]} for q in qs]
        if not getattr(t, "chat_template", None):
            rows = [{"prompt": q} for q in qs]
        max_prompt = 64
        rewards = [smoke_reward]

    max_completion = min(a.max_completion_length, a.max_seq_length - max_prompt)
    dropped = []
    ga = k.common_train_kwargs(a, out_dir)
    ga.update(per_device_train_batch_size=a.num_generations,
              optim="adamw_8bit" if a.preset in ("notebook", *P.GRPO) and a.backend != "hf" else "adamw_torch")
    if a.preset == "notebook":  # notebook GRPOConfig schedule
        # transformers 5 dropped warmup_ratio: float warmup_steps < 1 is the ratio
        import inspect
        from transformers import TrainingArguments
        ratio = "warmup_ratio" in inspect.signature(TrainingArguments).parameters
        ga.update(weight_decay=0.001, lr_scheduler_type="linear",
                  **({"warmup_ratio": 0.1, "warmup_steps": 0} if ratio else {"warmup_steps": 0.1}))
    cfg_kw = dict(num_generations=a.num_generations, max_prompt_length=max_prompt, max_completion_length=max_completion,
                  temperature=1.0, **ga)
    if a.fast_inference:
        from vllm import SamplingParams
        cfg_kw["vllm_sampling_params"] = SamplingParams(min_p=0.1, top_p=1.0, top_k=-1, seed=a.seed,
                                                        stop=[t.eos_token], include_stop_str_in_output=True)
    cfg = k.make(GRPOConfig, dropped, **cfg_kw)
    trainer = GRPOTrainer(model=model, reward_funcs=rewards, args=cfg, train_dataset=Dataset.from_list(rows),
                          callbacks=[c.metrics_callback(rec)], **k.tokenizer_kwarg(GRPOTrainer, tok))
    if GPT_OSS:
        rec.summary(moe=k.moe_info(model))
    rec.summary(preset=a.preset, dropped_config_args=dropped, grpo_rows=len(rows), max_prompt_length=max_prompt,
                max_completion_length=max_completion)

    if a.fast_inference:
        rec.check("vllm_engine_attached", hasattr(model, "vllm_engine"), "model.vllm_engine")
        rec.check("trainer_uses_vllm", bool(getattr(trainer.args, "use_vllm", False)) and getattr(trainer, "llm", None) is not None,
                  f"use_vllm={getattr(trainer.args, 'use_vllm', None)} llm={type(getattr(trainer, 'llm', None)).__name__}")

    before = c.adapter_fingerprint(model)
    out = trainer.train()
    after = c.adapter_fingerprint(model)
    rec.summary(train_runtime_s=out.metrics.get("train_runtime"), final_loss=out.training_loss)
    if gmm is not None:
        gmm.restore()
        rec.summary(grouped_mm_calls=gmm.calls)
        moe = rec.data["summary"]["moe"]
        if moe["grouped_mm_expected"] and (a.backend == "hf" or moe["unsloth_backend"] == "grouped_mm"):
            rec.check("grouped_mm_called", gmm.calls > 0, f"{gmm.calls} torch._grouped_mm calls; {moe}")

    rec.standard_training_checks()
    rec.backend_check()
    rec.check("adapter_changed", *c.adapter_changed(before, after))
    steps = rec.data["steps"]
    stds = [s.get("reward_std") for s in steps]
    if a.preset in ("smoke", *P.GRPO):
        # Per-step asserts of test_fast_inference.py.
        rec.check("reward_std_positive", all(x is not None and x > 0 for x in stds), stds)
        fz = [s.get("frac_reward_zero_std") for s in steps]
        rec.check("no_zero_std_groups", all(x in (None, 0, 0.0) for x in fz), fz)
    else:
        rec.check("reward_varies", any(x is not None and x > 0 for x in stds), stds)
    lens = [s.get("completions/mean_length", s.get("completion_length")) for s in steps]
    rec.check("completion_length_bounded", all(x is not None and 0 < x <= max_completion for x in lens), lens)
    kls = [s.get("kl") for s in steps if s.get("kl") is not None]
    if kls:  # unlogged kl (beta=0): no vacuous pass
        rec.check("kl_small", all(_finite_kl(x) and abs(x) < 1 for x in kls), kls)
    else:
        rec.summary(kl_small="not logged (beta=0)")
