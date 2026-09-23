"""Version-tolerant helpers shared by the torch jobs (sft, grpo, dpo, vision_sft, inference_smoke).

transformers 4.57.6..5.x and TRL 0.22.2..1.x rename and drop arguments; everything here filters
kwargs against the live signature instead of pinning one API. Import only after unsloth.
"""

import dataclasses
import inspect

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def accepted(cls):
    """Names cls(...) accepts, or None when it takes **kwargs (then keep everything)."""
    if dataclasses.is_dataclass(cls):
        return {f.name for f in dataclasses.fields(cls)}
    params = inspect.signature(cls).parameters
    if any(p.kind is p.VAR_KEYWORD for p in params.values()):
        return None
    return set(params)


def filter_kwargs(cls, kw, dropped=None):
    ok = accepted(cls)
    if ok is None:
        return dict(kw)
    out = {k: v for k, v in kw.items() if k in ok}
    if dropped is not None:
        dropped.extend(sorted(set(kw) - set(out)))
    return out


def make(cls, dropped=None, **kw):
    return cls(**filter_kwargs(cls, kw, dropped))


def tokenizer_kwarg(trainer_cls, tok):
    """TRL >= 0.12 takes processing_class; older (and some Unsloth wrappers) take tokenizer."""
    params = inspect.signature(trainer_cls.__init__).parameters
    return {"processing_class": tok} if "processing_class" in params else {"tokenizer": tok}


def backend_name(a, device):
    """From what actually got imported, not from --backend, so backend_matches_request can fail
    (e.g. --backend hf that still pulled in unsloth, or an unsloth run that never imported it)."""
    import sys
    return ("unsloth-" if "unsloth" in sys.modules else "hf-") + device


def load_text_model(a, lora_r=16, lora_alpha=16, load_in_4bit=False, **unsloth_kw):
    """(model, tokenizer) with LoRA attached, via Unsloth or plain transformers + PEFT."""
    import torch
    if a.backend != "hf":
        from unsloth import FastLanguageModel
        model, tok = FastLanguageModel.from_pretrained(
            model_name=a.model, max_seq_length=a.max_seq_length, load_in_4bit=load_in_4bit,
            dtype=None, **unsloth_kw)
        model = FastLanguageModel.get_peft_model(
            model, r=lora_r, lora_alpha=lora_alpha, lora_dropout=0, target_modules=LORA_TARGETS,
            use_gradient_checkpointing="unsloth", random_state=a.seed)
        return model, tok
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if dev == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dtype).to(dev)
    present = {n.split(".")[-1] for n, _ in model.named_modules()}
    model = get_peft_model(model, LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.0,
                                             target_modules=[t for t in LORA_TARGETS if t in present],
                                             task_type="CAUSAL_LM"))
    return model, tok


def ensure_pad(tok):
    t = getattr(tok, "tokenizer", tok)
    if t.pad_token is None:
        t.pad_token = t.eos_token
    return tok


def precision_kwargs():
    import torch
    if torch.cuda.is_available():
        bf16 = torch.cuda.is_bf16_supported()
        return {"bf16": bf16, "fp16": not bf16}
    return {"bf16": False, "fp16": False}


def common_train_kwargs(a, out_dir):
    return dict(output_dir=out_dir, max_steps=a.max_steps, per_device_train_batch_size=2,
                gradient_accumulation_steps=1, learning_rate=a.lr, warmup_steps=0,
                lr_scheduler_type="constant", logging_steps=1, save_strategy="no", seed=a.seed,
                report_to="none", optim="adamw_torch", weight_decay=0.0, max_grad_norm=1.0,
                include_num_input_tokens_seen=True, dataloader_num_workers=0,
                **precision_kwargs())


def chat_fixture(n=64):
    """Deterministic conversational rows; no network, same data for base and head."""
    facts = [("capital of France", "Paris"), ("2 + 2", "4"), ("color of the sky", "blue"),
             ("opposite of hot", "cold"), ("largest planet", "Jupiter"), ("H2O", "water"),
             ("first letter of the alphabet", "A"), ("number of legs on a spider", "8")]
    rows = []
    for i in range(n):
        q, ans = facts[i % len(facts)]
        rows.append({"messages": [{"role": "user", "content": f"What is the {q}? (#{i})"},
                                  {"role": "assistant", "content": f"The {q} is {ans}."}]})
    return rows


def render_chat(tok, rows):
    t = getattr(tok, "tokenizer", tok)
    if getattr(t, "chat_template", None):
        return [{"text": t.apply_chat_template(r["messages"], tokenize=False)} for r in rows]
    return [{"text": "\n".join(f"{m['role']}: {m['content']}" for m in r["messages"]) + (t.eos_token or "")}
            for r in rows]
