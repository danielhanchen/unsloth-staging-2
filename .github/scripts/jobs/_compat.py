"""Signature-filtered kwargs for the torch jobs across transformers 4.57.6..5.x and TRL
0.22.2..1.x. Import after unsloth.
"""

import dataclasses
import inspect

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def accepted(cls):
    """Params cls(...) accepts; None if it takes **kwargs."""
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
    """processing_class (TRL >= 0.12) or tokenizer (older, some Unsloth wrappers)."""
    params = inspect.signature(trainer_cls.__init__).parameters
    return {"processing_class": tok} if "processing_class" in params else {"tokenizer": tok}


def backend_name(a, device):
    """From sys.modules, not --backend, so backend_matches_request can actually fail."""
    import sys
    return ("unsloth-" if "unsloth" in sys.modules else "hf-") + device


def load_text_model(a, lora_r=16, lora_alpha=16, load_in_4bit=False, **unsloth_kw):
    """(model, tokenizer) + LoRA via Unsloth or transformers+PEFT."""
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
    """Deterministic offline chat rows."""
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


def vision_dataset(rows):
    """rows -> datasets.Dataset with the image column typed as datasets.Image, so rows read back as
    decoded PIL images. Plain Dataset.from_list infers {"bytes", "path"} structs, which neither the
    HF processor nor a chat template accepts (TRL 1.x also rejects a plain list)."""
    from datasets import Dataset, Features, Image, Value
    item = {"type": Value("string"), "image": Image(), "text": Value("string")}
    return Dataset.from_list(rows, features=Features(
        {"messages": [{"role": Value("string"), "content": [item]}]}))


def strip_none(example):
    """Arrow gives every content item every key: drop the None fillers ({"type": "text", "image":
    None}) so collators and chat templates see the notebook-shaped messages."""
    return {**example, "messages": [
        {**m, "content": [{k: v for k, v in x.items() if v is not None} for x in m["content"]]}
        for m in example["messages"]]}


class GroupedMMCounter:
    """Counts torch._grouped_mm calls (torch.nn.functional.grouped_mm routes through it). Install
    before the model's first forward; a Python wrapper, so compiled graphs that inline the aten op
    directly are not counted (a 0 there is "not observed", not "not used")."""

    def __init__(self):
        import torch
        self.calls, self._orig = 0, getattr(torch, "_grouped_mm", None)
        if self._orig is not None:
            def counted(*args, **kw):
                self.calls += 1
                return self._orig(*args, **kw)
            torch._grouped_mm = counted

    def restore(self):
        import torch
        if self._orig is not None:
            torch._grouped_mm = self._orig


def moe_info(model=None):
    """MoE backend facts: requested env, Unsloth's selection, transformers' experts implementation,
    device capability, and whether grouped_mm is expected to run (CUDA sm >= 8.0)."""
    import os
    import torch
    info = {"UNSLOTH_MOE_BACKEND": os.environ.get("UNSLOTH_MOE_BACKEND"), "unsloth_backend": None,
            "experts_implementation": None, "capability": None}
    import sys
    if "unsloth" in sys.modules:  # never import Unsloth into the plain-HF reference arm
        try:
            from unsloth_zoo.temporary_patches.moe_utils import select_moe_backend
            info["unsloth_backend"] = select_moe_backend()
        except Exception as e:
            info["unsloth_backend"] = f"unavailable: {type(e).__name__}"
    cfg = getattr(model, "config", None)
    info["experts_implementation"] = getattr(cfg, "_experts_implementation", None) if cfg else None
    if torch.cuda.is_available() and not getattr(torch.version, "hip", None):
        info["capability"] = ".".join(map(str, torch.cuda.get_device_capability()))
    info["grouped_mm_expected"] = bool(info["capability"]) and tuple(
        int(x) for x in info["capability"].split(".")) >= (8, 0)
    return info


def lora_param_names(model, needle):
    """Trainable (LoRA) parameter names containing needle, e.g. "experts"."""
    return [n for n, p in model.named_parameters() if p.requires_grad and needle in n]


def _bound_impl(fn):
    """Implementation a transformers `use_kernel_func_from_hub_with_fallback` wrapper resolved at
    import time (fla / hub kernel / the torch reference itself)."""
    import inspect
    try:
        impl = inspect.getclosurevars(fn).nonlocals.get("implementation", fn)
    except (TypeError, ValueError):
        impl = fn
    return f"{getattr(impl, '__module__', '?')}.{getattr(impl, '__name__', type(impl).__name__)}"


def gdn_info(model):
    """Which chunk_gated_delta_rule the Gated DeltaNet layers call: resolved from the modeling module
    (and Unsloth's compiled copy, if any) that defines each GatedDeltaNet class."""
    import sys
    mods = [m for m in model.modules() if type(m).__name__.endswith("GatedDeltaNet")]
    impls = {}
    for m in mods:
        src = sys.modules.get(type(m).__module__)
        fn = getattr(src, "torch_chunk_gated_delta_rule", None) or getattr(src, "chunk_gated_delta_rule", None)
        if fn is not None:
            impls[type(m).__module__] = _bound_impl(fn)
    fla = sys.modules.get("fla")
    return {"gdn_modules": len(mods), "gdn_impls": impls, "fla_module": getattr(fla, "__file__", None),
            "fla_used": any(v.startswith("fla.") for v in impls.values())}
