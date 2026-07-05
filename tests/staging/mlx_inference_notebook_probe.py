import os
# Real Apple Silicon MLX validation for unsloth-zoo#855: the notebook INFERENCE
# contract. Supported notebooks run the standard CUDA-style pattern
#   inputs  = tokenizer.apply_chat_template(..., return_tensors="pt", return_dict=True).to(...)
#   outputs = model.generate(**inputs, max_new_tokens=...)
#   resp    = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:])
# On MLX the model/tokenizer are backed by mlx-lm, so #855 makes generate() return
# a batched torch id tensor and apply_chat_template(return_dict=True) a BatchEncoding.
# This probe exercises that path end to end on REAL mlx and hard-fails (exit != 0)
# if any part of the contract regresses. No simulation shim.
print('=== MLX NOTEBOOK INFERENCE CONTRACT PROBE (zoo#855) ===')

from unsloth import FastModel

model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-0.5B-Instruct",
    max_seq_length = 2048,
    load_in_4bit = False,
    load_in_8bit = False,
    full_finetuning = False,
)

messages = [{"role": "user", "content": "What is 1+1? Reply with just the number."}]

# 1) apply_chat_template(return_tensors="pt", return_dict=True) must return a
#    mapping that expands via ** and moves via .to(...) -- the notebook shape.
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True,
    tokenize = True,
    return_tensors = "pt",
    return_dict = True,
)
assert hasattr(inputs, "keys"), f"apply_chat_template(return_dict=True) not a mapping: {type(inputs)}"
assert "input_ids" in inputs, f"missing input_ids: {list(inputs.keys())}"
inputs = inputs.to("cpu")  # notebooks call .to("cuda"); must exist and not crash on MLX
prompt_len = int(inputs["input_ids"].shape[-1])
print("input_ids shape:", tuple(inputs["input_ids"].shape))

# 2) generate(**inputs) must accept the expanded mapping.
out = model.generate(**inputs, max_new_tokens = 8)

# 3) output must be a Transformers-friendly batched id container (not a raw mx.array).
import torch
from transformers.tokenization_utils_base import to_py_obj
assert isinstance(out, torch.Tensor), f"generate() did not return a torch tensor: {type(out)}"
assert out.dtype == torch.long, out.dtype
assert out.dim() == 2 and out.shape[0] == 1, f"expected (1, N), got {tuple(out.shape)}"
assert out.shape[-1] >= prompt_len, f"output {tuple(out.shape)} shorter than prompt {prompt_len}"
assert to_py_obj(out) == out.tolist(), "transformers.to_py_obj failed on generate() output"

# 4) notebook slicing + batch_decode of only the newly generated ids.
generated = out[:, prompt_len:]
decoded = tokenizer.batch_decode(generated, skip_special_tokens = True)
assert isinstance(decoded, list) and len(decoded) == 1, decoded
print("prompt_len:", prompt_len, "| total:", int(out.shape[-1]), "| new tokens:", int(generated.shape[-1]))
print("decoded generated text:", repr(decoded[0]))

print('=== MLX INFERENCE CONTRACT PROBE OK ===')
