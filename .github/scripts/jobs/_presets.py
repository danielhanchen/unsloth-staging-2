"""Notebook presets for the jobs (torch-free, so tests can import them).

A preset = the notebook's model + trainer settings, capped for a few-step regression run.
`resolve(presets, name, tiny, model)` fills the model: explicit --model wins, then the preset's
tiny / full model.
"""

# unslothai/notebooks nb/*.ipynb each preset mirrors.
VISION_SFT = {
    "default": dict(notebook="Gemma3_(4B)-Vision", model="unsloth/Qwen2.5-VL-3B-Instruct",
                    tiny="trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",
                    lora_r=16, lora_alpha=16, target_modules="all-linear", batch=2, grad_accum=1, lr=2e-4,
                    max_seq_length=1024),
    "gemma4": dict(notebook="Gemma4_(E2B)-Vision", model="unsloth/gemma-4-E2B-it",
                   tiny="trl-internal-testing/tiny-Gemma4ForConditionalGeneration",
                   lora_r=32, lora_alpha=32, target_modules="all-linear", batch=1, grad_accum=4, lr=2e-4,
                   max_seq_length=2048),
    # Gated DeltaNet (linear attention) layers: exercises the fla chunk_gated_delta_rule kernels.
    # TRL tests the NoThink tiny variant (full_attention_interval 2).
    "qwen3_5": dict(notebook="Qwen3_5_(0_8B)_Vision", model="unsloth/Qwen3.5-0.8B",
                    tiny="trl-internal-testing/tiny-Qwen3_5ForConditionalGeneration-NoThink",
                    lora_r=16, lora_alpha=16, target_modules=None, batch=2, grad_accum=4, lr=2e-4,
                    max_seq_length=2048),
}

GRPO = {
    # gpt-oss-(20B)-GRPO recipe on the tiny gpt-oss (MoE experts -> torch._grouped_mm, torch.compile
    # paths). Deterministic length + tie-break reward: a random tiny model never writes a kernel.
    "gpt_oss_tiny": dict(notebook="gpt-oss-(20B)-GRPO", model="trl-internal-testing/tiny-GptOssForCausalLM",
                         tiny="trl-internal-testing/tiny-GptOssForCausalLM", lora_r=4, lora_alpha=8,
                         num_generations=2, max_completion_length=16, lr=5e-5, max_seq_length=256),
}

MOE_SFT = {
    "qwen3_moe_tiny": dict(model="trl-internal-testing/tiny-Qwen3MoeForCausalLM",
                           tiny="trl-internal-testing/tiny-Qwen3MoeForCausalLM", lora_r=8, lora_alpha=16),
}

AUDIO_SFT = {
    "gemma4": dict(notebook="Gemma4_(E2B)-Audio", model="unsloth/gemma-4-E2B-it",
                   # tiny-Gemma4 has audio_config null (no audio tower): --tiny only checks the plumbing.
                   tiny="trl-internal-testing/tiny-Gemma4ForConditionalGeneration",
                   lora_r=8, lora_alpha=16, lr=5e-5, batch=2, max_seq_length=2048,
                   target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                                   "post", "linear_start", "linear_end", "embedding_projection",
                                   "ffw_layer_1", "ffw_layer_2", "output_proj"]),
}

VISION_GRPO = {
    # Gemma3_(4B)-Vision-GRPO recipe with the model switched to Gemma 4 E2B.
    "gemma4": dict(notebook="Gemma3_(4B)-Vision-GRPO", model="unsloth/gemma-4-E2B-it",
                   tiny="trl-internal-testing/tiny-Gemma4ForConditionalGeneration",
                   lora_r=16, lora_alpha=16, lr=5e-6, num_generations=2, max_completion_length=32,
                   max_prompt_length=1024, image_size=512, loss_type="dr_grpo",
                   importance_sampling_level="sequence", max_grad_norm=0.1),
}


def resolve(presets, name, tiny=False, model=None):
    """(preset dict copy, model id). Unknown name -> KeyError listing the choices."""
    if name not in presets:
        raise KeyError(f"unknown preset {name!r}; choose from {sorted(presets)}")
    p = dict(presets[name])
    return p, model or (p["tiny"] if tiny else p["model"])


def model_given(argv):
    """True if --model was passed explicitly (argparse fills a default otherwise)."""
    return any(x == "--model" or x.startswith("--model=") for x in argv)
