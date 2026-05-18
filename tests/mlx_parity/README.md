# MLX vs HF parity probes

Seven small, focused probes designed to bisect the parity gap between MLX
training (via `unsloth_zoo.mlx.trainer`) and HF training (via
`transformers.SFTTrainer`) on the same hyperparameters.

Symptom: identical 7-step LoRA fine-tune of `unsloth/gemma-3-270m-it` on the
single row `"<<HELLO!!>> My name is Unsloth!"` produces:

| | step-1 loss | post-train loss | greedy generation |
|---|---|---|---|
| HF SFTTrainer (CUDA bf16) | 7.64 | 0.001 | `"... Unsloth! My personality is bubbly ..."` |
| MLX trainer | 10.55 | 0.009 | `"5 lbs!"` |

The 1.38x step-1 forward-pass gap is the root anomaly. Each probe answers
one question along the dispatch path:

| # | probe | question |
|---|---|---|
| 1 | `probe_1_tokenization.py` | does the tokenized input differ? |
| 2 | `probe_2_forward_logits.py` | does the base model emit different logits? |
| 3 | `probe_3_loss_reduction.py` | does CE-then-reduce produce different scalars? |
| 4 | `probe_4_lora_init.py` | does LoRA init produce different magnitudes? |
| 5 | `probe_5_single_grad.py` | does one backward produce different gradients? |
| 6 | `probe_6_adamw_step.py` | does one AdamW step produce different deltas? |
| 7 | `probe_7_loss_curve.py` | what does the 7-step curve look like end-to-end? |

Each probe prints diagnostic data, then asserts a numeric tolerance. The
workflow runs them with `continue-on-error: true` so even a single
diverging probe still prints subsequent diagnostic data.
