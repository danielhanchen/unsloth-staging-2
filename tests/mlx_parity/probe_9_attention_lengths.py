"""Probe 9 — attention mask / lengths inspection.

HF SFTTrainer's default collator and MLX trainer's create_batches both
build a (batch, lengths_or_mask) representation. Their masking
semantics may differ in subtle ways:

  HF SFTTrainer:
    * attention_mask is a (B, L) 0/1 tensor; 0 marks padding tokens.
    * labels = input_ids with padding positions set to -100.
    * loss is reduced over labels != -100.

  MLX trainer (unsloth_zoo.mlx):
    * batch is (B, L) padded with 0.
    * lengths is (B, 2) of [start, end] = [1, L-1] for this dataset
      (see trainer.py around batch_lengths.append([1, L-1])).
    * labels mirror input_ids with [-100]*pad_len trailing.
    * loss mask = (targets != -100) AND length_mask(start, end).

This probe enumerates what tokens are actually being supervised in
each case for our specific train row and confirms the two paths
supervise the SAME positional set.
"""

import json
import sys

import numpy as np

from _common import MODEL_NAME, TRAIN_TEXT, OUT_DIR, banner, section, report, seed_everything


def main() -> int:
    seed_everything()
    banner("Probe 9: attention mask / lengths inspection")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    ids = tok.encode(TRAIN_TEXT)
    if tok.eos_token_id is not None and ids[-1] != tok.eos_token_id:
        ids.append(tok.eos_token_id)
    L = len(ids)
    report("token_ids", ids)
    report("len", L)

    section("HF SFTTrainer style supervision mask")
    # No padding here (batch of 1, length L) -> attention_mask is all 1s,
    # labels mirror ids, all positions are supervised after shift.
    attn = [1] * L
    labels = list(ids)
    shifted_labels = labels[1:]
    hf_supervised_positions = list(range(L - 1))
    hf_supervised_tokens = [tok.decode([t]) for t in shifted_labels]
    report("attention_mask", attn)
    report("shifted target ids", shifted_labels)
    report("supervised positions (post-shift)", hf_supervised_positions)

    section("MLX trainer style supervision mask")
    # Mirrors the path in unsloth_zoo/mlx/trainer.py:
    #   batch_lengths.append([1, L - 1])
    #   length_mask = (steps >= lengths[:,0]) AND (steps <= lengths[:,1])
    #   steps = mx.arange(1, targets.shape[1] + 1) == [1..L-1]
    # so length_mask is TRUE for steps in [1, L-1], i.e. all post-shift
    # positions for our unpadded batch.
    lengths_pair = [1, L - 1]
    steps = list(range(1, L))  # = [1..L-1]
    length_mask = [(s >= lengths_pair[0]) and (s <= lengths_pair[1]) for s in steps]
    targets_mlx = labels[1:]
    mask_neg100 = [t != -100 for t in targets_mlx]
    combined_mask = [a and b for a, b in zip(length_mask, mask_neg100)]
    mlx_supervised_positions = [i for i, m in enumerate(combined_mask) if m]
    mlx_supervised_tokens = [tok.decode([targets_mlx[i]]) for i in mlx_supervised_positions]
    report("lengths_pair", lengths_pair)
    report("steps", steps)
    report("length_mask", length_mask)
    report("supervised positions (post-shift)", mlx_supervised_positions)

    section("comparison")
    matches = hf_supervised_positions == mlx_supervised_positions
    report("supervised positions match", matches)
    report("hf supervises N tokens", len(hf_supervised_positions))
    report("mlx supervises N tokens", len(mlx_supervised_positions))
    only_hf = set(hf_supervised_positions) - set(mlx_supervised_positions)
    only_mlx = set(mlx_supervised_positions) - set(hf_supervised_positions)
    if only_hf:
        report("only supervised by HF", list(only_hf))
    if only_mlx:
        report("only supervised by MLX", list(only_mlx))

    out = {
        "token_ids": ids,
        "hf_supervised_positions": hf_supervised_positions,
        "mlx_supervised_positions": mlx_supervised_positions,
        "match": matches,
        "n_supervised_hf": len(hf_supervised_positions),
        "n_supervised_mlx": len(mlx_supervised_positions),
        "lengths_pair": lengths_pair,
    }
    (OUT_DIR / "probe_9.json").write_text(json.dumps(out, indent=2))
    return 0 if matches else 2


if __name__ == "__main__":
    sys.exit(main())
