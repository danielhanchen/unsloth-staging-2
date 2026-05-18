"""Probe 1 — tokenization parity.

Compare two ways of tokenizing the same training text:

  (a) HF SFTTrainer path:        tokenizer(TRAIN_TEXT, return_tensors=...)
  (b) MLX trainer path:          tokenizer.encode(TRAIN_TEXT); maybe append EOS

Difference in token IDs / length here would explain a different per-token
denominator and thus a different reported scalar loss, even with identical
math downstream.

Exits 0 on parity, 2 on divergence (with diagnostic printout).
"""

import json
import sys

from _common import MODEL_NAME, TRAIN_TEXT, OUT_DIR, banner, section, report, seed_everything


def main() -> int:
    seed_everything()
    banner("Probe 1: tokenization parity")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    report("tokenizer class", type(tok).__name__)
    report("vocab_size", tok.vocab_size)
    report("bos_token_id", tok.bos_token_id)
    report("eos_token_id", tok.eos_token_id)
    report("pad_token_id", tok.pad_token_id)

    section("(a) HF SFTTrainer path: tokenizer(TRAIN_TEXT)")
    hf_enc = tok(TRAIN_TEXT, add_special_tokens=True)
    hf_ids = list(hf_enc["input_ids"])
    report("input_ids", hf_ids)
    report("len", len(hf_ids))
    report("first/last id", (hf_ids[0], hf_ids[-1]))
    report("decoded", repr(tok.decode(hf_ids)))

    section("(b) MLX trainer path: tokenizer.encode + EOS append")
    mlx_ids = tok.encode(TRAIN_TEXT)
    if tok.eos_token_id is not None and (not mlx_ids or mlx_ids[-1] != tok.eos_token_id):
        mlx_ids.append(tok.eos_token_id)
    report("input_ids", mlx_ids)
    report("len", len(mlx_ids))
    report("first/last id", (mlx_ids[0], mlx_ids[-1]))
    report("decoded", repr(tok.decode(mlx_ids)))

    section("comparison")
    same = hf_ids == mlx_ids
    delta_len = len(mlx_ids) - len(hf_ids)
    report("identical id list", same)
    report("len_mlx - len_hf", delta_len)
    if not same:
        only_a = [i for i in hf_ids if i not in mlx_ids]
        only_b = [i for i in mlx_ids if i not in hf_ids]
        report("ids only in HF path", only_a)
        report("ids only in MLX path", only_b)

    out = {
        "hf_ids": hf_ids,
        "mlx_ids": mlx_ids,
        "delta_len": delta_len,
        "identical": same,
    }
    (OUT_DIR / "probe_1.json").write_text(json.dumps(out, indent=2))
    return 0 if same else 2


if __name__ == "__main__":
    sys.exit(main())
