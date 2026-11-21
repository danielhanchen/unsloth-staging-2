"""Weight-layout fixes a baseline-UI file built from an official repo needs, kept next to the cards that cite them.

h3_qkv_regroup: MiniMax-H3's released ``blocks.N.attn.qkv_proj.weight`` runs PER HEAD ([q; k; v] of head 0, then head
1, ...: the diffusers to_k row 0 is row 128), while BlineUI's Attention splits qkv_proj by THIRDS ([all q; all k; all
v]). A BlineUI file converted from the release without this regroup loads without error and renders noise (trap
h3_qkv_regroup in models/minimax-h3*.json). Bline-Org's published files already carry the thirds layout. From
scripts/vidcmp_nov5/build_bline.py (h3_dit) and vscv_bline_hook ``qkv_per_head_to_thirds``; every H3 DiT format
built there (bf16, fp8_scaled, int8, nvfp4) is regrouped before quantisation.

Works on torch tensors and numpy arrays (anything with reshape / transpose-by-axes).
"""

from __future__ import annotations

H3_HEAD_DIM = 128
H3_QKV_SUFFIX = ".attn.qkv_proj.weight"


def h3_qkv_regroup(w, head_dim: int = H3_HEAD_DIM):
    """Per-head [q_h; k_h; v_h] rows -> [q; k; v] thirds. ``w`` is [3 * heads * head_dim, in_features]."""
    rows, cols = w.shape[0], w.shape[-1]
    if rows % (3 * head_dim):
        raise ValueError(f"qkv rows {rows} are not a multiple of 3 * head_dim ({3 * head_dim})")
    x = w.reshape(-1, 3, head_dim, cols)
    x = x.transpose(0, 1) if hasattr(x, "contiguous") else x.transpose(1, 0, 2, 3)
    x = x.reshape(rows, cols)
    return x.contiguous() if hasattr(x, "contiguous") else x


def h3_qkv_ungroup(w, head_dim: int = H3_HEAD_DIM):
    """The inverse: thirds -> per-head rows (to check a file's layout: ungroup(regroup(w)) == w)."""
    rows, cols = w.shape[0], w.shape[-1]
    x = w.reshape(3, -1, head_dim, cols)
    x = x.transpose(0, 1) if hasattr(x, "contiguous") else x.transpose(1, 0, 2, 3)
    x = x.reshape(rows, cols)
    return x.contiguous() if hasattr(x, "contiguous") else x


def h3_regroup_state_dict(sd: dict, head_dim: int = H3_HEAD_DIM) -> int:
    """Regroup every H3 qkv_proj weight of a released-layout state dict in place; returns how many were changed."""
    n = 0
    for k in [k for k in sd if k.endswith(H3_QKV_SUFFIX)]:
        sd[k] = h3_qkv_regroup(sd[k], head_dim)
        n += 1
    return n
