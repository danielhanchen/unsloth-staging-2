"""Qwen3.5-2B-4bit: mlx_lm vs mlx_vlm / Unsloth text_only (CE 0.816 vs 12.65 on mlx-vlm 0.6.4).

    python jobs/mlx_diag_qwen35.py [--model M] [--out diag.json] [--no-unsloth] [--mlx-vlm V]

Isolated probes (failures recorded, never raised): versions, config, param names/shapes/values,
raw vs loaded norms, per-layer hidden-state divergence, explicit text positions, Unsloth load.
Prints `DIAG {json}`; always exits 0.
"""

from __future__ import annotations

import argparse
import sys
import glob
import importlib.metadata as md
import json
import os
import traceback

TEXT = ("The capital of France is Paris. The capital of Germany is Berlin. "
        "The capital of Italy is Rome. Water boils at 100 degrees Celsius at sea level. ") * 2


def _ver(p):
    try:
        return md.version(p)
    except md.PackageNotFoundError:
        return None


def _guard(res, key, fn):
    try:
        res[key] = fn()
    except Exception as e:  # noqa: BLE001 - a diagnostic records every failure and moves on
        res[key] = {"error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-1500:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3.5-2B-4bit")
    ap.add_argument("--out", default="mlx_diag_qwen35.json")
    ap.add_argument("--no-unsloth", action="store_true")
    ap.add_argument("--tiny", action="store_true", help="ignored (staging_ci)")
    ap.add_argument("--mlx-vlm", default=None, metavar="VERSION",
                    help="pip install this mlx-vlm first, then re-exec; mutates the env, so last job")
    a = ap.parse_args()
    if a.mlx_vlm and _ver("mlx-vlm") != a.mlx_vlm:
        import subprocess
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", f"mlx-vlm=={a.mlx_vlm}"])
        print(f"DIAG_INSTALL mlx-vlm=={a.mlx_vlm} rc={r.returncode} "
              f"transformers={_ver('transformers')}", flush=True)
        argv = [x for x in sys.argv[1:] if x != "--mlx-vlm" and x != a.mlx_vlm]
        os.execv(sys.executable, [sys.executable, __file__, *argv, "--out",
                                  a.out.replace(".json", f"_vlm{a.mlx_vlm}.json")])

    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    res = {"model": a.model, "versions": {p: _ver(p) for p in
                                          ("mlx", "mlx-lm", "mlx-vlm", "unsloth", "unsloth_zoo",
                                           "transformers")}}

    def ce(logits, ids):
        logits = getattr(logits, "logits", logits)
        return round(nn.losses.cross_entropy(logits[:, :-1].astype(mx.float32), ids[:, 1:],
                                             reduction="mean").item(), 4)

    def fdiff(x, y):
        x, y = x.astype(mx.float32), y.astype(mx.float32)
        d = mx.abs(x - y).max().item()
        return d, d / max(mx.abs(x).max().item(), 1e-12)

    from huggingface_hub import snapshot_download
    path = snapshot_download(a.model)
    cfg = json.load(open(os.path.join(path, "config.json")))
    tc = cfg.get("text_config", cfg)
    res["config"] = {"model_type": cfg.get("model_type"), "architectures": cfg.get("architectures"),
                     "vision_config": "vision_config" in cfg, "tie": tc.get("tie_word_embeddings"),
                     "rope_parameters": tc.get("rope_parameters"),
                     "layer_types": tc.get("layer_types"), "quantization": cfg.get("quantization")}

    from mlx_lm import load as lm_load
    from mlx_vlm import load as vlm_load
    m_lm, tok = lm_load(path)
    m_vlm, _proc = vlm_load(path)
    ids = mx.array([tok.encode(TEXT)])
    res["n_tokens"] = int(ids.shape[1])
    lm_text = m_lm.language_model if hasattr(m_lm, "language_model") else m_lm
    vlm_text = m_vlm.language_model
    res["classes"] = {"mlx_lm": f"{type(m_lm).__module__}.{type(m_lm).__name__}",
                      "mlx_vlm_text": f"{type(vlm_text).__module__}.{type(vlm_text).__name__}"}

    # 1. CE per path.
    ces = {}
    _guard(ces, "mlx_lm", lambda: ce(m_lm(ids), ids))
    _guard(ces, "mlx_vlm_language_model", lambda: ce(vlm_text(ids), ids))
    _guard(ces, "mlx_vlm_full_model", lambda: ce(m_vlm(ids), ids))

    def _pos():
        L = ids.shape[1]
        pos = mx.tile(mx.arange(L)[None, None, :], (3, 1, 1))
        vlm_text._position_ids, vlm_text._rope_deltas = None, None
        return ce(vlm_text(ids, position_ids=pos), ids)
    _guard(ces, "mlx_vlm_explicit_text_positions", _pos)
    res["ce"] = ces

    # 2. Parameters: same names, shapes, values?
    def _params():
        p_lm = {k.removeprefix("language_model."): v for k, v in tree_flatten(lm_text.parameters())}
        p_vl = {k: v for k, v in tree_flatten(vlm_text.parameters())}
        only_lm, only_vl = sorted(set(p_lm) - set(p_vl)), sorted(set(p_vl) - set(p_lm))
        diffs, shape_mismatch = [], []
        for k in sorted(set(p_lm) & set(p_vl)):
            x, y = p_lm[k], p_vl[k]
            if x.shape != y.shape or x.dtype != y.dtype:
                shape_mismatch.append([k, list(x.shape), str(x.dtype), list(y.shape), str(y.dtype)])
                continue
            if x.dtype in (mx.uint32, mx.int32, mx.uint8):
                if not mx.array_equal(x, y).item():
                    diffs.append([k, "packed-differs", None])
                continue
            d, rel = fdiff(x, y)
            if d > 0:
                diffs.append([k, d, rel])
        diffs.sort(key=lambda r: -(r[1] if isinstance(r[1], float) else 1e9))
        return {"n_lm": len(p_lm), "n_vlm": len(p_vl), "only_mlx_lm": only_lm[:40],
                "only_mlx_vlm": only_vl[:40], "n_only_lm": len(only_lm), "n_only_vlm": len(only_vl),
                "shape_or_dtype_mismatch": shape_mismatch[:40], "n_value_diffs": len(diffs),
                "value_diffs_top": diffs[:40]}
    _guard(res, "params", _params)

    # 2b. Raw vs loaded norms (double +1 RMSNorm offset).
    def _norms():
        raw = {}
        for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
            for k, v in mx.load(f).items():
                if k.endswith("layers.0.input_layernorm.weight") or k.endswith("layers.3.self_attn.q_norm.weight") \
                        or k.endswith("model.norm.weight") or k.endswith("layers.0.linear_attn.norm.weight"):
                    raw[k] = v
        p_lm = dict(tree_flatten(lm_text.parameters()))
        p_vl = dict(tree_flatten(vlm_text.parameters()))
        out = {}
        for k, v in raw.items():
            short = k.split("language_model.", 1)[-1]
            row = {"raw_mean": round(v.astype(mx.float32).mean().item(), 4)}
            for name, p in (("mlx_lm", p_lm), ("mlx_vlm", p_vl)):
                hit = next((pv for pk, pv in p.items() if pk.endswith(short)), None)
                row[name] = None if hit is None else round(hit.astype(mx.float32).mean().item(), 4)
            out[k] = row
        return out
    _guard(res, "norms", _norms)

    # 3. Per-layer hidden states, same batch.
    def _record(text_model, store):
        layers = text_model.model.layers
        pos = {id(layer): i for i, layer in enumerate(layers)}  # nn.Module is a dict: key by id
        patched = []
        for cls in {type(layer) for layer in layers}:
            orig = cls.__call__

            def rec(self, *args, _orig=orig, **kw):
                out = _orig(self, *args, **kw)
                store.append((pos.get(id(self), -1),
                              getattr(self, "is_linear", None), out))
                return out
            cls.__call__ = rec
            patched.append((cls, orig))
        return patched

    def _layers():
        h_lm, h_vl = [], []
        emb_d = fdiff(lm_text.model.embed_tokens(ids), vlm_text.model.embed_tokens(ids))
        for model, store in ((lm_text, h_lm), (vlm_text, h_vl)):
            patched = _record(model, store)
            try:
                if model is vlm_text:
                    vlm_text._position_ids, vlm_text._rope_deltas = None, None
                out = model(ids)
                mx.eval(getattr(out, "logits", out))
            finally:
                for cls, orig in patched:
                    cls.__call__ = orig
        rows, first = [], None
        for (i1, lin1, o1), (i2, lin2, o2) in zip(h_lm, h_vl):
            o1, o2 = (o[0] if isinstance(o, tuple) else o for o in (o1, o2))
            d, rel = fdiff(o1, o2)
            rows.append({"layer": i1, "vlm_layer": i2, "is_linear_lm": lin1, "is_linear_vlm": lin2,
                         "max_abs": round(d, 5), "rel": round(rel, 5)})
            if first is None and rel > 1e-2:
                first = i1
        return {"embed_max_abs": round(emb_d[0], 6), "n_layers_lm": len(h_lm),
                "n_layers_vlm": len(h_vl), "first_diverging_layer": first, "rows": rows}
    _guard(res, "hidden", _layers)

    # 4. Unsloth text_only (routes to mlx-vlm today).
    if not a.no_unsloth:
        def _unsloth():
            import unsloth  # noqa: F401
            from unsloth import FastLanguageModel
            m_u, tok_u = FastLanguageModel.from_pretrained(a.model, max_seq_length=512, text_only=True)
            u_ids = mx.array([tok_u.encode(TEXT)])
            return {"class": f"{type(m_u).__module__}.{type(m_u).__name__}",
                    "text_only_vlm_flag": bool(getattr(m_u, "_unsloth_text_only_vlm", False)),
                    "ce": ce(m_u(u_ids), u_ids)}
        _guard(res, "unsloth_text_only", _unsloth)

    with open(a.out, "w") as f:
        json.dump(res, f, indent=2, default=str)
    print("DIAG " + json.dumps(res, default=str), flush=True)
    print("\n=== summary ===")
    print("versions", res["versions"])
    print("ce", res.get("ce"))
    p = res.get("params", {})
    print("params: only_lm", p.get("n_only_lm"), "only_vlm", p.get("n_only_vlm"),
          "shape_mismatch", len(p.get("shape_or_dtype_mismatch", []) or []), "value_diffs",
          p.get("n_value_diffs"))
    print("norms", res.get("norms"))
    h = res.get("hidden", {})
    print("embed diff", h.get("embed_max_abs"), "first diverging layer", h.get("first_diverging_layer"))
    for r in (h.get("rows") or [])[:8]:
        print("  ", r)
    print("unsloth", res.get("unsloth_text_only"))


if __name__ == "__main__":
    main()
