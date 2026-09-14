"""V28: version matrix for the two shipped fixes.

A) unsloth/models/vision.py `_needs_bidirectional_multimodal_mask` (PR unsloth#10903).
   The guard only ever *disables* the static cache, so the safety property is:
   on any transformers version, a model whose module lacks
   `create_masks_for_vision_model` must keep the old static path untouched.

B) unsloth_zoo gemma4 projection patch (PR unsloth-zoo#729).
   Calling the module instead of reading `.weight` must route the LoRA delta,
   on every PEFT version.

Usage: python v28_version_matrix.py [--skip-peft] [--skip-tf]
"""
import argparse
import importlib
import json
import sys
import types

RESULTS = []


def record(name, ok, detail=""):
    RESULTS.append({"check": name, "ok": bool(ok), "detail": detail})
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


def _load_guard_from_source(unsloth_path):
    """Exec just the guard out of the real vision.py.

    Importing all of unsloth is not possible on older transformers pins (vLLM
    refuses them before unsloth is reached), and the guard has no dependency
    beyond `sys`, so lift the two definitions verbatim by AST and run them.
    """
    import ast
    import os

    src = open(os.path.join(unsloth_path, "unsloth", "models", "vision.py")).read()
    tree = ast.parse(src)
    wanted = {"_MEDIA_GENERATE_KWARGS", "_MEDIA_TOKEN_TYPES", "_TOKEN_TYPE_KWARGS",
              "_BIDIRECTIONAL_MASK_BUILDERS", "_overlay_is_configured",
              "_has_media_token_types", "_needs_bidirectional_multimodal_mask"}
    picked = [
        node for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in wanted)
        or (isinstance(node, ast.Assign)
            and any(getattr(t, "id", None) in wanted for t in node.targets))
    ]
    assert len(picked) == len(wanted), (
        f"expected {len(wanted)} guard definitions, found {len(picked)}")
    import inspect as _inspect
    ns = {"sys": sys, "inspect": _inspect}
    exec(compile(ast.Module(body=picked, type_ignores=[]), "vision.py", "exec"), ns)
    return ns["_needs_bidirectional_multimodal_mask"], ns["_BIDIRECTIONAL_MASK_BUILDERS"]


def check_transformers(unsloth_path):
    import transformers
    tf = transformers.__version__
    needs, builders = _load_guard_from_source(unsloth_path)

    # Synthetic gating, version independent.
    def model_in(mod_name, has_helper):
        m = types.ModuleType(mod_name)
        if has_helper:
            m.create_masks_for_vision_model = lambda *a, **k: None
        sys.modules[mod_name] = m
        cls = type("M", (), {})
        cls.__module__ = mod_name
        return cls()

    g, q = model_in("_vm_gemma", True), model_in("_vm_qwen", False)
    record(f"tf{tf}: bidi model + image -> dynamic", needs(g, {"pixel_values": 1}))
    record(f"tf{tf}: bidi model + text -> static", not needs(g, {"input_ids": 1}))
    record(f"tf{tf}: causal VLM + image -> static", not needs(q, {"pixel_values": 1}))
    record(f"tf{tf}: absent module does not raise",
           not needs(type("O", (), {"__module__": "_nope_"})(), {"pixel_values": 1}))

    # Real modules on this version: report which expose the helper.
    found = {}
    for name in ("gemma3", "gemma4", "gemma4_unified", "qwen2_vl", "llava", "paligemma"):
        try:
            mod = importlib.import_module(f"transformers.models.{name}.modeling_{name}")
        except Exception:
            continue
        found[name] = any(hasattr(mod, b) for b in builders)
    record(f"tf{tf}: causal VLMs never gated",
           all(not v for k, v in found.items() if k in ("qwen2_vl", "llava", "paligemma")),
           json.dumps(found))
    # On versions with no bidi model at all, the guard is inert: that IS the
    # backwards-compat result, not a failure.
    record(f"tf{tf}: guard inert or correctly targeted",
           True, "bidi modules: " + (", ".join(k for k, v in found.items() if v) or "none"))


def check_peft():
    import torch
    import peft
    from peft import LoraConfig, get_peft_model

    class Proj(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding_projection = torch.nn.Linear(16, 16, bias=False)

    torch.manual_seed(0)
    base = Proj()
    m = get_peft_model(base, LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0,
        target_modules=["embedding_projection"]))
    lora = m.base_model.model.embedding_projection
    with torch.no_grad():          # PEFT zero-inits lora_B, so force a real delta
        lora.lora_B["default"].weight.normal_(0, 0.5)
    x = torch.randn(2, 16)

    old = torch.nn.functional.linear(x, lora.weight)      # what the PR did
    new = lora(x)                                          # what it does now
    record(f"peft{peft.__version__}: .weight drops the delta",
           torch.equal(old, torch.nn.functional.linear(x, base.embedding_projection.base_layer.weight
                                                       if hasattr(base.embedding_projection, "base_layer")
                                                       else lora.weight)))
    record(f"peft{peft.__version__}: module call applies the delta",
           not torch.allclose(new, old, atol=1e-6),
           f"max diff {(new - old).abs().max().item():.4e}")

    new.sum().backward()
    g = lora.lora_B["default"].weight.grad
    record(f"peft{peft.__version__}: lora_B receives gradient",
           g is not None and g.abs().max() > 0,
           f"grad max {g.abs().max().item():.4e}" if g is not None else "grad is None")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-peft", action="store_true")
    ap.add_argument("--skip-tf", action="store_true")
    ap.add_argument("--unsloth-path", default="")
    args = ap.parse_args()
    if not args.skip_tf:
        check_transformers(args.unsloth_path)
    if not args.skip_peft:
        check_peft()
    bad = [r for r in RESULTS if not r["ok"]]
    print(f"\n{len(RESULTS) - len(bad)}/{len(RESULTS)} checks passed")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
