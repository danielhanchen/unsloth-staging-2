# SPDX-License-Identifier: AGPL-3.0-only
"""Real Apple Silicon probe for the Studio MLX load failure on qwen3_5 / gemma4.

Reproduces the tester error "Received N parameters not in model:
language_model.model.layers.*.self_attn.k_norm.weight" and finds which
mlx-lm / mlx-vlm build actually loads AND generates coherently (QK-norm applied,
not just a successful strict-skipping load). Llama-3.2-1B (plain llama) is the
control that must always pass. Exits non-zero only if the control breaks, so
every version-sweep job completes and the per-model results can be compared.
"""
import platform
import sys
import traceback

MODELS = [
    ("gemma4 VLM   ", "unsloth/gemma-4-E2B-it-UD-MLX-4bit"),
    ("qwen3_5      ", "mlx-community/Qwen3.5-2B-8bit"),
    ("llama control", "mlx-community/Llama-3.2-1B-Instruct-4bit"),
]
PROMPT = "Explain what a hash map is in one short sentence."
CONTROL = "llama control"


def _sanity():
    assert platform.system() == "Darwin" and platform.machine() == "arm64", platform.platform()
    import mlx.core as mx

    assert "mlx_simulation" not in (getattr(mx, "__file__", "") or ""), mx.__file__
    import mlx_lm

    v = {"mlx": mx.__version__, "mlx_lm": getattr(mlx_lm, "__version__", "?")}
    try:
        import mlx_vlm

        v["mlx_vlm"] = getattr(mlx_vlm, "__version__", "?")
    except Exception as e:
        v["mlx_vlm"] = f"import-failed: {e}"
    print("VERSIONS:", v, flush=True)


def _coherent(text):
    if not text or len(text.strip()) < 12:
        return False
    words = text.split()
    # crude loop detector: healthy text has many distinct tokens
    return len(set(w.lower() for w in words)) >= max(5, len(words) // 4)


def _try_mlx_lm(repo):
    from mlx_lm import generate, load

    model, tok = load(repo)
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": PROMPT}], add_generation_prompt=True
    )
    return generate(model, tok, prompt=prompt, max_tokens=48, verbose=False)


def _try_mlx_vlm(repo):
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config

    model, proc = load(repo)
    cfg = load_config(repo)
    prompt = apply_chat_template(proc, cfg, PROMPT, num_images=0)
    out = generate(model, proc, prompt, max_tokens=48, verbose=False)
    # mlx-vlm may return a GenerationResult or a str depending on version
    return getattr(out, "text", out)


def main():
    _sanity()
    results = {}
    for label, repo in MODELS:
        print(f"\n===== {label.strip()} :: {repo} =====", flush=True)
        ok, note = False, ""
        for name, fn in (("mlx_lm", _try_mlx_lm), ("mlx_vlm", _try_mlx_vlm)):
            try:
                out = fn(repo)
                c = _coherent(out)
                print(f"[{name}] loaded+generated coherent={c}\n   >> {out[:220]!r}", flush=True)
                if c:
                    ok, note = True, name
                    break
                note = note or f"{name}:incoherent(loop?)"
            except Exception as e:
                first = (str(e).splitlines() or [""])[0][:220]
                print(f"[{name}] FAILED: {first}", flush=True)
                if "not in model" in str(e):
                    print("   (reproduces the tester's strict load_weights error)", flush=True)
                note = note or f"{name}:{first}"
        results[label.strip()] = (ok, note)

    print("\n===== SUMMARY =====", flush=True)
    all_ok = True
    for label, (ok, note) in results.items():
        print(f"{'PASS' if ok else 'FAIL'}  {label}  ({note})", flush=True)
        all_ok = all_ok and ok
    print("ALL_COHERENT:", all_ok, flush=True)

    if not results[CONTROL][0]:
        print("CONTROL FAILED -> environment broken, not an arch issue", flush=True)
        sys.exit(2)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(2)
