# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""
Targeted PR #5727 validation -- non-PEFT base model export on real MLX.

The bug the PR fixes: studio/backend/core/export/export.py:export_base_model
called save_pretrained_merged() on a non-PEFT model without an explicit
save_method, so the MLX implementation defaulted to save_method='lora' and
unsloth_zoo raised:

    Unsloth: save_method='lora' but the model has no LoRA layers --
    there's nothing to save. Use 'merged_16bit' instead.

This script exercises the exact code path the PR touches on a real Apple
Silicon runner. It loads unsloth/gemma-3-270m-it via FastMLXModel.from_pretrained
WITHOUT applying LoRA (i.e. the model is non-PEFT) and:

  1. Confirms the pre-PR behavior: save_pretrained_merged() without
     save_method raises the documented error against a non-PEFT model.
  2. Confirms the PR's fix: save_pretrained_merged(..., save_method='merged_16bit')
     succeeds on the same non-PEFT model and writes the expected
     safetensors + config artifacts.
  3. Drives the patched studio/backend/core/export/export.py:export_base_model
     end-to-end against the same non-PEFT model and asserts it returns
     (success=True, ..., output_path=<dir>) with the expected artifacts.

Only runnable on a real Apple Silicon host. Invoked from
.github/workflows/pr5727-mlx-base-export.yml on macos-14.
"""

from __future__ import annotations

import json
import os
import random as _random
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

import numpy as np

SEED = 3407
MODEL_NAME = "unsloth/gemma-3-270m-it"


def _seed_all() -> None:
    _random.seed(SEED)
    np.random.seed(SEED)
    try:
        import mlx.core as mx
        mx.random.seed(SEED)
    except Exception:
        pass


def _load_non_peft_mlx_model() -> tuple[Any, Any]:
    """Load the model fresh as a non-PEFT (no-LoRA) MLX model.

    This is the exact precondition that triggered the bug -- Studio's
    'Export Base Model' button calls export_base_model against a model
    that has never had get_peft_model applied to it.
    """
    from unsloth_zoo.mlx_loader import FastMLXModel
    model, tokenizer = FastMLXModel.from_pretrained(
        model_name = MODEL_NAME,
        random_state = SEED,
    )
    # Sanity: no LoRA layers attached.
    has_lora = False
    try:
        # FastMLXModel surfaces this on get_peft_model. We probe by
        # inspecting the module tree; if any module has a `lora_A` /
        # `lora_B` attribute, the model is PEFT.
        for _name, mod in getattr(model, "named_modules", lambda: [])():
            if hasattr(mod, "lora_A") or hasattr(mod, "lora_B"):
                has_lora = True
                break
    except Exception:
        pass
    assert not has_lora, (
        f"expected a non-PEFT base model from FastMLXModel.from_pretrained({MODEL_NAME!r}); "
        f"detected LoRA layers, which means the precondition for this test is wrong."
    )
    return model, tokenizer


def _assert_full_checkpoint(path: Path) -> None:
    """A merged_16bit save MUST write the full HF directory."""
    safetensors = sorted(path.glob("*.safetensors"))
    assert safetensors, f"no *.safetensors in {path} -- merged_16bit save did not write weights"
    cfg = path / "config.json"
    assert cfg.is_file(), f"missing config.json under {path}"
    # tokenizer files (gemma-3 ships tokenizer.json + tokenizer_config.json)
    tok_json = (path / "tokenizer.json").is_file()
    tok_cfg = (path / "tokenizer_config.json").is_file()
    assert tok_json or tok_cfg, f"no tokenizer artifacts under {path}"


def phase_1_repro_bug(model, tokenizer) -> dict[str, Any]:
    """Pre-PR behavior: default save_method must raise on non-PEFT model."""
    print("=== PHASE 1: confirm pre-PR error reproduces ===", flush = True)
    with tempfile.TemporaryDirectory() as td:
        try:
            model.save_pretrained_merged(td, tokenizer)
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            print(f"  GOT expected error: {msg}", flush = True)
            assert "lora" in msg.lower() and (
                "no lora layers" in msg.lower() or "merged_16bit" in msg.lower()
            ), (
                f"got an error, but it was not the documented 'no LoRA layers' / "
                f"'merged_16bit' message we expected: {msg}"
            )
            return {"phase_1_error": msg}
    raise AssertionError(
        "PHASE 1 FAILED: save_pretrained_merged() on a non-PEFT model "
        "succeeded WITHOUT save_method. The pre-PR bug no longer reproduces -- "
        "either unsloth_zoo changed its default (recheck save_pretrained_merged "
        "in unsloth_zoo/mlx/utils.py) or the model loaded with LoRA attached."
    )


def phase_2_fix_succeeds(model, tokenizer, workdir: Path) -> dict[str, Any]:
    """PR fix: explicit save_method='merged_16bit' must succeed and write a full ckpt."""
    print("=== PHASE 2: confirm PR fix (save_method=merged_16bit) succeeds ===", flush = True)
    out = workdir / "phase_2_save_pretrained_merged"
    out.mkdir(parents = True, exist_ok = True)
    model.save_pretrained_merged(
        str(out),
        tokenizer,
        save_method = "merged_16bit",
    )
    _assert_full_checkpoint(out)
    print(f"  OK: wrote full checkpoint under {out}", flush = True)
    return {"phase_2_path": str(out)}


def phase_3_export_backend(workdir: Path) -> dict[str, Any]:
    """End-to-end: ExportBackend.export_base_model() on the patched code path."""
    print("=== PHASE 3: drive ExportBackend.export_base_model end-to-end ===", flush = True)
    # Re-load the model fresh so we exercise the same flow Studio's
    # backend would: an ExportBackend instance with .current_model /
    # .current_tokenizer set, .is_peft = False, _IS_MLX = True.
    _seed_all()
    model, tokenizer = _load_non_peft_mlx_model()

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "studio"))
    # The Studio backend imports `from unsloth import _IS_MLX`; that must
    # already evaluate to True on this real Apple Silicon host. We assert
    # to surface a clear failure if not (the workflow installs unsloth
    # so the import chain is satisfied).
    import unsloth as _unsloth
    assert _unsloth._IS_MLX is True, (
        f"expected unsloth._IS_MLX=True on Apple Silicon, got {_unsloth._IS_MLX}"
    )

    # Pin UNSLOTH_STUDIO_HOME so storage_roots.resolve_export_dir
    # accepts a relative save_directory by anchoring it under
    # $UNSLOTH_STUDIO_HOME/exports/. resolve_export_dir refuses any
    # absolute path outside that sandbox.
    studio_home = workdir / "studio_home"
    studio_home.mkdir(parents = True, exist_ok = True)
    os.environ["UNSLOTH_STUDIO_HOME"] = str(studio_home)

    from backend.core.export.export import ExportBackend

    backend = ExportBackend()
    backend.current_model = model
    backend.current_tokenizer = tokenizer
    backend.is_peft = False
    # ExportBackend writes export metadata via self._write_export_metadata(...).
    # On a fresh ExportBackend with no checkpoint state the implementation
    # tolerates missing context, so we leave the rest of the attributes at
    # their defaults.

    # Use a relative export name so resolve_export_dir places it under
    # $UNSLOTH_STUDIO_HOME/exports/pr5727_phase3.
    success, message, output_path = backend.export_base_model(
        save_directory = "pr5727_phase3",
        push_to_hub = False,
    )
    print(f"  export_base_model -> success={success}, msg={message!r}, path={output_path}",
          flush = True)
    assert success is True, (
        f"PHASE 3 FAILED: export_base_model returned success=False. "
        f"Message: {message!r}"
    )
    assert output_path, "PHASE 3 FAILED: export_base_model returned no output_path"
    _assert_full_checkpoint(Path(output_path))
    print(f"  OK: ExportBackend wrote full checkpoint under {output_path}", flush = True)
    return {
        "phase_3_path": output_path,
        "phase_3_message": message,
    }


def main() -> int:
    print(f"[pr5727 smoke] starting; model={MODEL_NAME}", flush = True)
    _seed_all()

    metrics: dict[str, Any] = {}
    workdir = Path(os.environ.get("PR5727_WORKDIR", "/tmp/pr5727_workdir"))
    workdir.mkdir(parents = True, exist_ok = True)

    # PHASE 1 + PHASE 2 share a single non-PEFT model load
    model, tokenizer = _load_non_peft_mlx_model()
    metrics.update(phase_1_repro_bug(model, tokenizer))
    metrics.update(phase_2_fix_succeeds(model, tokenizer, workdir))
    # PHASE 3 reloads the model fresh inside its helper.
    metrics.update(phase_3_export_backend(workdir))

    out = workdir / "pr5727_metrics.json"
    out.write_text(json.dumps(metrics, indent = 2))
    print(f"[pr5727 smoke] metrics written to {out}", flush = True)
    print(json.dumps(metrics, indent = 2), flush = True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
