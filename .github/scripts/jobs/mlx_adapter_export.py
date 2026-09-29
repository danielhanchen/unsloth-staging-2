#!/usr/bin/env python3
"""Apple Silicon: Studio LoRA export in MLX, PEFT and GGUF adapter formats from a real MLX adapter.

Trains a tiny MLX LoRA with mlx_sft.py, loads it through Studio's ExportBackend, then exports:
native MLX, PEFT (twice into one folder), a refused MLX->PEFT mix, and a GGUF LoRA. The PEFT copy
is loaded with transformers + PEFT on torch CPU and must change the base model's logits.
Exit 3 off Apple Silicon.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

TINY = "mlx-community/SmolLM-135M-Instruct-4bit"
HF_BASE = "HuggingFaceTB/SmolLM-135M-Instruct"


def main(argv = None):
    p = C.base_parser("mlx_adapter_export", TINY, TINY, default_steps = 3)
    p.add_argument("--repo", default = ".", help = "unsloth checkout (studio/backend inside)")
    a = C.resolve_args(p, argv)
    with C.JobRecorder(a, backend_hint = "mlx") as rec:
        if not (platform.system() == "Darwin" and platform.machine() == "arm64"):
            rec.skip(f"MLX needs Apple Silicon, this is {platform.system()}-{platform.machine()}")
        repo = Path(a.repo).resolve()
        work = Path(tempfile.mkdtemp(prefix = "mlx_adapter_export_"))
        here = Path(__file__).resolve().parent
        r = subprocess.run(
            [
                sys.executable,
                str(here / "mlx_sft.py"),
                "--tiny",
                "--max-steps",
                str(a.max_steps),
                "--save-dir",
                str(work / "sft"),
                "--out",
                str(work / "sft.json"),
            ]
        )
        lora = work / "sft" / "lora"
        rec.check(
            "mlx_adapter_trained",
            r.returncode == 0 and (lora / "adapters.safetensors").exists(),
            f"mlx_sft exit {r.returncode}",
        )

        req = repo / "studio" / "backend" / "requirements" / "studio.txt"
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req)])
        if r.returncode:
            raise SystemExit(f"studio requirements install failed ({r.returncode})")
        home = work / "studio_home"
        os.environ["UNSLOTH_STUDIO_HOME"] = str(home)
        sys.path.insert(0, str(repo / "studio" / "backend"))
        from core.export.export import ExportBackend, _IS_MLX

        rec.check("studio_export_on_mlx", bool(_IS_MLX), f"_IS_MLX={_IS_MLX}")
        backend = ExportBackend()
        ok, msg = backend.load_checkpoint(str(lora))
        rec.check("load_checkpoint", ok and backend.is_peft, msg)

        out = home / "exports_test"
        d_mlx, d_peft, d_gguf = out / "mlx", out / "peft", out / "gguf"
        ok, msg, _ = backend.export_lora_adapter(str(d_mlx))
        rec.check(
            "native_mlx_export",
            ok
            and (d_mlx / "adapters.safetensors").exists()
            and not (d_mlx / "adapter_model.safetensors").exists(),
            msg,
        )
        ok, msg, _ = backend.export_lora_adapter(str(d_mlx), adapter_format = "peft")
        rec.check("mixed_format_refused", not ok and "mix" in msg, msg)

        results = [
            backend.export_lora_adapter(str(d_peft), adapter_format = "peft") for _ in range(2)
        ]
        cfg_path = d_peft / "adapter_config.json"
        cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        rec.check(
            "peft_export_twice_same_dir",
            all(r[0] for r in results)
            and (d_peft / "adapter_model.safetensors").exists()
            and not (d_peft / "adapters.safetensors").exists()
            and cfg.get("peft_type") == "LORA",
            [r[1] for r in results],
        )
        rec.summary(
            peft_config = {
                k: cfg.get(k)
                for k in ("r", "lora_alpha", "target_modules", "base_model_name_or_path")
            },
            peft_files = sorted(p.name for p in d_peft.iterdir()) if d_peft.exists() else [],
        )

        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tok = AutoTokenizer.from_pretrained(HF_BASE)
            ids = tok("The capital of France is", return_tensors = "pt").input_ids
            base = AutoModelForCausalLM.from_pretrained(HF_BASE, torch_dtype = torch.float32)
            with torch.no_grad():
                ref = base(ids).logits
            peft_model = PeftModel.from_pretrained(base, str(d_peft))
            with torch.no_grad():
                got = peft_model(ids).logits
            n_lora = sum(1 for n, _ in peft_model.named_parameters() if "lora_" in n)
            diff = float((got - ref).abs().max())
            rec.check(
                "peft_loads_in_transformers",
                n_lora > 0 and diff > 0,
                f"{n_lora} lora tensors, max |dlogit| {diff:.3e}",
            )
        except Exception as e:
            rec.check("peft_loads_in_transformers", False, f"{type(e).__name__}: {e}")

        ok, msg, _ = backend.export_lora_adapter(str(d_gguf), gguf = True, gguf_outtype = "f16")
        ggufs = sorted(p.name for p in d_gguf.glob("*.gguf")) if d_gguf.exists() else []
        rec.check("gguf_lora_export", ok and bool(ggufs), f"{msg} {ggufs}")
        rec.summary(gguf_files = ggufs)


if __name__ == "__main__":
    main()
