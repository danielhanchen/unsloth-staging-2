import sys
import torch
print("torch", torch.__version__, flush = True)
ok = torch.backends.mps.is_available()
print("mps available", ok, flush = True)
if not ok:
    sys.exit(3)
x = torch.randn(256, 256, device = "mps", dtype = torch.bfloat16)
print("bf16 matmul ok", float((x @ x).float().abs().mean()), flush = True)
open("mps_ok", "w").write(torch.__version__)
sys.path.insert(0, sys.argv[1])
from core.inference import diffusion_qwenimage21_mps as m
lib = m._library()
print("compile_shader", lib is not None, "failed", m._LIB[1], flush = True)
for k in ("q21_gated_residual", "q21_qk_norm_rope"):
    try:
        print(k, "probe", m._run_probe(k, torch.device("mps")), flush = True)
    except Exception as e:
        print(k, "probe error", repr(e)[:400], flush = True)
