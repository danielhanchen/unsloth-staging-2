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

# Fast-math / denormal check: compile_shader's flush behaviour vs the stock MPS ops, on bf16 subnormal inputs.
import torch
src = """
kernel void q21_ftz_probe(device const float* x [[buffer(0)]], device float* out [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    out[i] = x[i] * 1.0f + 0.0f;
}
"""
lib2 = torch.mps.compile_shader(src)
x = torch.tensor([1e-40, -3e-39, 1e-45, 1.0], device = "mps")
y = torch.empty_like(x)
lib2.q21_ftz_probe(x, y, threads = 4, group_size = 4)
print("compile_shader fp32 denormal passthrough:", y.cpu().tolist(), "(0.0 = flushed)", flush = True)
print("stock MPS fp32 mul denormal:", (x * 1.0).cpu().tolist(), flush = True)
sub = torch.tensor([2.0**-130, -(2.0**-128), 2.0**-133, 2.0**-127 * 1.5], dtype = torch.float32).to(torch.bfloat16)
print("bf16 subnormal inputs", sub.float().tolist(), flush = True)
d = torch.device("mps")
xs = sub.repeat(1, 1, 256).view(1, 4, 256).to(d)  # (1, 4, 256)
xs = torch.cat([xs, torch.randn(1, 4, 256).to(d, torch.bfloat16)], dim = 1)
rows = torch.cat([torch.full((1, 256), 0.5), torch.ones(1, 256)]).to(d, torch.bfloat16)  # 1 + scale = 1.5 / 1.0 ... t=0 row
mask = torch.arange(8, device = d) >= 2
want_mod = xs * (1 + m._module()._select_modulation_rows(rows, mask))
got_mod = torch.empty_like(xs)
lib.q21_modulate(xs, (1 + rows).contiguous(), got_mod, 64, 8, 2, 1, threads = (64, 8, 1), group_size = (64, 1, 1))
print("modulate on subnormals: bit-equal", bool(torch.equal(got_mod, want_mod)),
      "stock", want_mod[0, :4, 0].float().cpu().tolist(), "kernel", got_mod[0, :4, 0].float().cpu().tolist(), flush = True)
ys = xs.flip(1).contiguous()
g = rows.tanh().contiguous()
want_res = xs + m._module()._select_modulation_rows(rows, mask).tanh() * ys
got_res = torch.empty_like(xs)
lib.q21_gated_residual(xs, ys, g, got_res, 64, 8, 2, 1, threads = (64, 8, 1), group_size = (64, 1, 1))
print("residual on subnormals: bit-equal", bool(torch.equal(got_res, want_res)),
      "stock", want_res[0, :4, 0].float().cpu().tolist(), "kernel", got_res[0, :4, 0].float().cpu().tolist(), flush = True)
