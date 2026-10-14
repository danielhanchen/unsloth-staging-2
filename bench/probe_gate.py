import json, sys, types, torch
sys.path.insert(0, "head/studio/backend")
import core.inference.diffusion_native_quant as nq

out = {"torch": torch.__version__, "mps": torch.backends.mps.is_available()}
for k, n in [(3840, 10240), (10240, 3840), (3840, 3840)]:
    nq._INT8PACK_PROBED.clear()
    lin = torch.nn.Linear(k, n, bias = False).to(torch.bfloat16)
    mod = torch.nn.Sequential(nq.native_linear_class()(lin, "int8", scale_dtype = torch.bfloat16)).to("mps")
    layer = mod[0]
    k_s, d_s = nq._bench_int8pack_vs_dense(layer, torch.bfloat16, torch.device("mps"))
    nq._INT8PACK_PROBED.clear()
    verdict = nq.probe_mps_int8pack(mod, torch.bfloat16)
    out[f"{k}x{n}"] = {"kernel_ms": k_s * 1e3, "dense_bf16_ms": d_s * 1e3, "speedup": d_s / k_s, "verdict_kernel": verdict,
                       "engaged": layer.int8pack_engaged(torch.bfloat16)}
print(json.dumps(out, indent = 1))
