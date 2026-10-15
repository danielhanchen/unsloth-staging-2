#!/usr/bin/env python3
"""Add EXPS_CPU=1 (routed experts on the CPU buffer) to diffusion-gemma-eval so the full model fits a small Metal working set."""
import sys
p = sys.argv[1]
s = open(p).read()
anchor = '    mparams.n_gpu_layers = atoi(getenv("NGL") ? getenv("NGL") : "0");\n'
assert anchor in s, "anchor missing"
add = '''    const char * exps_pat = getenv("EXPS_CPU_PATTERN") ? getenv("EXPS_CPU_PATTERN") : "\\\\.ffn_(up|down|gate|gate_up)_exps";
    const llama_model_tensor_buft_override exps_cpu[] = {
        { exps_pat, ggml_backend_cpu_buffer_type() },
        { nullptr, nullptr },
    };
    if (getenv("EXPS_CPU") && atoi(getenv("EXPS_CPU"))) {
        mparams.tensor_buft_overrides = exps_cpu;
        fprintf(stderr, "routed experts on CPU: %s\\n", exps_pat);
    }
'''
if "EXPS_CPU" not in s:
    s = s.replace(anchor, anchor + add)
open(p, "w").write(s)
print("patched", p)
