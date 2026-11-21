#!/usr/bin/env python3
"""Re-container a Qwen-Image VAE (Qwen-Image-2.1: Wan 2.2 layout; Qwen-Image / Edit: Wan 2.1 layout, --wan21) (diffusers AutoencoderKLWan names) into the baseline UI's Wan 2.2 layout, same
tensors (renamed; 2D conv weights get the size-1 temporal axis back). The name map is diffusers' own
convert_vae_22 run on the baseline UI's WanVAE key list.

  python qwen21_vae_to_bline.py BLINE_DIR DIFFUSERS_SCRIPT SRC.safetensors OUT.safetensors
"""
import importlib.machinery
import importlib.util
import sys
import types

import torch
try:  # extnames-aware safetensors (builders/_st.py); as a script or as builders.<module>
    from _st import safe_open
except ImportError:
    from builders._st import safe_open
try:  # extnames-aware safetensors (builders/_st.py); as a script or as builders.<module>
    from _st import save_file
except ImportError:
    from builders._st import save_file

WAN21 = "--wan21" in sys.argv
bline_dir, conv_script, src, out = [x for x in sys.argv[1:] if x != "--wan21"][:4]
sys.path.insert(0, bline_dir)
import extnames as X  # noqa: E402 - the server package's real name in a public copy

wv = X.import_module("bline.ldm.wan.vae" if WAN21 else "bline.ldm.wan.vae2_2")

captured = {}


class _Dummy:
    def __init__(self, *a, **kw):
        pass

    def __call__(self, *a, **kw):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def load_state_dict(self, sd, **kw):
        captured.update(sd)


f = safe_open(src, "pt")
keys = list(f.keys())
shape = {k: f.get_slice(k).get_shape() for k in keys}
if WAN21:  # bline/sd.py "Wan 2.1 VAE" ddconfig
    cfg = {"dim": shape["encoder.conv_in.weight"][0], "z_dim": 16, "dim_mult": [1, 2, 4, 4], "num_res_blocks": 2,
           "attn_scales": [], "temperal_downsample": [False, True, True], "image_channels": shape["encoder.conv_in.weight"][1],
           "conv_out_channels": shape["decoder.conv_out.weight"][0], "dropout": 0.0}
else:
    cfg = {"dim": shape["encoder.conv_in.weight"][0], "dec_dim": shape["decoder.norm_out.gamma"][0], "z_dim": 64,
           "dim_mult": [1, 2, 4, 8, 8], "num_res_blocks": 2, "attn_scales": [], "temperal_downsample": [False, True, True, True],
           "dropout": 0.0, "image_channels": shape["decoder.conv_out.weight"][0], "patch_size": 1, "temporal_kernel": 1}
with torch.device("meta"):
    m = wv.WanVAE(**cfg)
orig = {k: tuple(v.shape) for k, v in m.state_dict().items()}

class _Any(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Dummy


for mod in ("accelerate", "diffusers", "diffusers.utils", "transformers"):
    sys.modules[mod] = _Any(mod)
    sys.modules[mod].__spec__ = importlib.machinery.ModuleSpec(mod, None)
spec = importlib.util.spec_from_file_location("cw", conv_script)
cw = importlib.util.module_from_spec(spec)
argv = sys.argv
sys.argv = [conv_script]
spec.loader.exec_module(cw)
sys.argv = argv
cw.hf_hub_download = lambda *a, **k: "x"
cw.AutoencoderKLWan = _Dummy
cw.init_empty_weights = _Dummy
cw.torch = types.SimpleNamespace(load=lambda *a, **k: {k2: k2 for k2 in orig})
cw.convert_vae() if WAN21 else cw.convert_vae_22()
d2o = dict(captured)
missing = [k for k in keys if k not in d2o]
assert not missing, missing[:10]
sd = {}
for k in keys:
    t = f.get_tensor(k)
    o = d2o[k]
    want = orig[o]
    if tuple(t.shape) != want:
        assert t.numel() == torch.Size(want).numel(), (k, t.shape, want)
        t = t.reshape(want)
    sd[o] = t.contiguous()
extra = sorted(set(orig) - set(sd))
print("tensors", len(sd), "orig keys without source", extra[:10])
save_file(sd, out)
print(out)
