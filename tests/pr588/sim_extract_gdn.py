"""Sim test: extract_gdn_layers shape / fp8 / output_sizes fallback."""
import sys, pytest
# sys.path: unsloth_zoo is pip-installed in CI

import torch
from unsloth_zoo.empty_model import extract_gdn_layers


class _Proj(torch.nn.Module):
    def __init__(self, out_features, in_features, dtype=torch.float32, output_sizes=None,
                 weight_scale=None, weight_block_size=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features, dtype=dtype), requires_grad=False)
        if output_sizes is not None:
            self.output_sizes = output_sizes
        if weight_scale is not None:
            self.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
        if weight_block_size is not None:
            self.weight_block_size = weight_block_size


class _GDN(torch.nn.Module):
    def __init__(self, num_k=2, num_v=2, hk=2, hv=4, hidden=8,
                 use_output_sizes=False, fp8=False):
        super().__init__()
        self.hidden_size = hidden
        self.num_k_heads = num_k
        self.num_v_heads = num_v
        self.head_k_dim = hk
        self.head_v_dim = hv
        self.key_dim = num_k * hk
        self.value_dim = num_v * hv

        qkvz_total = self.key_dim * 2 + self.value_dim * 2
        if use_output_sizes:
            sizes = [self.key_dim, self.key_dim, self.value_dim, self.value_dim]
        else:
            sizes = None

        dtype = torch.float8_e4m3fn if fp8 else torch.float32
        scale = None
        block = None
        if fp8:
            # Build a 1-D scale and a 2-D block-size scale path
            scale = torch.randn(qkvz_total, dtype=torch.float32)
            block = (1, 1)

        self.in_proj_qkvz = _Proj(qkvz_total, hidden, dtype=dtype, output_sizes=sizes,
                                  weight_scale=scale, weight_block_size=block)
        self.in_proj_ba = _Proj(num_v * 2, hidden)
        self.conv1d = _Proj(self.key_dim * 2 + self.value_dim, 4)
        self.dt_bias = torch.nn.Parameter(torch.randn(num_v), requires_grad=False)
        self.A_log = torch.nn.Parameter(torch.randn(num_v), requires_grad=False)
        self.norm = torch.nn.Module()
        self.norm.weight = torch.nn.Parameter(torch.randn(hv), requires_grad=False)
        self.out_proj = _Proj(hidden, self.value_dim)


def _fake_get_state_dict(prefix, kk, state_dict, module, slice_weights=True):
    state_dict[f"{prefix}.weight"] = module.weight.data


def test_plain_with_output_sizes_emits_all_keys():
    gdn = _GDN(use_output_sizes=True)
    sd, qsd = {}, {}
    extract_gdn_layers(gdn, "P", sd, qsd, _fake_get_state_dict)
    expected = {
        "P.in_proj_qkv.weight", "P.in_proj_z.weight",
        "P.in_proj_b.weight", "P.in_proj_a.weight",
        "P.conv1d.weight", "P.dt_bias", "P.A_log",
        "P.norm.weight", "P.out_proj.weight",
    }
    assert expected <= set(sd.keys()), f"missing keys: {expected - set(sd.keys())}"


def test_plain_without_output_sizes_falls_back_to_key_value_dim():
    gdn = _GDN(use_output_sizes=False)
    sd, qsd = {}, {}
    extract_gdn_layers(gdn, "P", sd, qsd, _fake_get_state_dict)
    # in_proj_qkv = key_dim * 2 + value_dim; in_proj_z = value_dim
    assert sd["P.in_proj_qkv.weight"].shape[0] == gdn.key_dim * 2 + gdn.value_dim
    assert sd["P.in_proj_z.weight"].shape[0] == gdn.value_dim


def test_non_square_dimensions():
    gdn = _GDN(num_k=3, num_v=2, hk=4, hv=5, use_output_sizes=True)
    sd, qsd = {}, {}
    extract_gdn_layers(gdn, "P", sd, qsd, _fake_get_state_dict)
    assert sd["P.in_proj_qkv.weight"].shape[0] == 3*4 + 3*4 + 2*5
    assert sd["P.in_proj_z.weight"].shape[0] == 2*5


def test_missing_key_dim_value_dim_raises_clearly():
    gdn = _GDN()
    del gdn.key_dim
    del gdn.value_dim
    with pytest.raises(RuntimeError, match="in_proj_qkvz"):
        extract_gdn_layers(gdn, "P", {}, {}, _fake_get_state_dict)


def test_in_proj_ba_split_at_midpoint():
    gdn = _GDN()
    sd, qsd = {}, {}
    extract_gdn_layers(gdn, "P", sd, qsd, _fake_get_state_dict)
    raw = gdn.in_proj_ba.weight.data
    mid = raw.shape[0] // 2
    torch.testing.assert_close(sd["P.in_proj_b.weight"], raw[:mid])
    torch.testing.assert_close(sd["P.in_proj_a.weight"], raw[mid:])


def test_norm_weight_emitted_when_present():
    gdn = _GDN()
    sd, qsd = {}, {}
    extract_gdn_layers(gdn, "P", sd, qsd, _fake_get_state_dict)
    assert "P.norm.weight" in sd


def test_norm_weight_skipped_when_module_lacks_weight():
    gdn = _GDN()
    gdn.norm = torch.nn.Module()  # no .weight
    sd, qsd = {}, {}
    extract_gdn_layers(gdn, "P", sd, qsd, _fake_get_state_dict)
    assert "P.norm.weight" not in sd


def test_dt_bias_a_log_are_plain_buffers():
    gdn = _GDN()
    sd, qsd = {}, {}
    extract_gdn_layers(gdn, "P", sd, qsd, _fake_get_state_dict)
    torch.testing.assert_close(sd["P.dt_bias"], gdn.dt_bias.data)
    torch.testing.assert_close(sd["P.A_log"], gdn.A_log.data)
