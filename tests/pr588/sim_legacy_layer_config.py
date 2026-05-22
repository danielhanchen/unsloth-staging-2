"""Sim test: get_model_layer_config() schema regressions for existing models."""
import sys
# sys.path: unsloth_zoo is pip-installed in CI

from unsloth_zoo.empty_model import get_model_layer_config


def test_schema_keys_present():
    cfg = get_model_layer_config()
    for key in ("standard_layers", "layernorms", "vision_layers", "non_layered_components"):
        assert key in cfg, f"missing schema key: {key}"


def test_existing_llama_qwen_paths_still_in_standard_layers():
    cfg = get_model_layer_config()
    std = cfg["standard_layers"]
    # Classic dense LM paths must remain
    must_exist = [
        "model.layers.{kk}.self_attn.q_proj",
        "model.layers.{kk}.self_attn.k_proj",
        "model.layers.{kk}.self_attn.v_proj",
        "model.layers.{kk}.self_attn.o_proj",
        "model.layers.{kk}.mlp.gate_proj",
        "model.layers.{kk}.mlp.up_proj",
        "model.layers.{kk}.mlp.down_proj",
        "model.language_model.layers.{kk}.self_attn.q_proj",
        "model.language_model.layers.{kk}.self_attn.k_proj",
        "model.language_model.layers.{kk}.self_attn.v_proj",
        "model.language_model.layers.{kk}.self_attn.o_proj",
        "model.language_model.layers.{kk}.mlp.gate_proj",
        "model.language_model.layers.{kk}.mlp.up_proj",
        "model.language_model.layers.{kk}.mlp.down_proj",
        # phi3 fused path
        "model.language_model.layers.{kk}.mlp.gate_up_proj",
        "model.layers.{kk}.mlp.gate_up_proj",
    ]
    for path in must_exist:
        assert path in std, f"regression: '{path}' no longer in standard_layers"


def test_existing_layernorms_preserved():
    cfg = get_model_layer_config()
    norms = cfg["layernorms"]
    must_exist = [
        "model.layers.{kk}.input_layernorm",
        "model.layers.{kk}.post_attention_layernorm",
        "model.language_model.layers.{kk}.input_layernorm",
        "model.language_model.layers.{kk}.post_attention_layernorm",
    ]
    for path in must_exist:
        assert path in norms, f"regression: layernorm '{path}' missing"


def test_visual_merger_linear_fc_fix():
    """Pre-fix: 'model.visual.merger.linear_fc{kk}' substituted to linear_fc0 / linear_fc1.
       Post-fix: moved to non_layered_components as explicit linear_fc1 / linear_fc2."""
    cfg = get_model_layer_config()
    nlc = cfg["non_layered_components"]
    assert "model.visual.merger.linear_fc1" in nlc
    assert "model.visual.merger.linear_fc2" in nlc
    # Make sure the old buggy template is gone
    additional = set()
    for section in ("standard_layers", "vision_layers", "layernorms"):
        additional |= set(cfg.get(section, ()))
    assert "model.visual.merger.linear_fc{kk}" not in additional


def test_new_gemma4_linear_attn_paths_added():
    cfg = get_model_layer_config()
    std = cfg["standard_layers"]
    # New GDN paths
    gdn_paths = [
        "model.layers.{kk}.linear_attn.in_proj_qkv",
        "model.layers.{kk}.linear_attn.in_proj_z",
        "model.layers.{kk}.linear_attn.in_proj_b",
        "model.layers.{kk}.linear_attn.in_proj_a",
        "model.layers.{kk}.linear_attn.conv1d",
        "model.layers.{kk}.linear_attn.out_proj",
        "model.layers.{kk}.linear_attn.dt_bias",
        "model.layers.{kk}.linear_attn.A_log",
        "model.language_model.layers.{kk}.linear_attn.in_proj_qkv",
    ]
    for p in gdn_paths:
        assert p in std, f"missing new GDN path: {p}"


def test_new_gemma4_per_layer_input_paths_added():
    cfg = get_model_layer_config()
    std = cfg["standard_layers"]
    nlc = cfg["non_layered_components"]
    assert "model.layers.{kk}.per_layer_input_gate" in std
    assert "model.layers.{kk}.per_layer_projection" in std
    assert "model.language_model.embed_tokens_per_layer" in nlc
    assert "model.language_model.per_layer_model_projection" in nlc
    assert "model.language_model.per_layer_projection_norm" in nlc


def test_new_layer_scalar_added():
    cfg = get_model_layer_config()
    std = cfg["standard_layers"]
    assert "model.layers.{kk}.layer_scalar" in std
    assert "model.language_model.layers.{kk}.layer_scalar" in std
