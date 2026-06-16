# Adversarial verification of Component B (dynamic transformers-version routing).
# Read-only verifier: does not modify source. Network is patched to RAISE so any
# accidental fetch on the hot path fails the test loudly.

import json
from pathlib import Path
from unittest.mock import patch

import pytest

import utils.transformers_version as tv


@pytest.fixture(autouse=True)
def _reset_caches():
    # Snapshot + reset all module caches so tests are independent.
    tv._config_json_cache.clear()
    tv._tokenizer_class_cache.clear()
    tv._sidecar_config_mapping_cache.clear()
    tv._MAIN_CONFIG_MODEL_TYPES = None
    yield
    tv._config_json_cache.clear()
    tv._tokenizer_class_cache.clear()
    tv._sidecar_config_mapping_cache.clear()
    tv._MAIN_CONFIG_MODEL_TYPES = None


def _no_network():
    # Any urlopen during a tier decision is a bug -> raise.
    return patch("urllib.request.urlopen", side_effect=AssertionError("network attempted"))


def _prime(name, cfg):
    """Pre-load the config + tokenizer caches so no network is needed."""
    tv._config_json_cache[name] = cfg
    tv._tokenizer_class_cache[name] = False


# ---- Item 2: dynamic correctness -----------------------------------------

def test_known_to_457_stays_default():
    # llama is in the in-process 4.57 CONFIG_MAPPING -> must NOT be sent to a sidecar.
    assert "llama" in tv._main_config_model_types()
    _prime("org/plain-llama", {"model_type": "llama", "architectures": ["LlamaForCausalLM"]})
    with _no_network():
        assert tv.get_transformers_tier("org/plain-llama") == "default"


def test_known_vision_to_457_stays_default():
    _prime("org/qv", {"model_type": "qwen3_vl", "architectures": ["Qwen3VLForConditionalGeneration"]})
    with _no_network():
        assert tv.get_transformers_tier("org/qv") == "default"


def test_real_glm47flash_name_routes_via_curated_530():
    # The canonical model name matches a curated substring -> 530, before any
    # config/network/dynamic layer is reached (pure string check).
    with _no_network():
        assert tv.get_transformers_tier("unsloth/GLM-4.7-Flash") == "530"


def test_glm4_moe_lite_synthetic_name_uses_newest_sidecar_when_present():
    # A name that MISSES the curated substrings reaches the dynamic layer. With the
    # real sidecars present, glm4_moe_lite is recognized by the newest (510=5.10.2)
    # so it routes there. Functionally valid (5.10.2 supports it). Documents the
    # "newest-compatible-first" selection order.
    assert "glm4_moe_lite" not in tv._main_config_model_types()
    _prime("org/glmflashy", {"model_type": "glm4_moe_lite", "transformers_version": "5.2.0.dev0"})
    with _no_network():
        tier = tv.get_transformers_tier("org/glmflashy")
    assert tier in {"510", "550", "530"}  # a 5.x sidecar that recognizes it


def test_version_hint_path_when_sidecars_absent():
    # Isolate the version-hint fallback by simulating absent sidecars.
    with patch.object(tv, "_enumerate_config_mapping", return_value=None):
        for ver, expected in [("5.2.0", "530"), ("5.4.0", "550"), ("5.10.0.dev0", "510")]:
            name = f"org/madeup-{ver}"
            _prime(name, {"model_type": f"madeup_{ver}", "transformers_version": ver})
            with _no_network():
                assert tv.get_transformers_tier(name) == expected, ver


def test_auto_map_repo_code_stays_default_no_crash():
    # deepseek_vl_v2 is unknown to 4.57 but is repo-code (auto_map) -> default (trust_remote_code on 4.57).
    _prime("org/dsocr", {
        "model_type": "deepseek_vl_v2",
        "auto_map": {"AutoConfig": "modeling_x.Cfg"},
        "vision_config": {},
    })
    with _no_network():
        assert tv.get_transformers_tier("org/dsocr") == "default"


def test_dynamic_never_overrides_curated():
    # gemma4_unified is in the curated 510 set: must resolve 510 regardless of any
    # version hint that would map elsewhere -> proves curated wins (dynamic is after).
    _prime("org/g4u", {"model_type": "gemma4_unified", "transformers_version": "5.2.0"})
    with _no_network():
        assert tv.get_transformers_tier("org/g4u") == "510"


def test_registry_unavailable_defers_to_curated():
    # If the in-process registry is empty, dynamic returns None -> default (no crash).
    with patch.object(tv, "_main_config_model_types", return_value=set()):
        _prime("org/whatever", {"model_type": "totally_unknown", "transformers_version": "5.9"})
        with _no_network():
            assert tv.get_transformers_tier("org/whatever") == "default"


# ---- Item 3: cache logic --------------------------------------------------

def _make_venv(tmp_path, version):
    di = tmp_path / f"transformers-{version}.dist-info"
    di.mkdir(parents=True)
    (di / "METADATA").write_text(f"Name: transformers\nVersion: {version}\n")
    return str(tmp_path)


def test_enumerate_caches_disk_and_memory(tmp_path):
    venv = _make_venv(tmp_path, "5.5.0")
    calls = []

    def fake_dump(vd):
        calls.append(vd)
        return {"gemma4", "glm4v"}

    with patch.object(tv, "_subprocess_dump_config_mapping", side_effect=fake_dump):
        r1 = tv._enumerate_config_mapping(venv)
        assert r1 == {"gemma4", "glm4v"}
        assert (Path(venv) / ".unsloth_config_mapping_cache.json").is_file()
        assert len(calls) == 1
        # 2nd call hits in-memory cache -> no new subprocess.
        r2 = tv._enumerate_config_mapping(venv)
        assert r2 == {"gemma4", "glm4v"} and len(calls) == 1
        # Clear in-memory -> disk cache (version matches) serves it, still no subprocess.
        tv._sidecar_config_mapping_cache.clear()
        r3 = tv._enumerate_config_mapping(venv)
        assert r3 == {"gemma4", "glm4v"} and len(calls) == 1


def test_enumerate_version_change_invalidates(tmp_path):
    venv = _make_venv(tmp_path, "5.5.0")
    calls = []
    with patch.object(tv, "_subprocess_dump_config_mapping", side_effect=lambda vd: (calls.append(1) or {"a"})):
        tv._enumerate_config_mapping(venv)
        assert len(calls) == 1
        # Bump the installed version + clear in-memory: disk cache version mismatches -> re-enumerate.
        for di in Path(venv).glob("transformers-*.dist-info"):
            di.rename(Path(venv) / "transformers-5.6.0.dist-info")
        (Path(venv) / "transformers-5.6.0.dist-info" / "METADATA").write_text(
            "Name: transformers\nVersion: 5.6.0\n"
        )
        tv._sidecar_config_mapping_cache.clear()
        tv._enumerate_config_mapping(venv)
        assert len(calls) == 2


def test_enumerate_missing_venv_returns_none():
    assert tv._enumerate_config_mapping("/nonexistent/venv/xyz") is None


def test_enumerate_subprocess_failure_returns_none(tmp_path):
    venv = _make_venv(tmp_path, "5.5.0")
    with patch.object(tv, "_subprocess_dump_config_mapping", return_value=None):
        assert tv._enumerate_config_mapping(venv) is None


# ---- Item 4: no accidental network on the registry helper -----------------

def test_main_registry_no_network():
    with _no_network():
        s = tv._main_config_model_types()
    assert isinstance(s, set) and "llama" in s and len(s) > 100
