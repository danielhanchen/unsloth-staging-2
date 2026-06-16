# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the dynamic, registry-driven transformers tier router (Component B).

The curated substring/architecture sets in ``utils/transformers_version.py`` are
kept as the authoritative fast path; on top of them a dynamic layer asks each
transformers environment's own ``CONFIG_MAPPING_NAMES`` whether it recognizes a
model's ``model_type``. These tests pin that the dynamic layer:

* routes architectures known to the in-process 4.57 registry to ``"default"``;
* upgrades architectures unknown to 4.57 to a 5.x tier (sidecar match or the
  ``transformers_version`` hint), without editing the hardcoded lists;
* treats ``auto_map`` (repo-code) models as ``"default"`` (trust_remote_code);
* caches sidecar enumerations on disk keyed on the sidecar's tf version;
* never overrides a positive curated decision (existing tests stay green).
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

import types as _types

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

from utils.transformers_version import (  # noqa: E402
    _config_json_cache,
    _config_needs_510_cache,
    _config_needs_550_cache,
    _dynamic_tier_from_cfg,
    _enumerate_config_mapping,
    _main_config_model_types,
    _sidecar_config_mapping_cache,
    _tier_from_transformers_version,
    _tokenizer_class_cache,
    get_transformers_tier,
    needs_transformers_5,
)


def _clear_caches():
    _config_json_cache.clear()
    _config_needs_510_cache.clear()
    _config_needs_550_cache.clear()
    _tokenizer_class_cache.clear()
    _sidecar_config_mapping_cache.clear()


# ---------------------------------------------------------------------------
# In-process registry helper
# ---------------------------------------------------------------------------


class TestMainConfigModelTypes:
    def test_known_text_types_present(self):
        types = _main_config_model_types()
        assert "llama" in types
        assert "gemma3" in types
        # 4.57 has a large registry.
        assert len(types) > 100

    def test_newer_only_types_absent(self):
        types = _main_config_model_types()
        # These ship only in transformers 5.x / as repo code.
        assert "glm4_moe_lite" not in types
        assert "gemma4_unified" not in types
        assert "deepseek_vl_v2" not in types


# ---------------------------------------------------------------------------
# transformers_version hint mapping
# ---------------------------------------------------------------------------


class TestTierFromVersion:
    @pytest.mark.parametrize(
        "tv,expected",
        [
            ("5.2.0.dev0", "530"),
            ("5.3.0", "530"),
            ("5.4.1", "550"),
            ("5.5.0", "550"),
            ("5.6", "510"),
            ("5.10.2", "510"),
            ("5.12.1", "510"),
            ("4.57.6", None),
            ("4.55.2", None),
            ("", None),
            (None, None),
            ("garbage", None),
        ],
    )
    def test_mapping(self, tv, expected):
        assert _tier_from_transformers_version(tv) == expected


# ---------------------------------------------------------------------------
# _dynamic_tier_from_cfg - the core decision (sidecars patched out / absent)
# ---------------------------------------------------------------------------


class TestDynamicTierFromCfg:
    def setup_method(self):
        _clear_caches()

    def test_none_config(self):
        assert _dynamic_tier_from_cfg(None) is None

    def test_known_type_is_default(self):
        # Recognized by in-process 4.57 -> default (None).
        assert _dynamic_tier_from_cfg({"model_type": "llama"}) is None
        assert _dynamic_tier_from_cfg({"model_type": "gemma3"}) is None

    def test_unknown_type_with_version_hint(self):
        with patch(
            "utils.transformers_version._enumerate_config_mapping", return_value = None
        ):
            cfg = {"model_type": "glm4_moe_lite", "transformers_version": "5.2.0.dev0"}
            assert _dynamic_tier_from_cfg(cfg) == "530"
            cfg2 = {"model_type": "some_future_arch", "transformers_version": "5.12.1"}
            assert _dynamic_tier_from_cfg(cfg2) == "510"

    def test_auto_map_repo_code_is_default(self):
        with patch(
            "utils.transformers_version._enumerate_config_mapping", return_value = None
        ):
            cfg = {
                "model_type": "deepseek_vl_v2",
                "transformers_version": "5.0.0",
                "auto_map": {"AutoConfig": "modeling_deepseekocr.DeepseekOCRConfig"},
            }
            # Repo-code arch: handled by trust_remote_code on 4.57 -> default.
            assert _dynamic_tier_from_cfg(cfg) is None

    def test_unknown_type_no_hint_is_default(self):
        with patch(
            "utils.transformers_version._enumerate_config_mapping", return_value = None
        ):
            cfg = {"model_type": "totally_unknown_arch"}
            assert _dynamic_tier_from_cfg(cfg) is None

    def test_sidecar_match_wins_over_hint(self):
        def fake_enum(venv_dir):
            # Only the 550 sidecar knows this type.
            if venv_dir.endswith("550"):
                return {"future_gemma"}
            return set()

        with patch(
            "utils.transformers_version._enumerate_config_mapping",
            side_effect = fake_enum,
        ):
            cfg = {"model_type": "future_gemma", "transformers_version": "5.12"}
            # Sidecar match (550) beats the version hint (which would say 510).
            assert _dynamic_tier_from_cfg(cfg) == "550"

    def test_registry_unavailable_defers(self):
        with patch(
            "utils.transformers_version._main_config_model_types", return_value = set()
        ):
            cfg = {"model_type": "glm4_moe_lite", "transformers_version": "5.2"}
            assert _dynamic_tier_from_cfg(cfg) is None


# ---------------------------------------------------------------------------
# get_transformers_tier - end to end via local config.json (no network)
# ---------------------------------------------------------------------------


class TestGetTransformersTierDynamic:
    def setup_method(self):
        _clear_caches()

    def _write_cfg(self, tmp_path, cfg):
        (tmp_path / "config.json").write_text(json.dumps(cfg))
        return str(tmp_path)

    def test_known_type_local_is_default(self, tmp_path):
        with patch(
            "utils.transformers_version._enumerate_config_mapping", return_value = None
        ):
            d = self._write_cfg(
                tmp_path, {"architectures": ["LlamaForCausalLM"], "model_type": "llama"}
            )
            assert get_transformers_tier(d) == "default"

    def test_unknown_type_local_routes_via_hint(self, tmp_path):
        with patch(
            "utils.transformers_version._enumerate_config_mapping", return_value = None
        ):
            d = self._write_cfg(
                tmp_path,
                {
                    "architectures": ["Glm4MoeLiteForCausalLM"],
                    "model_type": "glm4_moe_lite",
                    "transformers_version": "5.2.0.dev0",
                },
            )
            assert get_transformers_tier(d) == "530"

    def test_curated_510_still_wins(self, tmp_path):
        # gemma4_unified is caught by the curated 510 check BEFORE the dynamic
        # layer - confirm the curated decision is preserved.
        d = self._write_cfg(
            tmp_path,
            {
                "architectures": ["Gemma4UnifiedForConditionalGeneration"],
                "model_type": "gemma4_unified",
            },
        )
        assert get_transformers_tier(d) == "510"

    def test_auto_map_local_is_default(self, tmp_path):
        with patch(
            "utils.transformers_version._enumerate_config_mapping", return_value = None
        ):
            d = self._write_cfg(
                tmp_path,
                {
                    "architectures": ["DeepseekOCRForCausalLM"],
                    "model_type": "deepseek_vl_v2",
                    "transformers_version": "5.0.0",
                    "auto_map": {
                        "AutoConfig": "modeling_deepseekocr.DeepseekOCRConfig"
                    },
                },
            )
            assert get_transformers_tier(d) == "default"

    def test_no_unmocked_network_for_unknown_default(self, tmp_path):
        # Local llama with a parent dir name that contains no substring; the
        # dynamic layer must not trigger any network fetch.
        with patch("urllib.request.urlopen") as mock_urlopen:
            d = self._write_cfg(
                tmp_path, {"architectures": ["LlamaForCausalLM"], "model_type": "llama"}
            )
            assert get_transformers_tier(d) == "default"
            mock_urlopen.assert_not_called()


# ---------------------------------------------------------------------------
# Sidecar CONFIG_MAPPING enumeration + disk cache
# ---------------------------------------------------------------------------


class TestEnumerateConfigMapping:
    def setup_method(self):
        _clear_caches()

    def _make_fake_venv(self, tmp_path, version="5.5.0"):
        di = tmp_path / f"transformers-{version}.dist-info"
        di.mkdir(parents = True)
        (di / "METADATA").write_text(f"Name: transformers\nVersion: {version}\n")
        return str(tmp_path)

    def test_absent_venv_returns_none(self, tmp_path):
        missing = str(tmp_path / "does_not_exist")
        assert _enumerate_config_mapping(missing) is None

    def test_enumerates_and_caches_to_disk(self, tmp_path):
        venv = self._make_fake_venv(tmp_path)
        with patch(
            "utils.transformers_version._subprocess_dump_config_mapping",
            return_value = {"gemma4", "llama"},
        ) as mock_dump:
            result = _enumerate_config_mapping(venv)
        assert result == {"gemma4", "llama"}
        assert mock_dump.call_count == 1
        cache_file = Path(venv) / ".unsloth_config_mapping_cache.json"
        assert cache_file.is_file()
        data = json.loads(cache_file.read_text())
        assert data["transformers_version"] == "5.5.0"
        assert set(data["model_types"]) == {"gemma4", "llama"}

    def test_in_memory_cache_hit(self, tmp_path):
        venv = self._make_fake_venv(tmp_path)
        with patch(
            "utils.transformers_version._subprocess_dump_config_mapping",
            return_value = {"x"},
        ) as mock_dump:
            _enumerate_config_mapping(venv)
            _enumerate_config_mapping(venv)
        # Second call served from the in-memory cache.
        assert mock_dump.call_count == 1

    def test_disk_cache_hit_after_memory_clear(self, tmp_path):
        venv = self._make_fake_venv(tmp_path)
        with patch(
            "utils.transformers_version._subprocess_dump_config_mapping",
            return_value = {"x", "y"},
        ) as mock_dump:
            _enumerate_config_mapping(venv)
            assert mock_dump.call_count == 1
            _sidecar_config_mapping_cache.clear()  # drop the in-memory cache
            result = _enumerate_config_mapping(venv)
        # Served from the disk cache, subprocess not invoked again.
        assert mock_dump.call_count == 1
        assert result == {"x", "y"}

    def test_version_change_invalidates_disk_cache(self, tmp_path):
        venv = self._make_fake_venv(tmp_path, version = "5.5.0")
        with patch(
            "utils.transformers_version._subprocess_dump_config_mapping",
            return_value = {"old"},
        ):
            _enumerate_config_mapping(venv)
        # Bump the installed version + drop in-memory cache.
        for di in Path(venv).glob("transformers-*.dist-info"):
            import shutil

            shutil.rmtree(di)
        new_di = Path(venv) / "transformers-5.10.2.dist-info"
        new_di.mkdir()
        (new_di / "METADATA").write_text("Name: transformers\nVersion: 5.10.2\n")
        _sidecar_config_mapping_cache.clear()
        with patch(
            "utils.transformers_version._subprocess_dump_config_mapping",
            return_value = {"new"},
        ) as mock_dump:
            result = _enumerate_config_mapping(venv)
        assert mock_dump.call_count == 1  # re-enumerated due to version change
        assert result == {"new"}

    def test_subprocess_failure_returns_none(self, tmp_path):
        venv = self._make_fake_venv(tmp_path)
        with patch(
            "utils.transformers_version._subprocess_dump_config_mapping",
            return_value = None,
        ):
            assert _enumerate_config_mapping(venv) is None


# ---------------------------------------------------------------------------
# Backward-compat sanity: curated decisions unchanged
# ---------------------------------------------------------------------------


class TestBackwardCompatNeedsV5:
    def setup_method(self):
        _clear_caches()

    def test_substring_models_unchanged(self):
        assert needs_transformers_5("unsloth/gemma-4-12b-it") is True
        assert needs_transformers_5("google/gemma-4-E2B-it") is True

    def test_plain_llama_default(self):
        with patch(
            "utils.transformers_version._check_config_needs_550", return_value = False
        ), patch(
            "utils.transformers_version._check_config_needs_510", return_value = False
        ), patch(
            "utils.transformers_version._check_tokenizer_config_needs_v5",
            return_value = False,
        ):
            assert needs_transformers_5("meta-llama/Llama-3-8B") is False
