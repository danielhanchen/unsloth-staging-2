"""A sentence-transformers model can have weights at the root AND in subfolders.

`weights_at_root` splits repos two ways -- root weights, or per-subfolder weights
-- and `unsloth/embeddinggemma-300m` is both. It ships a root `model.safetensors`
(so `weights_at_root` is True) plus `2_Dense/model.safetensors` and
`3_Dense/model.safetensors`, which the ST load reads as part of the model.

Landing in the `weights_at_root` branch added `_SUBDIR_WEIGHT_IGNORE_PATTERNS`,
whose `*/*.safetensors` excluded both Dense weights from the download. The
post-download gate in unsloth_zoo then did exactly its job -- a KNOWN
weight-bearing ST module with no weight is breakage -- so the download was
retried over HTTP, excluded the same two files for the same reason, and raised

    DownloadStallError: Download for 'unsloth/embeddinggemma-300m' returned an
    incomplete snapshot even with HF_HUB_DISABLE_XET=1 -- missing files, check
    your network connection

The retry could never have succeeded: the request itself guaranteed the files
would be absent, and the message blamed the network. `EmbeddingGemma_(300M)`
passed three sweeps, then failed five in a row on two different backends, which
is what a deterministic bug wearing an infra label looks like.

Reproduced on a cold cache and confirmed fixed by the change under test here.
These tests are offline: the hub call is stubbed, because a test whose answer
depends on the network is a test that eventually reports a bug that is not there.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unsloth.models._utils as U  # noqa: E402


ROOT_ONLY = [
    {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
    {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
]

EMBEDDINGGEMMA = ROOT_ONLY + [
    {"idx": 2, "name": "2", "path": "2_Dense", "type": "sentence_transformers.models.Dense"},
    {"idx": 3, "name": "3", "path": "3_Dense", "type": "sentence_transformers.models.Dense"},
    {"idx": 4, "name": "4", "path": "", "type": "sentence_transformers.models.Normalize"},
]


@pytest.fixture
def modules_json(tmp_path, monkeypatch):
    """Stub hf_hub_download so it hands back a modules.json we control."""

    def _install(payload):
        if payload is None:                       # repo ships no modules.json
            def boom(*a, **k):
                raise OSError("404 modules.json")
            monkeypatch.setattr(U, "hf_hub_download", boom, raising=False)
            import huggingface_hub
            monkeypatch.setattr(huggingface_hub, "hf_hub_download", boom)
            return
        p = tmp_path / "modules.json"
        p.write_text(payload if isinstance(payload, str) else json.dumps(payload),
                     encoding="utf-8")
        import huggingface_hub
        monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                            lambda *a, **k: str(p))

    return _install


# ---- detection -----------------------------------------------------------

def test_embeddinggemma_layout_is_detected(modules_json):
    modules_json(EMBEDDINGGEMMA)
    assert U._repo_has_weighted_st_subfolders("unsloth/embeddinggemma-300m") is True


def test_a_root_only_st_model_is_not(modules_json):
    """Pooling lives in a subfolder but holds no weight, so nothing is at risk
    and the existing subdir pruning should stay in force."""
    modules_json(ROOT_ONLY)
    assert U._repo_has_weighted_st_subfolders("org/plain-st") is False


def test_a_repo_without_modules_json_is_not(modules_json):
    """The overwhelming majority. A plain causal LM must keep the old
    behaviour exactly, so a fix for one notebook does not enlarge 400 other
    downloads."""
    modules_json(None)
    assert U._repo_has_weighted_st_subfolders("unsloth/Qwen3-0.6B") is False


@pytest.mark.parametrize("payload", [
    "{ not json",
    json.dumps({"not": "a list"}),
    json.dumps([None, 3, "x"]),
    json.dumps([{"path": "2_Dense"}]),                       # no type
    json.dumps([{"type": "...Dense"}]),                      # no path
    json.dumps([{"path": "  ", "type": "...Dense"}]),        # blank path
    json.dumps([{"path": "/", "type": "...Dense"}]),         # root, slash only
])
def test_malformed_modules_json_falls_back_to_the_old_behaviour(modules_json, payload):
    """Best-effort by design: anything unreadable must not start failing loads
    that work today."""
    modules_json(payload)
    assert U._repo_has_weighted_st_subfolders("org/whatever") is False


def test_an_unknown_subfolder_module_type_is_not_assumed_weighted(modules_json):
    modules_json([{"path": "2_Custom", "type": "mypkg.WeirdModule"}])
    assert U._repo_has_weighted_st_subfolders("org/custom") is False


@pytest.mark.parametrize("leaf", ["Dense", "CNN", "LSTM", "dense"])
def test_every_weight_bearing_type_counts(modules_json, leaf):
    modules_json([{"path": f"2_{leaf}", "type": f"sentence_transformers.models.{leaf}"}])
    assert U._repo_has_weighted_st_subfolders("org/x") is True


def test_the_taxonomy_is_shared_with_unsloth_zoo_not_restated():
    """If these two ever disagree, unsloth would fetch a module the gate then
    rejects, or prune one it demands -- the exact shape of the original bug."""
    src = (Path(U.__file__)).read_text(encoding="utf-8")
    assert "_ST_WEIGHTED_MODULE_TYPES" in src
    assert '"dense"' not in src.split("_repo_has_weighted_st_subfolders")[1][:2000]


# ---- the behaviour that actually changed ---------------------------------

def _ignores(model_name, monkeypatch, **kw):
    """The ignore_patterns `maybe_prefetch_hf_snapshot` actually sends.

    Driven through the real function with the downloader stubbed, rather than
    through `_prefetch_ignore_patterns` -- that builds the static skip list and
    knows nothing about the subdir branch, so asserting on it would have tested
    the wrong function and passed either way.
    """
    seen = {}

    def fake_download(name, **kwargs):
        seen.update(kwargs)
        return "/nonexistent/snapshot"

    import unsloth_zoo.hf_xet_fallback as XF
    monkeypatch.setattr(XF, "snapshot_download_with_xet_fallback", fake_download)
    U.maybe_prefetch_hf_snapshot(model_name, weights_at_root=True, **kw)
    assert seen, "the downloader was never reached; the call bailed out early"
    return list(seen.get("ignore_patterns") or [])


def test_the_subdir_weight_patterns_are_dropped_for_such_a_repo(modules_json,
                                                                monkeypatch):
    modules_json(EMBEDDINGGEMMA)
    got = _ignores("unsloth/embeddinggemma-300m", monkeypatch)
    assert "*/*.safetensors" not in got, got


def test_the_subdir_weight_patterns_are_kept_for_everything_else(modules_json,
                                                                 monkeypatch):
    """The other half of the claim. Without this, the test above would pass
    just as well if the patterns had been deleted outright."""
    modules_json(None)
    got = _ignores("unsloth/Qwen3-0.6B", monkeypatch)
    assert "*/*.safetensors" in got, got


def test_only_the_subdir_weight_patterns_differ(modules_json, monkeypatch):
    """The fix must not quietly change anything else about the request."""
    modules_json(None)
    plain = set(_ignores("unsloth/Qwen3-0.6B", monkeypatch))
    modules_json(EMBEDDINGGEMMA)
    st = set(_ignores("unsloth/embeddinggemma-300m", monkeypatch))
    assert plain - st == set(U._SUBDIR_WEIGHT_IGNORE_PATTERNS)
    assert st - plain == set()


def test_the_patterns_still_exist(modules_json):
    """They are correct for the case they were written for -- an fp16/ or
    experimental/ directory a root load never reads. This fix narrows where
    they apply, it does not retire them."""
    assert "*/*.safetensors" in U._SUBDIR_WEIGHT_IGNORE_PATTERNS
    assert "*/*.bin" in U._SUBDIR_WEIGHT_IGNORE_PATTERNS


def test_a_hub_failure_keeps_the_patterns(monkeypatch):
    """Network trouble must not silently enlarge every download."""
    import huggingface_hub
    def boom(*a, **k):
        raise RuntimeError("hub down")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", boom)
    got = _ignores("org/anything", monkeypatch)
    assert "*/*.safetensors" in got


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
