import sys
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from unsloth import save as save_module


class _Sibling:
    def __init__(self, name):
        self.rfilename = name


class _RepoInfo:
    def __init__(self, siblings):
        self.siblings = siblings


class _FakeTok:
    def __init__(self, name_or_path):
        self.name_or_path = name_or_path


def _install_fake_hfapi(monkeypatch, siblings, counter):
    class FakeApi:
        def __init__(self, token = None):
            pass

        def model_info(self, source, files_metadata = False):
            counter["n"] += 1
            return _RepoInfo(siblings)

    monkeypatch.setattr(save_module, "HfApi", FakeApi)


def test_negative_lookup_is_cached(monkeypatch):
    save_module._TOKENIZER_MODEL_CACHE.clear()
    counter = {"n": 0}
    _install_fake_hfapi(monkeypatch, [_Sibling("tokenizer.json")], counter)

    tok = _FakeTok("fake-org/non-spm-repo")
    assert save_module._has_tokenizer_model(tok) is False
    assert save_module._has_tokenizer_model(tok) is False
    assert save_module._has_tokenizer_model(tok) is False
    assert counter["n"] == 1


def test_exception_path_is_cached(monkeypatch):
    save_module._TOKENIZER_MODEL_CACHE.clear()
    counter = {"n": 0}

    class RaisingApi:
        def __init__(self, token = None):
            pass

        def model_info(self, source, files_metadata = False):
            counter["n"] += 1
            raise RuntimeError("offline")

    monkeypatch.setattr(save_module, "HfApi", RaisingApi)

    tok = _FakeTok("fake-org/offline-repo")
    assert save_module._has_tokenizer_model(tok) is False
    assert save_module._has_tokenizer_model(tok) is False
    assert counter["n"] == 1


def test_siblings_none_does_not_raise(monkeypatch):
    save_module._TOKENIZER_MODEL_CACHE.clear()
    counter = {"n": 0}
    _install_fake_hfapi(monkeypatch, None, counter)

    tok = _FakeTok("fake-org/siblings-none-repo")
    assert save_module._has_tokenizer_model(tok) is False


def test_positive_lookup_is_cached(monkeypatch):
    save_module._TOKENIZER_MODEL_CACHE.clear()
    counter = {"n": 0}
    _install_fake_hfapi(
        monkeypatch,
        [_Sibling("tokenizer.json"), _Sibling("tokenizer.model")],
        counter,
    )

    tok = _FakeTok("fake-org/spm-repo")
    assert save_module._has_tokenizer_model(tok) is True
    assert save_module._has_tokenizer_model(tok) is True
    assert counter["n"] == 1
