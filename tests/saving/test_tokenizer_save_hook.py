import sys
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pytest

from unsloth import save as save_module


def _load_tiny_tokenizer():
    try:
        from transformers import AutoTokenizer
    except Exception:
        pytest.skip("transformers not available")
    try:
        return AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    except Exception:
        pytest.skip("tiny llama tokenizer not available (offline)")


def test_legacy_format_false_skips_preservation(tmp_path, monkeypatch):
    tok = _load_tiny_tokenizer()
    monkeypatch.setattr(save_module, "_has_tokenizer_model", lambda *a, **k: False)
    save_module.patch_saving_functions(tok)

    out_false = tmp_path / "json_only"
    out_false.mkdir()
    tok.save_pretrained(str(out_false), legacy_format = False)
    assert not (out_false / "tokenizer.model").exists()

    out_default = tmp_path / "default"
    out_default.mkdir()
    tok.save_pretrained(str(out_default))
    assert (out_default / "tokenizer.model").exists()


def test_push_to_hub_calls_upload_folder(tmp_path, monkeypatch):
    tok = _load_tiny_tokenizer()
    monkeypatch.setattr(save_module, "_has_tokenizer_model", lambda *a, **k: False)

    recorded = {"create": 0, "upload": 0, "folder": None, "repo_id": None}

    class FakeApi:
        def __init__(self, token = None):
            self.token = token

        def create_repo(self, **kwargs):
            recorded["create"] += 1
            return None

        def upload_folder(self, *, folder_path, repo_id, **kwargs):
            recorded["upload"] += 1
            recorded["folder"] = str(folder_path)
            recorded["repo_id"] = repo_id
            return None

    monkeypatch.setattr(save_module, "HfApi", FakeApi)
    save_module.patch_saving_functions(tok)

    out = tmp_path / "push"
    out.mkdir()
    tok.save_pretrained(str(out), push_to_hub = True, repo_id = "user/fake-repo")

    assert recorded["upload"] == 1
    assert recorded["repo_id"] == "user/fake-repo"
    assert recorded["folder"] == str(out)
    assert (out / "tokenizer.model").exists()
