import sys
import os
import json

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from unsloth import save as save_module


class _AddedTok:
    def __init__(self, content, special):
        self.content = content
        self.single_word = False
        self.lstrip = False
        self.rstrip = False
        self.normalized = False
        self.special = special


class _FakeTok:
    def __init__(self, name_or_path = "fake/repo", vocab_file = None, added_tokens_decoder = None):
        self.name_or_path = name_or_path
        if vocab_file is not None:
            self.vocab_file = vocab_file
        self.added_tokens_decoder = added_tokens_decoder or {}


def test_filename_prefix_honored(tmp_path, monkeypatch):
    save_dir = tmp_path / "out"
    save_dir.mkdir()
    (save_dir / "v2-tokenizer_config.json").write_text(
        json.dumps({"added_tokens_decoder": {"105": {"content": "<s>", "special": False}}})
    )

    monkeypatch.setattr(save_module, "_has_tokenizer_model", lambda *a, **k: False)

    tok = _FakeTok(
        added_tokens_decoder = {105: _AddedTok("<start_of_turn>", special = True)},
    )
    save_module._preserve_sentencepiece_tokenizer_assets(
        tok, str(save_dir), filename_prefix = "v2",
    )

    cfg = json.loads((save_dir / "v2-tokenizer_config.json").read_text())
    assert cfg["added_tokens_decoder"]["105"]["special"] is True
    assert cfg["added_tokens_decoder"]["105"]["content"] == "<start_of_turn>"
    assert not (save_dir / "tokenizer_config.json").exists()


def test_vocab_file_used_before_network(tmp_path, monkeypatch):
    save_dir = tmp_path / "out"
    save_dir.mkdir()
    cached = tmp_path / "cached_tokenizer.model"
    cached.write_bytes(b"spm-source-bytes")

    def boom(*a, **k):
        raise AssertionError("_has_tokenizer_model should not be consulted when vocab_file exists")

    monkeypatch.setattr(save_module, "_has_tokenizer_model", boom)

    tok = _FakeTok(vocab_file = str(cached))
    save_module._preserve_sentencepiece_tokenizer_assets(tok, str(save_dir))

    assert (save_dir / "tokenizer.model").read_bytes() == b"spm-source-bytes"


def test_existing_tokenizer_model_not_overwritten(tmp_path, monkeypatch):
    save_dir = tmp_path / "out"
    save_dir.mkdir()
    (save_dir / "tokenizer.model").write_bytes(b"already-saved")

    def boom(*a, **k):
        raise AssertionError("_has_tokenizer_model must not run when target already exists")

    monkeypatch.setattr(save_module, "_has_tokenizer_model", boom)

    tok = _FakeTok()
    save_module._preserve_sentencepiece_tokenizer_assets(tok, str(save_dir))

    assert (save_dir / "tokenizer.model").read_bytes() == b"already-saved"


def test_prefixed_tokenizer_model_target_name(tmp_path, monkeypatch):
    save_dir = tmp_path / "out"
    save_dir.mkdir()
    cached = tmp_path / "cached.model"
    cached.write_bytes(b"spm-bytes")

    monkeypatch.setattr(save_module, "_has_tokenizer_model", lambda *a, **k: False)

    tok = _FakeTok(vocab_file = str(cached))
    save_module._preserve_sentencepiece_tokenizer_assets(
        tok, str(save_dir), filename_prefix = "v2",
    )

    assert (save_dir / "v2-tokenizer.model").read_bytes() == b"spm-bytes"
    assert not (save_dir / "tokenizer.model").exists()
