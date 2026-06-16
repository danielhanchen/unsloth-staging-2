# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Security helpers for the ``trust_remote_code`` boundary.

Two orthogonal questions, two helpers:

* ``trusted_org.is_trusted_org_repo`` -- may we *auto-enable* remote code for
  this name without asking? Only for genuine first-party (``unsloth``/
  ``nvidia``) Hub repos; never for local paths or spoofed names.

* ``remote_code_scan`` -- *what* would run if the user opts in? Statically scan
  the repo's ``auto_map`` Python and surface findings for an informed decision.

Intended consent flow for a deliberate LOAD (train / infer / export), wired by
the load paths:

    1. Try to load with ``trust_remote_code=False`` (safe default).
    2. If transformers raises the "requires `trust_remote_code=True`" error
       (architecture defined as repo code via ``auto_map``):
         a. files = repo_remote_code_files(model_name)
         b. result = scan_remote_code_files(files)
         c. Surface result.summary() + result.fingerprint to the user and
            require explicit consent. ``should_block_remote_code(result)``
            recommends blocking by default on CRITICAL/HIGH findings.
         d. Remember approval pinned to result.fingerprint; re-prompt if the
            code changes on a later load.
    3. On approval, retry with ``trust_remote_code=True``; otherwise fail.

Detection (is-vision / version / size) never enters this flow -- it reads raw
``config.json`` and never needs remote code.
"""

from utils.security.consent import (  # noqa: F401
    RemoteCodeDecision,
    evaluate_remote_code_consent,
)
from utils.security.remote_code_scan import (  # noqa: F401
    CRITICAL,
    HIGH,
    MEDIUM,
    Finding,
    ScanResult,
    remote_code_fingerprint,
    repo_remote_code_files,
    scan_remote_code_files,
)
from utils.security.trusted_org import is_trusted_org_repo  # noqa: F401

__all__ = [
    "is_trusted_org_repo",
    "scan_remote_code_files",
    "repo_remote_code_files",
    "remote_code_fingerprint",
    "should_block_remote_code",
    "evaluate_remote_code_consent",
    "RemoteCodeDecision",
    "ScanResult",
    "Finding",
    "CRITICAL",
    "HIGH",
    "MEDIUM",
]


def should_block_remote_code(result: "ScanResult") -> bool:
    """Recommend blocking-by-default when the scan found CRITICAL/HIGH patterns.

    Non-blocking by itself: the caller still surfaces findings and takes explicit
    user consent. A clean/MEDIUM result returns False (warn but allow with consent);
    CRITICAL/HIGH returns True (block unless the user force-overrides).
    """
    sev = result.max_severity
    return sev in (CRITICAL, HIGH)
