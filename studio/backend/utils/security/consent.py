# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Consent gate for loads that would execute model repo code.

This is the LOAD-path counterpart to the capability probes. Detection never
calls this: it reads raw ``config.json`` and never needs remote code. A
deliberate load (train / infer / export) calls ``evaluate_remote_code_consent``
right before it would pass ``trust_remote_code=True`` to the loader.

The gate answers: *is it safe to run this model's repo code now?*

* No ``auto_map`` in ``config.json`` -> ``trust_remote_code`` executes nothing;
  allow (``has_remote_code=False``).
* ``auto_map`` present -> statically scan the repo's ``.py`` (reusing
  ``remote_code_scan``) and decide by severity and provenance:
    - CRITICAL findings (reverse shell, cloud-metadata/IMDS, credential theft,
      remote-code loaders, droppers) -> block even for first-party repos
      (defense against a compromised trusted repo), unless pinned-approved.
    - HIGH findings (``subprocess``/``exec``/``eval``/network/``b64decode``) are
      common in legitimate modeling code (DeepSeek-OCR, Kimi, etc. all trip
      HIGH), so they block only for UNTRUSTED third-party repos. First-party
      ``unsloth``/``nvidia`` repos (``is_trusted_org_repo``) load through -- the
      org is the trust anchor.
  In every blocking case the caller surfaces ``findings_summary`` +
  ``fingerprint`` so the user can make an informed, pinned decision and retry.
* ``auto_map`` present but the code cannot be fetched (gated/offline) -> cannot
  verify; allow with a warning, since the load only reaches here on an explicit
  opt-in (user toggle or trusted-org auto-enable). We never silently *run* code
  the scan flagged, but we do not break legitimate gated repos either.

The gate is hardening + consent-driven, not a hard sandbox: a determined attacker
can obfuscate past static patterns. Its job is to raise the bar and inform
consent; subprocess/venv isolation remains the containment layer.
"""

from dataclasses import dataclass
from typing import Optional

from loggers import get_logger

from utils.security.remote_code_scan import (
    CRITICAL,
    HIGH,
    remote_code_fingerprint,
    repo_remote_code_files,
    scan_remote_code_files,
)

logger = get_logger(__name__)


@dataclass
class RemoteCodeDecision:
    """Outcome of the consent gate for one (model, trust_remote_code) load."""

    model_name: str
    has_remote_code: bool
    blocked: bool
    fingerprint: Optional[str]
    max_severity: Optional[str]
    findings_summary: str
    reason: str

    def response_payload(self) -> dict:
        """Machine-readable detail for the frontend to render + pin approval."""
        return {
            "error_kind": "remote_code_blocked",
            "model_name": self.model_name,
            "fingerprint": self.fingerprint,
            "max_severity": self.max_severity,
            "findings": self.findings_summary,
            "reason": self.reason,
        }


def _config_has_auto_map(model_name: str, hf_token: Optional[str] = None) -> bool:
    """True if the model's ``config.json`` declares an ``auto_map`` (repo code).

    Reads raw JSON only (never executes). On any error returns False so we do not
    over-block; if there really is unfetchable code the file-scan branch handles
    it.
    """
    try:
        from utils.transformers_version import _load_config_json

        cfg = _load_config_json(model_name) or {}
        return bool(cfg.get("auto_map"))
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("auto_map check failed for %s: %s", model_name, exc)
        return False


def evaluate_remote_code_consent(
    model_name: str,
    hf_token: Optional[str] = None,
    *,
    trust_remote_code: bool,
    approved_fingerprint: Optional[str] = None,
    trusted_org: Optional[bool] = None,
) -> RemoteCodeDecision:
    """Decide whether a ``trust_remote_code=True`` load may proceed.

    Call this immediately before a load that would execute repo code. When the
    returned decision is ``blocked``, the caller must refuse the load and surface
    ``response_payload()`` so the user can review the findings and, if they
    accept, re-issue the load with ``approved_fingerprint`` set to
    ``decision.fingerprint``.

    ``trusted_org`` may be supplied if the caller already computed
    ``is_trusted_org_repo``; otherwise it is resolved lazily, and only when it
    matters (a HIGH-severity result), to avoid an extra Hub call.
    """
    if not trust_remote_code:
        return RemoteCodeDecision(
            model_name, False, False, None, None, "", "trust_remote_code disabled"
        )

    # ``trust_remote_code`` only executes code when the repo defines an auto_map.
    if not _config_has_auto_map(model_name, hf_token):
        return RemoteCodeDecision(
            model_name, False, False, None, None, "",
            "no auto_map; trust_remote_code is a no-op",
        )

    files = repo_remote_code_files(model_name, hf_token=hf_token)
    if not files:
        # auto_map present but unscannable (gated/offline). Reached only on an
        # explicit opt-in; allow but record that we could not verify.
        logger.warning(
            "Remote code for '%s' could not be fetched to scan; allowing on "
            "explicit opt-in but it was NOT verified.",
            model_name,
        )
        return RemoteCodeDecision(
            model_name, True, False, None, None,
            "Remote code present (auto_map) but could not be downloaded to scan.",
            "unscannable; allowed via explicit opt-in",
        )

    result = scan_remote_code_files(files)
    fingerprint = remote_code_fingerprint(files)
    sev = result.max_severity
    approved = approved_fingerprint is not None and approved_fingerprint == fingerprint

    if approved:
        blocked, reason = False, "approved by fingerprint"
    elif sev == CRITICAL:
        # High-confidence malicious patterns: block even first-party repos.
        blocked, reason = True, "blocked: scan found CRITICAL patterns"
    elif sev == HIGH:
        # exec/eval/subprocess/b64decode are common in legitimate modeling code,
        # so HIGH blocks only for untrusted third-party repos. First-party repos
        # (the org is the trust anchor) load through.
        trusted = trusted_org
        if trusted is None:
            from utils.security.trusted_org import is_trusted_org_repo

            trusted = is_trusted_org_repo(model_name, hf_token=hf_token)
        if trusted:
            blocked, reason = False, "allowed: first-party repo (HIGH findings hardening)"
        else:
            blocked, reason = True, "blocked: scan found HIGH patterns in third-party repo"
    else:
        blocked, reason = False, "allowed: no high-risk patterns"

    if blocked:
        logger.warning(
            "Blocking trust_remote_code load of '%s': scan severity %s (fingerprint %s)",
            model_name, sev, fingerprint[:12],
        )

    return RemoteCodeDecision(
        model_name,
        True,
        blocked,
        fingerprint,
        sev,
        result.summary(),
        reason,
    )
