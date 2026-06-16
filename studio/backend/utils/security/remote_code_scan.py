# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Static scan of a model's ``auto_map`` remote code, for the consent gate.

When a user opts into ``trust_remote_code`` for a model whose architecture is
defined by Python shipped in its HF repo (``config.json`` ``auto_map`` ->
``modeling_*.py`` / ``configuration_*.py``), we scan that code BEFORE executing
it and surface any suspicious patterns so the consent decision is informed.

This is a *warning aid*, not a hard boundary -- a determined attacker can
obfuscate past regexes. Its job is to raise the bar and inform the explicit,
hash-pinned user consent. Containment is provided separately (subprocess/venv
isolation); execution still only happens with the user's opt-in.

Single source of truth: the detection logic is the repo's canonical
supply-chain auditor, ``scripts/scan_packages.py`` -- the exact scanner CI runs
(``security-audit.yml``). We import its ``check_py_file`` so the load-time gate
inherits every improvement to the CI scanner with no drift. The combination
heuristics there are deliberately low-false-positive (bare ``subprocess`` /
``eval`` alone are not flagged; staged-payload / reverse-shell / IMDS
combinations are). When ``scripts/`` is not present on disk (a stripped
packaged install), we fall back to the vendored ``_FALLBACK_PATTERNS`` below; a
test asserts the canonical scanner loads in-repo so the fallback never silently
takes over in the canonical environment.
"""

from __future__ import annotations

import hashlib
import importlib.util
import pathlib
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

from loggers import get_logger

logger = get_logger(__name__)

CRITICAL = "CRITICAL"
HIGH = "HIGH"
MEDIUM = "MEDIUM"
_SEVERITY_ORDER = {CRITICAL: 0, HIGH: 1, MEDIUM: 2}

# --- Fallback patterns (only used if scripts/scan_packages.py is absent) -----
# (regex, check-name, severity). A flat subset of the canonical scanner, kept so
# a stripped packaged install without scripts/ still scans. The canonical
# scanner (imported below) supersedes this whenever the repo is present.
_FALLBACK_PATTERNS: tuple[tuple[re.Pattern, str, str], ...] = (
    (re.compile(
        r"\bexec\s*\(\s*(?:urllib|requests|httpx|urlopen)"
        r"|\bexec\s*\([^)]*\.(?:text|content|read)\s*\("
        r"|\beval\s*\([^)]*\.(?:text|content|read)\s*\("
        r"|\b__import__\s*\([^)]*\+", re.DOTALL),
        "loads-and-executes-remote-code", CRITICAL),
    (re.compile(
        r"\bsocket\b.*\bconnect\b.*\bsubprocess\b"
        r"|\bsocket\b.*\bconnect\b.*\b(?:sh|bash|cmd)\b"
        r"|\bpty\s*\.\s*spawn\b|\bos\s*\.\s*dup2\s*\(", re.DOTALL),
        "reverse/bind-shell", CRITICAL),
    (re.compile(
        r"169\.254\.169\.254|metadata\.google\.internal|/latest/meta-data"
        r"|/metadata/identity|169\.254\.170\.2"),
        "cloud-metadata/IMDS-access", CRITICAL),
    (re.compile(
        r"(?:open|Path|read_text|read_bytes)\s*\([^)]*?"
        r"(?:\.ssh[/\\]|\.aws[/\\]|\.kube[/\\]|\.gnupg[/\\]|id_rsa|id_ed25519"
        r"|credentials\.json|\.git-credentials|\.npmrc|\.pypirc|/etc/shadow)"
        r"|(?:open|Path)\(\s*['\"]\.env['\"]\s*[,)]", re.DOTALL),
        "credential-file-access", CRITICAL),
    (re.compile(r"/tmp/\S+.*(?:subprocess|os\.system|os\.popen|Popen|chmod.*\+x)", re.DOTALL),
        "tmp-staged-dropper", CRITICAL),
    (re.compile(r"\bopenssl\s+(enc|rand|rsautl|pkeyutl|genrsa|dgst|s_client)\b"),
        "openssl-cli-exfil", HIGH),
    (re.compile(
        r"\bsubprocess\s*\.\s*(Popen|call|run|check_call|check_output)\b"
        r"|\bos\s*\.\s*(system|popen|exec[lv]p?e?)\b"),
        "subprocess/os-exec", HIGH),
    # Bare exec()/eval() only; exclude attribute calls like torch's
    # ``module.eval()`` (eval-mode), which are ubiquitous false positives.
    (re.compile(r"(?<![\w.])(?:exec|eval)\s*\("), "exec/eval", HIGH),
    (re.compile(
        r"\burllib\.request\b|\burlopen\s*\("
        r"|\brequests\s*\.\s*(get|post|put|patch|delete|head|Session)\b"
        r"|\bhttpx\s*\.\s*(get|post|put|patch|delete|Client|AsyncClient)\b"
        r"|\bsocket\s*\.\s*(socket|create_connection)\b|\bhttp\.client\b"),
        "network-access", HIGH),
    (re.compile(
        r"\bmarshal\s*\.\s*(loads|load)\b"
        r"|\bcompile\s*\([^)]*['\"]exec['\"]\s*\)"
        r"|\b__import__\s*\(|\bgetattr\s*\(\s*__builtins__"
        r"|\b(?:b64decode|decodebytes)\s*\(.*(?:b64decode|decodebytes)\s*\(", re.DOTALL),
        "obfuscation", HIGH),
    (re.compile(
        r"-----BEGIN\s+(?:RSA\s+)?(?:PUBLIC|PRIVATE|ENCRYPTED|EC|DSA|OPENSSH)\s+KEY-----"
        r"|\bMII[A-Za-z0-9+/]{20,}", re.DOTALL),
        "embedded-key-material", HIGH),
    (re.compile(
        r"\bos\.environ\s*\.\s*copy\s*\(|\bdict\s*\(\s*os\.environ\s*\)"
        r"|\bjson\.dumps\s*\(\s*(?:dict\s*\(\s*)?os\.environ", re.IGNORECASE),
        "environment-harvest", HIGH),
    (re.compile(
        r"\bbase64\s*\.\s*(b64decode|decodebytes|b32decode|b16decode)\b"
        r"|\bcodecs\s*\.\s*decode\b"),
        "base64/encoding-decode", MEDIUM),
    (re.compile(r"[A-Za-z0-9+/=]{200,}"), "large-base64-blob", MEDIUM),
)


@dataclass
class Finding:
    severity: str
    filename: str
    check: str
    evidence: str = ""


@dataclass
class ScanResult:
    findings: list[Finding] = field(default_factory=list)
    fingerprint: str = ""

    @property
    def max_severity(self) -> Optional[str]:
        if not self.findings:
            return None
        return min((f.severity for f in self.findings), key=lambda s: _SEVERITY_ORDER[s])

    @property
    def clean(self) -> bool:
        return not self.findings

    def summary(self) -> str:
        if self.clean:
            return "no suspicious patterns found"
        by = {}
        for f in self.findings:
            by.setdefault(f.severity, set()).add(f.check)
        parts = []
        for sev in (CRITICAL, HIGH, MEDIUM):
            if sev in by:
                parts.append(f"{sev}: {', '.join(sorted(by[sev]))}")
        return "; ".join(parts)


# --- Canonical scanner: import scripts/scan_packages.py (the CI scanner) ------
# Loaded by file path (scripts/ is not an importable package from the backend
# root). scan_packages.py imports only stdlib at module level (requests/etc. are
# lazy) and guards its CLI under ``if __name__ == "__main__"``, so importing it
# is side-effect-free and dependency-light.
_CANON_SENTINEL = object()
_canon_cache = _CANON_SENTINEL


def _load_canonical_scanner():
    """Return the ``scripts/scan_packages.py`` module, or None if unavailable."""
    global _canon_cache
    if _canon_cache is not _CANON_SENTINEL:
        return _canon_cache

    module = None
    # Walk up from this file to a repo root that contains scripts/scan_packages.py.
    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "scripts" / "scan_packages.py"
        if candidate.is_file():
            try:
                spec = importlib.util.spec_from_file_location(
                    "unsloth_scan_packages", candidate
                )
                mod = importlib.util.module_from_spec(spec)
                sys.modules.setdefault("unsloth_scan_packages", mod)
                spec.loader.exec_module(mod)  # type: ignore[union-attr]
                if hasattr(mod, "check_py_file"):
                    module = mod
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Could not load canonical scan_packages.py: %s", exc)
            break

    if module is None:
        logger.warning(
            "scripts/scan_packages.py not found; remote-code scan using the "
            "vendored fallback patterns."
        )
    _canon_cache = module
    return module


# Model-context-strict patterns. The canonical scanner is tuned for *package*
# supply chain, where a bare ``subprocess`` (build scripts) or ``eval`` is
# common, so it only flags them in combinations. In a model's modeling_*.py
# none of these have a legitimate place -- a forward pass never shells out --
# so the load-time gate flags them on their own. This catches, for example, a
# bare ``subprocess.Popen`` sitting in a config ``__init__``.
_MODEL_STRICT_PATTERNS: tuple[tuple[re.Pattern, str, str], ...] = (
    (re.compile(
        r"\bsubprocess\s*\.\s*(Popen|call|run|check_call|check_output)\b"
        r"|\bos\s*\.\s*(system|popen|exec[lv]p?e?)\b"),
        "subprocess/os-exec (model code)", HIGH),
    # Bare exec()/eval(); excludes attribute calls like torch ``module.eval()``.
    (re.compile(r"(?<![\w.])(?:exec|eval)\s*\("),
        "exec/eval (model code)", HIGH),
)


def _scan_content(content: str, filename: str) -> list[Finding]:
    findings: list[Finding] = []

    canon = _load_canonical_scanner()
    if canon is not None:
        # Canonical Finding is (severity, package, filename, check, evidence);
        # adapt to the gate's (severity, filename, check, evidence).
        for f in canon.check_py_file(content, filename, ""):
            findings.append(Finding(f.severity, f.filename, f.check, (f.evidence or "")[:120]))
    else:
        for pat, check, sev in _FALLBACK_PATTERNS:
            m = pat.search(content)
            if m:
                findings.append(Finding(sev, filename, check, m.group(0)[:120]))

    # Augment with the model-context-strict patterns the package scanner omits.
    have = {f.check for f in findings}
    for pat, check, sev in _MODEL_STRICT_PATTERNS:
        if check in have:
            continue
        m = pat.search(content)
        if m:
            findings.append(Finding(sev, filename, check, m.group(0)[:120]))
    return findings


def scan_remote_code_files(files: dict[str, str]) -> ScanResult:
    """Scan a mapping of {filename: content} and return aggregated findings."""
    result = ScanResult(fingerprint=remote_code_fingerprint(files))
    for name, content in files.items():
        if not name.endswith(".py"):
            continue
        result.findings.extend(_scan_content(content or "", name))
    return result


def remote_code_fingerprint(files: dict[str, str]) -> str:
    """Stable sha256 over the (sorted) file contents, for pinning consent."""
    h = hashlib.sha256()
    for name in sorted(files):
        h.update(name.encode("utf-8"))
        h.update(b"\0")
        h.update((files[name] or "").encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def repo_remote_code_files(model_name: str, hf_token: Optional[str] = None) -> dict[str, str]:
    """Download a repo's executable ``.py`` (auto_map targets + modeling/config).

    Returns {filename: content}. Returns {} on any error / offline so callers can
    treat an unscannable repo as "unknown" and warn accordingly.
    """
    import json
    from pathlib import Path

    files: dict[str, str] = {}
    try:
        from utils.paths import is_local_path, normalize_path

        if is_local_path(model_name):
            root = Path(normalize_path(model_name)).expanduser()
            cfg_path = root / "config.json"
            cfg = json.loads(cfg_path.read_text()) if cfg_path.is_file() else {}
            wanted = _auto_map_py(cfg) | {p.name for p in root.glob("*.py")}
            for fn in wanted:
                fp = root / fn
                if fp.is_file():
                    files[fn] = fp.read_text(errors="replace")
            return files

        from huggingface_hub import hf_hub_download, list_repo_files

        cfg_path = hf_hub_download(model_name, "config.json", token=hf_token)
        cfg = json.loads(Path(cfg_path).read_text())
        wanted = set(_auto_map_py(cfg))
        try:
            wanted |= {f for f in list_repo_files(model_name, token=hf_token)
                       if f.endswith(".py")}
        except Exception:
            pass
        for fn in wanted:
            try:
                fp = hf_hub_download(model_name, fn, token=hf_token)
                files[fn] = Path(fp).read_text(errors="replace")
            except Exception:
                continue
    except Exception as exc:
        logger.warning("repo_remote_code_files(%s): could not fetch (%s)", model_name, exc)
        return {}
    return files


def _auto_map_py(cfg: dict) -> set[str]:
    """Extract ``modeling_x.py`` filenames referenced by config.json auto_map."""
    out: set[str] = set()
    am = cfg.get("auto_map") or {}
    if isinstance(am, dict):
        for ref in am.values():
            # ref like "modeling_deepseekocr.DeepseekOCRForCausalLM"
            if isinstance(ref, str) and "." in ref:
                module = ref.rsplit(".", 1)[0]
                out.add(module.split("--")[-1] + ".py")
    return out
