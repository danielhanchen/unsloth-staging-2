"""Real-world names in a public copy of the toolkit.

The AMD fork and the staging repos can be public, and nothing pushed there may name the baseline UI. So
``public_bundle.py`` writes a mechanically rewritten copy of this toolkit: every case variant of the baseline UI's
word becomes the sentinel ``bline`` (the module, card key, stack name, env names, file names), and the copy carries a
``PUBLIC_BUNDLE`` marker. The sentinel does not occur in this toolkit, so the rewrite is exactly invertible
(``public_bundle.restore_text`` maps returned records back).

Inside the toolkit everything stays self-consistent under the rewrite. Only data that crosses into the real world
needs the real spelling: an imported package, a safetensors key the server reads, a server log line, a requirement
name, a node class name, a download's repo / file name. Those crossings go through two functions:

  ext(s)   sentinel -> real word   (outgoing: imports, keys written, graph JSON submitted, download names, env keys)
  norm(s)  real word -> sentinel   (incoming: server logs, API responses, module names, requirement lines, headers)

In this (private) toolkit the sentinel never appears and nothing is a public copy, so both are the identity. In a
public copy the real word comes from the baseline repository URL at runtime: ``$BASELINE_REPO`` (a repository
variable the workflow passes, e.g. ``${{ vars.BASELINE_REPO }}``), else the private ``private_defaults.json`` that
is never bundled. The word is the repository name without its trailing "UI". Unset in a public copy: a clear error.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
SENTINEL = "bline"
MARKER = HERE / "PUBLIC_BUNDLE"
PRIVATE_DEFAULTS = HERE / "private_defaults.json"
_SENT_RE = re.compile(SENTINEL, re.IGNORECASE)


def is_public() -> bool:
    return MARKER.exists()


def private_defaults() -> dict:
    try:
        return json.loads(PRIVATE_DEFAULTS.read_text())
    except (OSError, ValueError):
        return {}


def baseline_repo(required: bool = True) -> Optional[str]:
    """The baseline UI's git URL: $BASELINE_REPO, else private_defaults.json (never in a public copy)."""
    url = (os.environ.get("BASELINE_REPO") or private_defaults().get("baseline_repo") or "").strip()
    if not url and required:
        raise RuntimeError("BASELINE_REPO is not set: export the baseline UI's git URL (in a workflow, the "
                           "repository variable vars.BASELINE_REPO); this copy carries no default")
    return url or None


def gguf_node_repo(required: bool = True) -> Optional[str]:
    """The GGUF loader node's git URL: $BASELINE_GGUF_REPO, else private_defaults.json, else derived from the
    baseline repository name (<owner city96>/<Name>-GGUF)."""
    url = (os.environ.get("BASELINE_GGUF_REPO") or private_defaults().get("gguf_node_repo") or "").strip()
    if not url:
        base = baseline_repo(required)
        url = f"https://github.com/city96/{base.rstrip('/').split('/')[-1].removesuffix('.git')}-GGUF" if base else ""
    if not url and required:
        raise RuntimeError("BASELINE_GGUF_REPO is not set and no BASELINE_REPO to derive it from")
    return url or None


def convrot_tool(required: bool = True) -> Optional[dict]:
    """The pinned int8 ConvRot quantizer {"repo", "sha", "path"}: $DBENCH_CONVROT_REPO / _SHA / _PATH, else
    private_defaults.json (never in a public copy). The tool is GPL-3 and not vendored: setup fetches it at the pin."""
    pin = dict(private_defaults().get("convrot_tool") or {})
    for k in ("repo", "sha", "path"):
        if os.environ.get(f"DBENCH_CONVROT_{k.upper()}"):
            pin[k] = os.environ[f"DBENCH_CONVROT_{k.upper()}"].strip()
    ok = bool(pin.get("repo")) and len(pin.get("sha") or "") == 40 and bool(pin.get("path"))
    if not ok and required:
        raise RuntimeError("the int8 ConvRot quantizer has no pin: export DBENCH_CONVROT_REPO (its git URL), "
                           "DBENCH_CONVROT_SHA (a 40-hex commit) and DBENCH_CONVROT_PATH (quant_int8_convrot.py), "
                           "or pass a local copy as DBENCH_CONVROT_TOOL")
    return pin if ok else None


def word() -> str:
    """The real word (lower case), from the baseline repository name: <Name>UI -> <name>."""
    name = baseline_repo(True).rstrip("/").split("/")[-1].removesuffix(".git")
    stem = name[:-2] if name.upper().endswith("UI") and len(name) > 2 else name
    if not stem:
        raise RuntimeError(f"cannot derive the baseline word from BASELINE_REPO {name!r}")
    return stem.lower()


def _case(src: str, w: str) -> str:
    if src.isupper():
        return w.upper()
    if src[:1].isupper():
        return w[:1].upper() + w[1:]
    return w


def ext(s):
    """Sentinel -> real word, case preserved (identity when the sentinel is absent)."""
    if not isinstance(s, str) or not _SENT_RE.search(s):
        return s
    w = word()
    return _SENT_RE.sub(lambda m: _case(m.group(0), w), s)


def norm(s):
    """Real word -> sentinel, case preserved; the identity outside a public copy."""
    if not isinstance(s, str) or not is_public():
        return s
    w = word()
    return re.sub(re.escape(w), lambda m: _case(m.group(0), SENTINEL), s, flags = re.IGNORECASE)


def ext_env(env: dict) -> dict:
    """``env`` plus each key holding the sentinel under its real spelling too (a variable the real package reads),
    both kept; values are never touched (they are paths)."""
    out = dict(env)
    for k, v in env.items():
        if isinstance(k, str) and _SENT_RE.search(k):
            out.setdefault(ext(k), v)
    return out


def import_module(name: str):
    import importlib

    return importlib.import_module(ext(name))
