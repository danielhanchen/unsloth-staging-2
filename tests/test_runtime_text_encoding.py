# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Guard: shipping code must name an encoding on every text read and write.

`Path.read_text()`, `Path.write_text()`, `Path.open()` and builtin `open()` fall
back to `locale.getencoding()` when no encoding is given: UTF-8 on the Linux and
macOS runners, cp1252 on a stock Windows install. Every file this repo reads at
runtime is UTF-8 -- HF `config.json` / `tokenizer_config.json` / `adapter_config.json`
(RFC 8259 mandates UTF-8 for JSON), Ollama manifests, GGUF export metadata -- so on
Windows those reads either crash or, worse, succeed with mojibake:

    A DeepSeek or Qwen tokenizer_config.json carries U+FF5C and U+2581 in its
    chat template. Under cp1252 that read raises UnicodeDecodeError, and at
    utils/models/model_config.py the call sits inside a broad `except Exception:
    logger.debug(...)`, so the token-pattern check silently returned the wrong
    answer with no visible error.

Unlike the import-time rule in test_source_read_encoding.py this is scope
agnostic, which makes it both simpler and stricter: runtime reads live inside
functions, and shipping code has no legitimate reason to want the operator's
locale to decide how a file is decoded. That leaves no reachability analysis to
get wrong, so there is no allowlist and no false positives.

Binary handles are skipped: they have no encoding to name, and passing one is a
ValueError. A non-constant mode is treated as unknown rather than assumed text,
for the same reason -- demanding `encoding =` on a call that may resolve to "rb"
would leave no compliant way to write it.
"""

# `str | None` below is evaluated at import on Python 3.9 without this, and
# pyproject declares requires-python = ">=3.9,<3.15".
from __future__ import annotations

import ast
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
# The three trees that ship to users. Test trees are covered by
# test_source_read_encoding.py under a narrower import-time rule.
ROOTS = (REPO / "unsloth", REPO / "studio" / "backend", REPO / "unsloth_cli")
GUARDED_METHODS = {"read_text", "write_text"}
# `<module>.open(...)` is somebody else's opener. fitz.open() takes filetype=,
# tarfile.open() takes a compression mode; neither has an encoding to name.
NOT_PATH_RECEIVERS = {
    "bz2", "codecs", "cv2", "dbm", "fitz", "gzip", "h5py", "Image", "io", "json",
    "lzma", "np", "numpy", "os", "pymupdf", "shelve", "socket", "sqlite3",
    "tarfile", "wave", "webbrowser", "zipfile",
}
# Distinct from None so that "no mode argument at all" still means text.
UNKNOWN_MODE = object()


def _mode(call: ast.Call, positional_index: int):
    """The call's mode, or UNKNOWN_MODE when it is not a literal."""
    if len(call.args) > positional_index:
        node = call.args[positional_index]
        return node.value if isinstance(node, ast.Constant) else UNKNOWN_MODE
    for kw in call.keywords:
        if kw.arg == "mode":
            return kw.value.value if isinstance(kw.value, ast.Constant) else UNKNOWN_MODE
    return "r"


def _names_encoding(call: ast.Call) -> bool:
    """True only for an encoding that actually pins one.

    `encoding = None` and `encoding = "locale"` both re-select the platform
    default, so the keyword being present is not enough.
    """
    for kw in call.keywords:
        if kw.arg != "encoding":
            continue
        if isinstance(kw.value, ast.Constant) and kw.value.value in (None, "locale"):
            return False
        return True
    return False


def _is_text(call: ast.Call, positional_index: int) -> bool:
    mode = _mode(call, positional_index)
    return mode is not UNKNOWN_MODE and "b" not in str(mode)


def _offender(call: ast.Call) -> str | None:
    """The call's name if it does text I/O without pinning an encoding."""
    func = call.func
    if isinstance(func, ast.Attribute):
        if func.attr in GUARDED_METHODS:
            # importlib.metadata's Distribution.read_text takes a positional
            # filename and has no encoding parameter at all, so a positional
            # argument means the receiver is not a Path.
            if func.attr == "read_text" and call.args:
                return None
            return None if _names_encoding(call) else f"{func.attr}()"
        if func.attr == "open":
            receiver = func.value
            if isinstance(receiver, ast.Name) and receiver.id in NOT_PATH_RECEIVERS:
                return None
            if not _is_text(call, 0):
                return None
            return None if _names_encoding(call) else "Path.open()"
        return None
    if isinstance(func, ast.Name) and func.id == "open":
        if not _is_text(call, 1):
            return None
        return None if _names_encoding(call) else "open()"
    return None


def _is_test_path(path: Path) -> bool:
    parts = path.relative_to(REPO).parts
    if "tests" in parts or "test" in parts:
        return True
    return path.name.startswith("test_") or path.name.endswith("_test.py")


def _offenders_in(src: str, label: str = "<snippet>"):
    tree = ast.parse(src, filename = label)
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _offender(node)
            if name is not None:
                found.append((node.lineno, name))
    return found


def test_shipping_code_names_an_encoding():
    offenders = []
    for root in ROOTS:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            if _is_test_path(path):
                continue
            src = path.read_text(encoding = "utf-8")
            try:
                tree = ast.parse(src, filename = str(path))
            except SyntaxError:
                continue
            rel = path.relative_to(REPO).as_posix()
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = _offender(node)
                if name is not None:
                    offenders.append(f"{rel}:{node.lineno}: {name}")
    assert offenders == [], (
        f"{len(offenders)} text read/write call sites in shipping code let the "
        "operator's locale decide the encoding, so they crash or silently "
        'produce mojibake on Windows. Pass encoding = "utf-8": ' + repr(offenders)
    )


# The repo-wide assertion above passes vacuously once the trees are clean, so it
# cannot tell a working detector from one that always returns None. These pin the
# detector itself.


def test_detects_the_plain_cases():
    assert _offenders_in("from pathlib import Path\np = Path('x')\ns = p.read_text()\n")
    assert _offenders_in("p.write_text('hi')\n")
    assert _offenders_in("f = open('x')\n")
    assert _offenders_in("f = open('x', 'w')\n")
    assert _offenders_in("f = p.open()\n")
    # Inside a function body too: shipping reads are not import-time.
    assert _offenders_in("def load(p):\n    return p.read_text()\n")


def test_rejects_encoding_that_reselects_the_platform_default():
    assert _offenders_in("s = p.read_text(encoding = None)\n")
    assert _offenders_in("s = p.read_text(encoding = 'locale')\n")


def test_accepts_a_pinned_encoding():
    assert not _offenders_in("s = p.read_text(encoding = 'utf-8')\n")
    assert not _offenders_in("f = open('x', 'w', encoding = 'utf-8')\n")
    assert not _offenders_in("f = p.open(encoding = 'utf-8')\n")
    assert not _offenders_in("s = p.read_text(encoding = 'utf-8', errors = 'replace')\n")


def test_skips_binary_handles():
    # Binary has no encoding to name; passing one is a ValueError.
    assert not _offenders_in("f = open('x', 'rb')\n")
    assert not _offenders_in("f = open('x', mode = 'wb')\n")
    assert not _offenders_in("f = p.open('rb')\n")


def test_skips_unknown_modes():
    # Demanding encoding = on a call that may resolve to "rb" would leave no
    # compliant way to write it, so an unresolvable mode is not an offence.
    assert not _offenders_in("mode = 'rb' if binary else 'r'\nf = open(path, mode)\n")
    assert not _offenders_in("f = open(path, mode = chosen)\n")


def test_skips_foreign_openers_and_readers():
    assert not _offenders_in("import fitz\nd = fitz.open(stream = b, filetype = 'pdf')\n")
    assert not _offenders_in("import tarfile\nt = tarfile.open(p, 'r:gz')\n")
    # importlib.metadata Distribution.read_text takes a positional filename.
    assert not _offenders_in("s = dist.read_text('direct_url.json')\n")


def test_test_trees_are_out_of_scope():
    assert _is_test_path(REPO / "tests" / "test_x.py")
    assert _is_test_path(REPO / "studio" / "backend" / "tests" / "helpers.py")
    assert not _is_test_path(REPO / "studio" / "backend" / "routes" / "inference.py")
