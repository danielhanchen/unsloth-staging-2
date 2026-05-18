# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Cross-OS regex unit test for the mid-plan EOS detectors used by the
auto-continue path in studio/backend/core/inference/llama_cpp.py.

Pulls the regex source straight out of llama_cpp.py via ast so the test
matches the live constants without depending on the heavy import chain
(httpx, structlog, the studio utils package). Runs on Ubuntu, macOS and
Windows with nothing but the stdlib and pytest.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LLAMA_PY = REPO_ROOT / "studio" / "backend" / "core" / "inference" / "llama_cpp.py"


def _extract_compiled(name: str) -> re.Pattern:
    """Find a top-level `name = re.compile(<args>)` and rebuild it."""
    tree = ast.parse(LLAMA_PY.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        if node.targets[0].id != name:
            continue
        if not isinstance(node.value, ast.Call):
            continue
        src = ast.unparse(node.value)
        return eval(src, {"re": re})
    raise LookupError(name)


INTENT_SIGNAL = _extract_compiled("_INTENT_SIGNAL")
PLAN_INTENT = _extract_compiled("_TRAILING_PLAN_INTENT")
PLAN_LIST = _extract_compiled("_TRAILING_PLAN_LIST")
PLAN_COLON = _extract_compiled("_TRAILING_PLAN_COLON")
WINDOW = 600


def _trailing_plan_hit(stripped: str) -> bool:
    if not stripped:
        return False
    tail = stripped[-WINDOW:]
    return (
        PLAN_INTENT.search(tail) is not None
        or PLAN_LIST.search(tail) is not None
        or PLAN_COLON.search(tail) is not None
    )


# ── shapes that must hit the trailing-plan detector ──
HIT_CASES = [
    "Let me clone the repo.",
    "Now let me check the file.",
    "I'll now run the script.",
    "I'm going to inspect this.",
    "I will now proceed.",
    "Let me clone the repo:",
    "Now I'll check the repo:",
    "I will inspect the file:",
    "I'll review the PR as follows:\n- read the diff\n- check tests\n- run pytest\n",
    "Here's my plan:\n1. clone\n2. build\n3. test\n",
    "Let me run the following steps:\n* clone\n* build\n* test\n",
]

# ── shapes that must NOT hit the trailing-plan detector ──
NO_HIT_CASES = [
    "",
    "The answer is 42.",
    "I can do that. Done.",
    "Let's see what the answer is. It is 42.",
    "I have finished. Result attached.",
]

# ── shapes that must hit the broader intent-signal regex ──
INTENT_HIT_CASES = [
    "I'll clone the repo and check it.",
    "I will run the script for you.",
    "Let me check the README.",
    "I am going to look at this.",
    "Step 1: clone the repo.",
    "Here's my plan: clone, build, test.",
    "Now I want to check the repo.",
]


def test_trailing_plan_intent_endings():
    assert PLAN_INTENT.search("Let me clone the repo.") is not None
    assert PLAN_INTENT.search("I'll now run the script.") is not None
    assert PLAN_INTENT.search("Done.") is None


def test_trailing_plan_list_endings():
    body = "I'll review:\n- A\n- B\n- C\n"
    assert PLAN_LIST.search(body) is not None
    body_num = "Here's my plan:\n1. clone\n2. build\n3. test\n"
    assert PLAN_LIST.search(body_num) is not None
    body_bullet = "Here's the steps:\n* clone\n* build\n* test\n"
    assert PLAN_LIST.search(body_bullet) is not None


def test_trailing_plan_colon_endings():
    assert PLAN_COLON.search("Now I'll check the repo:") is not None
    assert PLAN_COLON.search("Let me inspect the file:") is not None


def test_trailing_plan_hit_positive():
    for s in HIT_CASES:
        assert _trailing_plan_hit(s), f"expected hit on: {s!r}"


def test_trailing_plan_hit_negative():
    for s in NO_HIT_CASES:
        assert not _trailing_plan_hit(s), f"expected miss on: {s!r}"


def test_intent_signal_positive():
    for s in INTENT_HIT_CASES:
        assert INTENT_SIGNAL.search(s) is not None, f"expected hit on: {s!r}"


def test_curly_apostrophe_handled():
    # The model often emits U+2019 (right single quote) instead of ASCII '.
    assert PLAN_INTENT.search("Let’s now look at the file.") is not None
    assert PLAN_COLON.search("Now I’ll check the repo:") is not None


def test_window_bounds_long_prefix():
    # Garbage filler longer than WINDOW followed by a fresh plan ending.
    filler = "x " * (WINDOW * 4)
    assert _trailing_plan_hit(filler + "Let me clone the repo.")
    # Plan cue further back than WINDOW: NOT a hit (window is the cap).
    far_prefix = "Let me clone the repo." + ("y " * (WINDOW * 4))
    assert not _trailing_plan_hit(far_prefix)
