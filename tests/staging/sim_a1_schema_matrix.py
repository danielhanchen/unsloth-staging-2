#!/usr/bin/env python3
"""A1/A2: validation matrix for the three clip knobs, pre-PR vs post-PR schema.

Self-contained: needs only pydantic, so the same file runs unchanged inside each
uv venv of A2's pydantic-drift sweep and on the Windows/macOS staging runners.

The field definitions below are transcribed from the two revisions of
studio/backend/models/training.py. sim_a1_verify_defs.py checks that transcription
against the live source so this file cannot silently drift.

Why it matters: studio/backend/requirements/studio.txt pins pydantic with no version
specifier at all, so whatever pydantic resolves to on a user's machine decides whether
allow_inf_nan=False actually binds on an Optional[float]. The pydantic docs warn that
constraints on a union may need the Annotated form.
"""

import json
import math
import sys
from typing import Optional

import pydantic
from pydantic import BaseModel, Field, ValidationError

PYD = pydantic.__version__
FAILURES = []


class OldModel(BaseModel):
    """Pre-PR: max_grad_norm required-with-default, no finite constraint anywhere."""

    max_grad_norm: float = Field(0.0, ge=0)
    max_grad_value: Optional[float] = Field(None, ge=0)
    max_grad_leaf_norm: Optional[float] = Field(None, ge=0)


class NewModel(BaseModel):
    """Post-PR (incl. my sibling commit): all three optional, all three finite."""

    max_grad_norm: Optional[float] = Field(None, ge=0, allow_inf_nan=False)
    max_grad_value: Optional[float] = Field(None, ge=0, allow_inf_nan=False)
    max_grad_leaf_norm: Optional[float] = Field(None, ge=0, allow_inf_nan=False)


ACCEPT = "accept"
REJECT = "reject"

# (label, python value, expect_old, expect_new)
# `_ABSENT` means the key is omitted entirely, which is what the new frontend sends.
_ABSENT = object()
VALUES = [
    ("absent",            _ABSENT,          ACCEPT, ACCEPT),
    ("explicit null",     None,             REJECT, ACCEPT),  # old: not Optional
    ("0",                 0,                ACCEPT, ACCEPT),
    ("0.0",               0.0,              ACCEPT, ACCEPT),
    ("0.5",               0.5,              ACCEPT, ACCEPT),
    ("numeric string",    "1.5",            ACCEPT, ACCEPT),
    ("negative",          -1,               REJECT, REJECT),
    # ge=0 already catches nan and -inf: `nan >= 0` and `-inf >= 0` are both False.
    # The one value that slipped through pre-PR is +inf, since `inf >= 0` is True.
    # So the hole allow_inf_nan closes is +inf (and 1e309, which floats to +inf),
    # NOT nan. Narrower than it first looks, but still a live hole.
    ("inf",               float("inf"),     ACCEPT, REJECT),
    ("-inf",              float("-inf"),    REJECT, REJECT),
    ("nan",               float("nan"),     REJECT, REJECT),
    ("1e309 -> inf",      1e309,            ACCEPT, REJECT),
    ("bool True",         True,             ACCEPT, ACCEPT),
]

# The sibling knobs were already Optional pre-PR, so "explicit null" is accepted by both.
SIBLING_OVERRIDE = {"explicit null": (ACCEPT, ACCEPT)}


def outcome(model, field, value):
    payload = {} if value is _ABSENT else {field: value}
    try:
        model(**payload)
        return ACCEPT
    except ValidationError:
        return REJECT


def run_python_matrix():
    print(f"\n== A1: python-level validation matrix (pydantic {PYD}) ==")
    header = f"  {'field':<20} {'input':<16} {'old':<8} {'new':<8} verdict"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for field in ("max_grad_norm", "max_grad_value", "max_grad_leaf_norm"):
        for label, value, exp_old, exp_new in VALUES:
            if field != "max_grad_norm" and label in SIBLING_OVERRIDE:
                exp_old, exp_new = SIBLING_OVERRIDE[label]
            got_old = outcome(OldModel, field, value)
            got_new = outcome(NewModel, field, value)
            ok = (got_old == exp_old) and (got_new == exp_new)
            if not ok:
                FAILURES.append(f"{field}/{label}: old {got_old}!={exp_old} new {got_new}!={exp_new}")
            print(f"  {field:<20} {label:<16} {got_old:<8} {got_new:<8} {'PASS' if ok else 'FAIL'}")


def run_json_matrix():
    """The wire path. jiter parses bare Infinity/NaN by default, so a JSON body can
    smuggle a non-finite float past a ge=0-only constraint."""
    print(f"\n== A1b: JSON-wire matrix (the literals a real HTTP body can carry) ==")
    literals = [
        ('{"max_grad_norm": Infinity}',  ACCEPT, REJECT),
        ('{"max_grad_norm": NaN}',       REJECT, REJECT),  # ge=0 already blocks nan
        ('{"max_grad_norm": 1e309}',     ACCEPT, REJECT),
        ('{"max_grad_norm": null}',      REJECT, ACCEPT),
        ('{"max_grad_norm": 0}',         ACCEPT, ACCEPT),
        ('{}',                           ACCEPT, ACCEPT),
        ('{"max_grad_value": Infinity}', ACCEPT, REJECT),
        ('{"max_grad_leaf_norm": NaN}',  REJECT, REJECT),
    ]
    for raw, exp_old, exp_new in literals:
        def via_json(model):
            try:
                model.model_validate_json(raw)
                return ACCEPT
            except ValidationError:
                return REJECT
        got_old, got_new = via_json(OldModel), via_json(NewModel)
        ok = (got_old == exp_old) and (got_new == exp_new)
        if not ok:
            FAILURES.append(f"json {raw}: old {got_old}!={exp_old} new {got_new}!={exp_new}")
        print(f"  {raw:<34} old={got_old:<7} new={got_new:<7} {'PASS' if ok else 'FAIL'}")


def run_none_survives():
    """The single most important assertion: allow_inf_nan=False on an Optional[float]
    must not accidentally reject None. If it did, every new-frontend request would 422."""
    print("\n== A1c: None still validates under allow_inf_nan=False (Optional union) ==")
    m = NewModel()
    ok = m.max_grad_norm is None and m.max_grad_value is None and m.max_grad_leaf_norm is None
    print(f"  [{'PASS' if ok else 'FAIL'}] defaults are None: {m.model_dump()}")
    if not ok:
        FAILURES.append("None default did not survive")
    m2 = NewModel(max_grad_norm=None)
    ok2 = m2.max_grad_norm is None
    print(f"  [{'PASS' if ok2 else 'FAIL'}] explicit null accepted: {m2.max_grad_norm!r}")
    if not ok2:
        FAILURES.append("explicit None rejected")
    # And the serialized form must omit / null it rather than emit 0.0.
    dumped = m.model_dump(exclude_unset=True)
    ok3 = "max_grad_norm" not in dumped
    print(f"  [{'PASS' if ok3 else 'FAIL'}] exclude_unset omits the key: {dumped}")
    if not ok3:
        FAILURES.append("exclude_unset did not omit max_grad_norm")


def run_serialization_hazard():
    """Starlette encodes responses with json.dumps(allow_nan=False). Anything that
    stores a non-finite float therefore 500s on read-back. Show that the old schema
    lets one through and the new one does not."""
    print("\n== A1d: stored-inf serialization hazard ==")
    old = OldModel(max_grad_norm=float("inf"))
    try:
        json.dumps({"max_grad_norm": old.max_grad_norm}, allow_nan=False)
        got = "serialized"
    except ValueError as e:
        got = f"ValueError: {e}"
    ok = got.startswith("ValueError")
    print(f"  [{'PASS' if ok else 'FAIL'}] old schema stores inf, then read-back raises: {got}")
    if not ok:
        FAILURES.append("expected the old schema to produce an unserializable value")
    try:
        NewModel(max_grad_norm=float("inf"))
        blocked = False
    except ValidationError:
        blocked = True
    print(f"  [{'PASS' if blocked else 'FAIL'}] new schema rejects it at the boundary: {blocked}")
    if not blocked:
        FAILURES.append("new schema failed to reject inf")


run_python_matrix()
run_json_matrix()
run_none_survives()
run_serialization_hazard()

print("\n" + "=" * 70)
if FAILURES:
    print(f"A1 FAILED on pydantic {PYD} ({len(FAILURES)}):")
    for f in FAILURES:
        print("   -", f)
    sys.exit(1)
print(f"A1 PASSED on pydantic {PYD} (all knobs, all inputs, both schemas)")
sys.exit(0)
