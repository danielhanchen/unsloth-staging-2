#!/usr/bin/env python3
"""A4 + A7: end-to-end config pipeline and mixed-version compatibility.

Runs against the REAL Studio backend modules (no stubs), so it exercises the actual
TrainingStartRequest, _build_training_worker_config and _resolve_mlx_max_grad_norm.

A4 - every caller shape (REST / CLI / raw kwargs) produces the key, with the right value.
A7 - the four legacy shapes that decide whether existing installs break:
     (a) an old cached frontend bundle POSTing max_grad_norm: 0.0 to a NEW backend
     (b) a NEW frontend omitting the key against the OLD schema
     (c) an old DB row containing 0.0, through sanitize -> json -> reload
     (d) the OpenAPI schema delta an old generated client would see
"""

import json
import math
import os
import sys
from pathlib import Path

REPO = Path(os.environ.get("UNSLOTH_REPO") or Path(__file__).resolve().parents[2])
BACKEND = REPO / "studio" / "backend"
sys.path.insert(0, str(BACKEND))
os.environ.setdefault("UNSLOTH_ALLOW_CPU", "1")
os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")

from core.training.training import (  # noqa: E402
    _build_training_worker_config,
    _coerce_optional_nonneg_float,
    _sanitize_db_config,
)
from core.training.worker import _resolve_mlx_max_grad_norm  # noqa: E402
from models.training import TrainingStartRequest  # noqa: E402

FAILURES = []


def check(label, got, want):
    ok = got == want or (isinstance(want, float) and isinstance(got, float)
                         and math.isclose(got, want))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: got {got!r}, want {want!r}")
    if not ok:
        FAILURES.append(label)


def request_kwargs(**over):
    base = dict(model_name="unsloth/test", training_type="LoRA/QLoRA", format_type="auto")
    base.update(over)
    return base


def full_pipeline(values):
    """values -> worker config -> the threshold the MLX trainer would receive."""
    base = {"model_name": "unsloth/test", "training_type": "LoRA/QLoRA"}
    base.update(values)
    cfg = _build_training_worker_config(base)
    return cfg, _resolve_mlx_max_grad_norm(cfg.get("max_grad_norm"))


# ---------------------------------------------------------------- A4
print("\n== A4: caller shapes through the real pipeline ==")

print("  -- REST caller (validated through TrainingStartRequest) --")
for label, over, want_req, want_resolved in [
    ("new frontend omits the key", {}, None, 0.0),
    ("explicit null", {"max_grad_norm": None}, None, 0.0),
    ("explicit 0.0 (old frontend)", {"max_grad_norm": 0.0}, 0.0, 0.0),
    ("explicit 0.3", {"max_grad_norm": 0.3}, 0.3, 0.3),
    ("numeric string", {"max_grad_norm": "2"}, 2.0, 2.0),
]:
    req = TrainingStartRequest(**request_kwargs(**over))
    check(f"{label}: request field", req.max_grad_norm, want_req)
    cfg, resolved = full_pipeline({"max_grad_norm": req.max_grad_norm})
    check(f"{label}: key present in worker config", "max_grad_norm" in cfg, True)
    check(f"{label}: threshold reaching MLX trainer", resolved, want_resolved)

print("  -- CLI caller (unsloth_cli exposes no clip knob at all) --")
cfg, resolved = full_pipeline({})
check("CLI: key still present", "max_grad_norm" in cfg, True)
check("CLI: value is None", cfg["max_grad_norm"], None)
check("CLI: MLX threshold (unchanged default)", resolved, 0.0)

print("  -- raw **kwargs caller (bypasses the HTTP schema entirely) --")
for bad in (-1, "nope", float("inf"), float("nan")):
    try:
        _build_training_worker_config({"model_name": "u/t", "max_grad_norm": bad})
        got = "accepted"
    except ValueError:
        got = "ValueError"
    check(f"raw caller passing {bad!r}", got, "ValueError")
check("raw caller passing 0.5", _build_training_worker_config({"model_name": "u/t", "max_grad_norm": 0.5})["max_grad_norm"], 0.5)

print("  -- sibling knobs keep None (the trainer picks their default) --")
cfg = _build_training_worker_config({"model_name": "u/t"})
check("max_grad_value stays None", cfg["max_grad_value"], None)
check("max_grad_leaf_norm stays None", cfg["max_grad_leaf_norm"], None)


# ---------------------------------------------------------------- A7
print("\n== A7a: OLD cached frontend bundle -> NEW backend ==")
# A user upgrades the backend but their browser still serves the previous SPA, which
# hardcoded max_grad_norm: 0.0. Must not error; must reproduce the OLD behavior.
old_payload = {"max_grad_norm": 0.0, "max_grad_value": None}
req = TrainingStartRequest(**request_kwargs(**old_payload))
cfg, resolved = full_pipeline({"max_grad_norm": req.max_grad_norm,
                               "max_grad_value": req.max_grad_value})
check("old bundle validates against the new schema", req.max_grad_norm, 0.0)
check("old bundle behaves exactly as before", resolved, 0.0)

print("\n== A7b: NEW frontend (key omitted) -> OLD backend schema ==")
# The reverse skew: a user updates the desktop app / SPA but runs an older backend.
# Reconstruct the old field definition and confirm the omission is benign.
from typing import Optional  # noqa: E402

from pydantic import BaseModel, Field  # noqa: E402


class OldRequest(BaseModel):
    max_grad_norm: float = Field(0.0, ge=0)
    max_grad_value: Optional[float] = Field(None, ge=0)


old = OldRequest()  # new frontend sends nothing for either knob
check("old backend falls back to its 0.0 default", old.max_grad_norm, 0.0)
check("=> old backend keeps pre-PR behavior, no crash", True, True)

print("\n== A7c: OLD DB row containing 0.0, through sanitize -> json -> reload ==")
legacy_config = {"model_name": "unsloth/test", "max_grad_norm": 0.0,
                 "hf_token": "secret", "max_grad_value": None}
sanitized = _sanitize_db_config(legacy_config)
blob = json.dumps(sanitized)
reloaded = json.loads(blob)
check("secrets stripped", "hf_token" in sanitized, False)
check("legacy 0.0 survives the round-trip", reloaded["max_grad_norm"], 0.0)
check("a NEW run's None also round-trips as JSON null",
      json.loads(json.dumps(_sanitize_db_config({"max_grad_norm": None})))["max_grad_norm"], None)
# And the crucial part: a None in the blob must be JSON-serializable, unlike inf.
try:
    json.dumps({"max_grad_norm": None}, allow_nan=False)
    ok = True
except ValueError:
    ok = False
check("None is safe for Starlette's allow_nan=False encoder", ok, True)

print("\n== A7d: OpenAPI schema delta seen by an old generated client ==")
new_schema = TrainingStartRequest.model_json_schema()["properties"]["max_grad_norm"]
old_schema = OldRequest.model_json_schema()["properties"]["max_grad_norm"]
print(f"  old: {json.dumps(old_schema, sort_keys=True)[:160]}")
print(f"  new: {json.dumps(new_schema, sort_keys=True)[:200]}")
check("max_grad_norm was never in `required` (had a default)",
      "max_grad_norm" in TrainingStartRequest.model_json_schema().get("required", []), False)
# The only client-visible change is that null became legal and the default moved to null.
check("new schema still accepts the number an old client sends",
      TrainingStartRequest(**request_kwargs(max_grad_norm=0.0)).max_grad_norm, 0.0)


print("\n" + "=" * 70)
if FAILURES:
    print(f"A4/A7 FAILED ({len(FAILURES)}):")
    for f in FAILURES:
        print("   -", f)
    sys.exit(1)
print("A4/A7 PASSED: every caller shape and every legacy shape behaves as designed.")
sys.exit(0)
