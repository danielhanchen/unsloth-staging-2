"""Record schema v2: one record.json per cell, the same shape for every backend, stack and target.

v2 is a superset of the v1 record (``renders`` / ``long`` / ``short`` / ``wall_s_median`` / ``peak_smi_gib`` ...),
which run_cell.py still writes next to the v2 blocks so score.py and older readers keep working. ``upgrade()`` maps
a v1-only record (older rounds) into v2, marked ``legacy``.

  SCHEMA_VERSION = 2
  new_record(cell) -> dict           every v2 top-level key, cell / target / arm copied from the cell options
  validate(record) -> list[str]      problems; [] = valid. An "ok" record must carry its evidence (requests, a VRAM
                                     figure, engagement for requested levers), else it is not ok.
  dumps(record) / loads(text)        strictly finite JSON: NaN / Infinity raise instead of being written or read
  sanitize(record) -> (record, paths) non-finite floats replaced by None, their paths returned (the caller VOIDs)
  upgrade(record) -> dict            v1 -> v2 (idempotent on v2)

Top-level keys (see outputs/dbench_unify/CONTRACT.md):
  schema_version, tag, verdict (ok|BROKEN|VOID|NOT_RUN), reason, cell, target, arm, revisions, gpu, requests[],
  timing, memory, telemetry, parity, engagement, media[]
Request: {kind: cold|warmup|new|same, id, seed, prompt_id, total_s, steps, step_s[], first_step_s, phases{te_s,
  denoise_s, vae_s, source}, bline{executed_nodes, cached_nodes, sampler_cached}, peak_vram_over_base_gib, clocks}
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import uuid
from typing import Any, Iterable, Optional

SCHEMA_VERSION = 2
VERDICTS = ("ok", "BROKEN", "VOID", "NOT_RUN")
REQUEST_KINDS = ("cold", "warmup", "new", "same")
TOP_KEYS = ("schema_version", "tag", "verdict", "reason", "cell", "target", "arm", "revisions", "gpu", "requests",
            "timing", "memory", "telemetry", "parity", "engagement", "media")
REVISION_KEYS = ("studio", "zoo", "blineui_tag", "blineui_sha", "custom_nodes", "torch", "cuda", "triton",
                 "scripts_sha")
GPU_KEYS = ("name", "uuid", "driver", "pcie_gen", "pcie_width", "cc")
TIMING_KEYS = ("load_s", "cold_s", "first_render_s", "compile_s", "compile_s_source", "first_step_s",
               "steady_s_median", "new_s_median", "step_s_polled_median", "step_s_derived", "step_s_derived_flag")
MEMORY_KEYS = ("peak_vram_own_gib", "peak_vram_device_gib", "peak_vram_over_base_gib", "peak_rss_tree_gib",
               "peak_rss_server_gib", "budget", "vram_source", "vram_unavailable_reason")
TELEMETRY_KEYS = ("sm_mhz_mean", "temp_c_max", "power_w_mean", "power_limit_w", "samples", "source")


def _empty(keys: Iterable[str]) -> dict:
    return {k: None for k in keys}


def new_record(cell: Optional[dict] = None) -> dict:
    """A v2 record skeleton for ``cell``; verdict "running" until run_cell decides."""
    cell = dict(cell or {})
    opts = cell.get("options") or {}
    target = dict(opts.get("target") or {})
    target.setdefault("name", None)
    target.setdefault("vm_id", None)
    arm = dict(opts.get("arm") or {})
    arm.setdefault("stack", stack_of(cell.get("backend")))
    return {
        "schema_version": SCHEMA_VERSION,
        # one id per cell execution, written once: copies of the same record (artifact.tgz and the base64 block in a
        # notebook / log) share it, real repeats never do (run_identity / dedup)
        "run_id": uuid.uuid4().hex,
        "tag": cell.get("tag"),
        "verdict": "running",
        "reason": None,
        "cell": cell,
        "target": target,
        "arm": arm,
        "revisions": _empty(REVISION_KEYS),
        "gpu": _empty(GPU_KEYS),
        "requests": [],
        "timing": _empty(TIMING_KEYS),
        "memory": {**_empty(MEMORY_KEYS), "budget": {"gib": None, "simulated": False}},
        "telemetry": _empty(TELEMETRY_KEYS),
        "parity": {},
        "engagement": {"levers": {}, "status_after": None, "bline_log_lines": []},
        "media": [],
    }


def stack_of(backend: Optional[str]) -> Optional[str]:
    """The stack a backend measures: studio (in process or HTTP), blineui, sdcpp, diffusers, fake."""
    if backend in ("studio", "studio_http", "studio_mlx"):
        return "studio"
    return backend


# ---------------------------------------------------------------------------------------------- strict JSON
def _reject_constant(name: str):
    raise ValueError(f"non-finite JSON constant {name!r}")


def loads(text: str) -> Any:
    """json.loads that refuses NaN / Infinity / -Infinity (Python's json accepts them by default)."""
    return json.loads(text, parse_constant = _reject_constant)


def dumps(obj: Any, **kw) -> str:
    """json.dumps with allow_nan=False: a non-finite float raises ValueError instead of writing invalid JSON."""
    kw.setdefault("indent", 1)
    return json.dumps(obj, allow_nan = False, default = str, **kw)


def nonfinite_paths(obj: Any, path: str = "$") -> list:
    out = []
    if isinstance(obj, float) and not math.isfinite(obj):
        out.append(path)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out += nonfinite_paths(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out += nonfinite_paths(v, f"{path}[{i}]")
    return out


def sanitize(obj: Any) -> tuple:
    """(copy with every non-finite float replaced by None, the JSON paths that were replaced)."""
    paths = nonfinite_paths(obj)
    if not paths:
        return obj, []

    def fix(x):
        if isinstance(x, float) and not math.isfinite(x):
            return None
        if isinstance(x, dict):
            return {k: fix(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [fix(v) for v in x]
        return x

    return fix(obj), paths


# ---------------------------------------------------------------------------------------------- validation
def _num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def vram_metric(rec: dict) -> Optional[float]:
    mem = rec.get("memory") or {}
    for key in ("peak_vram_own_gib", "peak_vram_over_base_gib", "peak_vram_device_gib"):
        if _num(mem.get(key)):
            return float(mem[key])
    return None


def cpu_run(rec: dict) -> bool:
    """A record that ran on the CPU (the cpu target, or a stack that states a CPU device): no VRAM metric exists."""
    if ((rec.get("target") or {}).get("name") or "") == "cpu":
        return True
    st = rec.get("status") if isinstance(rec.get("status"), dict) else {}
    dev = str(st.get("device") or (st.get("facts") or {}).get("device") or
              ((rec.get("cell") or {}).get("options") or {}).get("device") or "").lower()
    return dev.startswith("cpu")


def metal_target(rec: dict) -> bool:
    """A record from the Apple Silicon target (mac:*): unified memory, no NVML / SMI device reading."""
    return str((rec.get("target") or {}).get("name") or "").startswith("mac:")


def vram_unavailable(rec: dict) -> bool:
    """The VRAM metric is explicitly UNAVAILABLE: only on the Metal target, for a stack that reports no Metal device
    memory, with the reason recorded (report.py lists it under could-not-test). Never on an NVIDIA / AMD target."""
    mem = rec.get("memory") or {}
    return metal_target(rec) and mem.get("vram_source") == "unavailable" and bool(mem.get("vram_unavailable_reason"))


def engagement_problems(rec: dict) -> list:
    """A lever the cell asked for must show evidence of engaging; no evidence is a silent fallback, not a pass."""
    out = []
    for name, lev in ((rec.get("engagement") or {}).get("levers") or {}).items():
        if not isinstance(lev, dict) or not lev.get("requested"):
            continue
        if lev.get("engaged") is None:
            out.append(f"lever {name}: requested, no engagement evidence")
        elif lev.get("engaged") is False:
            out.append(f"lever {name}: requested but not engaged ({lev.get('evidence')})")
    return out


def validate(rec: Any) -> list:
    """Problems with ``rec`` as a v2 record; [] = valid. Never raises."""
    if not isinstance(rec, dict):
        return ["record is not an object"]
    p = []
    if rec.get("schema_version") != SCHEMA_VERSION:
        p.append(f"schema_version {rec.get('schema_version')!r} != {SCHEMA_VERSION}")
    missing = [k for k in TOP_KEYS if k not in rec]
    if missing:
        p.append(f"missing keys {missing}")
    verdict = rec.get("verdict")
    if verdict not in VERDICTS:
        p.append(f"verdict {verdict!r} not in {VERDICTS}")
    if verdict in ("BROKEN", "VOID", "NOT_RUN") and not (rec.get("reason") or rec.get("void_reason")):
        p.append(f"verdict {verdict} without a reason")
    bad = nonfinite_paths(rec)
    if bad:
        p.append(f"non-finite values at {bad[:5]}")
    for i, r in enumerate(rec.get("requests") or []):
        if not isinstance(r, dict):
            p.append(f"requests[{i}] is not an object")
            continue
        if r.get("kind") not in REQUEST_KINDS:
            p.append(f"requests[{i}].kind {r.get('kind')!r} not in {REQUEST_KINDS}")
        if r.get("total_s") is not None and not _num(r.get("total_s")):
            p.append(f"requests[{i}].total_s not a finite number")
        steps = r.get("step_s")
        if steps is not None and (not isinstance(steps, list) or not all(_num(s) for s in steps)):
            p.append(f"requests[{i}].step_s is not a list of finite numbers")
    if verdict == "ok":
        new = [r for r in rec.get("requests") or [] if isinstance(r, dict) and r.get("kind") == "new"]
        if not new:
            p.append("ok record with no new-prompt request")
        # a CPU run has host RSS only (VRAM stays null); a Metal stack with no device-memory source is UNAVAILABLE
        if vram_metric(rec) is None and not cpu_run(rec) and not vram_unavailable(rec):
            p.append("ok record without a VRAM metric (memory.peak_vram_own_gib / over_base / device all missing)")
        if not _num((rec.get("timing") or {}).get("new_s_median")):
            p.append("ok record without timing.new_s_median")
        p += engagement_problems(rec)
        for i, r in enumerate(rec.get("requests") or []):
            c = r.get("bline") if isinstance(r, dict) else None
            if c and r.get("kind") in ("new", "same"):
                if not c.get("executed_nodes"):
                    p.append(f"requests[{i}] ({r.get('kind')}): BlineUI executed 0 nodes (served from cache)")
                elif c.get("sampler_cached"):
                    p.append(f"requests[{i}] ({r.get('kind')}): BlineUI sampler served from cache")
    return p


# ---------------------------------------------------------------------------------------------- v1 -> v2
def upgrade(rec: dict) -> dict:
    """A v1 record (older rounds) as v2, marked legacy. v1 fields are kept; v2 blocks are filled from them."""
    if not isinstance(rec, dict) or rec.get("schema_version") == SCHEMA_VERSION:
        return rec
    out = new_record(rec.get("cell") or {"tag": rec.get("tag"), "backend": rec.get("backend")})
    out.update({k: v for k, v in rec.items() if k not in out or k in ("tag",)})
    out["schema_version"] = SCHEMA_VERSION
    out["run_id"] = rec.get("run_id") or run_identity(rec)[4:]  # stable across loads: the v1 content digest
    out["legacy"] = {"from_schema": rec.get("schema_version") or 1}
    v = rec.get("verdict")
    out["verdict"] = v if v in VERDICTS else "BROKEN"
    out["reason"] = rec.get("reason") or rec.get("error") or (None if out["verdict"] == "ok" else f"v1 verdict {v!r}")
    reqs = []
    if rec.get("cold_s") is not None:
        reqs.append({"kind": "cold", "total_s": rec.get("cold_s")})
    cell = rec.get("cell") or {}
    for r in rec.get("renders") or []:
        reqs.append({"kind": "new", "id": r.get("id"), "seed": r.get("seed"), "total_s": r.get("wall_s"),
                     "steps": cell.get("steps"), "step_s": None, "step_s_reported": r.get("step_s")})
    for s in rec.get("long") or []:
        reqs.append({"kind": "same", "total_s": s, "steps": cell.get("steps")})
    for s in rec.get("short") or []:
        reqs.append({"kind": "same", "total_s": s, "steps": cell.get("short_steps")})
    out["requests"] = reqs
    out["timing"].update({"load_s": rec.get("load_s"), "first_render_s": rec.get("cold_s"),
                          "new_s_median": rec.get("wall_s_median"), "steady_s_median": rec.get("steady_s_median"),
                          "step_s_derived": rec.get("step_s_derived")})
    if rec.get("load_s") is not None and rec.get("cold_s") is not None:
        out["timing"]["cold_s"] = round(rec["load_s"] + rec["cold_s"], 3)
    out["memory"].update({"peak_vram_own_gib": rec.get("peak_own_gib") or None,
                          "peak_vram_device_gib": rec.get("peak_smi_gib") or None,
                          "peak_vram_over_base_gib": rec.get("peak_smi_delta_gib") or None})
    env = rec.get("env") or {}
    gpu = env.get("gpu") or {}
    out["gpu"].update({"name": env.get("device_name") or gpu.get("name"),
                       "cc": ".".join(map(str, env["device_capability"])) if env.get("device_capability") else None})
    out["revisions"].update({"torch": (env.get("packages") or {}).get("torch"), "cuda": env.get("torch_cuda")})
    trees = env.get("trees") or {}
    out["revisions"]["studio"] = trees.get("studio")
    out["revisions"]["blineui_sha"] = trees.get("blineui")
    return out


def run_identity(rec: dict) -> str:
    """The execution a record describes: its run_id, else (older records) a digest of its content without the
    in-memory "_" keys, so two byte-identical copies of one run are one run."""
    if isinstance(rec, dict) and rec.get("run_id"):
        return f"run:{rec['run_id']}"
    body = {k: v for k, v in (rec or {}).items() if not str(k).startswith("_")} if isinstance(rec, dict) else rec
    return "sha:" + hashlib.sha256(json.dumps(body, sort_keys = True, default = str).encode()).hexdigest()


def dedup(items: Iterable, rec_of=lambda x: x) -> tuple:
    """(items with one entry per run_identity, first kept, number of duplicate copies dropped)."""
    items = list(items)
    seen, out = set(), []
    for it in items:
        key = run_identity(rec_of(it))
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out, len(items) - len(out)


def load(path) -> dict:
    """A record file as v2 (v1 upgraded). Raises on unreadable or non-finite JSON."""
    from pathlib import Path

    return upgrade(loads(Path(path).read_text(encoding = "utf-8")))


def deep_copy(rec: dict) -> dict:
    return copy.deepcopy(rec)
