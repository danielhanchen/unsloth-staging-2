#!/usr/bin/env python3
"""CPU-fallback detection for Mac (MPS / Metal) cells: a cell meant for the GPU that ran on the CPU measured the
wrong thing, so its record becomes VOID with the evidence lines. Run after matrix.py on the runner, or locally on a
downloaded artifact (collect() calls it).

  python diffusion_bench/targets/mac_check.py RUN_DIR [--write]

Evidence per stack (from the cell's log, the BlineUI server log and the record):
  studio   torch device lines ("device=cpu", "on cpu", "MPS not available", "falling back to CPU"), and any
           "not currently implemented for the MPS device" (PYTORCH_ENABLE_MPS_FALLBACK=0 turns that into an error,
           so it shows as BROKEN; with the fallback on it would be a silent CPU op);
  blineui  the server's "Device: cpu" / "Set vram state to: DISABLED" vs "Device: mps";
  sdcpp    "Using CPU backend" / no ggml_metal_init vs "ggml_metal_init" / "Metal" backend lines.
A cell that asked for the CPU (options.bline_args has --cpu, options.device / env says cpu) is left alone.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

CPU_RE = {
    "studio": re.compile(r"device[=: ]+['\"]?cpu\b|running on cpu|MPS (?:backend )?(?:is )?not available|"
                         r"falling back to (?:the )?CPU|not currently implemented for the MPS device", re.I),
    "blineui": re.compile(r"^Device: cpu|Set vram state to: DISABLED|Using CPU", re.I | re.M),
    "sdcpp": re.compile(r"Using CPU backend|backend[: ]+cpu\b", re.I),
}
GPU_RE = {
    "studio": re.compile(r"device[=: ]+['\"]?mps|\bmps:0\b|on mps\b|mlx", re.I),
    "blineui": re.compile(r"^Device: mps", re.I | re.M),
    "sdcpp": re.compile(r"ggml_metal_init|Using Metal backend|MTL", re.I),
}


def wants_cpu(cell: dict) -> bool:
    opts = cell.get("options") or {}
    env = cell.get("env") or {}
    return ("--cpu" in str(opts.get("bline_args") or "") or str(opts.get("device") or "").lower() == "cpu"
            or str(env.get("DBENCH_DEVICE") or "").lower() == "cpu" or "--backend cpu" in str(opts.get("extra_args") or ""))


def check_cell(cell_dir: Path) -> dict | None:
    rp = cell_dir / "record.json"
    if not rp.exists():
        return None
    rec = json.loads(rp.read_text())
    cell = rec.get("cell") or {}
    stack = cell.get("backend") or (rec.get("arm") or {}).get("stack")
    if stack not in CPU_RE or wants_cpu(cell):
        return {"tag": cell_dir.name, "stack": stack, "status": "skipped"}
    text = ""
    for p in [cell_dir / "cell.log", cell_dir / "_bline" / "server.log"] + list(cell_dir.glob("*.log")):
        if p.exists():
            text += p.read_text(errors = "replace") + "\n"
    text += json.dumps(rec.get("engagement") or {}) + json.dumps(rec.get("gpu") or {})
    cpu = [m.group(0) for m in CPU_RE[stack].finditer(text)][:5]
    gpu = [m.group(0) for m in GPU_RE[stack].finditer(text)][:5]
    if cpu or not gpu:
        status = "cpu_fallback" if cpu else "no_gpu_evidence"
    else:
        status = "gpu"
    return {"tag": cell_dir.name, "stack": stack, "status": status, "cpu_lines": cpu, "gpu_lines": gpu,
            "verdict_before": rec.get("verdict"), "_rec": rec, "_path": str(rp)}


def run(run_dir: Path, write: bool) -> list:
    out = []
    for rp in sorted(Path(run_dir).rglob("record.json")):
        r = check_cell(rp.parent)
        if not r:
            continue
        rec = r.pop("_rec", None)
        path = r.pop("_path", None)
        if write and rec is not None and r["status"] in ("cpu_fallback", "no_gpu_evidence") and rec.get("verdict") == "ok":
            rec["verdict"] = "VOID"
            rec["reason"] = (f"Mac CPU fallback: {r['cpu_lines']}" if r["status"] == "cpu_fallback"
                             else "Mac: no MPS / Metal evidence in the logs (engagement unproven)")
            Path(path).write_text(json.dumps(rec, indent = 1))
        out.append(r)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--write", action = "store_true", help = "turn ok records with a fallback into VOID")
    a = ap.parse_args()
    res = run(Path(a.run_dir), a.write)
    Path(a.run_dir, "mac_fallback.json").write_text(json.dumps(res, indent = 1))
    for r in res:
        print(f"{r['tag']:60s} {r['stack']:8s} {r['status']} {r.get('cpu_lines') or ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
