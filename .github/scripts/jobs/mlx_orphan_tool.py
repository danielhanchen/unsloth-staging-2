"""Orphan role=tool replay on the real Studio MLX backend (unslothai/unsloth#11669, PR #11733).

    python jobs/mlx_orphan_tool.py [--model M] [--pr 11733] [--out job.json]

Run from the root of a checkout holding the PR (staging branch = fresh main + PR). Two arms, each
in its own process so nothing is shared through the module cache:
  head  the checkout as is
  base  a `git worktree` of HEAD with the PR's chat_template_helpers.py hunk reverse-applied
        (fallback: that file from upstream main)
Each arm loads --model through MLXInferenceBackend.load_model (real mlx-lm on Apple Silicon) and
generates a few greedy tokens with generate_chat_response for every (template, history) cell:
  native   the model's own template
  strict   the same template plus the guard the issue quotes (its "strict Qwen-family template")
  gptoss   unsloth/gpt-oss-20b's real template, the one shipped template found to reject an orphan
Verdict (exit 1 on FAIL): every valid history renders and generates identically in both arms; every
cell the base rejects, the head generates for; the head never fails where the base succeeded.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.request

HELPERS = "studio/backend/core/inference/chat_template_helpers.py"
STRICT_GUARD = (
    "{%- for m in messages %}{%- if m.role == 'tool' and (loop.first or "
    "messages[loop.index0 - 1].role not in ['assistant', 'tool']) %}"
    "{{- raise_exception('A tool message must follow an assistant or tool message.') }}"
    "{%- endif %}{%- endfor %}"
)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "search the web",
            "parameters": {
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
            },
        },
    }
]
CALL = {
    "id": "call_1",
    "type": "function",
    "function": {"name": "web_search", "arguments": '{"q": "weather"}'},
}
RESULT = {"role": "tool", "tool_call_id": "call_1", "name": "web_search", "content": "21C sunny"}
U = {"role": "user", "content": "What is the weather?"}
# content "" as the route sends it (_extract_content_parts): Qwen3 reads `"</think>" in content`.
HISTORIES = {
    "valid": [U, {"role": "assistant", "content": "", "tool_calls": [CALL]}, RESULT],
    "user->tool": [U, RESULT],
    "user->tool(no id/name)": [U, {"role": "tool", "content": "21C sunny"}],
    "valid,then orphan": [
        U,
        {"role": "assistant", "content": "", "tool_calls": [CALL]},
        RESULT,
        {"role": "assistant", "content": "It is 21C."},
        {"role": "user", "content": "And tomorrow?"},
        {"role": "tool", "tool_call_id": "call_9", "name": "web_search", "content": "rain"},
    ],
}


def _sh(*cmd, cwd = None):
    return subprocess.run(cmd, cwd = cwd, capture_output = True, text = True)


def make_base(pr: int, workdir: str) -> dict:
    """A worktree of HEAD with the PR's helper change undone; returns how it was built."""
    head = _sh("git", "rev-parse", "HEAD").stdout.strip()
    base_dir = os.path.join(workdir, "base")
    r = _sh("git", "worktree", "add", "--detach", base_dir, "HEAD")
    if r.returncode:
        raise SystemExit(f"git worktree add failed: {r.stderr}")
    info = {"head_sha": head, "base_dir": base_dir}
    diff_path = os.path.join(workdir, f"pr{pr}.diff")
    try:
        with urllib.request.urlopen(
            f"https://patch-diff.githubusercontent.com/raw/unslothai/unsloth/pull/{pr}.diff",
            timeout = 60,
        ) as resp:
            open(diff_path, "wb").write(resp.read())
        r = _sh("git", "apply", "-R", f"--include={HELPERS}", diff_path, cwd = base_dir)
        if r.returncode == 0:
            info["method"] = f"reverse-applied PR #{pr} diff to {HELPERS}"
    except Exception as e:  # noqa: BLE001
        r = None
        info["diff_error"] = f"{type(e).__name__}: {e}"
    if "method" not in info:
        if r is not None:
            info["reverse_apply_error"] = r.stderr[-500:]
        with urllib.request.urlopen(
            f"https://raw.githubusercontent.com/unslothai/unsloth/main/{HELPERS}", timeout = 60
        ) as resp:
            open(os.path.join(base_dir, HELPERS), "wb").write(resp.read())
        info["method"] = f"{HELPERS} from upstream main"
    changed = _sh("git", "diff", "--stat", cwd = base_dir).stdout.strip()
    info["base_vs_head"] = changed
    if HELPERS.split("/")[-1] not in changed:
        raise SystemExit(f"base arm is identical to head ({info}); a void comparison")
    return info


def run_arm(backend_dir: str, model: str, out: str) -> None:
    """One arm: load the model through Studio's MLX backend and generate every cell."""
    sys.path.insert(0, backend_dir)
    from types import SimpleNamespace

    from huggingface_hub import hf_hub_download
    from core.inference import chat_template_helpers
    from core.inference.mlx_inference import MLXInferenceBackend

    res = {
        "backend_dir": backend_dir,
        "helpers_file": chat_template_helpers.__file__,
        "has_repair": hasattr(chat_template_helpers, "_repair_orphan_tool_results"),
        "cells": {},
    }
    import gc

    backend = None
    native = None
    gptoss = open(hf_hub_download("unsloth/gpt-oss-20b", "chat_template.jinja")).read()
    templates = {"native": None, "strict": "STRICT", "gptoss": gptoss}
    for tname, override in templates.items():
        t0 = time.time()
        # A fresh backend per template: each load goes through Studio's own override installation.
        backend = None
        gc.collect()
        backend = MLXInferenceBackend()
        if override == "STRICT":
            override = STRICT_GUARD + native
        ok = backend.load_model(
            SimpleNamespace(identifier = model, is_vision = False, is_lora = False, is_gguf = False),
            max_seq_length = 4096,
            chat_template_override = override,
        )
        installed = getattr(backend._tokenizer, "chat_template", None)
        if tname == "native":
            native = installed
        print(f"LOAD {tname} ok={ok} {round(time.time() - t0, 1)}s", flush = True)
        res[f"load_{tname}"] = {
            "ok": bool(ok),
            "secs": round(time.time() - t0, 1),
            "template_is_override": override is None or installed == override,
            "override_state": {
                k: v
                for k, v in (getattr(backend, "_template_override", {}) or {}).items()
                if isinstance(v, (str, bool, int, type(None)))
            },
        }
        if override is not None and installed != override:
            # Studio refused the override: install it directly so the cell still tests this template.
            backend._tokenizer.chat_template = override
            res[f"load_{tname}"]["installed_directly"] = True
        for hname, msgs in HISTORIES.items():
            for with_tools in (False, True):
                key = f"{tname} | {hname} | tools={with_tools}"
                t1 = time.time()
                try:
                    pieces = list(
                        backend.generate_chat_response(
                            json.loads(json.dumps(msgs)),
                            tools = TOOLS if with_tools else None,
                            temperature = 0.0,
                            top_p = 1.0,
                            top_k = 0,
                            max_new_tokens = 8,
                            seed = 0,
                            enable_thinking = False,
                        )
                    )
                    text = (
                        pieces[-1]
                        if pieces and all(isinstance(p, str) for p in pieces)
                        else str(pieces)
                    )
                    res["cells"][key] = {
                        "status": "OK",
                        "n_chunks": len(pieces),
                        "text": text[-200:],
                    }
                except Exception as e:  # noqa: BLE001
                    res["cells"][key] = {
                        "status": "FAIL",
                        "error": f"{type(e).__name__}: {str(e)[:300]}",
                        "tb": traceback.format_exc()[-800:],
                    }
                res["cells"][key]["secs"] = round(time.time() - t1, 1)
                print(
                    f"ARM {backend_dir.split('/studio/')[0].rsplit('/', 1)[-1]} {key:55s} "
                    f"{res['cells'][key]['status']} {res['cells'][key]['secs']}s "
                    f"{res['cells'][key].get('error', '')[:120]}",
                    flush = True,
                )
                if res["cells"][key]["status"] == "FAIL":
                    print(res["cells"][key]["tb"], flush = True)
                # Written as it goes, so a timeout still leaves the finished cells behind.
                json.dump(res, open(out, "w"), indent = 1)
    json.dump(res, open(out, "w"), indent = 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default = "mlx-community/Qwen3-0.6B-4bit")
    ap.add_argument("--pr", type = int, default = 11733)
    ap.add_argument("--out", default = "mlx_orphan_tool.json")
    ap.add_argument("--tiny", action = "store_true", help = "ignored (staging_ci)")
    ap.add_argument("--arm", help = argparse.SUPPRESS)
    ap.add_argument("--arm-out", help = argparse.SUPPRESS)
    a = ap.parse_args()
    if a.arm:
        return run_arm(a.arm, a.model, a.arm_out)

    import platform
    import importlib.metadata as md

    def ver(p):
        try:
            return md.version(p)
        except md.PackageNotFoundError:
            return None

    work = tempfile.mkdtemp(prefix = "orphan_tool_")
    report = {
        "model": a.model,
        "pr": a.pr,
        "machine": platform.machine(),
        "platform": platform.platform(),
        "versions": {p: ver(p) for p in ("mlx", "mlx-lm", "mlx-vlm", "transformers")},
    }
    report["base"] = make_base(a.pr, work)
    arms = {
        "head": os.path.abspath("studio/backend"),
        "base": os.path.join(report["base"]["base_dir"], "studio/backend"),
    }
    for arm, backend_dir in arms.items():
        out = os.path.join(work, f"{arm}.json")
        r = subprocess.run(
            [
                sys.executable,
                os.path.abspath(__file__),
                "--model",
                a.model,
                "--arm",
                backend_dir,
                "--arm-out",
                out,
            ]
        )
        report[arm] = (
            json.load(open(out)) if os.path.exists(out) else {"error": f"arm exited {r.returncode}"}
        )

    problems = []
    head, base = report["head"].get("cells"), report["base"].get("cells")
    if not head or not base:
        problems.append("an arm produced no cells")
    else:
        if not report["head"]["has_repair"] or report["base"]["has_repair"]:
            problems.append("arm identity wrong: head must carry the repair and base must not")
        for key, h in head.items():
            b = base.get(key, {})
            if key.split(" | ")[1] == "valid":
                if h["status"] != "OK" or b.get("status") != "OK" or h.get("text") != b.get("text"):
                    problems.append(f"valid history changed or failed: {key}")
            elif b.get("status") == "FAIL" and h["status"] != "OK":
                problems.append(f"base fails and head still fails: {key}: {h.get('error')}")
            elif b.get("status") == "OK" and h["status"] != "OK":
                problems.append(f"head regressed: {key}")
        base_fail = [k for k, v in base.items() if v["status"] == "FAIL"]
        report["base_failing_cells"] = base_fail
        report["fixed_cells"] = [k for k in base_fail if head.get(k, {}).get("status") == "OK"]
        if not base_fail:
            problems.append("base reproduced nothing: VOID")
    report["problems"] = problems
    report["verdict"] = "PASS" if not problems else "FAIL"
    json.dump(report, open(a.out, "w"), indent = 1)
    print(
        f"VERDICT {report['verdict']} fixed={len(report.get('fixed_cells', []))} "
        f"base_failing={len(report.get('base_failing_cells', []))} problems={problems}",
        flush = True,
    )
    sys.exit(0 if not problems else 1)


if __name__ == "__main__":
    main()
