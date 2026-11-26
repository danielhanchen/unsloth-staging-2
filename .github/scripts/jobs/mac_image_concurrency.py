"""Apple Silicon (MPS): do two concurrent Studio image requests from different accounts share one pipeline call, and
does each member's image match the same prompt and seed rendered alone? Correctness only, tiny model.

Run from a Studio checkout root (the staging branch):
    python .github/scripts/jobs/mac_image_concurrency.py --out job.json

Installs Studio's backend requirements, fetches hf-internal-testing/tiny-stable-diffusion-xl-pipe into the checkout,
and runs _image_concurrency_driver.py with a blocker render holding the slot while 2 clients send their requests.
Exit 1 unless the render ran on MPS, a call of 2 images was observed, and every batched member matches its solo
render bit for bit or within LPIPS 1e-3."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "hf-internal-testing/tiny-stable-diffusion-xl-pipe"


def _sh(cmd, timeout = 1800):
    t = time.time()
    r = subprocess.run(cmd, capture_output = True, text = True, timeout = timeout)
    return (
        r.returncode,
        round(time.time() - t, 1),
        (r.stdout or "")[-3000:] + (r.stderr or "")[-3000:],
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action = "store_true", help = "accepted for the staging --job contract")
    ap.add_argument("--out", required = True)
    ap.add_argument("--skip-install", action = "store_true")
    a = ap.parse_args()
    rec: dict = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
    }
    py = sys.executable
    req = "studio/backend/requirements"
    if not a.skip_install:
        steps = [
            [py, "-m", "pip", "install", "-q", "-r", f"{req}/studio.txt", "psutil", "httpx"],
            [py, "-m", "pip", "install", "-q", "-r", f"{req}/diffusers-pin.txt"],
            [
                py,
                "-m",
                "pip",
                "install",
                "-q",
                "transformers",
                "accelerate",
                "safetensors",
                "pillow",
            ],
            # torch 2.15 skips the hosted runners' paravirtual GPU (pytorch/pytorch#202529), so MPS reads as
            # unavailable there; 2.14 still sees it. torchvision resolves to the matching release.
            [py, "-m", "pip", "install", "-q", "torch<2.15", "torchvision"],
        ]
        rec["install"] = []
        for cmd in steps:
            rc, s, log = _sh(cmd)
            rec["install"].append(
                {
                    "cmd": " ".join(cmd[3:])[:200],
                    "rc": rc,
                    "s": s,
                    "tail": log[-1500:] if rc else "",
                }
            )
            print("INSTALL", rc, s, " ".join(cmd[3:])[:200], flush = True)
        # Best effort: the pixel diff stands without it.
        rc, s, log = _sh([py, "-m", "pip", "install", "-q", "lpips", "torchvision"])
        rec["lpips_install_rc"] = rc
    rc, s, log = _sh(
        [
            py,
            "-c",
            "import torch; print(torch.__version__, torch.backends.mps.is_built(), "
            "torch.backends.mps.is_available()); print(torch.ones(2, device='mps').sum().item())",
        ]
    )
    rec["mps_probe"] = {"rc": rc, "out": log[-800:]}
    print("MPS_PROBE", rc, log[-800:], flush = True)
    from huggingface_hub import snapshot_download

    local = snapshot_download(REPO, local_dir = os.path.abspath("tiny_sdxl"))
    out = os.path.abspath("mac_driver.json")
    cmd = [
        py,
        "-u",
        os.path.join(HERE, "_image_concurrency_driver.py"),
        "--tree",
        "studio/backend",
        "--repo",
        local,
        "--family",
        "sdxl",
        "--clients",
        "2",
        "--requests",
        "1",
        "--width",
        "256",
        "--height",
        "256",
        "--steps",
        "4",
        "--guidance",
        "5",
        "--blocker",
        "--solo-checks",
        "2",
        "--account-role",
        "owner",
        "--label",
        "mac",
        "--out",
        out,
    ]
    t = time.time()
    r = subprocess.run(cmd, capture_output = True, text = True, timeout = 3600)
    rec["driver_rc"], rec["driver_s"] = r.returncode, round(time.time() - t, 1)
    print((r.stdout or "")[-6000:], (r.stderr or "")[-6000:], flush = True)
    row = json.load(open(out, encoding = "utf-8")) if os.path.isfile(out) else {}
    rec["driver"] = row
    checks = row.get("solo_checks") or []
    batched = [c for c in checks if c.get("batched_in", 1) > 1]
    rec["verdict"] = {
        "on_mps": row.get("device") == "mps",
        "batch_of_2_observed": (row.get("call_sizes") or {}).get("2", 0) >= 1,
        "accounts_in_batch": row.get("accounts_per_batch"),
        "own_prompt_ok": row.get("own_prompt_ok"),
        "members_match_solo": bool(batched)
        and all(
            c.get("bitwise") or (c.get("lpips") is not None and c["lpips"] <= 1e-3) for c in batched
        ),
        "request_batching": row.get("request_batching"),
    }
    v = rec["verdict"]
    rec["ok"] = bool(
        v["on_mps"] and v["batch_of_2_observed"] and v["own_prompt_ok"] and v["members_match_solo"]
    )
    with open(a.out, "w", encoding = "utf-8") as fh:
        json.dump(rec, fh, indent = 1, default = str)
    print("JOB_DONE", json.dumps(rec["verdict"]), "ok", rec["ok"], flush = True)
    return 0 if rec["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
