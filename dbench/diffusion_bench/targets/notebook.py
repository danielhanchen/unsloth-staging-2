"""ONE notebook builder for every rented VM (Colab tiers and Kaggle T4x2), generalised from
scripts/moss_speed/build_nb.py, scripts/cloud_bench/build_nb.py and scripts/imgcmp_nov5/build_nb.py.

The notebook: hardware facts -> unpack the toolkit bundle (verified against its sha256) -> vm_setup.sh (Studio at
UNSLOTH_SHA, BlineUI at BLINE_REF) -> targets/vm_driver.py over the VM's plan slice. The code that ran is the bundle
at scripts_sha, a copy of which stays in the outputs dir next to the notebook."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import sys
from pathlib import Path

R = "/" + "tmp/dbench"  # on the rented VM


def check_revisions(rev: dict) -> dict:
    sha, ref = rev.get("unsloth_sha"), rev.get("bline_ref")
    if not sha or not re.fullmatch(r"[0-9a-f]{40}", str(sha)):
        raise SystemExit(f"unsloth_sha must be a full 40-hex commit (got {sha!r}): a branch name is not a recorded "
                         "revision; pass --unsloth-sha or set plan revisions")
    if not ref:
        raise SystemExit("bline_ref (a BlineUI release tag or sha) is required: pass --bline-ref or set plan revisions")
    zoo = rev.get("zoo_sha")
    if not zoo or not re.fullmatch(r"[0-9a-f]{40}", str(zoo)):
        # Studio's install.sh --local overlays unsloth-zoo from git main at install time unless UNSLOTH_ZOO_REF is
        # set, so without a pin the Studio arm runs whatever zoo main was when the VM happened to install
        raise SystemExit(f"zoo_sha must be a full 40-hex unsloth-zoo commit (got {zoo!r}): pass --zoo-sha (a branch "
                         "is resolved at prepare time) or set plan revisions.zoo_sha")
    return rev


def needs_gguf_node(cells) -> bool:
    """A baseline-UI cell loads a .gguf model file: the GGUF loader custom node must be installed on the VM."""
    def gguf(v):
        return any(gguf(x) for x in v) if isinstance(v, list) else \
            str((v or {}).get("file") if isinstance(v, dict) else v or "").lower().endswith(".gguf")
    return any(c.get("backend") == "blineui" and any(gguf(v) for v in ((c.get("options") or {}).get("files") or {}).values())
               for c in cells)


def needs_sdcpp(sl: dict) -> bool:
    return any(c.get("backend") == "sdcpp" for c in sl["cells"].values())


def convrot_env(sl: dict) -> str:
    """vm_setup.sh's CONVROT_* variables when a planned file needs the int8 ConvRot quantizer (it fetches the pinned
    copy and the driver hands it to the builders); refuses at prepare time when no pin is configured."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import builders  # noqa: PLC0415
    import extnames  # noqa: PLC0415

    need = builders.needs_tool(sl["cells"].values(), sl.get("downloads"))
    if not need:
        return ""
    try:
        pin = extnames.convrot_tool()
    except RuntimeError as exc:
        raise SystemExit(f"{sl['vm']['vm_id']}: {', '.join(need)} need the int8 ConvRot quantizer; {exc}") from None
    return f" CONVROT_REPO={pin['repo']} CONVROT_SHA={pin['sha']} CONVROT_PATH={pin['path']}"


def build(sl: dict, bundle_bytes: bytes, bundle_meta: dict, rev: dict, platform: str) -> dict:
    """nbformat-v4 notebook dict for one VM slice."""
    vm_id = sl["vm"]["vm_id"]
    sha256 = hashlib.sha256(bundle_bytes).hexdigest()
    assert sha256 == bundle_meta["sha256"], "bundle bytes do not match bundle.json"
    out = f"{R}/out_{vm_id}"
    slice_json = json.dumps(sl)
    cells = [
        "%%bash\nnvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version,compute_cap,clocks.max.sm,"
        "pcie.link.gen.max,pcie.link.width.max --format=csv || echo NO_NVIDIA_SMI\n"
        "free -g; df -h / | tail -1; nproc; grep 'model name' /proc/cpuinfo | head -1; python3 --version\n"
        "python3 -c \"import os;print('HF_TOKEN set', bool(os.environ.get('HF_TOKEN')))\"",
        f"B64 = {base64.b64encode(bundle_bytes).decode()!r}\n"
        "import base64, hashlib, io, tarfile, os, shutil, json\n"
        "raw = base64.b64decode(B64)\n"
        f"assert hashlib.sha256(raw).hexdigest() == {sha256!r}, 'bundle corrupted in transit'\n"
        f"shutil.rmtree('{R}/bench', ignore_errors=True); os.makedirs('{R}/bench', exist_ok=True)\n"
        f"tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz').extractall('{R}/bench')\n"
        f"os.makedirs('{out}', exist_ok=True)\n"
        f"open('{R}/slice_{vm_id}.json', 'w').write({json.dumps(slice_json)})\n"
        # a Python string literal holding the JSON (a JSON object literal is not Python: false / null)
        f"open('{out}/setup_bundle.txt', 'w').write({json.dumps(json.dumps(bundle_meta))})\n"
        f"print('BUNDLE scripts_sha', {bundle_meta['scripts_sha']!r}, 'sha256', {sha256!r}, 'dirty', "
        f"{bundle_meta['dirty']!r})",
        f"%%bash\nset -o pipefail\nexport ROOT={R} BENCH={R}/bench UNSLOTH_SHA={rev['unsloth_sha']} "
        f"ZOO_SHA={rev['zoo_sha']} "
        + (f"BLINE_GGUF_SHA={rev['bline_gguf_sha']} " if needs_gguf_node(sl["cells"].values()) else "") +
        f"BLINE_REF={rev['bline_ref']} HF_HOME={R}/hf NEED_SDCPP={1 if needs_sdcpp(sl) else 0}" + convrot_env(sl)
        + (f" SDCPP_REF={rev['sdcpp_ref']}" if rev.get("sdcpp_ref") else "") + "\n"
        f"bash {R}/bench/diffusion_bench/targets/vm_setup.sh 2>&1 | tee {out}/setup_log.txt | tail -80 || echo SETUP_FAILED",
        "import subprocess, sys, os\n"
        f"spy = open('{R}/studio_py.txt').read().strip() if os.path.exists('{R}/studio_py.txt') else sys.executable\n"
        f"p = subprocess.Popen([spy, '-u', '{R}/bench/diffusion_bench/targets/vm_driver.py', '--slice', "
        f"'{R}/slice_{vm_id}.json', '--root', '{R}', '--bench', '{R}/bench', '--out', '{out}'], "
        "stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)\n"
        "for line in p.stdout:\n    print(line.rstrip(), flush=True)\n"
        "print('DRIVER rc', p.wait(), flush=True)",
    ]
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                       "language_info": {"name": "python"},
                       "dbench": {"vm_id": vm_id, "target": sl["vm"]["target"], "platform": platform,
                                  "scripts_sha": bundle_meta["scripts_sha"], "bundle_sha256": sha256,
                                  "plan_sha256": sl.get("plan_sha256"), "revisions": rev}},
          "cells": []}
    for i, src in enumerate(cells):
        nb["cells"].append({"cell_type": "code", "id": f"c{i}", "metadata": {}, "execution_count": None,
                            "outputs": [], "source": src})
    return nb


def write(nb: dict, path: Path) -> Path:
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text(json.dumps(nb, indent = 1))
    return path
