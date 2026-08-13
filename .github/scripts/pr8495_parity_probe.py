# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Does PR #8495/#8599 change what the installer does ON THIS MACHINE?

Loads install_python_stack.py from the PR tree and from the merge base, on the
real runner, and compares the flags each computes and the exact uv command each
renders for a requirements install. Any platform other than native ARM64 Windows
must produce identical output, which is the PR's central claim.

Cheap enough to run on every runner in the matrix: nothing is installed, uv is
never invoked, subprocess.run is stubbed out.

    python pr8495_parity_probe.py <pr-tree> <base-tree> [--expect-change]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROBE = r"""
import json, os, sys, types, subprocess, tempfile, sysconfig, platform
import urllib.request  # before anything patches sys.platform

tree = sys.argv[1]
sys.prefix = tempfile.mkdtemp(prefix = "parity-")   # no local manifest may answer
sys.path.insert(0, tree)
sys.path.insert(0, os.path.join(tree, "backend"))

calls = []
def fake_run(cmd, *a, **k):
    if isinstance(cmd, (list, tuple)):
        calls.append([str(c) for c in cmd])
    return types.SimpleNamespace(returncode = 0, stdout = "", stderr = "")
subprocess.run = fake_run

import install_python_stack as ips
ips.USE_UV = True
ips.VERBOSE = False
ips._install_env_for_cmd = lambda cmd: os.environ.copy()

out = {
    "platform_tag": sysconfig.get_platform(),
    "machine": platform.machine(),
    "sys_platform": sys.platform,
    "NO_TORCH": bool(getattr(ips, "NO_TORCH", False)),
    "NO_DATASETS": bool(getattr(ips, "NO_DATASETS", False)),
    "IS_WINDOWS_ARM64_PYTHON": bool(getattr(ips, "IS_WINDOWS_ARM64_PYTHON", False)),
    "PLATFORM_LACKS_TORCHCODEC_WHEEL": bool(ips.PLATFORM_LACKS_TORCHCODEC_WHEEL),
    "UV_OVERRIDE": os.environ.get("UV_OVERRIDE", ""),
}

from pathlib import Path
req = Path(tree) / "backend" / "requirements" / "studio.txt"
calls.clear()
ips.pip_install("studio deps", "--no-cache-dir", req = req)
out["studio_cmd"] = calls[0] if calls else []
calls.clear()
ips.pip_install("core", "--no-cache-dir", "--upgrade-package", "unsloth", "unsloth")
out["core_cmd"] = calls[0] if calls else []
print("@@JSON@@" + json.dumps(out))
"""


def probe(tree: Path) -> dict:
    result = subprocess.run(
        [sys.executable, "-c", PROBE, str(tree)],
        capture_output = True, text = True, timeout = 900,
    )
    if "@@JSON@@" not in result.stdout:
        raise SystemExit(
            f"probe of {tree} produced no verdict:\n{(result.stderr or result.stdout)[-3000:]}"
        )
    return json.loads(result.stdout.split("@@JSON@@", 1)[1].splitlines()[0])


def normalise(cmd: list[str], tree: Path) -> list[str]:
    """Absolute paths and per-run temp names differ by construction.

    The tree path is resolved first: the two checkouts are passed as `.` and
    `../basetree`, and substituting those verbatim replaced nothing on one side and
    the bare word "studio" everywhere on the other, which turned studio.txt into
    <tree>.txt and made two identical commands look different.
    """
    root = str(tree.resolve()).replace("\\", "/")
    out = []
    for token in cmd:
        token = token.replace("\\", "/").replace(root, "<tree>")
        if "-filtered-" in token or "-arm64-" in token or "parity-" in token:
            token = "<generated>"
        out.append(token)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pr_tree", type = Path)
    parser.add_argument("base_tree", type = Path)
    parser.add_argument("--expect-change", action = "store_true",
                        help = "native ARM64 Windows: the commands SHOULD differ there")
    args = parser.parse_args()

    pr = probe(args.pr_tree / "studio")
    base = probe(args.base_tree / "studio")

    print(f"runner: {pr['sys_platform']} {pr['machine']} interpreter {pr['platform_tag']}")
    print(f"flags (PR):   NO_DATASETS={pr['NO_DATASETS']} "
          f"IS_WINDOWS_ARM64_PYTHON={pr['IS_WINDOWS_ARM64_PYTHON']} "
          f"UV_OVERRIDE={'set' if pr['UV_OVERRIDE'] else 'unset'}")
    print(f"flags (base): NO_DATASETS={base['NO_DATASETS']}")

    problems = []
    for key in ("studio_cmd", "core_cmd"):
        before = normalise(base[key], args.base_tree / "studio")
        after = normalise(pr[key], args.pr_tree / "studio")
        same = before == after
        print(f"\n{key}: {'identical' if same else 'DIFFERENT'}")
        if not same:
            print(f"  base: {' '.join(before)}")
            print(f"  pr  : {' '.join(after)}")
            if not args.expect_change:
                problems.append(f"{key} differs on a platform the PR must not touch")
    if pr["PLATFORM_LACKS_TORCHCODEC_WHEEL"] != base["PLATFORM_LACKS_TORCHCODEC_WHEEL"]:
        problems.append("the torchcodec gate changed")

    is_native_arm64_windows = pr["platform_tag"].lower() == "win-arm64"
    if pr["NO_DATASETS"] and not is_native_arm64_windows:
        problems.append("the inference-only tier switched itself on off ARM64 Windows")
    if args.expect_change and not pr["NO_DATASETS"]:
        problems.append("the tier did NOT engage on a native ARM64 Windows interpreter")

    print()
    if problems:
        for problem in problems:
            print(f"FAIL: {problem}")
        return 1
    print("PASS: this runner renders the same install commands as before the PR"
          if not args.expect_change else
          "PASS: the tier engaged, as intended on native ARM64 Windows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
