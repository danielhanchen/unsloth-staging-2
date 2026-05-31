"""Live simulation against the real ggml-org and unslothai/llama.cpp release
feeds. Confirms the new selector picks a sane bundle for the actual published
assets, and that the old selector exhibits the bug.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(fn, name):
    spec = importlib.util.spec_from_file_location(name, HERE / fn)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


OLD = load("old_ilp.py", "old_live")
NEW = load("new_ilp.py", "new_live")


def gh(url):
    req = urllib.request.Request(url, headers={"User-Agent": "sim"})
    tok = os.environ.get("GH_TOKEN")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def host(mod, system, driver, sm="120"):
    return mod.HostInfo(
        system=system, machine=("AMD64" if system == "Windows" else "x86_64"),
        is_windows=system == "Windows", is_linux=system == "Linux", is_macos=False,
        is_x86_64=True, is_arm64=False, nvidia_smi="nvidia-smi",
        driver_cuda_version=driver, compute_caps=[sm], visible_cuda_devices=None,
        has_physical_nvidia=True, has_usable_nvidia=True,
    )


def main():
    # --- Windows: ggml-org latest ---
    rel = gh("https://api.github.com/repos/ggml-org/llama.cpp/releases/latest")
    tag = rel["tag_name"]
    assets = {a["name"]: a["browser_download_url"] for a in rel["assets"]
              if "win-cuda" in a["name"]}
    print(f"ggml-org {tag} win-cuda assets: {sorted(n for n in assets if n.startswith('llama-'))}")

    def win(mod, driver):
        h = host(mod, "Windows", driver)
        mod.detected_windows_runtime_lines = lambda: ([], {})
        a = mod.windows_cuda_attempts(h, tag, assets, None)
        return a[0].name if a else None

    print("\nWindows selection (driver -> old | new):")
    bug_seen = fix_seen = False
    for d in [(12, 8), (13, 1), (13, 3)]:
        o, n = win(OLD, d), win(NEW, d)
        print(f"  {d[0]}.{d[1]:<2} -> {o}  |  {n}")
        if d[0] == 13 and o and "12.4" in o:
            bug_seen = True
        if d[0] == 13 and n and "13" in n.split("cuda-")[1]:
            fix_seen = True
    assert bug_seen, "expected old code to mis-select 12.4 for a cuda13 driver"
    assert fix_seen, "expected new code to select a cuda13 build"
    print("  -> OLD reproduces the bug; NEW fixes it. OK")

    # --- Linux: unslothai/llama.cpp latest ---
    lrel = gh("https://api.github.com/repos/unslothai/llama.cpp/releases/latest")
    ltag = lrel["tag_name"]
    bundle = NEW.parse_direct_linux_release_bundle("unslothai/llama.cpp", lrel)
    cuda_arts = sorted({a.asset_name for a in bundle.artifacts if a.install_kind == "linux-cuda"})
    print(f"\nunslothai/llama.cpp {ltag} parsed linux-cuda artifacts: {len(cuda_arts)}")
    for n in cuda_arts:
        print(f"    {n}")

    def lin(driver, sm="120"):
        h = host(NEW, "Linux", driver, sm=sm)
        NEW.detected_linux_runtime_lines = lambda: (["cuda13", "cuda12"], {})
        sel = NEW.linux_cuda_choice_from_release(h, bundle)
        return sel.attempts[0].name if sel and sel.attempts else None

    print("\nLinux selection (driver, sm120 -> bundle):")
    for d in [(12, 8), (13, 3)]:
        print(f"  {d[0]}.{d[1]:<2} -> {lin(d)}")
    assert lin((13, 3)) and "cuda13" in lin((13, 3)), "expected a cuda13 linux bundle for a 13.x driver"
    print("  -> Linux selects the major-matched bundle. OK")

    print("\nLIVE SIMULATION PASSED")


if __name__ == "__main__":
    main()
