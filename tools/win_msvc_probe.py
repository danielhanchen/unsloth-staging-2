#!/usr/bin/env python3
"""Windows probe for PR 7704: does the CRT gate behave against REAL triton-windows?

Everything I could test on Linux stubbed Triton's discovery through `sys.modules`, so
it answered whatever the stub said. The two things only a real Windows host with a real
triton-windows install can settle are:

  1. The private API contract. `crt_headers_reachable()` calls
     `triton.windows_utils.find_msvc_winsdk` and `triton.runtime.build.get_cc /
     is_msvc / is_clang_cl`. Whether those exist, and whether `find_msvc_winsdk`
     really returns a 3-tuple, is an assumption everywhere else.
  2. Whether the gate leaves a working Windows box alone. The runner has VS Build
     Tools and no AMD GPU, which is exactly the "ordinary Windows install" shape that
     must never be gated.

AMD is spoofed by planting the ROCm wheel's clang-cl where `get_cc()` looks for it
(`<platlib>/_rocm_sdk_core/lib/llvm/bin/clang-cl.exe`). That is the real trigger for
the AMD path -- `get_cc()` prefers that binary over everything else -- so this exercises
the actual decision on a real host rather than a stub. It does NOT give the runner an
AMD GPU, and nothing here should be read as saying a HIP kernel compiled.

Writes JSON to --out. Never raises: a failed reading is recorded, not thrown.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import sysconfig
from pathlib import Path


def _api_contract() -> dict:
    """What the installed Triton actually exposes, before any of it is relied on."""
    rec: dict = {}
    try:
        import triton  # noqa: PLC0415
        rec["triton_version"] = getattr(triton, "__version__", None)
        rec["triton_file"] = getattr(triton, "__file__", None)
    except Exception as e:  # noqa: BLE001
        rec["triton_import_error"] = f"{type(e).__name__}: {e}"
        return rec

    try:
        import importlib.metadata as md  # noqa: PLC0415
        rec["owning_distributions"] = md.packages_distributions().get("triton") or []
        # `triton.__version__` is "3.6.0" for every 3.6.0.postN, and the post number is
        # what decides whether get_cc() has the ROCm clang-cl branch at all. Record the
        # DIST version or the contract reading below cannot be attributed to a build.
        for dist in ("triton-windows", "triton"):
            try:
                rec[f"dist_version_{dist}"] = md.version(dist)
            except Exception:  # noqa: BLE001
                pass
    except Exception as e:  # noqa: BLE001
        rec["distribution_error"] = f"{type(e).__name__}: {e}"

    # Recompute get_cc()'s own ROCm precondition exactly as it does, so a spoof that
    # did not take can be told apart from a build whose get_cc never looks there.
    try:
        rec["CC_env"] = os.environ.get("CC")
        rocm_cc = os.path.join(
            sysconfig.get_path("platlib"), "_rocm_sdk_core", "lib", "llvm", "bin",
            "clang-cl.exe")
        rec["rocm_cc_path"] = rocm_cc
        rec["rocm_cc_exists"] = os.path.exists(rocm_cc)
        rec["platlib"] = sysconfig.get_path("platlib")
        import triton.runtime.build as _tb  # noqa: PLC0415
        src = Path(_tb.__file__).read_text(
            encoding = "utf-8", errors = "replace")
        rec["get_cc_mentions_rocm_sdk_core"] = "_rocm_sdk_core" in src
    except Exception as e:  # noqa: BLE001
        rec["rocm_precondition_error"] = f"{type(e).__name__}: {e}"

    try:
        from triton.windows_utils import find_msvc_winsdk  # noqa: PLC0415
        rec["has_find_msvc_winsdk"] = True
        try:
            got = find_msvc_winsdk()
            rec["find_msvc_winsdk_len"] = len(got) if hasattr(got, "__len__") else None
            rec["find_msvc_winsdk_type"] = type(got).__name__
            if hasattr(got, "__len__") and len(got) == 3:
                _, inc, _ = got
                rec["include_dir_count"] = len(list(inc))
                rec["stdlib_h_found"] = any(
                    d and os.path.isfile(os.path.join(d, "stdlib.h")) for d in inc
                )
        except Exception as e:  # noqa: BLE001
            rec["find_msvc_winsdk_call_error"] = f"{type(e).__name__}: {e}"
    except Exception as e:  # noqa: BLE001
        rec["has_find_msvc_winsdk"] = False
        rec["find_msvc_winsdk_import_error"] = f"{type(e).__name__}: {e}"

    try:
        from triton.runtime.build import get_cc, is_clang_cl, is_msvc  # noqa: PLC0415
        rec["has_compiler_helpers"] = True
        try:
            cc = get_cc()
            rec["get_cc"] = cc
            rec["get_cc_basename"] = os.path.basename(cc or "")
            rec["is_msvc"] = bool(is_msvc(cc))
            rec["is_clang_cl"] = bool(is_clang_cl(cc))
        except Exception as e:  # noqa: BLE001
            rec["get_cc_error"] = f"{type(e).__name__}: {e}"
    except Exception as e:  # noqa: BLE001
        rec["has_compiler_helpers"] = False
        rec["compiler_helpers_import_error"] = f"{type(e).__name__}: {e}"
    return rec


def _plant_rocm_clang_cl() -> dict:
    """Put a clang-cl where get_cc() looks for the ROCm wheel's copy.

    This is the spoof. `_rocm_clang_cl_present()` tests exactly this path, and real
    `get_cc()` prefers it, so planting it moves the host onto the AMD branch of the
    decision without an AMD GPU. Copies a real executable rather than writing a stub,
    so anything that tries to run it gets a working binary.
    """
    rec: dict = {}
    target = Path(sysconfig.get_path("platlib")) / "_rocm_sdk_core" / "lib" / "llvm" / "bin"
    try:
        target.mkdir(parents = True, exist_ok = True)
        dest = target / "clang-cl.exe"
        src = None
        for cand in (Path(sys.executable),):
            if cand.is_file():
                src = cand
                break
        if src is not None:
            dest.write_bytes(src.read_bytes())
        else:
            dest.write_text("stub", encoding = "utf-8")
        rec["planted"] = str(dest)
        rec["exists"] = dest.is_file()

        # `get_cc` is @functools.lru_cache'd in triton.runtime.build. Reading the API
        # contract before planting warms that cache, and every later call then returns
        # the pre-spoof answer -- which looks exactly like "the ROCm branch does not
        # work" and is not. Drop the cache so the spoof is actually observable.
        try:
            from triton.runtime.build import get_cc  # noqa: PLC0415
            cache_clear = getattr(get_cc, "cache_clear", None)
            rec["get_cc_was_cached"] = cache_clear is not None
            if cache_clear is not None:
                cache_clear()
        except Exception as e:  # noqa: BLE001
            rec["cache_clear_error"] = f"{type(e).__name__}: {e}"
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


def _gate(label: str, checkout: Path, strip_msvc_env: bool) -> dict:
    """Import the checkout's gate and run it. Fresh module each time."""
    rec: dict = {"scenario": label}
    saved = {}
    if strip_msvc_env:
        # The PR's central claim is that clearing these changes nothing, because
        # Triton's discovery reads vswhere, the registry and Program Files instead.
        # Measuring that on a real host is the point.
        for k in ("INCLUDE", "LIB", "LIBPATH", "VCINSTALLDIR", "WindowsSdkDir"):
            saved[k] = os.environ.pop(k, None)
    try:
        backend = str(checkout / "studio" / "backend")
        if backend not in sys.path:
            sys.path.insert(0, backend)
        for stale in [m for m in list(sys.modules) if m.startswith("core.")]:
            del sys.modules[stale]
        os.environ.pop("TORCHDYNAMO_DISABLE", None)

        from core._msvc_env import crt_headers_reachable, gate_torch_compile_on_windows  # noqa: PLC0415
        rec["reachable"] = bool(crt_headers_reachable())

        import logging  # noqa: PLC0415
        gate_torch_compile_on_windows(logging.getLogger("winprobe"))
        rec["dynamo_disabled"] = os.environ.get("TORCHDYNAMO_DISABLE") == "1"
    except Exception as e:  # noqa: BLE001
        import traceback  # noqa: PLC0415
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["tb"] = traceback.format_exc()[-1200:]
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    args = ap.parse_args()

    obs: dict = {
        "platform": sys.platform,
        "python": sys.version.split()[0],
        "has_msvc_env_module": (
            args.checkout / "studio" / "backend" / "core" / "_msvc_env.py"
        ).is_file(),
    }

    obs["api_contract_before_spoof"] = _api_contract()
    if obs["has_msvc_env_module"]:
        # The ordinary Windows install: VS present, no ROCm wheel. Must not be gated.
        obs["stock_runner"] = _gate("stock_runner", args.checkout, False)
        obs["stock_runner_env_stripped"] = _gate(
            "stock_runner_env_stripped", args.checkout, True)

    obs["rocm_spoof"] = _plant_rocm_clang_cl()
    obs["api_contract_after_spoof"] = _api_contract()
    if obs["has_msvc_env_module"]:
        # AMD shape: get_cc() should now prefer the planted clang-cl, so the gate has
        # to actually consult the headers rather than short-circuiting on TinyCC.
        obs["amd_spoofed"] = _gate("amd_spoofed", args.checkout, False)
        obs["amd_spoofed_env_stripped"] = _gate(
            "amd_spoofed_env_stripped", args.checkout, True)

    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    print(json.dumps(obs, indent = 2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
