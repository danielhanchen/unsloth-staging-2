"""Print (and check) what this runner reports for CPU frequency.

Cross-OS evidence for unslothai/unsloth#8519: psutil <= 7.2.2 reads the Apple
Silicon pmgr voltage-state tables in the wrong unit on M4+, so a 4.5 GHz part
comes back as "4 MHz". Both correction sites are exercised here:

  * unsloth/import_fixes.py   -> patch_psutil_cpu_freq (loaded BY PATH, so the
                                 runner needs neither torch nor transformers)
  * studio/backend/utils/hardware/hardware.py -> cpu_frequency_mhz

On Linux and Windows both must be exact no-ops. On Apple Silicon the reported
value must be plausible; on Intel macOS the patch must decline to install.
"""

import importlib.util
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IS_MAC = platform.system() == "Darwin"
IS_APPLE_SILICON = IS_MAC and platform.machine() == "arm64"


def load_import_fixes():
    spec = importlib.util.spec_from_file_location(
        "unsloth_import_fixes_standalone", ROOT / "unsloth" / "import_fixes.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    import psutil

    print(f"platform      : {platform.platform()}")
    print(f"machine       : {platform.machine()}")
    print(f"psutil        : {psutil.__version__}")

    try:
        before = psutil.cpu_freq()
    except Exception as e:
        before = None
        print(f"psutil.cpu_freq() raised: {e!r}")
    print(f"psutil raw    : {before}")

    fixes = load_import_fixes()
    fixes.patch_psutil_cpu_freq()
    # psutil gates cpu_freq on a RUNTIME cext.has_cpu_freq() probe on macOS, so
    # on Apple Silicon hosts without readable pmgr tables (GitHub's virtualised
    # runners) the attribute is absent entirely and there is nothing to wrap.
    patched = getattr(getattr(psutil, "cpu_freq", None), "__unsloth_patched__", False)
    print(f"patch applied : {patched}")

    if not IS_APPLE_SILICON:
        # Nothing wrapped, so nothing can be rescaled. Compared by identity of
        # the callable: a live Linux clock moves between two calls, so equal
        # readings would be the wrong assertion.
        assert not patched, "the patch must only install on Apple Silicon"

    try:
        after = psutil.cpu_freq()
    except Exception as e:
        after = None
        print(f"patched cpu_freq() raised: {e!r}")
    print(f"psutil patched: {after}")

    sys.path.insert(0, str(ROOT / "studio" / "backend"))
    from utils.hardware import cpu_frequency_mhz

    studio_mhz = cpu_frequency_mhz()
    print(f"studio /api/system frequency_mhz: {studio_mhz}")

    if IS_APPLE_SILICON:
        tables = fixes._apple_cpu_freq_range_mhz()
        print(f"ioreg voltage-state CPU range (MHz): {tables}")
        if studio_mhz is None:
            # Both sources declined: psutil's has_cpu_freq() probe said no and
            # the pmgr tables are not exposed (virtualised runner). Reporting
            # nothing is correct -- the UI simply omits the row.
            assert tables is None, "ioreg found tables but no frequency was reported"
            print("no CPU clock exposed on this host; nothing to correct")
        else:
            assert 500 <= studio_mhz <= 20000, f"implausible frequency: {studio_mhz}"
        if after is not None:
            assert 500 <= after.current <= 20000, (
                f"implausible patched psutil reading: {after}"
            )
    elif before is not None and before.current:
        # Same reading path as before the change, so only the magnitude can be
        # asserted: the clock itself drifts between samples.
        assert studio_mhz is not None and 0 < studio_mhz < 20000, (
            f"non-Apple frequency looks rescaled: {before.current} -> {studio_mhz}"
        )

    print("OK")


if __name__ == "__main__":
    main()
