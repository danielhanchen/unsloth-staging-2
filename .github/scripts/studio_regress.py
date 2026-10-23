#!/usr/bin/env python3
"""Studio / Desktop / Core PR regression switchboard.

    python studio_regress.py select  --pr N [--json]        what a PR needs, with reasons
    python studio_regress.py run     --pr N [--only a,b] [--tier fast] [--all]
    python studio_regress.py diff    --root outputs/studio_regress/prN
    python studio_regress.py gate    --pr N                  UI gate (FULL|PROBE), studio_regress/gate.py
    python studio_regress.py publish --pr N --report R       differing pairs -> PR body, studio_regress/publish.py
    python studio_regress.py prefetch [--only a,b]           download fixture models for selected targets
    python studio_regress.py gpu-pack --status|--plan F      GPU leases / packing
    python studio_regress.py gc [--days 3] [--dry-run]       reclaim installs / trees unused for days
    python studio_regress.py prebuild --pr N [--main]        build installs into the shared cache ahead of `run`
    python studio_regress.py status  --pr N [--root DIR]     is the recorded run still evidence for the PR head?

Studio commands need Playwright: when the current interpreter lacks it, this re-execs under
$WORKSPACE/temp/venv_pw (created on first use, with Chromium in temp/pw_browsers).
Core targets (jobs/*.py via jobs/ab.py, the regression/ suite) run in --core-python (default
$STUDIO_REGRESS_CORE_PYTHON, else temp/venv_core) and need no Studio; `--gh-repo unslothai/unsloth-zoo` for zoo PRs.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pw_env  # noqa: E402

COMMANDS = {
    "select": "studio_regress.selection",
    "run": "studio_regress.run",
    "diff": "studio_regress.diff",
    "gate": "studio_regress.gate",
    "publish": "studio_regress.publish",
    "prefetch": "studio_regress.prefetch",
    "gpu-pack": "studio_regress.gpu_pack",
    "gc": "studio_regress.gc",
    "prebuild": "studio_regress.prebuild",
    "status": "studio_regress.provenance",
}


# Commands whose modules import Playwright / numpy / PIL / httpx (select, gc, gpu-pack, status are stdlib).
PW_COMMANDS = {"run", "diff", "gate", "publish", "prefetch", "prebuild"}
PW_MODULES = pw_env.PW_MODULES
PW_PACKAGES = pw_env.PW_PACKAGES
WS = pw_env.base_dir()


def ensure_pw_python(cmd, argv):
    """Re-exec under temp/venv_pw when this interpreter cannot run `cmd` (pw_env.py creates the
    venv and Chromium once, under a lock). Returns only when the current interpreter is fine."""
    os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(WS / "temp" / "pw_browsers"))
    if cmd not in PW_COMMANDS or os.environ.get(pw_env.REEXEC_ENV):
        return
    pw_env.ensure(PW_MODULES, tag = "studio_regress", argv = [str(Path(__file__).resolve()), *argv])


def main(argv = None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help") or argv[0] not in COMMANDS:
        print(__doc__)
        return 0 if argv and argv[0] in ("-h", "--help") else 2
    cmd, rest = argv[0], argv[1:]
    ensure_pw_python(cmd, argv)
    import pw_fast

    pw_fast.patch()
    from studio_regress import engine

    os.environ.update(
        engine._nvsmi_path_env()
    )  # gpu_pack, Studio and every child use the nvidia-smi shim
    try:
        mod = importlib.import_module(COMMANDS[cmd])
    except ModuleNotFoundError as e:
        if e.name == COMMANDS[cmd]:
            print(
                f"`{cmd}` is not implemented in this checkout ({COMMANDS[cmd]} missing)",
                file = sys.stderr,
            )
            return 2
        raise
    return mod.main(rest)


if __name__ == "__main__":
    sys.exit(main())
