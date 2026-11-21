"""Run a target adapter the way ``python -m targets.<kind>`` does, from any directory and under PYTHONSAFEPATH=1
(safe-path mode keeps the cwd off sys.path, so ``-m targets.<kind>`` cannot find the package there):

  python diffusion_bench/target.py colab prepare PLAN.json --out DIR --unsloth-sha SHA --bline-ref TAG
  python diffusion_bench/target.py amd collect DIR_OF_gh_run_download --plan PLAN.json
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    sys.path.insert(0, str(HERE))
    from targets import KINDS  # noqa: PLC0415

    kinds = [k for k in KINDS if k != "tester"]
    if len(sys.argv) < 2 or sys.argv[1] not in kinds:
        raise SystemExit(f"usage: target.py {{{','.join(kinds)}}} prepare|collect ... (got {sys.argv[1:2]})")
    kind = sys.argv[1]
    sys.argv = [f"targets.{kind}", *sys.argv[2:]]
    runpy.run_module(f"targets.{kind}", run_name = "__main__", alter_sys = True)


if __name__ == "__main__":
    main()
