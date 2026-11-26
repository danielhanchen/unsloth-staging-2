"""Apple Silicon staging legs only run mac_* / mlx_* jobs: this runs mxc_bench.py's Seatbelt path there."""

import runpy
import sys
from pathlib import Path

target = Path(__file__).with_name("mxc_bench.py")
sys.argv[0] = str(target)
runpy.run_path(str(target), run_name = "__main__")
