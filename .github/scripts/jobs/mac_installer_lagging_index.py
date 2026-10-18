"""macOS (Apple Silicon) leg of installer_lagging_index: staging runs only mac_* / mlx_* jobs there."""

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).with_name("installer_lagging_index.py")), run_name = "__main__")
