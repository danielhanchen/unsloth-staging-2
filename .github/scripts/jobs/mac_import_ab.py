"""Apple Silicon leg of import_ab.py (staging runs only mac_/mlx_ jobs on Apple Silicon)."""

import os
import runpy
import sys

sys.argv[0] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "import_ab.py")
runpy.run_path(sys.argv[0], run_name = "__main__")
