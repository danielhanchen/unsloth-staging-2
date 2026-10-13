"""Apple Silicon leg of install_equiv (staging runs mac_* jobs on macOS)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from install_equiv import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
