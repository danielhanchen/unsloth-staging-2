"""Apple Silicon leg of pytest_ab (staging runs mac_* jobs on macOS)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pytest_ab import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
