# SPDX-License-Identifier: AGPL-3.0-only
import sys

from mlx_speculative_verify import main


if __name__ == "__main__":
    if "--tiny" not in sys.argv:
        sys.argv.append("--tiny")
    main()
