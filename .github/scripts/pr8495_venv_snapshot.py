# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""What is installed in THIS interpreter, as a stable, diffable list.

`pip freeze` is not stable across the comparison this CI makes: it reports an
editable or path install as the URL or directory it came from, and the old/new
comparison deliberately installs from two different checkouts, so those lines
differ for a reason that has nothing to do with the code under test. Their
distribution names still have to match, so they are emitted without a version.

Run with the venv's own interpreter; prints `name==version` per line, sorted.
"""

from __future__ import annotations

import importlib.metadata as metadata


def main() -> None:
    rows = []
    for dist in metadata.distributions():
        name = (dist.metadata["Name"] or "").strip().lower().replace("_", "-")
        if not name:
            continue
        direct_url = None
        try:
            direct_url = dist.read_text("direct_url.json")
        except Exception:  # noqa: BLE001
            direct_url = None
        if direct_url:
            # Installed from a path, VCS or editable checkout: the version is the
            # checkout's, so record presence only.
            rows.append(f"{name}==<local checkout>")
        else:
            rows.append(f"{name}=={dist.version}")
    print("\n".join(sorted(set(rows))))


if __name__ == "__main__":
    main()
