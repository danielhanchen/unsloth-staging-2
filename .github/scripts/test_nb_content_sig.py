#!/usr/bin/env python3
"""Portable, Docker-free test of the body-aware notebook comparison.

Proves the same logic the Linux container uses to decide "did only the install
header / announcements / footer change?" runs identically on macOS and Windows
runners (it is pure stdlib). Builds three notebooks that differ from a base only
in the boilerplate (header, footer) or in the tutorial body, and asserts the
helper reports SAME for boilerplate-only diffs and DIFF for a body change.
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unsloth_nb_content_sig import middle_digest


def write_nb(path, announce, pin, body, footer):
    def md(s):
        return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(True)}

    def code(s):
        return {"cell_type": "code", "metadata": {}, "execution_count": None,
                "outputs": [], "source": s.splitlines(True)}

    cells = [
        md('To run this, press "*Runtime*" and press "*Run all*". rev %s.' % announce),
        md("### News"),
        md("Introducing **Unsloth Studio** - announcement rev %s." % announce),
        code("%%capture\n!pip install unsloth==%s" % pin),
        md("## Data Prep\nLoad the dataset."),
        code("print('train')\n# BODY %s" % body),
        md("And we're done! Join Discord if you need help. Star us on Github. "
           "Footer rev %s.\n\nThis notebook and all Unsloth notebooks are "
           "licensed [LGPL-3.0]." % footer),
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"cells": cells, "metadata": {}, "nbformat": 4,
                   "nbformat_minor": 5}, f)


def main():
    d = tempfile.mkdtemp()
    base = os.path.join(d, "base.ipynb")
    footer_only = os.path.join(d, "footer.ipynb")
    header_only = os.path.join(d, "header.ipynb")
    body_change = os.path.join(d, "body.ipynb")
    write_nb(base, 1, 1, 1, 1)
    write_nb(footer_only, 1, 1, 1, 2)   # only footer differs
    write_nb(header_only, 2, 2, 1, 1)   # only announce + install pin differ
    write_nb(body_change, 1, 1, 2, 1)   # tutorial body differs

    failures = []
    if middle_digest(base) != middle_digest(footer_only):
        failures.append("footer-only diff should be SAME")
    if middle_digest(base) != middle_digest(header_only):
        failures.append("header-only diff should be SAME")
    if middle_digest(base) == middle_digest(body_change):
        failures.append("body change should be DIFF")

    if failures:
        print("FAIL: " + "; ".join(failures))
        sys.exit(1)
    print("OK: body-aware helper portable here -- header/footer ignored, body detected")
    sys.exit(0)


if __name__ == "__main__":
    main()
