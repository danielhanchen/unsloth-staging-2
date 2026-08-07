# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Dump an OS-independent inventory of a cache directory.

Emits one sorted line per entry so two OSes' inventories can be diffed
byte-for-byte, plus a verdict on cross-OS portability of every name.
"""

import hashlib
import os
import sys

# Reserved on Windows in any path segment, with or without an extension.
WIN_RESERVED = {"CON", "PRN", "AUX", "NUL"} | {f"COM{i}" for i in range(1, 10)} | {
    f"LPT{i}" for i in range(1, 10)
}
WIN_ILLEGAL = set('<>:"|?*\\') | {chr(c) for c in range(32)}


def name_problems(name):
    bad = []
    hit = sorted(set(name) & WIN_ILLEGAL)
    if hit:
        bad.append("illegal_chars=" + "".join(repr(c) for c in hit))
    if name.rstrip(". ") != name:
        bad.append("trailing_dot_or_space")
    if name.split(".")[0].upper() in WIN_RESERVED:
        bad.append("reserved_device_name")
    if len(name.encode("utf-8")) != len(name):
        bad.append("non_ascii")
    return bad


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    root = sys.argv[1]
    label = sys.argv[2]
    # Hashing every blob is slow on a 3 GiB model; opt in.
    do_hash = "--hash" in sys.argv

    if not os.path.isdir(root):
        print(f"INVENTORY {label}: MISSING {root}")
        return 0

    rows, problems, lower_seen = [], [], {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(dirnames) + sorted(filenames):
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root).replace(os.sep, "/")

            if os.path.islink(full):
                kind = "LINK"
                target = os.readlink(full).replace(os.sep, "/")
                size = -1
                extra = f"-> {target} dangling={not os.path.exists(full)}"
            elif os.path.isdir(full):
                kind, size, extra = "DIR", -1, ""
            else:
                kind = "FILE"
                size = os.path.getsize(full)
                extra = f"sha256={sha256(full)}" if do_hash else ""

            rows.append(f"{kind}\t{size}\t{rel}\t{extra}".rstrip())

            for seg in rel.split("/"):
                for p in name_problems(seg):
                    problems.append(f"{rel}: {p}")
            lower_seen.setdefault(rel.lower(), []).append(rel)

    collisions = {k: v for k, v in lower_seen.items() if len(set(v)) > 1}

    print(f"===== INVENTORY {label} root={root} entries={len(rows)} =====")
    for r in rows:
        print(r)
    print(f"===== PORTABILITY {label} =====")
    print(f"WIN_NAME_PROBLEMS={len(problems)}")
    for p in problems:
        print("  PROBLEM " + p)
    print(f"CASE_COLLISIONS={len(collisions)}")
    for k, v in collisions.items():
        print(f"  COLLISION {k} <- {sorted(set(v))}")
    n_links = sum(1 for r in rows if r.startswith("LINK"))
    print(f"SYMLINKS={n_links}")
    print(f"VERDICT {label}: "
          f"win_safe_names={'YES' if not problems and not collisions else 'NO'} "
          f"symlinks={n_links}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
