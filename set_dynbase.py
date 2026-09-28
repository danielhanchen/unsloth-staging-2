"""Harness only: mark PE images DYNAMICBASE so mandatory ASLR maps them at one per-boot base shared by every process.

Usage: python set_dynbase.py <dir> [<dir> ...]
Skips images without a relocation table (they cannot be rebased); prints a count summary.
"""

import struct
import sys
from pathlib import Path

IMAGE_DLLCHARACTERISTICS_DYNAMIC_BASE = 0x0040
IMAGE_FILE_RELOCS_STRIPPED = 0x0001


def patch(path):
    data = bytearray(path.read_bytes())
    if data[:2] != b"MZ":
        return "not-pe"
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe:pe + 4] != b"PE\0\0":
        return "not-pe"
    characteristics = struct.unpack_from("<H", data, pe + 4 + 18)[0]
    opt = pe + 24
    magic = struct.unpack_from("<H", data, opt)[0]
    if magic != 0x20B:
        return "not-pe32plus"
    reloc_size = struct.unpack_from("<I", data, opt + 112 + 5 * 8 + 4)[0]
    if characteristics & IMAGE_FILE_RELOCS_STRIPPED or reloc_size == 0:
        return "no-relocs"
    dll_chars = struct.unpack_from("<H", data, opt + 70)[0]
    if dll_chars & IMAGE_DLLCHARACTERISTICS_DYNAMIC_BASE:
        return "already"
    struct.pack_into("<H", data, opt + 70, dll_chars | IMAGE_DLLCHARACTERISTICS_DYNAMIC_BASE)
    path.write_bytes(bytes(data))
    return "patched"


if __name__ == "__main__":
    counts = {}
    for root in sys.argv[1:]:
        for f in sorted(Path(root).iterdir()):
            if f.suffix.casefold() not in {".exe", ".dll"} or not f.is_file():
                continue
            try:
                result = patch(f)
            except Exception as exc:  # noqa: BLE001 - harness diagnostics
                result = f"error {type(exc).__name__}"
            counts[result] = counts.get(result, 0) + 1
            if f.name in {"msys-2.0.dll", "bash.exe", "ls.exe", "cat.exe"} or result.startswith("error"):
                print(f"DYNBASE {f.name} {result}", flush = True)
    print("DYNBASE_SUMMARY", counts, flush = True)
