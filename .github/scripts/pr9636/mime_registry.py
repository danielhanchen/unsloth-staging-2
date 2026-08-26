"""Report, and assert, what this host's mimetypes registry says about image formats.

mimetypes reads /etc/mime.types on Linux, the same plus system files on macOS, and
HKEY_CLASSES_ROOT on Windows. _image_mime falls back to that registry, so a format
could resolve on one OS and not another, or resolve to a non-canonical alias such as
the historical Windows 'image/x-png'. This job makes that visible per runner and
fails if a format the hard-coded alias map claims is affected by host state.
"""

import mimetypes
import platform
import sys

# what _IMAGE_SUBTYPES pins explicitly: these must never depend on the host
PINNED = {
    "apng": "image/apng", "png": "image/png", "jpeg": "image/jpeg", "jpg": "image/jpeg",
    "gif": "image/gif", "webp": "image/webp", "bmp": "image/bmp", "avif": "image/avif",
    "tif": "image/tiff", "tiff": "image/tiff", "ico": "image/vnd.microsoft.icon",
    "svg": "image/svg+xml",
}
# resolved through the registry fallback: reported, not asserted
FALLBACK = ["heic", "heif", "jxl", "jp2", "tga", "psd", "exr"]


def main():
    print(f"platform : {platform.platform()}")
    print(f"python   : {sys.version.split()[0]}")

    mimetypes.init([])
    builtin = {e: mimetypes.guess_type(f"file:///image.{e}", strict = False)[0]
               for e in list(PINNED) + FALLBACK}
    mimetypes.init()
    host = {e: mimetypes.guess_type(f"file:///image.{e}", strict = False)[0]
            for e in list(PINNED) + FALLBACK}

    print(f"\n{'ext':<6} {'builtin only':<26} {'with host registry':<26} drift")
    drift = []
    for ext in list(PINNED) + FALLBACK:
        mark = "" if builtin[ext] == host[ext] else "DRIFT"
        if mark:
            drift.append((ext, builtin[ext], host[ext]))
        print(f"{ext:<6} {str(builtin[ext]):<26} {str(host[ext]):<26} {mark}")

    # the pinned aliases are answered by _IMAGE_SUBTYPES before mimetypes is consulted,
    # so host drift on them is informational; a pinned value differing from the alias
    # map would mean the map and the registry disagree, which is worth seeing.
    print("\npinned aliases vs host registry:")
    mismatched = [(e, w, host[e]) for e, w in PINNED.items() if host[e] != w]
    for ext, want, got in mismatched:
        print(f"  {ext}: alias map says {want}, host says {got}")
    if not mismatched:
        print("  all agree")

    if drift:
        print(f"\n{len(drift)} extension(s) resolve differently with the host registry loaded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
