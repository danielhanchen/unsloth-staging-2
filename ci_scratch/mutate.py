"""Remove `.no_proxy()` from the loopback helper to simulate the pre-PR code."""
import pathlib
import sys

root = pathlib.Path(__file__).resolve().parent.parent
helper = root / "studio" / "src-tauri" / "src" / "loopback_http.rs"
lines = helper.read_text().splitlines(keepends=True)
kept = [line for line in lines if line.strip() != ".no_proxy()"]
if len(kept) == len(lines):
    sys.exit("no `.no_proxy()` line found - the helper is not what this review assumed")
helper.write_text("".join(kept))
print("mutated: removed %d .no_proxy() line(s)" % (len(lines) - len(kept)))
