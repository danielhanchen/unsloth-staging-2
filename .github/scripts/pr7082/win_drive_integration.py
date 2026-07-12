"""Real Windows drive-root + system-denylist integration for PR #7082.

Runs on a real windows-latest GitHub runner (which has C: and usually D:).
Unlike the mocked unit tests, this calls GetLogicalDrives for real and drives
the real browse resolver against real C:\\Windows, proving both the headline
feature (browse across real drives) and the security fix (system dirs blocked)
on genuine Windows path semantics. Exits non-zero on any failure.
"""
import ast
import os
import platform
import sys
import types
from pathlib import Path
from typing import Optional

sys.path.insert(0, "studio/backend")

assert platform.system() == "Windows", f"expected Windows, got {platform.system()}"

from storage.studio_db import is_denied_system_path  # noqa: E402
from utils.paths.external_media import windows_drive_roots  # noqa: E402

fails = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  :: {detail}" if detail else ""))
    if not cond:
        fails.append(name)


# 1) Real drive discovery via GetLogicalDrives on the actual runner.
roots = windows_drive_roots()
print("windows_drive_roots() ->", roots)
check("drive discovery returns C:\\", any(str(r).upper().startswith("C:") for r in roots), str(roots))

# 2) Denylist on real Windows system dirs (case-insensitive, real normcase).
check("C:\\Windows denied", is_denied_system_path(r"C:\Windows") is True)
check("ProgramFiles denied",
      is_denied_system_path(os.environ.get("ProgramFiles", r"C:\Program Files")) is True)
check("C:\\Users allowed", is_denied_system_path(r"C:\Users") is False)

# 3) AST-extract the real resolver + allowlist check and drive them on the real FS.
src = Path("studio/backend/routes/models.py").read_text(encoding="utf-8")
tree = ast.parse(src)
names = {
    "_is_path_inside_allowlist",
    "_normalize_browse_request_path",
    "_browse_relative_parts",
    "_match_browse_child",
    "_resolve_browse_target",
}
funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
mod = ast.Module(body=funcs, type_ignores=[])
ast.fix_missing_locations(mod)


class HTTP(Exception):
    def __init__(self, status_code, detail=""):
        self.status_code = status_code
        self.detail = detail


ns = {
    "os": os,
    "Path": Path,
    "Optional": Optional,
    "HTTPException": HTTP,
    "logger": types.SimpleNamespace(warning=lambda *a, **k: None, debug=lambda *a, **k: None),
}
exec(compile(mod, "<models>", "exec"), ns)
resolve = ns["_resolve_browse_target"]
is_inside = ns["_is_path_inside_allowlist"]

# Allowlist containment: a drive root authorizes its descendants.
check("C:\\ root allows C:\\Users", is_inside(Path(r"C:\Users"), [Path("C:\\")]) is True)

# Headline security fix on REAL Windows: browsing C:\Windows via a C:\ allowlist
# root is refused by the system denylist even though C:\ is browseable.
try:
    resolve(r"C:\Windows", [Path("C:\\")])
    check("browse C:\\Windows via C:\\ -> 403", False, "no exception raised")
except HTTP as e:
    check("browse C:\\Windows via C:\\ -> 403",
          e.status_code == 403 and "System directories" in e.detail, f"{e.status_code}:{e.detail}")

# A real non-system dir on the drive stays browseable.
try:
    r = resolve(r"C:\Users", [Path("C:\\")])
    check("browse C:\\Users allowed", str(r).upper().startswith("C:\\USERS"), str(r))
except HTTP as e:
    check("browse C:\\Users allowed", False, f"403: {e.detail}")

print("\nRESULT:", "ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
sys.exit(1 if fails else 0)
