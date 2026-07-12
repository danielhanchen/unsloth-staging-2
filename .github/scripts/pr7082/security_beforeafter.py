"""Before/after security probe for PR #7082 (runs on ubuntu).

Extracts the browse resolver from two revisions and shows the '/' -> /etc
path-traversal hole existed in the PR as originally submitted and is closed by
the fix commit:

  BEFORE = 8c58a449f  (PR head before the denylist fix)
  AFTER  = HEAD        (PR head with the fix)

Needs full history (fetch-depth: 0). Exits non-zero if the outcome is not
"hole present before, closed after".
"""
import ast
import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Optional

sys.path.insert(0, "studio/backend")

BEFORE_REV = "8c58a449f"
AFTER_REV = "HEAD"


class HTTP(Exception):
    def __init__(self, status_code, detail=""):
        self.status_code = status_code
        self.detail = detail


def load_resolver(rev):
    src = subprocess.check_output(
        ["git", "show", f"{rev}:studio/backend/routes/models.py"]
    ).decode("utf-8")
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
    ns = {
        "os": os,
        "Path": Path,
        "Optional": Optional,
        "HTTPException": HTTP,
        "logger": types.SimpleNamespace(warning=lambda *a, **k: None, debug=lambda *a, **k: None),
    }
    exec(compile(mod, f"<{rev}>", "exec"), ns)
    return ns["_resolve_browse_target"]


def browse_etc(resolve):
    try:
        r = resolve("/etc", [Path("/")])
        return ("REACHABLE", str(r))
    except HTTP as e:
        return (f"BLOCKED_{e.status_code}", e.detail)


before = browse_etc(load_resolver(BEFORE_REV))
after = browse_etc(load_resolver(AFTER_REV))
print(f"BEFORE fix ({BEFORE_REV}): browse /etc via '/' -> {before}")
print(f"AFTER  fix (HEAD):        browse /etc via '/' -> {after}")

ok = before[0] == "REACHABLE" and after[0].startswith("BLOCKED_403")
print("\nRESULT:", "PASS (hole existed before the fix, closed after)" if ok else f"UNEXPECTED before={before} after={after}")
sys.exit(0 if ok else 1)
