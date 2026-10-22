#!/usr/bin/env python3
"""Deterministic targeted test selection for a base..head diff of unsloth / unsloth-zoo.

    python test_select.py --repo PATH --base SHA --head SHA [--json] [--scope DIR]... [--full-tests]

Picks the pytest files (and the Studio Playwright suites) a change can reach, or says FULL with the
reason when it cannot prove that. Stdlib only. The graph is read from the HEAD commit's tree via git
(never the working tree), so the same (repo, base, head) always gives byte-identical output.

Edges, file -> what it depends on (all conservative; a guess always adds an edge, never drops one):
  import      `import a.b` (and the package __init__ chain), `from a import b` (b if a submodule),
              relative imports, importlib.import_module / __import__ with literals, literal-prefix
              f-strings / concatenations (every module under the prefix), code inside `-c` strings.
              Names resolve under the repo root, studio/backend, every ancestor dir of the importer
              and sys.path literals; a name found under none of them matches any file with that path
              suffix.
  literal     a string literal (or shell / PowerShell / data-file token) naming a tracked path, a
              path suffix ("routes/inference.py"), a basename ("install.sh") or a dotted module.
  walker      glob / rglob / iterdir / os.walk / os.listdir / os.scandir on a directory the file
              names: every file under it.
  opaque      a non-literal dynamic import: every module in the importer's own package.
  conftest    each test depends on every conftest.py from the repo root down to its directory.
  table       EXPLICIT_MAP below, for the known non-Python cases.

FULL (today's behaviour) when: a changed file is test infrastructure or environment; a changed
Python module has no static or literal importer at all while the repo has opaque dynamic imports;
a changed file maps to nothing; or the selection is empty for a non-docs change. Docs-only diffs
that nothing references select nothing (mode NONE).
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import json
import os
import re
import subprocess
import sys
import textwrap
from collections import defaultdict, deque
from pathlib import PurePosixPath

VERSION = 1
FULL_ENV = "STUDIO_REGRESS_FULL_TESTS"

# Playwright suites of the studio_regress upstream_ci journey (tests/studio/<script>).
PLAYWRIGHT_SUITES = ("tests/studio/playwright_chat_ui.py", "tests/studio/playwright_extra_ui.py")

# Extra module roots besides the repo root and the importer's ancestors (backend CI runs with
# PYTHONPATH=studio/backend; the Studio server does the same).
EXTRA_ROOTS = ("studio/backend",)

_DOCS_RX = re.compile(
    r"(^|/)[^/]+\.(md|mdx|rst)$|^docs/|^images/|"
    r"^(LICENSE|COPYING|NOTICE|CODE_OF_CONDUCT|CONTRIBUTING|AUTHORS)(\.[a-z]+)?$|"
    r"^\.github/(ISSUE_TEMPLATE|PULL_REQUEST_TEMPLATE)"
)

# Test infrastructure / environment: any change here reruns everything (FULL).
_INFRA_GLOBS = (
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "pytest.ini",
    "tox.ini",
    "noxfile.py",
    "requirements*",
    "**/requirements*",
    "**/requirements/**",
    "constraints*.txt",
    "**/constraints*.txt",
    "uv.lock",
    "**/uv.lock",
    "poetry.lock",
    "Pipfile",
    "Pipfile.lock",
    ".python-version",
    ".github/workflows/**",
    ".github/actions/**",
)
# tests/**/fixtures/** (any fixtures dir inside a test tree).
_FIXTURE_RX = re.compile(r"(^|/)tests?/(.*/)?(fixtures?|fixture_data|testdata|test_data)/")

# Explicit mappings for the known non-Python cases: glob -> what it selects. `tests` entries are
# globs over test files; `suites` are PLAYWRIGHT_SUITES entries; "mapped" with nothing else means
# "known to affect no pytest file / suite beyond what literal references find".
EXPLICIT_MAP = (
    # The UI both suites drive. A syntax error anywhere in the app breaks the build, so any
    # frontend source change selects both, not just the one whose route it looks like.
    {
        "glob": "studio/frontend/**",
        "suites": PLAYWRIGHT_SUITES,
        "exclude": (
            "studio/frontend/tests/**",
            "**/*.test.ts",
            "**/*.test.tsx",
            "**/*.test.mts",
            "**/*.test.js",
            "**/*.test.mjs",
            "**/*.spec.ts",
            "**/*.spec.tsx",
        ),
    },
    # Frontend unit tests (node --test) and their helpers: the per-OS node step runs a PR's own;
    # nothing in pytest or the Playwright suites executes them.
    {"glob": "studio/frontend/**", "mapped": True},
    # Backend data files (json / yaml / jinja / templates): the backend suite plus the chat suite.
    # Python files under studio/backend go through the import graph instead.
    {
        "glob": "studio/backend/**",
        "tests": ("studio/backend/tests/**",),
        "suites": PLAYWRIGHT_SUITES,
        "non_python_only": True,
    },
    # Installers: the Studio each suite drives is installed with them.
    {"glob": "install.sh", "suites": PLAYWRIGHT_SUITES},
    {"glob": "install.ps1", "suites": PLAYWRIGHT_SUITES},
    {"glob": "studio/setup.sh", "suites": PLAYWRIGHT_SUITES},
    {"glob": "studio/setup.ps1", "suites": PLAYWRIGHT_SUITES},
    {"glob": "studio/*.sh", "suites": PLAYWRIGHT_SUITES},
    {"glob": "studio/*.ps1", "suites": PLAYWRIGHT_SUITES},
    # Desktop (Rust / Tauri) is covered by the desktop journey, not pytest or these suites.
    {"glob": "studio/src-tauri/**", "mapped": True},
)

# Python entry points of the Studio a Playwright suite drives; their import closure is runtime.
STUDIO_RUNTIME_ENTRIES = (
    "studio/backend/main.py",
    "studio/backend/run.py",
    "unsloth_cli/__init__.py",
    "unsloth_cli/__main__.py",
    "cli.py",
    "unsloth-cli.py",
)

_TEXT_SCAN_EXT = (
    ".sh",
    ".bash",
    ".ps1",
    ".psm1",
    ".psd1",
    ".bat",
    ".cmd",
    ".toml",
    ".json",
    ".yaml",
    ".yml",
    ".cfg",
    ".ini",
    ".txt",
    ".in",
    ".j2",
    ".jinja",
    ".tmpl",
    ".spec",
    ".mk",
)
_TEXT_SCAN_MAX = 2_000_000
_TOKEN_RX = re.compile(r"[A-Za-z0-9_.\-/\\${}]+")
_WALK_ATTRS = {"glob", "rglob", "iterdir", "walk"}
_WALK_FUNCS = {
    ("os", "walk"),
    ("os", "listdir"),
    ("os", "scandir"),
    ("glob", "glob"),
    ("glob", "iglob"),
}
_STDLIB = frozenset(getattr(sys, "stdlib_module_names", ()))


def _git(
    repo,
    *args,
    input_ = None,
):
    r = subprocess.run(["git", "-C", repo, *args], capture_output = True, input = input_, check = False)
    if r.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)}: {r.stderr.decode(errors = 'replace').strip()[:300]}"
        )
    return r.stdout


def changed_files(repo, base, head):
    """[(status, path)] of base..head, renames split into delete + add, sorted."""
    out = _git(repo, "diff", "--no-renames", "--name-status", "-z", base, head).decode()
    parts = [p for p in out.split("\0") if p]
    res = []
    for i in range(0, len(parts) - 1, 2):
        res.append((parts[i][:1], parts[i + 1]))
    return sorted(res, key = lambda x: (x[1], x[0]))


def _ls_tree(repo, rev):
    out = _git(repo, "ls-tree", "-r", "-z", "--full-tree", rev).decode()
    files = {}
    for rec in out.split("\0"):
        if not rec:
            continue
        meta, path = rec.split("\t", 1)
        mode, typ, sha = meta.split()
        if typ == "blob":
            files[path] = sha
    return files


def _cat_blobs(repo, shas):
    """{sha: bytes} via one `git cat-file --batch`."""
    shas = sorted(set(shas))
    if not shas:
        return {}
    out = _git(repo, "cat-file", "--batch", input_ = ("\n".join(shas) + "\n").encode())
    res, i = {}, 0
    for sha in shas:
        nl = out.index(b"\n", i)
        header = out[i:nl].split()
        size = int(header[2])
        res[sha] = out[nl + 1 : nl + 1 + size]
        i = nl + 1 + size + 1
    return res


def is_test_file(path):
    name = PurePosixPath(path).name
    return path.endswith(".py") and (name.startswith("test_") or name.endswith("_test.py"))


def _in_test_tree(path):
    return (
        is_test_file(path)
        or any(part in ("tests", "test") for part in path.split("/")[:-1])
        or PurePosixPath(path).name == "conftest.py"
    )


def is_docs(path):
    return bool(_DOCS_RX.search(path))


def _glob(path, pattern):
    if fnmatch.fnmatchcase(path, pattern):
        return True
    if pattern.startswith("**/") and fnmatch.fnmatchcase(path, pattern[3:]):
        return True
    return "/**/" in pattern and fnmatch.fnmatchcase(path, pattern.replace("/**/", "/"))


def infra_reason(path, conftest_roots):
    if path in conftest_roots:
        return f"{path}: root conftest of a test tree"
    for g in _INFRA_GLOBS:
        if _glob(path, g):
            return f"{path}: test infrastructure / environment ({g})"
    if _FIXTURE_RX.search(path):
        return f"{path}: test fixtures"
    return None


# ---------------------------------------------------------------------------------------------
# Graph


class Graph:
    """File-level dependency graph of one commit. deps[f] = set of files f depends on."""

    def __init__(
        self,
        files,
        phantom = (),
    ):
        self.files = sorted(set(files) | set(phantom))
        self.fileset = set(self.files)
        self.deps = defaultdict(set)
        self.kinds = defaultdict(set)  # (src, dst) -> {"import", "literal", ...}
        self.opaque = defaultdict(list)  # file -> [line] non-literal dynamic imports
        self.py_loaders = set()  # files that load modules by path / dynamically
        self.runners = set()  # files that start processes
        self.parse_errors = []
        self.suffix = defaultdict(list)  # "b/c.py" / "c.py" -> [paths]
        self.dirs = defaultdict(list)  # "a/b" -> [paths under it, recursive]
        for p in self.files:
            parts = p.split("/")
            for k in range(len(parts)):
                self.suffix["/".join(parts[k:])].append(p)
            for k in range(1, len(parts)):
                self.dirs["/".join(parts[:k])].append(p)
        self.dirset = set(self.dirs)
        self.dir_suffix = defaultdict(list)  # "backend/routes" -> ["studio/backend/routes"]
        for d in self.dirs:
            parts = d.split("/")
            for k in range(len(parts)):
                self.dir_suffix["/".join(parts[k:])].append(d)

    def add(self, src, dst, kind):
        if dst == src or dst not in self.fileset:
            return
        self.deps[src].add(dst)
        self.kinds[(src, dst)].add(kind)

    def rdeps(self):
        """(exec, read) reverse maps. exec: src runs dst's code (imports, dynamic imports, a script
        it launches, a module it loads by path), so whatever changes dst's behaviour changes src's.
        read: src only reads dst's text / lists it, so only a change to dst itself matters."""
        rx, rr = defaultdict(set), defaultdict(set)
        for s, ds in self.deps.items():
            for d in ds:
                (rx if self.is_exec(s, d) else rr)[d].add(s)
        return rx, rr

    def exec_capable(self, src):
        """src (or a test helper it imports) starts processes or loads code by path. Package
        code has to do it itself; a test may do it through its tree's helpers."""
        cache = self.__dict__.setdefault("_capable", {})
        if src in cache:
            return cache[src]
        seen, stack, hit = {src}, [src], False
        while stack and not hit:
            x = stack.pop()
            if x in self.runners or x in self.py_loaders:
                hit = True
                break
            if not _in_test_tree(x) and x != src:
                continue
            for y in self.deps.get(x, ()):
                if (
                    y not in seen
                    and _in_test_tree(y)
                    and self.kinds[(x, y)] & {"import", "dynamic"}
                ):
                    seen.add(y)
                    stack.append(y)
        cache[src] = hit
        return hit

    def is_exec(self, src, dst):
        kinds = self.kinds[(src, dst)]
        if kinds & {"import", "dynamic", "opaque"}:
            return True
        if not _executable(dst) or is_test_file(dst):
            return False
        if "literal" in kinds:
            # naming a script / module path runs it only in a file that runs or loads things
            return self.exec_capable(src)
        if "walker" in kinds:
            # only a file that loads the .py files / runs the scripts it finds
            return src in (self.py_loaders if dst.endswith(".py") else self.runners)
        return False  # "mention": prose in a test


_EXEC_EXT = (".py", ".sh", ".bash", ".ps1", ".psm1", ".bat", ".cmd")


def _executable(path):
    return path.endswith(_EXEC_EXT) or PurePosixPath(path).name in ("Makefile", "Dockerfile")


def _pkg_inits(path, root):
    """__init__.py files of every package between root and path's directory."""
    out = []
    parts = path.split("/")
    rparts = root.split("/") if root else []
    for k in range(len(rparts) + 1, len(parts)):
        out.append("/".join(parts[:k] + ["__init__.py"]))
    return out


def _conftests_above(fileset, path):
    parts = path.split("/")[:-1]
    out = []
    for k in range(len(parts) + 1):
        c = "/".join(parts[:k] + ["conftest.py"])
        if c in fileset and c != path:
            out.append(c)
    return out


class _Resolver:
    def __init__(self, g):
        self.g = g
        self.conftest_syspath = {}  # conftest.py -> sys.path dirs it adds (for its whole tree)
        self.conftest_unresolved = set()

    def roots_for(
        self,
        path,
        extra = (),
    ):
        """sys.path entries a file's imports can resolve under: the repo root, EXTRA_ROOTS, the
        pytest rootdir-insertion dir (first ancestor that is not a package; a script's own dir),
        sys.path literals of the file and of its conftest chain."""
        roots = {""} | set(EXTRA_ROOTS) | set(extra)
        d = _parent(path)
        while d and (d + "/__init__.py") in self.g.fileset:
            d = _parent(d)
        roots.add(d)
        for c in _conftests_above(self.g.fileset, path):
            roots |= self.conftest_syspath.get(c, set())
        return roots

    def module(
        self,
        dotted,
        roots,
        allow_fallback = True,
    ):
        """Files a dotted module name can load (module file or package __init__, plus the inits
        of the packages above it)."""
        if not dotted or dotted.startswith("."):
            return set()
        rel = dotted.replace(".", "/")
        cands = self.g.suffix.get(rel + ".py", []) + self.g.suffix.get(rel + "/__init__.py", [])
        hits = set()
        for c in cands:
            root = (
                c[: len(c) - len(rel + ".py")]
                if c.endswith(rel + ".py")
                else c[: len(c) - len(rel + "/__init__.py")]
            )
            root = root.rstrip("/")
            if root in roots:
                hits.add(c)
                hits.update(i for i in _pkg_inits(c, root) if i in self.g.fileset)
        # A namespace package (no __init__) as a directory: `import a.b` where a/b/ is a dir.
        if not hits:
            for d in self.g.dir_suffix.get(rel, []):
                root = d[: len(d) - len(rel)].rstrip("/")
                if root in roots:
                    hits.update(i for i in _pkg_inits(d + "/x", root) if i in self.g.fileset)
        if not hits and allow_fallback and dotted.split(".")[0] not in _STDLIB:
            for c in cands:
                hits.add(c)
        return hits

    def package_dir_modules(self, dotted, roots):
        """Every .py under the package `dotted` (for prefix / wildcard dynamic imports)."""
        rel = dotted.strip(".").replace(".", "/")
        out = set()
        for d in self.g.dir_suffix.get(rel, []):
            root = d[: len(d) - len(rel)].rstrip("/")
            if root in roots or not rel:
                out.update(p for p in self.g.dirs[d] if p.endswith(".py"))
        return out


def _const_str(node):
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _str_prefix(node):
    """Literal prefix of an f-string / 'a' + x / '%s' % x expression, else None."""
    if isinstance(node, ast.JoinedStr) and node.values:
        return _const_str(node.values[0])
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mod)):
        return _const_str(node.left) or _str_prefix(node.left)
    return None


def _is_dunder_name(node):
    return isinstance(node, ast.Name) and node.id in ("__name__", "__package__", "__spec__")


def _relative_base(path, level):
    parts = path.split("/")[:-1]
    if level - 1 > len(parts):
        return None
    return "/".join(parts[: len(parts) - (level - 1)])


def _scope_assigns(body_nodes):
    """name -> [value nodes] for assignments in these statements, not descending into nested
    functions / classes."""
    out = defaultdict(list)
    stack = list(body_nodes)
    while stack:
        n = stack.pop()
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    out[t.id].append(n.value)
        elif (
            isinstance(n, (ast.AnnAssign, ast.AugAssign))
            and isinstance(n.target, ast.Name)
            and n.value is not None
        ):
            out[n.target.id].append(n.value)
        elif isinstance(n, ast.NamedExpr) and isinstance(n.target, ast.Name):
            out[n.target.id].append(n.value)
        elif isinstance(n, (ast.For, ast.AsyncFor, ast.comprehension)) and isinstance(
            n.target, ast.Name
        ):
            out[n.target.id].append(
                ast.Call(func = ast.Name(id = "__iter_of__"), args = [n.iter], keywords = [])
            )
        stack.extend(ast.iter_child_nodes(n))
    return out


def _params(fn):
    a = fn.args
    names = [x.arg for x in a.posonlyargs + a.args + a.kwonlyargs]
    names += [x.arg for x in (a.vararg, a.kwarg) if x is not None]
    return set(names)


class _PyScanner(ast.NodeVisitor):
    def visit_Constant(self, node):  # NodeVisitor's default is a slow deprecation shim
        pass

    def __init__(
        self,
        g,
        res,
        path,
        tree,
        has_syspath = True,
    ):
        self.g, self.res, self.path, self.tree = g, res, path, tree
        self.module_assign = _scope_assigns(tree.body)
        self.fn_info = {}
        self.stack = []
        self.syspath = set()
        self.syspath_unresolved = False
        self.collect_only = True
        self.roots = res.roots_for(path)
        if has_syspath:
            self.visit(tree)  # pass 1: sys.path literals (need scopes)
        self.collect_only = False
        self.roots = res.roots_for(path, self.syspath)
        # A name that resolves under none of the roots may still come from a sys.path entry
        # this file (or its conftest chain) adds but that could not be evaluated: match it by
        # path suffix anywhere. Package code without such hacks never needs that.
        self.fallback = self.syspath_unresolved or any(
            c in res.conftest_unresolved for c in _conftests_above(g.fileset, path)
        )

    def _fn(self, fn):
        info = self.fn_info.get(id(fn))
        if info is None:
            body = fn.body if isinstance(fn.body, list) else [fn.body]
            info = self.fn_info[id(fn)] = (_params(fn), _scope_assigns(body))
        return info

    def _visit_fn(self, node):
        self.stack.append(node)
        self.generic_visit(node)
        self.stack.pop()

    visit_FunctionDef = visit_AsyncFunctionDef = visit_Lambda = _visit_fn

    def _lookup(self, name):
        """[(value node, scope depth)] a name can hold at this point, or None when unknown."""
        for i in range(len(self.stack) - 1, -1, -1):
            params, assigns = self._fn(self.stack[i])
            if name in assigns:
                return [(v, i) for v in assigns[name]]
            if name in params:
                return None
        if name in self.module_assign:
            return [(v, -1) for v in self.module_assign[name]]
        return None

    # -- tiny symbolic evaluator for repo paths -----------------------------------------------
    def _eval_dirs(
        self,
        node,
        depth = 0,
    ):
        """{(repo-relative path, anchored)} an expression can denote; anchored = derived from
        __file__ (else a cwd-relative literal). Empty set when unknown."""
        return self._ev(node, depth, len(self.stack))

    def _ev(self, node, depth, scope):
        if node is None or depth > 10:
            return set()
        if isinstance(node, ast.Name):
            if node.id == "__file__":
                return {(self.path, True)}
            saved = self.stack
            self.stack = saved[:scope]
            try:
                vals = self._lookup(node.id)
            finally:
                self.stack = saved
            if not vals:
                return set()
            out = set()
            for v, sc in vals:
                out |= self._ev(v, depth + 1, sc + 1)
            return out
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            s = node.value.strip().replace("\\", "/")
            while s.startswith("./"):
                s = s[2:]
            return {("" if s in (".", "") else s, False)}
        if isinstance(node, ast.Call):
            f = _dotted(node.func)
            if f == "__iter_of__":
                it = node.args[0]
                if isinstance(it, (ast.Tuple, ast.List, ast.Set)):
                    out = set()
                    for e in it.elts:
                        out |= self._ev(e, depth + 1, scope)
                    return out
                return set()
            if f in (
                "Path",
                "pathlib.Path",
                "PurePath",
                "pathlib.PurePath",
                "PosixPath",
                "PurePosixPath",
                "str",
                "os.fspath",
                "os.path.abspath",
                "os.path.realpath",
                "os.path.normpath",
                "Path.resolve",
            ):
                if not node.args:
                    return set()
                acc = self._ev(node.args[0], depth + 1, scope)
                for x in node.args[1:]:
                    nxt = self._ev(x, depth + 1, scope)
                    acc = {(_join(p, q), a) for p, a in acc for q, _ in nxt}
                return acc
            if f == "os.path.dirname" and node.args:
                return {(_parent(p), a) for p, a in self._ev(node.args[0], depth + 1, scope)}
            if f == "os.path.join" and node.args:
                acc = self._ev(node.args[0], depth + 1, scope)
                for x in node.args[1:]:
                    nxt = self._ev(x, depth + 1, scope)
                    acc = {(_join(p, q), a) for p, a in acc for q, _ in nxt}
                return acc
            if isinstance(node.func, ast.Attribute) and node.func.attr in (
                "resolve",
                "absolute",
                "expanduser",
            ):
                return self._ev(node.func.value, depth + 1, scope)
            if isinstance(node.func, ast.Attribute) and node.func.attr == "joinpath":
                acc = self._ev(node.func.value, depth + 1, scope)
                for x in node.args:
                    nxt = self._ev(x, depth + 1, scope)
                    acc = {(_join(p, q), a) for p, a in acc for q, _ in nxt}
                return acc
            return set()
        if isinstance(node, ast.Attribute) and node.attr == "parent":
            return {(_parent(p), a) for p, a in self._ev(node.value, depth + 1, scope)}
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "parents"
        ):
            k = (
                node.slice.value
                if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, int)
                else None
            )
            if k is None:
                return set()
            out = set()
            for p, a in self._ev(node.value.value, depth + 1, scope):
                for _ in range(k + 1):
                    p = _parent(p)
                out.add((p, a))
            return out
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            left, right = (
                self._ev(node.left, depth + 1, scope),
                self._ev(node.right, depth + 1, scope),
            )
            return {(_join(p, q), a) for p, a in left for q, _ in right}
        return set()

    # -- visitors ----------------------------------------------------------------------------
    def _mod(
        self,
        dotted,
        kind = "import",
    ):
        for f in self.res.module(dotted, self.roots, allow_fallback = self.fallback):
            self.g.add(self.path, f, kind)

    def _rel_from(self, level, module):
        base = _relative_base(self.path, level)
        if base is None:
            return None
        return (base + "/" + module.replace(".", "/")) if module else base

    def _add_relpath(
        self,
        relpath,
        kind = "import",
    ):
        for cand in (relpath + ".py", relpath + "/__init__.py"):
            if cand in self.g.fileset:
                self.g.add(self.path, cand, kind)

    def visit_Import(self, node):
        if not self.collect_only:
            for a in node.names:
                self._mod(a.name)

    def visit_ImportFrom(self, node):
        if self.collect_only:
            return
        if node.level:
            base = self._rel_from(node.level, node.module or "")
            if base is None:
                return
            pb = _relative_base(self.path, node.level)
            # the package inits between the importer's package and the target
            self._add_relpath(pb)
            if node.module:
                parts = node.module.split(".")
                for k in range(1, len(parts) + 1):
                    self._add_relpath((pb + "/" if pb else "") + "/".join(parts[:k]))
            for a in node.names:
                if a.name != "*":
                    self._add_relpath((base + "/" if base else "") + a.name)
        else:
            self._mod(node.module)
            for a in node.names:
                if a.name != "*":
                    for f in self.res.module(
                        node.module + "." + a.name, self.roots, allow_fallback = False
                    ):
                        self.g.add(self.path, f, "import")

    def visit_Call(self, node):
        f = _dotted(node.func)
        tail = f.rsplit(".", 1)[-1] if f else ""
        if self.collect_only:
            if (
                tail in ("insert", "append", "extend")
                and f.endswith("sys.path." + tail)
                and node.args
            ):
                got = self._eval_dirs(node.args[-1])
                for d, _a in got:
                    self.syspath.add(d.strip("/"))
                if not got:
                    self.syspath_unresolved = True
            self.generic_visit(node)
            return
        if tail in _PY_LOADER_CALLS:
            self.g.py_loaders.add(self.path)
        if tail in _RUNNER_CALLS or tail in _PY_LOADER_CALLS:
            self.g.runners.add(self.path)
        if (
            f in ("os.path.join", "Path", "pathlib.Path", "PurePath", "PurePosixPath")
            and len(node.args) > 1
        ):
            self._path_expr(node, list(node.args))
        elif tail == "joinpath" and isinstance(node.func, ast.Attribute):
            self._path_expr(node, [node.func.value, *node.args])
        if tail in ("import_module", "__import__", "find_spec", "run_module") or f in (
            "pytest.importorskip",
            "importlib.util.find_spec",
        ):
            arg = (
                node.args[0]
                if node.args
                else next((k.value for k in node.keywords if k.arg in ("name", "modname")), None)
            )
            pkg = (
                node.args[1]
                if len(node.args) > 1 and tail == "import_module"
                else next((k.value for k in node.keywords if k.arg == "package"), None)
            )
            lit = _const_str(arg) if arg is not None else None
            if lit is not None:
                if lit.startswith("."):
                    lvl = len(lit) - len(lit.lstrip("."))
                    base = self._rel_from(lvl, lit.lstrip("."))
                    if base is not None:
                        self._add_relpath(base, "dynamic")
                    if pkg is not None and not _is_dunder_name(pkg):
                        self._opaque(node)  # relative to some other package
                else:
                    self._mod(lit, "dynamic")
            elif arg is not None:
                prefix = _str_prefix(arg)
                if (
                    isinstance(arg, ast.JoinedStr)
                    and arg.values
                    and isinstance(arg.values[0], ast.FormattedValue)
                    and _is_dunder_name(arg.values[0].value)
                ):
                    self._own_package(node, "dynamic")
                elif prefix and prefix.strip("."):
                    top = prefix.strip(".")
                    pkgname = (
                        top
                        if prefix.endswith(".")
                        else top.rsplit(".", 1)[0]
                        if "." in top
                        else top
                    )
                    mods = self.res.package_dir_modules(pkgname, self.roots)
                    for m in mods:
                        self.g.add(self.path, m, "dynamic")
                    if not mods and top.split(".")[0] not in _STDLIB | _KNOWN_EXTERNAL:
                        self._opaque(node)
                elif prefix == "." or (pkg is not None and _is_dunder_name(pkg)):
                    self._own_package(node, "dynamic")
                else:
                    self._opaque(node)
        elif tail in _WALK_ATTRS and isinstance(node.func, ast.Attribute) and tail != "walk":
            pat = _const_str(node.args[0]) if node.args else None
            self._walker(node.func.value, pat)
        elif f and tuple(f.split(".")[-2:]) in _WALK_FUNCS and node.args:
            self._walker(node.args[0])
        self.generic_visit(node)

    def _own_package(self, node, kind):
        """Modules directly in the importer's package (only if it is a real package; never its
        test files, which no plugin loader imports)."""
        d = _parent(self.path)
        if (d + "/__init__.py" if d else "__init__.py") not in self.g.fileset:
            return
        for p in self.g.dirs.get(d, []) if d else self.g.files:
            rest = p[len(d) + 1 :] if d else p
            if is_test_file(p) or PurePosixPath(p).name == "conftest.py":
                continue
            if p.endswith(".py") and (
                "/" not in rest or rest.count("/") == 1 and rest.endswith("/__init__.py")
            ):
                self.g.add(self.path, p, kind)

    def _opaque(self, node):
        self.g.opaque[self.path].append(getattr(node, "lineno", 0))
        if not _in_test_tree(self.path):
            # a test's dynamic import names a module it spells out as a literal elsewhere (literal
            # edges cover that); a package module's may load a sibling plugin
            self._own_package(node, "opaque")

    def visit_BinOp(self, node):
        if not self.collect_only and isinstance(node.op, ast.Div):
            self._path_expr(node, _div_parts(node))
        self.generic_visit(node)

    def _path_expr(self, node, parts):
        """`ROOT / "unsloth" / "__init__.py"`, os.path.join(...), Path(a, b): the file it builds.
        Anchored (from __file__) -> that exact path; otherwise the trailing literal components
        as a path suffix."""
        for path, anchored in self._eval_dirs(node):
            if anchored and path in self.g.fileset:
                self.g.add(self.path, path, "literal")
        tail = []
        for x in reversed(parts):
            c = _const_str(x)
            if c is None:
                break
            tail.insert(0, c.replace("\\", "/").strip("/"))
        joined = "/".join(t for t in tail if t and t != ".")
        if joined and len(tail) >= 2:
            for f in _path_targets(self.g, joined):
                self.g.add(self.path, f, "literal")

    def _walker(
        self,
        target,
        pattern = None,
    ):
        for d, anchored in self._eval_dirs(target):
            if pattern and "/" in pattern.replace("\\", "/"):
                d = _join(d, pattern.replace("\\", "/"))
            d = re.split(r"[*?\[]", d)[0]
            d = (
                d.rstrip("/")
                if d.endswith("/") or d in self.g.dirset
                else _parent(d)
                if (d not in self.g.fileset and "*" in (pattern or "") and False)
                else d.rstrip("/")
            )
            if not anchored and (not d or d.startswith("/")):
                continue  # cwd-relative with no repo path in it: not a walk over the checkout
            name_pat = pattern.replace("\\", "/").rsplit("/", 1)[-1] if pattern else None
            keep = (
                (lambda p: fnmatch.fnmatchcase(p.rsplit("/", 1)[-1], name_pat))
                if name_pat and name_pat != "*"
                else (lambda p: True)
            )
            if d == "":
                for p in self.g.files:
                    if keep(p):
                        self.g.add(self.path, p, "walker")
            elif d in self.g.dirset:
                for p in self.g.dirs[d]:
                    if keep(p):
                        self.g.add(self.path, p, "walker")
            elif d in self.g.fileset:
                self.g.add(self.path, d, "walker")
            elif _parent(d) in self.g.dirset:
                # "dir/prefix" from a glob like dir/prefix*.py: the whole dir
                for p in self.g.dirs[_parent(d)]:
                    self.g.add(self.path, p, "walker")


_PY_LOADER_CALLS = frozenset(
    {
        "exec_module",
        "run_path",
        "run_module",
        "load_module",
        "import_module",
        "__import__",
        "SourceFileLoader",
        "spec_from_file_location",
        "exec",
        "compile",
    }
)
_RUNNER_CALLS = frozenset(
    {
        "run",
        "Popen",
        "call",
        "check_call",
        "check_output",
        "create_subprocess_exec",
        "create_subprocess_shell",
        "system",
        "execv",
        "execvp",
        "execve",
        "spawnv",
        "startfile",
    }
)

_KNOWN_EXTERNAL = frozenset(
    {
        "transformers",
        "torch",
        "peft",
        "trl",
        "vllm",
        "diffusers",
        "accelerate",
        "datasets",
        "huggingface_hub",
        "tokenizers",
        "bitsandbytes",
        "triton",
        "mlx",
        "mlx_lm",
        "mlx_vlm",
        "numpy",
        "safetensors",
        "xformers",
        "torchao",
    }
)


def _div_parts(node):
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return _div_parts(node.left) + [node.right]
    return [node]


def _dotted(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return ""


def _parent(p):
    return p.rsplit("/", 1)[0] if "/" in p else ""


def _join(a, b):
    b = b.replace("\\", "/")
    if b.startswith("/"):
        return b.lstrip("/")
    parts = [x for x in (a + "/" + b).split("/") if x and x != "."]
    out = []
    for x in parts:
        if x == "..":
            if out:
                out.pop()
        else:
            out.append(x)
    return "/".join(out)


_COMMAND_RX = re.compile(
    r"(sudo\s+)?(bash|sh|zsh|python[0-9.]*|py|pwsh|powershell(\.exe)?|cmd(\.exe)?|uv|pip|"
    r"pytest|node|npm|npx|source|\.|exec|env|\./|/bin/|&)(\s|$|/)",
    re.I,
)
_GENERIC_MAX = 8  # a bare basename shared by more files than this names none of them
_EXT_RX = re.compile(r"\.[A-Za-z0-9]{1,8}$")


def _path_targets(
    g,
    cand,
    allow_basename = True,
):
    """Tracked files a path-like string names: the longest matching path suffix, down to two
    components; a bare basename only when few files share it (so "__init__.py" names nothing)."""
    cand = cand.strip("/")
    if not cand:
        return ()
    if cand in g.fileset:
        return (cand,)
    parts = cand.split("/")
    if len(parts) > 1:
        # the whole literal must be a path suffix: "transformers/utils/x.py" names no repo file
        return g.suffix.get(cand, ())
    base = parts[-1]
    if allow_basename and (_EXT_RX.search(base) or base in ("Makefile", "Dockerfile")):
        hits = g.suffix.get(base, ())
        if 0 < len(hits) <= _GENERIC_MAX:
            return hits
    return ()


def _literal_targets(
    g,
    res,
    s,
    roots,
    allow_basename = True,
):
    """Files a string literal (or text token) can name."""
    out = set()
    s = s.strip()
    if not s or len(s) > 400:
        return out
    toks = [s] if not re.search(r"\s", s) else _TOKEN_RX.findall(s)
    for t in toks:
        t = t.replace("\\", "/").strip("\"'`,;:()[]{}")
        if not t or len(t) < 3:
            continue
        # "$ROOT/install.sh", "${HERE}/../x.sh", "{ROOT}/x.py" (an f-string field), "./x", "~/x"
        t2 = re.sub(
            r"^(\$?\{?[A-Za-z_][A-Za-z0-9_]*\}?/|\$\{?[A-Za-z_][A-Za-z0-9_]*\}?|\.{1,2}/|/|~/)+",
            "",
            t,
        )
        if "$" in t2 or "{" in t2:
            continue
        if "/" in t2 or _EXT_RX.search(t2):
            out.update(_path_targets(g, t2, allow_basename))
        # dotted module name ("core.inference.x", `-m unsloth_cli`)
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)+", t) and t.split(".")[
            -1
        ] not in (
            "py",
            "sh",
            "json",
            "toml",
            "yaml",
            "yml",
            "txt",
            "md",
            "ps1",
            "cfg",
            "ini",
            "lock",
        ):
            out.update(res.module(t, roots, allow_fallback = False))
    return out


def _code_imports(s):
    """Module names imported by Python source embedded in a string (`python -c "..."`)."""
    if "import" not in s:
        return []
    try:
        tree = ast.parse(textwrap.dedent(s))
    except (SyntaxError, ValueError):
        return []
    names = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            names += [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom) and n.module and not n.level:
            names.append(n.module)
            names += [n.module + "." + a.name for a in n.names if a.name != "*"]
    return names


def _docstring_ids(tree):
    out = set()
    stack = [tree]
    while stack:
        n = stack.pop()
        body = getattr(n, "body", None)
        if isinstance(body, list):
            if (
                isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                and body
            ):
                first = body[0]
                if isinstance(first, ast.Expr) and _const_str(first.value) is not None:
                    out.add(id(first.value))
            stack.extend(
                x
                for x in body
                if isinstance(
                    x,
                    (
                        ast.ClassDef,
                        ast.FunctionDef,
                        ast.AsyncFunctionDef,
                        ast.If,
                        ast.Try,
                        ast.With,
                        ast.For,
                        ast.While,
                    ),
                )
            )
            for attr in ("orelse", "finalbody", "handlers"):
                stack.extend(x for x in getattr(n, attr, ()) or () if hasattr(x, "body"))
    return out


def _parse(p, data):
    try:
        return ast.parse(data.decode("utf-8", errors = "replace"), filename = p)
    except (SyntaxError, ValueError):
        return None


def _scan_file(g, res, p, data):
    if not p.endswith(".py"):
        if len(data) <= _TEXT_SCAN_MAX:
            _scan_text(g, res, p, data)
        return
    t = _parse(p, data)
    if t is None:
        g.parse_errors.append(p)
        _scan_text(g, res, p, data)
        return
    sc = _PyScanner(g, res, p, t, has_syspath = b"sys.path" in data)
    sc.visit(t)
    docs = _docstring_ids(t)
    # A bare basename ("utils.py") in package code is almost always some other project's file;
    # package code names its own files through path expressions.
    testy = _in_test_tree(p)
    seen = set()
    for node in ast.walk(t):
        s = _const_str(node)
        if s is None or id(node) in docs or s in seen:
            continue
        seen.add(s)
        # Prose (a message, an f-string fragment with spaces) that is not a command line only
        # mentions a file: from a test that may be what it checks (read), from package code it
        # is nothing.
        prose = bool(re.search(r"\s", s.strip())) and not _COMMAND_RX.match(s.strip())
        if prose and not testy:
            pass
        else:
            for f in _literal_targets(g, res, s, sc.roots, allow_basename = testy):
                g.add(p, f, "mention" if prose else "literal")
        if "\n" in s or ";" in s or s.startswith(("import ", "from ")):
            for name in _code_imports(s):
                for f in res.module(name, sc.roots):
                    g.add(p, f, "import")


_POOL_CTX = None  # (graph, resolver, blobs by path) inherited by forked workers


def _scan_chunk(paths):
    import warnings

    g, res, blobs = _POOL_CTX
    g.deps, g.kinds = defaultdict(set), defaultdict(set)
    g.opaque, g.py_loaders, g.runners, g.parse_errors = defaultdict(list), set(), set(), []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for p in paths:
            _scan_file(g, res, p, blobs[p])
    return (
        {k: sorted(v) for k, v in g.kinds.items()},
        dict(g.opaque),
        sorted(g.py_loaders),
        sorted(g.runners),
        list(g.parse_errors),
    )


def build_graph(
    repo,
    rev,
    phantom = (),
    workers = None,
):
    """Graph of `rev`. `phantom`: paths absent at rev (deleted by the diff) that may still be named.
    Files are scanned in forked workers where the platform allows; the result is identical."""
    global _POOL_CTX
    import warnings

    tree = _ls_tree(repo, rev)
    g = Graph(tree, phantom)
    res = _Resolver(g)
    scan = sorted(
        p
        for p in tree
        if p.endswith(".py")
        or p.endswith(_TEXT_SCAN_EXT)
        or PurePosixPath(p).name in ("Makefile", "Dockerfile")
    )
    raw = _cat_blobs(repo, [tree[p] for p in scan])
    blobs = {p: raw.get(tree[p], b"") for p in scan}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # SyntaxWarning noise from the target repo's sources
        # conftest.py files first: the sys.path entries they add apply to their whole tree
        for p in (x for x in scan if PurePosixPath(x).name == "conftest.py"):
            t = _parse(p, blobs[p])
            if t is not None:
                sc = _PyScanner(g, res, p, t, has_syspath = True)
                res.conftest_syspath[p] = set(sc.syspath)
                if sc.syspath_unresolved:
                    res.conftest_unresolved.add(p)
    workers = workers if workers is not None else min(16, os.cpu_count() or 1)
    chunks = [scan[i :: max(1, workers * 4)] for i in range(max(1, workers * 4))]
    results = None
    if workers > 1 and len(scan) > 200:
        try:
            import multiprocessing as mp
            ctx = mp.get_context("fork")
        except (ImportError, ValueError):
            ctx = None
        if ctx is not None:
            _POOL_CTX = (g, res, blobs)
            try:
                with ctx.Pool(workers) as pool:
                    results = pool.map(_scan_chunk, chunks)
            finally:
                _POOL_CTX = None
    if results is None:
        _POOL_CTX = (g, res, blobs)
        try:
            results = [_scan_chunk(c) for c in chunks]
        finally:
            _POOL_CTX = None
        g.deps, g.kinds = defaultdict(set), defaultdict(set)
        g.opaque, g.py_loaders, g.runners, g.parse_errors = defaultdict(list), set(), set(), []
    for kinds, opaque, loaders, runners, errors in results:
        for (src, dst), ks in kinds.items():
            g.deps[src].add(dst)
            g.kinds[(src, dst)].update(ks)
        for f, lines in opaque.items():
            g.opaque[f].extend(lines)
        g.py_loaders.update(loaders)
        g.runners.update(runners)
        g.parse_errors.extend(errors)
    for f in g.opaque:
        g.opaque[f].sort()
    g.parse_errors.sort()
    return g


_SHELLISH = (
    ".sh",
    ".bash",
    ".ps1",
    ".psm1",
    ".psd1",
    ".toml",
    ".yaml",
    ".yml",
    ".cfg",
    ".ini",
    ".txt",
    ".in",
    ".mk",
)


def _strip_comments(p, text):
    """Drop `#` comment lines / tails (and PowerShell <# #> blocks): a comment names a file, it
    does not run or read it."""
    if p.endswith((".ps1", ".psm1", ".psd1")):
        text = re.sub(r"<#.*?#>", " ", text, flags = re.S)
    if not p.endswith(_SHELLISH) and PurePosixPath(p).name not in ("Makefile", "Dockerfile"):
        return text
    out = []
    for line in text.splitlines():
        st = line.lstrip()
        if st.startswith("#") and not st.startswith("#!"):
            continue
        out.append(re.sub(r"\s#\s.*$", "", line))
    return "\n".join(out)


def _scan_text(g, res, p, data):
    text = _strip_comments(p, data.decode("utf-8", errors = "replace"))
    roots = res.roots_for(p)
    seen = set()
    for tok in _TOKEN_RX.findall(text):
        if tok in seen or len(tok) < 3:
            continue
        seen.add(tok)
        for f in _literal_targets(g, res, tok, roots):
            g.add(p, f, "literal")


# ---------------------------------------------------------------------------------------------
# Selection


def conftest_chain(g, path):
    parts = path.split("/")[:-1]
    out = []
    for k in range(len(parts) + 1):
        c = "/".join(parts[:k] + ["conftest.py"])
        if c in g.fileset and c != path:
            out.append(c)
    return out


def conftest_roots(files):
    """Topmost conftest.py of each test tree (no conftest.py in any ancestor dir)."""
    fs = set(files)
    out = set()
    for f in fs:
        if PurePosixPath(f).name != "conftest.py":
            continue
        parts = f.split("/")[:-1]
        if not any("/".join(parts[:k] + ["conftest.py"]) in fs for k in range(len(parts))):
            out.add(f)
    return out


def _table_hits(path):
    hits = []
    for rule in EXPLICIT_MAP:
        if not _glob(path, rule["glob"]):
            continue
        if any(_glob(path, x) for x in rule.get("exclude", ())):
            continue
        if rule.get("non_python_only") and (path.endswith(".py") or is_test_file(path)):
            continue
        if rule.get("non_python_only") and "/tests/" in "/" + path:
            continue
        hits.append(rule)
    return hits


def _in_scope(path, scope):
    return not scope or any(path == s or path.startswith(s.rstrip("/") + "/") for s in scope)


def select(
    repo,
    base,
    head,
    scope = None,
    full = False,
    graph = None,
):
    """Selection dict (see module docstring). Deterministic for identical inputs."""
    scope = sorted(set(scope or ()))
    changes = changed_files(repo, base, head)
    changed = sorted({p for _, p in changes})
    deleted = sorted({p for s, p in changes if s == "D"} - {p for s, p in changes if s != "D"})
    out = {
        "version": VERSION,
        "base": base,
        "head": head,
        "scope": scope,
        "changed": changed,
        "mode": "TARGETED",
        "reasons": [],
        "tests": [],
        "suites": [],
        "by_changed": {},
        "total_tests": 0,
        "selected_tests": 0,
        "fraction": 0.0,
    }
    head_files = set(g.files) - set(deleted) if graph is not None else set(_ls_tree(repo, head))
    all_tests = sorted(p for p in head_files if is_test_file(p) and _in_scope(p, scope))
    all_suites = [x for x in PLAYWRIGHT_SUITES if x in head_files]
    out["total_tests"] = len(all_tests)
    if full or os.environ.get(FULL_ENV) == "1":
        out.update(mode = "FULL", reasons = ["override: --full-tests / %s=1" % FULL_ENV])
        return _finish(out, all_tests, all_suites, head_files)
    if not changed:
        out.update(mode = "NONE", reasons = ["no changed files"])
        return _finish(out, [], [], head_files)

    croots = conftest_roots(head_files | set(deleted))
    infra = [r for r in (infra_reason(p, croots) for p in changed) if r]
    if infra:
        out.update(mode = "FULL", reasons = infra)
        return _finish(out, all_tests, all_suites, head_files)
    g = graph or build_graph(repo, head, phantom = deleted)

    rx, rr = g.rdeps()
    # conftest chains: a test runs every conftest above it
    tests_all = sorted(p for p in head_files if is_test_file(p))
    for t in tests_all:
        for c in conftest_chain(g, t):
            rx[c].add(t)
        # pytest imports a test inside its package (every __init__.py up the chain) at collection
        d = _parent(t)
        while d and (d + "/__init__.py") in head_files:
            rx[d + "/__init__.py"].add(t)
            d = _parent(d)
    # A Playwright suite drives a Studio started from these entry points: their code is its code.
    anchors = {s: {s} for s in PLAYWRIGHT_SUITES if s in head_files}
    entries = {e for e in STUDIO_RUNTIME_ENTRIES if e in head_files}

    sel_tests, sel_suites = set(), set()
    reasons = []
    for c in changed:
        affected = affected_by(rx, rr, [c])
        t_hit = [p for p in affected if is_test_file(p) and p in head_files]
        s_hit = {s for s, a in anchors.items() if a & affected}
        if entries & affected and not is_test_file(c):
            s_hit.update(anchors)
        via = []
        for rule in _table_hits(c):
            via.append(rule)
            s_hit.update(x for x in rule.get("suites", ()) if x in head_files)
            for tg in rule.get("tests", ()):
                t_hit += [p for p in tests_all if _glob(p, tg)]
        if PurePosixPath(c).name == "conftest.py" and c in head_files:
            d = _parent(c)
            t_hit += [p for p in tests_all if p.startswith(d + "/" if d else "")]
        t_hit = sorted(set(t_hit))
        entry = {"tests": len(t_hit), "suites": sorted(s_hit)}
        if not t_hit and not s_hit and not via:
            inbound = (rx.get(c, set()) | rr.get(c, set())) - {c}
            if is_docs(c):
                entry["note"] = "docs"
            elif c.endswith(".py") and not inbound and g.opaque:
                sites = sorted(f"{f}:{ln}" for f, lns in g.opaque.items() for ln in lns)
                reasons.append(
                    f"{c}: no static or literal importer, and {len(sites)} non-literal dynamic "
                    f"imports could load it (e.g. {', '.join(sites[:3])})"
                )
            else:
                reasons.append(f"{c}: maps to no test (unknown file / directory)")
        out["by_changed"][c] = entry
        sel_tests.update(t_hit)
        sel_suites.update(s_hit)
    if reasons:
        out.update(mode = "FULL", reasons = reasons)
        return _finish(out, all_tests, all_suites, head_files)
    non_docs = [c for c in changed if not is_docs(c)]
    if not sel_tests and not sel_suites:
        if non_docs and not all(any(r.get("mapped") for r in _table_hits(c)) for c in non_docs):
            out.update(mode = "FULL", reasons = ["selection is empty for a non-docs change"])
            return _finish(out, all_tests, all_suites, head_files)
        out.update(
            mode = "NONE",
            reasons = [
                "docs only" if not non_docs else "only files the map says no test or suite covers"
            ],
        )
        return _finish(out, [], [], head_files)
    scoped = sorted(p for p in sel_tests if _in_scope(p, scope))
    out["reasons"] = [
        f"{len(sel_tests)} test files reach the change"
        + (f" ({len(scoped)} in scope)" if scope else "")
    ]
    return _finish(out, scoped, sorted(sel_suites), head_files)


def affected_by(rx, rr, changed):
    """Files whose behaviour a change to `changed` can alter: the changed files, every file that
    reads one of them, and everything that runs (transitively) any affected file."""
    seen = set(changed)
    for c in changed:
        seen |= rr.get(c, set())
    q = deque(sorted(seen))
    while q:
        x = q.popleft()
        for y in rx.get(x, ()):
            if y not in seen:
                seen.add(y)
                q.append(y)
    return seen


def _finish(out, tests, suites, head_files):
    out["tests"] = sorted(set(tests))
    out["suites"] = sorted(set(suites))
    out["selected_tests"] = len(out["tests"])
    out["fraction"] = (
        round(len(out["tests"]) / out["total_tests"], 4) if out["total_tests"] else 0.0
    )
    out["by_changed"] = {k: out["by_changed"][k] for k in sorted(out["by_changed"])}
    return out


def select_playwright(
    repo,
    base,
    head,
    full = False,
    graph = None,
):
    """{mode, suites, reasons}: which upstream_ci Playwright scripts the diff can affect."""
    s = select(repo, base, head, full = full, graph = graph)
    return {
        "mode": s["mode"],
        "suites": [PurePosixPath(x).name for x in s["suites"]],
        "reasons": s["reasons"],
        "changed": s["changed"],
    }


def dumps(sel):
    return json.dumps(sel, indent = 1, sort_keys = True, ensure_ascii = True) + "\n"


def main(argv = None):
    p = argparse.ArgumentParser(description = "Deterministic targeted test selection for base..head")
    p.add_argument("--repo", required = True)
    p.add_argument("--base", required = True)
    p.add_argument("--head", required = True)
    p.add_argument("--scope", action = "append", help = "only report tests under this dir (repeatable)")
    p.add_argument("--full-tests", action = "store_true", help = "select everything (old behaviour)")
    p.add_argument("--json", action = "store_true")
    a = p.parse_args(argv)
    sel = select(a.repo, a.base, a.head, scope = a.scope, full = a.full_tests)
    if a.json:
        sys.stdout.write(dumps(sel))
        return 0
    print(
        f"MODE {sel['mode']}  {sel['selected_tests']}/{sel['total_tests']} test files "
        f"({sel['fraction']:.1%})  suites: {', '.join(PurePosixPath(s).name for s in sel['suites']) or '-'}"
    )
    for r in sel["reasons"]:
        print(f"  reason: {r}")
    for t in sel["tests"]:
        print(t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
