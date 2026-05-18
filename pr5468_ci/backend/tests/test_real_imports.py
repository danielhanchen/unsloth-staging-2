"""
Real-world Python imports must still work through the wrapper.

The whole point of the LLM tool layer is that user code runs; the new
sandbox layer must not regress legitimate workloads. We probe a range
of common imports and behaviors.

Note: AST gate may flag `socket.create_connection(...)` and similar
network primitives even when used legitimately. We assert what the
gate does, not what we wish it did.
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from core.inference import tools  # noqa: E402


@pytest.fixture
def session(tmp_path, monkeypatch):
    sid = f"_imp_{uuid.uuid4().hex[:8]}"
    monkeypatch.setitem(tools._workdirs, sid, str(tmp_path))
    return sid


# ---------------------------------------------------------------------------
# Stdlib modules — all common ones
# ---------------------------------------------------------------------------

class TestStdlibImports:
    @pytest.mark.parametrize(
        "module",
        [
            "os", "sys", "json", "math", "re", "time", "datetime",
            "collections", "itertools", "functools", "pathlib",
            "tempfile", "io", "csv", "html", "hashlib", "hmac",
            "base64", "binascii", "struct", "array",
            "threading", "queue", "concurrent.futures",
            "ssl", "ipaddress", "urllib.parse",
            "asyncio", "contextlib", "weakref",
            "logging", "warnings", "traceback",
            "ast", "tokenize", "dis",
            "pickle", "shelve", "sqlite3",
            "gzip", "tarfile", "zipfile", "bz2", "lzma",
            "email", "calendar", "uuid", "secrets",
            "argparse",
        ],
    )
    def test_stdlib_import(self, session, module):
        out = tools._python_exec(
            f"import {module}; print('OK', {module}.__name__)",
            session_id=session, timeout=10,
        )
        assert "OK" in out, out


# ---------------------------------------------------------------------------
# Common patterns that should work
# ---------------------------------------------------------------------------

class TestCommonPatterns:
    def test_class_inheritance(self, session):
        code = (
            "class A:\n"
            "    def f(self): return 1\n"
            "class B(A):\n"
            "    def f(self): return super().f() + 1\n"
            "print(B().f())\n"
        )
        out = tools._python_exec(code, session_id=session, timeout=10)
        assert "2" in out

    def test_dataclass(self, session):
        code = (
            "from dataclasses import dataclass\n"
            "@dataclass\n"
            "class Point:\n"
            "    x: int\n"
            "    y: int\n"
            "p = Point(3, 4)\n"
            "print(p.x + p.y)\n"
        )
        out = tools._python_exec(code, session_id=session, timeout=10)
        assert "7" in out

    def test_pathlib_glob(self, session, tmp_path):
        sid = session
        (tmp_path / "a.txt").write_text("1")
        (tmp_path / "b.txt").write_text("2")
        code = (
            "from pathlib import Path\n"
            "for p in sorted(Path('.').glob('*.txt')):\n"
            "    print(p.name)\n"
        )
        out = tools._python_exec(code, session_id=sid, timeout=10)
        assert "a.txt" in out and "b.txt" in out

    def test_json_roundtrip(self, session):
        code = (
            "import json\n"
            "d = {'k': [1, 2, 3], 'nested': {'a': True}}\n"
            "s = json.dumps(d)\n"
            "back = json.loads(s)\n"
            "assert back == d\n"
            "print('JSON_OK')\n"
        )
        out = tools._python_exec(code, session_id=session, timeout=10)
        assert "JSON_OK" in out

    def test_async_io(self, session):
        code = (
            "import asyncio\n"
            "async def go():\n"
            "    await asyncio.sleep(0.05)\n"
            "    return 42\n"
            "print(asyncio.run(go()))\n"
        )
        out = tools._python_exec(code, session_id=session, timeout=10)
        assert "42" in out

    def test_threading_no_deadlock_at_exit(self, session):
        code = (
            "import threading\n"
            "result = []\n"
            "def worker(i): result.append(i*i)\n"
            "threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]\n"
            "for t in threads: t.start()\n"
            "for t in threads: t.join()\n"
            "print(sum(result))\n"
        )
        out = tools._python_exec(code, session_id=session, timeout=15)
        # 0+1+4+9+16 = 30
        assert "30" in out

    def test_subprocess_run_safe_command(self, session):
        """subprocess.run with a literal safe argv must NOT be blocked
        by the AST gate."""
        code = (
            "import subprocess\n"
            "r = subprocess.run(['echo', 'hello-subproc'], "
            "capture_output=True, text=True)\n"
            "print(r.stdout.strip())\n"
        )
        out = tools._python_exec(code, session_id=session, timeout=10)
        # The AST gate's shell-escape detector treats subprocess.run() with
        # a non-literal-False shell= as potentially dangerous, but only
        # flags if args are dynamic OR if blocked words appear in args.
        # ['echo', 'hello-subproc'] is all literal + no blocked words → OK.
        assert "hello-subproc" in out, out


# ---------------------------------------------------------------------------
# Numerical / scientific (only if installed; we ship a minimal venv)
# ---------------------------------------------------------------------------

class TestOptionalScientific:
    """If numpy / pandas / etc. are installed in the venv, they should
    still work; if not, the test is a no-op."""

    @pytest.mark.parametrize("module", ["numpy", "pandas", "torch"])
    def test_optional_import(self, session, module):
        try:
            __import__(module)
        except ImportError:
            pytest.skip(f"{module} not installed in test venv")
        out = tools._python_exec(
            f"import {module}; print('OK', {module}.__name__)",
            session_id=session, timeout=15,
        )
        assert "OK" in out


# ---------------------------------------------------------------------------
# Unicode / non-ASCII handling
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Windows cp1252 console can't encode CJK without PYTHONIOENCODING; "
    "_build_safe_env sets PYTHONIOENCODING=utf-8 for the child but the "
    "Popen text-mode parent decode still uses the system default",
)
class TestUnicode:
    def test_unicode_print(self, session):
        out = tools._python_exec("print('héllo 世界 🦥')", session_id=session, timeout=10)
        assert "héllo" in out and "世界" in out

    def test_unicode_filename(self, session, tmp_path):
        code = (
            "from pathlib import Path\n"
            "Path('文件.txt').write_text('内容')\n"
            "print(Path('文件.txt').read_text())\n"
        )
        out = tools._python_exec(code, session_id=session, timeout=10)
        assert "内容" in out


# ---------------------------------------------------------------------------
# Exception handling and traceback formatting
# ---------------------------------------------------------------------------

class TestExceptionFormatting:
    def test_uncaught_exception(self, session):
        code = "raise ValueError('boom')\n"
        out = tools._python_exec(code, session_id=session, timeout=10)
        assert "ValueError" in out
        assert "boom" in out
        # Non-zero exit reflected in output
        assert "Exit code" in out

    def test_zero_division(self, session):
        out = tools._python_exec("print(1/0)", session_id=session, timeout=10)
        assert "ZeroDivisionError" in out

    def test_caught_exception_prints_normally(self, session):
        code = (
            "try:\n"
            "    1/0\n"
            "except ZeroDivisionError as e:\n"
            "    print('caught:', e)\n"
        )
        out = tools._python_exec(code, session_id=session, timeout=10)
        assert "caught:" in out


# ---------------------------------------------------------------------------
# Bash legitimate use cases
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Unix shell builtins (awk, tr, seq, printf) not available in cmd /c",
)
class TestBashUseCases:
    @pytest.mark.parametrize(
        "cmd,expect",
        [
            ("echo hello", "hello"),
            ("printf '%s\\n' world", "world"),
            ("date +%Y", "20"),  # year starts with 20
            ("uname -s", "Linux" if sys.platform == "linux" else "Darwin"),
            ("pwd", "/"),
            ("ls -la / | head -3", "total"),
            ("env | sort | head -1", "="),  # something with KEY=VAL
            ("python3 --version 2>&1", "Python"),
            ("which python3", "python"),
            ("cat /etc/hostname || echo NA", ""),  # might fail in sandbox; OK
            ("seq 1 5", "1"),
            ("echo a b c | awk '{print $2}'", "b"),
            ("echo hello | tr a-z A-Z", "HELLO"),
        ],
    )
    def test_bash_command(self, session, cmd, expect):
        out = tools._bash_exec(cmd, session_id=session, timeout=10)
        assert "Blocked" not in out, (cmd, out)
        if expect:
            assert expect in out, (cmd, out)
