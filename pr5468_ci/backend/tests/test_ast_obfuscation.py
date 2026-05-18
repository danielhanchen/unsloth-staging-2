"""
Obfuscation-resistance sweep for the AST gate.

Tests here document the gate's actual scope: where it succeeds, and
where it falls back on the OS sandbox. Each gap is annotated because
the OS sandbox is the intended backstop — and on hosts where the
sandbox falls open (the P1 finding), these gaps reach the host.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from core.inference import tools  # noqa: E402


def _check(code: str) -> str | None:
    return tools._check_code_safety(code)


# ---------------------------------------------------------------------------
# Direct shell-escape variants WITH a known-blocked command word
# ---------------------------------------------------------------------------

class TestDirectShellEscapesWithBlockedWord:
    """The static gate flags os.system/subprocess.* etc. when the argument
    contains a known-blocked command (sudo, rm, curl, ssh, passwd, eval,
    source, …) at a shell boundary. These should always be caught."""

    @pytest.mark.parametrize(
        "code",
        [
            "import os; os.system('sudo whoami')",
            "import os as o; o.system('sudo whoami')",
            "from os import system; system('sudo whoami')",
            "from os import system as s; s('sudo whoami')",
            "import os; os.popen('sudo whoami')",
            "import subprocess; subprocess.call(['sudo','x'])",
            "import subprocess; subprocess.run(['sudo','x'])",
            "import subprocess; subprocess.check_output(['sudo','x'])",
            "import subprocess; subprocess.Popen(['sudo','x'])",
            "from subprocess import run; run(['sudo','x'])",
            "from subprocess import Popen as P; P(['sudo','x'])",
            # Path prefix is allowed by the regex
            "import os; os.system('/usr/bin/sudo whoami')",
            # subprocess via opaque **kwargs
            "import subprocess; kw={'args':['x']}; subprocess.run(**kw)",
        ],
    )
    def test_blocked(self, code):
        msg = _check(code)
        assert msg is not None, code
        assert "unsafe" in msg, (code, msg)


# ---------------------------------------------------------------------------
# Shell-exec calls with DYNAMIC (non-literal) args
# ---------------------------------------------------------------------------

class TestDynamicShellExec:
    @pytest.mark.parametrize(
        "code",
        [
            # variable as command
            "import os; cmd='id'; os.system(cmd)",
            # f-string
            "import os; x='i'; os.system(f'{x}d')",
            # concatenation
            "import os; os.system('i' + 'd')",
            # str.format
            "import os; os.system('{}'.format('id'))",
            # chr building
            "import os; os.system(chr(105)+chr(100))",
            # subprocess shell=True with variable
            "import subprocess; c='id'; subprocess.run(c, shell=True)",
        ],
    )
    def test_dynamic_args_flagged(self, code):
        msg = _check(code)
        assert msg is not None, code
        assert "dynamic" in msg or "unsafe" in msg, (code, msg)


# ---------------------------------------------------------------------------
# Documented gaps the OS sandbox is supposed to backstop
# ---------------------------------------------------------------------------

class TestKnownGapsInAstGate:
    """These all pass the AST gate. Each is documented as a known gap the
    OS sandbox layer (Seatbelt / bwrap) is expected to enforce. When the
    sandbox falls open (P1 in the review), these leak."""

    @pytest.mark.parametrize(
        "code,reason",
        [
            # Literal commands not in blocklist
            ("import os; os.system('id')", "'id' not in blocklist"),
            ("import os; os.system('whoami')", "'whoami' not in blocklist"),
            ("import os; os.system('cat /home/u/secret.txt')",
             "'cat' not in blocklist; path doesn't match passwd/shadow"),
            ("import os; os.system('find /home -name *.key')",
             "'find' not in blocklist"),
            ("import os; os.system('ls /home/u/.aws')",
             "'ls' not in blocklist"),
            # Bare-name HF upload (no receiver, no module prefix)
            ("from huggingface_hub import upload_file\nupload_file('x','y','a/b')",
             "bare-name HF upload not caught"),
            ("from huggingface_hub import upload_folder\nupload_folder('x','a/b')",
             "bare-name HF upload not caught"),
            # eval / exec body is a constant string — visitor recurses into the
            # Call node, but the body is just a string Constant, not parsed
            ("eval(\"__import__('os').system('id')\")",
             "eval body is a string Constant"),
            ("exec(\"import os; os.system('id')\")",
             "exec body is a string Constant"),
            ("__import__('os').system('id')",
             "__import__ chained call doesn't resolve to os.system fq"),
            # Dynamic file path (concat is BinOp, not Constant)
            ("p = '/etc/' + 'shadow'; open(p)",
             "dynamic path concatenation not resolved"),
            # getattr lookup
            ("import os; getattr(os, 'sys' + 'tem')('id')",
             "getattr access not modeled"),
            # base64 decode + exec
            ("import base64\nexec(base64.b64decode('cHJpbnQoMSk=').decode())",
             "exec of decoded content"),
        ],
    )
    def test_gap(self, code, reason):
        msg = _check(code)
        # Documenting the gap. If a future PR closes it, this test will
        # start passing the "blocked" assertion; for now we record it.
        if msg is None:
            pytest.skip(f"AST gap (sandbox-backed): {reason}")
        else:
            # Bonus: gate evolved to catch it.
            assert "unsafe" in msg


# ---------------------------------------------------------------------------
# Network policy specifics
# ---------------------------------------------------------------------------

class TestNetworkPolicy:
    @pytest.mark.parametrize(
        "code,phrase",
        [
            ("import requests; requests.get('http://169.254.169.254/')",
             "cloud-metadata host"),
            ("import requests; requests.get('http://metadata.google.internal/')",
             "cloud-metadata host"),
            ("import requests; requests.get('https://attacker.example.com/')",
             "not in sandbox allowlist"),
            ("import urllib.request; urllib.request.urlopen('http://[fd00:ec2::254]/')",
             "cloud-metadata host"),
            ("import socket; s=socket.socket(); s.connect(('169.254.169.254', 80))",
             "cloud-metadata host"),
            ("import requests; requests.get('https://wikipedia.org@169.254.169.254/')",
             "cloud-metadata"),
            # IPv4 prefix metadata
            ("import requests; requests.get('http://169.254.170.2/')",
             "cloud-metadata host"),
            # Alibaba ECS
            ("import socket; s=socket.socket(); s.connect(('100.100.100.200', 80))",
             "cloud-metadata host"),
        ],
    )
    def test_blocked(self, code, phrase):
        msg = _check(code)
        assert msg is not None, code
        assert phrase in msg, (code, msg, phrase)

    @pytest.mark.parametrize(
        "code",
        [
            "import requests; requests.get('https://wikipedia.org/wiki/Foo')",
            "import requests; requests.get('https://huggingface.co/foo/bar')",
            "import requests; requests.get('https://github.com/foo/bar')",
            "import requests; requests.get('https://arxiv.org/abs/1234')",
            "import requests; requests.get('https://stackoverflow.com/q/123')",
            "import requests; requests.get('https://EN.WIKIPEDIA.ORG/wiki/X')",
            # subdomain of trusted suffix
            "import requests; requests.get('https://fr.wikipedia.org/wiki/Foo')",
            # trailing dot
            "import requests; requests.get('https://wikipedia.org./')",
            # explicit port
            "import requests; requests.get('https://wikipedia.org:443/')",
        ],
    )
    def test_trusted_allowed(self, code):
        assert _check(code) is None, code

    def test_dynamic_url_not_statically_blocked(self):
        """Static gate can't resolve runtime URLs. By design — bash side
        catches dynamic hosts via the blocklist."""
        assert _check("import requests; u='http://evil.example/'; requests.get(u)") is None


# ---------------------------------------------------------------------------
# Upload denylist (method-style)
# ---------------------------------------------------------------------------

class TestUploadDenylistMethodStyle:
    """The HF upload denylist catches method-style calls (`<receiver>.upload_*()`)
    regardless of receiver name. Bare-name from-imports are a known gap
    (see TestKnownGapsInAstGate)."""

    @pytest.mark.parametrize(
        "code",
        [
            "import requests; requests.post('https://huggingface.co/x', "
            "files={'f': open('a','rb')})",
            "import httpx; httpx.post('https://huggingface.co/x', files={'f': open('a','rb')})",
            "from huggingface_hub import HfApi; "
            "HfApi().upload_file(path_or_fileobj='x', path_in_repo='y', repo_id='a/b')",
            "import huggingface_hub; "
            "huggingface_hub.upload_folder(folder_path='.', repo_id='a/b')",
            "import huggingface_hub as hf; "
            "hf.upload_file(path_or_fileobj='x', path_in_repo='y', repo_id='a/b')",
            # via HfApi() chain — any receiver
            "x = object(); x.upload_file(path_or_fileobj='x', path_in_repo='y', repo_id='a/b')",
            # data=bytes
            "import requests; requests.put('https://huggingface.co/x', data=b'AB')",
            # data=open()
            "import requests; requests.post('https://huggingface.co/x', data=open('a','rb'))",
        ],
    )
    def test_blocked(self, code):
        msg = _check(code)
        assert msg is not None, code
        assert "upload" in msg.lower() or "unsafe" in msg, (code, msg)


# ---------------------------------------------------------------------------
# Loop / signal escapes
# ---------------------------------------------------------------------------

class TestSignalEscapes:
    @pytest.mark.parametrize(
        "code,phrase",
        [
            ("import signal; signal.signal(signal.SIGALRM, signal.SIG_IGN)",
             "signal_handler_override"),
            ("import signal; signal.alarm(0)",
             "alarm_manipulation"),
            ("import signal; signal.setitimer(signal.ITIMER_REAL, 0)",
             "timer_manipulation"),
            ("import signal; signal.pthread_sigmask(signal.SIG_BLOCK, [signal.SIGALRM])",
             "signal_mask"),
            ("while True:\n    try: pass\n    except TimeoutError: pass",
             "catches_TimeoutError_in_loop"),
            ("while True:\n    try: pass\n    except BaseException: pass",
             "catches_BaseException_in_loop"),
            ("while True:\n    try: pass\n    except: pass",
             "bare_except_in_loop"),
            # For-loop variant
            ("for _ in range(10):\n    try: pass\n    except TimeoutError: pass",
             "catches_TimeoutError_in_loop"),
            # Tuple in except
            ("while True:\n    try: pass\n    except (ValueError, TimeoutError): pass",
             "catches_TimeoutError_in_loop"),
        ],
    )
    def test_blocked(self, code, phrase):
        msg = _check(code)
        assert msg is not None, code
        assert phrase in msg or "unsafe" in msg, (code, msg, phrase)

    def test_except_exception_in_loop_not_flagged(self):
        """`except Exception` should NOT be flagged (false-positive guard)."""
        msg = _check("while True:\n    try: pass\n    except Exception: pass")
        assert msg is None


# ---------------------------------------------------------------------------
# Sensitive file reads (literal open() only — dynamic is a known gap)
# ---------------------------------------------------------------------------

class TestSensitiveFileReads:
    @pytest.mark.parametrize(
        "code",
        [
            "open('/etc/passwd')",
            "open('/etc/shadow')",
            "open('/etc/sudoers')",
            "open('/etc/ssh/sshd_config')",
            "open('/proc/self/environ')",
            "open('/proc/123/environ')",
            "open('/proc/self/task/1/environ')",
            "open('/proc/self/cmdline')",
        ],
    )
    def test_blocked(self, code):
        msg = _check(code)
        assert msg is not None, code


# ---------------------------------------------------------------------------
# Parse-error pass-through
# ---------------------------------------------------------------------------

class TestSyntaxError:
    def test_syntax_error_passes_through(self):
        """ast.parse fails → gate returns None so the child gets a real
        Python traceback instead of "unsafe code detected" confusion."""
        msg = _check("def x:\n    not valid python")
        assert msg is None
