"""
startup_banner.py: stdlib-only banner emission.

Cover color/no-color, IPv4/IPv6 binds, loopback vs wildcard, port-in-use
notice, sandbox-unavailable notice on Linux vs macOS.
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

import startup_banner as sbn  # noqa: E402


# ---------------------------------------------------------------------------
# Color detection
# ---------------------------------------------------------------------------

class TestColorDetection:
    def test_no_color_env_disables_color(self, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        assert sbn.stdout_supports_color() is False

    def test_force_color_env_enables_color(self, monkeypatch):
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setenv("FORCE_COLOR", "1")
        assert sbn.stdout_supports_color() is True

    def test_no_tty_disables_color_by_default(self, monkeypatch):
        # Both vars unset
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.delenv("FORCE_COLOR", raising=False)
        # stdout from pytest isn't a TTY, so this should be False
        assert sbn.stdout_supports_color() is False


# ---------------------------------------------------------------------------
# Sandbox unavailable notice
# ---------------------------------------------------------------------------

class TestSandboxUnavailableNotice:
    def test_includes_apt_install_hint_on_linux(self, monkeypatch):
        monkeypatch.setattr(sbn.sys, "platform", "linux")
        monkeypatch.setenv("NO_COLOR", "1")
        buf = io.StringIO()
        with redirect_stdout(buf):
            sbn.print_sandbox_unavailable_notice()
        out = buf.getvalue()
        assert "Sandbox unavailable" in out
        assert "apt install bubblewrap" in out
        assert "dnf install bubblewrap" in out
        assert "pacman -S bubblewrap" in out

    def test_no_install_hints_on_mac(self, monkeypatch):
        monkeypatch.setattr(sbn.sys, "platform", "darwin")
        monkeypatch.setenv("NO_COLOR", "1")
        buf = io.StringIO()
        with redirect_stdout(buf):
            sbn.print_sandbox_unavailable_notice()
        out = buf.getvalue()
        assert "Sandbox unavailable" in out
        assert "bubblewrap" not in out
        assert "apt install" not in out

    def test_no_install_hints_on_windows(self, monkeypatch):
        monkeypatch.setattr(sbn.sys, "platform", "win32")
        monkeypatch.setenv("NO_COLOR", "1")
        buf = io.StringIO()
        with redirect_stdout(buf):
            sbn.print_sandbox_unavailable_notice()
        out = buf.getvalue()
        assert "Sandbox unavailable" in out
        assert "bubblewrap" not in out


# ---------------------------------------------------------------------------
# Access banner — IPv4 / IPv6, loopback / wildcard
# ---------------------------------------------------------------------------

class TestAccessBanner:
    def test_loopback_ipv4(self, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        buf = io.StringIO()
        with redirect_stdout(buf):
            sbn.print_studio_access_banner(
                port=8888, bind_host="127.0.0.1", display_host="127.0.0.1",
            )
        out = buf.getvalue()
        assert "Unsloth Studio is running" in out
        assert "http://127.0.0.1:8888" in out
        assert "localhost:8888" in out

    def test_loopback_ipv6(self, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        buf = io.StringIO()
        with redirect_stdout(buf):
            sbn.print_studio_access_banner(
                port=8888, bind_host="::1", display_host="::1",
            )
        out = buf.getvalue()
        assert "[::1]:8888" in out

    def test_wildcard_with_lan_ip(self, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        buf = io.StringIO()
        with redirect_stdout(buf):
            sbn.print_studio_access_banner(
                port=8000, bind_host="0.0.0.0", display_host="10.0.0.5",
            )
        out = buf.getvalue()
        # Loopback URL still shown for local access
        assert "127.0.0.1:8000" in out
        # External URL shown for LAN access
        assert "10.0.0.5:8000" in out

    def test_ipv6_wildcard(self, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        buf = io.StringIO()
        with redirect_stdout(buf):
            sbn.print_studio_access_banner(
                port=8000, bind_host="::", display_host="2001:db8::1",
            )
        out = buf.getvalue()
        assert "[::1]:8000" in out
        # IPv6 external address bracketed
        assert "[2001:db8::1]:8000" in out

    def test_api_health_paths(self, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        buf = io.StringIO()
        with redirect_stdout(buf):
            sbn.print_studio_access_banner(
                port=8888, bind_host="127.0.0.1", display_host="127.0.0.1",
            )
        out = buf.getvalue()
        assert "/api" in out
        assert "/api/health" in out


# ---------------------------------------------------------------------------
# Port-in-use notice
# ---------------------------------------------------------------------------

class TestPortInUseNotice:
    def test_basic_message(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            sbn.print_port_in_use_notice(8888, 8889)
        out = buf.getvalue()
        assert "8888" in out
        assert "8889" in out


# ---------------------------------------------------------------------------
# ANSI codes appear when colored
# ---------------------------------------------------------------------------

class TestAnsiCodes:
    def test_force_color_emits_ansi(self, monkeypatch):
        monkeypatch.setenv("FORCE_COLOR", "1")
        monkeypatch.delenv("NO_COLOR", raising=False)
        buf = io.StringIO()
        with redirect_stdout(buf):
            sbn.print_studio_access_banner(
                port=8888, bind_host="127.0.0.1", display_host="127.0.0.1",
            )
        out = buf.getvalue()
        # ANSI CSI sequences start with ESC[
        assert "\x1b[" in out

    def test_no_color_strips_ansi(self, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        monkeypatch.delenv("FORCE_COLOR", raising=False)
        buf = io.StringIO()
        with redirect_stdout(buf):
            sbn.print_studio_access_banner(
                port=8888, bind_host="127.0.0.1", display_host="127.0.0.1",
            )
        out = buf.getvalue()
        assert "\x1b[" not in out
