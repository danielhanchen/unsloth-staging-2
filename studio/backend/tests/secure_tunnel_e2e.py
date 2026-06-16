#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""End-to-end Cloudflare quick-tunnel check using Studio's own cloudflare_tunnel.py.

What it proves, per OS/arch, with no torch / llama.cpp / GPU:
  1. _asset_name() picks the right cloudflared release for this platform.
  2. ensure_cloudflared() downloads + (on macOS) extracts that binary.
  3. start_studio_tunnel() mints a https://*.trycloudflare.com URL AND registers
     an edge connection (the same readiness gate Studio uses).
  4. Fetching that public URL routes back through Cloudflare to a local loopback
     server -- i.e. the tunnel actually carries traffic.

This is exactly the code path `unsloth studio --secure` runs; only the cloudflared
cache location is redirected to a temp dir so the harness needs no backend deps.
Exit 0 on success, non-zero otherwise.
"""
from __future__ import annotations

import http.server
import os
import secrets
import socket
import socketserver
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path


def _find_backend() -> Path:
    """Locate studio/backend (dir holding cloudflare_tunnel.py) from anywhere."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "cloudflare_tunnel.py").is_file():
            return parent
        cand = parent / "studio" / "backend"
        if (cand / "cloudflare_tunnel.py").is_file():
            return cand
    raise SystemExit("FATAL: could not locate studio/backend/cloudflare_tunnel.py")


BACKEND = _find_backend()
sys.path.insert(0, str(BACKEND))

import cloudflare_tunnel as ct  # noqa: E402

TOKEN = "unsloth-tunnel-" + secrets.token_hex(8)


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = TOKEN.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _fetch_token(url: str) -> "tuple[bool, str]":
    """Fetch `url` and report whether TOKEN came back.

    Primary path is a normal DNS-based HTTPS GET (what a browser does, and what
    GitHub runners can do). Fallback, for hosts whose resolver won't return the
    wildcard *.trycloudflare.com record, dials a routable Cloudflare edge IP
    directly with SNI/Host set to the tunnel host -- still a real round trip out
    to Cloudflare and back down the tunnel, just without the wildcard DNS lookup.
    """
    import ssl
    from urllib.parse import urlparse

    # 1) Normal DNS-based fetch.
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "unsloth-e2e"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            code = resp.status
            text = resp.read().decode("utf-8", "replace").strip()
        if code == 200 and text == TOKEN:
            return True, f"HTTP {code} (dns) body={text!r}"
        return False, f"HTTP {code} (dns) body={text!r}"
    except Exception as exc:  # noqa: BLE001
        dns_err = f"{type(exc).__name__}: {exc}"

    # 2) DNS-independent fallback: edge IP + SNI.
    host = urlparse(url).hostname or ""
    try:
        edge_ip = socket.gethostbyname("trycloudflare.com")
        ctx = ssl.create_default_context()
        raw = socket.create_connection((edge_ip, 443), timeout=15)
        tls = ctx.wrap_socket(raw, server_hostname=host)
        request = (
            f"GET / HTTP/1.1\r\nHost: {host}\r\nUser-Agent: unsloth-e2e\r\n"
            f"Accept: */*\r\nConnection: close\r\n\r\n"
        ).encode()
        tls.sendall(request)
        buf = b""
        while True:
            chunk = tls.recv(65536)
            if not chunk:
                break
            buf += chunk
        tls.close()
        # TOKEN is a unique random string, so a substring hit (past chunked
        # framing) is a reliable signal the body came from our origin.
        if TOKEN.encode() in buf:
            status_line = buf.split(b"\r\n", 1)[0].decode("latin1", "replace")
            return True, f"{status_line} (edge {edge_ip}, sni {host})"
        head = buf.split(b"\r\n\r\n", 1)[0].decode("latin1", "replace")[:120]
        return False, f"edge fetch missing token; head={head!r}"
    except Exception as exc:  # noqa: BLE001
        return False, f"dns_err={dns_err}; edge_err={type(exc).__name__}: {exc}"


def main() -> int:
    print(f"== platform: {sys.platform} / asset: {ct._asset_name()}", flush=True)

    # Redirect the cloudflared cache to a temp dir so we don't need the backend's
    # utils.paths (structlog etc.). The download + tunnel logic stays the real thing.
    cache_dir = Path(tempfile.mkdtemp(prefix="cf_e2e_"))
    bin_name = "cloudflared.exe" if sys.platform == "win32" else "cloudflared"
    ct._cache_path = lambda: cache_dir / bin_name  # type: ignore[assignment]

    binary = ct.ensure_cloudflared()
    print(f"== cloudflared binary: {binary}", flush=True)
    if not binary:
        print("FAIL: could not obtain cloudflared for this platform", flush=True)
        return 1

    port = _free_port()
    httpd = socketserver.TCPServer(("127.0.0.1", port), _Handler)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    print(f"== local origin: http://127.0.0.1:{port}", flush=True)

    url = None
    try:
        url = ct.start_studio_tunnel(port, timeout=90)
        print(f"== tunnel url: {url}", flush=True)
        if not url or "trycloudflare.com" not in url:
            print("FAIL: tunnel did not mint a trycloudflare URL", flush=True)
            return 1

        # Edge propagation can lag the 'registered' marker; retry the public fetch.
        last = ""
        for attempt in range(12):
            ok, last = _fetch_token(url)
            if ok:
                print(f"== public fetch (attempt {attempt + 1}): {last}", flush=True)
                print("PASS: public trycloudflare URL routed to the local origin", flush=True)
                return 0
            time.sleep(4)
        print(f"== public fetch: {last}", flush=True)
        print("FAIL: public URL did not return the expected token", flush=True)
        return 1
    finally:
        try:
            ct.stop_studio_tunnel()
        except Exception:
            pass
        httpd.shutdown()


if __name__ == "__main__":
    sys.exit(main())
