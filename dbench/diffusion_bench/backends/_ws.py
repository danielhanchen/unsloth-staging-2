"""A minimal RFC 6455 client (stdlib only) for BlineUI's /ws progress feed: text frames in, pings answered, no
extensions, so the bench venv needs no websocket-client package."""

from __future__ import annotations

import base64
import json
import os
import socket
import struct
from typing import Optional


class WSClosed(ConnectionError):
    pass


class MiniWS:
    def __init__(self, host: str, port: int, path: str, timeout: float = 30):
        self.sock = socket.create_connection((host, port), timeout = timeout)
        key = base64.b64encode(os.urandom(16)).decode()
        req = (f"GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
               f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n")
        self.sock.sendall(req.encode())
        head = b""
        while b"\r\n\r\n" not in head:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise WSClosed("websocket handshake: connection closed")
            head += chunk
        status, _, rest = head.partition(b"\r\n\r\n")
        if b" 101 " not in status.split(b"\r\n", 1)[0]:
            raise WSClosed(f"websocket handshake refused: {status[:200]!r}")
        self.buf = rest

    def _read(self, n: int) -> bytes:
        while len(self.buf) < n:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise WSClosed("websocket closed")
            self.buf += chunk
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def _send(self, opcode: int, payload: bytes = b"") -> None:
        mask = os.urandom(4)
        head = bytes([0x80 | opcode])
        n = len(payload)
        if n < 126:
            head += bytes([0x80 | n])
        elif n < 65536:
            head += bytes([0x80 | 126]) + struct.pack(">H", n)
        else:
            head += bytes([0x80 | 127]) + struct.pack(">Q", n)
        self.sock.sendall(head + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(payload)))

    def settimeout(self, t: Optional[float]) -> None:
        self.sock.settimeout(t)

    def recv(self):
        """The next text (str) or binary (bytes) message; raises socket.timeout on the socket timeout."""
        parts, kind = [], None
        while True:
            b1, b2 = self._read(2)
            fin, opcode = b1 & 0x80, b1 & 0x0F
            n = b2 & 0x7F
            if n == 126:
                n = struct.unpack(">H", self._read(2))[0]
            elif n == 127:
                n = struct.unpack(">Q", self._read(8))[0]
            mask = self._read(4) if b2 & 0x80 else None
            data = self._read(n)
            if mask:
                data = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
            if opcode == 0x9:
                self._send(0xA, data)
                continue
            if opcode == 0xA:
                continue
            if opcode == 0x8:
                raise WSClosed("websocket closed by server")
            if opcode in (0x1, 0x2):
                kind = opcode
            parts.append(data)
            if fin:
                payload = b"".join(parts)
                return payload.decode(errors = "replace") if kind == 0x1 else payload

    def close(self) -> None:
        try:
            self._send(0x8)
        except Exception:  # noqa: BLE001
            pass
        try:
            self.sock.close()
        except Exception:  # noqa: BLE001
            pass


def connect(host: str, port: int, path: str, timeout: float = 30):
    return MiniWS(host, port, path, timeout)


def parse(msg) -> Optional[dict]:
    if isinstance(msg, (bytes, bytearray)):
        return None  # preview images
    try:
        import extnames  # node ids / class names under the toolkit's spelling (identity outside a public copy)

        return json.loads(extnames.norm(msg))
    except Exception:  # noqa: BLE001
        return None
