"""Launch a headless Studio behind a public `--secure` Cloudflare link.

    python -m studio_test_kit.secure_launch --home ./studio_home --port 8901

Prints the URL and exits 0 once it answers, non-zero if no tunnel appears.

Verified on Linux 2026-09: `--secure` fetches `cloudflared` to `$UNSLOTH_STUDIO_HOME/bin/`, binds
`127.0.0.1` ONLY (`ss -ltnp`), and publishes a `*.trycloudflare.com` URL, so the raw port is never
public. Default username is `unsloth`, not `admin`. The URL lands in the log seconds AFTER
"Application startup complete", so poll rather than sleep.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
READY_RE = re.compile(r"Unsloth Studio is running|Application startup complete")


def launch(home: Path, port: int, password: str, log: Path, timeout: float) -> str:
    """Start `unsloth studio --secure` detached and return the public URL."""
    binary = home / "bin" / "unsloth"
    if not binary.exists():
        raise SystemExit(f"no Studio at {binary}; install with UNSLOTH_STUDIO_HOME={home} first")

    env = {**os.environ, "UNSLOTH_STUDIO_HOME": str(home), "UNSLOTH_STUDIO_PASSWORD": password}
    log.parent.mkdir(parents=True, exist_ok=True)
    # A real file, never a pipe: if the pipe's reader dies, request logging raises
    # BrokenPipeError and endpoints start 500ing on an otherwise-healthy server.
    with log.open("ab") as sink:
        subprocess.Popen(
            [str(binary), "studio", "--secure", "-p", str(port)],
            env=env, stdout=sink, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True,
        )

    deadline = time.time() + timeout
    while time.time() < deadline:
        text = log.read_text(errors="replace") if log.exists() else ""
        found = URL_RE.search(text)
        if found:
            return found.group(0)
        if "Address already in use" in text:
            raise SystemExit(f"port {port} is taken")
        time.sleep(2)
    raise SystemExit(f"no trycloudflare URL in {log} after {timeout:.0f}s")


def verify(url: str, timeout: float = 30.0) -> int:
    """GET /api/health over the link: resolving is not the same as serving."""
    with urllib.request.urlopen(f"{url}/api/health", timeout=timeout) as r:
        return r.status


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--home", type=Path, required=True, help="UNSLOTH_STUDIO_HOME, absolute")
    p.add_argument("--port", type=int, default=8901)
    p.add_argument("--password", default=os.environ.get("UNSLOTH_STUDIO_PASSWORD", ""))
    p.add_argument("--log", type=Path, default=None)
    p.add_argument("--timeout", type=float, default=180.0)
    a = p.parse_args()
    if not a.password:
        raise SystemExit("--password or UNSLOTH_STUDIO_PASSWORD required (a bare --secure prompts)")

    log = a.log or a.home.parent / f"studio_secure_{a.port}.log"
    url = launch(a.home.resolve(), a.port, a.password, log, a.timeout)
    status = verify(url)
    print(f"url    {url}")
    print(f"health {status}")
    print(f"local  http://127.0.0.1:{a.port}  (loopback only; the raw port is never public)")
    print(f"log    {log}")
    return 0 if status == 200 else 1


if __name__ == "__main__":
    sys.exit(main())
