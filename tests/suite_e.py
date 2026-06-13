#!/usr/bin/env python3
"""
Suite E: real `uv` against a mock package index that resets the TCP connection
mid-download, reproducing issue #6274's error chain and proving the fix layers:

  E0  healthy server                              -> uv installs OK (plumbing sane)
  E1  always-reset, UV_HTTP_RETRIES=0             -> uv FAILS with connection reset (repro)
  E2  always-reset, UV_HTTP_RETRIES honored       -> more attempts at the WIRE than retries=0
  E3  reset-first-2, UV_HTTP_RETRIES=8            -> uv RECOVERS (layer 2: uv-level retry)
  E4  reset-first-K, UV_HTTP_RETRIES=0:
        single `uv pip install`                   -> FAILS
        wrapped in run_install_cmd_retry          -> RECOVERS (layer 3: external wrapper)

Everything runs in ./temp. A handmade pure-python wheel is served so a recovered
download actually installs.
"""
import base64, hashlib, io, os, socket, struct, subprocess, sys, threading, time, zipfile
from shutil import which

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TEMP = os.path.join(ROOT, "temp", "suite_e")
os.makedirs(TEMP, exist_ok=True)

def _find(name):
    """Locate an installer file whether the layout is ROOT/<name> (repo root,
    as in CI) or ROOT/unsloth/<name> (this workspace)."""
    for c in (os.path.join(ROOT, name), os.path.join(ROOT, "unsloth", name)):
        if os.path.isfile(c):
            return c
    return os.path.join(ROOT, name)

INSTALL_SH = _find("install.sh")
INSTALL_PS1 = _find("install.ps1")

WHEEL_NAME = "dummypkg-0.0.1-py3-none-any.whl"

# ---------------- build a valid, RECORD-correct, pure-python wheel ----------------
def build_wheel(path):
    init_py = b"VERSION = '0.0.1'\n" + b"# pad\n" * 16384  # ~100KB so a mid-stream reset is meaningful
    files = {
        "dummypkg/__init__.py": init_py,
        "dummypkg-0.0.1.dist-info/METADATA":
            b"Metadata-Version: 2.1\nName: dummypkg\nVersion: 0.0.1\nSummary: test\n",
        "dummypkg-0.0.1.dist-info/WHEEL":
            b"Wheel-Version: 1.0\nGenerator: handmade\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    def record_hash(b):
        d = hashlib.sha256(b).digest()
        return "sha256=" + base64.urlsafe_b64encode(d).decode().rstrip("=")
    rec_lines = [f"{n},{record_hash(b)},{len(b)}" for n, b in files.items()]
    rec_lines.append("dummypkg-0.0.1.dist-info/RECORD,,")
    files["dummypkg-0.0.1.dist-info/RECORD"] = ("\n".join(rec_lines) + "\n").encode()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for n, b in files.items():
            z.writestr(n, b)
    data = buf.getvalue()
    with open(path, "wb") as f:
        f.write(data)
    return data

WHEEL_PATH = os.path.join(TEMP, WHEEL_NAME)
WHEEL_BYTES = build_wheel(WHEEL_PATH)

# ---------------- mock index server (raw sockets, controllable reset) ----------------
class Server:
    def __init__(self):
        self.lock = threading.Lock()
        self.reset_n = 0          # reset the first N GET attempts to the wheel
        self.attempts = 0         # GET attempts to the wheel since last config
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(64)
        self.port = self.sock.getsockname()[1]
        self.running = True

    def configure(self, reset_n):
        with self.lock:
            self.reset_n = reset_n
            self.attempts = 0

    def get_attempts(self):
        with self.lock:
            return self.attempts

    def serve_forever(self):
        while self.running:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                break
            threading.Thread(target=self.handle, args=(conn,), daemon=True).start()

    def handle(self, conn):
        try:
            conn.settimeout(5)
            data = b""
            while b"\r\n\r\n" not in data:
                chunk = conn.recv(4096)
                if not chunk:
                    conn.close(); return
                data += chunk
            line = data.split(b"\r\n", 1)[0].decode("latin1")
            parts = line.split(" ")
            method, path = (parts + ["", ""])[:2]

            if path.startswith("/simple/dummypkg"):
                body = (b'<!DOCTYPE html><html><body>'
                        b'<a href="/files/' + WHEEL_NAME.encode() + b'">' + WHEEL_NAME.encode() + b'</a>'
                        b'</body></html>')
                self.send(conn, 200, "text/html", body); conn.close(); return

            if path.startswith("/files/" + WHEEL_NAME):
                if method == "HEAD":
                    self.send_headers(conn, 200, "application/octet-stream", len(WHEEL_BYTES))
                    conn.close(); return
                with self.lock:
                    self.attempts += 1
                    n = self.attempts
                    do_reset = n <= self.reset_n
                if do_reset:
                    # Headers claim the full length, then we send a substantial but
                    # incomplete body and RST mid-stream. uv classifies this as a
                    # transient transfer/body error (and RETRIES it per UV_HTTP_RETRIES);
                    # only after exhausting retries does it surface a final error.
                    self.send_headers(conn, 200, "application/octet-stream", len(WHEEL_BYTES))
                    try:
                        conn.sendall(WHEEL_BYTES[: len(WHEEL_BYTES) // 2])
                    except OSError:
                        pass
                    # SO_LINGER (1, 0) -> close() sends a RST -> client sees connection reset.
                    conn.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
                    conn.close(); return
                self.send(conn, 200, "application/octet-stream", WHEEL_BYTES); conn.close(); return

            self.send(conn, 404, "text/plain", b"nope"); conn.close()
        except Exception:
            try: conn.close()
            except OSError: pass

    def send_headers(self, conn, code, ctype, length):
        hdr = (f"HTTP/1.1 {code} OK\r\nContent-Type: {ctype}\r\n"
               f"Content-Length: {length}\r\nConnection: close\r\n\r\n").encode()
        conn.sendall(hdr)

    def send(self, conn, code, ctype, body):
        self.send_headers(conn, code, ctype, len(body))
        try: conn.sendall(body)
        except OSError: pass

# ---------------- orchestration ----------------
PASS = 0; FAIL = 0
def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  PASS  {name}")
    else:
        FAIL += 1; print(f"  FAIL  {name}" + (f"\n        | {detail}" if detail else ""))

def run(cmd, env=None, timeout=120):
    e = dict(os.environ); e.update(env or {})
    p = subprocess.run(cmd, env=e, capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout + p.stderr)

def main():
    srv = Server()
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.port}/simple/"
    venv = os.path.join(TEMP, "venv")
    cache = os.path.join(TEMP, "uvcache")
    # Extract run_install_cmd + run_install_cmd_retry from install.sh so E4 works
    # standalone (no dependency on having run suite_a first).
    funcs = os.path.join(HERE, "_funcs.sh")
    _emit, _depth = False, 0
    with open(INSTALL_SH) as fh, open(funcs, "w") as out:
        for ln in fh:
            if ln.startswith("run_install_cmd() {"):
                _emit = True
            if _emit:
                out.write(ln)
                if ln.startswith("run_install_cmd_retry() {"):
                    _depth = 1
                elif _depth and ln.rstrip() == "}":
                    break

    # fresh venv
    rc, out = run(["uv", "venv", "--clear", venv, "--python", sys.executable])
    check("create uv venv", rc == 0, out)
    vpy = (os.path.join(venv, "Scripts", "python.exe") if os.name == "nt"
           else os.path.join(venv, "bin", "python"))

    # base uv install args (http localhost needs allow-insecure-host on modern uv)
    def uv_install(retries):
        return (["uv", "pip", "install", "--python", vpy, "--no-cache",
                 "--reinstall", "--index-url", base,
                 "--allow-insecure-host", "127.0.0.1", "dummypkg"],
                {"UV_HTTP_RETRIES": str(retries), "UV_CACHE_DIR": cache,
                 "UV_HTTP_TIMEOUT": "30"})

    def installed_ok():
        rc, _ = run([vpy, "-c", "import dummypkg; assert dummypkg.VERSION=='0.0.1'"])
        return rc == 0
    def uninstall():
        run(["uv", "pip", "uninstall", "--python", vpy, "dummypkg"])

    # ---- E0: healthy server -> success ----
    srv.configure(reset_n=0); uninstall()
    cmd, env = uv_install(0)
    rc, out = run(cmd, env)
    check("E0 healthy: uv install succeeds", rc == 0 and installed_ok(), out[-800:])

    # ---- E1: always reset, retries=0 -> failure with connection-reset chain ----
    srv.configure(reset_n=10_000); uninstall()
    cmd, env = uv_install(0)
    rc, out = run(cmd, env)
    n0 = srv.get_attempts()
    low = out.lower()
    # The user's exact "connection reset" wording is HTTP/2-specific; on HTTP/1.1
    # the same mid-download interruption may surface as any of these. The point is
    # that an interrupted download makes uv FAIL (and E2 proves uv retried it).
    transient_markers = ("connection reset", "error reading a body",
                         "error decoding response body", "connection closed",
                         "failed to fetch", "failed to download", "error sending request",
                         "invalid package format", "failed to read")
    repro = (rc != 0) and any(m in low for m in transient_markers)
    sig = next((m for m in transient_markers if m in low), "(no marker)")
    check("E1 repro: interrupted download makes uv FAIL (retries=0)", repro, out[-800:])
    print(f"        (wire attempts with retries=0: {n0}; failure signature: '{sig}')")

    # ---- E2: UV_HTTP_RETRIES honored at the wire (more attempts than retries=0) ----
    srv.configure(reset_n=10_000); uninstall()
    cmd, env = uv_install(5)
    rc, out = run(cmd, env)
    n5 = srv.get_attempts()
    check("E2 UV_HTTP_RETRIES honored: attempts(retries=5) > attempts(retries=0)",
          n5 > n0, f"n0={n0} n5={n5}")
    print(f"        (wire attempts with retries=5: {n5})")

    # ---- E3: reset first 2, retries=8 -> uv recovers (layer 2) ----
    srv.configure(reset_n=2); uninstall()
    cmd, env = uv_install(8)
    rc, out = run(cmd, env)
    check("E3 layer2: uv RECOVERS after 2 transient resets (retries=8)",
          rc == 0 and installed_ok(), f"attempts={srv.get_attempts()} :: {out[-600:]}")

    # ---- E4: external wrapper recovers where a single retries=0 invocation cannot ----
    # (POSIX-sh path; skipped on hosts without sh, e.g. bare Windows -> see E5.)
    if not which("sh"):
        print("  SKIP  E4 (sh not available; covered by E5/PowerShell on this host)")
    else:
        # single invocation first (must fail):
        srv.configure(reset_n=3); uninstall()
        cmd, env = uv_install(0)
        rc_single, out_single = run(cmd, env)
        single_failed = rc_single != 0 and not installed_ok()

        # same uv config wrapped in run_install_cmd_retry (server keeps resetting
        # the first 3 cumulative attempts; the wrapper's repeated invocations push past):
        srv.configure(reset_n=3); uninstall()
        cmd, env = uv_install(0)
        quoted = " ".join("'%s'" % c.replace("'", "'\\''") for c in cmd)
        wrap = f"""
set -e
C_OK= ; C_DIM= ; C_WARN= ; C_ERR= ; C_RST=
_is_verbose() {{ return 1; }}
step()    {{ :; }}
substep() {{ printf '    [substep] %s\\n' "$1"; }}
. "{funcs}"
export TMPDIR="{TEMP}/tmp"; mkdir -p "$TMPDIR"
UNSLOTH_INSTALL_RETRIES=5 UNSLOTH_INSTALL_RETRY_DELAY=1 run_install_cmd_retry "install dummypkg" {quoted}
"""
        e = dict(os.environ); e.update(env)
        p = subprocess.run(["sh", "-c", wrap], env=e, capture_output=True, text=True, timeout=180)
        wrapped_ok = p.returncode == 0 and installed_ok()
        check("E4 layer3: single retries=0 invocation FAILS", single_failed,
              f"rc={rc_single} :: {out_single[-300:]}")
        check("E4 layer3: run_install_cmd_retry RECOVERS the same uv config", wrapped_ok,
              f"rc={p.returncode} :: {(p.stdout + p.stderr)[-600:]}")
        retr = (p.stdout + p.stderr).count("retrying ")
        print(f"        (external retries observed: {retr}; wire attempts: {srv.get_attempts()})")

    # ---- E5: same recovery via the PowerShell wrapper (real uv through pwsh) ----
    pwsh = None
    for cand in ("pwsh", "pwsh-preview", "powershell"):
        if which(cand):
            pwsh = cand; break
    if pwsh and os.path.isfile(INSTALL_PS1):
        install_ps1 = INSTALL_PS1
        srv.configure(reset_n=3); uninstall()
        cmd, env = uv_install(0)  # UV_HTTP_RETRIES=0 -> rely on the external PS retry
        uv_args = " ".join("'%s'" % c.replace("'", "''") for c in cmd[1:])  # drop leading 'uv'
        ps = f"""
$ErrorActionPreference = 'Stop'
$src = Get-Content -Raw '{install_ps1}'
$ast = [System.Management.Automation.Language.Parser]::ParseInput($src,[ref]$null,[ref]$null)
$fns = $ast.FindAll({{ param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -in @('Invoke-InstallCommand','Invoke-InstallCommandRetry') }}, $true)
foreach ($f in $fns) {{ Invoke-Expression $f.Extent.Text }}
$script:UnslothVerbose = $false
function substep {{ param($m,$c) Write-Host "    [substep] $m" }}
$rc = Invoke-InstallCommandRetry -Label 'install dummypkg' {{ & uv {uv_args} }}
exit [int]$rc
"""
        e = dict(os.environ); e.update(env)
        e["UNSLOTH_INSTALL_RETRIES"] = "5"; e["UNSLOTH_INSTALL_RETRY_DELAY"] = "1"
        p = subprocess.run([pwsh, "-NoProfile", "-Command", ps], env=e,
                           capture_output=True, text=True, timeout=180)
        e5_ok = p.returncode == 0 and installed_ok()
        retr = (p.stdout + p.stderr).count("retrying ")
        check("E5 layer3 (PowerShell): Invoke-InstallCommandRetry recovers real uv reset",
              e5_ok, f"rc={p.returncode} :: {(p.stdout + p.stderr)[-600:]}")
        print(f"        (PowerShell external retries observed: {retr}; wire attempts: {srv.get_attempts()})")
    else:
        print("  SKIP  E5 (pwsh not available)")

    srv.running = False
    try: srv.sock.close()
    except OSError: pass
    print("  ------------------------------------")
    print(f"  SUITE E: {PASS} passed, {FAIL} failed")
    sys.exit(0 if FAIL == 0 else 1)

if __name__ == "__main__":
    main()
