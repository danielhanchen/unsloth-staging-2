"""Windows: llama-server working set with every layer on an RPC device, base vs head, ABAB.

The offloaded weights live in rpc-server; what llama-server keeps of them is the mmap'd file pages
that unmap_fragment should release."""

import ctypes
import ctypes.wintypes as wt
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

MIB = 1024**2


class PMC(ctypes.Structure):
    _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD)] + [
        (n, ctypes.c_size_t)
        for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                  "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                  "PagefileUsage", "PeakPagefileUsage", "PrivateUsage", "PrivateWorkingSetSize")
    ] + [("SharedCommitUsage", ctypes.c_ulonglong)]


def procmem(pid):
    k = ctypes.windll.kernel32
    h = k.OpenProcess(0x1000, False, pid)
    c = PMC()
    c.cb = ctypes.sizeof(c)
    k.K32GetProcessMemoryInfo(h, ctypes.byref(c), c.cb)
    k.CloseHandle(h)
    return {"ws_mib": c.WorkingSetSize // MIB, "private_ws_mib": c.PrivateWorkingSetSize // MIB,
            "private_mib": c.PrivateUsage // MIB}


def wait_health(port, proc, timeout = 600):
    t = time.time()
    while time.time() - t < timeout:
        if proc.poll() is not None:
            return f"exited {proc.returncode}"
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout = 5) as r:
                if r.status == 200:
                    return "ready"
        except Exception:
            pass
        time.sleep(1)
    return "timeout"


def one(arm, bindir, model, mode, logs, i):
    rpc_log = open(logs / f"{i}_{arm}_{mode}_rpc.log", "wb")
    rpc = subprocess.Popen([str(bindir / "ggml-rpc-server.exe"), "-H", "127.0.0.1", "-p", "50052"],
                           stdout = rpc_log, stderr = subprocess.STDOUT)
    time.sleep(3)
    log = logs / f"{i}_{arm}_{mode}_server.log"
    srv = subprocess.Popen([str(bindir / "llama-server.exe"), "-m", str(model), "--port", "18316",
                            "--rpc", "127.0.0.1:50052", "-ngl", "99", "-c", "2048", "--parallel", "1",
                            "--fit", "off", "--load-mode", mode],
                           stdout = open(log, "wb"), stderr = subprocess.STDOUT)
    res = {"arm": arm, "mode": mode, "status": wait_health(18316, srv)}
    if res["status"] == "ready":
        body = json.dumps({"prompt": "The capital of France is", "n_predict": 24, "temperature": 0,
                           "seed": 1}).encode()
        req = urllib.request.Request("http://127.0.0.1:18316/completion", data = body,
                                     headers = {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout = 600) as r:
            res["text"] = json.loads(r.read())["content"]
        time.sleep(5)
        res["server"] = procmem(srv.pid)
        res["rpc"] = procmem(rpc.pid)
    for p in (srv, rpc):
        subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output = True)
        p.wait(timeout = 60)
    text = log.read_text(errors = "replace")
    res["log"] = [ln for ln in text.splitlines()
                  if any(k in ln for k in ("offloaded", "model buffer size", "VirtualUnlock", "warning", "error"))][:20]
    time.sleep(5)
    return res


def main():
    base, head, model, out = map(Path, sys.argv[1:5])
    logs = out.parent / "logs"
    logs.mkdir(parents = True, exist_ok = True)
    runs = []
    i = 0
    for mode in ("mmap", "none"):
        for arm, bindir in (("base", base), ("head", head), ("base", base), ("head", head)):
            runs.append(one(arm, bindir, model, mode, logs, i))
            i += 1
            print(json.dumps({k: v for k, v in runs[-1].items() if k != "log"}), flush = True)
    out.write_text(json.dumps(runs, indent = 2))


main()
