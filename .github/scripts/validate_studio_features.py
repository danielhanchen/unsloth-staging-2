#!/usr/bin/env python3
"""Headless Unsloth Studio feature battery (Docker + native).

Drives the Studio HTTP API end to end and reports a JSON summary, exiting
nonzero if any REQUIRED check fails. Works against a Studio in a Docker
container (--mode docker --container NAME) or a native install
(--mode native --studio-home DIR --log-file FILE).

Checks: health, login+change-password, model load, /v1/chat/completions,
/v1/responses, terminal tool, python code-exec, web search (real page content),
MCP (registers a local fastmcp server and calls its tool), and the Cloudflare
quick-tunnel URL. Optional --mlx-model loads a safetensors model to exercise the
Apple MLX backend (macOS lane).

Secrets (bootstrap password, JWT, tunnel URL) are emitted as GitHub
::add-mask:: directives so they never appear raw in CI logs.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
import urllib.error

NEWPW = "UnslothVerify_2026!"


def mask(v):
    if v:
        print(f"::add-mask::{v}", flush=True)


def http(method, url, token=None, body=None, timeout=120, stream=False):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    resp = urllib.request.urlopen(req, timeout=timeout)
    if stream:
        return resp  # caller iterates lines
    raw = resp.read().decode("utf-8", "replace")
    ctype = resp.headers.get("Content-Type", "")
    return resp.status, (json.loads(raw) if "json" in ctype and raw.strip() else raw)


def http_code(url, token=None, timeout=20):
    try:
        s, _ = http("GET", url, token=token, timeout=timeout)
        return s
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:
        return 0


def sse_tool_events(base, token, prompt, tools, timeout=480):
    """POST a streaming chat with tools; return (tool_args[], tool_results[])."""
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 900, "temperature": 0.0,
        "enable_tools": True, "enabled_tools": tools, "stream": True,
    }
    calls, results = [], []
    resp = http("POST", f"{base}/v1/chat/completions", token=token, body=body,
                timeout=timeout, stream=True)
    for ln in resp:
        ln = ln.decode("utf-8", "replace").strip()
        if not ln.startswith("data: "):
            continue
        b = ln[6:].strip()
        if b == "[DONE]":
            break
        try:
            d = json.loads(b)
        except Exception:
            continue
        if d.get("type") == "tool_start":
            calls.append(d.get("arguments"))
        elif d.get("type") == "tool_end":
            results.append(str(d.get("result", "")))
    return calls, results


def get_password(args):
    if args.mode == "docker":
        out = subprocess.run(
            ["docker", "exec", args.container, "cat",
             f"{args.studio_home}/auth/.bootstrap_password"],
            capture_output=True, text=True, timeout=30)
        return out.stdout.strip()
    with open(os.path.join(args.studio_home, "auth", ".bootstrap_password")) as f:
        return f.read().strip()


def read_logs(args):
    if args.mode == "docker":
        out = subprocess.run(["docker", "logs", args.container],
                             capture_output=True, text=True, timeout=30)
        return out.stdout + out.stderr
    if args.log_file and os.path.exists(args.log_file):
        with open(args.log_file, "r", errors="replace") as f:
            return f.read()
    return ""


def start_mcp_server(args, port):
    """Launch mcp_test_server.py where Studio can reach it on 127.0.0.1:port."""
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(here, "mcp_test_server.py")
    if args.mode == "docker":
        subprocess.run(["docker", "cp", src, f"{args.container}:/tmp/mcp_test_server.py"],
                       check=True, timeout=30)
        py = f"{args.studio_home}/unsloth_studio/bin/python"
        subprocess.Popen(["docker", "exec", "-d", args.container, py,
                          "/tmp/mcp_test_server.py", str(port)])
    else:
        py = args.mcp_python or sys.executable
        subprocess.Popen([py, src, str(port)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return f"http://127.0.0.1:{port}/mcp"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["docker", "native"], required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--container", default="")
    ap.add_argument("--studio-home", default="/opt/unsloth-studio")
    ap.add_argument("--log-file", default="")
    ap.add_argument("--model", default="unsloth/Qwen3.5-0.8B-MTP-GGUF")
    ap.add_argument("--gguf-variant", default="UD-Q4_K_XL")
    ap.add_argument("--mlx-model", default="")
    ap.add_argument("--mcp-python", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    base = args.base_url.rstrip("/")
    results = {}

    def record(name, ok, detail="", required=True):
        results[name] = {"ok": bool(ok), "required": required, "detail": str(detail)[:300]}
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {str(detail)[:160]}", flush=True)

    # 1) health
    ok = False
    for _ in range(90):
        if http_code(f"{base}/api/health") == 200:
            ok = True
            break
        time.sleep(4)
    record("health", ok, "/api/health 200")
    if not ok:
        return finish(results, args)

    # 2) auth: login -> change-password -> relogin (idempotent across re-runs)
    token = ""

    def _login(password):
        try:
            _, d = http("POST", f"{base}/api/auth/login",
                        body={"username": "unsloth", "password": password})
            return d
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                return None
            raise

    try:
        pw = get_password(args)
        mask(pw)
        d = _login(pw)
        if d is None:
            # bootstrap password already rotated by a prior run -> use the rotated one
            d = _login(NEWPW)
            if d is None:
                raise RuntimeError("login failed with bootstrap and rotated password")
            token = d.get("access_token", "")
            mask(token)
            record("auth", bool(token), "jwt via rotated password (re-run)")
        else:
            token = d.get("access_token", "")
            mask(token)
            must = d.get("must_change_password")
            if must:
                http("POST", f"{base}/api/auth/change-password", token=token,
                     body={"current_password": pw, "new_password": NEWPW})
                d = _login(NEWPW)
                token = d.get("access_token", "")
                mask(token)
            record("auth", bool(token), f"jwt acquired, must_change_password={must}")
    except Exception as e:
        record("auth", False, repr(e))
        return finish(results, args)

    # 3) load model
    try:
        s, d = http("POST", f"{base}/api/inference/load", token=token, timeout=600,
                    body={"model_path": args.model, "gguf_variant": args.gguf_variant,
                          "max_seq_length": 4096})
        status = d.get("status") if isinstance(d, dict) else d
        ready = str(status) in ("loaded", "already_loaded")
        for _ in range(60):
            if ready:
                break
            try:
                _, p = http("GET", f"{base}/api/inference/load-progress", token=token, timeout=20)
                if isinstance(p, dict) and p.get("phase") == "ready":
                    ready = True
                    break
            except Exception:
                pass
            time.sleep(5)
        record("model_load", ready, f"status={status}")
    except Exception as e:
        record("model_load", False, repr(e))

    # 4) chat/completions
    try:
        s, d = http("POST", f"{base}/v1/chat/completions", token=token, timeout=120,
                    body={"messages": [{"role": "user", "content": "Reply with exactly: API_OK /no_think"}],
                          "max_tokens": 48, "temperature": 0.0, "stream": False})
        txt = d["choices"][0]["message"]["content"]
        record("chat_completions", "api_ok" in txt.lower(), repr(txt[:80]))
    except Exception as e:
        record("chat_completions", False, repr(e))

    # 5) responses
    try:
        s, d = http("POST", f"{base}/v1/responses", token=token, timeout=120,
                    body={"model": args.model, "input": "Reply with exactly: RESPONSES_OK /no_think",
                          "max_output_tokens": 48, "stream": False})
        blob = json.dumps(d)
        record("responses", "responses_ok" in blob.lower() and not d.get("error"), blob[:80])
    except Exception as e:
        record("responses", False, repr(e))

    # 6) terminal tool
    try:
        # `echo TERMINAL_OK` is portable across bash / cmd / PowerShell (the
        # native lanes run different shells); bash arithmetic is not.
        calls, res = sse_tool_events(
            base, token,
            "Use the terminal tool to run exactly: echo TERMINAL_OK -- then report the output. /no_think",
            ["terminal"])
        record("tool_terminal", any("TERMINAL_OK" in r for r in res),
               f"calls={calls[:1]} results={[r[:40] for r in res[:1]]}")
    except Exception as e:
        record("tool_terminal", False, repr(e))

    # 7) python code-exec
    try:
        calls, res = sse_tool_events(
            base, token,
            "Use the python tool to run exactly this code: print(6*7) -- then report the printed integer. /no_think",
            ["python"])
        record("code_exec_python", any("42" in r for r in res),
               f"calls={calls[:1]} results={[r[:40] for r in res[:1]]}")
    except Exception as e:
        record("code_exec_python", False, repr(e))

    # 8) web search (assert real page content)
    try:
        # Hand the model the explicit URL so the tool call is trivial (it just
        # fetches), and retry: a tiny model on a slow CPU runner is flaky about
        # deciding to emit a tool call. Asserting on fetched page content proves
        # web_search returned the real page, not just a model guess.
        joined, calls, res = "", [], []
        for _ in range(4):
            calls, res = sse_tool_events(
                base, token,
                "Call the web_search tool with url=\"https://www.rust-lang.org\" to fetch that page, "
                "then quote one sentence from it. You MUST use the tool. /no_think",
                ["web_search"])
            joined = " ".join(res).lower()
            if res and "rust" in joined:
                break
        record("web_search", bool(res) and "rust" in joined,
               f"calls={calls[:1]} snippet={joined[:80]}")
    except Exception as e:
        record("web_search", False, repr(e))

    # 9) MCP
    try:
        port = 9137
        url = start_mcp_server(args, port)
        time.sleep(6)
        s, d = http("POST", f"{base}/api/mcp/servers/", token=token, timeout=60,
                    body={"display_name": "unsloth-test-mcp", "url": url,
                          "is_enabled": True, "use_oauth": False})
        sid = d.get("id") if isinstance(d, dict) else None
        tool_count = 0
        if sid is not None:
            try:
                _, r = http("POST", f"{base}/api/mcp/servers/{sid}/refresh", token=token, timeout=60)
                tool_count = r.get("tool_count", 0) if isinstance(r, dict) else 0
            except Exception:
                pass
        calls, res = sse_tool_events(
            base, token,
            "Use the add_numbers MCP tool to add 19 and 23, then report the sum. /no_think",
            None, timeout=180) if False else ([], [])
        # MCP tools come via mcp_enabled; disable local tools (enabled_tools=[])
        # so the tiny model can't satisfy the request with the local python tool
        # and must call the MCP add_numbers tool.
        body = {"messages": [{"role": "user",
                "content": "Use the add_numbers tool to add 19 and 23, then report the sum. /no_think"}],
                "max_tokens": 600, "temperature": 0.0,
                "enable_tools": True, "enabled_tools": [], "mcp_enabled": True, "stream": True}
        resp = http("POST", f"{base}/v1/chat/completions", token=token, body=body, timeout=180, stream=True)
        names, mres = [], []
        for ln in resp:
            ln = ln.decode("utf-8", "replace").strip()
            if not ln.startswith("data: "):
                continue
            b = ln[6:].strip()
            if b == "[DONE]":
                break
            try:
                d2 = json.loads(b)
            except Exception:
                continue
            if d2.get("type") == "tool_start":
                names.append(str(d2.get("name", "")) + str(d2.get("arguments", "")))
            elif d2.get("type") == "tool_end":
                mres.append(str(d2.get("result", "")))
        got42 = any('"sum":42' in r.replace(" ", "") or "42" in r for r in mres)
        # MCP is proven by the server's tool being discovered (tool_count>=1) and
        # its result (sum=42) coming back through the chat loop.
        record("mcp", (tool_count >= 1) and got42,
               f"tool_count={tool_count} names={names[:1]} results={[r[:40] for r in mres[:1]]}")
    except Exception as e:
        record("mcp", False, repr(e))

    # 10) cloudflare
    try:
        logs = read_logs(args)
        urls = re.findall(r"https://[A-Za-z0-9-]+\.trycloudflare\.com", logs)
        url = urls[-1] if urls else ""
        mask(url)
        reach = False
        if url:
            try:
                code = http_code(url, timeout=30)
                reach = code in (200, 301, 302, 401, 403, 404, 502, 530)
            except Exception:
                reach = False
        record("cloudflare", bool(url) and reach,
               f"found={bool(url)} reachable={reach}")
    except Exception as e:
        record("cloudflare", False, repr(e))

    # 11) optional MLX (safetensors) for macOS lane
    if args.mlx_model:
        try:
            http("POST", f"{base}/api/inference/load", token=token, timeout=900,
                 body={"model_path": args.mlx_model, "max_seq_length": 1024})
            s, d = http("POST", f"{base}/v1/chat/completions", token=token, timeout=180,
                        body={"model": args.mlx_model,
                              "messages": [{"role": "user", "content": "Reply with exactly: MLX_OK"}],
                              "max_tokens": 32, "temperature": 0.0, "stream": False})
            txt = d["choices"][0]["message"]["content"]
            record("mlx_safetensors", "mlx_ok" in txt.lower(), repr(txt[:80]))
        except Exception as e:
            record("mlx_safetensors", False, repr(e))

    # 12) notebook refresh (docker only): delete/edit/restore behavior in the
    # real container. A probe deletes one baked notebook, edits another, runs the
    # boot-time sync, and reports whether the deleted one healed back, the edited
    # one was left intact, and a third stayed unchanged. Native lanes have no
    # in-container notebook sync, so this check is docker-only.
    if args.mode == "docker":
        try:
            here = os.path.dirname(os.path.abspath(__file__))
            probe = os.path.join(here, "notebook_refresh_probe.sh")
            subprocess.run(["docker", "cp", probe, f"{args.container}:/tmp/nb_probe.sh"],
                           check=True, timeout=30)
            out = subprocess.run(["docker", "exec", args.container, "bash", "/tmp/nb_probe.sh"],
                                 capture_output=True, text=True, timeout=300)
            line = ""
            for ln in (out.stdout + out.stderr).splitlines():
                if ln.startswith("NBRESULT"):
                    line = ln.strip()
            flags = dict(kv.split("=", 1) for kv in line.split()[1:]) if line else {}
            a_off = flags.get("a_off") == "1"     # deleted -> restored (offline)
            b_kept = flags.get("b_kept") == "1"   # edited  -> kept
            c_same = flags.get("c_same") == "1"   # other   -> unchanged
            a_net = flags.get("a_net") == "1"     # deleted -> restored from upstream (bonus)
            record("notebook_refresh", a_off and b_kept and c_same,
                   f"deleted_restored={a_off} edited_kept={b_kept} "
                   f"others_unchanged={c_same} upstream_restore={a_net}")
        except Exception as e:
            record("notebook_refresh", False, repr(e))

    return finish(results, args)


def finish(results, args):
    required_fail = [k for k, v in results.items() if v["required"] and not v["ok"]]
    summary = {"results": results, "required_failures": required_fail,
               "passed": len(required_fail) == 0}
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
    sys.exit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
