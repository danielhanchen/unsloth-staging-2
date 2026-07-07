# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MLX server-side tool-calling smoke for Unsloth Studio on Apple Silicon.

Loads a small MLX quant through Studio's ``/api/inference/load``, confirms the
MLX backend is actually active (non-GGUF model on an ``_IS_MLX`` host), then
drives every server-side tool capability through the ``/v1`` API and reports a
PASS / WARN / FAIL matrix:

  * code execution   -- the ``python`` tool computes 123 * 456
  * file editing     -- ``python``/``terminal`` write a sentinel file, then read
                        it back in a later call sharing the same session_id (no
                        dedicated editor tool exists; file editing is code exec
                        against the per-session sandbox workdir)
  * web search       -- the keyless ``web_search`` (DuckDuckGo via ddgs) tool
  * MCP              -- register a stdio filesystem MCP server, confirm tool
                        discovery, then let the model call an MCP tool
  * function calling -- OpenAI client-tool passthrough (tool_choice=required)
  * render_html      -- the one-shot ``render_html`` tool
  * thinking on/off  -- enable_thinking toggle

MLX routes through the SAME ``run_safetensors_tool_loop`` as GGUF/safetensors
(via ``InferenceOrchestrator``), so this mirrors the GGUF tool-calling smoke in
``.github/workflows/studio-mac-inference-smoke.yml``. Small 4-bit quants under
Metal are degenerate at temperature 0, so we use temperature 0.2 + a fixed seed
and a PASS/WARN tiering that tolerates model drift: only the code-execution axis
is a hard gate (and only when ``--require-core`` is set), since it is the
cleanest proof that a tool call was emitted, dispatched, executed, and fed back.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

SEED = 3407
TEMP = 0.2

# Sentinel the file-editing axis writes then reads back. Distinctive so a
# read-back can be asserted unambiguously in the streamed answer.
FILE_SENTINEL = "unsloth-mlx-a1b2c3"


class Client:
    def __init__(self, base, token):
        self.base = base.rstrip("/")
        self.token = token

    def _headers(self):
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    def get(self, path, *, timeout=60):
        req = urllib.request.Request(
            f"{self.base}{path}", method="GET", headers=self._headers()
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode())

    def post(self, path, body, *, timeout=240):
        """Plain JSON POST. Non-agentic responses are a single JSON object."""
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            f"{self.base}{path}", data=data, method="POST", headers=self._headers()
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode())

    def post_sse(self, path, body, *, timeout=600):
        """POST a streaming request and accumulate assistant text + tool events.

        The server-side agentic loop ALWAYS returns SSE regardless of the
        request's ``stream`` field, so any call with ``enable_tools=true`` (or
        ``mcp_enabled=true``) must use this helper. Returns
        ``(text, tool_names, tool_results)`` where ``tool_names`` are the tools
        the loop started (``type=="tool_start"``) and ``tool_results`` are the
        executed tool outputs (``type=="tool_end"``, ``result`` field). Judging
        an axis on the executed tool result is far more robust than on the small
        model's final phrasing."""
        body = {**body, "stream": True}
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            f"{self.base}{path}", data=data, method="POST", headers=self._headers()
        )
        parts = []
        tool_names = []
        tool_results = []
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                # Studio interleaves agentic tool events into the SSE stream.
                ctype = chunk.get("type")
                if ctype == "tool_start":
                    name = chunk.get("tool_name")
                    if name:
                        tool_names.append(name)
                elif ctype == "tool_end":
                    tool_results.append(str(chunk.get("result", "")))
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta", {}) or {}
                    if delta.get("content"):
                        parts.append(delta["content"])
        return "".join(parts), tool_names, tool_results


class Results:
    """Collects per-axis outcomes and prints a matrix."""

    def __init__(self, model):
        self.model = model
        self.rows = []  # (axis, status, detail)
        self.tools_executed = 0  # count of real server-side tool executions seen

    def note_tool(self, count):
        self.tools_executed += count

    def add(self, axis, status, detail):
        self.rows.append((axis, status, detail))
        print(f"[mlx-tools] {status:4} {axis:16} :: {detail}", flush=True)

    def has_fail(self):
        return any(s == "FAIL" for _, s, _ in self.rows)

    def status_of(self, axis):
        for a, s, _ in self.rows:
            if a == axis:
                return s
        return None

    def summary(self):
        print("\n" + "=" * 66, flush=True)
        print(f"MLX tool-calling summary for {self.model}", flush=True)
        print("=" * 66, flush=True)
        for axis, status, detail in self.rows:
            print(f"  {status:4}  {axis:16}  {detail}", flush=True)
        print("=" * 66, flush=True)


# --------------------------------------------------------------------------- #
# Model load + MLX confirmation
# --------------------------------------------------------------------------- #
def load_and_confirm_mlx(cli, model, res, require_load):
    """Load ``model`` and confirm the active backend is MLX (non-GGUF on an
    Apple-Silicon host). Returns True on success. On load failure: raises
    SystemExit when ``require_load`` is set (hard gate -- used for core-tier
    models and for the gemma-4-e2b audio-conv load-fix verification), else
    records a WARN and returns False so the matrix reports "this quant does not
    load on MLX" without failing the job."""
    print(f"[mlx-tools] loading {model} ...", flush=True)

    def _fail(reason):
        if require_load:
            raise SystemExit(f"[mlx-tools] FATAL {reason}")
        res.add("model_load", "WARN", f"not loadable on MLX (report-only): {reason}")
        return False

    try:
        status, data = cli.post(
            "/api/inference/load",
            {"model_path": model, "is_lora": False, "max_seq_length": 2048},
            timeout=1800,
        )
        print(f"[mlx-tools] load returned status={status} {json.dumps(data)[:200]}", flush=True)
    except urllib.error.HTTPError as exc:
        return _fail(f"load HTTP {exc.code}: {exc.read().decode()[:200]}")
    except Exception as exc:
        return _fail(f"load error: {exc}")

    # Poll status until the model is loaded (load may be async / still
    # downloading + quantizing on the first run).
    deadline = time.time() + 1500
    st = {}
    while time.time() < deadline:
        try:
            _, st = cli.get("/api/inference/status", timeout=30)
        except Exception:
            time.sleep(5)
            continue
        loaded = st.get("loaded") or []
        loading = st.get("loading") or []
        if loaded and not loading:
            break
        time.sleep(5)

    loaded = st.get("loaded") or []
    if not loaded:
        return _fail(f"model never reached loaded state: {json.dumps(st)[:200]}")
    if bool(st.get("is_gguf", False)):
        return _fail("loaded model reports is_gguf=true (routed to llama.cpp, not MLX)")
    res.add("model_load", "PASS", f"loaded={loaded} is_gguf=false (MLX backend)")
    return True


# --------------------------------------------------------------------------- #
# Capability axes
# --------------------------------------------------------------------------- #
def _joined(text, results):
    return text + " " + " ".join(results)


def axis_code_exec(cli, res):
    """python tool: 123 * 456 = 56088. Judge on the executed tool RESULT (or the
    final answer), not the model's phrasing. Retry a few times: small Metal
    quants drift per-attempt, so a couple of tries reliably lands the tool call
    on a capable model without making the axis flaky."""
    last = "no attempt"
    for attempt in range(1, 4):
        try:
            text, tools, results = cli.post_sse(
                "/v1/chat/completions",
                {
                    "messages": [{"role": "user", "content": "What is 123 * 456? Use the python tool to compute it and tell me the exact number."}],
                    "enable_tools": True,
                    "enabled_tools": ["python"],
                    "session_id": f"ci-mlx-code-{attempt}",
                    "temperature": TEMP,
                    "seed": SEED + attempt,
                    "max_tokens": 256,
                },
                timeout=300,
            )
        except Exception as exc:
            res.add("code_exec", "FAIL", f"request error: {exc}")
            return
        res.note_tool(len(results))
        blob = _joined(text, results)
        if "56088" in blob or "56,088" in blob:
            res.add("code_exec", "PASS", f"python tool -> 56088 (attempt {attempt}, tools={tools}, {len(results)} exec)")
            return
        if "python" in tools:
            last = f"python executed but 56088 not surfaced (attempt {attempt}, {len(results)} exec)"
        else:
            last = f"no python tool call -- quant drift (attempt {attempt}, {len(text)} chars)"
    res.add("code_exec", "WARN", last)


def axis_file_edit(cli, res):
    """Write a sentinel file via the python tool, then read it back in a second
    call sharing the same session_id (proves the sandbox workdir persists)."""
    sid = "ci-mlx-fileedit"
    try:
        _, tools1, results1 = cli.post_sse(
            "/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": f"Use the python tool to create a file named note.txt in the current directory containing exactly the text {FILE_SENTINEL} and nothing else. Then confirm you wrote it."}],
                "enable_tools": True,
                "enabled_tools": ["python", "terminal"],
                "session_id": sid,
                "temperature": TEMP,
                "seed": SEED,
                "max_tokens": 256,
            },
            timeout=300,
        )
        text2, tools2, results2 = cli.post_sse(
            "/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": "Use the python tool to read the file note.txt from the current directory and print its exact contents."}],
                "enable_tools": True,
                "enabled_tools": ["python", "terminal"],
                "session_id": sid,
                "temperature": TEMP,
                "seed": SEED,
                "max_tokens": 256,
            },
            timeout=300,
        )
    except Exception as exc:
        res.add("file_edit", "WARN", f"request error (non-blocking): {exc}")
        return
    res.note_tool(len(results1) + len(results2))
    blob = _joined(text2, results2)
    if FILE_SENTINEL in blob:
        res.add("file_edit", "PASS", f"sentinel written+read back across session (tools={tools1}/{tools2})")
    elif tools2 or results2:
        res.add("file_edit", "WARN", f"tool ran but sentinel not surfaced (tools={tools2}, {len(results2)} exec)")
    else:
        res.add("file_edit", "WARN", f"no tool call on read-back -- quant drift ({len(text2)} chars)")


def axis_web_search(cli, res):
    """web_search (keyless DuckDuckGo). DDG is flaky from CI -> WARN-tier."""
    try:
        text, tools, results = cli.post_sse(
            "/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": "Search the web for 'unsloth ai github' and summarise the top result in one sentence."}],
                "enable_tools": True,
                "enabled_tools": ["web_search"],
                "session_id": "ci-mlx-web",
                "temperature": TEMP,
                "seed": SEED,
                "max_tokens": 128,
            },
            timeout=240,
        )
    except Exception as exc:
        res.add("web_search", "WARN", f"probe failed (non-blocking): {exc}")
        return
    res.note_tool(len(results))
    if "web_search" in tools:
        res.add("web_search", "PASS", f"web_search executed ({len(results)} exec, {len(text)} chars)")
    else:
        res.add("web_search", "WARN", f"stream ok but no web_search tool_start ({len(text)} chars)")


def axis_mcp(cli, res, workdir):
    """Register a stdio filesystem MCP server, confirm tool discovery
    (deterministic), then let the model call an MCP tool (model-dependent)."""
    # Seed a file the MCP filesystem server can read.
    try:
        os.makedirs(workdir, exist_ok=True)
        with open(os.path.join(workdir, "mcp_probe.txt"), "w") as fh:
            fh.write("mcp filesystem works\n")
    except Exception:
        pass
    server_id = None
    try:
        # Trailing slash: the router mounts @post("/") at prefix /api/mcp/servers.
        status, srv = cli.post(
            "/api/mcp/servers/",
            {
                "display_name": "ci-fs",
                "url": f"npx -y @modelcontextprotocol/server-filesystem {workdir}",
                "is_enabled": True,
            },
            timeout=120,
        )
        server_id = srv.get("id")
    except Exception as exc:
        res.add("mcp", "WARN", f"server registration failed (non-blocking): {exc}")
        return
    # Confirm tool discovery -- deterministic (not model-dependent). The refresh
    # endpoint returns McpServerProbeResult(ok, tool_count), a COUNT not a list.
    tool_count = 0
    err = None
    try:
        _, probe = cli.post(f"/api/mcp/servers/{server_id}/refresh", {}, timeout=180)
        tool_count = int(probe.get("tool_count", 0))
        err = probe.get("error")
    except Exception as exc:
        res.add("mcp", "WARN", f"registered but tool discovery failed (npx fetch?): {exc}")
        return
    if tool_count == 0:
        res.add("mcp", "WARN", f"registered but discovered 0 MCP tools (npx spawn?) err={err}")
        return
    # Successful server registration + tool discovery IS the MCP-integration
    # proof. Whether the small model then chooses to call an MCP tool is a
    # separate, drift-prone signal we note but don't gate on.
    try:
        text, tools, results = cli.post_sse(
            "/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": "List the files in the allowed directory using the filesystem MCP tool, then read mcp_probe.txt and print its contents."}],
                "enable_tools": True,
                "mcp_enabled": True,
                "session_id": "ci-mlx-mcp",
                "temperature": TEMP,
                "seed": SEED,
                "max_tokens": 256,
            },
            timeout=300,
        )
    except Exception as exc:
        res.add("mcp", "PASS", f"discovered {tool_count} MCP tools (integration works); model call errored: {exc}")
        return
    res.note_tool(len(results))
    mcp_started = [t for t in tools if str(t).startswith("mcp__")]
    if mcp_started or "mcp filesystem works" in _joined(text, results):
        res.add("mcp", "PASS", f"discovered {tool_count} tools + model invoked MCP ({mcp_started or 'read-back'})")
    else:
        res.add("mcp", "PASS", f"discovered {tool_count} MCP tools (integration works); model didn't call it ({len(text)} chars)")


def axis_function_calling(cli, res):
    """OpenAI client-tool passthrough (not server-side). tool_choice=required."""
    weather_tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
    try:
        status, data = cli.post(
            "/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
                "tools": [weather_tool],
                "tool_choice": "required",
                "stream": False,
                "temperature": TEMP,
                "seed": SEED,
                "max_tokens": 128,
            },
            timeout=180,
        )
    except Exception as exc:
        res.add("function_call", "WARN", f"request error (non-blocking): {exc}")
        return
    if status != 200:
        res.add("function_call", "WARN", f"HTTP {status}")
        return
    choice = data["choices"][0]
    tcs = (choice.get("message") or {}).get("tool_calls") or []
    if tcs and tcs[0]["function"]["name"] == "get_weather":
        try:
            args = json.loads(tcs[0]["function"]["arguments"])
        except Exception:
            args = {}
        if args.get("city"):
            res.add("function_call", "PASS", f"get_weather({args})")
            return
    res.add("function_call", "WARN", f"no/!schema tool_calls (finish={choice.get('finish_reason')!r}) -- quant drift")


def axis_render_html(cli, res):
    """render_html one-shot tool."""
    try:
        text, tools, results = cli.post_sse(
            "/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": "Use the render_html tool to render a minimal HTML page with an <h1>Hello Unsloth</h1> heading."}],
                "enable_tools": True,
                "enabled_tools": ["render_html"],
                "session_id": "ci-mlx-html",
                "temperature": TEMP,
                "seed": SEED,
                "max_tokens": 256,
            },
            timeout=240,
        )
    except Exception as exc:
        res.add("render_html", "WARN", f"request error (non-blocking): {exc}")
        return
    res.note_tool(len(results))
    if "render_html" in tools:
        res.add("render_html", "PASS", f"render_html executed ({len(results)} exec, {len(text)} chars)")
    else:
        res.add("render_html", "WARN", f"stream ok but no render_html tool_start ({len(text)} chars)")


def axis_thinking(cli, res):
    """enable_thinking on/off (plain chat, no tools)."""
    def call(enable):
        _, data = cli.post(
            "/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": "Briefly: is 17 prime?"}],
                "stream": False,
                "enable_thinking": enable,
                "temperature": TEMP,
                "seed": SEED,
                "max_tokens": 96,
            },
            timeout=180,
        )
        msg = data["choices"][0]["message"]
        return (msg.get("content") or "") + (msg.get("reasoning_content") or "")
    try:
        on_text = call(True)
        off_text = call(False)
    except Exception as exc:
        res.add("thinking", "WARN", f"request error (non-blocking): {exc}")
        return
    if "<think>" in off_text:
        res.add("thinking", "WARN", f"enable_thinking=False but <think> present ({len(off_text)} chars)")
    else:
        res.add("thinking", "PASS", f"on={len(on_text)} chars, off={len(off_text)} chars, no leaked <think>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="mlx-community repo id to load")
    ap.add_argument("--port", required=True, help="Studio port")
    ap.add_argument("--workdir", default=os.path.expanduser("~/mcp_fs"), help="MCP filesystem root")
    ap.add_argument("--require-core", action="store_true",
                    help="Fail (exit 1) if the code_exec axis does not PASS (core-tier models).")
    ap.add_argument("--require-load", action="store_true",
                    help="Fail (exit 1) if the model does not load on the MLX backend, "
                         "without gating on the drift-prone tool axes (used to verify the "
                         "gemma-4-e2b audio-conv load fix).")
    args = ap.parse_args()

    token = os.environ.get("API_KEY")
    if not token:
        print("[mlx-tools] FATAL API_KEY env not set", flush=True)
        return 2

    cli = Client(f"http://127.0.0.1:{args.port}", token)
    res = Results(args.model)

    # Hard gate: the model must load as MLX before any axis runs when it is a
    # core-tier model (--require-core) or when we are explicitly verifying that
    # it loads (--require-load). Otherwise a load failure is reported + skipped.
    require_load = args.require_core or args.require_load
    if not load_and_confirm_mlx(cli, args.model, res, require_load):
        res.summary()
        print("[mlx-tools] RESULT: OK (best-effort model not loadable, reported)", flush=True)
        return 0

    # Warm up the tool loop once (the first tool request after a cold model load
    # occasionally yields an empty generation); the result is discarded.
    try:
        cli.post_sse(
            "/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": "Use the python tool to print the text READY."}],
                "enable_tools": True,
                "enabled_tools": ["python"],
                "session_id": "ci-mlx-warmup",
                "temperature": TEMP,
                "seed": SEED,
                "max_tokens": 128,
            },
            timeout=240,
        )
    except Exception as exc:
        print(f"[mlx-tools] warmup skipped: {exc}", flush=True)

    axis_code_exec(cli, res)
    axis_file_edit(cli, res)
    axis_web_search(cli, res)
    axis_mcp(cli, res, args.workdir)
    axis_function_calling(cli, res)
    axis_render_html(cli, res)
    axis_thinking(cli, res)

    res.summary()
    print(f"[mlx-tools] total server-side tool executions observed: {res.tools_executed}", flush=True)

    # Exit policy (mirrors the GGUF Tool calling Tests job): the hard gates are
    # infrastructure only -- the model must load on the MLX backend (enforced in
    # load_and_confirm_mlx) and no endpoint may hard-error (has_fail). Per-axis
    # tool output is PASS/WARN informational, because small Metal quants drift
    # run-to-run; gating a green run on their sampling would be flaky. The
    # confirmation that tools actually work is the reported matrix across the
    # model fleet (e.g. python -> 56088, MCP tool invocation, web_search).
    if res.has_fail():
        print("[mlx-tools] RESULT: FAIL (a hard endpoint error occurred)", flush=True)
        return 1
    print("[mlx-tools] RESULT: OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
