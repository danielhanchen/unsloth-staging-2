"""Studio tool-calling + OS-sandbox latency bench (staging CI job; Linux, macOS, Windows, WSL2).

Arms (all CPU, no model download):
  tools     execute_tool per tool in the OS sandbox (python, terminal, curl via the egress proxy,
            file grep/read, edit_file, web_search against stubbed engines), with per-stage timers
  loop_ext  stream_with_studio_tools driven by a scripted fake provider: reasoning -> tool calls ->
            answer; canned and real tool bodies; per-round Studio overhead, per-chunk relay cost
  loop_gguf LlamaCppBackend.generate_chat_completion_with_tools with stubbed llama-server streams
Plus cProfile per arm, and a same-runner base/head A/B when jobs/tool_loop_patch.py is vendored.

    python .github/scripts/jobs/tool_loop_bench.py --tiny --out job.json [--wsl]
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import cProfile
import hashlib
import io
import json
import os
import platform
import pstats
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

REPO = Path.cwd()
SESSION = "__LOCALID_toolbench"


def _stats(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    return {
        "n": len(xs),
        "median_ms": round(statistics.median(xs), 2),
        "min_ms": round(xs[0], 2),
        "p90_ms": round(xs[min(len(xs) - 1, int(0.9 * len(xs)))], 2),
        "max_ms": round(xs[-1], 2),
    }


# ---------------------------------------------------------------- instrumentation


class Stages:
    """Wrap named functions with wall-clock timers; nested calls are recorded separately."""

    def __init__(self):
        self.times: dict[str, list[float]] = {}
        self.counts: dict[str, int] = {}
        self._undo = []

    def wrap(
        self,
        module,
        attr,
        label = None,
    ):
        fn = getattr(module, attr, None)
        if fn is None or not callable(fn):
            return
        label = label or f"{module.__name__.rsplit('.', 1)[-1]}.{attr}"
        times = self.times.setdefault(label, [])

        def timed(*a, **k):
            t = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                times.append((time.perf_counter() - t) * 1000)

        timed.__wrapped__ = fn
        setattr(module, attr, timed)
        self._undo.append((module, attr, fn))

    def count(self, module, attr, label):
        fn = getattr(module, attr, None)
        if fn is None:
            return
        self.counts.setdefault(label, 0)

        def counted(*a, **k):
            self.counts[label] += 1
            return fn(*a, **k)

        setattr(module, attr, counted)
        self._undo.append((module, attr, fn))

    def reset(self):
        for v in self.times.values():
            v.clear()
        for k in self.counts:
            self.counts[k] = 0

    def snapshot(self, calls):
        return {
            "stages_ms_per_call": {k: round(sum(v) / calls, 2) for k, v in self.times.items() if v},
            "stage_calls_per_call": {
                k: round(len(v) / calls, 2) for k, v in self.times.items() if v
            },
            "counters_per_call": {k: round(v / calls, 2) for k, v in self.counts.items()},
        }

    def undo(self):
        for module, attr, fn in reversed(self._undo):
            setattr(module, attr, fn)
        self._undo.clear()


def instrument(stages: Stages):
    from core.inference import os_sandbox, tools
    import core.inference.tools as tools_mod

    for attr in (
        "_check_code_safety",
        "_snapshot_and_history_begin",
        "_snapshot_workdir_views",
        "_prepare_tool_launch",
        "_drain_process_output",
        "_wait_process",
        "_truncate",
        "_tool_result_char_budget",
        "_adopt_tool_pid",
        "_forget_tool_pid",
        "_terminal_profile",
        "_references_studio_credential_here",
        "_python_exec",
        "_bash_exec",
        "_edit_file",
        "_web_search",
    ):
        stages.wrap(tools_mod, attr)
    for attr in (
        "capability_snapshot",
        "model_library_roots",
        "spawn_prepared_launch",
        "verify_prepared_completion",
        "finalize_prepared_cleanup",
        "trusted_tool_folder_roots",
    ):
        stages.wrap(os_sandbox, attr)
    with contextlib.suppress(Exception):
        from core.inference import sandbox_linux
        for attr in ("prepare", "_model_cache_binds", "_runtime_read_paths", "_start_egress"):
            stages.wrap(sandbox_linux, attr)
    with contextlib.suppress(Exception):
        from core.inference import sandbox_macos
        for attr in ("prepare", "build_profile"):
            stages.wrap(sandbox_macos, attr)
    with contextlib.suppress(Exception):
        from core.inference import sandbox_egress
        stages.wrap(sandbox_egress.EgressProxy, "close", "egress.EgressProxy.close")
    with contextlib.suppress(Exception):
        from core.inference import mxc_policy, sandbox_windows_mxc, mxc_runtime, mxc_adapter

        stages.wrap(sandbox_windows_mxc, "prepare", "mxc.prepare")
        stages.wrap(mxc_policy, "build_launch_request")
        stages.wrap(mxc_adapter, "spawn", "mxc_adapter.spawn")
        stages.count(mxc_runtime, "_sha256_file", "sha256_of_runtime")
    with contextlib.suppress(Exception):
        from storage import studio_db
        stages.count(studio_db, "get_connection", "db_connect")
    with contextlib.suppress(Exception):
        from utils import process_lifetime
        stages.wrap(process_lifetime, "_write_breadcrumb")
    real_popen_init = subprocess.Popen.__init__
    stages.counts["subprocess_spawn"] = 0

    def popen_init(self, *a, **k):
        stages.counts["subprocess_spawn"] += 1
        return real_popen_init(self, *a, **k)

    subprocess.Popen.__init__ = popen_init
    stages._undo.append((subprocess.Popen, "__init__", real_popen_init))
    real_start = threading.Thread.start
    stages.counts["thread_start"] = 0

    def start(self, *a, **k):
        stages.counts["thread_start"] += 1
        return real_start(self, *a, **k)

    threading.Thread.start = start
    stages._undo.append((threading.Thread, "start", real_start))
    return tools


# ---------------------------------------------------------------- local fakes


def _http_server():
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            body = b"toolbench-ok\n"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target = srv.serve_forever, daemon = True).start()
    return srv


def _stub_search_engines():
    """web_search runs its real tiering, filtering and formatting; only the engine HTTP is stubbed."""
    from core.inference import tools

    try:
        from ddgs.ddgs import DDGS
        from ddgs.engines import ENGINES
        from ddgs.results import TextResult
    except Exception as exc:  # noqa: BLE001
        return f"ddgs unavailable: {exc!r}"
    if callable(getattr(DDGS, "_get_network_client", None)):
        DDGS._get_network_client = lambda self: None
    tools._wikipedia_search = lambda *a, **k: []

    def search(engine, query, **kwargs):
        return [
            TextResult(
                title = f"{query} result {i}",
                href = f"https://example{i}.test/page",
                body = f"Synthetic snippet {i} about {query}.",
            )
            for i in range(5)
        ]

    for cls in ENGINES["text"].values():
        cls.search = search
    return "stubbed"


# ---------------------------------------------------------------- arm: tools


def _cases(port):
    win = sys.platform == "win32"
    grep = "findstr /s /n needle_17 corpus\\*.txt" if win else "grep -rn needle_17 corpus | head -5"
    read = "type corpus\\d3\\f_42.txt" if win else "cat corpus/d3/f_42.txt"
    cases = [
        ("python_print", "python", {"code": "print(1)"}),
        (
            "python_imports",
            "python",
            {"code": "import json, os, re, pathlib\nprint(len(os.listdir('.')))"},
        ),
        ("terminal_echo", "terminal", {"command": "echo 1"}),
        ("file_grep", "terminal", {"command": grep}),
        ("file_read", "terminal", {"command": read}),
        (
            "edit_file",
            "edit_file",
            {"path": "notes.txt", "edits": [{"old_string": "", "new_string": "x\n"}]},
        ),
        ("web_search", "web_search", {"query": "toolbench"}),
    ]
    if not win:
        cases.append(
            ("terminal_curl", "terminal", {"command": f"curl -s http://127.0.0.1:{port}/"})
        )
    return cases


def arm_tools(n, profile_out):
    stages = Stages()
    tools = instrument(stages)
    out = {"search_stub": _stub_search_engines()}
    srv = _http_server()
    port = srv.server_address[1]
    try:
        # Seed a 2,000-file corpus in the session workdir through the python tool itself.
        seed = (
            "import os\nfor d in range(20):\n os.makedirs(f'corpus/d{d}', exist_ok=True)\n"
            " for f in range(100):\n  open(f'corpus/d{d}/f_{f}.txt','w').write(f'line needle_{f}\\n'*4)\n"
            "print('seeded')"
        )
        out["seed"] = tools.execute_tool(
            "python", {"code": seed}, session_id = SESSION, tool_execution_mode = "required"
        )[-80:]
        for label, name, args in _cases(port):
            sessions = [("session", SESSION)] + (
                [("no_session", None)] if name in ("python", "terminal") else []
            )
            for slabel, sid in sessions:
                key = f"{label}|{slabel}"
                times, results, backends = [], [], set()
                for i in range(n + 2):
                    a = json.loads(json.dumps(args))
                    if name == "edit_file":
                        a["path"] = f"notes_{label}_{slabel}_{i}.txt"
                    if i == 2:
                        stages.reset()
                    t = time.perf_counter()
                    try:
                        r = tools.execute_tool(
                            name, a, session_id = sid, tool_execution_mode = "required"
                        )
                    except Exception as exc:  # noqa: BLE001
                        r = f"EXC {exc!r}"
                    if i >= 2:
                        times.append((time.perf_counter() - t) * 1000)
                        results.append(str(r))
                        if name in ("python", "terminal"):
                            backends.add(
                                str(getattr(tools._last_tool_execution_record, "backend", None))
                            )
                out[key] = {
                    "warm": _stats(times),
                    "backends": sorted(backends),
                    "sample": results[-1][-160:] if results else None,
                    **stages.snapshot(max(1, n)),
                }
        # Attribution profile: python + terminal, separate from the headline numbers.
        for label, name, args in (
            ("python", "python", {"code": "print(1)"}),
            ("terminal", "terminal", {"command": "echo 1"}),
        ):
            pr = cProfile.Profile()
            pr.enable()
            for _ in range(10):
                tools.execute_tool(
                    name, dict(args), session_id = SESSION, tool_execution_mode = "required"
                )
            pr.disable()
            s = io.StringIO()
            st = pstats.Stats(pr, stream = s)
            st.sort_stats("cumulative").print_stats("studio[/\\\\]backend", 45)
            st.sort_stats("tottime").print_stats(25)
            Path(profile_out + f".{label}.txt").write_text(s.getvalue())
            out[f"profile_{label}_head"] = s.getvalue()[:7000]
    finally:
        srv.shutdown()
        stages.undo()
    return out


# ---------------------------------------------------------------- arms: loops


def _sse(
    delta = None,
    finish = None,
    **extra,
):
    choice = {"index": 0, "delta": delta if delta is not None else {}}
    if finish is not None:
        choice["finish_reason"] = finish
    payload = {"choices": [choice], **extra}
    return "data: " + json.dumps(payload, ensure_ascii = False)


def _script(
    rounds,
    reasoning_chunks = 40,
    answer_chunks = 200,
    parallel = False,
):
    """Provider turns: per round reasoning + tool call(s); then a long streamed answer."""
    turns = []
    for r in range(rounds):
        lines = [_sse({"role": "assistant"})]
        lines += [
            _sse({"reasoning_content": f"thinking step {r}.{i} "}) for i in range(reasoning_chunks)
        ]
        calls = [("python", {"code": f"print({r})"})] + (
            [("terminal", {"command": f"echo r{r}"})] if parallel else []
        )
        for idx, (name, args) in enumerate(calls):
            raw = json.dumps(args)
            cut = len(raw) // 2
            cid = f"call_{r}_{idx}"
            lines.append(
                _sse(
                    {
                        "tool_calls": [
                            {
                                "index": idx,
                                "id": cid,
                                "type": "function",
                                "function": {"name": name, "arguments": raw[:cut]},
                            }
                        ]
                    }
                )
            )
            lines.append(
                _sse({"tool_calls": [{"index": idx, "function": {"arguments": raw[cut:]}}]})
            )
        lines.append(_sse({}, finish = "tool_calls"))
        lines.append("data: [DONE]")
        turns.append(lines)
    answer = [_sse({"role": "assistant"})] + [
        _sse({"content": f"word{i} "}) for i in range(answer_chunks)
    ]
    answer += [_sse({}, finish = "stop"), "data: [DONE]"]
    turns.append(answer)
    return turns


class _Transport:
    heals_text_tool_calls = True
    sanitizes_provider_frames = False

    def __init__(self, turns):
        self.turns = [list(t) for t in turns]
        self.requests = 0

    def stream(self, *, messages, tools, tool_choice, cancel_event):
        self.requests += 1
        lines = self.turns.pop(0) if self.turns else ["data: [DONE]"]

        async def gen():
            for line in lines:
                yield line

        return gen()


def _tool_defs():
    from core.inference import tools
    return [tools.PYTHON_TOOL, tools.TERMINAL_TOOL, tools.WEB_SEARCH_TOOL, tools.EDIT_FILE_TOOL]


def _timed_execute(module, attr, real, bodies, canned):
    def execute(name, arguments, **kwargs):
        t = time.perf_counter()
        try:
            if canned:
                return f"RESULT<{name}>"
            kwargs.setdefault("tool_execution_mode", "required")
            return real(name, arguments, **kwargs)
        finally:
            bodies.append((time.perf_counter() - t) * 1000)

    setattr(module, attr, execute)


def arm_loop_ext(n):
    import asyncio
    import threading as th
    from core.inference import studio_tool_loop as loop_mod

    real = loop_mod.execute_tool
    out = {}
    with contextlib.suppress(Exception):
        loop_mod.build_rag_autoinject = lambda *a, **k: None
    policy = loop_mod.ToolLoopPolicy(
        tools = _tool_defs(),
        max_calls = 25,
        timeout = 300,
        permission_mode = "off",
        confirm_calls = False,
        bypass_permissions = False,
        rag_scope = None,
    )

    async def drive(turns):
        transport = _Transport(turns)
        lines, first = 0, None
        t0 = time.perf_counter()
        agen = loop_mod.stream_with_studio_tools(
            transport,
            run = loop_mod.ToolLoopRun(
                messages = [{"role": "user", "content": "go"}],
                session_id = SESSION,
                thread_id = "toolbench",
            ),
            policy = policy,
            cancel_event = th.Event(),
        )
        async for line in agen:
            lines += 1
            if first is None and ("reasoning" in line or "content" in line):
                first = (time.perf_counter() - t0) * 1000
        return (time.perf_counter() - t0) * 1000, lines, first, transport.requests

    try:
        for label, rounds, parallel, canned, answer in (
            ("answer_only_2000_chunks", 0, False, True, 2000),
            ("3_rounds_canned", 3, False, True, 200),
            ("3_rounds_parallel2_canned", 3, True, True, 200),
            ("3_rounds_real_tools", 3, False, False, 200),
            ("3_rounds_parallel2_real", 3, True, False, 200),
        ):
            walls, overheads, ttft, nlines, reqs = [], [], [], 0, 0
            for i in range(n + 1):
                bodies = []
                _timed_execute(loop_mod, "execute_tool", real, bodies, canned)
                w, nlines, first, reqs = asyncio.run(
                    drive(_script(rounds, answer_chunks = answer, parallel = parallel))
                )
                if i == 0:
                    continue
                walls.append(w)
                overheads.append(w - sum(bodies))
                ttft.append(first)
            per = max(1, rounds)
            out[label] = {
                "wall": _stats(walls),
                "studio_overhead": _stats(overheads),
                "overhead_per_round_ms": round(statistics.median(overheads) / per, 2)
                if rounds
                else None,
                "us_per_answer_chunk": round(statistics.median(walls) * 1000 / answer, 1)
                if not rounds
                else None,
                "ttft": _stats(ttft),
                "yielded_lines": nlines,
                "provider_requests": reqs,
            }
        pr = cProfile.Profile()
        _timed_execute(loop_mod, "execute_tool", real, [], True)
        pr.enable()
        for _ in range(5):
            asyncio.run(drive(_script(3, answer_chunks = 400)))
        pr.disable()
        s = io.StringIO()
        st = pstats.Stats(pr, stream = s)
        st.sort_stats("cumulative").print_stats("studio[/\\\\]backend", 40)
        st.sort_stats("tottime").print_stats(20)
        out["profile_canned_head"] = s.getvalue()[:7000]
    finally:
        loop_mod.execute_tool = real
    return out


def arm_loop_gguf(n):
    import copy
    from core.inference.llama_cpp import LlamaCppBackend
    import core.inference.tools as tools_mod

    real = tools_mod.execute_tool

    def make_backend(streams):
        backend = LlamaCppBackend.__new__(LlamaCppBackend)
        backend._process = object()
        backend._healthy = True
        backend._port = 48847
        backend._api_key = None
        backend._effective_context_length = 32768
        backend._supports_reasoning = True
        backend._reasoning_always_on = False
        backend._reasoning_style = "enable_thinking"
        backend._supports_preserve_thinking = False

        @contextlib.contextmanager
        def fake_stream_with_retry(
            _client,
            _url,
            payload,
            _cancel_event,
            headers = None,
            first_token_deadline = None,
        ):
            stream = streams.pop(0) if streams else ["data: [DONE]\n"]
            yield type(
                "FakeResponse",
                (),
                {"status_code": 200, "chunks": stream, "close": lambda self: None},
            )()

        def fake_iter(
            response,
            _cancel_event,
            first_token_deadline = None,
        ):
            yield from response.chunks

        backend._stream_with_retry = fake_stream_with_retry
        backend._iter_text_cancellable = fake_iter
        backend._maybe_recover_from_mtp_crash = lambda *_a, **_k: False
        return backend

    out = {}
    try:
        for label, rounds, parallel, canned, answer in (
            ("answer_only_2000_chunks", 0, False, True, 2000),
            ("3_rounds_canned", 3, False, True, 200),
            ("3_rounds_real_tools", 3, False, False, 200),
            ("3_rounds_parallel2_real", 3, True, False, 200),
        ):
            walls, overheads, ttft, events = [], [], [], 0
            for i in range(n + 1):
                bodies = []
                _timed_execute(tools_mod, "execute_tool", real, bodies, canned)
                streams = [
                    [line + "\n" for line in turn]
                    for turn in _script(rounds, answer_chunks = answer, parallel = parallel)
                ]
                backend = make_backend(copy.deepcopy(streams))
                t0 = time.perf_counter()
                first, events = None, 0
                for ev in backend.generate_chat_completion_with_tools(
                    messages = [{"role": "user", "content": "go"}],
                    tools = _tool_defs(),
                    max_tool_iterations = rounds + 2,
                    permission_mode = "off",
                    session_id = SESSION,
                    enable_thinking = True,
                ):
                    events += 1
                    if (
                        first is None
                        and isinstance(ev, dict)
                        and ev.get("type") in ("content", "reasoning")
                    ):
                        first = (time.perf_counter() - t0) * 1000
                w = (time.perf_counter() - t0) * 1000
                if i == 0:
                    continue
                walls.append(w)
                overheads.append(w - sum(bodies))
                ttft.append(first)
            out[label] = {
                "wall": _stats(walls),
                "studio_overhead": _stats(overheads),
                "overhead_per_round_ms": round(statistics.median(overheads) / rounds, 2)
                if rounds
                else None,
                "us_per_answer_chunk": round(statistics.median(walls) * 1000 / answer, 1)
                if not rounds
                else None,
                "ttft": _stats(ttft),
                "events": events,
            }
        pr = cProfile.Profile()
        _timed_execute(tools_mod, "execute_tool", real, [], True)
        pr.enable()
        for _ in range(5):
            streams = [[line + "\n" for line in turn] for turn in _script(3, answer_chunks = 400)]
            list(
                make_backend(streams).generate_chat_completion_with_tools(
                    messages = [{"role": "user", "content": "go"}],
                    tools = _tool_defs(),
                    max_tool_iterations = 5,
                    permission_mode = "off",
                    session_id = SESSION,
                    enable_thinking = True,
                )
            )
        pr.disable()
        s = io.StringIO()
        st = pstats.Stats(pr, stream = s)
        st.sort_stats("cumulative").print_stats("studio[/\\\\]backend", 40)
        st.sort_stats("tottime").print_stats(20)
        out["profile_canned_head"] = s.getvalue()[:7000]
    finally:
        tools_mod.execute_tool = real
    return out


# ---------------------------------------------------------------- arm: mid-turn setting changes


class _FlipPermissions:
    """A live tool loop's permissions; flips to full access (the approval card's upgrade) at a chosen round."""

    def __init__(self):
        self.full = False

    def apply(self, permission_mode, bypass_permissions):
        return ("full", True) if self.full else (permission_mode, bypass_permissions)


def arm_midturn(_n):
    """Settings change between tool rounds of one turn, and the sandbox toggle changes between turns.

    Turn A (sandbox high): r0 reads an unregistered folder (denied); the folder is registered as a scan
    folder; r1 reads it (allowed, still read-only); the tool output limit is lowered; r2 prints 5,000 chars
    (truncated); the approval card's full-access upgrade fires; r3 runs without prompts, still sandboxed.
    Turn B (sandbox low): r0 runs on the host (reads and writes). Turn C (sandbox high again): r0 is
    sandboxed again. Every row must match on base and head.
    """
    import copy
    from core.inference.llama_cpp import LlamaCppBackend
    import core.inference.tools as tools_mod
    from storage import studio_db
    from utils import tool_result_limit

    base_dir = Path(os.environ.get("TOOLBENCH_MODEL_DIR") or Path.home())
    folder = base_dir / f"toolbench_models_{os.getpid()}"
    folder.mkdir(parents = True, exist_ok = True)
    (folder / "weights.txt").write_text("MODEL-BYTES-OK")
    target = str(folder / "weights.txt").replace("\\", "/")
    probe = (
        f"p = {target!r}\n"
        "try:\n print('READ', open(p).read())\nexcept OSError as e:\n print('READ-DENIED', type(e).__name__)\n"
        "try:\n open(p, 'a').write('x'); print('WRITE-OK')\nexcept OSError as e:\n print('WRITE-DENIED', type(e).__name__)\n"
    )
    perms = _FlipPermissions()
    real = tools_mod.execute_tool
    rows = []
    mutations = {}

    def execute(name, arguments, **kwargs):
        key = (rows_turn[0], sum(1 for r in rows if r["turn"] == rows_turn[0]))
        if key in mutations:
            mutations[key]()
        out = real(name, arguments, **kwargs)
        rec = tools_mod._last_tool_execution_record
        rows.append(
            {
                "turn": key[0],
                "round": key[1],
                "backend": getattr(rec, "backend", None),
                "effective": getattr(rec, "effective_mode", None),
                "out_len": len(str(out)),
                "out": str(out)[:160],
            }
        )
        return out

    def make_backend(codes):
        turns = []
        for r, code in enumerate(codes):
            raw = json.dumps({"code": code + f"\n#{r}"})
            turns.append(
                [
                    _sse({"role": "assistant"}),
                    _sse(
                        {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": f"m{r}",
                                    "type": "function",
                                    "function": {"name": "python", "arguments": raw},
                                }
                            ]
                        }
                    ),
                    _sse({}, finish = "tool_calls"),
                    "data: [DONE]",
                ]
            )
        turns.append([_sse({"content": "done"}), _sse({}, finish = "stop"), "data: [DONE]"])
        streams = [[line + "\n" for line in t] for t in turns]
        backend = LlamaCppBackend.__new__(LlamaCppBackend)
        backend._process, backend._healthy, backend._port, backend._api_key = (
            object(),
            True,
            48847,
            None,
        )
        backend._effective_context_length = 32768
        backend._supports_reasoning = backend._reasoning_always_on = (
            backend._supports_preserve_thinking
        ) = False
        backend._reasoning_style = "enable_thinking"

        @contextlib.contextmanager
        def fake_stream(
            _c,
            _u,
            payload,
            _e,
            headers = None,
            first_token_deadline = None,
        ):
            yield type(
                "R",
                (),
                {
                    "status_code": 200,
                    "chunks": streams.pop(0) if streams else ["data: [DONE]\n"],
                    "close": lambda self: None,
                },
            )()

        backend._stream_with_retry = fake_stream
        backend._iter_text_cancellable = lambda resp, _e, first_token_deadline = None: iter(
            resp.chunks
        )
        backend._maybe_recover_from_mtp_crash = lambda *_a, **_k: False
        return backend

    mutations[("A", 1)] = lambda: studio_db.add_scan_folder_with_status(str(folder))
    mutations[("A", 2)] = lambda: tool_result_limit.set_max_chars(tool_result_limit.MIN_MAX_CHARS)
    mutations[("A", 3)] = lambda: setattr(perms, "full", True)
    plan = [
        ("A", "high", [probe, probe, "print('Z' * 5000)", probe]),
        ("B", "low", [probe]),
        ("C", "high", [probe]),
    ]
    rows_turn = [None]
    tools_mod.execute_tool = execute
    errors = []
    try:
        for turn, level, codes in plan:
            rows_turn[0] = turn
            try:
                for _ in make_backend(codes).generate_chat_completion_with_tools(
                    messages = [{"role": "user", "content": "go"}],
                    tools = [copy.deepcopy(tools_mod.PYTHON_TOOL)],
                    max_tool_iterations = len(codes) + 2,
                    permission_mode = "off",
                    session_id = SESSION,
                    live_permissions = perms if turn == "A" else None,
                    deduplicate_tool_calls = False,
                    sandbox_level = level,
                ):
                    pass
            except Exception:  # noqa: BLE001
                errors.append(traceback.format_exc()[-1500:])
    finally:
        tools_mod.execute_tool = real
        with contextlib.suppress(Exception):
            for row in studio_db.list_scan_folders():
                if row.get("path") == str(folder):
                    studio_db.remove_scan_folder(row["id"])
        with contextlib.suppress(Exception):
            studio_db.upsert_app_settings({tool_result_limit.MAX_CHARS_SETTING_KEY: None})
        shutil.rmtree(folder, ignore_errors = True)

    def row(turn, rnd):
        return next(
            (r for r in rows if r["turn"] == turn and r["round"] == rnd),
            {"out": "", "effective": None},
        )

    checks = {
        "A0_unregistered_folder_not_readable": "READ-DENIED" in row("A", 0)["out"],
        "A1_scan_folder_added_mid_turn_is_readable": "MODEL-BYTES-OK" in row("A", 1)["out"],
        "A1_scan_folder_is_read_only": "WRITE-DENIED" in row("A", 1)["out"],
        "A2_lowered_output_limit_applies_next_call": 0 < row("A", 2).get("out_len", 0) < 4000,
        "A3_full_access_upgrade_keeps_os_sandbox": row("A", 3)["effective"] == "os_isolated",
        "B0_sandbox_low_runs_on_host": "WRITE-OK" in row("B", 0)["out"]
        and row("B", 0)["effective"] != "os_isolated",
        "C0_sandbox_high_again_is_isolated": row("C", 0)["effective"] == "os_isolated",
    }
    return {"rows": rows, "checks": checks, "errors": errors, "rows_ran": len(rows)}


# ---------------------------------------------------------------- child / orchestrator


def child(args):
    if not os.environ.get("UNSLOTH_STUDIO_HOME"):
        raise SystemExit("refusing to run against the default Studio home: set UNSLOTH_STUDIO_HOME")
    backend = os.environ.get("TOOLBENCH_BACKEND") or str(REPO / "studio" / "backend")
    sys.path.insert(0, backend)
    os.chdir(backend)
    import logging

    logging.disable(logging.INFO)
    with contextlib.suppress(Exception):
        import structlog
        structlog.configure(wrapper_class = structlog.make_filtering_bound_logger(logging.WARNING))
    t = time.perf_counter()
    from core.inference import tools

    res = {"backend_dir": backend, "import_ms": round((time.perf_counter() - t) * 1000, 1)}
    assert os.path.realpath(tools.__file__).startswith(os.path.realpath(backend)), tools.__file__
    for arm in args.arms.split(","):
        t = time.perf_counter()
        try:
            res[arm] = {
                "tools": lambda: arm_tools(args.calls, args.phase_out),
                "loop_ext": lambda: arm_loop_ext(max(3, args.calls // 4)),
                "loop_gguf": lambda: arm_loop_gguf(max(3, args.calls // 4)),
                "midturn": lambda: arm_midturn(args.calls),
            }[arm]()
        except Exception:  # noqa: BLE001
            res[arm] = {"error": traceback.format_exc()[-3000:]}
        res[arm]["_arm_s"] = round(time.perf_counter() - t, 1)
    Path(args.phase_out).write_text(json.dumps(res, indent = 1, default = str))


def _ensure_deps():
    probe = [
        sys.executable,
        "-c",
        "import sys; sys.path.insert(0, r'%s'); import core.inference.tools"
        % (REPO / "studio" / "backend"),
    ]
    if subprocess.run(probe, capture_output = True, timeout = 600).returncode == 0:
        return {"importable": True}
    failed = []
    for line in (
        (REPO / "studio" / "backend" / "requirements" / "studio.txt").read_text().splitlines()
    ):
        spec = line.split("#", 1)[0].strip()
        if spec and not spec.startswith("-"):
            r = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "--only-binary", ":all:", spec],
                capture_output = True,
                timeout = 900,
            )
            if r.returncode:
                failed.append(spec)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "pytest", "psutil", "python-multipart"],
        capture_output = True,
        timeout = 900,
    )
    r = subprocess.run(probe, capture_output = True, text = True, timeout = 600)
    return {"importable": r.returncode == 0, "failed": failed, "err": r.stderr[-800:]}


def _ensure_os_sandbox(env):
    """Linux: bwrap + AppArmor profile. Windows: pinned MXC (+ DACL tier on hosts without BaseContainer)."""
    info = {}
    if sys.platform == "linux":
        script = (
            "command -v bwrap >/dev/null || { apt-get install -y --no-install-recommends bubblewrap || "
            "{ apt-get update && apt-get install -y --no-install-recommends bubblewrap; }; }; "
            'if [ "$(cat /proc/sys/kernel/apparmor_restrict_unprivileged_userns 2>/dev/null || echo 0)" = 1 ]; then '
            "apt-get install -y --no-install-recommends apparmor-profiles || true; "
            "[ -f /etc/apparmor.d/bwrap-userns-restrict ] || install -m 644 "
            "/usr/share/apparmor/extra-profiles/bwrap-userns-restrict /etc/apparmor.d/ || true; "
            "apparmor_parser -r /etc/apparmor.d/bwrap-userns-restrict || true; fi"
        )
        pre = ["sudo", "-n"] if shutil.which("sudo") and os.geteuid() != 0 else []
        r = subprocess.run(pre + ["sh", "-c", script], capture_output = True, text = True, timeout = 900)
        info["bwrap_setup_rc"] = r.returncode
    elif sys.platform == "win32":
        sys.path.insert(0, str(REPO / "studio" / "backend"))
        os.environ["UNSLOTH_STUDIO_HOME"] = env["UNSLOTH_STUDIO_HOME"]
        from core.inference import mxc_runtime

        pkg = mxc_runtime._installed_package_root()
        inst = [sys.executable, str(REPO / "studio" / "install_mxc_prebuilt.py")]
        r = subprocess.run(
            inst + ["--install-dir", str(pkg)], capture_output = True, text = True, env = env, timeout = 900
        )
        info["install"] = (r.stdout + r.stderr)[-300:]
        steps = None
        with contextlib.suppress(Exception):
            steps = mxc_runtime.probe_host_prep_steps(package_root = pkg)
        info["host_prep_steps"] = steps
        probe = subprocess.run(
            [str(pkg / "wxc-exec.exe"), "--probe"],
            capture_output = True,
            text = True,
            cwd = str(pkg),
            timeout = 120,
        )
        tier = None
        with contextlib.suppress(Exception):
            tier = json.loads(probe.stdout).get("tier")
        info["tier"] = tier
        if tier != "base-container":
            r = subprocess.run(
                inst + ["--prepare-host", "--install-dir", str(pkg)],
                capture_output = True,
                text = True,
                env = env,
                timeout = 1200,
            )
            info["prepare_host"] = (r.stdout + r.stderr)[-300:]
            env["UNSLOTH_MXC_ALLOW_DACL_FALLBACK"] = "1"
            env["UNSLOTH_MXC_PERSISTENT_READ_GRANTS"] = "1"
    return info


def _run_child(arms, backend, n, work, tag, env):
    f = work / f"{tag}.json"
    e = dict(env, TOOLBENCH_BACKEND = backend, PYTHONPATH = backend)
    r = subprocess.run(
        [
            sys.executable,
            os.path.abspath(__file__),
            "--phase",
            "child",
            "--arms",
            arms,
            "--calls",
            str(n),
            "--phase-out",
            str(f),
        ],
        capture_output = True,
        text = True,
        errors = "replace",
        env = e,
        timeout = 5400,
    )
    data = json.loads(f.read_text()) if f.is_file() else {}
    data["rc"] = r.returncode
    if r.returncode or not data:
        data["stderr_tail"] = r.stderr[-3000:]
    return data


def _head_tree(work):
    patch_mod = Path(os.path.abspath(__file__)).with_name("tool_loop_patch.py")
    if not patch_mod.is_file():
        return None, "no tool_loop_patch.py"
    ns = {}
    exec(compile(patch_mod.read_text(), str(patch_mod), "exec"), ns)  # noqa: S102 - our own vendored file
    head = work / "head_tree"
    r = subprocess.run(
        ["git", "-C", str(REPO), "worktree", "add", "--detach", str(head), "HEAD"],
        capture_output = True,
        text = True,
    )
    if r.returncode:
        return None, "worktree: " + r.stderr[-400:]
    pf = work / "head.patch"
    pf.write_bytes(base64.b64decode(ns["PATCH_B64"]))
    r = subprocess.run(
        ["git", "-C", str(head), "apply", "--whitespace=nowarn", str(pf)],
        capture_output = True,
        text = True,
    )
    if r.returncode:
        return None, "apply: " + r.stderr[-600:]
    return str(head / "studio" / "backend"), hashlib.sha256(pf.read_bytes()).hexdigest()


def _wsl(work, n):
    """Run the tools + loop arms inside an imported WSL2 Ubuntu (bwrap backend) on the staged tree."""
    if sys.platform != "win32" or not shutil.which("wsl.exe"):
        return {"skipped": "not Windows"}
    st = subprocess.run(["wsl.exe", "--status"], capture_output = True, timeout = 60)
    if st.returncode:
        return {"skipped": "WSL not installed", "rc": st.returncode}
    import urllib.request

    arch = "arm64" if "ARM" in (os.environ.get("PROCESSOR_ARCHITECTURE") or "").upper() else "amd64"
    tar = work / "rootfs.tar.gz"
    url = f"https://cloud-images.ubuntu.com/wsl/releases/noble/current/ubuntu-noble-wsl-{arch}-wsl.rootfs.tar.gz"
    with (
        urllib.request.urlopen(
            urllib.request.Request(url, headers = {"User-Agent": "toolbench"}), timeout = 600
        ) as r,
        tar.open("wb") as fh,
    ):
        shutil.copyfileobj(r, fh)
    name = "ToolBench"
    r = subprocess.run(
        ["wsl.exe", "--import", name, str(work / "wsl"), str(tar), "--version", "2"],
        capture_output = True,
        timeout = 900,
    )
    if r.returncode:
        return {
            "import_rc": r.returncode,
            "out": (r.stdout + r.stderr).decode("utf-16-le", "replace")[-500:],
        }

    def winpath(p):
        p = str(p).replace("\\", "/")
        return f"/mnt/{p[0].lower()}{p[2:]}"

    src = winpath(REPO)
    me = winpath(os.path.abspath(__file__))
    patch = Path(os.path.abspath(__file__)).with_name("tool_loop_patch.py")
    script = f"""set -e
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq python3-venv python3-pip bubblewrap git curl >/dev/null
rm -rf /root/repo && mkdir -p /root/repo && cp -r {src}/studio /root/repo/studio
cd /root/repo && git init -q && git add -A >/dev/null && git -c user.email=b@b -c user.name=b commit -qm base >/dev/null
mkdir -p /root/jobs && cp {me} /root/jobs/ && {'cp ' + winpath(patch) + ' /root/jobs/' if patch.is_file() else 'true'}
python3 -m venv /root/v && /root/v/bin/pip install -q -r studio/backend/requirements/studio.txt pytest psutil python-multipart
/root/v/bin/python /root/jobs/tool_loop_bench.py --calls {n} --out /root/wsl.json --no-sandbox-setup
cat /root/wsl.json
"""
    r = subprocess.run(
        ["wsl.exe", "-d", name, "-u", "root", "--", "bash", "-lc", script],
        capture_output = True,
        timeout = 5400,
    )
    text = r.stdout.decode("utf-8", "replace")
    subprocess.run(["wsl.exe", "--unregister", name], capture_output = True, timeout = 300)
    start = text.find("{\n")
    try:
        return {"rc": r.returncode, "result": json.loads(text[start:])}
    except Exception:  # noqa: BLE001
        return {
            "rc": r.returncode,
            "stdout_tail": text[-3000:],
            "stderr_tail": r.stderr.decode("utf-8", "replace")[-2000:],
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tiny", action = "store_true")
    p.add_argument("--out", default = "job_tool_loop_bench.json")
    p.add_argument("--calls", type = int, default = None)
    p.add_argument("--arms", default = "tools,loop_ext,loop_gguf,midturn")
    p.add_argument("--rounds", type = int, default = 3)
    p.add_argument("--phase", default = None)
    p.add_argument("--phase-out", default = None)
    p.add_argument("--wsl", action = "store_true")
    p.add_argument("--no-sandbox-setup", action = "store_true")
    args = p.parse_args()
    if args.calls is None:
        args.calls = 12 if args.tiny else 20
    if args.phase == "child":
        return child(args)

    out_path = Path(args.out).resolve()
    result = {"job": "tool_loop_bench", "passed": False, "finished": False}

    def flush():
        out_path.write_text(json.dumps(result, indent = 1, default = str))

    result["host"] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpus": os.cpu_count(),
        "python": sys.version.split()[0],
        "image": os.environ.get("ImageOS"),
        "image_version": os.environ.get("ImageVersion"),
    }
    work = Path(tempfile.mkdtemp(prefix = "toolbench-"))
    env = dict(os.environ, UNSLOTH_STUDIO_HOME = str(work / "home"), PYTHONIOENCODING = "utf-8")
    result["deps"] = _ensure_deps()
    if not args.no_sandbox_setup:
        try:
            result["sandbox_setup"] = _ensure_os_sandbox(env)
        except Exception:  # noqa: BLE001
            result["sandbox_setup"] = {"error": traceback.format_exc()[-1500:]}
    flush()
    base = str(REPO / "studio" / "backend")
    head, patch_id = _head_tree(work)
    result["head"] = {"backend": head, "patch": patch_id}
    rounds = []
    for i in range(args.rounds if head else 1):
        row = {}
        order = ("base", "head") if i % 2 == 0 else ("head", "base")
        for arm in order:
            if arm == "head" and not head:
                continue
            row[arm] = _run_child(
                args.arms, base if arm == "base" else head, args.calls, work, f"{arm}_{i}", env
            )
            flush()
        rounds.append(row)
        result["rounds"] = rounds
        flush()
    if args.wsl:
        try:
            result["wsl"] = _wsl(work, args.calls)
        except Exception:  # noqa: BLE001
            result["wsl"] = {"error": traceback.format_exc()[-1500:]}
    result["summary"] = summarize(rounds)
    ok = [r.get("base", {}).get("tools", {}) for r in rounds]
    result["passed"] = any(
        any(
            k.startswith("python_print") and "None" not in str(v.get("backends"))
            for k, v in t.items()
            if isinstance(v, dict)
        )
        for t in ok
    )
    result["finished"] = True
    flush()
    with contextlib.suppress(Exception):
        subprocess.run(
            ["git", "-C", str(REPO), "worktree", "remove", "--force", str(work / "head_tree")],
            capture_output = True,
        )
    print("JOB_RESULT " + json.dumps(result["summary"], default = str)[:20000])
    return 0


def summarize(rounds):
    """Median over rounds of each arm's headline metric, per side."""
    summ = {}
    for side in ("base", "head"):
        acc = {}
        for row in rounds:
            data = row.get(side) or {}
            for key, val in (data.get("tools") or {}).items():
                if isinstance(val, dict) and val.get("warm"):
                    acc.setdefault(f"tools|{key}", []).append(val["warm"]["median_ms"])
                    acc.setdefault(f"backend|{key}", [str(val.get("backends"))])
            for arm in ("loop_ext", "loop_gguf"):
                for key, val in (data.get(arm) or {}).items():
                    if isinstance(val, dict) and val.get("studio_overhead"):
                        acc.setdefault(f"{arm}|{key}|overhead_ms", []).append(
                            val["studio_overhead"]["median_ms"]
                        )
                        if val.get("us_per_answer_chunk") is not None:
                            acc.setdefault(f"{arm}|{key}|us_per_chunk", []).append(
                                val["us_per_answer_chunk"]
                            )
        summ[side] = {
            k: (statistics.median(v) if v and isinstance(v[0], (int, float)) else v[0])
            for k, v in acc.items()
        }
        summ[side + "_rounds"] = {
            k: v for k, v in acc.items() if v and isinstance(v[0], (int, float))
        }
    return summ


if __name__ == "__main__":
    raise SystemExit(main())
