# Simulation matrix for unslothai/unsloth PR #10810.
#
# Runs unchanged against BOTH the PR head and its merge base. Behaviour that differs between the
# two is *recorded* into a JSON report (compared afterwards); behaviour that must hold on both
# revisions is *asserted* here, so a regression fails loudly on whichever tree it appears in.
#
# Copy into <tree>/studio/backend/tests/ and run with cwd <tree>/studio/backend.
# Report path comes from $PR10810_SIM_OUT.

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)
# The sibling suites are plain modules in this directory; borrow their harnesses.
_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from core.inference import tool_loop_controller as tlc
from core.inference.tool_loop_controller import ToolLoopController, canonical_tool_call_key
from core.inference.safetensors_agentic import run_safetensors_tool_loop

# Helpers borrowed from the suites already in this directory rather than re-made.
from test_llama_cpp_tool_loop import (  # noqa: E402
    _backend_and_payloads,
    _done,
    _run_tool_loop,
    _sse,
    _tool_call_sse,
)
from test_safetensors_tool_loop import FakeExecuteTool, _collect_events  # noqa: E402

# "head" iff the PR's constant exists. Nothing else in the file branches on the revision.
REVISION = "head" if hasattr(tlc, "_WORKSPACE_TOOLS") else "base"
_REPORT: dict[str, object] = {"revision": REVISION}


def _record(name: str, value) -> None:
    _REPORT[name] = value


def teardown_module(module):  # noqa: ARG001
    out = os.environ.get("PR10810_SIM_OUT")
    if out:
        Path(out).write_text(json.dumps(_REPORT, indent = 2, sort_keys = True))


# ---------------------------------------------------------------------------
# shared fixtures for the controller-level traces
# ---------------------------------------------------------------------------

def _tool(name: str) -> dict:
    return {"type": "function", "function": {"name": name}}


def _call(name: str, args, call_id: str = "call_0") -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(args) if isinstance(args, dict) else args,
        },
    }


ALL = [
    _tool("terminal"),
    _tool("python"),
    _tool("edit_file"),
    _tool("web_search"),
    _tool("render_html"),
]

R = _call("terminal", {"command": "cat notes.txt"})
R2 = _call("terminal", {"command": "cat other.txt"})
E = _call("edit_file", {"path": "notes.txt", "edits": [{"old_string": "a", "new_string": "b"}]})
P = _call("python", {"code": "open('notes.txt','w').write('x')"})
S = _call("web_search", {"query": "gpu prices"})
H = _call("render_html", {"html": "<p>hi</p>"})

OK = "done"
ERR = "Error: wrote the file then exited 1"


def _drive(controller: ToolLoopController, trace):
    """Run (call, result) pairs through the controller; return the action of each step."""
    actions = []
    for call, result in trace:
        decision = controller.prepare_call(call)
        actions.append(decision.action)
        if decision.action == "execute":
            controller.record_result(decision, result)
        else:
            controller.record_noop(decision)
    return actions


# ---------------------------------------------------------------------------
# Group 1 - controller decision traces
# ---------------------------------------------------------------------------

# name -> (trace, expected action of the FINAL step on head, expected on base)
_TRACES = {
    # The bug in #10792 and its immediate neighbours.
    "read_edit_read":              ([(R, OK), (E, OK), (R, OK)],  "execute",   "duplicate"),
    "read_failing_edit_read":      ([(R, OK), (E, ERR), (R, OK)], "execute",   "duplicate"),
    "read_python_read":            ([(R, OK), (P, OK), (R, OK)],  "execute",   "duplicate"),
    "edit_read_edit":              ([(E, OK), (R, OK), (E, OK)],  "execute",   "duplicate"),
    # Guards that must SURVIVE the change.
    "read_read":                   ([(R, OK), (R, OK)],           "duplicate", "duplicate"),
    "read_search_read":            ([(R, OK), (S, OK), (R, OK)],  "duplicate", "duplicate"),
    "search_edit_search":          ([(S, OK), (E, OK), (S, OK)],  "duplicate", "duplicate"),
    "edit_edit":                   ([(E, OK), (E, OK)],           "duplicate", "duplicate"),
    "read_edit_read_read":         ([(R, OK), (E, OK), (R, OK), (R, OK)],
                                    "duplicate", "duplicate"),
    "oneshot_edit_oneshot":        ([(H, OK), (E, OK), (H, OK)],
                                    "render_html_repeat", "render_html_repeat"),
    # Deliberately broader than the issue asked for: an unrelated read also clears.
    "read_otherread_read":         ([(R, OK), (R2, OK), (R, OK)], "execute",   "duplicate"),
    # A failure never entered the ledger before the PR either.
    "failed_read_read":            ([(R, ERR), (R, OK)],          "execute",   "execute"),
}


@pytest.mark.parametrize("name", sorted(_TRACES))
def test_controller_trace(name):
    trace, on_head, on_base = _TRACES[name]
    actions = _drive(ToolLoopController(tools = ALL), trace)
    _record(f"trace.{name}", actions)
    expected = on_head if REVISION == "head" else on_base
    assert actions[-1] == expected, f"{name}: {actions}"


def test_canonical_key_ignores_argument_order():
    """Argument spelling must not let a repeat slip past the guard on either revision."""
    controller = ToolLoopController(tools = ALL)
    first = _call("terminal", {"command": "ls", "cwd": "/tmp"})
    reordered = _call("terminal", {"cwd": "/tmp", "command": "ls"})
    controller.record_result(controller.prepare_call(first), OK)
    action = controller.prepare_call(reordered).action
    _record("canonical_key.reordered_args", action)
    assert action == "duplicate"
    assert canonical_tool_call_key("terminal", {"a": 1, "b": 2}) == canonical_tool_call_key(
        "terminal", {"b": 2, "a": 1}
    )


def test_mcp_named_tool_is_not_mistaken_for_a_workspace_tool():
    """`mcp__srv__python` must not evict workspace keys via key.partition(':')."""
    tools = ALL + [_tool("mcp__srv__python")]
    controller = ToolLoopController(tools = tools)
    mcp_call = _call("mcp__srv__python", {"code": "1"})
    controller.record_result(controller.prepare_call(R), OK)
    controller.record_result(controller.prepare_call(mcp_call), OK)
    action = controller.prepare_call(R).action
    _record("mcp.namespaced_tool_does_not_evict", action)
    # An MCP tool is not a workspace tool on either revision, so the read stays suppressed.
    assert action == "duplicate"


def test_a_colon_bearing_tool_name_cannot_reach_record_result():
    """The one way partition(':') could misfire, shown to be unreachable when tools are gated."""
    controller = ToolLoopController(tools = ALL)
    hallucinated = _call("terminal:remote", {"command": "cat notes.txt"})
    decision = controller.prepare_call(hallucinated)
    _record("colon_name.action", decision.action)
    # Not an advertised tool, so it is refused before it can ever be recorded.
    assert decision.action == "disabled"


@pytest.mark.parametrize("limit", [1, 2, 3])
def test_duplicate_noop_limit_still_latches(limit):
    """Repeating one call with nothing in between must still end the reply."""
    controller = ToolLoopController(tools = ALL, duplicate_noop_limit = limit)
    controller.record_result(controller.prepare_call(R), OK)
    for _ in range(limit):
        controller.record_noop(controller.prepare_call(R))
    _record(f"latch.limit_{limit}", controller.force_final_answer)
    assert controller.force_final_answer
    assert controller.active_tools() == []


def test_latch_is_never_unset_by_a_later_workspace_execution():
    """Codex's 'stale latch' item: once latched, no tool can run, so it cannot go stale."""
    controller = ToolLoopController(tools = ALL)
    controller.record_result(controller.prepare_call(R), OK)
    controller.record_noop(controller.prepare_call(R))
    controller.record_noop(controller.prepare_call(R))
    assert controller.force_final_answer
    edit = controller.prepare_call(E)
    controller.record_result(edit, OK)
    _record("latch.survives_workspace_execution", controller.force_final_answer)
    assert controller.force_final_answer
    assert controller.active_tools() == []


def test_one_shot_completion_survives_workspace_eviction():
    controller = ToolLoopController(tools = ALL)
    controller.record_result(controller.prepare_call(H), OK)
    controller.record_result(controller.prepare_call(E), OK)
    names = [tool["function"]["name"] for tool in controller.active_tools()]
    _record("oneshot.active_tools_after_eviction", names)
    assert "render_html" not in names


def test_disabled_and_denied_calls_do_not_evict():
    """A call that never executes must not clear the ledger on either revision."""
    controller = ToolLoopController(tools = [_tool("terminal"), _tool("web_search")])
    controller.record_result(controller.prepare_call(R), OK)
    # edit_file is not advertised here, so it is a `disabled` no-op, not an execution.
    denied = controller.prepare_call(E)
    assert denied.action == "disabled"
    controller.record_noop(denied)
    action = controller.prepare_call(R).action
    _record("no_eviction_without_execution", action)
    assert action == "duplicate"


# ---------------------------------------------------------------------------
# Group 2 - termination bounds (the runaway question)
# ---------------------------------------------------------------------------

_CAP = 100


def _alternating_execution_count(calls, results):
    """Feed a cycle of calls forever; count executions until the controller forces a final answer."""
    controller = ToolLoopController(tools = ALL)
    executed = 0
    for i in range(_CAP):
        call = calls[i % len(calls)]
        decision = controller.prepare_call(call)
        if decision.action == "execute":
            controller.record_result(decision, results[i % len(results)])
            executed += 1
        else:
            controller.record_noop(decision)
        if controller.force_final_answer:
            return executed, "force_final_answer", i + 1
    return executed, "HARNESS_CAP", _CAP


@pytest.mark.parametrize(
    "name,calls,results",
    [
        ("two_reads",            [R, R2],  [OK, OK]),
        ("read_and_edit",        [R, E],   [OK, OK]),
        ("read_and_failing_edit", [R, E],  [OK, ERR]),
        ("read_and_search",      [R, S],   [OK, OK]),
        ("single_read",          [R],      [OK]),
    ],
)
def test_alternating_cycle_termination(name, calls, results):
    executed, reason, steps = _alternating_execution_count(calls, results)
    _record(f"runaway.{name}", {"executed": executed, "reason": reason, "steps": steps})
    # The controller alone is NOT the only bound (the loops cap iterations), so reaching the
    # harness cap here is information, not automatically a failure. Recorded either way.
    assert executed <= _CAP


# ---------------------------------------------------------------------------
# Group 3 - safetensors loop, end to end
# ---------------------------------------------------------------------------

_READ_TXT = '<tool_call>{"name":"terminal","arguments":{"command":"cat notes.txt"}}</tool_call>'
_READ2_TXT = '<tool_call>{"name":"terminal","arguments":{"command":"cat other.txt"}}</tool_call>'
_EDIT_TXT = '<tool_call>{"name":"edit_file","arguments":{"path":"notes.txt","edits":[]}}</tool_call>'
_LOOP_TOOLS = [
    {"type": "function", "function": {"name": name}} for name in ("terminal", "edit_file")
]


def _safetensors_calls(turn_texts, exec_results, *, max_tool_iterations = 5):
    turns = iter(turn_texts)

    def single_turn(messages, **kwargs):  # noqa: ARG001
        yield next(turns, "Done.")

    executor = FakeExecuteTool(list(exec_results))
    _collect_events(
        run_safetensors_tool_loop(
            single_turn = single_turn,
            messages = [{"role": "user", "content": "read, edit, verify"}],
            tools = _LOOP_TOOLS,
            execute_tool = executor,
            max_tool_iterations = max_tool_iterations,
        ),
        max_events = 4000,
    )
    return [name for name, _ in executor.calls]


def test_safetensors_read_edit_read_one_turn():
    names = _safetensors_calls(
        [_READ_TXT + _EDIT_TXT + _READ_TXT, "Done."], ["v1", "Edited", "v2"]
    )
    _record("safetensors.read_edit_read_one_turn", names)
    expected = ["terminal", "edit_file", "terminal"] if REVISION == "head" else [
        "terminal", "edit_file"
    ]
    assert names == expected


def test_safetensors_read_edit_read_across_turns():
    names = _safetensors_calls(
        [_READ_TXT, _EDIT_TXT, _READ_TXT, "Done."], ["v1", "Edited", "v2"]
    )
    _record("safetensors.read_edit_read_across_turns", names)
    expected = ["terminal", "edit_file", "terminal"] if REVISION == "head" else [
        "terminal", "edit_file"
    ]
    assert names == expected


def test_safetensors_eight_reads_do_not_crowd_out_a_later_edit():
    names = _safetensors_calls(
        [_READ_TXT * 8 + _EDIT_TXT + _READ_TXT, "Done."], ["v1", "Edited", "v2"]
    )
    _record("safetensors.eight_reads_then_edit", names)
    if REVISION == "head":
        assert names == ["terminal", "edit_file", "terminal"]


def test_safetensors_batch_cap_still_holds():
    """Nine DISTINCT calls must still be capped at _MAX_TOOL_CALLS_PER_TURN on both revisions."""
    distinct = "".join(
        '<tool_call>{"name":"terminal","arguments":{"command":"cat f%d"}}</tool_call>' % i
        for i in range(12)
    )
    names = _safetensors_calls([distinct, "Done."], ["r"] * 12)
    _record("safetensors.batch_cap", len(names))
    assert len(names) <= 8, f"batch cap breached: {len(names)}"


def test_safetensors_alternating_reads_terminate():
    """The runaway question, on the real loop: alternate two reads for far more turns than allowed."""
    turns = [_READ_TXT if i % 2 == 0 else _READ2_TXT for i in range(60)]
    names = _safetensors_calls(turns, ["r"] * 200, max_tool_iterations = 25)
    _record("safetensors.alternating_reads", {"executed": len(names)})
    # Must terminate well inside the harness budget, whatever the duplicate guard decides.
    assert len(names) <= 60, f"did not terminate: {len(names)} calls"


def test_safetensors_alternating_read_and_failing_edit_terminate():
    turns = [_READ_TXT if i % 2 == 0 else _EDIT_TXT for i in range(60)]
    results = ["r" if i % 2 == 0 else ERR for i in range(200)]
    names = _safetensors_calls(turns, results, max_tool_iterations = 25)
    _record("safetensors.alternating_read_failing_edit", {"executed": len(names)})
    assert len(names) <= 60, f"did not terminate: {len(names)} calls"


# ---------------------------------------------------------------------------
# Group 4 - llama.cpp / GGUF loop, textual fallback AND structured tool calls
# ---------------------------------------------------------------------------

def _gguf_calls(monkeypatch, streams, *, max_tool_iterations = 5, results = None):
    backend, _ = _backend_and_payloads(monkeypatch, streams)
    calls: list[str] = []
    supply = iter(results or [])

    def execute(name, arguments, **kwargs):  # noqa: ARG001
        calls.append(name)
        return next(supply, "OK")

    monkeypatch.setattr("core.inference.tools.execute_tool", execute)
    _run_tool_loop(
        backend,
        [{"role": "user", "content": "read, edit, verify"}],
        _LOOP_TOOLS,
        max_tool_iterations = max_tool_iterations,
    )
    return calls


def test_gguf_textual_read_edit_read_one_turn(monkeypatch):
    names = _gguf_calls(
        monkeypatch,
        [[_sse({"content": _READ_TXT + _EDIT_TXT + _READ_TXT}), _done()],
         [_sse({"content": "Done."}), _done()]],
        results = ["v1", "Edited", "v2"],
    )
    _record("gguf.textual_read_edit_read", names)
    expected = ["terminal", "edit_file", "terminal"] if REVISION == "head" else [
        "terminal", "edit_file"
    ]
    assert names == expected


def test_gguf_structured_read_edit_read_across_turns(monkeypatch):
    """Structured delta.tool_calls bypass the changed prefilter entirely; the controller carries it."""
    names = _gguf_calls(
        monkeypatch,
        [
            [_tool_call_sse("terminal", {"command": "cat notes.txt"}, "c1"), _done()],
            [_tool_call_sse("edit_file", {"path": "notes.txt", "edits": []}, "c2"), _done()],
            [_tool_call_sse("terminal", {"command": "cat notes.txt"}, "c3"), _done()],
            [_sse({"content": "Done."}), _done()],
        ],
        results = ["v1", "Edited", "v2"],
    )
    _record("gguf.structured_read_edit_read", names)
    expected = ["terminal", "edit_file", "terminal"] if REVISION == "head" else [
        "terminal", "edit_file"
    ]
    assert names == expected


def test_gguf_structured_immediate_repeat_is_still_suppressed(monkeypatch):
    """Immediate-repeat safety must not depend on the textual prefilter."""
    names = _gguf_calls(
        monkeypatch,
        [
            [_tool_call_sse("terminal", {"command": "cat notes.txt"}, "c1"), _done()],
            [_tool_call_sse("terminal", {"command": "cat notes.txt"}, "c2"), _done()],
            [_sse({"content": "Done."}), _done()],
        ],
        results = ["v1", "v1"],
    )
    _record("gguf.structured_immediate_repeat", names)
    assert names == ["terminal"], f"an identical back-to-back call executed twice: {names}"


def test_gguf_textual_eight_reads_do_not_crowd_out_a_later_edit(monkeypatch):
    names = _gguf_calls(
        monkeypatch,
        [[_sse({"content": _READ_TXT * 8 + _EDIT_TXT + _READ_TXT}), _done()],
         [_sse({"content": "Done."}), _done()]],
        results = ["v1", "Edited", "v2"],
    )
    _record("gguf.textual_eight_reads_then_edit", names)
    if REVISION == "head":
        assert names == ["terminal", "edit_file", "terminal"]


def test_gguf_textual_batch_cap_still_holds(monkeypatch):
    distinct = "".join(
        '<tool_call>{"name":"terminal","arguments":{"command":"cat f%d"}}</tool_call>' % i
        for i in range(12)
    )
    names = _gguf_calls(
        monkeypatch,
        [[_sse({"content": distinct}), _done()], [_sse({"content": "Done."}), _done()]],
        results = ["r"] * 12,
    )
    _record("gguf.textual_batch_cap", len(names))
    assert len(names) <= 8, f"batch cap breached: {len(names)}"


def test_gguf_alternating_structured_reads_terminate(monkeypatch):
    streams = []
    for i in range(60):
        command = "cat notes.txt" if i % 2 == 0 else "cat other.txt"
        streams.append([_tool_call_sse("terminal", {"command": command}, f"c{i}"), _done()])
    streams.append([_sse({"content": "Done."}), _done()])
    names = _gguf_calls(monkeypatch, streams, max_tool_iterations = 25, results = ["r"] * 200)
    _record("gguf.alternating_structured_reads", {"executed": len(names)})
    assert len(names) <= 60, f"did not terminate: {len(names)} calls"
