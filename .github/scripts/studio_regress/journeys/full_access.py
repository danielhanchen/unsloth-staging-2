"""Journey 5: Full Access on, web search off, code execution on, with the whole run inside
studio_regress.isolation (no network but loopback). Model: unsloth/Qwen3.5-0.8B-MTP-GGUF
(tool-capable; gemma-3-270m reports supports_tools=false).

The python tool is asked to run a fixed probe that WRITES its findings to a file in its own
working dir, so the evidence does not depend on how the model paraphrases tool output:
  internet  urlopen("https://example.com") and a raw TCP connect to 1.1.1.1:53 must fail,
  loopback  a local socket pair must work,
  sentinel  the host file named in STUDIO_REGRESS_SENTINEL must be unreadable,
  artifact  the probe file itself proves permitted writes work.
The server log must show `execute_tool: name=python` and never `name=web_search`.
Outside isolation (STUDIO_REGRESS_ISOLATION unset) the network / sentinel checks cannot hold, so
the step fails with that reason instead of passing on a non-isolated host."""

from __future__ import annotations

import json
import os
import re
import time

from .. import timeouts
from ..contract import Journey, Step, StepFailed
from . import _b_common as b
from . import _b_ui as ui
from .train_responses_only import log_marks, log_since

PROBE_FILE = "regress_isolation_probe.json"


def probe_code(sentinel):
    return f"""import json, socket, urllib.request
r = {{}}
try:
    urllib.request.urlopen("https://example.com", timeout=5); r["internet_http"] = "reachable"
except Exception as e:
    r["internet_http"] = "blocked: " + type(e).__name__
try:
    socket.create_connection(("1.1.1.1", 53), timeout=5).close(); r["internet_tcp"] = "reachable"
except OSError as e:
    r["internet_tcp"] = "blocked: " + type(e).__name__
s = socket.socket(); s.bind(("127.0.0.1", 0)); s.listen()
socket.create_connection(s.getsockname()).close(); r["loopback"] = "ok"
try:
    open({sentinel!r}).read(); r["sentinel"] = "readable"
except OSError as e:
    r["sentinel"] = "unreadable: " + type(e).__name__
open("{PROBE_FILE}", "w").write(json.dumps(r))
print(json.dumps(r))"""


def isolation_mode():
    return os.environ.get("STUDIO_REGRESS_ISOLATION")


async def owner_full_access(ctx):
    st = await b.get(ctx, "/api/auth/status")
    fa = st.get("full_access")
    if fa is False:
        raise StepFailed(f"Full Access is not permitted on this Studio (managed accounts?): {st}")
    return {"full_access_permitted": fa, "isolation": isolation_mode() or "none"}


async def load_tool_model(ctx):
    repo, variant = b.model(ctx, "gguf_tools", b.GGUF_QWEN35_08B)
    # Setup, not the subject (the tool run inside isolation is): keep an identical load.
    st, reused = await b.ensure_gguf(ctx, repo, variant, 8192)
    await b.goto(ctx, "/chat")
    await ui.need(ctx.page.get_by_role("textbox", name = "Message input"), "chat composer", 60_000)
    return {
        "model": repo,
        "context_length": b.effective_ctx(st),
        "supports_tools": st.get("supports_tools"),
        "_reused_load": reused,
    }


async def _toggle(page, noun, want_on):
    """Composer toggles carry their state in the NAME ("Enable code execution" while off,
    "Disable code execution" while on), not in aria-pressed. Returns (on, label)."""
    loc = page.get_by_role("button", name = re.compile(rf"^(Enable|Disable) {noun}$", re.I))
    btn = await ui.need(loc, f"{noun} toggle")
    label = await btn.get_attribute("aria-label") or ""
    if label.lower().startswith("disable") != want_on:
        await btn.click()
        # Was a blind 300 ms: wait (bounded) for the toggle's name to flip to the new state.
        want = "Disable" if want_on else "Enable"
        try:
            await page.get_by_role(
                "button", name = re.compile(rf"^{want} {noun}$", re.I)
            ).first.wait_for(state = "visible", timeout = timeouts.scaled_ms(3_000))
        except Exception:
            pass  # read back below; the assertion in ui_toggles reports the state it found
        btn = await ui.need(loc, f"{noun} toggle")
        label = await btn.get_attribute("aria-label") or ""
    return label.lower().startswith("disable"), label


async def ui_toggles(ctx):
    page = ctx.page
    code_state, code_label = await _toggle(page, "code execution", True)
    search_state, search_label = await _toggle(page, "web search", False)
    perm = await ui.need(
        page.get_by_role("button", name = "Permission level for tool calls"), "permission menu"
    )
    perm = page.locator(
        f'[id="{await perm.get_attribute("id")}"]'
    )  # its name changes with the level
    await perm.click()
    item = page.get_by_role("menuitemradio", name = re.compile("full access", re.I))
    if not await item.count():
        item = page.get_by_role("menuitem", name = re.compile("full access", re.I))
    picked = False
    if await item.count():
        await item.first.click()
        picked = True
        ok = page.get_by_role("button", name = "I understand")  # "Enable Full access?" confirm
        try:
            await ok.first.wait_for(timeout = 5_000)
            await ok.first.click()
        except Exception:
            pass
    else:
        await page.keyboard.press("Escape")
    if picked:  # was a blind 500 ms: wait (bounded) for the control to read the new level
        try:
            await page.wait_for_function(
                "el => /full/i.test(el.innerText || '')",
                arg = await perm.element_handle(),
                timeout = timeouts.scaled_ms(3_000),
            )
        except Exception:
            pass  # the assertion below reports the label it found
    else:
        await page.wait_for_timeout(500)  # menu closing: nothing observable to wait on
    label = (await perm.inner_text()).strip()
    if picked and "full" not in label.lower():
        raise StepFailed(f"Full access chosen but the permission control reads {label!r}")
    if not code_state or search_state:
        raise StepFailed(
            f"toggles: code={code_state} ({code_label}) search={search_state} ({search_label})"
        )
    return {
        "code_execution": code_state,
        "web_search": search_state,
        "full_access_item": picked,
        "permission_label": label,
    }


async def python_probe(ctx):
    sentinel = os.environ.get("STUDIO_REGRESS_SENTINEL", "/nonexistent/sentinel")
    marks = log_marks(ctx)
    prompt = (
        "Use the python tool to run exactly this code, unchanged, then reply DONE.\n```python\n"
        + probe_code(sentinel)
        + "\n```"
    )
    t0 = time.time()
    found = None
    for attempt in range(3):  # a 0.8B model occasionally answers instead of calling the tool
        code, body = await b.chat(
            ctx,
            [{"role": "user", "content": prompt}],
            max_tokens = 1024,
            enable_tools = True,
            enabled_tools = ["python", "terminal"],
            permission_mode = "full",
            bypass_permissions = True,
            session_id = f"regress-iso-{attempt}",
            timeout = timeouts.inner(600, ctx.state),
        )
        if code != 200:
            raise StepFailed(f"chat with tools: HTTP {code} {str(body)[:300]}")
        home = b.studio_home(ctx)
        cands = (
            [p for p in (home / "sandbox").rglob(PROBE_FILE) if p.stat().st_mtime >= t0 - 1]
            if home and (home / "sandbox").is_dir()
            else []
        )
        if not cands and home:
            cands = [p for p in home.rglob(PROBE_FILE) if p.stat().st_mtime >= t0 - 1]
        if cands:
            found = max(cands, key = lambda p: p.stat().st_mtime)
            break
    log = log_since(ctx, marks)
    facts = {
        "attempts": attempt + 1,
        "python_calls": log.count("execute_tool: name=python"),
        "web_search_calls": log.count("execute_tool: name=web_search"),
        "isolation": isolation_mode() or "none",
    }
    if not found:
        raise StepFailed(f"the python tool never wrote {PROBE_FILE}: {facts}")
    r = json.loads(found.read_text())
    facts.update(r)
    if facts["web_search_calls"]:
        raise StepFailed(f"web_search executed although disabled: {facts}")
    if r.get("loopback") != "ok":
        raise StepFailed(f"loopback broken inside the tool: {r}")
    mode = isolation_mode()
    if mode in ("bwrap", "docker"):
        leaks = [k for k in ("internet_http", "internet_tcp") if r.get(k) == "reachable"]
        if leaks or r.get("sentinel") == "readable":
            raise StepFailed(f"isolation={mode} leaked: {r}")
    elif mode == "weak":
        if r.get("internet_http") == "reachable":
            raise StepFailed(
                f"weak isolation: proxy-honouring HTTP still reached the internet: {r}"
            )
        facts["note"] = "weak mode: raw TCP and host files are NOT isolated"
    else:
        raise StepFailed(f"not running inside studio_regress.isolation; probe says {r}")
    return facts


# run.py runs this journey in a nested run.py inside studio_regress.isolation, per side.
ISOLATED = True

JOURNEY = Journey(
    name = "full_access_isolated",
    tier = "model",
    needs = ("gguf_tools",),
    routes = ("/chat",),
    steps = (
        Step("owner_full_access", owner_full_access, shot = False),
        Step("load_tool_model", load_tool_model, timeout_s = 600),
        Step("ui_toggles", ui_toggles),
        Step("python_probe", python_probe, shot = False, timeout_s = 1500),
    ),
)
