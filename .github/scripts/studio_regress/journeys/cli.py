"""Journey 12: `unsloth run` and `unsloth start` (CLI surfaces, no browser screenshots).

Own fresh-state home + port per side. Steps:
  s01_run_ready      `unsloth run --model <tiny GGUF> --context-length 2048 --password P --yes`
                     comes up: /api/health answers as Studio and the model is loaded
  s02_completion     OpenAI-style /v1/chat/completions (temperature 0, seed 0, 8 tokens) -> 200
                     with content (content itself is not compared: sampling details are not
                     the subject)
  s03_port_in_use    a second `unsloth run` (fresh home) on the SAME port either exits non-zero
                     or moves to the next free port and announces it; the outcome is a fact
  s04_start_stub     `unsloth start claude --no-serve` against the running server with a stub
                     `claude` on PATH: the stub records argv/env; assert ANTHROPIC_BASE_URL is
                     this loopback server, a token is handed over, and the stub's exit code
                     (7) propagates
  s05_sigint_cleanup SIGINT to `unsloth run`: it exits and leaves no child (llama-server) behind
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path

from studio_regress import engine, timeouts
from studio_regress.contract import Journey, Step, StepFailed, StepUnreachable

READY_TIMEOUT_S = 600   # idle-host cap; `unsloth run` imports the full stack before binding. Scaled at
                        # runtime by timeouts.inner (host load, and never past the step budget)
STUB_EXIT = 7
PASSWORD = "CliJourney-Pass-2026"

STUB = r"""#!/usr/bin/env python3
import json, os, sys
keep = {k: v for k, v in os.environ.items() if k.startswith(("ANTHROPIC_", "OPENAI_", "CLAUDE_"))}
open(os.environ["SR_STUB_OUT"], "w").write(json.dumps({"argv": sys.argv[1:], "env": keep}))
sys.exit(%d)
""" % STUB_EXIT


def _model(ctx):
    return ctx.models.get("gguf_270m", {"repo": "unsloth/gemma-3-270m-it-GGUF", "variant": "UD-Q4_K_XL"})


_AGENT_ENV = ("ANTHROPIC_", "OPENAI_", "CLAUDE_")   # ambient agent env would leak into the stub's record


def _env(ctx, home, extra=None):
    base = {k: v for k, v in os.environ.items() if not k.startswith(_AGENT_ENV)}
    return {**base, **engine.STUDIO_ENV, "UNSLOTH_STUDIO_HOME": str(home), "PYTHONUNBUFFERED": "1",
            **(extra or {})}


def _run_cmd(ctx, home, port):
    m = _model(ctx)
    return [str(Path(home) / "unsloth_studio" / "bin" / "unsloth"), "run", "--model", m["repo"],
            "--gguf-variant", m.get("variant", "UD-Q4_K_XL"), "--context-length", "2048",
            "--port", str(port), "--host", "127.0.0.1", "--password", PASSWORD, "--yes"]


def aux_base(ctx):
    """Prefix for this journey's extra homes: the unit's own state home (parallel mode: unique per
    side and unit, same length for all), else state/<root name> as before."""
    return ctx.state.get("state_base") or str(engine.WS / "temp" / "studio_regress" / "state" /
                                              Path(ctx.state["root"]).name)


async def _get(url, token=None, timeout=5):
    import httpx
    h = {"Authorization": f"Bearer {token}"} if token else {}
    async with httpx.AsyncClient(timeout=timeout) as c:
        return await c.get(url, headers=h)


# On a shared Studio (sequential, --instance shared) run_ready's extra home is state/<root>_cli on
# both sides, so the two sides must never run at once (scheduler.build_plan -> "sequence"). Per-unit
# Studios (private_fs) give each side <unit home>_cli (aux_base), so there they may overlap.
OVERLAP_SIDES = False


async def run_ready(ctx):
    from pr_ui_scenes._common import pick_free_ports
    home = engine.state_home(ctx.state["install_home"], Path(aux_base(ctx) + "_cli"), venv_shim=True)
    port = pick_free_ports(1)[0]
    log_dir = Path(ctx.out_dir) / "cli"
    log_dir.mkdir(parents=True, exist_ok=True)
    log = open(log_dir / "unsloth_run.log", "w")
    proc = subprocess.Popen(_run_cmd(ctx, home, port), env=_env(ctx, home), stdout=log,
                            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    ctx.state.update(cli_home=str(home), cli_port=port, cli_pid=proc.pid)
    CLI_PROCS[ctx.side] = proc
    base = f"http://127.0.0.1:{port}"
    budget = timeouts.inner(READY_TIMEOUT_S, ctx.state)
    deadline = time.time() + budget
    health, iv = None, 0.25

    def exited():   # a dead `unsloth run` fails at once, in either wait, never at the deadline
        if proc.poll() is not None:
            raise StepFailed(f"`unsloth run` exited {proc.returncode} before ready; see {log_dir}/unsloth_run.log")
    while time.time() < deadline:
        exited()
        try:
            r = await _get(base + "/api/health")
            if r.status_code == 200:
                health = r.json()
                break
        except Exception:
            pass
        await asyncio.sleep(iv)
        iv = min(2.0, iv * 1.5)
    if health is None:
        raise StepFailed(f"not healthy within {budget:.0f}s")
    tok = (await asyncio.to_thread(engine.api_login, base, PASSWORD))["access_token"]
    ctx.state["cli_token"] = tok
    loaded, iv = None, 0.25
    while time.time() < deadline:
        exited()
        try:
            st = (await _get(base + "/api/inference/status", tok)).json()
            loaded = st.get("model_path") or st.get("active_model") or st.get("loaded")
            if loaded:
                break
        except Exception:
            pass
        await asyncio.sleep(iv)
        iv = min(2.0, iv * 1.5)
    if not loaded:
        raise StepFailed(f"server up but no model loaded within {budget:.0f}s")
    return {"service": health.get("service"), "model_loaded": True}


async def completion(ctx):
    import httpx
    base = f"http://127.0.0.1:{ctx.state['cli_port']}"
    body = {"model": _model(ctx)["repo"], "messages": [{"role": "user", "content": "Reply with OK."}],
            "max_tokens": 8, "temperature": 0, "seed": 0, "stream": False}
    async with httpx.AsyncClient(timeout=timeouts.inner(180, ctx.state)) as c:
        r = await c.post(base + "/v1/chat/completions", json=body,
                         headers={"Authorization": f"Bearer {ctx.state['cli_token']}"})
    if r.status_code != 200:
        raise StepFailed(f"/v1/chat/completions {r.status_code}: {r.text[:200]}")
    ch = (r.json().get("choices") or [{}])[0]
    content = ((ch.get("message") or {}).get("content") or "")
    if not content.strip():
        raise StepFailed("empty completion")
    return {"status": 200, "object": r.json().get("object"), "has_content": True}


_SERVING_RX = re.compile(r"running at http://127\.0\.0\.1:(\d+)")


async def port_in_use(ctx):
    """Second `unsloth run` on the busy port. Studio's contract (base 2026-09): it moves to the
    next free port and says so in its banner. Either that or a non-zero exit passes; the
    outcome is a fact, so a behaviour change between base and head shows in the facts diff."""
    # A second FRESH home: reusing the first would fail on "password already set", not on the port.
    home = engine.state_home(ctx.state["install_home"], Path(ctx.state["cli_home"] + "_2"), venv_shim=True)
    port = ctx.state["cli_port"]
    log_path = Path(ctx.out_dir) / "cli" / "port_in_use.log"
    t0 = time.time()
    with open(log_path, "w") as log:
        proc = subprocess.Popen(_run_cmd(ctx, home, port), env=_env(ctx, home), stdout=log,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    try:
        moved, budget, iv = None, timeouts.inner(READY_TIMEOUT_S, ctx.state, margin_s=70), 0.25
        while time.time() - t0 < budget and proc.poll() is None:
            m = _SERVING_RX.search(log_path.read_text(errors="replace"))
            if m:
                moved = int(m.group(1))
                break
            await asyncio.sleep(iv)
            iv = min(1.0, iv * 1.5)
        if moved is None:
            if proc.poll() is None:
                raise StepFailed("second `unsloth run` on a busy port neither failed nor announced a port")
            if proc.returncode == 0:
                raise StepFailed("second `unsloth run` on a busy port exited 0 without serving")
            return {"outcome": "exit_nonzero", "_s_to_fail": round(time.time() - t0, 1)}
        if moved == port:
            raise StepFailed(f"second `unsloth run` claims the busy port {port}")
        r = await _get(f"http://127.0.0.1:{moved}/api/health")
        if r.status_code != 200:
            raise StepFailed(f"announced port {moved} not healthy ({r.status_code})")
        return {"outcome": "moved_to_next_free_port", "_moved_to": moved, "_s": round(time.time() - t0, 1)}
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGINT)
            try:
                await asyncio.wait_for(asyncio.to_thread(proc.wait), 60)
            except asyncio.TimeoutError:
                os.killpg(proc.pid, signal.SIGKILL)


async def start_stub(ctx):
    d = Path(ctx.out_dir) / "cli" / "stub_bin"
    d.mkdir(parents=True, exist_ok=True)
    stub = d / "claude"
    stub.write_text(STUB)
    stub.chmod(0o755)
    out_json = Path(ctx.out_dir) / "cli" / "stub_out.json"
    home, port = ctx.state["cli_home"], ctx.state["cli_port"]
    env = _env(ctx, home, {"PATH": f"{d}:{os.environ.get('PATH', '')}", "SR_STUB_OUT": str(out_json),
                           "UNSLOTH_STUDIO_URL": f"http://127.0.0.1:{port}"})
    m = _model(ctx)
    cmd = [str(Path(home) / "unsloth_studio" / "bin" / "unsloth"), "start", "claude", "--no-serve",
           "--model", m["repo"], "--gguf-variant", m.get("variant", "UD-Q4_K_XL")]
    proc = await asyncio.create_subprocess_exec(*cmd, env=env, stdout=asyncio.subprocess.PIPE,
                                                stderr=asyncio.subprocess.STDOUT,
                                                stdin=asyncio.subprocess.DEVNULL)
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeouts.inner(400, ctx.state))
    except asyncio.TimeoutError:
        proc.kill()
        raise StepFailed("`unsloth start claude` did not return") from None
    (Path(ctx.out_dir) / "cli" / "start_claude.log").write_text(out.decode(errors="replace"))
    if not out_json.exists():
        raise StepFailed(f"stub agent never ran (exit {proc.returncode})")
    got = json.loads(out_json.read_text())
    base = got["env"].get("ANTHROPIC_BASE_URL", "")
    if f":{port}" not in base:
        raise StepFailed(f"ANTHROPIC_BASE_URL={base!r} is not this server")
    if proc.returncode != STUB_EXIT:
        raise StepFailed(f"agent exit {STUB_EXIT} not propagated (got {proc.returncode})")
    return {"base_url_is_server": True,
            "token_handed_over": bool(got["env"].get("ANTHROPIC_AUTH_TOKEN") or got["env"].get("ANTHROPIC_API_KEY")),
            "exit_propagated": True, "env_keys": sorted(got["env"])}


async def sigint_cleanup(ctx):
    proc = CLI_PROCS.pop(ctx.side, None)
    if proc is None:
        raise StepUnreachable("no `unsloth run` to stop")
    home = ctx.state["cli_home"]
    os.killpg(proc.pid, signal.SIGINT)
    wait_s = timeouts.inner(90, ctx.state, margin_s=10)
    try:
        await asyncio.wait_for(asyncio.to_thread(proc.wait), wait_s)
    except asyncio.TimeoutError:
        os.killpg(proc.pid, signal.SIGKILL)
        raise StepFailed(f"`unsloth run` ignored SIGINT for {wait_s:.0f}s") from None

    def survivors():
        out = subprocess.run(["pgrep", "-f", str(home)], capture_output=True, text=True).stdout.split()
        # Only processes running with THIS home (engine.stop's rule): the pattern also matches
        # <home>_2 (s03's second home) and anything else whose argv merely mentions the path.
        return [p for p in out if int(p) != os.getpid() and engine._ours(int(p), home)]
    # Children get the same 3 s grace as before to exit after the parent, but the check ends as
    # soon as none is left instead of always sleeping 3 s.
    grace = time.monotonic() + 3
    left = await asyncio.to_thread(survivors)
    while left and time.monotonic() < grace:
        await asyncio.sleep(0.25)
        left = await asyncio.to_thread(survivors)
    for p in left:
        try:
            os.kill(int(p), signal.SIGKILL)
        except ProcessLookupError:
            pass
    if left:
        raise StepFailed(f"{len(left)} process(es) survived SIGINT")
    return {"exited": True, "leftover": 0}


CLI_PROCS: dict = {}


async def teardown(ctx):
    proc = CLI_PROCS.pop(ctx.side, None)
    if proc is not None and proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

JOURNEY = Journey(
    name="cli", tier="model", needs=("gguf_270m",), routes=(), teardown=teardown,
    steps=(
        Step("s01_run_ready", run_ready, shot=False, timeout_s=READY_TIMEOUT_S + 60),
        Step("s02_completion", completion, shot=False, timeout_s=240),
        Step("s03_port_in_use", port_in_use, shot=False, timeout_s=READY_TIMEOUT_S + 90),
        Step("s04_start_stub", start_stub, shot=False, timeout_s=460),
        Step("s05_sigint_cleanup", sigint_cleanup, shot=False, timeout_s=120),
    ),
)
