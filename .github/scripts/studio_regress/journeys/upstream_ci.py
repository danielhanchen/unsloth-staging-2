"""Journey 9: unsloth's own Playwright suites, run from EACH side's source checkout.

Base passes and head fails -> FAIL_HEAD (functional regression) through the normal diff.
Each script gets its own fresh-state Studio in bootstrap state (they drive first-run
change-password themselves) on a private port, with the tiny CI GGUF. Their screenshots are
kept as `_evidence_png` (reviewable, NOT pixel-diffed: the suites take full-page shots with
live content, which would only add noise; the pass / fail and the last reached step are
the signal).

Only the suites the diff can reach run (test_select.py); the others record SKIPPED_NOT_TOUCHED,
never a pass. Which ones, in order of precedence:
  STUDIO_REGRESS_FULL_TESTS=1              every suite (the old behaviour)
  STUDIO_REGRESS_UPSTREAM_SUITES=a.py,b.py  exactly these (ALL / NONE); staging_ci.py sets it on CI legs
  STUDIO_REGRESS_BASE_SHA / _HEAD_SHA / _SRC_REPO (run.py exports them) -> test_select.select_playwright
  none of those                             every suite (nothing to select from)
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from pathlib import Path

from studio_regress import engine, timeouts
from studio_regress.contract import Journey, Step, StepFailed, StepNotTouched, StepUnreachable

SCRIPTS = {"s01_chat_ui": "playwright_chat_ui.py", "s02_extra_ui": "playwright_extra_ui.py"}
TIMEOUT_S = 1200
_STEP_RX = re.compile(r"^\[ui\]\s*(?:==>|STEP|step)?\s*(.+)$")


FULL_ENV = "STUDIO_REGRESS_FULL_TESTS"
SUITES_ENV = "STUDIO_REGRESS_UPSTREAM_SUITES"
SHA_ENV = ("STUDIO_REGRESS_BASE_SHA", "STUDIO_REGRESS_HEAD_SHA", "STUDIO_REGRESS_SRC_REPO")
_SELECTION = {}


def selected_scripts(root = None):
    """(set of script names to run, or None for every suite; reason). Cached per (base, head) in
    `root`/upstream_ci_selection.json so both sides (and isolated sub-runs) agree."""
    if os.environ.get(FULL_ENV) == "1":
        return None, f"{FULL_ENV}=1"
    v = os.environ.get(SUITES_ENV)
    if v is not None:
        v = v.strip()
        if v.upper() in ("ALL", "FULL"):
            return None, f"{SUITES_ENV}={v}"
        if v.upper() in ("", "NONE"):
            return set(), f"{SUITES_ENV}=NONE"
        return {x.strip() for x in v.split(",") if x.strip()}, f"{SUITES_ENV}={v}"
    base, head, repo = (os.environ.get(k) for k in SHA_ENV)
    if not (base and head and repo):
        return None, "no base / head to select from: every suite runs"
    key = (repo, base, head)
    if key in _SELECTION:
        return _SELECTION[key]
    cache = Path(root) / "upstream_ci_selection.json" if root else None
    sel = None
    if cache and cache.exists():
        try:
            got = json.loads(cache.read_text())
            if (got.get("base"), got.get("head")) == (base, head):
                sel = got
        except (OSError, ValueError):
            sel = None
    if sel is None:
        try:
            scripts = str(Path(__file__).resolve().parents[2])
            if scripts not in sys.path:
                sys.path.append(scripts)
            import test_select

            sel = {"base": base, "head": head, **test_select.select_playwright(repo, base, head)}
        except Exception as e:  # never lose coverage to a selector fault
            sel = {
                "base": base,
                "head": head,
                "mode": "FULL",
                "suites": [],
                "reasons": [f"selector error: {type(e).__name__}: {e}"[:300]],
            }
        if cache:
            try:
                cache.write_text(json.dumps(sel, indent = 1, sort_keys = True))
            except OSError:
                pass
    reason = "; ".join(sel.get("reasons") or [])[:500]
    out = (
        (None, f"test_select FULL: {reason}")
        if sel.get("mode") == "FULL"
        else (set(sel.get("suites") or []), f"test_select {sel.get('mode')}: {reason}")
    )
    _SELECTION[key] = out
    return out


def _last_step(text):
    last = None
    for line in text.splitlines():
        if line.startswith("[ui]") or line.startswith("STEP") or line.startswith("==>"):
            last = line.strip()[:160]
    return last


def _suite_step(step_id, script):
    async def act(ctx):
        chosen, why = selected_scripts(ctx.state.get("root"))
        if chosen is not None and script not in chosen:
            raise StepNotTouched(f"{script} not selected ({why})")
        src = ctx.state.get("src")
        if not src or not (Path(src) / "tests" / "studio" / script).exists():
            raise StepUnreachable(f"{script} not in this side's source ({src})")
        from pr_ui_scenes._common import pick_free_ports

        root = Path(ctx.state["root"])
        home = engine.state_home(
            ctx.state["install_home"],
            Path(os.environ.get("WORKSPACE", "."))
            / "temp"
            / "studio_regress"
            / "state"
            / f"{root.name}_upstream",
        )
        port = pick_free_ports(1, seed = f"{root.name}_{step_id}")[0]
        log_dir = Path(ctx.out_dir) / "upstream_ci"
        log_dir.mkdir(parents = True, exist_ok = True)
        inst = await asyncio.to_thread(engine.launch, home, port, log_dir / f"{step_id}_studio.log")
        port = inst.port  # launch moves to a fresh port if this one was taken
        art = log_dir / step_id
        art.mkdir(exist_ok = True)
        m = ctx.models.get(
            "gguf_270m", {"repo": "unsloth/gemma-3-270m-it-GGUF", "variant": "UD-Q4_K_XL"}
        )
        import pw_fast

        env = {
            **pw_fast.site_env(),
            "BASE_URL": f"http://127.0.0.1:{port}",
            "STUDIO_OLD_PW": inst.bootstrap_password or "",
            "STUDIO_NEW_PW": "UpstreamNew-2026!x",
            "GGUF_REPO": m["repo"],
            "GGUF_VARIANT": m.get("variant", "UD-Q4_K_XL"),
            "PW_ART_DIR": str(art),
            "STUDIO_UI_STRICT": "1",
            "PYTHONUNBUFFERED": "1",
        }
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                str(Path(src) / "tests" / "studio" / script),
                cwd = str(Path(src) / "tests" / "studio"),
                env = env,
                stdout = asyncio.subprocess.PIPE,
                stderr = asyncio.subprocess.STDOUT,
            )
            try:
                # Timeout policy (timeouts.py): load-scaled, and inside the step budget so the
                # suite is killed here rather than orphaned by the engine's outer cancel.
                out, _ = await asyncio.wait_for(
                    proc.communicate(), timeouts.inner(TIMEOUT_S, ctx.state, 60)
                )
                rc = proc.returncode
            except asyncio.TimeoutError:
                proc.kill()
                out, rc = b"", -9
        finally:
            await asyncio.to_thread(engine.stop, port)
        text = out.decode(errors = "replace")
        (log_dir / f"{step_id}.log").write_text(text)
        root_side = Path(ctx.out_dir)
        pngs = sorted(str(p.relative_to(root_side)) for p in art.glob("*.png"))
        facts = {
            "script": script,
            "exit": rc,
            "passed": rc == 0,
            "_evidence_png": pngs,
            "_last_step": _last_step(text),
            "_selection": why,
        }
        if rc != 0:
            tail = "\n".join(text.splitlines()[-8:])
            raise StepFailed(f"{script} exit {rc}; last: {facts['_last_step']}\n{tail}"[:700])
        return facts

    act.__name__ = step_id
    return act


# Pass / fail only (no pixel diff), and each suite takes 10-20 min per side: run.py runs the head
# first and the base only for the suites that fail on head (external.py's rule for the diffusion
# suites). A head pass is SAME without a base run.
HEAD_FIRST = True

JOURNEY = Journey(
    name = "upstream_ci",
    tier = "model",
    needs = ("gguf_270m",),
    routes = ("/chat", "/settings"),
    independent = True,
    steps = tuple(
        Step(sid, _suite_step(sid, s), shot = False, timeout_s = TIMEOUT_S + 120)
        for sid, s in SCRIPTS.items()
    ),
)
