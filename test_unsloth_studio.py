#!/usr/bin/env python3
"""test_unsloth_studio.py -- ad-hoc Studio probe via Playwright MCP.

Drive Unsloth Studio's UI with a free-form natural-language prompt,
answered by a ``claude --print`` subprocess whose tool allow-list is
limited to the Playwright MCP ``browser_*`` family. The MCP itself is
loaded per-invocation via ``--mcp-config <json> --strict-mcp-config``;
no user-scope registration is required, and launcher.sh does not
install it ambient.

Smart-mode dispatch (no explicit --attach/--standalone required):

    * --port N                : attach to N (must be alive)
    * --pr <URL|N|owner/repo#N>: install the PR's head, launch, drive
    * --branch <ref>          : install that ref, launch, drive
    * --attach-only           : attach or fail (never install)
    * (no flags)              : attach to 8888..8895 if reachable,
                                otherwise install unslothai/unsloth@main

Output is a plain-text report on stdout. No JSON verdict -- the review
phase (studio_browser_test.py) handles PASS/FAIL; this tool is for
humans asking the agent "tell me what you found."
"""

from __future__ import annotations

import argparse
import json
import os
import re
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from claude_cmd import build_claude_cmd
import studio_browser_test as sbt


ATTACH_PROBE_PORTS = [8888, 8889, 8890, 8891, 8892, 8893, 8894, 8895]
DEFAULT_TIMEOUT = 600
HEALTH_ENDPOINT = "/api/health"
DEFAULT_REPO = "unslothai/unsloth"
INSTALL_TIMEOUT = sbt.INSTALL_TIMEOUT


# ---------------------------------------------------------------------------
# Env-mode workspace helpers (UNSLOTH_STUDIO_HOME -- requires the install.sh
# changes from PR #5190 to be present in the cloned repo). Used when the user
# passes --workspace, --temp-workspace, --install-main, or --pr / --branch
# without a pre-set UNSLOTH_STUDIO_HOME env var.
# ---------------------------------------------------------------------------


def _make_temp_workspace_under_cwd() -> Path:
    """Create cwd/temp/unsloth_studio_<random>/.

    Uses tempfile.mkdtemp so two probes started in the same second
    cannot collide on the directory name. Naming matches the layout
    used by studio_browser_test.run_studio_browser_test and the
    parallel-install smoke test in unslothai/unsloth
    tests/studio/install/, so all three studio drivers leave the
    same kind of folder behind.
    """
    base = Path.cwd() / "temp"
    base.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix="unsloth_studio_", dir=base))
    return workdir.resolve()


def _envmode_install_dirs(workspace: Path) -> tuple[Path, Path]:
    """Return (repo_dir, install_root) under the workspace."""
    return workspace / "repo", workspace / "studio"


def _envmode_install_studio(
    repo_dir: Path, install_root: Path, timeout_s: int = INSTALL_TIMEOUT,
) -> tuple[bool, str]:
    """Run install.sh --local with UNSLOTH_STUDIO_HOME pinned to install_root."""
    install_root.mkdir(parents=True, exist_ok=True)
    script = repo_dir / "install.sh"
    if not script.is_file():
        return False, f"[install] install.sh not found at {script}"
    env = os.environ.copy()
    env["UNSLOTH_STUDIO_HOME"] = str(install_root)
    try:
        proc = subprocess.run(
            ["bash", str(script), "--local"],
            cwd=str(repo_dir),
            env=env,
            capture_output=True, text=True, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return False, f"[install] TIMEOUT after {timeout_s}s"
    log = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, log


def _envmode_unsloth_bin(install_root: Path) -> str | None:
    """Locate the unsloth CLI inside the env-mode install."""
    bin_path = install_root / "bin" / "unsloth"
    if bin_path.is_file() and os.access(bin_path, os.X_OK):
        return str(bin_path)
    return None


def _envmode_venv_python(install_root: Path) -> Path:
    return install_root / "unsloth_studio" / "bin" / "python"


def _envmode_verify_pr_code(install_root: Path, repo_dir: Path) -> tuple[bool, str]:
    """Confirm `import unsloth` resolves under the cloned PR repo (editable install)."""
    py = _envmode_venv_python(install_root)
    if not py.is_file():
        return False, f"[verify] venv python missing at {py}"
    code = (
        "import unsloth, sys; "
        "sys.stdout.write('\\n<<<UNSLOTH_FILE>>>' + unsloth.__file__ + '<<<END>>>\\n')"
    )
    env = os.environ.copy()
    env["UNSLOTH_STUDIO_HOME"] = str(install_root)
    try:
        proc = subprocess.run(
            [str(py), "-c", code],
            env=env, capture_output=True, text=True, timeout=30,
        )
    except subprocess.TimeoutExpired:
        return False, "[verify] import unsloth timed out after 30s"
    if proc.returncode != 0:
        return False, f"[verify] import unsloth failed: {(proc.stderr or '').strip()[:400]}"
    m = re.search(r"<<<UNSLOTH_FILE>>>(.*?)<<<END>>>", proc.stdout or "", re.DOTALL)
    if not m:
        return False, f"[verify] sentinel missing: {(proc.stdout or '').strip()[:400]!r}"
    loaded = m.group(1).strip()
    repo_abs = str(repo_dir.resolve())
    if not loaded.startswith(repo_abs):
        return False, (
            f"[verify] unsloth loaded from {loaded!r}, expected under {repo_abs!r}"
        )
    return True, f"[verify] OK -- unsloth loaded from {loaded}"


def _envmode_start_server(install_root: Path, port: int) -> subprocess.Popen | None:
    """Spawn unsloth studio with UNSLOTH_STUDIO_HOME pinned."""
    bin_path = _envmode_unsloth_bin(install_root)
    if not bin_path:
        return None
    env = os.environ.copy()
    env["UNSLOTH_STUDIO_HOME"] = str(install_root)
    return subprocess.Popen(
        [bin_path, "studio", "-H", "127.0.0.1", "-p", str(port)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env=env, preexec_fn=os.setsid,
    )


def _envmode_read_bootstrap(install_root: Path) -> str | None:
    """Read the first-boot admin password from the env-mode install."""
    path = install_root / "auth" / ".bootstrap_password"
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _resolve_install_workspace(args) -> Path | None:
    """Resolve the install root per priority order, or None for legacy mode.

    Priority (highest first):
      1. --workspace PATH (explicit)
      2. --temp-workspace (auto-create under cwd/temp/)
      3. UNSLOTH_STUDIO_HOME / STUDIO_HOME env var (global)
      4. --pr / --branch / --install-main given AND no env var: auto temp
      5. None -> legacy HOME-redirect mode (current default behaviour)
    """
    if args.workspace:
        ws = Path(args.workspace).resolve()
        ws.mkdir(parents=True, exist_ok=True)
        return ws
    if args.temp_workspace:
        return _make_temp_workspace_under_cwd()
    env_override = os.environ.get("UNSLOTH_STUDIO_HOME") or os.environ.get("STUDIO_HOME")
    if env_override:
        return Path(env_override).expanduser().resolve()
    if args.pr is not None or args.branch is not None or getattr(args, "install_main", False):
        return _make_temp_workspace_under_cwd()
    return None


# ---------------------------------------------------------------------------
# port discovery
# ---------------------------------------------------------------------------

def _port_is_studio(port: int, timeout_s: float = 1.5) -> bool:
    """Return True iff 127.0.0.1:<port> answers HTTP with <500."""
    url = f"http://127.0.0.1:{port}{HEALTH_ENDPOINT}"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            return resp.status < 500
    except (urllib.error.URLError, ConnectionError, OSError, TimeoutError):
        return False


def _discover_port(explicit: int | None) -> int | None:
    """Return the port of a running Studio, or None."""
    if explicit is not None:
        return explicit if _port_is_studio(explicit) else None
    for port in ATTACH_PROBE_PORTS:
        if _port_is_studio(port):
            return port
    return None


def _read_host_bootstrap_password() -> str | None:
    """Read ~/.unsloth/studio/auth/.bootstrap_password from the REAL HOME."""
    home = os.environ.get("HOME")
    if not home:
        return None
    path = Path(home) / ".unsloth" / "studio" / "auth" / ".bootstrap_password"
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Playwright MCP self-install
# ---------------------------------------------------------------------------

def _install_chromium_via_scratch_project() -> bool:
    """Install Playwright chromium via a scratch npm project.

    ``npx -y playwright install`` silently no-ops in an empty cwd; the
    workaround (mirroring launcher.sh:1250-1255) is a throwaway project
    that depends on @playwright/test, from which playwright install
    actually downloads the chromium bundle.
    """
    if subprocess.run(
        ["which", "npm"], capture_output=True,
    ).returncode != 0:
        print(
            "[test_unsloth_studio] npm not found; cannot install chromium",
            file=sys.stderr,
        )
        return False

    scratch = Path(tempfile.mkdtemp(prefix="pw_install_"))
    try:
        (scratch / "package.json").write_text(
            '{"name":"pw-install","version":"1.0.0"}', encoding="utf-8",
        )
        rc = subprocess.run(
            ["npm", "install", "-D", "@playwright/test", "--silent"],
            cwd=str(scratch), capture_output=True, text=True, timeout=300,
        )
        if rc.returncode != 0:
            tail = (rc.stderr or rc.stdout or "")[-500:]
            print(
                f"[test_unsloth_studio] npm install @playwright/test failed: {tail}",
                file=sys.stderr,
            )
            return False

        pw_bin = scratch / "node_modules" / ".bin" / "playwright"
        if not pw_bin.is_file():
            print(
                "[test_unsloth_studio] playwright binary missing after npm install",
                file=sys.stderr,
            )
            return False

        install_args = ["install"]
        if subprocess.run(
            ["sudo", "-n", "true"], capture_output=True,
        ).returncode == 0:
            install_args = ["install", "--with-deps"]

        rc = subprocess.run(
            [str(pw_bin)] + install_args + ["chromium"],
            cwd=str(scratch), capture_output=True, text=True, timeout=900,
        )
        if rc.returncode != 0:
            tail = (rc.stderr or rc.stdout or "")[-500:]
            print(
                f"[test_unsloth_studio] playwright install chromium failed: {tail}",
                file=sys.stderr,
            )
            return False
        return True
    finally:
        shutil.rmtree(str(scratch), ignore_errors=True)


# MCP config + writer live in studio_browser_test (single source of
# truth). The probe imports it via `import studio_browser_test as sbt`
# at the top of this file, so use `sbt.write_playwright_mcp_config(...)`.


def _ensure_playwright_chromium() -> bool:
    """Ensure the Playwright chromium bundle is cached on disk.

    The MCP itself is loaded per-invocation via ``claude --print
    --mcp-config`` (see ``_run_probe``), so we do NOT register a
    user-scope MCP here. But the MCP's headless chromium still needs
    the browser binary on disk -- which Playwright only fetches via
    its scratch-project npm trick.
    """
    browser_root = Path(os.environ.get(
        "PLAYWRIGHT_BROWSERS_PATH",
        str(Path.home() / ".cache" / "ms-playwright"),
    ))
    if browser_root.is_dir() and any(
        p.is_dir() and p.name.startswith("chromium-")
        for p in browser_root.iterdir()
    ):
        return True

    print(
        "[test_unsloth_studio] installing chromium for Playwright "
        "(one-time, ~180 MB)...",
    )
    return _install_chromium_via_scratch_project()


# ---------------------------------------------------------------------------
# PR / branch resolution
# ---------------------------------------------------------------------------

_PR_URL_RE = re.compile(
    r"^https?://github\.com/([^/\s]+)/([^/\s]+)/pull/(\d+)(?:[/?#].*)?$"
)
_PR_HASH_RE = re.compile(r"^([^/\s#]+/[^/\s#]+)#(\d+)$")


def _parse_pr_arg(value: str) -> tuple[str, int]:
    """argparse ``type=`` for --pr.

    Accepts:
      - bare number             -> (DEFAULT_REPO, N)
      - github.com pull URL     -> (owner/repo, N)
      - owner/repo#N            -> (owner/repo, N)
    """
    s = (value or "").strip()
    if not s:
        raise argparse.ArgumentTypeError("--pr value is empty")

    m = _PR_URL_RE.match(s)
    if m:
        return f"{m.group(1)}/{m.group(2)}", int(m.group(3))

    m = _PR_HASH_RE.match(s)
    if m:
        return m.group(1), int(m.group(2))

    if s.isdigit():
        return DEFAULT_REPO, int(s)

    raise argparse.ArgumentTypeError(
        f"--pr: cannot parse {value!r}; expected a PR number, "
        f"a github.com pull URL, or owner/repo#N"
    )


def _resolve_pr_to_ref(owner_repo: str, pr_num: int) -> tuple[str, str]:
    """Resolve a PR to (head_owner_repo, head_ref) via ``gh pr view``.

    Returns the fork's owner/repo when the PR comes from a fork so the
    caller can clone the right tree. Raises ``RuntimeError`` on any gh
    failure so the caller can surface a friendly error instead of a
    Python traceback.
    """
    try:
        out = subprocess.check_output(
            [
                "gh", "pr", "view", str(pr_num), "-R", owner_repo,
                "--json", "headRefName,headRepositoryOwner,headRepository",
            ],
            text=True, timeout=30, stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        raise RuntimeError("gh CLI not found in PATH")
    except subprocess.TimeoutExpired:
        raise RuntimeError("gh pr view timed out after 30s")
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or "").strip()
        raise RuntimeError(f"gh pr view failed: {stderr or e}")

    try:
        data = json.loads(out)
        head_owner = data["headRepositoryOwner"]["login"]
        head_name = data["headRepository"]["name"]
        head_ref = data["headRefName"]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise RuntimeError(f"gh pr view returned unexpected JSON: {e}")

    return f"{head_owner}/{head_name}", head_ref


# ---------------------------------------------------------------------------
# prompt builder
# ---------------------------------------------------------------------------

def _build_adhoc_prompt(
    user_prompt: str,
    *,
    port: int,
    screenshot_dir: str,
    bootstrap_password: str | None,
    new_password: str,
) -> str:
    """Assemble the prompt for the claude --print subprocess.

    Deliberately looser than studio_browser_test._build_browser_prompt:
    no JSON verdict, no smoke/dynamic schema -- just a human-readable
    report answering the user's ad-hoc request.
    """
    bootstrap_line = bootstrap_password if bootstrap_password else "(none -- bootstrap file absent; password already changed)"
    return f"""<task>
You are driving the Unsloth Studio UI at http://127.0.0.1:{port} via
Playwright MCP. This is a focused probe, not an exhaustive audit --
answer the user's request with concrete evidence and stop.

<auth>
  - Admin username: unsloth
  - Bootstrap password (first-boot only): {bootstrap_line}
  - If you land on /change-password, set a new password to: {new_password}
  - If you land on /login, use {new_password} (or the bootstrap above if
    the password hasn't been changed yet).
  - If auth fails and neither password works, STOP and report that in
    the final report -- do not loop on /login.
</auth>

<studio-ui-tips>
  - Train page is at /studio. Combobox dropdowns (HF model search,
    dataset search) populate a detached listbox that
    browser_snapshot does NOT capture. To read/click options, use
    browser_evaluate with:
      document.querySelectorAll('[role="option"]')
  - Save screenshots under {screenshot_dir}/ with descriptive names.
</studio-ui-tips>

<user-request>
{user_prompt}
</user-request>

<how-to-work>
  - Use browser_snapshot before each click to confirm the element
    exists; hallucinated selectors produce wrong answers.
  - Use browser_console_messages (level=error) and
    browser_network_requests to catch JS errors and 4xx/5xx.
  - Use browser_take_screenshot for visual evidence of anything
    interesting.
  - Keep the probe focused. "Find as many bugs as possible" still
    means a single pass over the main surfaces, not an hour-long
    audit.
</how-to-work>

<output>
At the end, write a plain-text report. No JSON required. The report
MUST:
  - Directly answer the user's request (lead with the answer).
  - Cite specific evidence: URLs, element text, console errors,
    network status codes, screenshot filenames.
  - List any bugs/anomalies observed with severity (low/med/high) if
    you can tell.
  - Be readable by a human skimming the output; no markdown fences
    required, but use headings or bullets where they help.
</output>

Allowed tools: mcp__playwright__* (UI driving), Bash, Read, Edit, Write, Glob, Grep (file / shell), WebFetch, WebSearch (research). Use Bash sparingly -- the probe is meant to drive Studio's UI, not run arbitrary scripts. Reach for WebFetch / WebSearch only if the user's prompt requires comparing Studio behaviour to upstream docs or known issues.

<screenshots mandatory="true">
Take a screenshot at every meaningful UI state, including:
  - the initial landing page (or any auth/redirect page),
  - after every significant navigation,
  - before AND after any form submission or state-changing click,
  - any error banner, dialog, modal, or unexpected state,
  - every page that evidences part of your final conclusion.
Use browser_take_screenshot with a descriptive RELATIVE filename
(e.g. "01_landing.png", "02_train_page.png", "03_error_toast.png").
Every screenshot MUST be saved under {screenshot_dir}/ and every
screenshot MUST be referenced by filename in the final report --
un-cited screenshots are as bad as missing ones.
</screenshots>
</task>
"""


# ---------------------------------------------------------------------------
# claude subprocess
# ---------------------------------------------------------------------------


def _run_probe(
    prompt: str,
    *,
    cwd: Path,
    model: str,
    timeout: int,
    verbose: bool,
) -> int:
    """Spawn ``claude --print``, stream stdout live, return exit code.

    Loads Playwright MCP dynamically via ``--mcp-config <json>
    --strict-mcp-config`` so the ambient user-scope MCP registry is
    ignored -- the MCP is only alive for the duration of this probe.
    Tool gating relies on the deny-list defaults in
    :func:`claude_cmd.build_claude_cmd` (no per-call --tools allow-list).
    """
    mcp_config = cwd / ".playwright_mcp.json"
    sbt.write_playwright_mcp_config(mcp_config)

    # Use the standard subprocess deny-list from claude_cmd.py
    # (_DEFAULT_DISALLOWED_TOOLS): drops worktree, notebook, streaming
    # watcher, agent-teams, plan-mode, Cron, AskUserQuestion. Everything
    # else is allowed, so research-flavoured prompts ("compare to
    # upstream", "search source for this error", "fetch the issue")
    # work without the model bumping into missing tools.
    # Playwright MCP (mcp__playwright__browser_*) comes in separately
    # via --mcp-config below.
    # Opt in to the project's interactive system prompt (-1.7k system
    # tokens vs the Claude Code default; UI-driving + research workload
    # is interactive-shaped). Falls back to the default automatically if
    # the file is missing -- build_claude_cmd only adds the flag when
    # system_prompt_file is truthy.
    interactive_sp = SCRIPTS_DIR / "interactive_system_prompt.txt"
    cmd = build_claude_cmd(
        print_flag=True,
        model=model,
        verbose=verbose,
        system_prompt_file=str(interactive_sp) if interactive_sp.is_file() else None,
    )
    cmd += ["--mcp-config", str(mcp_config), "--strict-mcp-config"]

    # ENABLE_TOOL_SEARCH=false per-invocation (NOT in claude/settings.json
    # since interactive launcher sessions still want tool search). This
    # probe always uses Playwright MCP -- deferring its definitions and
    # paying a search round-trip on first use is pure overhead. Per docs
    # https://code.claude.com/docs/en/agent-sdk/tool-search the env var
    # disables tool search and eager-loads MCP tool definitions instead.
    probe_env = os.environ.copy()
    probe_env["ENABLE_TOOL_SEARCH"] = "false"
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(cwd),
            env=probe_env,
            # New process group so Ctrl-C / timeout kills the whole
            # tree (npx, chromium, headless wrapper) rather than just
            # the top-level claude binary.
            start_new_session=True,
        )
    except FileNotFoundError:
        print("[test_unsloth_studio] ERROR: 'claude' CLI not found in PATH", file=sys.stderr)
        return 1

    assert proc.stdin is not None and proc.stdout is not None
    try:
        proc.stdin.write(prompt)
        proc.stdin.close()
    except (BrokenPipeError, OSError) as e:
        print(f"[test_unsloth_studio] ERROR writing prompt: {e}", file=sys.stderr)
        _kill_process_group(proc)
        return 1

    # Use select to poll stdout with a 1 s tick so the --timeout deadline
    # fires even when claude buffers output (claude --print without a
    # streaming --output-format may be silent for minutes at a time).
    deadline = time.time() + timeout
    fd = proc.stdout.fileno()
    try:
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                print(f"\n[test_unsloth_studio] TIMEOUT after {timeout}s", file=sys.stderr)
                _kill_process_group(proc)
                return 1
            ready, _, _ = select.select([fd], [], [], min(remaining, 1.0))
            if ready:
                chunk = os.read(fd, 4096)
                if not chunk:
                    break
                sys.stdout.write(chunk.decode("utf-8", errors="replace"))
                sys.stdout.flush()
            elif proc.poll() is not None:
                break
    except KeyboardInterrupt:
        _kill_process_group(proc)
        raise
    return proc.wait()


def _kill_process_group(proc: "subprocess.Popen") -> None:
    """SIGTERM the whole pgid; SIGKILL after 5s if it refuses to die.

    Pairs with ``start_new_session=True`` on Popen to ensure orphan
    npx/chromium children die with the parent claude process.
    """
    if proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass


# Exit codes: 0 success, 1 generic / probe error, 2 zero screenshots
# produced when the prompt mandates them. 130 = KeyboardInterrupt.
_EXIT_NO_SCREENSHOTS = 2

# Skip retry on these exit codes: success or genuine user-interrupt.
_NO_RETRY_EXITS = frozenset({0, 130, _EXIT_NO_SCREENSHOTS})


def _count_pngs(screenshot_dir: Path) -> int:
    """Count PNG files (recursive) under screenshot_dir."""
    if not screenshot_dir.is_dir():
        return 0
    return sum(1 for _ in screenshot_dir.rglob("*.png"))


def _run_probe_with_retries(
    prompt: str,
    *,
    cwd: Path,
    model: str,
    timeout: int,
    verbose: bool,
    retries: int,
    screenshot_dir: Path,
    allow_no_screenshots: bool,
) -> int:
    """Wrap _run_probe with retry/backoff + post-run screenshot check."""
    attempts = max(1, retries + 1)
    backoff = 5  # seconds; doubles each retry
    for attempt in range(1, attempts + 1):
        rc = _run_probe(
            prompt, cwd=cwd, model=model, timeout=timeout, verbose=verbose,
        )
        if rc in _NO_RETRY_EXITS or attempt == attempts:
            break
        print(
            f"[test_unsloth_studio] probe attempt {attempt}/{attempts} "
            f"failed (rc={rc}); retrying in {backoff}s",
            file=sys.stderr,
        )
        time.sleep(backoff)
        backoff *= 2

    if rc == 0 and not allow_no_screenshots:
        n = _count_pngs(screenshot_dir)
        if n == 0:
            print(
                f"[test_unsloth_studio] WARNING: probe exited 0 but produced "
                f"0 screenshots under {screenshot_dir}. Failing with rc={_EXIT_NO_SCREENSHOTS}; "
                "pass --allow-no-screenshots to suppress.",
                file=sys.stderr,
            )
            return _EXIT_NO_SCREENSHOTS
        print(f"[test_unsloth_studio] {n} screenshot(s) saved to {screenshot_dir}")
    return rc


# ---------------------------------------------------------------------------
# attach / install drivers
# ---------------------------------------------------------------------------

def _drive_attached(args, port: int) -> int:
    """Drive an already-running Studio on <port>."""
    bootstrap = args.password or _read_host_bootstrap_password()
    new_password = args.password or sbt.ISOLATED_ADMIN_PASSWORD

    screenshot_dir = Path(args.screenshot_dir)
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    prompt = _build_adhoc_prompt(
        args.prompt,
        port=port,
        screenshot_dir=str(screenshot_dir),
        bootstrap_password=bootstrap,
        new_password=new_password,
    )

    print(f"[test_unsloth_studio] attach port={port} screenshots={screenshot_dir}")
    return _run_probe_with_retries(
        prompt,
        cwd=screenshot_dir,
        model=args.model,
        timeout=args.timeout,
        verbose=args.verbose,
        retries=args.retries,
        screenshot_dir=screenshot_dir,
        allow_no_screenshots=args.allow_no_screenshots,
    )


def _run_install(args, owner_repo: str, head_ref: str) -> int:
    """Clone owner_repo@head_ref, install, launch, drive, tear down."""
    workdir = Path(tempfile.mkdtemp(prefix="test_unsloth_studio_"))
    popen: subprocess.Popen | None = None
    try:
        print(f"[test_unsloth_studio] workdir={workdir} repo={owner_repo} ref={head_ref}")

        ok, log = sbt._clone_pr_branch(owner_repo, head_ref, workdir)
        if not ok:
            print(log, file=sys.stderr)
            return 1

        ok, log = sbt._install_studio(workdir)
        if not ok:
            print(log[-4000:], file=sys.stderr)
            return 1

        ok, log = sbt._verify_pr_code_loaded(workdir)
        print(log)
        if not ok:
            return 1

        port = sbt._pick_free_port()
        popen = sbt._start_studio_server(workdir, port)
        if popen is None:
            print("[test_unsloth_studio] unsloth CLI not found after install", file=sys.stderr)
            return 1

        if not sbt._wait_for_health(port):
            print(f"[test_unsloth_studio] studio failed to bind on port {port}", file=sys.stderr)
            return 1

        bootstrap = args.password or sbt._read_bootstrap_password(workdir)
        new_password = args.password or sbt.ISOLATED_ADMIN_PASSWORD

        screenshot_dir = Path(args.screenshot_dir)
        screenshot_dir.mkdir(parents=True, exist_ok=True)

        prompt = _build_adhoc_prompt(
            args.prompt,
            port=port,
            screenshot_dir=str(screenshot_dir),
            bootstrap_password=bootstrap,
            new_password=new_password,
        )

        print(f"[test_unsloth_studio] installed pid={popen.pid} port={port} screenshots={screenshot_dir}")
        return _run_probe_with_retries(
            prompt,
            cwd=screenshot_dir,
            model=args.model,
            timeout=args.timeout,
            verbose=args.verbose,
            retries=args.retries,
            screenshot_dir=screenshot_dir,
            allow_no_screenshots=args.allow_no_screenshots,
        )
    finally:
        sbt._teardown(popen, workdir)


def _run_install_envmode(args, owner_repo: str, head_ref: str, workspace: Path) -> int:
    """Workspace-scoped install via UNSLOTH_STUDIO_HOME (PR #5190 feature).

    Layout:
        <workspace>/repo/      -- clone of <owner_repo>@<head_ref>
        <workspace>/studio/    -- Studio install root (UNSLOTH_STUDIO_HOME)
                                  -> studio/unsloth_studio/  venv
                                  -> studio/bin/unsloth      shim
                                  -> studio/share/           launcher + studio.conf
    """
    workspace = workspace.resolve()
    repo_dir, install_root = _envmode_install_dirs(workspace)
    repo_dir.mkdir(parents=True, exist_ok=True)

    popen: subprocess.Popen | None = None
    try:
        print(f"[test_unsloth_studio] env-mode workspace={workspace} "
              f"repo={owner_repo}@{head_ref} install_root={install_root}")

        ok, log = sbt._clone_pr_branch(owner_repo, head_ref, repo_dir)
        if not ok:
            print(log, file=sys.stderr); return 1

        ok, log = _envmode_install_studio(repo_dir, install_root)
        if not ok:
            print(log[-4000:], file=sys.stderr); return 1

        ok, log = _envmode_verify_pr_code(install_root, repo_dir)
        print(log)
        if not ok:
            return 1

        port = sbt._pick_free_port()
        popen = _envmode_start_server(install_root, port)
        if popen is None:
            print("[test_unsloth_studio] unsloth CLI missing under "
                  f"{install_root}/bin/unsloth", file=sys.stderr)
            return 1

        if not sbt._wait_for_health(port):
            print(f"[test_unsloth_studio] studio failed to bind on port {port}",
                  file=sys.stderr)
            return 1

        bootstrap = args.password or _envmode_read_bootstrap(install_root)
        new_password = args.password or sbt.ISOLATED_ADMIN_PASSWORD

        screenshot_dir = Path(args.screenshot_dir)
        screenshot_dir.mkdir(parents=True, exist_ok=True)

        prompt = _build_adhoc_prompt(
            args.prompt, port=port,
            screenshot_dir=str(screenshot_dir),
            bootstrap_password=bootstrap, new_password=new_password,
        )

        print(f"[test_unsloth_studio] env-mode pid={popen.pid} port={port} "
              f"screenshots={screenshot_dir}")
        return _run_probe_with_retries(
            prompt, cwd=screenshot_dir, model=args.model,
            timeout=args.timeout, verbose=args.verbose, retries=args.retries,
            screenshot_dir=screenshot_dir,
            allow_no_screenshots=args.allow_no_screenshots,
        )
    finally:
        # Reuse sbt._teardown for the popen kill; we manage workspace separately
        # because env-mode uses a different layout (no isolated $HOME).
        if popen is not None:
            try:
                os.killpg(os.getpgid(popen.pid), signal.SIGTERM)
                popen.wait(timeout=10)
            except (ProcessLookupError, PermissionError, subprocess.TimeoutExpired):
                try:
                    os.killpg(os.getpgid(popen.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass


def _install_from_flags(args) -> int:
    """Resolve --pr/--branch/--repo into (owner_repo, ref) and install."""
    if args.pr is not None:
        owner_repo, pr_num = args.pr
        if args.repo:
            owner_repo = args.repo
        try:
            head_repo, head_ref = _resolve_pr_to_ref(owner_repo, pr_num)
        except RuntimeError as e:
            print(f"[test_unsloth_studio] {e}", file=sys.stderr)
            return 1
        print(
            f"[test_unsloth_studio] resolved PR {owner_repo}#{pr_num} "
            f"-> {head_repo}@{head_ref}"
        )
        return _dispatch_install(args, head_repo, head_ref)

    owner_repo = args.repo or DEFAULT_REPO
    return _dispatch_install(args, owner_repo, args.branch)


def _dispatch_install(args, owner_repo: str, head_ref: str) -> int:
    """Route to env-mode install when a workspace is resolved, else legacy."""
    workspace = _resolve_install_workspace(args)
    if workspace is not None:
        return _run_install_envmode(args, owner_repo, head_ref, workspace)
    return _run_install(args, owner_repo, head_ref)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_EPILOG = """\
Examples:
  # Attach to a running Studio, or install main if none is running:
  python test_unsloth_studio.py --prompt "Find as many bugs as possible in Studio and list them all"
  python test_unsloth_studio.py --prompt "Test the stop button to see if it works"
  python test_unsloth_studio.py --prompt "Try Mistral finetuning and check logs"

  # Probe a specific PR (installs from the PR's head branch into a fresh
  # ./temp/test_unsloth_studio_<ts> workspace via UNSLOTH_STUDIO_HOME):
  python test_unsloth_studio.py --pr 4302 --prompt "Does this PR break the Train page?"

  # Probe a specific branch into an explicit workspace:
  python test_unsloth_studio.py --branch nightly --workspace ./ws_nightly --prompt "smoke-test"

  # Install Studio from main (no PR overlay) into a fresh temp workspace:
  python test_unsloth_studio.py --install-main --prompt "Run a Mistral finetune"

  # Attach only (no auto-install):
  python test_unsloth_studio.py --attach-only --prompt "..."

Returns a plain-text report on stdout. Screenshots land under
--screenshot-dir (default /tmp/test_unsloth_studio/<timestamp>/).
"""


def _default_screenshot_dir() -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    return f"/tmp/test_unsloth_studio/{ts}"


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="test_unsloth_studio.py",
        description=(
            "Drive Unsloth Studio's UI with a natural-language prompt via "
            "Playwright MCP. Fully standalone: auto-installs Playwright MCP "
            "and Studio if neither is present."
        ),
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--prompt", required=True,
                   help="Natural-language task for the agent (required).")

    p.add_argument("--port", type=int, default=None,
                   help="Attach to this port (must be alive). Skips auto-install.")
    p.add_argument("--attach-only", action="store_true",
                   help="Attach to a running Studio or fail. Never install.")
    p.add_argument("--pr", type=_parse_pr_arg, default=None, metavar="PR",
                   help="Install from this PR. Accepts a bare number "
                        "(defaults to unslothai/unsloth), a pull URL, "
                        "or owner/repo#N. Mutually exclusive with --branch.")
    p.add_argument("--branch", default=None, metavar="REF",
                   help="Install from this branch/ref (e.g. main, nightly, "
                        "a SHA). Mutually exclusive with --pr.")
    p.add_argument("--repo", default=None, metavar="OWNER/REPO",
                   help="Override base repo for --branch (and for bare-"
                        "number --pr). Default: unslothai/unsloth.")
    p.add_argument("--password", default=None,
                   help="Override auto-detected bootstrap password.")
    p.add_argument("--screenshot-dir", default=_default_screenshot_dir(),
                   help="Where screenshots land. Default: /tmp/test_unsloth_studio/<ts>/")
    p.add_argument("--model", default="sonnet",
                   help="Model for `claude --print` (default: sonnet).")
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                   help=f"Subprocess budget in seconds (default: {DEFAULT_TIMEOUT}).")
    p.add_argument("--no-playwright-install", action="store_true",
                   help="Skip the Playwright chromium auto-install step.")
    p.add_argument("--verbose", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Stream claude --print output live "
                        "(default: on; pass --no-verbose to disable).")
    p.add_argument("--retries", type=int, default=2, metavar="N",
                   help="Retry the probe N times on transient failure "
                        "(network/MCP/timeout). Exponential backoff "
                        "5s/15s/45s. Default: 2.")
    p.add_argument("--allow-no-screenshots", action="store_true",
                   help="Don't fail with rc=2 if the probe produces zero "
                        "PNGs in --screenshot-dir. Default: fail.")
    p.add_argument("--workspace", default=None, metavar="PATH",
                   help="Install Studio under PATH (sets UNSLOTH_STUDIO_HOME "
                        "for install + run). Highest priority among workspace "
                        "options.")
    p.add_argument("--temp-workspace", action="store_true",
                   help="Auto-create a fresh workspace under ./temp/. Implied "
                        "by --pr / --branch / --install-main when neither "
                        "--workspace nor UNSLOTH_STUDIO_HOME is set.")
    p.add_argument("--install-main", action="store_true",
                   help="Install Unsloth Studio from unslothai/unsloth main "
                        "(no PR overlay). Shortcut for --branch main.")
    return p


def _validate(args, parser: argparse.ArgumentParser) -> None:
    """Reject flag combos that don't make sense."""
    if args.pr is not None and args.branch is not None:
        parser.error("--pr and --branch are mutually exclusive")
    if args.install_main and (args.pr is not None or args.branch is not None):
        parser.error("--install-main is a shortcut for --branch main; do not "
                     "combine with --pr / --branch")
    if args.attach_only and (args.pr is not None or args.branch is not None
                              or args.install_main):
        parser.error("--attach-only cannot be combined with --pr / --branch / "
                     "--install-main (those imply install)")
    if args.port is not None and (args.pr is not None or args.branch is not None
                                   or args.install_main):
        parser.error("--port attaches to an existing Studio; it cannot be "
                     "combined with --pr/--branch/--install-main")
    if args.workspace and args.temp_workspace:
        parser.error("--workspace and --temp-workspace are mutually exclusive")
    if args.install_main:
        # Promote into --branch main so the rest of the code path is uniform.
        args.branch = "main"


def main(argv: list[str] | None = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)
    _validate(args, parser)

    if not args.no_playwright_install:
        _ensure_playwright_chromium()

    # --port: explicit attach, must be alive.
    if args.port is not None:
        port = _discover_port(args.port)
        if port is None:
            print(
                f"[test_unsloth_studio] no Studio responding on 127.0.0.1:{args.port}",
                file=sys.stderr,
            )
            return 2
        return _drive_attached(args, port)

    # --pr / --branch: install the named ref.
    if args.pr is not None or args.branch is not None:
        return _install_from_flags(args)

    # No explicit port/install flags. Try to attach first.
    port = _discover_port(None)
    if port is not None:
        return _drive_attached(args, port)

    # Nothing reachable.
    if args.attach_only:
        print(
            "[test_unsloth_studio] no Studio found on 127.0.0.1:8888..8895 "
            "(--attach-only, not installing)",
            file=sys.stderr,
        )
        return 2

    print(
        f"[test_unsloth_studio] no running Studio found; installing "
        f"{DEFAULT_REPO}@main (first run takes 3-6 min: clone + "
        f"uv pip install + chromium)...",
        file=sys.stderr,
    )
    return _dispatch_install(args, DEFAULT_REPO, "main")


if __name__ == "__main__":
    sys.exit(main())
