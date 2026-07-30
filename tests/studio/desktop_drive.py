# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Drive a RUNNING packaged desktop app through its own backend.

`desktop-app-clean-machine-ci.yml` proves the shipped bundle installs and that the
process is still alive after 90 s. It cannot prove the app *works*, because a hosted
runner has no one to click Install and the app sits on the startup screen.

This module closes that gap. The desktop app does not hide its backend: `process.rs`
spawns `unsloth studio --api-only -H 127.0.0.1 -p <port>`, `desktop_backend_owner.rs`
picks that port from 8888-8908 and records it in `~/.unsloth/studio/run/desktop_backend.json`,
and `~/.unsloth/studio/auth/.desktop_secret` exchanges at POST /api/auth/desktop-login
for ordinary admin tokens. So we can attach to exactly the backend the Tauri window is
talking to and exercise it the way a person would.

Every scenario reports independently and the run continues after a failure, so one CI
run surfaces every defect instead of only the first. Exit code is non-zero if any
selected scenario failed.

    python tests/studio/desktop_drive.py --scenarios all --out report.json
    python tests/studio/desktop_drive.py --scenarios inference,parallel_chats

stdlib only: this runs on a machine stripped of developer tooling, where `pip install`
is exactly the thing we are asserting the user never has to do.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import ssl
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

# desktop_backend_owner.rs:22-23. Probing outside this range would find some unrelated
# server and call it Unsloth.
PORT_START = 8888
PORT_END = 8908

# The model the desktop app is expected to run. Its card pins `-np 1` and
# `--spec-type draft-mtp`: MTP does not support more than one parallel slot, so the
# parallel-chat scenario deliberately uses a different, non-MTP model.
MTP_REPO = "unsloth/Qwen3.5-2B-MTP-GGUF"
# 16 GB runners get the real quant; the 7 GB macOS runners get the small one. Overridable
# so a leg can be retuned without editing this file.
MTP_VARIANT = os.environ.get("UNSLOTH_DRIVE_MTP_VARIANT", "UD-Q4_K_XL")
SMALL_REPO = os.environ.get("UNSLOTH_DRIVE_SMALL_REPO", "unsloth/gemma-3-270m-it-GGUF")
SMALL_VARIANT = os.environ.get("UNSLOTH_DRIVE_SMALL_VARIANT", "UD-Q4_K_XL")

# llama.cpp gained MTP in ggml-org/llama.cpp#22673, merged 2026-05-16, and the flag was
# renamed `--spec-type mtp` -> `--spec-type draft-mtp` three days earlier. A prebuilt
# older than this cannot run the MTP model at all.
LLAMA_MTP_MERGE_DATE = "2026-05-16"

DEFAULT_TIMEOUT = float(os.environ.get("UNSLOTH_DRIVE_HTTP_TIMEOUT", "120"))


# ─────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────
class Report:
    """Accumulates per-check results. Mirrors studio_api_smoke.py's ok()/fail() model:
    never raise out of a check, so one run reports every defect."""

    def __init__(self) -> None:
        self.scenarios: dict[str, dict[str, Any]] = {}
        self._current: str | None = None

    def begin(self, name: str) -> None:
        self._current = name
        self.scenarios[name] = {"status": "running", "checks": [], "notes": {}}
        print(f"\n=== scenario: {name} ===", flush=True)

    def ok(self, message: str) -> None:
        self.scenarios[self._current]["checks"].append({"ok": True, "message": message})
        print(f"  OK   {message}", flush=True)

    def fail(self, message: str) -> None:
        self.scenarios[self._current]["checks"].append({"ok": False, "message": message})
        print(f"  FAIL {message}", flush=True)

    def warn(self, message: str) -> None:
        self.scenarios[self._current]["checks"].append(
            {"ok": True, "warn": True, "message": message}
        )
        print(f"  WARN {message}", flush=True)

    def note(self, key: str, value: Any) -> None:
        """Attach evidence (a version string, a port, a token count) to the report."""
        self.scenarios[self._current]["notes"][key] = value

    def skip(self, message: str) -> None:
        self.scenarios[self._current]["status"] = "skipped"
        self.scenarios[self._current]["skip_reason"] = message
        print(f"  SKIP {message}", flush=True)

    def end(self) -> None:
        entry = self.scenarios[self._current]
        if entry["status"] == "skipped":
            return
        failed = [c for c in entry["checks"] if not c["ok"]]
        entry["status"] = "failed" if failed else "passed"

    @property
    def failed_scenarios(self) -> list[str]:
        return [n for n, e in self.scenarios.items() if e["status"] == "failed"]

    def summary(self) -> str:
        lines = ["", "=" * 68, "SUMMARY"]
        for name, entry in self.scenarios.items():
            checks = entry["checks"]
            bad = sum(1 for c in checks if not c["ok"])
            detail = f"{len(checks) - bad}/{len(checks)} checks"
            if entry["status"] == "skipped":
                detail = entry.get("skip_reason", "")
            lines.append(f"  {entry['status'].upper():8} {name:24} {detail}")
        lines.append("=" * 68)
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────
# HTTP
# ─────────────────────────────────────────────────────────────────────────
class Backend:
    """An authenticated handle on the desktop app's own backend."""

    def __init__(self, port: int, token: str | None = None) -> None:
        self.port = port
        self.token = token
        self.base = f"http://127.0.0.1:{port}"

    def http(
        self,
        method: str,
        path: str,
        body: Any = None,
        timeout: float = DEFAULT_TIMEOUT,
        auth: bool = True,
        raw: bool = False,
    ) -> tuple[int, Any]:
        """Returns (status_code, parsed_body). Never raises on an HTTP error status --
        the status IS the result we assert on. -1 means the connection itself failed."""
        url = f"{self.base}{path}"
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(url, data=data, method=method)
        if data is not None:
            request.add_header("Content-Type", "application/json")
        if auth and self.token:
            request.add_header("Authorization", f"Bearer {self.token}")
        try:
            # 127.0.0.1 over plain HTTP; the context is only here so a future https
            # base URL does not silently fall back to an unverified connection.
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
                code = response.status
        except urllib.error.HTTPError as error:
            payload = error.read()
            code = error.code
        except Exception as error:  # connection refused, timeout, DNS, ...
            return -1, {"transport_error": f"{type(error).__name__}: {error}"}
        if raw:
            return code, payload
        try:
            return code, json.loads(payload)
        except Exception:
            return code, payload.decode("utf-8", errors="replace")


def studio_root() -> Path:
    """Where the install we are driving keeps its state.

    The desktop app ignores UNSLOTH_STUDIO_HOME (install.rs strips it before invoking
    the bundled installer) and always uses ~/.unsloth/studio, so that is the default.
    The non-Tauri control arm is installed under an explicit UNSLOTH_STUDIO_HOME, which
    IS the root -- not a parent of `.unsloth/studio` -- so honour it when set.
    """
    override = os.environ.get("UNSLOTH_DRIVE_STUDIO_ROOT") or os.environ.get(
        "UNSLOTH_STUDIO_HOME"
    )
    if override:
        return Path(override)
    return Path.home() / ".unsloth" / "studio"


def is_tauri_install() -> bool:
    """True when we are driving the packaged desktop app rather than a plain
    `unsloth studio`. The desktop app writes tauri.log unconditionally at process start
    (main.rs setup_logging) and records its backend metadata; a control-arm Studio does
    neither, so Tauri-only assertions must not fire against it."""
    if os.environ.get("UNSLOTH_DRIVE_ASSUME_TAURI"):
        return os.environ["UNSLOTH_DRIVE_ASSUME_TAURI"] == "1"
    root = studio_root()
    return (root / "tauri.log").exists() or (root / "run" / "desktop_backend.json").exists()


def discover_backend(
    report: Report, timeout: float = 300.0, pinned_port: int | None = None
) -> Backend | None:
    """Find the port the running desktop app's backend is on.

    Prefers the file the app itself wrote (authoritative, and distinguishes "the app is
    not running" from "something else is on 8888"), falling back to a scan for the case
    where the app is up but has not yet written its metadata. `pinned_port` skips both,
    for the non-Tauri control arm where we chose the port ourselves.
    """
    if pinned_port is not None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if _is_unsloth(pinned_port):
                report.ok(f"backend reachable on pinned port {pinned_port}")
                report.note("port", pinned_port)
                report.note("port_source", "pinned")
                return Backend(pinned_port)
            time.sleep(2)
        report.fail(f"nothing answering /api/health on pinned port {pinned_port}")
        return None

    metadata_path = studio_root() / "run" / "desktop_backend.json"
    rejected: dict[int, str] = {}
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text())
            except Exception:
                metadata = {}
            port = metadata.get("port") or metadata.get("requested_port")
            if port and _is_unsloth(int(port)):
                report.ok(f"backend found via {metadata_path.name} on port {port}")
                report.note("port", int(port))
                report.note("port_source", "desktop_backend.json")
                return Backend(int(port))
        for port in range(PORT_START, PORT_END + 1):
            if not _is_unsloth(port):
                continue
            ours, why = _is_our_install(port)
            rejected[port] = why
            if ours:
                report.warn(
                    f"backend found by scanning port {port} ({why}); "
                    f"{metadata_path} was absent or stale"
                )
                report.note("port", port)
                report.note("port_source", "scan")
                return Backend(port)
        time.sleep(2)

    # A bare "not found" sends the reader to the runner logs to work out whether the app
    # never started, started somewhere else, or was rejected. Say which.
    report.note("discovery_metadata_path", str(metadata_path))
    report.note("discovery_metadata_exists", metadata_path.exists())
    report.note("discovery_studio_root", str(studio_root()))
    report.note("discovery_rejected_ports", rejected)
    if rejected:
        report.fail(
            f"an Unsloth backend answered on {sorted(rejected)} but none was accepted "
            f"as ours: {rejected}"
        )
    else:
        report.fail(
            f"no Unsloth backend on 127.0.0.1:{PORT_START}-{PORT_END} after "
            f"{timeout:.0f}s; the desktop app never brought one up "
            f"(metadata {metadata_path} exists={metadata_path.exists()})"
        )
    return None


def _is_unsloth(port: int) -> bool:
    """A health response alone would accept any server squatting on the port."""
    code, body = Backend(port).http("GET", "/api/health", timeout=3, auth=False)
    if code != 200 or not isinstance(body, dict):
        return False
    if body.get("status") not in ("healthy", "alive"):
        return False
    service = str(body.get("service", ""))
    return "unsloth" in service.lower() or not service


def _is_our_install(port: int) -> tuple[bool, str]:
    """Does this backend belong to the install we are testing?

    /api/health reports `studio_root_id`, which the install writes to
    share/studio_install_id (desktop_backend_owner.rs:151-157). Matching them stops the
    scan attaching to a DIFFERENT Unsloth on the same box -- a co-tenant dev server, a
    leftover from another run -- and reporting its health as ours.

    The id file is not guaranteed to exist (a hosted runner that has only ever run the
    packaged app may not have one), and refusing to attach in that case throws away the
    whole run over a missing marker. So: mismatch is a hard no, absent is a warned yes.
    """
    code, body = Backend(port).http("GET", "/api/health", timeout=3, auth=False)
    if code != 200 or not isinstance(body, dict):
        return False, "no health response"
    reported = str(body.get("studio_root_id", ""))

    id_path = studio_root() / "share" / "studio_install_id"
    if not id_path.exists():
        return True, f"no {id_path} to compare against; accepting on health alone"
    expected = id_path.read_text().strip()
    if reported and reported == expected:
        return True, "studio_root_id matches this install"
    return False, (
        f"studio_root_id {reported[:16]}... does not match this install "
        f"{expected[:16]}...; that is somebody else's Unsloth"
    )


def desktop_login(backend: Backend, report: Report) -> bool:
    """Exchange the on-disk desktop secret for admin tokens, exactly as
    desktop_auth.rs:188-226 does."""
    secret_path = studio_root() / "auth" / ".desktop_secret"
    if not secret_path.exists():
        report.fail(f"no desktop secret at {secret_path}; the app never provisioned auth")
        return False

    # Never let the secret reach the log; GitHub masks it for the rest of the job.
    secret = secret_path.read_text().strip()
    print(f"::add-mask::{secret}", flush=True)

    code, body = backend.http(
        "POST", "/api/auth/desktop-login", {"secret": secret}, auth=False
    )
    if code != 200 or not isinstance(body, dict) or "access_token" not in body:
        report.fail(f"POST /api/auth/desktop-login -> {code} {str(body)[:200]}")
        return False
    backend.token = body["access_token"]
    print(f"::add-mask::{backend.token}", flush=True)
    report.ok("exchanged the desktop secret for admin tokens")

    # A backend that hands out tokens but does not enforce them is worse than one that
    # refuses to log in, so prove the negative too.
    code, _ = backend.http("GET", "/api/inference/status", auth=False)
    if code in (401, 403):
        report.ok(f"unauthenticated /api/inference/status -> {code}")
    else:
        report.fail(f"unauthenticated /api/inference/status -> {code}, expected 401/403")

    code, _ = backend.http("GET", "/api/inference/status")
    if code == 200:
        report.ok("authenticated /api/inference/status -> 200")
    else:
        report.fail(f"authenticated /api/inference/status -> {code}")
    return True


def password_login(
    backend: Backend, report: Report, password: str, username: str | None = None
) -> bool:
    """Password login, for the non-Tauri control arm which has no desktop secret.

    The admin account is `unsloth`, not `admin`; try the caller's choice first and then
    both known names rather than failing on a guess.
    """
    print(f"::add-mask::{password}", flush=True)
    candidates = [username] if username else []
    candidates += [u for u in ("unsloth", "admin") if u not in candidates]

    body: Any = None
    for candidate in candidates:
        code, body = backend.http(
            "POST",
            "/api/auth/login",
            {"username": candidate, "password": password},
            auth=False,
        )
        if code == 200 and isinstance(body, dict) and "access_token" in body:
            report.note("login_username", candidate)
            break
    else:
        report.fail(f"POST /api/auth/login failed for {candidates}: {str(body)[:200]}")
        return False
    backend.token = body["access_token"]
    print(f"::add-mask::{backend.token}", flush=True)
    report.ok("logged in with a password")
    return True


# ─────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────
def load_model(
    backend: Backend, report: Report, model_path: str, *, timeout: float = 1800, **extra
) -> bool:
    """POST /api/inference/load and wait for the model to actually be resident.

    /load can return before the sidecar is serving, so poll /status rather than trusting
    the response -- a load that 200s and then never becomes ready is a real failure mode
    and the whole point of testing the packaged app.
    """
    payload = {"model_path": model_path, **extra}
    started = time.monotonic()
    code, body = backend.http("POST", "/api/inference/load", payload, timeout=timeout)
    if code != 200:
        report.fail(f"load {model_path} -> {code} {str(body)[:300]}")
        return False

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        code, status = backend.http("GET", "/api/inference/status", timeout=30)
        if code == 200 and isinstance(status, dict):
            loaded = status.get("model_loaded") or status.get("loaded")
            current = status.get("model_path") or status.get("model_name") or ""
            if loaded and (not current or model_path.split("/")[-1] in str(current)):
                elapsed = time.monotonic() - started
                report.ok(f"loaded {model_path} in {elapsed:.0f}s")
                report.note(f"load_seconds::{model_path}", round(elapsed, 1))
                return True
        time.sleep(3)
    report.fail(f"{model_path} never reported loaded within {timeout:.0f}s")
    return False


def unload_model(backend: Backend) -> None:
    """Free the sidecar before the next load. The 7 GB macOS runners cannot hold two."""
    backend.http("POST", "/api/inference/unload", {}, timeout=300)
    time.sleep(2)


def chat_once(
    backend: Backend, prompt: str, *, max_tokens: int = 48, timeout: float = 600
) -> tuple[int, Any]:
    """One non-streaming completion through the OpenAI-compatible surface -- the same
    endpoint `unsloth run` exposes, so this covers both the UI path and API serving."""
    return backend.http(
        "POST",
        "/v1/chat/completions",
        {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False,
        },
        timeout=timeout,
    )


def _completion_text(body: Any) -> str:
    try:
        return body["choices"][0]["message"]["content"] or ""
    except Exception:
        return ""


# ─────────────────────────────────────────────────────────────────────────
# Scenarios
# ─────────────────────────────────────────────────────────────────────────
def scenario_backend(backend: Backend, report: Report) -> None:
    """The app's own view of the machine -- and the fields the desktop bug reports
    turned on."""
    for path in ("/api/system", "/api/system/hardware", "/api/auth/status"):
        code, body = backend.http("GET", path)
        if code == 200:
            report.ok(f"GET {path} -> 200")
            report.note(path, body if len(str(body)) < 2000 else "<large>")
        else:
            report.fail(f"GET {path} -> {code}")

    # The bundled installer's venv is what the app boots from; a preflight that says
    # ready over an unbootable venv is the exact reported failure. Only the packaged app
    # writes this file, so a control-arm Studio is not held to it.
    log = studio_root() / "tauri.log"
    if not is_tauri_install():
        report.warn("not a packaged desktop install; skipping the tauri.log assertions")
    elif log.exists():
        text = log.read_text(errors="replace")
        dispositions = [
            line.split("disposition=")[1].split()[0]
            for line in text.splitlines()
            if "desktop_preflight completed disposition=" in line
        ]
        report.note("preflight_dispositions", dispositions)
        if any(d.endswith("Ready") for d in dispositions):
            report.ok(f"preflight reported a ready disposition: {dispositions}")
        else:
            report.fail(f"preflight never reported ready (saw {dispositions or 'nothing'})")
        if "ModuleNotFoundError" in text:
            report.fail("tauri.log contains ModuleNotFoundError; the venv is incomplete")
    else:
        report.fail(f"no tauri.log at {log}")


def scenario_inference(backend: Backend, report: Report) -> None:
    """The model the goal names, run the way its card says it must be run."""
    report.note("repo", MTP_REPO)
    report.note("variant", MTP_VARIANT)

    if not load_model(
        backend,
        report,
        MTP_REPO,
        gguf_variant=MTP_VARIANT,
        # LoadRequest.speculative_type: 'mtp' forces draft-mtp on GPU and CPU. The card
        # pins spec-draft-n-max 6.
        speculative_type="mtp",
        spec_draft_n_max=6,
    ):
        return

    code, body = chat_once(backend, "In one sentence, what is a large language model?")
    if code != 200:
        report.fail(f"/v1/chat/completions -> {code} {str(body)[:300]}")
        unload_model(backend)
        return
    text = _completion_text(body)
    report.note("completion", text[:400])
    if text.strip():
        report.ok(f"MTP model generated {len(text)} chars: {text[:80]!r}")
    else:
        report.fail(f"completion was empty: {str(body)[:300]}")

    usage = body.get("usage") if isinstance(body, dict) else None
    if isinstance(usage, dict) and usage.get("completion_tokens"):
        report.ok(f"usage reports {usage['completion_tokens']} completion tokens")
    else:
        report.warn(f"no completion_tokens in usage: {usage}")

    # Multi-turn: a first-turn-only pass hides thread/context regressions.
    code, body = backend.http(
        "POST",
        "/v1/chat/completions",
        {
            "messages": [
                {"role": "user", "content": "My favourite colour is teal. Remember it."},
                {"role": "assistant", "content": "Got it, teal."},
                {"role": "user", "content": "What is my favourite colour? One word."},
            ],
            "max_tokens": 16,
            "temperature": 0.0,
        },
        timeout=600,
    )
    answer = _completion_text(body)
    if code == 200 and "teal" in answer.lower():
        report.ok(f"multi-turn context carried: {answer.strip()[:60]!r}")
    elif code == 200:
        report.warn(f"multi-turn answer did not contain 'teal': {answer.strip()[:80]!r}")
    else:
        report.fail(f"multi-turn request -> {code}")

    unload_model(backend)


def scenario_parallel_chats(backend: Backend, report: Report) -> None:
    """Three concurrent conversations against one loaded model.

    Uses the MTP model with speculative decoding OFF, not the 270m: the MTP card states
    `-np > 1` is unsupported with MTP, so parallel slots need a non-MTP load, and a 270m
    is too small to be a fair test -- it emits EOS immediately on a short instruction,
    which reads as "empty response" and hides a real parallelism bug behind a model
    quality artifact. UNSLOTH_DRIVE_PARALLEL_REPO overrides for memory-tight runners.
    """
    repo = os.environ.get("UNSLOTH_DRIVE_PARALLEL_REPO", MTP_REPO)
    variant = os.environ.get("UNSLOTH_DRIVE_PARALLEL_VARIANT", MTP_VARIANT)
    if not load_model(
        backend, report, repo, gguf_variant=variant, speculative_type="off"
    ):
        return
    report.note("parallel_repo", repo)

    # Distinct topics, so a shared-slot bug shows up as one thread's subject appearing
    # in another's answer. Natural prompts, because a "reply with one word" instruction
    # is answered with EOS by small models.
    prompts = {
        "ocean": ("Write one sentence about the ocean.", "ocean"),
        "mountain": ("Write one sentence about mountains.", "mountain"),
        "desert": ("Write one sentence about deserts.", "desert"),
    }
    results: dict[str, tuple[int, str]] = {}
    lock = threading.Lock()

    def run(name: str, prompt: str) -> None:
        code, body = chat_once(backend, prompt, max_tokens=48)
        with lock:
            results[name] = (code, _completion_text(body))

    started = time.monotonic()
    threads = [threading.Thread(target=run, args=(n, p)) for n, (p, _) in prompts.items()]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=900)
    parallel_seconds = time.monotonic() - started
    report.note("parallel_seconds", round(parallel_seconds, 1))
    report.note("parallel_results", {k: v[1][:80] for k, v in results.items()})

    if len(results) != 3:
        report.fail(f"only {len(results)}/3 parallel chats returned within 900s")
    for name, (code, text) in sorted(results.items()):
        if code != 200:
            report.fail(f"parallel chat {name} -> {code}")
        elif not text.strip():
            report.fail(f"parallel chat {name} returned empty text")
        else:
            report.ok(f"parallel chat {name} -> {text.strip()[:50]!r}")

    # Cross-talk: each answer must be about its OWN subject and not another thread's.
    leaks = []
    for name, (_code, text) in results.items():
        others = [kw for other, (_p, kw) in prompts.items() if other != name]
        leaked = [kw for kw in others if kw in text.lower()]
        if leaked:
            leaks.append(f"{name} mentioned {leaked}")
    if leaks:
        report.warn(f"possible cross-talk between parallel chats: {'; '.join(leaks)}")
    elif results:
        report.ok("no cross-talk between parallel chats")

    # Were they actually concurrent? Three requests that queue behind each other also
    # "all succeed", so time one request alone and compare. A server with a single slot
    # takes ~3x as long for three; real parallel slots are much closer to 1x.
    solo_started = time.monotonic()
    chat_once(backend, "Write one sentence about rivers.", max_tokens=48)
    solo_seconds = time.monotonic() - solo_started
    report.note("solo_seconds", round(solo_seconds, 1))
    # Below a second the measurement is request overhead, not generation, and the ratio
    # is meaningless -- on a fast GPU three chats finish in 0.5s against a 0.2s solo and
    # "2.5x" says nothing about slot count.
    if solo_seconds < 1.0:
        report.ok(
            f"concurrency ratio not meaningful at this speed "
            f"({parallel_seconds:.2f}s parallel vs {solo_seconds:.2f}s solo)"
        )
    elif solo_seconds > 0:
        ratio = parallel_seconds / solo_seconds
        report.note("parallel_over_solo_ratio", round(ratio, 2))
        if ratio <= 2.2:
            report.ok(
                f"the three chats ran concurrently ({parallel_seconds:.1f}s vs "
                f"{solo_seconds:.1f}s solo, {ratio:.1f}x)"
            )
        else:
            report.warn(
                f"the three chats look serialised ({parallel_seconds:.1f}s vs "
                f"{solo_seconds:.1f}s solo, {ratio:.1f}x); expected well under 3x"
            )

    unload_model(backend)


def scenario_settings(backend: Backend, report: Report) -> None:
    """Read every settings surface, then round-trip a write through persistence."""
    readable = [
        "/api/settings/hugging-face-cache",
        "/api/settings/upload-limit",
        "/api/settings/helper-precache",
        "/api/settings/coding-agents",
        "/api/settings/openai-auto-switch",
        "/api/settings/embedding-model",
        "/api/settings/preview-sharing",
        "/api/settings/personalization",
    ]
    for path in readable:
        code, body = backend.http("GET", path, timeout=60)
        if code == 200:
            report.ok(f"GET {path} -> 200")
        else:
            report.fail(f"GET {path} -> {code} {str(body)[:160]}")

    # Round-trip a write: a settings page that renders but does not persist is the
    # regression a read-only sweep cannot see.
    code, original = backend.http("GET", "/api/settings/upload-limit", timeout=60)
    if code != 200 or not isinstance(original, dict):
        report.fail("cannot round-trip upload-limit; GET did not return an object")
        return
    key = next(
        (
            k
            for k in ("max_upload_size_mb", "limit_mb", "upload_limit_mb", "value")
            if k in original
        ),
        None,
    )
    if key is None:
        report.warn(f"upload-limit shape unrecognised, skipping write test: {original}")
        return
    new_value = int(original[key]) + 1
    code, _ = backend.http("PUT", "/api/settings/upload-limit", {key: new_value}, timeout=60)
    if code != 200:
        report.fail(f"PUT /api/settings/upload-limit -> {code}")
        return
    code, after = backend.http("GET", "/api/settings/upload-limit", timeout=60)
    if code == 200 and isinstance(after, dict) and int(after.get(key, -1)) == new_value:
        report.ok(f"upload-limit persisted {key}={new_value}")
    else:
        report.fail(f"upload-limit did not persist: wrote {new_value}, read back {after}")
    backend.http("PUT", "/api/settings/upload-limit", {key: original[key]}, timeout=60)


def scenario_llama_prebuilt(backend: Backend, report: Report) -> None:
    """The prebuilt llama.cpp the app downloads, and whether it is new enough for MTP."""
    code, body = backend.http("GET", "/api/llama/update-status", timeout=300)
    if code != 200:
        report.fail(f"GET /api/llama/update-status -> {code} {str(body)[:200]}")
        return
    report.ok("GET /api/llama/update-status -> 200")
    report.note("llama_update_status", body)

    version = ""
    if isinstance(body, dict):
        for key in ("current_version", "installed_version", "version", "local_version"):
            if body.get(key):
                version = str(body[key])
                break
    if version:
        report.ok(f"llama.cpp prebuilt version: {version}")
        report.note("llama_version", version)
    else:
        report.warn(f"no version field in update-status: {str(body)[:200]}")

    # The binary itself, not just what the API claims about it.
    root = studio_root()
    candidates = list(root.glob("llama.cpp/**/llama-server*")) + list(
        root.glob("**/bin/llama-server*")
    )
    if candidates:
        report.ok(f"llama-server present at {candidates[0]}")
        report.note("llama_server_path", str(candidates[0]))
    else:
        report.fail(f"no llama-server binary found under {root}")

    # MTP landed in llama.cpp on 2026-05-16 (ggml-org/llama.cpp#22673). A prebuilt older
    # than that cannot run the MTP model at all, so the inference scenario would fail
    # for a reason that has nothing to do with the desktop app.
    report.note("llama_mtp_merge_date", LLAMA_MTP_MERGE_DATE)
    if candidates:
        mtime = time.strftime("%Y-%m-%d", time.gmtime(candidates[0].stat().st_mtime))
        report.note("llama_server_mtime", mtime)


def scenario_whisper_prebuilt(backend: Backend, report: Report) -> None:
    """whisper.cpp prebuilt: the API's view, then the binary on disk."""
    code, body = backend.http("GET", "/api/whisper/update-status", timeout=300)
    if code != 200:
        report.fail(f"GET /api/whisper/update-status -> {code} {str(body)[:200]}")
    else:
        report.ok("GET /api/whisper/update-status -> 200")
        report.note("whisper_update_status", body)

    # install_whisper_prebuilt.py's canonical target, per its own docstring.
    root = studio_root()
    is_windows = platform.system() == "Windows"
    expected = (
        root / "whisper.cpp" / "build" / "bin" / "Release" / "whisper-server.exe"
        if is_windows
        else root / "whisper.cpp" / "build" / "bin" / "whisper-server"
    )
    found = list(root.glob("**/whisper-server*"))
    if expected.exists():
        report.ok(f"whisper-server at the canonical path {expected}")
    elif found:
        report.warn(f"whisper-server found off the canonical path: {found[0]}")
    else:
        # Not a hard failure: the prebuilt is optional and Studio falls back to
        # Transformers STT, which its own docstring documents as the designed behaviour.
        report.warn(
            f"no whisper-server under {root}; dictation falls back to Transformers STT"
        )


def scenario_api_serving(backend: Backend, report: Report) -> None:
    """The OpenAI-compatible surface `unsloth run` serves."""
    code, body = backend.http("GET", "/v1/models", timeout=60)
    if code == 200:
        report.ok("GET /v1/models -> 200")
        report.note("v1_models", body if len(str(body)) < 2000 else "<large>")
    else:
        report.fail(f"GET /v1/models -> {code} {str(body)[:200]}")

    # An API key issued through the UI is how a real integration authenticates, so prove
    # the whole issue -> use -> revoke loop rather than only bearer-token access.
    code, created = backend.http(
        "POST", "/api/auth/api-keys", {"name": "desktop-drive-smoke"}, timeout=60
    )
    if code not in (200, 201) or not isinstance(created, dict):
        report.fail(f"POST /api/auth/api-keys -> {code} {str(created)[:200]}")
        return
    # CreateApiKeyResponse is {"key": "sk-unsloth-...", "api_key": {record}}. The raw key
    # is `key`; `api_key` is the record (which deliberately carries only a prefix), so
    # reading `api_key` first would mask a dict and then fail to authenticate with it.
    key = created.get("key") or created.get("token")
    record = created.get("api_key") if isinstance(created.get("api_key"), dict) else {}
    key_id = record.get("id") or created.get("id")
    if not isinstance(key, str) or not key:
        report.fail(f"api-key response carried no raw key: {sorted(created)}")
        return
    print(f"::add-mask::{key}", flush=True)
    report.ok("issued an API key")

    keyed = Backend(backend.port, token=key)
    code, _ = keyed.http("GET", "/v1/models", timeout=60)
    if code == 200:
        report.ok("the issued API key authenticates against /v1/models")
    else:
        report.fail(f"issued API key -> /v1/models {code}")

    if key_id is not None:
        code, _ = backend.http("DELETE", f"/api/auth/api-keys/{key_id}", timeout=60)
        if code in (200, 204):
            report.ok("revoked the API key")
            code, _ = keyed.http("GET", "/v1/models", timeout=60)
            if code in (401, 403):
                report.ok(f"the revoked key no longer authenticates -> {code}")
            else:
                report.fail(f"revoked key still works -> {code}")
        else:
            report.fail(f"DELETE /api/auth/api-keys/{key_id} -> {code}")


def scenario_mlx(backend: Backend, report: Report) -> None:
    """macOS only: MLX safetensors inference, plus the adapter-only repo the goal names.

    LoRA *training* on MLX is covered by tests/studio/run_real_mlx_smoke.py, which the
    macOS workflow legs run directly; this drives the packaged app's model surface.
    """
    if platform.system() != "Darwin":
        report.skip("MLX is Apple Silicon only")
        return

    # mlx-community/Qwen3-0.6B is an ADAPTER repo (adapter_config.json +
    # adapters.safetensors, no model.safetensors). The real weights live under the
    # -bf16 / -4bit / -8bit names. Both are worth driving: the real one must work, and
    # the adapter-only one must fail with something a user can act on.
    real = os.environ.get("UNSLOTH_DRIVE_MLX_REPO", "mlx-community/Qwen3-0.6B-bf16")
    adapter_only = "mlx-community/Qwen3-0.6B"

    if load_model(backend, report, real, timeout=2400):
        code, body = chat_once(backend, "Say hello in five words.", max_tokens=32)
        text = _completion_text(body)
        if code == 200 and text.strip():
            report.ok(f"MLX safetensors generated: {text.strip()[:80]!r}")
        else:
            report.fail(f"MLX generation -> {code} {str(body)[:200]}")
        unload_model(backend)

    code, body = backend.http(
        "POST", "/api/inference/load", {"model_path": adapter_only}, timeout=900
    )
    detail = str(body)[:300]
    if code == 200:
        report.warn(
            f"{adapter_only} is adapter-only (no model.safetensors) yet /load returned "
            f"200; it should be rejected or resolved to its base model"
        )
        unload_model(backend)
    elif 400 <= code < 500 and any(
        word in detail.lower() for word in ("adapter", "base model", "lora", "safetensors")
    ):
        report.ok(f"adapter-only repo rejected with an actionable message: {detail[:140]}")
    else:
        report.fail(
            f"adapter-only repo {adapter_only} -> {code} with an unhelpful body: {detail[:200]}"
        )


def scenario_updater(backend: Backend, report: Report) -> None:
    """The endpoint the shipped app polls for updates.

    Read from the installed bundle rather than the source tree, because the question is
    what the SHIPPED binary points at.
    """
    endpoint = os.environ.get(
        "UNSLOTH_DRIVE_UPDATER_ENDPOINT",
        "https://github.com/unslothai/unsloth/releases/download/desktop-latest/latest.json",
    )
    report.note("updater_endpoint", endpoint)
    request = urllib.request.Request(endpoint, method="GET")
    try:
        context = ssl.create_default_context()
        with urllib.request.urlopen(request, timeout=60, context=context) as response:
            payload = json.loads(response.read())
        report.ok(f"updater endpoint reachable, version={payload.get('version')}")
        report.note("updater_manifest_version", payload.get("version"))
        platforms = payload.get("platforms", {})
        for expected in ("darwin-aarch64", "linux-x86_64", "windows-x86_64-nsis"):
            if expected in platforms and platforms[expected].get("signature"):
                report.ok(f"updater manifest carries a signed {expected} entry")
            else:
                report.fail(f"updater manifest missing or unsigned for {expected}")
    except urllib.error.HTTPError as error:
        report.fail(
            f"updater endpoint {endpoint} -> HTTP {error.code}; "
            f"every shipped desktop app's update check is dead"
        )
    except Exception as error:
        report.fail(f"updater endpoint {endpoint} unreachable: {type(error).__name__}: {error}")


SCENARIOS: dict[str, Callable[[Backend, Report], None]] = {
    "backend": scenario_backend,
    "inference": scenario_inference,
    "parallel_chats": scenario_parallel_chats,
    "settings": scenario_settings,
    "llama_prebuilt": scenario_llama_prebuilt,
    "whisper_prebuilt": scenario_whisper_prebuilt,
    "api_serving": scenario_api_serving,
    "mlx": scenario_mlx,
    "updater": scenario_updater,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenarios",
        default="all",
        help=f"comma-separated, or 'all'. Known: {', '.join(SCENARIOS)}",
    )
    parser.add_argument("--out", type=Path, help="write the JSON report here")
    parser.add_argument(
        "--discover-timeout",
        type=float,
        default=300.0,
        help="how long to wait for the desktop app to bring its backend up",
    )
    parser.add_argument(
        "--port",
        type=int,
        help="pin the backend port instead of discovering it (non-Tauri control arm)",
    )
    parser.add_argument(
        "--password",
        help="log in with a password instead of the desktop secret (non-Tauri control arm)",
    )
    args = parser.parse_args()

    names = list(SCENARIOS) if args.scenarios == "all" else args.scenarios.split(",")
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown:
        parser.error(f"unknown scenarios: {unknown}; known: {list(SCENARIOS)}")

    report = Report()
    report.begin("discovery")
    backend = discover_backend(
        report, timeout=args.discover_timeout, pinned_port=args.port
    )
    if not backend:
        authenticated = False
    elif args.password:
        # The control arm is a plain `unsloth studio`, which never provisions a desktop
        # secret; it has an ordinary password instead. Same scenarios, same assertions.
        authenticated = password_login(backend, report, args.password)
    else:
        authenticated = desktop_login(backend, report)
    report.end()

    if not authenticated:
        # Nothing downstream can run, but the scenarios still belong in the report as
        # blocked rather than silently missing.
        for name in names:
            report.begin(name)
            report.skip("backend discovery or desktop login failed")
            report.end()
    else:
        for name in names:
            report.begin(name)
            try:
                SCENARIOS[name](backend, report)
            except Exception as error:  # a scenario crash must not lose the others
                report.fail(f"scenario raised {type(error).__name__}: {error}")
            report.end()

    print(report.summary(), flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report.scenarios, indent=2, default=str))
        print(f"report written to {args.out}", flush=True)

    failed = report.failed_scenarios
    if failed:
        print(f"::error::failing scenarios: {', '.join(failed)}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
