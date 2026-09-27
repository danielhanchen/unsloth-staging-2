"""Scene: the SAME reply bytes rendered by both builds, read back out of the DOM.

For PRs 9012, 9014 and 9054, all three of which are pure performance changes whose
entire claim is that the UI does NOT change. Identical halves are therefore the
THESIS, not the failure, which is what the registry's `parity=True` tells the
driver, and it puts the whole weight on the registry's `expect` strings.

So this scene is written the other way round from the picker scenes: instead of
proving that something MOVED, it has to prove that the specific content named in
`expect` was actually ON SCREEN on both sides. Two matching blank bubbles are the
exact output a missed send, an un-intercepted request or a Studio that never
rebuilt would produce, and they would read as "this PR changed nothing". Every
fact below is therefore read out of the live DOM, and anything missing raises
rather than returning an empty string.

The input is byte-identical on both sides by construction: a fetch shim swaps the
completions request for a crafted SSE body served by an out-of-page HTTP server,
so free-form generation cannot make the two halves render different essays. That
shim needs `open_chat(bypass_csp=True)`, since the served bytes come from another
port and Studio ships `connect-src 'self'`. Everything downstream of the
socket is the real thing -- the real chat adapter, assistant-ui, React, markdown,
Shiki -- built from that side's own tree.

Three payloads, one per PR:

  think_and_placeholder (9012)  a <think> block, an answer, a COMPLETED trailing
                                `${answer}` fragment that must be stripped, and an
                                INCOMPLETE `${unclosed` that must survive. The
                                strip only runs when the adapter decides the
                                request is external, which it does from the
                                selected checkpoint being prefixed `external::`,
                                so `external=True` seeds a connection and selects
                                it. Run non-external the strip is NOT exercised
                                and the facts say so (`strip_exercised`).
  parts_and_tool (9014)         a finished text part, a tool call caught RUNNING
                                at the mid-stream shot and COMPLETE at the settled
                                one, and a closing text part. The running ->
                                complete transition is the point of the pair, so a
                                mid-stream shot that misses it raises.
  plain_thread (9054)           a finished reply with a fenced code block (Shiki on
                                screen) plus a second, truncated turn so the
                                Continue bar renders on the newest message only,
                                with text typed into the composer.

`reply_sha256` is over the settled reply text with the reasoning pane's contents
removed, so the halves can be compared exactly instead of by eye. The reasoning
TRIGGER carries a "Thought for N seconds" duration that is wall clock, not
content, which is why the pane is excluded from the hash and reported separately.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

# Workspace root ONLY. `unsloth_codex/scripts` also holds a copy of
# `studio_test_kit`, and it can be older than the workspace one. This scene needs
# an `open_chat` that accepts `bypass_csp`; without it the shim's cross-port
# fetch is blocked by Studio's `connect-src 'self'` and delivers nothing, at
# which point both sides photograph an empty bubble -- which on a parity pair
# reads as a clean pass rather than a failure.
WORKSPACE = Path(os.environ.get("WORKSPACE") or os.environ.get("UNSLOTH_WORKSPACE") or Path(__file__).resolve().parents[2])
sys.path.insert(0, str(WORKSPACE / "scripts"))
sys.path.insert(0, str(WORKSPACE))

from pr_ui_scenes._common import Session, api_delete, api_get, api_post  # noqa: E402
from studio_test_kit.auth import ProviderSeed, seed_init_script  # noqa: E402
from studio_test_kit.ui import open_chat, send_prompt  # noqa: E402

# ── the fetch shim ───────────────────────────────────────────────────────────
#
# Replace the completions request with a fixed body from a server the page cannot
# slow down, so neither build can render a different essay. Extended to number the
# requests, because plain_thread drives two turns and each needs its own script,
# and because the COUNT is itself a fact -- an unexpected extra completions call
# (auto-title, a retry) would silently eat the next turn's script.
SHIM = r"""
(() => {
  window.__run = { urls: [], intercepted: [], count: 0, done: [], received: [] };
  const SOURCE = "__SOURCE__";
  const realFetch = window.fetch;
  window.fetch = async (...args) => {
    const url = typeof args[0] === "string" ? args[0] : (args[0] && args[0].url) || "";
    window.__run.urls.push(url);
    if (!/chat\/completions|generate\/stream/.test(url)) return realFetch(...args);
    const i = window.__run.count++;
    window.__run.intercepted.push(url);
    window.__run.done[i] = false;
    const upstream = await realFetch(SOURCE + "?i=" + i, { method: "POST" });
    const [a, b] = upstream.body.tee();
    (async () => {
      const reader = b.getReader();
      let n = 0;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        n += value.length;
      }
      window.__run.received[i] = n;
      window.__run.done[i] = true;
    })();
    return new Response(a, {
      status: 200,
      headers: { "Content-Type": "text/event-stream" },
    });
  };
})();
"""

# ── the DOM reader ───────────────────────────────────────────────────────────
#
# Everything the facts are built from. Installed as one object so the driver-side
# polling and the fact collection read the SAME definitions -- a mid-stream
# predicate that disagreed with the settled fact would be its own quiet bug.
#
# Text parts are `div[data-status]` (MarkdownText's root) and tool cards are
# `[data-slot="tool-fallback-root"]`; both are matched inside the message CONTENT
# node, which excludes the footer action bar and its per-run tok/s timing.
READER = r"""
(() => {
  const q = (sel, root) => Array.from((root || document).querySelectorAll(sel));
  const lastContent = () => {
    const n = q('[data-role="assistant"] .aui-assistant-message-content');
    return n.length ? n[n.length - 1] : null;
  };
  // Top-level parts only: a tool card's result is itself rendered with
  // MarkdownText, so a nested div[data-status] would be counted twice.
  const partNodes = (content) => {
    if (!content) return [];
    const all = q('[data-status], [data-slot="tool-fallback-root"]', content)
      .filter((n) => !n.closest('[data-slot="reasoning-root"]'));
    return all.filter((n) => !all.some((m) => m !== n && m.contains(n)));
  };
  const text = (n) => ((n && n.innerText) || "").replace(/\r/g, "");
  window.__parity = {
    replyText() {
      return partNodes(lastContent())
        .map(text)
        .join("\n")
        .trim();
    },
    partCount() {
      return partNodes(lastContent()).length;
    },
    reasoningGroups() {
      const c = lastContent();
      return c ? q('[data-slot="reasoning-root"]', c).length : 0;
    },
    // The CONTENT, not the trigger: the trigger reads "Thought for N seconds",
    // which is wall clock and would differ between the halves for no reason
    // connected to any of these PRs.
    reasoningText() {
      const c = lastContent();
      if (!c) return "";
      // textContent, not innerText: a collapsed pane is hidden and innerText
      // reads "" there, which is indistinguishable from a pane that never got
      // any reasoning at all. Whether it is ON SCREEN is `reasoningExpanded`.
      return q('[data-slot="reasoning-content"]', c)
        .map((n) => (n.textContent || "").replace(/\r/g, ""))
        .join("\n")
        .trim();
    },
    // LAID OUT, not merely mounted. Radix keeps the collapsed content in the
    // DOM with `hidden`, and `innerText` on a hidden node is "" -- so a
    // presence test reported the pane open while the settled shot showed a
    // collapsed bar and the think text read as missing (observed).
    reasoningExpanded() {
      const c = lastContent();
      if (!c) return false;
      return q('[data-slot="reasoning-content"]', c).some(
        (n) => n.offsetHeight > 0,
      );
    },
    toolCards() {
      const c = lastContent();
      return c ? q('[data-slot="tool-fallback-root"]', c).length : 0;
    },
    toolState() {
      const c = lastContent();
      const cards = c ? q('[data-slot="tool-fallback-root"]', c) : [];
      if (!cards.length) return "absent";
      const card = cards[cards.length - 1];
      // The running trigger renders a Spinner in place of the status icon and a
      // duplicate shimmer label; the finished one renders the tick.
      if (card.querySelector('[data-slot="tool-fallback-trigger-shimmer"]')) {
        return "running";
      }
      const result = card.querySelector('[data-slot="tool-fallback-result"]');
      if (result) return "complete";
      // Collapsed: the result lives in a CollapsibleContent that unmounts when
      // shut, so a closed finished card cannot be called complete from here.
      if (card.querySelector('[data-slot="tool-fallback-trigger-icon"]')) {
        return "settled-collapsed";
      }
      return "unknown";
    },
    toolLabel() {
      const c = lastContent();
      const cards = c ? q('[data-slot="tool-fallback-root"]', c) : [];
      if (!cards.length) return "";
      return text(
        cards[cards.length - 1].querySelector(
          '[data-slot="tool-fallback-trigger-label"]',
        ),
      ).trim();
    },
    toolResultText() {
      const c = lastContent();
      const cards = c ? q('[data-slot="tool-fallback-root"]', c) : [];
      if (!cards.length) return "";
      return text(
        cards[cards.length - 1].querySelector(
          '[data-slot="tool-fallback-result"]',
        ),
      ).trim();
    },
    assistantCount() {
      return q('[data-role="assistant"]').length;
    },
    assistantTexts() {
      return q('[data-role="assistant"] .aui-assistant-message-content').map(text);
    },
    continueBars() {
      return q(".aui-continue-bar").length;
    },
    // Which assistant message carries the bar, counted from the end. The whole
    // claim is "newest only", so 0 is the pass and the count alone is not.
    continueBarIndexFromEnd() {
      const msgs = q('[data-role="assistant"]');
      for (let i = 0; i < msgs.length; i += 1) {
        if (msgs[i].querySelector(".aui-continue-bar")) return msgs.length - 1 - i;
      }
      return -1;
    },
    continueBarText() {
      const bar = q(".aui-continue-bar");
      return bar.length ? text(bar[bar.length - 1]).replace(/\n+/g, " ").trim() : "";
    },
    // Shiki emits one styled <span> per token inside the fence. Zero means the
    // block rendered unhighlighted, which IS a visible change.
    //
    // Any inline style counts, not just `color:`. Streamdown is configured with
    // a light/dark theme pair, so the tokens carry `--shiki-light` /
    // `--shiki-dark` custom properties and a substring test for "color" scored
    // a fully highlighted block as zero on the first run.
    highlightedTokens() {
      const c = lastContent();
      if (!c) return 0;
      return q("pre span[style]", c).length;
    },
    highlightStyleSample() {
      const c = lastContent();
      if (!c) return "";
      const s = q("pre span[style]", c);
      return s.length ? s[0].getAttribute("style") : "";
    },
    codeHtmlSample() {
      const c = lastContent();
      if (!c) return "";
      const pre = q("pre", c);
      return pre.length ? pre[0].innerHTML.slice(0, 600) : "";
    },
    codeText() {
      const c = lastContent();
      if (!c) return "";
      const pre = q("pre", c);
      return pre.length ? text(pre[0]).trim() : "";
    },
    composerText() {
      const box = document.querySelector("form:has(textarea) textarea");
      return box ? box.value : null;
    },
  };
})();
"""


def _sse(obj: dict) -> bytes:
    return b"data: " + json.dumps(obj).encode() + b"\n\n"


def _delta(content: str = "", finish: Optional[str] = None) -> bytes:
    choice: dict = {"index": 0, "delta": {"content": content} if content else {}}
    if finish:
        choice["finish_reason"] = finish
    return _sse({"id": "parity", "object": "chat.completion.chunk",
                 "model": "parity-fixed", "choices": [choice]})


_HEAD = _sse({"id": "parity", "object": "chat.completion.chunk",
              "model": "parity-fixed",
              "choices": [{"index": 0,
                           "delta": {"role": "assistant", "content": ""}}]})
_DONE = b"data: [DONE]\n\n"

# A pause long enough that a 250 ms poll cannot miss the mid-stream state. The
# whole point of the mid-stream shot is the transient condition, so it is held
# open deliberately rather than raced for.
HOLD = 4.5
STEP = 0.12


def _scripts(payload: str) -> list[list[tuple[float, bytes]]]:
    """`(delay_before_frame_seconds, frame_bytes)` per SSE response, in order."""
    if payload == "think_and_placeholder":
        return [[
            (0.0, _HEAD),
            (STEP, _delta("<think>weighing the two ")),
            (STEP, _delta("options</think>")),
            # Mid-stream window: reasoning on screen, answer not started.
            (HOLD, _delta("\n\nBoth shapes compile and both keep the tests green, ")),
            (STEP, _delta("but only one of them keeps the reducer flat, ")),
            (STEP, _delta("so the second one wins.")),
            # Completed fragment: the adapter must strip this on an external
            # stream, so it must NOT be on screen at the settled shot.
            (STEP, _delta(" ${answer}")),
            # Incomplete fragment: no closing brace, so the same pattern cannot
            # match and it must SURVIVE. This is the half that catches a strip
            # that got greedier.
            (0.6, _delta(" ${unclosed")),
            (STEP, _delta("", finish="stop")),
            (0.05, _DONE),
        ]]
    if payload == "parts_and_tool":
        return [[
            (0.0, _HEAD),
            (STEP, _delta("Here is what I found:")),
            (STEP, _delta("\n\n")),
            (STEP, _sse({"type": "tool_start", "tool_call_id": "call_0",
                         "tool_name": "lookup_release_notes",
                         "arguments": {"query": "studio streaming notes"}})),
            # Mid-stream window: the card spins while the earlier finished text
            # part sits above it. If hoisting the components map froze that part,
            # this is the shot where it goes stale.
            (HOLD, _sse({"type": "tool_end", "tool_call_id": "call_0",
                         "tool_name": "lookup_release_notes",
                         "result": "3 notes: streaming, tool cards, themes."})),
            (STEP, _delta("\n\nThat is the whole answer.")),
            (STEP, _delta("", finish="stop")),
            (0.05, _DONE),
        ]]
    if payload == "plain_thread":
        return [
            [
                (0.0, _HEAD),
                (STEP, _delta("Here is the loader, formatted:")),
                (STEP, _delta("\n\n```python\n")),
                (STEP, _delta("def load(path):\n")),
                # Mid-stream window: the fence is open and half painted.
                (HOLD, _delta("    with open(path) as handle:\n")),
                (STEP, _delta("        return handle.read()\n```\n")),
                (STEP, _delta("\nThat is the whole loader.")),
                (STEP, _delta("", finish="stop")),
                (0.05, _DONE),
            ],
            [
                (0.0, _HEAD),
                (STEP, _delta("The second answer starts here and then stops")),
                # `length` is what stamps the incomplete marker the Continue bar
                # reads, so this turn is the truncated one.
                (STEP, _delta("", finish="length")),
                (0.05, _DONE),
            ],
        ]
    raise ValueError(f"unknown payload {payload!r}")


class _ScriptedStreamServer:
    """Serves one scripted SSE body per numbered request, out of the page.

    Numbered rather than round-robin: `plain_thread` drives two turns with
    different endings, and an unexpected extra completions request (auto-title,
    a silent retry) would otherwise consume the next turn's script and the run
    would still look clean. `served` records what was actually asked for.
    """

    def __init__(self, scripts: list[list[tuple[float, bytes]]]):
        self.scripts = scripts
        self.served: list[int] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a):  # keep the scene's output readable
                pass

            def _cors(self):
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Headers", "*")
                self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")

            def do_OPTIONS(self):
                self.send_response(204)
                self._cors()
                self.end_headers()

            def do_POST(self):
                query = parse_qs(urlparse(self.path).query)
                try:
                    index = int(query.get("i", ["0"])[0])
                except ValueError:
                    index = 0
                outer.served.append(index)
                script = outer.scripts[min(index, len(outer.scripts) - 1)]
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self._cors()
                self.end_headers()
                for delay, frame in script:
                    if delay:
                        time.sleep(delay)
                    try:
                        self.wfile.write(frame)
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        return

        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        self.port = sock.getsockname()[1]
        sock.close()
        self._srv = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        self._thread = threading.Thread(target=self._srv.serve_forever, daemon=True)
        self._thread.start()

    def close(self):
        self._srv.shutdown()
        self._srv.server_close()


# ── driver-side helpers ──────────────────────────────────────────────────────


async def _wait_for(page, js: str, what: str, timeout_s: float = 90.0,
                    poll_s: float = 0.2):
    """Poll a page predicate, raising with what WAS on screen when it never held.

    The failure message matters more than usual here: a scene whose mid-stream
    condition never fired would otherwise take two settled shots and the pair
    would look perfectly ordinary.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if await page.evaluate(js):
            return
        await asyncio.sleep(poll_s)
    seen = await page.evaluate(
        "() => ({ reply: (window.__parity ? window.__parity.replyText() : '')"
        ".slice(0, 400), tool: window.__parity ? window.__parity.toolState() : '?',"
        " reasoning: (window.__parity ? window.__parity.reasoningText() : '')"
        ".slice(0, 200), run: window.__run })"
    )
    raise RuntimeError(
        f"{what} never happened within {timeout_s}s. On screen: {json.dumps(seen)[:900]}"
    )


async def _await_stream_end(page, index: int, timeout_s: float = 180.0) -> None:
    await _wait_for(page, f"() => window.__run.done[{index}] === true",
                    f"SSE response {index} never finished", timeout_s)
    # The socket closing is not the paint. Wait for the composer's Stop control
    # to go away, which is the app's own "the run is over".
    stop = page.locator('button[aria-label="Stop generating"], '
                        'form:has(textarea) button:has-text("Stop")').first
    try:
        await stop.wait_for(state="hidden", timeout=60_000)
    except Exception:  # noqa: BLE001 -- some turns finish before it ever paints
        pass
    await page.wait_for_timeout(1_200)


def _load_and_wait(session: Session, model: str, variant: str,
                   timeout_s: int = 1800) -> dict:
    """Load the local model both sides will run under.

    The shim replaces the reply, not the app state: with no checkpoint selected
    Studio never issues the completion request at all and the bubble sits on
    "Generating...", which photographs as an empty pair.
    """
    # UnloadRequest.model_path is REQUIRED, so an empty body is a 422 the except
    # below would swallow: the unload would then be a silent no-op on every run,
    # and a reused home would keep whatever model it already had resident. Ask
    # the server what is loaded and unload those by name. `loaded` is a LIST of
    # model ids, not a bool.
    resident = api_get(session, "/api/inference/status").get("loaded") or []
    if isinstance(resident, str):
        resident = [resident]
    for loaded_model in resident:
        api_post(session, "/api/inference/unload",
                 {"model_path": loaded_model}, timeout=300)
    api_post(session, "/api/inference/load",
             {"model_path": model, "gguf_variant": variant, "max_seq_length": 4096},
             timeout=timeout_s)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        time.sleep(5)
        status = api_get(session, "/api/inference/status")
        if status.get("loaded"):
            return status
    raise RuntimeError(f"model did not load within {timeout_s}s")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# Seeded so the scene's turns are the only completions requests on the wire.
# Auto-title in particular issues its own, which would consume a numbered script.
EXTERNAL_MODEL = "parity-model"

_SETTINGS = {
    "unsloth_chat_auto_title": "false",
    "unsloth_chat_tools_enabled": "false",
    "unsloth_chat_code_tools_enabled": "false",
    "unsloth_chat_mcp_enabled": "false",
    "unsloth_chat_confirm_tool_calls": "false",
    "unsloth_chat_deep_research_enabled": "false",
    "unsloth_chat_reasoning_enabled": "true",
}


async def drive(session: Session, out_dir: Path, label: str,
                payload: str = "plain_thread",
                external: bool = False,
                type_in_composer: bool = False,
                model: str = "unsloth/Qwen3.5-2B-MTP-GGUF",
                variant: str = "UD-Q4_K_XL",
                **_: object) -> tuple[list[Path], dict]:
    scripts = _scripts(payload)
    server = _ScriptedStreamServer(scripts)
    shim = SHIM.replace("__SOURCE__", f"http://127.0.0.1:{server.port}/stream")

    facts: dict = {
        "payload": payload,
        "external_requested": bool(external),
        "_server_port": server.port,
        "expected_requests": len(scripts),
        "backend": "scripted SSE, byte-identical on both sides",
    }

    # The scripted SSE server binds a port and runs handler threads, and
    # this scene raises on roughly fifteen paths. Closing it only on the way
    # out leaks both on every one of them.
    try:
        extra: dict = dict(_SETTINGS)
        providers: list[ProviderSeed] = []
        if external:
            # The trailing-fragment strip runs only when the adapter sees an external
            # request, and it decides that from `params.checkpoint` starting with
            # `external::`. The checkpoint itself IS hydrated from localStorage, so
            # that half is seeded below -- but the CONNECTION it names is not:
            # `sync-external-providers` rebuilds `unsloth_chat_external_providers`
            # from `GET /api/providers/` on boot and overwrites whatever was seeded,
            # so a purely localStorage connection is erased a second after the page
            # loads and the turn dies on "Connection not found." (observed).
            #
            # So the connection is created through the API, on the same server that
            # is about to be photographed. `display_name` deliberately differs from
            # the registry's OpenAI entry: the sync then resolves it to the CUSTOM
            # provider type, which is the one type allowed to carry no API key, so
            # nothing here needs a real credential.
            # Idempotent: the homes are reused between runs, so without this every
            # attempt leaves another "Parity Scene" row behind and the two sides
            # stop having the same connection list.
            for existing in api_get(session, "/api/providers/") or []:
                if existing.get("display_name") == "Parity Scene":
                    api_delete(session, f"/api/providers/{existing['id']}")
            created = api_post(session, "/api/providers/", {
                "provider_type": "openai",
                "display_name": "Parity Scene",
                "base_url": None,
                "models": [EXTERNAL_MODEL],
                "available_models": [EXTERNAL_MODEL],
            })
            provider_id = created.get("id")
            if not provider_id:
                raise RuntimeError(f"provider creation returned no id: {created}")
            extra["unsloth_chat_last_external_checkpoint"] = (
                f"external::{provider_id}::{EXTERNAL_MODEL}"
            )
            facts["_external_provider_id"] = provider_id
            facts["_external_checkpoint"] = extra["unsloth_chat_last_external_checkpoint"]
            # No local model on this side: `useChatModelRuntime` re-derives the
            # checkpoint from the backend's active_model, which would take the
            # external selection straight back off again.
        else:
            status = _load_and_wait(session, model, variant)
            # `loaded` is the list of resident model ids; the status payload carries
            # no `model_path`, so there is nothing to fall back to.
            facts["model_loaded"] = status.get("loaded")
            facts["active_model"] = status.get("active_model")

        init = seed_init_script(
            type("A", (), {"access_token": session.access_token,
                           "refresh_token": session.refresh_token})(),
            providers, connections_enabled=True, extra_local_storage=extra,
        )

        shots: list[Path] = []
        video_dir = out_dir / f"{label}_video"
        async with open_chat(session.base_url, init_scripts=[init, shim, READER],
                             viewport=(1500, 1000), headless=True,
                             video_dir=video_dir, video_name=label,
                             bypass_csp=True) as sp:
            page = sp.page
            errors: list[str] = []
            page.on("pageerror", lambda e: errors.append(str(e)[:300]))
            # add_init_script runs before the SPA boots, but READER is re-evaluated
            # here as well so a navigation inside the app cannot leave it undefined.
            await page.evaluate(READER)

            if external:
                # The provider list is rebuilt from the backend a moment after boot.
                # Sending before that lands raises "Connection not found." and the
                # turn never reaches the wire.
                await _wait_for(
                    page,
                    "() => (localStorage.getItem('unsloth_chat_external_providers')"
                    f" || '').includes('{facts['_external_provider_id']}')",
                    "the external connection never synced into the app", 60)
                facts["external_providers_synced"] = await page.evaluate(
                    "() => localStorage.getItem('unsloth_chat_external_providers')")

            if payload == "think_and_placeholder":
                shots += await _turn_think(page, out_dir, label, facts)
            elif payload == "parts_and_tool":
                shots += await _turn_tool(page, out_dir, label, facts)
            elif payload == "plain_thread":
                shots += await _turn_thread(page, out_dir, label, facts,
                                            type_in_composer)
            else:
                raise ValueError(f"unknown payload {payload!r}")

            run = await page.evaluate("() => window.__run") or {}
            facts["intercept_count"] = run.get("count", 0)
            facts["intercepted_urls"] = run.get("intercepted") or []
            facts["received_sse_bytes"] = run.get("received") or []
            facts["scripts_served"] = list(server.served)
            facts["page_errors"] = errors[:5]

            if not facts["intercepted_urls"]:
                facts["urls_seen"] = (run.get("urls") or [])[:15]
                raise RuntimeError(
                    "the shim never intercepted a completions request, so this side "
                    "photographed whatever the app did on its own. urls seen: "
                    f"{facts['urls_seen']}"
                )
            if facts["intercept_count"] != len(scripts):
                raise RuntimeError(
                    f"expected {len(scripts)} completions request(s), saw "
                    f"{facts['intercept_count']} (served {server.served}); an extra "
                    "request consumes a numbered script and the turns diverge"
                )

        if getattr(sp, "video_webm", None):
            facts["_video"] = str(sp.video_webm)
    finally:
        server.close()
    return shots, facts


# ── per-payload turns ────────────────────────────────────────────────────────


async def _shot(page, out_dir: Path, name: str) -> Path:
    path = out_dir / name
    # animations="disabled" is not cosmetic here. The mid-stream shots are taken
    # on purpose while a shimmer or spinner is running, and the two halves are
    # photographed minutes apart, so with Playwright's default ("allow") a looping
    # animation is caught at a different phase on each side. That is a visible
    # difference with no cause in the PR, in a composite whose whole claim is that
    # the halves match.
    await page.screenshot(path=str(path), animations="disabled")
    return path


async def _require(facts: dict, text: str, needle: str, where: str) -> None:
    if needle not in text:
        raise RuntimeError(
            f"{where} does not contain {needle!r}. This is the failure the "
            f"`expect` string exists to catch: two halves can match and still "
            f"show nothing. Got: {text[:500]!r}"
        )


async def _turn_think(page, out_dir: Path, label: str, facts: dict) -> list[Path]:
    await send_prompt(_SP(page), "Which of the two shapes should we ship?")
    await _wait_for(page, "() => window.__run.count > 0",
                    "the completions request was never issued")
    shots = []
    await _wait_for(
        page,
        "() => window.__parity.reasoningExpanded()"
        " && window.__parity.reasoningText().includes('weighing the two options')"
        " && !window.__parity.replyText().includes('second one wins')",
        "the reasoning block never appeared, expanded, before the answer",
    )
    facts["reasoning_text_midstream"] = await page.evaluate(
        "() => window.__parity.reasoningText()")
    facts["reasoning_expanded_midstream"] = await page.evaluate(
        "() => window.__parity.reasoningExpanded()")
    shots.append(await _shot(page, out_dir, f"{label}_0_midstream.png"))

    await _await_stream_end(page, 0)
    # The pane auto-opens while streaming and collapses when the run ends, so it
    # is reopened here -- identically on both sides -- because `expect` asks for
    # the think text to be legible in the settled shot.
    if not await page.evaluate("() => window.__parity.reasoningExpanded()"):
        trigger = page.locator('[data-slot="reasoning-trigger"]').last
        await trigger.click()
        await _wait_for(page, "() => window.__parity.reasoningExpanded()",
                        "the reasoning pane would not reopen", 20)
    await page.wait_for_timeout(600)

    reply = await page.evaluate("() => window.__parity.replyText()")
    reasoning = await page.evaluate("() => window.__parity.reasoningText()")
    facts.update({
        "reply_text": reply,
        "reply_sha256": _sha(reply),
        "reply_chars": len(reply),
        "reasoning_text": reasoning,
        "reasoning_sha256": _sha(reasoning),
        "reasoning_groups": await page.evaluate(
            "() => window.__parity.reasoningGroups()"),
        "reasoning_expanded_settled": await page.evaluate(
            "() => window.__parity.reasoningExpanded()"),
        "part_count": await page.evaluate("() => window.__parity.partCount()"),
        "has_unclosed_fragment": "${unclosed" in reply,
        "completed_fragment_survived": "${answer}" in reply,
        "reply_ends_with_unclosed": reply.rstrip().endswith("${unclosed"),
        # Read off the wire, not off our own kwargs. Whether the scene SEEDED an
        # external connection says nothing about whether the app selected it and
        # the adapter took the external branch, and the strip only runs there.
        "strip_exercised": any(
            "/chat/completions" in url
            for url in await page.evaluate("() => window.__run.intercepted")
        ),
    })
    shots.append(await _shot(page, out_dir, f"{label}_1_settled.png"))

    if not reply:
        raise RuntimeError("the settled reply is empty")
    await _require(facts, reasoning, "weighing the two options", "the reasoning pane")
    await _require(facts, reply, "so the second one wins.", "the settled reply")
    await _require(facts, reply, "${unclosed", "the settled reply")
    # The strip is the entire reason 9012 has a scene, and until this raised it was
    # only RECORDED: a tree where the strip never ran shows `${answer}` on both
    # halves, every other assertion here still passes, and the matching pair gets
    # posted as evidence that the strip is intact.
    if facts["completed_fragment_survived"]:
        raise RuntimeError(
            "the completed `${answer}` fragment is still in the settled reply, so "
            "the trailing-placeholder strip did not run. Both halves would show it "
            "and the pair would read as a clean parity result")
    if not facts["reply_ends_with_unclosed"]:
        raise RuntimeError(
            "the settled reply does not end on the INCOMPLETE `${unclosed`, which "
            "is the fragment the strip must leave alone")
    if not facts["strip_exercised"]:
        raise RuntimeError(
            "no /chat/completions request was intercepted, so the external path "
            "the strip lives on was never taken; `${answer}` being absent would "
            "prove nothing")
    if facts["reasoning_groups"] < 1:
        raise RuntimeError("no reasoning pane rendered at all")
    if not facts["reasoning_expanded_settled"]:
        raise RuntimeError(
            "the reasoning pane is collapsed in the settled shot, so the think "
            "text `expect` asks for is not on screen")
    return shots


async def _turn_tool(page, out_dir: Path, label: str, facts: dict) -> list[Path]:
    await send_prompt(_SP(page), "What is in the release notes?")
    await _wait_for(page, "() => window.__run.count > 0",
                    "the completions request was never issued")
    shots = []
    await _wait_for(
        page,
        "() => window.__parity.toolState() === 'running'"
        " && window.__parity.replyText().includes('Here is what I found:')",
        "the tool card was never caught RUNNING with the earlier text above it",
    )
    facts["tool_state_midstream"] = await page.evaluate(
        "() => window.__parity.toolState()")
    facts["tool_label_midstream"] = await page.evaluate(
        "() => window.__parity.toolLabel()")
    facts["reply_text_midstream"] = await page.evaluate(
        "() => window.__parity.replyText()")
    facts["earlier_part_intact_midstream"] = (
        "Here is what I found:" in facts["reply_text_midstream"])
    shots.append(await _shot(page, out_dir, f"{label}_0_midstream.png"))

    await _await_stream_end(page, 0)
    # The finished card is collapsed by default and its result row unmounts with
    # it, so it is opened -- identically on both sides -- to make "complete"
    # readable as content rather than inferred from a missing spinner.
    if await page.evaluate("() => window.__parity.toolState()") != "complete":
        await page.locator('[data-slot="tool-fallback-trigger"]').last.click()
        await _wait_for(page, "() => window.__parity.toolState() === 'complete'",
                        "the finished tool card never showed its result row", 20)
    await page.wait_for_timeout(600)

    reply = await page.evaluate("() => window.__parity.replyText()")
    facts.update({
        "reply_text": reply,
        "reply_sha256": _sha(reply),
        "reply_chars": len(reply),
        "tool_state_settled": await page.evaluate("() => window.__parity.toolState()"),
        "tool_label_settled": await page.evaluate("() => window.__parity.toolLabel()"),
        "tool_result_text": await page.evaluate(
            "() => window.__parity.toolResultText()"),
        "tool_cards": await page.evaluate("() => window.__parity.toolCards()"),
        "part_count": await page.evaluate("() => window.__parity.partCount()"),
        "earlier_part_intact": "Here is what I found:" in reply,
        "closing_part_present": "That is the whole answer." in reply,
    })
    shots.append(await _shot(page, out_dir, f"{label}_1_settled.png"))

    if not reply:
        raise RuntimeError("the settled reply is empty")
    if facts["tool_state_midstream"] != "running":
        raise RuntimeError(
            f"the mid-stream tool state was {facts['tool_state_midstream']!r}, "
            "not 'running'; the running -> complete transition is the whole "
            "point of this pair"
        )
    if facts["tool_state_settled"] != "complete":
        raise RuntimeError(
            f"the settled tool state was {facts['tool_state_settled']!r}, "
            "not 'complete'")
    await _require(facts, reply, "Here is what I found:", "the settled reply")
    await _require(facts, reply, "That is the whole answer.", "the settled reply")
    await _require(facts, facts["tool_result_text"], "3 notes", "the tool result row")
    return shots


async def _turn_thread(page, out_dir: Path, label: str, facts: dict,
                       type_in_composer: bool) -> list[Path]:
    shots = []
    await send_prompt(_SP(page), "Show me the loader.")
    await _wait_for(page, "() => window.__run.count > 0",
                    "the first completions request was never issued")
    await _wait_for(
        page,
        "() => window.__parity.replyText().includes('def load(path)')"
        " && !window.__parity.replyText().includes('That is the whole loader')",
        "the code fence was never caught half painted",
    )
    facts["reply_text_midstream"] = await page.evaluate(
        "() => window.__parity.replyText()")
    shots.append(await _shot(page, out_dir, f"{label}_0_midstream.png"))
    await _await_stream_end(page, 0)

    # Shiki loads its grammar and theme asynchronously, so the fence paints
    # plain for a moment after the stream ends. Waited for rather than sampled
    # once, or a slow side scores zero and the pair reads as "the PR broke
    # highlighting".
    await _wait_for(page, "() => window.__parity.highlightedTokens() > 0",
                    "the code fence never picked up Shiki highlighting", 30)

    first = await page.evaluate("() => window.__parity.replyText()")
    facts.update({
        "highlight_style_sample": await page.evaluate(
            "() => window.__parity.highlightStyleSample()"),
        "code_html_sample": await page.evaluate(
            "() => window.__parity.codeHtmlSample()"),
        "first_turn_text": first,
        "first_turn_sha256": _sha(first),
        "highlighted_token_count": await page.evaluate(
            "() => window.__parity.highlightedTokens()"),
        "code_text": await page.evaluate("() => window.__parity.codeText()"),
        "continue_bars_after_first_turn": await page.evaluate(
            "() => window.__parity.continueBars()"),
    })
    await _require(facts, first, "def load(path):", "the first reply")
    await _require(facts, first, "That is the whole loader.", "the first reply")
    if facts["highlighted_token_count"] <= 0:
        raise RuntimeError(
            "the code fence rendered with zero Shiki-coloured tokens; the "
            "highlighting is part of what both halves must show")
    if facts["continue_bars_after_first_turn"] != 0:
        raise RuntimeError(
            "a Continue bar rendered on a turn that finished normally "
            f"({facts['continue_bars_after_first_turn']} bars)")

    # Second turn, cut off at `length`, so the bar has a newest message to sit on
    # and an older one it must stay off.
    await send_prompt(_SP(page), "And the second half?")
    await _wait_for(page, "() => window.__run.count > 1",
                    "the second completions request was never issued")
    await _await_stream_end(page, 1)
    await _wait_for(page, "() => window.__parity.continueBars() > 0",
                    "the truncated turn never produced a Continue bar", 30)

    if type_in_composer:
        typed = "does the composer still update"
        box = page.locator("form:has(textarea) textarea").first
        await box.click()
        # type_delay, not fill: this PR is about the work done PER KEYSTROKE, so
        # the composer has to be driven one key at a time.
        if hasattr(box, "press_sequentially"):
            await box.press_sequentially(typed, delay=45)
        else:
            await box.type(typed, delay=45)
        await page.wait_for_timeout(500)
        facts["composer_text"] = await page.evaluate(
            "() => window.__parity.composerText()")
        facts["composer_typed"] = typed
        if facts["composer_text"] != typed:
            raise RuntimeError(
                f"the composer holds {facts['composer_text']!r}, not {typed!r}; "
                "a composer that drops keystrokes is the regression this PR "
                "could cause")

    await page.wait_for_timeout(400)
    reply = await page.evaluate("() => window.__parity.replyText()")
    texts = await page.evaluate("() => window.__parity.assistantTexts()")
    facts.update({
        "reply_text": reply,
        "reply_sha256": _sha(reply),
        "reply_chars": len(reply),
        "assistant_count": await page.evaluate(
            "() => window.__parity.assistantCount()"),
        "assistant_texts": texts,
        "thread_sha256": _sha("\n\n".join(texts)),
        "continue_bar_count": await page.evaluate(
            "() => window.__parity.continueBars()"),
        "continue_bar_index_from_end": await page.evaluate(
            "() => window.__parity.continueBarIndexFromEnd()"),
        "continue_bar_text": await page.evaluate(
            "() => window.__parity.continueBarText()"),
        "part_count": await page.evaluate("() => window.__parity.partCount()"),
    })
    shots.append(await _shot(page, out_dir, f"{label}_1_settled.png"))

    if not reply:
        raise RuntimeError("the settled reply is empty")
    await _require(facts, reply, "The second answer starts here", "the newest reply")
    if facts["continue_bar_count"] != 1:
        raise RuntimeError(
            f"{facts['continue_bar_count']} Continue bars on screen, expected 1")
    if facts["continue_bar_index_from_end"] != 0:
        raise RuntimeError(
            "the Continue bar is not on the newest message "
            f"(index from end {facts['continue_bar_index_from_end']})")
    if facts["assistant_count"] != 2:
        raise RuntimeError(
            f"{facts['assistant_count']} assistant messages, expected 2")
    return shots


class _SP:
    """`send_prompt` wants the kit's StudioPage; only `.page` is read."""

    def __init__(self, page):
        self.page = page
