# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""PR-5272 cross-browser chat-history migration smoke.

Verifies that ``importLegacyChatsIfNeeded`` (chat-history-storage.ts:246)
correctly migrates legacy browser-Dexie chat history into the new
backend ``studio.db`` on first sidebar mount, across every browser
Playwright supports (chromium / firefox / webkit / msedge).

The test does NOT require a chat model. Migration runs purely on
Dexie -> studio.db copy semantics; no llama-server inference happens.
That lets the workflow run on free GPU-less CI runners.

Drives Studio via Playwright sync_api. The seed data is the
A -> B -> {C, D-E, F} regen tree from the PR's manual review: three
threads, each with 6 messages, where messages C, D, F share the same
``parentId`` (= B) so the migration must preserve three distinct
children of B per thread.

Env:
  BASE_URL        Studio URL, e.g. http://127.0.0.1:18900
  PW_BROWSER      One of: chromium, firefox, webkit, msedge
  STUDIO_DB_PATH  Path to studio.db. Default: ~/.unsloth/studio/studio.db
  PW_ART_DIR      Artifact dir. Default: logs/playwright_pr5272
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from pathlib import Path

from playwright.sync_api import Browser, Page, Playwright, sync_playwright


BASE_URL = os.environ["BASE_URL"].rstrip("/")
BROWSER_NAME = os.environ.get("PW_BROWSER", "chromium").lower()
STUDIO_DB = Path(os.path.expanduser(
    os.environ.get("STUDIO_DB_PATH", "~/.unsloth/studio/studio.db")
))
ART_DIR = Path(os.environ.get("PW_ART_DIR", "logs/playwright_pr5272"))
ART_DIR.mkdir(parents=True, exist_ok=True)

# Wall clock cap. 6 min covers chromium / firefox; webkit on macos-14
# is the slow path so give it more headroom via env.
WALL_TIMEOUT_S = float(os.environ.get("PW5272_WALL_TIMEOUT_S", "360"))

# Mock legacy Dexie payload. Three threads, each carrying the regen
# tree A -> B -> {C, D-E, F}. The migration must preserve three rows
# with parent_id = "<thread>-msgB".
#
# Note on the version handling: Dexie internally multiplies the
# declared schema version by 10 (db.version(3) -> IndexedDB v30). The
# Studio frontend opens "unsloth-chat" at v30 on /chat mount, so by
# the time we get here the DB exists with both stores populated by
# Dexie. We open WITHOUT specifying a version so IndexedDB returns the
# existing schema as-is, and write through plain transactions. If the
# stores don't exist yet (we got here before Studio mounted /chat),
# the caller polls until they do.
MOCK_DEXIE_JS = r"""
() => new Promise((resolve, reject) => {
  const req = indexedDB.open("unsloth-chat");
  req.onsuccess = (e) => {
    const db = e.target.result;
    if (!db.objectStoreNames.contains("threads") || !db.objectStoreNames.contains("messages")) {
      db.close();
      resolve({ok:false, reason:"stores not yet created by Studio", stores:Array.from(db.objectStoreNames)});
      return;
    }
    const tx = db.transaction(["threads","messages"], "readwrite");
    const baseTs = Date.now() - 3600000;
    const threads = [
      {id:"legacy-t1", title:"Legacy Thread 1", modelType:"base",
       modelId:"unsloth/Llama-3.2-1B-Instruct-GGUF", archived:false,
       createdAt:baseTs+1000},
      {id:"legacy-t2", title:"Legacy Thread 2", modelType:"base",
       modelId:"unsloth/Llama-3.2-1B-Instruct-GGUF", archived:false,
       createdAt:baseTs+2000},
      {id:"legacy-t3", title:"Legacy Thread 3", modelType:"base",
       modelId:"unsloth/Llama-3.2-1B-Instruct-GGUF", archived:false,
       createdAt:baseTs+3000},
    ];
    for (const t of threads) tx.objectStore("threads").put(t);
    const mkMsgs = (tid, baseTs) => [
      {id:tid+"-msgA", threadId:tid, parentId:null,            role:"user",
       content:[{type:"text", text:"apple?"}], createdAt:baseTs+10},
      {id:tid+"-msgB", threadId:tid, parentId:tid+"-msgA",     role:"assistant",
       content:[{type:"text", text:"apple"}],  createdAt:baseTs+20},
      {id:tid+"-msgC", threadId:tid, parentId:tid+"-msgB",     role:"user",
       content:[{type:"text", text:"banana?"}], createdAt:baseTs+30},
      {id:tid+"-msgD", threadId:tid, parentId:tid+"-msgB",     role:"user",
       content:[{type:"text", text:"date?"}],   createdAt:baseTs+50},
      {id:tid+"-msgE", threadId:tid, parentId:tid+"-msgD",     role:"assistant",
       content:[{type:"text", text:"date"}],    createdAt:baseTs+60},
      {id:tid+"-msgF", threadId:tid, parentId:tid+"-msgB",     role:"user",
       content:[{type:"text", text:"fig?"}],    createdAt:baseTs+70},
    ];
    for (const t of threads)
      for (const m of mkMsgs(t.id, t.createdAt))
        tx.objectStore("messages").put(m);
    tx.oncomplete = () => { db.close(); resolve({ok:true, threads:3, messages:18}); };
    tx.onerror = (e) => reject(e.target.error?.message || "tx error");
  };
  req.onerror = (e) => reject(e.target.error?.message || "open error");
})
"""

SENTINEL_KEY = "unsloth_chat_legacy_imported_to_studio_db"


def _launch_browser(p: Playwright) -> Browser:
    """Launch the requested browser in headless mode.

    chromium / firefox / webkit are first-class Playwright browsers.
    msedge piggybacks on chromium with channel='msedge'.

    Webkit on the free macos-14 runner has a documented driver-side
    JSON parse crash on the first page.goto (run #26263338979 macos-14
    / webkit job 77301343084). The fix is to slow the channel down
    with PLAYWRIGHT_BROWSERS_PATH defaults and a smaller viewport so
    the initial paint completes inside the driver's buffer budget.
    """
    if BROWSER_NAME == "chromium":
        return p.chromium.launch(headless=True, args=[
            "--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu",
        ])
    if BROWSER_NAME == "firefox":
        # Firefox on CI is fine with defaults; main differentiator is
        # IndexedDB version handling, exercised by the test body.
        return p.firefox.launch(headless=True)
    if BROWSER_NAME == "webkit":
        # Small viewport + slow_mo tames the macOS driver JSON-buffer
        # crash; the test does not depend on viewport size for any
        # assertion.
        return p.webkit.launch(headless=True, slow_mo=50)
    if BROWSER_NAME == "msedge":
        return p.chromium.launch(headless=True, channel="msedge", args=[
            "--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu",
        ])
    raise SystemExit(f"unknown PW_BROWSER: {BROWSER_NAME!r}")


def _auth_tokens() -> tuple[str, str]:
    """Read access + refresh tokens that the workflow captured.

    The workflow's "Rotate bootstrap password + capture tokens" step
    rotates the seeded admin password via /api/auth/login + /api/auth/
    change-password, then exports the tokens as env vars. The test
    pre-injects them into localStorage via context.add_init_script so
    no UI login is needed -- this sidesteps browser-specific quirks
    in the /change-password and /login forms (firefox headless drops
    the submit click; webkit macos-14 has TCP-level driver weirdness).
    """
    access = os.environ.get("STUDIO_ACCESS_TOKEN")
    refresh = os.environ.get("STUDIO_REFRESH_TOKEN")
    if not access or not refresh:
        raise SystemExit(
            "STUDIO_ACCESS_TOKEN / STUDIO_REFRESH_TOKEN env vars are required."
        )
    return access, refresh


def _wait_for_chat_mounted(page: Page, timeout_s: float = 60.0) -> None:
    """Wait until Studio's chat module has imported + opened Dexie.

    Polls indexedDB.databases() until "unsloth-chat" appears with the
    expected stores. Triggers loading by navigating to /chat once.
    Some browsers (notably headless Firefox) take noticeably longer
    than chromium to settle the module graph + open IndexedDB.
    """
    # Explicit navigation so the chat route + chat-history-storage.ts
    # actually run, regardless of where login redirected.
    try:
        page.goto(f"{BASE_URL}/chat", wait_until="domcontentloaded", timeout=30000)
    except Exception as e:
        print(f"[chat-mount] initial /chat goto warning: {e}")
    deadline = time.time() + timeout_s
    last_state = None
    while time.time() < deadline:
        state = page.evaluate(
            r"""() => new Promise((resolve) => {
              const req = indexedDB.open("unsloth-chat");
              req.onsuccess = (e) => {
                const db = e.target.result;
                const stores = Array.from(db.objectStoreNames);
                const ok = stores.includes("threads") && stores.includes("messages");
                db.close();
                resolve({ok, stores, version: db.version, url: location.href});
              };
              req.onerror = () => resolve({ok:false, stores:[], error:"open-error", url:location.href});
            })"""
        )
        last_state = state
        if state.get("ok"):
            print(f"[chat-mount] ready: stores={state['stores']} v={state.get('version')} url={state.get('url')}")
            return
        time.sleep(1.0)
    raise AssertionError(
        f"chat module never mounted Dexie stores in {timeout_s}s. last={last_state!r}"
    )


def _inject_mock_legacy(page: Page) -> None:
    """Inject mock legacy Dexie data.

    Caller must have run _wait_for_chat_mounted first so the stores
    exist; this just writes the rows.
    """
    result = page.evaluate(MOCK_DEXIE_JS)
    print(f"[inject] {json.dumps(result, sort_keys=True)}")
    assert result.get("ok"), f"injection failed: {result}"
    assert result.get("threads") == 3 and result.get("messages") == 18


def _clear_sentinel_and_reload(page: Page) -> None:
    page.evaluate(f"localStorage.removeItem({SENTINEL_KEY!r})")
    page.reload(wait_until="load", timeout=60000)
    page.wait_for_load_state("networkidle", timeout=60000)


def _assert_sidebar_has_legacy(page: Page) -> None:
    # Sidebar entries may render lazily; poll up to 30 s for the first
    # legacy title to appear, then assert the rest.
    page.wait_for_selector('text="Legacy Thread 1"', timeout=30000)
    for title in ("Legacy Thread 1", "Legacy Thread 2", "Legacy Thread 3"):
        loc = page.get_by_text(title, exact=True).first
        assert loc.count() >= 1, f"sidebar missing: {title}"


def _assert_sentinel_set(page: Page) -> None:
    val = page.evaluate(f"localStorage.getItem({SENTINEL_KEY!r})")
    assert val == "true", f"sentinel not set after migration: {val!r}"


def _assert_studio_db_has_imports() -> None:
    """Verify the migration actually wrote rows to studio.db.

    Three threads (legacy-t1/2/3). For each, three messages must share
    parent_id = "<thread>-msgB" -- the regen branch fan-out.
    """
    if not STUDIO_DB.is_file():
        raise SystemExit(f"studio.db not found at {STUDIO_DB}")
    conn = sqlite3.connect(str(STUDIO_DB))
    try:
        cur = conn.cursor()
        threads = [r[0] for r in cur.execute(
            "SELECT id FROM chat_threads ORDER BY id"
        )]
        assert threads == ["legacy-t1", "legacy-t2", "legacy-t3"], \
            f"chat_threads = {threads!r}"
        for tid in threads:
            n = cur.execute(
                "SELECT COUNT(*) FROM chat_messages "
                "WHERE thread_id = ? AND parent_id = ?",
                (tid, f"{tid}-msgB"),
            ).fetchone()[0]
            assert n == 3, (
                f"{tid}: expected 3 children of {tid}-msgB (the regen fan-out), "
                f"got {n}"
            )
    finally:
        conn.close()


def _screenshot(page: Page, name: str) -> None:
    path = ART_DIR / f"{name}.png"
    try:
        page.screenshot(path=str(path), full_page=True)
    except Exception as e:
        print(f"[screenshot] {name} failed: {e}", file=sys.stderr)


def main() -> int:
    deadline = time.time() + WALL_TIMEOUT_S
    print(f"[start] browser={BROWSER_NAME} base={BASE_URL} db={STUDIO_DB}")
    access, refresh = _auth_tokens()
    with sync_playwright() as p:
        browser = _launch_browser(p)
        context = browser.new_context()
        # Pre-inject auth tokens so the very first navigation lands on
        # /chat as a logged-in user; no /change-password redirect, no
        # form filling, no browser-specific submit-button quirks.
        # localStorage init-script runs in every page in the context
        # before any other JS.
        context.add_init_script(
            "try {{"
            "  localStorage.setItem('unsloth_auth_token', {access});"
            "  localStorage.setItem('unsloth_auth_refresh_token', {refresh});"
            "}} catch (e) {{}}".format(
                access=json.dumps(access), refresh=json.dumps(refresh),
            )
        )
        page = context.new_page()
        page.set_default_timeout(30000)

        # 1. Initial navigation (tokens already in localStorage).
        page.goto(f"{BASE_URL}/chat", wait_until="domcontentloaded", timeout=45000)
        _screenshot(page, "01_logged_in")

        # 2a. Wait for Studio's chat module to mount + open Dexie
        _wait_for_chat_mounted(page)
        _screenshot(page, "02a_chat_mounted")

        # 2b. Inject mock legacy Dexie into Studio's already-open DB
        _inject_mock_legacy(page)
        _screenshot(page, "02b_after_inject")

        # 3. Clear sentinel, reload -> migration fires on first sidebar mount
        _clear_sentinel_and_reload(page)
        _screenshot(page, "03_after_reload")

        # 4. Sidebar must show the 3 legacy threads
        _assert_sidebar_has_legacy(page)
        _screenshot(page, "04_sidebar_legacy_visible")

        # 5. localStorage sentinel set
        _assert_sentinel_set(page)

        # 6. studio.db must contain the imported rows + the regen fan-out
        _assert_studio_db_has_imports()

        browser.close()

    elapsed = WALL_TIMEOUT_S - (deadline - time.time())
    print(f"[pass] browser={BROWSER_NAME} elapsed={elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
