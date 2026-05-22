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
MOCK_DEXIE_JS = r"""
() => new Promise((resolve, reject) => {
  const req = indexedDB.open("unsloth-chat", 3);
  req.onupgradeneeded = (e) => {
    const db = e.target.result;
    if (!db.objectStoreNames.contains("threads")) {
      const ts = db.createObjectStore("threads", {keyPath:"id"});
      ts.createIndex("modelType", "modelType");
      ts.createIndex("pairId", "pairId");
      ts.createIndex("archived", "archived");
      ts.createIndex("createdAt", "createdAt");
    }
    if (!db.objectStoreNames.contains("messages")) {
      const ms = db.createObjectStore("messages", {keyPath:"id"});
      ms.createIndex("threadId", "threadId");
      ms.createIndex("createdAt", "createdAt");
    }
  };
  req.onsuccess = (e) => {
    const db = e.target.result;
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
    """
    if BROWSER_NAME == "chromium":
        return p.chromium.launch(headless=True, args=[
            "--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu",
        ])
    if BROWSER_NAME == "firefox":
        return p.firefox.launch(headless=True)
    if BROWSER_NAME == "webkit":
        return p.webkit.launch(headless=True)
    if BROWSER_NAME == "msedge":
        return p.chromium.launch(headless=True, channel="msedge", args=[
            "--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu",
        ])
    raise SystemExit(f"unknown PW_BROWSER: {BROWSER_NAME!r}")


def _read_bootstrap_password() -> str:
    """Studio writes a one-shot admin password on first boot.

    The CI workflow exports STUDIO_BOOTSTRAP_PASSWORD pulled from
    ~/.unsloth/studio/auth/.bootstrap_password so this script doesn't
    have to know the OS-specific home path.
    """
    pw = os.environ.get("STUDIO_BOOTSTRAP_PASSWORD")
    if pw:
        return pw
    candidates = [
        Path("~/.unsloth/studio/auth/.bootstrap_password").expanduser(),
        Path(os.environ.get("UNSLOTH_STUDIO_HOME", "")) / "auth" / ".bootstrap_password",
    ]
    for c in candidates:
        if c.is_file():
            return c.read_text().strip()
    raise SystemExit(
        "Could not find bootstrap password. Set STUDIO_BOOTSTRAP_PASSWORD."
    )


def _login_admin(page: Page) -> None:
    """First-boot Studio shows /change-password. Fill it once."""
    page.goto(BASE_URL, wait_until="load", timeout=60000)
    page.wait_for_load_state("networkidle", timeout=30000)
    if "/change-password" in page.url:
        old = _read_bootstrap_password()
        new = "CrossBrowserCI2026!"
        pw_inputs = page.locator('input[type="password"]')
        # Old / new / confirm. Some flows omit "old"; handle both.
        count = pw_inputs.count()
        if count == 3:
            pw_inputs.nth(0).fill(old)
            pw_inputs.nth(1).fill(new)
            pw_inputs.nth(2).fill(new)
        elif count == 2:
            pw_inputs.nth(0).fill(new)
            pw_inputs.nth(1).fill(new)
        else:
            raise SystemExit(f"unexpected password input count: {count}")
        page.locator('button[type="submit"]').first.click()
        page.wait_for_url("**/chat**", timeout=30000)
    elif "/chat" not in page.url:
        # No change-password redirect; click into chat manually.
        page.goto(f"{BASE_URL}/chat", wait_until="load", timeout=30000)


def _inject_mock_legacy(page: Page) -> None:
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
    with sync_playwright() as p:
        browser = _launch_browser(p)
        context = browser.new_context()
        page = context.new_page()
        page.set_default_timeout(30000)

        # 1. Login
        _login_admin(page)
        _screenshot(page, "01_logged_in")

        # 2. Inject mock legacy Dexie
        _inject_mock_legacy(page)
        _screenshot(page, "02_after_inject")

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
