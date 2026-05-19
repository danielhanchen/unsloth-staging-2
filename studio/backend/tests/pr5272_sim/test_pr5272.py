"""
Comprehensive simulation tests for PR #5272.

Each test is self-contained (own tmpdir + own studio_db module reload) so we
can isolate state. Covers:
  - schema idempotency + ALTER ADD COLUMN migration from old shapes
  - basic CRUD round-trips
  - thread + message dedup, chunking boundaries (900 + 1500 + 2000 ids)
  - sync_chat_messages prune_missing semantics
  - cross-thread message hijack via ON CONFLICT(id)
  - clear_chat_history blast radius
  - explicit BEGIN behavior
  - unicode / large content / NULL metadata
  - SQL injection attempts in id/title
  - concurrent settings writers (lost update window)
  - subject-scoping leakage between two subjects
  - export bytesize unbounded behavior
  - cross-platform path safety (PurePosix vs PureWindows, env-var driven roots)
  - legacy preservation contract: PR never deletes Dexie tables (frontend code)
  - HTTP layer: confirm 404/400/auth surfaces from chat_history.py routes

Run: pytest -v test_pr5272.py
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import sim_harness


# ---------- Fixtures -------------------------------------------------- #


@pytest.fixture()
def env():
    """Fresh tmp home + fresh studio_db module for every test."""
    home, db, ch = sim_harness.mount()
    yield home, db, ch
    sim_harness.remove_tmp(home)


@pytest.fixture()
def http_env(monkeypatch):
    """Same but returns a TestClient wired to the FastAPI router."""
    app, db, ch = sim_harness.fresh_app()
    client = TestClient(app)
    yield client, db, ch
    home = Path(os.environ["UNSLOTH_STUDIO_HOME"])
    sim_harness.remove_tmp(home)


def _thread(thread_id="t1", title="T", model_type="base", model_id="m", pair_id=None, archived=False, created_at=1_700_000_000_000):
    return {
        "id": thread_id,
        "title": title,
        "modelType": model_type,
        "modelId": model_id,
        "pairId": pair_id,
        "archived": archived,
        "createdAt": created_at,
    }


def _msg(message_id, thread_id="t1", role="user", content="hi", created_at=1, parent_id=None, metadata=None, attachments=None):
    return {
        "id": message_id,
        "threadId": thread_id,
        "parentId": parent_id,
        "role": role,
        "content": [{"type": "text", "text": content}] if isinstance(content, str) else content,
        "attachments": attachments,
        "metadata": metadata,
        "createdAt": created_at,
    }


# ==========================================================================
# A) SCHEMA + MIGRATION
# ==========================================================================


def test_schema_idempotent_called_twice(env):
    home, db, _ = env
    # Call ensure_schema again on a fresh conn — should not throw
    conn = db.get_connection()
    db._ensure_schema(conn)  # idempotent
    conn.close()
    # And tables still there
    tables = {
        r[0]
        for r in db.get_connection()
        .execute("select name from sqlite_master where type='table'")
        .fetchall()
    }
    assert {"chat_threads", "chat_messages", "chat_settings"} <= tables


def test_migration_from_pre_pr_db(env):
    """Simulate an older studio.db that has chat_threads WITHOUT the
    openai_code_exec_container_id / anthropic_code_exec_container_id columns
    (e.g., from an intermediate PR build). The PR's ALTER ADD COLUMN should
    upgrade it in place without data loss."""
    home, db, _ = env
    db_path = db.studio_db_path()
    # Force-create an "older" version manually, then reset module flag
    # so the PR code runs migration on next get_connection.
    conn = sqlite3.connect(str(db_path))
    conn.execute("DROP TABLE IF EXISTS chat_threads")
    conn.execute("DROP TABLE IF EXISTS chat_messages")
    conn.execute("DROP TABLE IF EXISTS chat_settings")
    conn.execute(
        """CREATE TABLE chat_threads (
            id TEXT NOT NULL PRIMARY KEY,
            title TEXT NOT NULL,
            model_type TEXT NOT NULL,
            model_id TEXT,
            pair_id TEXT,
            archived INTEGER NOT NULL DEFAULT 0,
            created_at INTEGER NOT NULL
        )"""
    )
    conn.execute(
        "INSERT INTO chat_threads (id, title, model_type, created_at) VALUES ('old-1', 'Pre-PR thread', 'base', 1)"
    )
    conn.commit()
    conn.close()
    db._schema_ready = False  # force re-run
    # Open via studio_db — migration should add the two columns
    got = db.get_chat_thread("old-1")
    assert got is not None
    assert got["title"] == "Pre-PR thread"
    cols = {
        r[1]
        for r in db.get_connection()
        .execute("PRAGMA table_info(chat_threads)")
        .fetchall()
    }
    assert "openai_code_exec_container_id" in cols
    assert "anthropic_code_exec_container_id" in cols


# ==========================================================================
# B) CRUD ROUND-TRIPS
# ==========================================================================


def test_basic_thread_crud(env):
    _, db, _ = env
    db.upsert_chat_thread(_thread("t1", title="hi"))
    g = db.get_chat_thread("t1")
    assert g["title"] == "hi"
    db.update_chat_thread("t1", {"title": "renamed"})
    assert db.get_chat_thread("t1")["title"] == "renamed"
    db.delete_chat_threads(["t1"])
    assert db.get_chat_thread("t1") is None


def test_message_round_trip_with_metadata(env):
    _, db, _ = env
    db.upsert_chat_thread(_thread("t1"))
    db.upsert_chat_message(
        _msg("m1", metadata={"timing": {"ms": 42}, "custom": {"k": "v"}})
    )
    got = db.list_chat_messages("t1")
    assert len(got) == 1
    assert got[0]["metadata"]["timing"]["ms"] == 42


def test_cascade_delete_messages_when_thread_deleted(env):
    _, db, _ = env
    db.upsert_chat_thread(_thread("t1"))
    db.upsert_chat_message(_msg("m1"))
    db.delete_chat_threads(["t1"])
    # FK cascade should wipe the messages
    assert db.list_chat_messages("t1") == []


# ==========================================================================
# C) sync_chat_messages PRUNE SEMANTICS + CHUNKING
# ==========================================================================


def test_sync_prune_missing(env):
    _, db, _ = env
    db.upsert_chat_thread(_thread("t1"))
    db.sync_chat_messages(
        "t1",
        [_msg("m1", created_at=1), _msg("m2", created_at=2), _msg("m3", created_at=3)],
        prune_missing=True,
    )
    got = db.sync_chat_messages("t1", [_msg("m2", created_at=2)], prune_missing=True)
    assert [m["id"] for m in got] == ["m2"]


def test_chunking_boundary_999_threads(env):
    """list_chat_messages_for_threads chunks at 900; submit > 999 ids
    (the SQLITE_MAX_VARIABLE_NUMBER on old builds) and confirm all rows return."""
    _, db, _ = env
    n = 1500
    for i in range(n):
        db.upsert_chat_thread(_thread(f"t{i}", model_id="m", created_at=i))
    for i in range(n):
        db.upsert_chat_message(_msg(f"m{i}", thread_id=f"t{i}", created_at=i))
    ids = [f"t{i}" for i in range(n)]
    got = db.list_chat_messages_for_threads(ids)
    assert len(got) == n
    # ordering by created_at
    cas = [m["createdAt"] for m in got]
    assert cas == sorted(cas)


def test_chunking_boundary_at_exactly_900_and_901(env):
    _, db, _ = env
    for n in (900, 901):
        for i in range(n):
            db.upsert_chat_thread(_thread(f"x{n}-{i}", created_at=i))
            db.upsert_chat_message(_msg(f"xm{n}-{i}", thread_id=f"x{n}-{i}", created_at=i))
        ids = [f"x{n}-{i}" for i in range(n)]
        got = db.list_chat_messages_for_threads(ids)
        assert len(got) == n, f"failed at n={n}"


# ==========================================================================
# D) CROSS-THREAD MESSAGE HIJACK (Fork A finding #5/#6)
# ==========================================================================


def test_cross_thread_message_hijack_via_upsert(env):
    """The PR's upsert_chat_message has ON CONFLICT(id) DO UPDATE SET thread_id = excluded.thread_id.
    Demonstrate that a client submitting an existing message id under a DIFFERENT thread
    rewrites the message's thread_id, effectively stealing it across threads."""
    _, db, _ = env
    db.upsert_chat_thread(_thread("t_owner"))
    db.upsert_chat_thread(_thread("t_attacker"))
    db.upsert_chat_message(_msg("shared-id", thread_id="t_owner", content="owner-secret"))
    # Sanity: lives under owner
    assert len(db.list_chat_messages("t_owner")) == 1
    assert len(db.list_chat_messages("t_attacker")) == 0
    # Attacker upserts SAME id under their thread
    db.upsert_chat_message(_msg("shared-id", thread_id="t_attacker", content="stolen"))
    # BUG: owner now has 0 messages, attacker has the message
    assert len(db.list_chat_messages("t_owner")) == 0
    attacker_msgs = db.list_chat_messages("t_attacker")
    assert len(attacker_msgs) == 1
    # And the content was overwritten too
    assert "stolen" in json.dumps(attacker_msgs[0]["content"])


def test_cross_thread_message_hijack_via_sync(env):
    """Same hijack but via sync_chat_messages without prune_missing."""
    _, db, _ = env
    db.upsert_chat_thread(_thread("a"))
    db.upsert_chat_thread(_thread("b"))
    db.upsert_chat_message(_msg("victim", thread_id="a", content="orig"))
    db.sync_chat_messages("b", [_msg("victim", thread_id="b", content="hijack")])
    assert db.list_chat_messages("a") == []
    assert len(db.list_chat_messages("b")) == 1


# ==========================================================================
# E) clear_chat_history BLAST RADIUS
# ==========================================================================


def test_clear_chat_history_wipes_everything(env):
    _, db, _ = env
    for i in range(5):
        db.upsert_chat_thread(_thread(f"t{i}"))
        db.upsert_chat_message(_msg(f"m{i}", thread_id=f"t{i}"))
    db.clear_chat_history()
    assert db.count_chat_threads() == 0
    # Cascade should have removed messages too
    conn = db.get_connection()
    n_msgs = conn.execute("SELECT COUNT(*) FROM chat_messages").fetchone()[0]
    conn.close()
    assert n_msgs == 0


# ==========================================================================
# F) EDGE CASES — unicode, big payloads, NULL metadata, weird ids
# ==========================================================================


def test_unicode_emoji_thread_and_message(env):
    _, db, _ = env
    db.upsert_chat_thread(_thread("t-✨", title="Café 中文 🍜"))
    db.upsert_chat_message(_msg("m-🦄", thread_id="t-✨", content="Hello 世界! 😀"))
    g = db.get_chat_thread("t-✨")
    assert g["title"] == "Café 中文 🍜"
    msgs = db.list_chat_messages("t-✨")
    assert "世界" in json.dumps(msgs[0]["content"], ensure_ascii=False)


def test_large_content_2mb(env):
    _, db, _ = env
    db.upsert_chat_thread(_thread("big"))
    big_text = "A" * (2 * 1024 * 1024)
    db.upsert_chat_message(_msg("bm", thread_id="big", content=big_text))
    msgs = db.list_chat_messages("big")
    assert msgs[0]["content"][0]["text"] == big_text


def test_null_metadata_round_trips(env):
    _, db, _ = env
    db.upsert_chat_thread(_thread("t1"))
    db.upsert_chat_message(_msg("m1", metadata=None, attachments=None))
    got = db.list_chat_messages("t1")
    # PR's row decoder returns None for metadata when the column is NULL
    assert got[0].get("metadata") in (None, {}, [])
    assert got[0].get("attachments") in (None, [])


def test_sql_injection_in_id_and_title(env):
    """SQL injection should fail to inject because the PR uses parameter binding."""
    _, db, _ = env
    nasty_id = "x'); DROP TABLE chat_threads; --"
    db.upsert_chat_thread(_thread(nasty_id, title="'); DROP TABLE chat_threads; --"))
    # If injection worked, the table is gone. Confirm it still works:
    tables = {
        r[0]
        for r in db.get_connection()
        .execute("select name from sqlite_master where type='table'")
        .fetchall()
    }
    assert "chat_threads" in tables
    # And the row is there with the literal id
    got = db.get_chat_thread(nasty_id)
    assert got is not None
    assert got["title"].startswith("')")


def test_message_id_with_slashes_and_paths(env):
    """Test thread/message ids that look like URLs/paths/CRLF (cross-platform)."""
    _, db, _ = env
    for bad in (
        "id/with/slashes",
        "id\\with\\backslashes",
        "id\nwith\nnewlines",
        "id\rwith\rcr",
        "../etc/passwd",
        "C:\\Windows\\System32",
        "id with spaces and 'quotes'",
    ):
        db.upsert_chat_thread(_thread(bad, title=f"title for {bad!r}"))
        db.upsert_chat_message(_msg(f"m-{hash(bad)}", thread_id=bad))
        assert db.get_chat_thread(bad) is not None


# ==========================================================================
# G) CONCURRENT WRITERS — settings lost update window (read-modify-write)
# ==========================================================================


def test_concurrent_settings_writers_can_lose_update(env):
    """upsert_chat_settings is one transaction, but the deep-merge in the
    route is NOT inside a transaction. Concurrent writers can lose updates.
    Simulate by manually doing the read-modify-write race with a tiny sleep."""
    _, db, _ = env
    db.upsert_chat_settings({"autoTitle": False, "reasoningEffort": "low"})

    barrier = threading.Barrier(2)
    final = {}

    def worker(key, value):
        barrier.wait()
        current = db.list_chat_settings()
        time.sleep(0.05)  # widen the window
        merged = dict(current)
        merged[key] = value
        db.upsert_chat_settings(merged)
        final[key] = value

    t1 = threading.Thread(target=worker, args=("autoTitle", True))
    t2 = threading.Thread(target=worker, args=("reasoningEffort", "high"))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    result = db.list_chat_settings()
    # If lost update occurs, ONE of the writes is missing
    lost = (result.get("autoTitle") != True) or (result.get("reasoningEffort") != "high")
    # We expect THIS BUG to manifest sometimes. Document either outcome.
    if lost:
        pytest.skip("Confirmed lost-update window in route-level read-modify-write")


# ==========================================================================
# H) MULTI-SUBJECT LEAKAGE — Fork A finding #1
# ==========================================================================


def test_two_subjects_share_one_chat_space():
    """Confirm: two distinct authenticated identities see/wipe each other's
    chats. Use app.dependency_overrides so FastAPI actually re-resolves."""
    app, db, ch = sim_harness.fresh_app()
    from auth.authentication import get_current_subject

    current = {"sub": "alice"}

    def _dep():
        return current["sub"]

    app.dependency_overrides[get_current_subject] = _dep
    client = TestClient(app)

    r = client.post(
        "/api/chat/threads",
        json={
            "id": "alice-private",
            "title": "Alice's diary",
            "modelType": "base",
            "modelId": "m",
            "pairId": None,
            "archived": False,
            "createdAt": 1,
        },
    )
    assert r.status_code == 200, r.text

    # Switch to bob
    current["sub"] = "bob"
    r = client.get("/api/chat/threads")
    assert r.status_code == 200
    threads = r.json()["threads"]
    bob_sees_alice = any(t["id"] == "alice-private" for t in threads)
    assert bob_sees_alice, "PR currently has no subject scoping — Bob sees Alice"

    # Bob can wipe it
    r = client.delete("/api/chat")
    assert r.status_code == 200

    # Alice now sees nothing
    current["sub"] = "alice"
    r = client.get("/api/chat/threads")
    assert r.json()["threads"] == []


# ==========================================================================
# I) HTTP LAYER — 404, 400, message-id-mismatch
# ==========================================================================


def test_http_404_on_unknown_thread(http_env):
    client, *_ = http_env
    r = client.get("/api/chat/threads/does-not-exist")
    assert r.status_code == 404


def test_http_400_on_message_id_mismatch(http_env):
    client, *_ = http_env
    client.post(
        "/api/chat/threads",
        json=_thread("t1"),
    )
    # Provide a body whose id mismatches the URL message_id
    r = client.put(
        "/api/chat/threads/t1/messages/m1",
        json=_msg("DIFFERENT_ID"),
    )
    assert r.status_code == 400


def test_http_clear_endpoint_unprotected(http_env):
    client, db, ch = http_env
    for i in range(3):
        client.post("/api/chat/threads", json=_thread(f"x{i}"))
    # No confirm param required
    r = client.delete("/api/chat")
    assert r.status_code == 200
    assert client.get("/api/chat/count").json()["count"] == 0


# ==========================================================================
# J) EXPORT BEHAVIOR — unbounded memory; verify it still returns something
# ==========================================================================


def test_export_returns_all_threads_and_messages(http_env):
    client, db, ch = http_env
    for i in range(20):
        client.post("/api/chat/threads", json=_thread(f"e{i}", created_at=i))
        client.put(
            f"/api/chat/threads/e{i}/messages/em{i}",
            json=_msg(f"em{i}", thread_id=f"e{i}", created_at=i),
        )
    r = client.get("/api/chat/export")
    assert r.status_code == 200
    body = r.json()
    assert body["threadCount"] == 20
    assert len(body["threads"]) == 20
    assert len(body["messages"]) == 20


# ==========================================================================
# K) BACKWARDS-COMPAT — confirm no field disappears from frontend Pydantic shape
# ==========================================================================


def test_thread_response_shape_matches_old_dexie_record(http_env):
    """The Dexie ThreadRecord (frontend) has these fields. The route's
    ChatThread Pydantic model must accept and return them all."""
    client, *_ = http_env
    payload = {
        "id": "compat-1",
        "title": "compat",
        "modelType": "base",
        "modelId": "m",
        "pairId": None,
        "archived": False,
        "createdAt": 1,
        "openaiCodeExecContainerId": "oai-c-1",
        "anthropicCodeExecContainerId": "ant-c-1",
    }
    r = client.post("/api/chat/threads", json=payload)
    assert r.status_code == 200
    got = r.json()
    for k in payload:
        assert k in got, f"field {k} missing in response"
    assert got["openaiCodeExecContainerId"] == "oai-c-1"
    assert got["anthropicCodeExecContainerId"] == "ant-c-1"


# ==========================================================================
# L) CROSS-PLATFORM PATH HANDLING
# ==========================================================================


def test_studio_db_path_uses_pathlib(env):
    """studio_db_path returns a pathlib.Path that uses the host OS native
    separator -- `\\` on Windows, `/` on POSIX."""
    import os as _os

    _, db, _ = env
    p = db.studio_db_path()
    assert isinstance(p, Path)
    # Native separator present (Windows: '\\', POSIX: '/')
    assert _os.sep in str(p)
    # Round-trips through Path without changing
    assert Path(str(p)) == p
    # Ends with studio.db on every platform
    assert p.name == "studio.db"


def test_no_hardcoded_posix_paths_in_chat_history_routes():
    """Static check that the PR's new files don't hardcode '/' separators
    in path-like operations."""
    bad = []
    for f in (
        sim_harness.PR_ROOT / "routes" / "chat_history.py",
        sim_harness.PR_ROOT / "storage" / "studio_db.py",
    ):
        text = f.read_text()
        # Look for likely-bad patterns (excluding URL routes and SQL)
        for lineno, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if (
                stripped.startswith("#")
                or stripped.startswith('"""')
                or stripped.startswith("@router")
            ):
                continue
            # File-path concatenation with literal "/" inside open() / Path()
            if 'open("' in line and '"/' in line:
                bad.append((str(f), lineno, line))
    assert not bad, "Likely POSIX-only paths found:\n" + "\n".join(map(str, bad))


# ==========================================================================
# M) LEGACY-PRESERVATION CONTRACT (static frontend check)
# ==========================================================================


def test_frontend_legacy_dexie_is_never_cleared_implicitly():
    """The PR keeps Dexie as a read fallback. Verify the import path does
    NOT call db.threads.clear() or db.messages.clear() automatically — those
    only run on explicit user clear-all."""
    storage = (
        sim_harness.PR_ROOT.parent
        / "frontend"
        / "src"
        / "features"
        / "chat"
        / "utils"
        / "chat-history-storage.ts"
    ).read_text()
    # In importLegacyChatsIfNeeded(), NO db.*.clear() should appear.
    # Match boundary on ANY top-level function/export, not just `async function`.
    import re

    fn_start = storage.find("async function importLegacyChatsIfNeeded")
    assert fn_start >= 0
    rest = storage[fn_start + 1 :]
    # next top-level decl
    m = re.search(r"\n(?:async function |function |export )", rest)
    fn_end = fn_start + 1 + m.start() if m else len(storage)
    body = storage[fn_start:fn_end]
    assert "db.threads.clear" not in body
    assert "db.messages.clear" not in body
    # Confirm those clears DO appear, but only in clearStoredChats (user-initiated).
    clear_idx = storage.find("export async function clearStoredChats")
    assert clear_idx >= 0
    assert "db.threads.clear" in storage[clear_idx : clear_idx + 1000]
    assert "db.messages.clear" in storage[clear_idx : clear_idx + 1000]


def test_frontend_legacy_import_writes_via_upsert_so_idempotent_on_retry():
    """saveChatThread + syncChatMessages used by import are upserts;
    a partial import that retries won't duplicate threads."""
    storage = (
        sim_harness.PR_ROOT.parent
        / "frontend"
        / "src"
        / "features"
        / "chat"
        / "utils"
        / "chat-history-storage.ts"
    ).read_text()
    assert "saveChatThread(thread)" in storage
    assert "syncChatMessages(\n          thread.id" in storage or "syncChatMessages(thread.id" in storage


# ==========================================================================
# N) BEGIN-TRANSACTION ROBUSTNESS (Fork A finding #4)
# ==========================================================================


def test_explicit_begin_works_after_pragma_only(env):
    """Explicit BEGIN must work because PRAGMA foreign_keys doesn't open a
    txn. This is the current-Python behavior, but flags brittleness if a
    future Python changes default isolation_level."""
    _, db, _ = env
    db.upsert_chat_thread(_thread("t"))
    out = db.sync_chat_messages("t", [_msg("a"), _msg("b")], prune_missing=True)
    assert len(out) == 2


# ==========================================================================
# O) MIGRATION SAFETY — adding empty rows during ALTER ADD COLUMN
# ==========================================================================


def test_alter_add_column_default_null_for_existing_rows(env):
    home, db, _ = env
    db_path = db.studio_db_path()
    # Wipe + recreate as pre-PR
    conn = sqlite3.connect(str(db_path))
    conn.execute("DROP TABLE IF EXISTS chat_threads")
    conn.execute("DROP TABLE IF EXISTS chat_messages")
    conn.execute("DROP TABLE IF EXISTS chat_settings")
    conn.execute(
        """CREATE TABLE chat_threads (
            id TEXT NOT NULL PRIMARY KEY,
            title TEXT NOT NULL,
            model_type TEXT NOT NULL,
            model_id TEXT,
            pair_id TEXT,
            archived INTEGER NOT NULL DEFAULT 0,
            created_at INTEGER NOT NULL
        )"""
    )
    conn.execute(
        "INSERT INTO chat_threads (id, title, model_type, created_at) VALUES ('r1','r','base',1)"
    )
    conn.commit()
    conn.close()
    db._schema_ready = False
    got = db.get_chat_thread("r1")
    assert got is not None
    assert got["openaiCodeExecContainerId"] is None
    assert got["anthropicCodeExecContainerId"] is None
