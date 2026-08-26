"""Old installs, upgrades, downgrades, and the SQLite feature floor.

An existing Studio install has a populated studio.db and, on some machines, a SQLite
that predates half of SQL. These scenarios ask whether this PR can break either one.
"""

import shutil
import sqlite3
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import _harness as H


def _fresh_schema():
    from storage import studio_db

    conn = studio_db.get_connection()
    conn.close()


def _names(kind, like):
    return {
        r["name"] for r in H.raw_query(
            "SELECT name FROM sqlite_master WHERE type=? AND name LIKE ?", (kind, like)
        )
    }


# --- the SQLite floor ---------------------------------------------------------------


def test_the_new_schema_adds_no_sqlite_feature_beyond_what_studio_already_used():
    """WITHOUT ROWID is 3.8.2 (2013). Studio already requires 3.35 via RETURNING.

    If this ever regresses -- someone adds a window function or a STRICT table to the
    generation schema -- an install that works today would stop opening its database.
    """
    from conftest import _BACKEND_ROOT as backend
    new_sql = (backend / "storage" / "chat_generation_runs_db.py").read_text()
    schema = (backend / "storage" / "studio_db.py").read_text()
    start = schema.index("chat_generation_runs")
    block = schema[start - 2000 : start + 6000]

    forbidden = ("STRICT", " OVER (", "GENERATED ALWAYS", "->>", "json_each", "json_extract")
    for token in forbidden:
        assert token not in new_sql, f"new storage module uses {token!r}"
        assert token not in block, f"new schema block uses {token!r}"


def test_the_running_sqlite_is_new_enough_for_the_whole_app():
    version = tuple(int(p) for p in sqlite3.sqlite_version.split("."))
    assert version >= (3, 8, 2), "WITHOUT ROWID needs 3.8.2"
    assert version >= (3, 35, 0), (
        f"Studio already uses RETURNING; {sqlite3.sqlite_version} predates it"
    )


# --- upgrade ------------------------------------------------------------------------


def test_a_legacy_database_keeps_its_history_through_the_migration(tmp_path):
    """Rows written before the PR must survive the new tables being created."""
    from storage import studio_db

    H.seed_thread("legacy-thread", "legacy-user")
    before = H.raw_query("SELECT id, title FROM chat_threads ORDER BY id")
    assert before, "the fixture did not write anything"

    studio_db._schema_ready = False
    conn = studio_db.get_connection()
    conn.close()

    assert H.raw_query("SELECT id, title FROM chat_threads ORDER BY id") == before
    assert "chat_generation_runs" in _names("table", "chat_generation%")
    assert "chat_generation_events" in _names("table", "chat_generation%")


def test_the_migration_is_idempotent_across_repeated_boots():
    from storage import studio_db

    _fresh_schema()
    first = H.raw_query("SELECT name, sql FROM sqlite_master ORDER BY name")
    for _ in range(3):
        studio_db._schema_ready = False
        studio_db.get_connection().close()
    assert H.raw_query("SELECT name, sql FROM sqlite_master ORDER BY name") == first
    assert H.raw_query("PRAGMA integrity_check")[0]["integrity_check"] == "ok"


def test_the_worker_token_column_is_added_and_backfilled_on_an_older_run_table():
    """The one real ALTER: an install already running an earlier commit of this branch.

    Fresh installs get NOT NULL; this path cannot (SQLite has no ADD COLUMN NOT NULL
    without a default), so the backfill is what has to hold the invariant.
    """
    from storage import studio_db

    _fresh_schema()
    conn = sqlite3.connect(str(H.db_path()))
    try:
        # Rebuild the table as it looked before the fencing commit: no worker_token.
        conn.executescript("""
            PRAGMA foreign_keys=OFF;
            DROP TABLE IF EXISTS chat_generation_events;
            DROP TABLE IF EXISTS chat_generation_runs;
            CREATE TABLE chat_generation_runs (
                id TEXT PRIMARY KEY,
                owner_subject TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                user_message_id TEXT NOT NULL,
                assistant_message_id TEXT NOT NULL,
                request_hash TEXT NOT NULL,
                request_json TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'queued',
                cancel_requested INTEGER NOT NULL DEFAULT 0,
                last_event_seq INTEGER NOT NULL DEFAULT 0,
                finish_reason TEXT,
                error_message TEXT,
                created_at INTEGER NOT NULL DEFAULT 0,
                updated_at INTEGER NOT NULL DEFAULT 0,
                started_at INTEGER,
                completed_at INTEGER
            );
            INSERT INTO chat_generation_runs
                (id, owner_subject, thread_id, user_message_id, assistant_message_id,
                 request_hash, request_json)
            VALUES ('old-run', 'alice', 't', 'u', 'a', 'h', '{}');
        """)
        conn.commit()
    finally:
        conn.close()

    studio_db._schema_ready = False
    studio_db.get_connection().close()

    cols = {c["name"]: c for c in H.raw_query("PRAGMA table_info(chat_generation_runs)")}
    assert "worker_token" in cols, "the ALTER did not run"
    token = H.raw_query("SELECT worker_token t FROM chat_generation_runs WHERE id='old-run'")
    assert token and token[0]["t"], "the pre-existing row was not backfilled"
    assert len(token[0]["t"]) == 32, "the backfilled token is not a 16-byte hex value"
    assert H.raw_query("PRAGMA integrity_check")[0]["integrity_check"] == "ok"


def test_two_first_boots_racing_the_schema_do_not_corrupt_it(tmp_path):
    """Two Studio processes started at once against one home.

    Documented, not enforced: measured 10-way, the merge base fails 5/8 rounds with
    `database is locked` or `duplicate column name: settings_seqs` and this branch
    fails 3/8, both from schema code that predates this PR. So the bar here is that
    the file is still a valid database afterwards, which is what a user would notice.
    """
    home = tmp_path / "raced-home"
    home.mkdir()
    from conftest import _BACKEND_ROOT as backend
    script = textwrap.dedent(f"""
        import os, sys
        sys.path.insert(0, {str(backend)!r})
        os.environ["UNSLOTH_STUDIO_HOME"] = {str(home)!r}
        os.environ.setdefault("UNSLOTH_ALLOW_CPU", "1")
        os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
        from storage import studio_db
        studio_db.get_connection().close()
        print("ok")
    """)
    procs = [
        subprocess.Popen([sys.executable, "-c", script],
                         stdout = subprocess.PIPE, stderr = subprocess.PIPE, text = True)
        for _ in range(3)
    ]
    outs = [p.communicate(timeout = 600) for p in procs]
    winners = sum(1 for (out, _err), p in zip(outs, procs) if p.returncode == 0)
    assert winners >= 1, "every racing boot failed:\n" + "\n".join(e[-1500:] for _o, e in outs)

    conn = sqlite3.connect(f"file:{home / 'studio.db'}?mode=ro", uri = True)
    try:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE name LIKE 'chat_generation%'")}
        assert {"chat_generation_runs", "chat_generation_events"} <= tables
    finally:
        conn.close()


# --- downgrade ----------------------------------------------------------------------


@pytest.mark.slow
def test_the_previous_studio_can_still_open_and_use_a_migrated_database(tmp_path):
    """Forwards compatibility: a user upgrades, hits a problem, and rolls back.

    The new BEFORE DELETE trigger and the two new tables stay in the file. Old code
    knows nothing about them, but its own thread delete still fires the trigger, so
    this has to be proven rather than assumed.
    """
    base_backend = Path(__file__).resolve().parents[1] / "wt_base_9187" / "studio" / "backend"
    if not base_backend.is_dir():
        pytest.skip("merge-base worktree not present")

    _fresh_schema()
    H.seed_run()
    assert H.raw_query("SELECT id FROM chat_generation_runs")

    home = tmp_path / "downgrade-home"
    home.mkdir()
    shutil.copy(H.db_path(), home / "studio.db")

    script = textwrap.dedent(f"""
        import os, sys
        sys.path.insert(0, {str(base_backend)!r})
        os.environ["UNSLOTH_STUDIO_HOME"] = {str(home)!r}
        os.environ.setdefault("UNSLOTH_ALLOW_CPU", "1")
        os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
        from storage import studio_db
        studio_db.get_connection().close()
        threads = studio_db.list_chat_threads()
        print("THREADS", len(threads))
        studio_db.delete_chat_threads(["thread-1"])
        print("DELETED")
        studio_db.get_connection().close()
        print("ok")
    """)
    proc = subprocess.run([sys.executable, "-c", script], capture_output = True,
                          text = True, timeout = 600)
    assert proc.returncode == 0, (
        "the previous Studio could not use a migrated database:\n"
        f"{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    )
    assert "DELETED" in proc.stdout and "ok" in proc.stdout

    conn = sqlite3.connect(f"file:{home / 'studio.db'}?mode=ro", uri = True)
    try:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        # The cascade must have taken the run with the thread.
        left = conn.execute("SELECT COUNT(*) FROM chat_generation_runs").fetchone()[0]
        assert left == 0, f"{left} run rows survived a delete by the old build"
    finally:
        conn.close()


# --- growth -------------------------------------------------------------------------


def test_the_tombstone_and_event_rows_are_measured_not_assumed(monkeypatch):
    """No retention policy exists. Record what a run actually costs."""
    import asyncio

    H.seed_run()
    H.script(monkeypatch, [H.text_chunk("hello world ") for _ in range(500)]
             + [H.stop_chunk()])
    asyncio.run(H.supervisor()._produce("run-1"))

    stats = H.raw_query(
        "SELECT COUNT(*) n, SUM(length(payload_json)) b FROM chat_generation_events"
    )[0]
    per_chunk = stats["b"] / max(1, stats["n"])
    print(f"\n  durable cost: {stats['n']} rows, {stats['b']} payload bytes, "
          f"{per_chunk:.0f} bytes/event")
    assert stats["n"] >= 500
    # WITHOUT ROWID is documented as suited to narrow rows; flag if a row gets fat.
    assert per_chunk < 1000, (
        f"{per_chunk:.0f} bytes/event in a WITHOUT ROWID table is outside its design range"
    )
