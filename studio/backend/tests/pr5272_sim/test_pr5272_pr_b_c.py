"""
Simulation tests for PR-B + PR-C fixes layered on top of PR-A.

PR-B (frontend correctness) is static-checked via grep against the
TypeScript source — runtime testing is the Studio dev-server's job, but
the patches are deterministic so a grep contract is sufficient to detect
regressions.

PR-C (perf) gets both:
  - PR-C2 batched messages endpoint: real HTTP test against the FastAPI
    router (this is the only PR-C item with a backend half).
  - PR-C1, PR-C3 are pure frontend; grep contract only.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import sim_harness

SUB = "test-subject"
OTHER = "other-subject"


@pytest.fixture()
def env():
    home, db, ch = sim_harness.mount()
    yield home, db, ch
    sim_harness.remove_tmp(home)


@pytest.fixture()
def http_env():
    app, db, ch = sim_harness.fresh_app()
    client = TestClient(app)
    yield client, db, ch
    home = Path(os.environ["UNSLOTH_STUDIO_HOME"])
    sim_harness.remove_tmp(home)


def _thread(thread_id="t1", title="T", model_type="base", model_id="m", pair_id=None, archived=False, created_at=1):
    return {
        "id": thread_id, "title": title, "modelType": model_type,
        "modelId": model_id, "pairId": pair_id, "archived": archived,
        "createdAt": created_at,
    }


def _msg(message_id, thread_id="t1", role="user", content="hi", created_at=1):
    return {
        "id": message_id, "threadId": thread_id, "parentId": None, "role": role,
        "content": [{"type": "text", "text": content}],
        "attachments": None, "metadata": None,
        "createdAt": created_at,
    }


# ==========================================================================
# PR-C2: batched messages endpoint
# ==========================================================================


def test_batch_messages_returns_one_per_thread(http_env):
    client, *_ = http_env
    for i in range(3):
        client.post("/api/chat/threads", json=_thread(f"t{i}", created_at=i))
        client.put(f"/api/chat/threads/t{i}/messages/m{i}",
                   json=_msg(f"m{i}", thread_id=f"t{i}", created_at=i))
    r = client.post("/api/chat/messages:batch",
                    json={"thread_ids": ["t0", "t1", "t2"]})
    assert r.status_code == 200
    body = r.json()
    assert set(body["threads"].keys()) == {"t0", "t1", "t2"}
    for k in ("t0", "t1", "t2"):
        assert len(body["threads"][k]) == 1
        assert body["threads"][k][0]["id"] == f"m{k[-1:]}"


def test_batch_messages_unknown_id_returns_empty_list(http_env):
    """Endpoint must not 404 on unknown ids — the typical caller is
    rebuilding a UI index and partial failure is the wrong default."""
    client, *_ = http_env
    client.post("/api/chat/threads", json=_thread("known"))
    client.put("/api/chat/threads/known/messages/m1", json=_msg("m1", thread_id="known"))
    r = client.post("/api/chat/messages:batch",
                    json={"thread_ids": ["known", "ghost"]})
    assert r.status_code == 200
    body = r.json()
    assert len(body["threads"]["known"]) == 1
    assert body["threads"]["ghost"] == []


def test_batch_messages_empty_request(http_env):
    client, *_ = http_env
    r = client.post("/api/chat/messages:batch", json={"thread_ids": []})
    assert r.status_code == 200
    assert r.json()["threads"] == {}


def test_batch_messages_subject_scoped(http_env):
    """PR-A1 + PR-C2 interaction: a caller cannot see another subject's
    messages via the batch endpoint."""
    app, _, ch = sim_harness.fresh_app()
    from auth.authentication import get_current_subject

    current = {"sub": "alice"}
    app.dependency_overrides[get_current_subject] = lambda: current["sub"]
    client = TestClient(app)

    client.post("/api/chat/threads", json=_thread("alice-t1"))
    client.put("/api/chat/threads/alice-t1/messages/am1",
               json=_msg("am1", thread_id="alice-t1"))

    current["sub"] = "bob"
    r = client.post("/api/chat/messages:batch",
                    json={"thread_ids": ["alice-t1"]})
    assert r.status_code == 200
    assert r.json()["threads"]["alice-t1"] == []


def test_batch_messages_chunks_over_900_ids(http_env):
    """The endpoint must handle id lists larger than SQLITE_MAX_VARIABLE_NUMBER
    (defaults to 999 on older builds; the storage layer chunks at 900)."""
    client, *_ = http_env
    n = 1200
    for i in range(n):
        client.post("/api/chat/threads", json=_thread(f"b{i}", created_at=i))
        client.put(f"/api/chat/threads/b{i}/messages/bm{i}",
                   json=_msg(f"bm{i}", thread_id=f"b{i}", created_at=i))
    r = client.post("/api/chat/messages:batch",
                    json={"thread_ids": [f"b{i}" for i in range(n)]})
    assert r.status_code == 200
    body = r.json()
    assert len(body["threads"]) == n
    for i in range(n):
        assert len(body["threads"][f"b{i}"]) == 1


def test_batch_messages_preserves_per_thread_order(http_env):
    """Within each thread the messages must be in created_at ASC order."""
    client, *_ = http_env
    client.post("/api/chat/threads", json=_thread("t"))
    for i in (3, 1, 2):  # insert out of order
        client.put(f"/api/chat/threads/t/messages/m{i}",
                   json=_msg(f"m{i}", thread_id="t", created_at=i))
    r = client.post("/api/chat/messages:batch", json={"thread_ids": ["t"]})
    assert r.status_code == 200
    msgs = r.json()["threads"]["t"]
    assert [m["createdAt"] for m in msgs] == [1, 2, 3]


# ==========================================================================
# PR-B static contracts
# ==========================================================================


def _read(rel: str) -> str:
    return (sim_harness.PR_ROOT.parent / "frontend" / "src" / rel).read_text()


def test_B1_hydrate_failure_sets_hydrated_true():
    """PR-B1: catch block in hydratePersistedSettings must set
    settingsHydrated:true so writes resume after a transient failure."""
    src = _read("features/chat/stores/chat-runtime-store.ts")
    # Locate the catch in hydratePersistedSettings (the one near
    # `warnSettingsPersistenceFailure()` in the hydration body) and confirm
    # it sets settingsHydrated: true.
    idx = src.find("hydratePersistedSettings: async")
    assert idx >= 0
    end = src.find("\n  setModelLoading:", idx)
    body = src[idx : end if end > 0 else len(src)]
    assert "warnSettingsPersistenceFailure()" in body
    assert "set({ settingsHydrated: true })" in body


def test_B2_setparams_bumps_versions_unconditionally():
    """PR-B2: setParams must call getChangedInferenceParams (which bumps
    inferenceParamMutationVersions) even when not hydrated, so late
    hydration's version check doesn't clobber the user's pre-hydrate edit."""
    src = _read("features/chat/stores/chat-runtime-store.ts")
    # Locate setParams handler
    idx = src.find("setParams: (params)")
    assert idx >= 0
    end = src.find("\n  setCustomPresets:", idx)
    body = src[idx : end if end > 0 else len(src)]
    # The buggy version had `if (!state.settingsHydrated) { return { params }; }`
    # BEFORE the getChangedInferenceParams call. The fixed version no longer
    # short-circuits before computing changed params.
    assert "getChangedInferenceParams" in body
    # The early-return-skip-bump pattern must be gone
    assert "if (!state.settingsHydrated)\n        return { params };" not in body
    # Save is now conditional on settingsHydrated, version-bump is not
    assert "state.settingsHydrated && hasKeys(changedParams)" in body or (
        "if (state.settingsHydrated" in body and "saveSettingsPatch" in body
    )


def test_B5_optimistic_delete_tombstone_before_await():
    """PR-B5: deleteChatItem must tombstone synchronously before awaiting
    the backend round-trip, with rollback on failure."""
    src = _read("features/chat/hooks/use-chat-sidebar-items.ts")
    idx = src.find("export async function deleteChatItem")
    assert idx >= 0
    body = src[idx : idx + 2000]
    assert "markChatThreadsDeleted(threadIds)" in body
    # Tombstone call must come BEFORE the await
    tombstone_at = body.find("markChatThreadsDeleted")
    await_at = body.find("await deleteStoredChatThreads")
    assert 0 <= tombstone_at < await_at, (
        "tombstone must run before the backend await"
    )
    # Rollback path must exist
    assert "removeChatThreadTombstones" in body


def test_B6_tombstones_carry_deletedAt_and_have_gc():
    """PR-B6: tombstones store {id, deletedAt} tuples and GC after 90 days."""
    src = _read("features/chat/utils/chat-thread-tombstones.ts")
    assert "deletedAt" in src
    assert "TOMBSTONE_MAX_AGE_MS" in src
    assert "90 * 24 * 60 * 60 * 1000" in src
    assert "removeChatThreadTombstones" in src
    assert "clearAllChatThreadTombstones" in src
    # Legacy plain-string format must still load (back-compat)
    assert 'typeof item === "string"' in src


def test_B4_clearStoredChats_returns_partial_result():
    """PR-B4: clearStoredChats must return a result object distinguishing
    backend/legacy outcomes instead of swallowing failures."""
    src = _read("features/chat/utils/chat-history-storage.ts")
    assert "ClearStoredChatsResult" in src
    assert "result.backend = " in src and "result.legacy = " in src
    assert '"cleared" | "failed" | "skipped"' in src


# ==========================================================================
# PR-C static contracts
# ==========================================================================


def test_C1_sidebar_debounce_and_seq_guard():
    """PR-C1: useChatSidebarItems must debounce the event handler and
    discard stale responses."""
    src = _read("features/chat/hooks/use-chat-sidebar-items.ts")
    assert "SIDEBAR_REFRESH_DEBOUNCE_MS" in src
    assert "300" in src
    assert "requestSeq" in src or "request-id" in src.lower()


def test_C2_frontend_batchListChatMessages_exported():
    """PR-C2 frontend half: batchListChatMessages must exist and fall back
    to per-thread listChatMessages on 404/405."""
    src = _read("features/chat/api/chat-api.ts")
    assert "export async function batchListChatMessages" in src
    assert "/api/chat/messages:batch" in src
    assert 'response.status === 404 || response.status === 405' in src
    # Consumer: listStoredChatThreadsWithMessages must use it
    storage = _read("features/chat/utils/chat-history-storage.ts")
    assert "batchListChatMessages(threadIds)" in storage


def test_C3_settings_write_debounce_with_coalesce():
    """PR-C3: saveSettingsPatch must coalesce into pendingPatch and flush
    on a trailing-edge timer (default 400ms)."""
    src = _read("features/chat/stores/chat-runtime-store.ts")
    assert "SETTINGS_DEBOUNCE_MS" in src
    assert "pendingPatch" in src
    assert "mergePatch" in src
    assert "beforeunload" in src
    # The old "serial chain" pattern must be gone
    assert "settingsSaveQueue = settingsSaveQueue" not in src
