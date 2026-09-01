# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Keep the turns the rolling context window evicts, and hand them back on request.

Each evicted turn is indexed into a per-thread RAG scope, reusing the store, chunker,
embedder and hybrid retrieval that back attached documents. The stored transcript is
never touched; this is only a search index over turns the projection cannot carry.

Two properties are load-bearing and easy to break:

* The archive is CUMULATIVE. Nothing is cleared or re-scoped per compaction, so the fifth
  compaction can still find what the first evicted. On MRCR v2 over 11 compaction events,
  cumulative scored 0.450 against 0.058 for latest-compaction-only.
* Nothing here may break a chat. Every entry point no-ops on failure, because the
  alternative to a degraded archive is a working conversation, not an error.
"""

from __future__ import annotations

import hashlib
import logging
import bisect
import json
import re
from typing import Optional

from storage import rag_db

from . import config, embeddings, retrieval, store, tool
from .chunking import chunk_pages
from .parsers import Page

logger = logging.getLogger(__name__)

# System prompts are instructions, not conversation: archived, one could come back quoted as an earlier turn.
_SKIP_ROLES = frozenset({"system", "developer"})

# Their results are retrieved passages, so archiving them would feed retrieved text back into its own index.
_INJECTED_CALL_PREFIXES = ("rag_auto_", "conv_recall_")

# A probe carrying render_turn's truncation marker can only be a PREFIX of the live text.
_TRUNCATION_MARKER = " ..."
_MAX_TOOL_RESULT_CHARS = 4000
# Tool arguments are usually short; the cap only stops a pathological blob dominating.
_MAX_TOOL_ARGS_CHARS = 1000
# Over-fetch ahead of the live-branch filter, else stale turns from an abandoned branch starve live matches.
_BRANCH_FILTER_OVERFETCH = 4
# One over-fetch is not always enough: a rewound long continuation can fill any fixed candidate window.
_BRANCH_FILTER_MAX_CANDIDATES = 256


def _text_of(content, *, include_tool_calls: bool = False) -> str:
    """Flatten OpenAI message content to plain text, dropping non-text parts.

    ``include_tool_calls`` also flattens assistant-ui's persisted ``tool-call`` parts,
    whose call lives in structured ``toolName``/``args``/``result`` fields. The branch
    check needs them, or a transcript can never contain the lines ``render_turn`` wrote
    and every archived tool turn looks rolled back.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") in ("text", "input_text") or "text" in part:
                parts.append(str(part.get("text") or ""))
            elif include_tool_calls and part.get("type") == "tool-call":
                for value in (part.get("toolName"), part.get("args"), part.get("result")):
                    if value in (None, "", {}, []):
                        continue
                    parts.append(value if isinstance(value, str) else json.dumps(value))
        return "\n".join(p for p in parts if p)
    return "" if content is None else str(content)


def _probe_text(message: dict) -> str:
    """One message flattened in the ORDER ``render_turn`` writes it.

    The branch check matches in order, so both must agree on it. Bucketing the whole
    message as call, text, result only holds while the text came BEFORE the call. A
    persisted row whose tool call is followed by the model's final answer -- the ordinary
    agent turn -- renders as call, result, answer, and the buckets put the answer in the
    middle: `_scan_probes` advanced its cursor past the answer to find the result and then
    could not find the answer again, so an unchanged evicted tool exchange was classified
    off-branch and no query could return it. Measured end to end: recall came back with
    the user's question alone and the document holding the answer was filtered out.

    So the buckets are flushed the way the replay serializer flushes its pending calls,
    when text arrives after a call. Text before a call still rides ahead of it, which is
    the shape it is written in.
    """
    chunks: list[str] = []
    calls: list[str] = []
    texts: list[str] = []
    results: list[str] = []

    def flush() -> None:
        chunks.extend(part for part in calls + texts + results if part)
        calls.clear()
        texts.clear()
        results.clear()

    def rendered(value) -> list[str]:
        """A structured value as the strings a wire-format copy of it could look like.

        The store keeps tool arguments as an object; the archived copy is the model's raw
        string, with its own spacing. This is a haystack, so offering both is free.
        """
        if isinstance(value, str):
            return [value]
        try:
            spaced = json.dumps(value, ensure_ascii = False)
            compact = json.dumps(value, ensure_ascii = False, separators = (",", ":"))
        except Exception:
            return [str(value)]
        return [spaced] if spaced == compact else [spaced, compact]

    content = message.get("content")
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "tool-call":
                for value in (part.get("toolName"), part.get("args")):
                    if value not in (None, "", {}, []):
                        calls.extend(rendered(value))
                result = part.get("result")
                if result not in (None, "", {}, []):
                    results.extend(rendered(result))
            elif part.get("type") == "reasoning":
                # Skip reasoning on both sides: the wire copy carries text only, so folding it in makes the stored
                # probe longer than the branch and _sticky_compaction_boundary returns 0 forever.
                continue
            elif part.get("type") in ("text", "input_text") or "text" in part:
                if calls:
                    flush()
                texts.append(str(part.get("text") or ""))
    else:
        texts.append(_text_of(content))
    for call in message.get("tool_calls") or []:
        function = (call or {}).get("function") or {}
        for value in (function.get("name"), function.get("arguments")):
            if value:
                calls.append(str(value))
    flush()
    return "\n".join(chunks)


def _is_injected(message: dict) -> bool:
    call_id = str(message.get("tool_call_id") or "")
    if call_id.startswith(_INJECTED_CALL_PREFIXES):
        return True
    for call in message.get("tool_calls") or []:
        if str((call or {}).get("id") or "").startswith(_INJECTED_CALL_PREFIXES):
            return True
    return False


def render_turn(group: list[dict]) -> str:
    """Render one evicted turn group as the text that gets indexed and quoted back.

    Both sides go in one document on purpose, so retrieval returns the question with its
    answer rather than a floating fragment.
    """
    lines: list[str] = []
    for message in group:
        role = str(message.get("role") or "")
        if role in _SKIP_ROLES:
            continue
        text = _text_of(message.get("content")).strip()
        calls = message.get("tool_calls") or []
        if calls:
            # Archive the arguments, bounded, not just the name: the command or query that ran is the searchable
            # substance.
            for call in calls:
                function = (call or {}).get("function") or {}
                name = str(function.get("name") or "tool")
                arguments = str(function.get("arguments") or "").strip()
                if len(arguments) > _MAX_TOOL_ARGS_CHARS:
                    arguments = arguments[:_MAX_TOOL_ARGS_CHARS] + _TRUNCATION_MARKER
                lines.append(
                    f"assistant called {name}: {arguments}"
                    if arguments
                    else f"assistant called {name}"
                )
            if text:
                lines.append(f"{role or 'assistant'}: {text}")
            continue
        if not text:
            continue
        if role == "tool":
            # Tool results are huge and the least useful thing to quote back verbatim: keep enough to be
            # searchable without bloating the index.
            if len(text) > _MAX_TOOL_RESULT_CHARS:
                text = text[:_MAX_TOOL_RESULT_CHARS] + _TRUNCATION_MARKER
            lines.append(f"tool result: {text}")
        else:
            lines.append(f"{role or 'message'}: {text}")
    return "\n".join(lines).strip()


def _archivable(group: list[dict]) -> list[dict]:
    """The part of an evicted turn worth archiving, or an empty list.

    Our own injections come out, the rest stays. Rejecting the whole group threw away real
    answers: ``group_turns`` keeps a tool call, its result and the following reply in ONE
    group, so an injection on that turn took the model's answer down with it, leaving the
    question archived and the answer not -- on compaction turns, which are the point here.
    """
    if any(str(message.get("role") or "") in _SKIP_ROLES for message in group):
        return []
    kept = [message for message in group if not _is_injected(message)]
    return _without_retrieval(kept)


def _retrieval_names() -> frozenset:
    """The tool names whose results are retrieved passages, not conversation."""
    try:
        from core.inference.tool_call_parser import RAG_SEARCH_TOOLS
        return frozenset(RAG_SEARCH_TOOLS)
    except Exception:  # noqa: BLE001 -- fall back to the name this module owns
        return frozenset({"search_conversation", "search_knowledge_base"})


def _without_retrieval(group: list[dict]) -> list[dict]:
    """The group with retrieval calls and their results removed.

    `_is_injected` recognises the ids this feature and the RAG auto-inject generate, but a
    search the MODEL asked for gets an ordinary `call_N` id from the parser, so both the
    call and its retrieved passage were archived as new conversation. Measured: a second
    search then indexed the first one's output inside its own, one nesting level per
    distinct search, each copy competing for the four recall slots.

    Removed by NAME, and only the retrieval parts: an assistant message can carry a
    retrieval call alongside an ordinary one, and the reply that follows a search is real
    conversation that must still be archived.
    """
    names = _retrieval_names()
    dropped_ids = set()
    out = []
    for message in group:
        calls = message.get("tool_calls") or []
        if calls:
            keep_calls = []
            for call in calls:
                function = (call or {}).get("function") or {}
                if str(function.get("name") or "") in names:
                    dropped_ids.add(str((call or {}).get("id") or ""))
                else:
                    keep_calls.append(call)
            if len(keep_calls) != len(calls):
                if not keep_calls and not _text_of(message.get("content")).strip():
                    continue
                message = {**message, "tool_calls": keep_calls}
        if str(message.get("role") or "") == "tool":
            if str(message.get("tool_call_id") or "") in dropped_ids:
                continue
        out.append(message)
    return out


def enabled() -> bool:
    """Whether the archive can actually run here.

    ``rag_available()`` and not ``RAG_AVAILABLE``: the flag only records that
    ``import sqlite_vec`` worked, while the vec0 native library is a separate file a venv
    can be missing (the common macOS case). Trusting it there is worse than no feature at
    all: the fit still reserves room, then write and recall both fail, so the user pays
    extra eviction for content that never arrives.
    """
    return bool(config.CONVERSATION_ARCHIVE) and bool(rag_db.rag_available())


def can_archive(thread_id: Optional[str]) -> bool:
    """Whether this thread's evicted turns can be archived at all.

    A temporary (incognito) chat is never written to studio.db, yet the frontend still
    sends its thread_id and the request carries no incognito flag, so saved messages are
    the only signal. Archiving one would persist exactly what the user asked not to keep,
    into a scope with no thread row that no deletion flow could reach. An API client that
    sends a thread_id without persisting anything is excluded for the same reason.

    Also gates the fit's recall reserve: a thread that cannot be archived cannot be
    recalled, so reserving there would evict history for content that cannot arrive.
    """
    if not thread_id or not enabled():
        return False
    try:
        from storage import studio_db
        return bool(studio_db.chat_thread_has_messages(thread_id))
    except Exception:
        return False


def archive_turns(
    thread_id: str,
    evicted: list[dict],
    live: Optional[list[dict]] = None,
    branch: Optional[list[dict]] = None,
) -> int:
    """Index the evicted turns for ``thread_id``. Returns how many were newly written.

    Idempotent by content hash: the same turns are evicted again on every later request
    in the session, so re-archiving has to be free. After the first write each repeat
    costs one indexed SELECT and writes nothing.

    ``live`` is the fitted conversation, i.e. what the model can still see. It bounds how
    many copies of a repeated turn the archive is allowed to hold right now; see
    ``_write_budget``. Optional, and omitting it keeps the previous behaviour.
    """
    if not thread_id or not evicted or not enabled():
        return 0
    # Incognito and API-only threads are excluded; see can_archive.
    if not can_archive(thread_id):
        logger.debug("conversation_archive.skipped_unpersisted_thread thread_id=%s", thread_id)
        return 0

    # The evictor's own grouper, so an archived unit is exactly the unit the window drops.
    # Imported lazily because the inference layer imports this module.
    from core.inference.context_window import group_turns

    # Keep the turn's transcript span too: _archivable can drop messages, and bounding the branch check
    # by the shorter figure rejected valid exchanges as off-branch.
    groups = [
        (archivable, len(group))
        for group, archivable in ((group, _archivable(group)) for group in group_turns(evicted))
        if archivable
    ]
    if not groups:
        return 0

    model = config.effective_embedding_model()
    scope = store.conversation_archive_scope(thread_id)
    # Conversation order from the persisted thread, not arrival order (`_transcript_positions`).
    positions = _transcript_positions(thread_id, branch or live)
    live_positions = _live_positions(live)
    written = 0
    conn = None
    try:
        count = embeddings.token_counter(model)
        conn = rag_db.get_connection()
        # Only a prediction until the encode reports what it used, so the authoritative check happens under
        # the write lock.
        expected_identity = embeddings.embedding_identity(model)

        # Chunk everything first, then embed in ONE pass: both backends serialise per-group jobs.
        pending = []
        for group, span in groups:
            text = render_turn(group)
            if not text:
                continue
            digest = hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()
            seats = _occurrences(positions, group)
            budget = _write_budget(positions, seats, live_positions, group)
            if _archived_under(conn, scope, digest, expected_identity, occurrences = budget):
                # Commit here: this path holds no transaction of its own and can return before the write lock, so it
                # is the only chance an upgraded archive has to converge.
                # Bounded by the LIVE-AWARE budget, not every seat: a copy per seat left byte-identical documents
                # competing for the same recall slots.
                _restamp(
                    conn,
                    scope,
                    digest,
                    seats[: max(1, budget)] if seats else seats,
                    commit = True,
                )
                _widen_span(conn, scope, digest, span)
                continue
            chunks = chunk_pages(
                [Page(text = text, page_number = None, char_count = len(text))],
                max_tokens = config.CHUNK_TOKENS,
                overlap = config.CHUNK_OVERLAP,
                count = count,
            )
            if chunks:
                pending.append((group, span, digest, chunks, seats, budget))
        if not pending:
            return 0

        # Identity from the encode that produced these vectors; a concurrent embedder swap would label them
        # with a space they were never in.
        vectors, identity = embeddings.encode_with_identity(
            [chunk.text for entry in pending for chunk in entry[3]],
            model_name = model,
            normalize = True,
        )
        offset = 0
        for group, span, digest, chunks, seats, budget in pending:
            group_vectors = vectors[offset : offset + len(chunks)]
            offset += len(chunks)
            roles = " + ".join(
                dict.fromkeys(str(message.get("role") or "message") for message in group)
            )
            # commit=False until the chunks are in: a document committed first leaves an empty row marked
            # completed, which document_by_hash skips forever.
            # Re-check under the write lock: (scope, sha256) is a plain index, so two concurrent passes both insert.
            _write_lock = False
            try:
                conn.execute("BEGIN IMMEDIATE")
                _write_lock = True
            except Exception:
                # Already in a transaction: the insert is still atomic with the re-check.
                logger.debug("conversation_archive.no_write_lock", exc_info = True)
            copies = store.documents_by_hash(conn, scope, digest)
            stale = _stale_document(conn, scope, digest, identity, occurrences = budget)
            if stale is _ARCHIVED:
                _restamp(conn, scope, digest, seats, copies = copies)
                if _write_lock:
                    conn.commit()
                # Widen here too: both turns can arrive in ONE compaction, leaving the longer occurrence unsearchable.
                _widen_span(conn, scope, digest, span)
                continue
            ordinal = None
            archived_at = None
            if stale is not None:
                # Replace rather than deduplicate: skipping would leave the turn invisible to dense search forever.
                # Keep POSITION and TIMESTAMP: a re-embed rewrites how a turn is indexed, not when it was said, and
                # NULL stays NULL.
                previous = store.get_document(conn, stale) or {}
                ordinal = previous.get("archive_ordinal")
                archived_at = previous.get("created_at")
                store.delete_document(conn, stale, commit = False)
            else:
                # The nth copy of a repeated turn takes the nth occurrence's position, so a verbatim repeat lands
                # where it was said.
                ordinal = (
                    seats[len(copies)]
                    if seats and len(copies) < len(seats)
                    else _fallback_ordinal(conn, scope, positions)
                )
            document_id = store.create_document(
                conn,
                scope = scope,
                thread_id = thread_id,
                filename = f"earlier turn ({roles})",
                sha256 = digest,
                status = "completed",
                embedding_model = identity,
                # The turn's real size in the TRANSCRIPT, so the branch check bounds its run exactly; label counts
                # only approximate it.
                archive_messages = span,
                # Allocated inside the write lock in group_turns order; created_at cannot order turns one compaction
                # writes microseconds apart.
                archive_ordinal = ordinal,
                # When the turn was archived, not when this row was written.
                created_at = archived_at,
                commit = False,
            )
            try:
                store.add_chunks(conn, scope, document_id, chunks, group_vectors)
            except Exception:
                conn.rollback()
                raise
            # Surplus only, NOT a restamp: restamping here would renumber a pre-column row.
            _retire_surplus(conn, scope, digest, seats)
            # Restamp against the seats that remain: retiring the first of two twins left the survivor rendering
            # as the later, superseding statement.
            if stale is not None:
                _restamp(conn, scope, digest, seats, skip_null = True)
            conn.commit()
            written += 1
            # The re-embed branch swaps one copy's vectors and keeps the count, so top the copies up to the budget here.
            # Only where the budget counts EVICTED occurrences, else this writes a copy of a turn whose repeat
            # is still in the prompt.
            while (
                stale is not None
                and live_positions is not None
                and len(store.documents_by_hash(conn, scope, digest)) < budget
            ):
                if not _write_copy(
                    conn,
                    scope = scope,
                    thread_id = thread_id,
                    roles = roles,
                    digest = digest,
                    identity = identity,
                    group = group,
                    span = span,
                    chunks = chunks,
                    vectors = group_vectors,
                    seats = seats,
                ):
                    break
                _retire_surplus(conn, scope, digest, seats)
                conn.commit()
                written += 1
            if _INGEST_FAILED:
                globals()["_INGEST_FAILED"] = False
    except Exception:
        # A chat that cannot archive still beats one that raises.
        globals()["_INGEST_FAILED"] = True
        logger.warning("conversation_archive.ingest_failed thread_id=%s", thread_id, exc_info = True)
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    # A delete landing during chunk+embed sweeps the scope BEFORE this commit puts rows back; re-
    # checking after the commit converges either way.
    if written and not can_archive(thread_id):
        logger.info("conversation_archive.thread_deleted_mid_ingest thread_id=%s", thread_id)
        delete_for_thread(thread_id)
        return 0
    return written


# Process-wide on purpose: what fails here is the embedder or the store, not one thread.
_INGEST_FAILED = False


def degraded() -> bool:
    """Whether the last archive attempt failed outright.

    The window reserves recall room before any of this runs and ``archive_turns``
    swallows its failures, so a machine whose embedder cannot start would pay the reserve
    on every compaction for nothing. The caller checks this and stops reserving.
    """
    return _INGEST_FAILED


# Short enough that a store or embedder coming back is noticed within a turn or two, long enough that one
# request's refits share a single probe.
def reachable() -> bool:
    """Whether an archive write attempted RIGHT NOW could reach its store and embedder.

    ``degraded`` is the verdict on the LAST write, the wrong tense for a caller deciding
    whether to reset: this request's write runs afterwards and swallows its own failure, so
    the first request after the store or embedder dies would commit a reset claiming the
    dropped turns are searchable while nothing was indexed.

    A probe, not a promise: it cannot see a failure starting after it returns, leaving the
    same one-turn window for a store that dies mid-request (the turns survive anyway, since
    the client re-sends the branch and the write is idempotent).

    The counter is CALLED, not merely constructed. `embedding_identity` is string
    formatting over resolver metadata and `token_counter` hands back a lazy closure, so
    both reported a healthy archive while the embedder could not initialize; calling it
    forces the load.

    And a real ENCODE, because the tokenizer is not the forward pass. `_st_token_counter`
    reaches only `_get(model).tokenizer`, so a runtime encode failure with no llama binary
    to fall back to -- CUDA OOM against the co-resident chat model, a driver fault, the
    half-precision path -- left this answering yes while `archive_turns` was about to
    raise and swallow it. The reset would already have dropped the history and told the
    model it was searchable, and the epoch is replayed from the boundary, so the loss is
    durable rather than one turn.

    NOT memoised across requests. A time-boxed cache handed back a stale yes for the whole
    window after the store or embedder died, and one request inside it is enough: the epoch
    starts, `archive_turns` swallows the write failure, and the fitted prompt has already
    dropped the turns it says are searchable. The caller memoises per fit instead, which is
    the only span where the answer cannot change under it.

    `"x"` rather than `""`: an empty input is documented to upset the llama embedding
    server. An encode rather than `dim()`, which caches and on the sentence-transformers
    path runs no forward at all -- that would reintroduce exactly the bug this closes.
    """
    if not enabled():
        return False
    conn = None
    try:
        conn = rag_db.get_connection()
        # WRITABLE, not merely open: get_connection succeeds against a read-only or full database and the
        # archive write swallows its own failure.
        # A real statement, rolled back: BEGIN IMMEDIATE alone is not enough, since sqlite defers the check
        # until a write.
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute("CREATE TABLE IF NOT EXISTS _archive_write_probe(x)")
        finally:
            conn.rollback()
        model = config.effective_embedding_model()
        embeddings.embedding_identity(model)
        embeddings.token_counter(model)("")
        embeddings.encode(["x"], model_name = model, normalize = True)
        return True
    except Exception:  # noqa: BLE001 -- an unreachable archive is "no", never an error
        logger.debug("conversation_archive.unreachable", exc_info = True)
        return False
    finally:
        # The probe runs on every checkpoint-eligible overflow, and a connection left to cyclic collection is a
        # descriptor held for an unbounded time: measured, 50 calls leaked 50 open handles on rag.db. Every
        # other connection in this module is closed the same way.
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


# Returned by ``_stale_document`` for a turn that is already archived under vectors the query side still
# accepts.
_ARCHIVED = "archived"


# A tight leaf cap excluded the older branch the user went BACK to; cost is amortised because
# sibling chains share nearly all ancestors.
_BRANCH_SEED_MAX_LEAVES = 512


def _walk_from(by_id: dict, parent_of: dict, leaf) -> list[dict]:
    """The rows from ``leaf`` back to the root, oldest first."""
    chain: list[dict] = []
    seen: set = set()
    current = leaf
    while current is not None and current not in seen:
        seen.add(current)
        record = by_id.get(current)
        if record is None:
            break
        chain.append(record)
        current = parent_of.get(current)
    chain.reverse()
    return chain


def _branch_seed(messages: list[dict], by_id: dict, parent_of: dict, branch) -> Optional[str]:
    """Which stored leaf the REQUEST is on, by matching its text. None when nothing does.

    The newest stored row is not the branch the request is on. Switching to a sibling,
    continuing there and switching back leaves the abandoned branch holding the greatest
    created_at, and the frontend says so outright in `refresh-context-usage.ts`: after a
    retry the newest stored leaf is a branch the user left. Measured on a thread with one
    such switch, the walk read the abandoned branch and BOTH of the request branch's
    evicted turns matched no position, taking MAX + 1 over an archive the other branch had
    already pushed up, which the recall header presents as superseding.

    Matched on text, not id: the wire carries no message ids. Scored rather than compared,
    since the newest branch message is usually not persisted yet and the evicted turns are
    no longer in the fitted conversation.
    """
    if not branch:
        return None
    # A LIST, in order, not a set: sets lose repetition and ordering, so an abandoned sibling could tie
    # and win on recency.
    wanted = [
        text
        for text in (
            _normalise_cased(_probe_text(message))
            for message in _as_wire(branch)
            if str(message.get("role") or "") not in ("system", "developer")
        )
        if text
    ]
    if not wanted:
        return None
    # Allow skipping an entry with no stored counterpart; a strict cursor stalled on the first one and
    # every leaf scored zero.
    where: dict = {}
    for index, text in enumerate(wanted):
        where.setdefault(text, []).append(index)
    parents = {parent for parent in parent_of.values() if parent is not None}
    order = {message.get("id"): index for index, message in enumerate(messages)}
    leaves = sorted(
        (identifier for identifier in by_id if identifier not in parents),
        key = lambda identifier: order.get(identifier, -1),
        reverse = True,
    )
    best = None
    best_score = 0
    # Rendered once per STORED ROW, not per leaf: `_as_wire` expands a row the same way whichever chain it is
    # walked in.
    rendered: dict = {}

    def _texts_of(record: dict) -> list:
        identifier = record.get("id")
        if identifier is None:
            return [_normalise_cased(_probe_text(m)) for m in _as_wire([record])]
        if identifier not in rendered:
            rendered[identifier] = [_normalise_cased(_probe_text(m)) for m in _as_wire([record])]
        return rendered[identifier]

    for leaf in leaves[:_BRANCH_SEED_MAX_LEAVES]:
        # Greedy in-order scan with gaps on both sides: neither the live branch nor the stored chain is a
        # subsequence of the other.
        cursor = 0
        score = 0
        for record in _walk_from(by_id, parent_of, leaf):
            for text in _texts_of(record):
                spots = where.get(text) if text else None
                if not spots:
                    continue
                index = bisect.bisect_left(spots, cursor)
                if index < len(spots):
                    cursor = spots[index] + 1
                    score += 1
        if score > best_score:
            best, best_score = leaf, score
        if best_score >= len(wanted):
            # Every message the request carries is on this chain; nothing can beat it.
            break
    return best


def _active_chain(messages: list[dict], branch = None) -> list[dict]:
    """The rows on ONE branch, oldest first, rather than the whole stored DAG.

    `list_chat_messages` is an unfiltered SELECT ordered by time, but a thread is a tree:
    `parent_id` is a real column and Retry leaves the replaced reply in place as a sibling.
    Read as a flat list, an abandoned sibling lands between two live turns and the grouper
    glues it onto whichever turn precedes it, so the regenerated turn matches no position
    at all and falls back to MAX + 1. Measured on five turns with one Retry: the
    regenerated turn 2 came back numbered 5 out of 4 live turns, colliding with live turn
    3, under the header that says the higher number supersedes. The cumulative archive is
    what makes MAX + 1 land past everything -- the abandoned branch has already taken the
    high numbers -- so this does not depend on the eviction order.

    Walked newest leaf back to root, the same shape the frontend's `orderBySelectedBranch`
    uses to decide what the model is actually shown. `parent_id` is missing on rows written
    before that column, so the previous row stands in for it, which is exactly a flat list
    when nothing branches.
    """
    if not messages:
        return []
    by_id: dict = {}
    parent_of: dict = {}
    previous = None
    for message in messages:
        identifier = message.get("id")
        if identifier is None:
            continue
        by_id[identifier] = message
        parent_of[identifier] = message.get("parentId") or message.get("parent_id") or previous
        previous = identifier
    if not by_id:
        return list(messages)
    # Fall back to the newest row: empty positions send every turn to MAX + 1, which is worse than
    # reading the wrong branch.
    seed = _branch_seed(messages, by_id, parent_of, branch) or messages[-1].get("id")
    return _walk_from(by_id, parent_of, seed) or list(messages)


# JSON.stringify puts no space after the colon and every comparison downstream is exact.
_EMPTY_TOOL_RESULT = '{"result":""}'


# SERVER_SIDE_BUILTIN_TOOL_NAMES in chat-adapter.ts. The NAME alone never decides: a user function
# may legitimately be called web_search.
_SERVER_BUILTIN_NAMES = frozenset({"web_search", "web_fetch", "code_execution", "image_generation"})
# `SANDBOX_FILE_TOOLS`, and `tool_loop_controller._SANDBOX_TOOLS`. Only these two wrap.
_SANDBOX_TOOL_NAMES = frozenset({"python", "terminal"})


def _server_builtin(part: dict) -> tuple[bool, bool]:
    """Whether a persisted call is a provider-side builtin, and whether it has a native part.

    The frontend drops a builtin from the replayed history entirely when there is no
    native part, and replays one WITH a native part as a call carrying no `tool` result.
    Reconstructing either as an ordinary local call inserted a phantom exchange, so the
    request-shaped turn no longer matched the persisted run and the turn took a fallback
    ordinal. Both signals ride on the persisted `args`, so this is decidable here.
    """
    args = part.get("args")
    args = args if isinstance(args, dict) else {}
    if str(part.get("toolName") or "").lower() not in _SERVER_BUILTIN_NAMES:
        return False, False
    google = args.get("google")
    native = isinstance(google, dict) and isinstance(google.get("native_part"), dict)
    return bool(args.get("_server_tool") is True or native), native


def _unwrapped(result, tool_name: str):
    """A sandbox or MCP-image wrapper reduced to the text the model actually saw.

    `python` and `terminal` results are wrapped in `{text, images, sessionId, files}` on
    EVERY call, and the replay adapter sends `result.text` alone rather than feeding the
    model a session id and file metadata. Serialising the whole wrapper reconstructed a
    tool message that can never equal the archived one.

    Both gates are the frontend's: the name, because a third-party tool answering with
    `{text, sessionId, images}` is someone else's and unwrapping it would drop every other
    field, and the shape.
    """
    if not isinstance(result, dict) or not isinstance(result.get("text"), str):
        return None
    images = result.get("images")
    if not isinstance(images, list):
        return None
    if tool_name in _SANDBOX_TOOL_NAMES and isinstance(result.get("sessionId"), str):
        files = result.get("files")
        if files is None or (
            isinstance(files, list)
            and all(isinstance(f, dict) and isinstance(f.get("name"), str) for f in files)
        ):
            return result["text"]
    if (
        result.get("sessionId") is None
        and images
        and all(
            isinstance(image, dict)
            and isinstance(image.get("data"), str)
            and isinstance(image.get("mimeType"), str)
            for image in images
        )
    ):
        return result["text"]
    return None


def _tool_result_content(result, tool_name: str = "") -> str:
    """A persisted tool result in the string the replay serializer would have sent.

    An empty string becomes the sentinel above, because the backend's ChatMessage
    validator rejects a `tool` message with empty content; everything else is JSON with
    JavaScript's separators. Rendering it any other way makes the reconstructed message
    differ from the one that was actually archived, which is the whole point of this
    module comparing the two.
    """
    if isinstance(result, str):
        return result if result else _EMPTY_TOOL_RESULT
    unwrapped = _unwrapped(result, tool_name)
    if unwrapped is not None:
        return unwrapped if unwrapped else _EMPTY_TOOL_RESULT
    # ensure_ascii=False because JSON.stringify does not escape non-ASCII; otherwise an accented tool
    # result loses its transcript seat.
    return json.dumps(result, ensure_ascii = False, separators = (",", ":"))


def _replayable(part: dict) -> bool:
    """Whether the serializer replays this tool call at all.

    `chat-adapter.ts` drops the whole call when it has no result to send and cannot be
    replayed without one: `if (!toolResult && !canReplayToolCallWithoutRoleTool(part))
    continue`. That is every cancelled or still-running LOCAL card. Keeping it here and
    merely omitting its `tool` message invented an assistant `tool_calls` message the
    request never carried, which shifts the groups either side of it and can send an
    otherwise live turn to a wrong ordinal or out of the branch check entirely.

    A provider-side builtin is replayable without a result, but one with no native part
    is dropped by `serializeAssistantToolCallPart` instead, so it is not a call here.
    """
    builtin, native = _server_builtin(part)
    if builtin:
        return native
    return part.get("result") is not None


def _local_round_id(part: dict):
    """`codexLocalToolRoundId`: the round a LOCAL tool call belongs to, or None."""
    provenance = part.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("source") != "local":
        return None
    round_id = provenance.get("round_id")
    if isinstance(round_id, bool) or not isinstance(round_id, int):
        return None
    return round_id


def _flushes_local_pair(part: dict) -> bool:
    """`shouldFlushCompletedLocalToolPair`: a completed local call is its own round.

    A local tool card that already has its result is flushed both BEFORE and AFTER it, so
    two adjacent completed local calls replay as two sequential assistant/tool groups and
    not as one parallel group. Batching them made `group_turns`, the occurrence ordinals
    and the live-branch check all describe a transcript shape the request never sent.
    """
    provenance = part.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("source") != "local":
        return False
    if _server_builtin(part)[0]:
        return False
    return part.get("result") is not None


def _as_wire(messages: list[dict]) -> list[dict]:
    """Persisted chat rows in the shape the inference layer sends them.

    The store keeps a tool call as a ``tool-call`` CONTENT PART carrying its own result,
    while the wire form is three messages: the assistant's `tool_calls`, a `tool` result,
    then the assistant's reply. Nothing put them back, and everything here reads the
    persisted rows as if they were wire messages, so an agent turn was invisible twice
    over. `group_turns` splits on `tool_calls`, which a persisted row never has, so the
    whole exchange folded into the preceding user group and the evicted tool group found
    no position of its own -- measured, seats came back empty and the exchange took
    MAX + 1, the number its own opening question already had. And `_live_transcript`
    probes these rows in the same shape, so the branch check compared a call/result/reply
    render against a row reading call/reply/result: measured, the archived agent turn
    failed the live-branch filter and NO query could return it, on every tool-using turn.

    The result is stripped from the call part and carried by the `tool` message instead,
    so `_probe_text` renders each piece exactly once and in the order `render_turn` wrote
    it. Only the id goes into `tool_calls`: the arguments stay on the content part, where
    `_probe_text` already offers both JSON spellings, and `_is_injected` still sees the id
    it filters our own injections by.
    """
    wire: list[dict] = []
    for message in messages:
        content = message.get("content")
        parts = content if isinstance(content, list) else None
        # Reasoning is not content: forwarding the stored list puts a model's thinking inline and the turn
        # matches no transcript seat. Nothing reads reasoning_content on the live side.
        if parts is not None and any(
            isinstance(part, dict) and part.get("type") == "reasoning" for part in parts
        ):
            parts = [
                part
                for part in parts
                if not (isinstance(part, dict) and part.get("type") == "reasoning")
            ]
            content = parts
        # A provider-side builtin with no native part is not replayed, so counting it invents an exchange
        # the request never carried.
        calls = [
            part
            for part in (parts or [])
            if isinstance(part, dict) and part.get("type") == "tool-call"
        ]
        if not calls:
            wire.append(
                {
                    "role": message.get("role"),
                    "content": content,
                    "tool_calls": message.get("tool_calls"),
                }
            )
            # Dropped whole, call and result, exactly as the serializer does.
            continue
        # Replay parts in order the way chat-adapter.ts does, flushing pending calls whenever text
        # arrives; collecting every call into one message rebuilt a different order.
        pending_calls: list[dict] = []
        pending_text: list = []
        # A one-slot box because _flush resets it and Python closures cannot rebind.
        pending_round: list = [None]

        def _flush(role = message.get("role")) -> None:
            pending_round[0] = None
            if not pending_calls and not pending_text:
                return
            body: list = [
                {key: value for key, value in call.items() if key != "result"}
                for call in pending_calls
            ] + pending_text
            entry: dict = {"role": role, "content": body}
            if pending_calls:
                entry["tool_calls"] = [
                    {"id": call.get("toolCallId"), "function": {}} for call in pending_calls
                ]
            wire.append(entry)
            for call in pending_calls:
                if _server_builtin(call)[0]:
                    # A builtin WITH a native part replays as a call and no tool message: its result travels in the
                    # provider's native part.
                    continue
                if "result" not in call or call.get("result") is None:
                    # The serializer skips exactly undefined and null, so treating empty string / {} / [] as absent
                    # dropped a message the wire carries.
                    continue
                wire.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("toolCallId"),
                        "content": _tool_result_content(
                            call.get("result"), str(call.get("toolName") or "")
                        ),
                    }
                )
            pending_calls.clear()
            pending_text.clear()

        for part in parts:
            if isinstance(part, dict) and part.get("type") == "tool-call":
                if not _replayable(part):
                    continue
                round_id = _local_round_id(part)
                if (
                    pending_calls
                    and pending_round[0] is not None
                    and round_id is not None
                    and pending_round[0] != round_id
                ):
                    # `startsNewCodexToolRound`: a new local round is a new group.
                    _flush()
                if round_id is not None:
                    pending_round[0] = round_id
                flush_pair = round_id is None and _flushes_local_pair(part)
                if flush_pair and pending_calls:
                    _flush()
                pending_calls.append(part)
                if flush_pair:
                    _flush()
                # A row archived before the column existed stays unnumbered, or it moves to the end of its own
                # conversation and the header calls the oldest statement the latest one.
                continue
            if pending_calls:
                _flush()
            pending_text.append(part)
        _flush()
    return wire


def _fallback_ordinal(conn, scope: str, positions: Optional[list[list[str]]]) -> int:
    """Where a turn that matched no seat goes: past the archive AND past the transcript.

    The two numbering spaces are different. Seats are TRANSCRIPT positions, while
    `next_archive_ordinal` counts what has been ARCHIVED, and the newest user group is
    protected from eviction, so during a long tool loop it is in the transcript and not
    in the archive. An in-flight tool group evicted before its assistant row is persisted
    matches no seat and took the archive's next number, which the user turn then claimed
    from the transcript: measured, both documents landed on ordinal 0, and since
    `created_at` breaks the tie the tool answer rendered ahead of the prompt that caused
    it, under the header saying a higher number was said later.

    The transcript length is the right floor because an unmatched group is unmatched for
    being newer than the saved rows. A gap in the numbering costs nothing: ordinals only
    have to order.
    """
    return max(store.next_archive_ordinal(conn, scope), len(positions or []))


def _transcript_positions(thread_id: str, branch = None) -> Optional[list[str]]:
    """The thread's saved messages as normalised probe text, in the order they were said.

    The ordinal has to come from the CONVERSATION, not from the moment a turn happened to
    be archived. Eviction is not strictly oldest-first: `truncate_oldest_messages` always
    protects the newest user group, so an agent turn's tool groups are evicted while the
    user message that opened them is held, and a pinned instruction is evicted only once
    it stops being pinned. Numbering by archive time then records the genuinely oldest
    turn as the newest, and `format_conversation_recall` states outright that the higher
    number "was said later and supersedes the earlier one". Measured before this: a
    standing instruction archived second came back as turn 3 of 3, presented to the model
    as superseding the two turns that actually followed it.

    Read from the persisted transcript rather than the request, because four of the five
    archive call sites pass the tool loop's already-fitted messages, whose indices shift
    as earlier groups drop out. `studio_db` holds the whole thread and is never truncated.
    """
    try:
        from core.inference.context_window import group_turns
        from storage import studio_db
        messages = studio_db.list_chat_messages(thread_id)
    except Exception:
        return None
    if not messages:
        return None
    # Grouped with the evictor's grouper, so a position is a TURN index, which is what the ordinal is compared against.
    wire = _as_wire(_active_chain(messages, branch))
    return [
        [_normalise_cased(_probe_text(message)) for message in group]
        for group in group_turns(wire)
        if group
    ]


def _occurrences(positions: Optional[list[list[str]]], group: list[dict]) -> list[int]:
    """Where this turn sits in the transcript, every time it was said.

    The WHOLE turn, not just its opening line. Matching on the first message alone made
    two different turns that merely start the same -- a repeated "continue", the same
    question re-asked, a regenerated reply -- claim each other's seats: measured, both
    were stamped with ordinal 0, and because each then believed it had two occurrences to
    fill, the next compaction wrote both of them AGAIN. Four documents for two turns,
    twice the recall slots spent on the same content, and the older answer quoted under
    the higher turn number, which the header presents as the one that supersedes.

    Compared as a prefix, because the transcript legitimately lags the evicted group: a
    turn is archived mid-request, before its own reply has been persisted. A turn that
    matches nothing yields no seats and falls back to the previous allocator, so this can
    never do worse than not looking.

    A list rather than one index because a conversation may legitimately contain the same
    turn twice, and the later occurrence is usually the one that matters -- "set X to 1",
    "set X to 2", "set X to 1" ends with X at 1.
    """
    if not positions or not group:
        return []
    texts = [_normalise_cased(_probe_text(message)) for message in group]
    if not texts or not texts[0]:
        return []
    calls = [bool(message.get("tool_calls")) for message in group]

    def _same(stored: str, live: str, is_call: bool) -> bool:
        # A tool call is compared as needle in haystack: the store keeps arguments as an OBJECT and the
        # request carries the model's raw string.
        # Allow exactly ONE split point: _probe_text inserts the second spelling between the arguments and
        # what followed them.
        if is_call:
            if not live:
                return False
            if live in stored:
                return True
            words = live.split(" ")
            for cut in range(1, len(words)):
                head = " ".join(words[:cut])
                tail = " ".join(words[cut:])
                at = stored.find(head)
                if at >= 0 and stored.find(tail, at + len(head)) >= 0:
                    return True
            return False
        return stored == live

    return [
        index
        for index, position in enumerate(positions)
        if position
        # zip stops at the shorter side, so an orphan user row prefix-matched and gave the answered turn two
        # seats; >= because a position may legitimately run longer.
        and (len(position) >= len(texts) or index == len(positions) - 1)
        and all(
            _same(stored, live, is_call) for stored, live, is_call in zip(position, texts, calls)
        )
    ]


def _write_copy(
    conn,
    *,
    scope: str,
    thread_id: str,
    roles: str,
    digest: str,
    identity: str,
    group: list[dict],
    span: int,
    chunks,
    vectors,
    seats: list[int],
) -> bool:
    """One more copy of an already-embedded turn, at the next unfilled seat.

    Only for topping up after a re-embed, which replaces a copy rather than adding one.
    Returns False when there is no seat left to fill, so the caller stops rather than
    allocating a fresh ordinal and putting a repeat at the end of its own conversation.

    ``span`` is the turn's size in the TRANSCRIPT and not ``len(group)``: `_archivable`
    strips a retrieval call and its result, so a group of three can span four live
    messages. Written short, the branch check bounds its run by the smaller figure and
    filters this copy out of every recall -- the same bug the primary write path was
    fixed for, reappearing on the top-up path.
    """
    copies = store.documents_by_hash(conn, scope, digest)
    if not seats or len(copies) >= len(seats):
        return False
    document_id = store.create_document(
        conn,
        scope = scope,
        thread_id = thread_id,
        filename = f"earlier turn ({roles})",
        sha256 = digest,
        status = "completed",
        embedding_model = identity,
        archive_messages = span,
        archive_ordinal = seats[len(copies)],
        commit = False,
    )
    try:
        store.add_chunks(conn, scope, document_id, chunks, vectors)
    except Exception:
        conn.rollback()
        raise
    return True


def _live_positions(live: Optional[list[dict]]) -> Optional[list[list[str]]]:
    """The fitted conversation grouped exactly as ``_transcript_positions`` groups the
    stored one, so a live turn can be matched by the same rules a stored seat is."""
    if live is None:
        return None
    try:
        from core.inference.context_window import group_turns
    except Exception:
        return None
    positions = [
        [_normalise_cased(_probe_text(message)) for message in group]
        for group in group_turns(_as_wire(live))
        if group
    ]
    return positions or None


def _write_budget(
    positions: Optional[list[list[str]]],
    seats: list[int],
    live_positions: Optional[list[list[str]]],
    group: Optional[list[dict]] = None,
) -> int:
    """How many copies of a repeated turn the archive may hold RIGHT NOW.

    ``seats`` is every place the turn was said, which is what allocates ordinals, but it
    is the wrong number to spend on writes. A thread can hold the same turn twice with
    only the older one evicted; the archive then sees one stored copy against two seats,
    decides it is short, and writes a second document for the occurrence still sitting in
    the prompt. Both are then recallable, so identical text takes two of the four recall
    slots and one of them repeats what the model can already read.

    Counted against the LIVE conversation instead: a seat whose every message is still in
    the prompt has not been evicted and buys no write. When that turn is evicted later the
    budget rises on its own and the copy is written then, at its own ordinal.

    COUNTED, not tested for membership. A set of live texts cannot tell "one of three
    identical turns is still in the prompt" from "all three are", so every seat looked
    live and a turn said three times with two of them evicted was archived once. Counting
    the live occurrences with `_occurrences`, the same matcher that finds the seats,
    subtracts exactly as many as the prompt really holds.

    Floors at 1 so a turn whose seats cannot be told apart from live text is still stored;
    and with no ``live`` to compare against, this is exactly the old ``len(seats)``.
    """
    if not seats:
        return 1
    if not live_positions or not positions:
        return len(seats)
    live = len(_occurrences(live_positions, group)) if group else 0
    return max(len(seats) - live, 1)


def _retire_surplus(
    conn,
    scope: str,
    digest: str,
    seats: list[int],
    *,
    rows = None,
) -> bool:
    """Delete copies of this turn the conversation no longer holds. True if any went.

    More copies than occurrences means a rewind removed one. The survivors are
    byte-identical, so the branch filter validates every copy against the single remaining
    occurrence and `recall` dedups on chunk id, which differs: measured, a recall slot went
    on quoting one turn twice, and the surplus kept an ordinal a genuinely later turn had
    since taken.

    Does nothing without seats, so a turn that failed to match its occurrences at all --
    an unreconstructable transcript, a thread with no persisted rows -- never loses a copy.
    """
    if not seats:
        return False
    try:
        copies = rows if rows is not None else store.documents_by_hash(conn, scope, digest)
        surplus = copies[len(seats) :]
        for copy in surplus:
            store.delete_document(conn, copy["id"], commit = False)
        return bool(surplus)
    except Exception:  # noqa: BLE001 -- tidying an archive is not worth a chat
        logger.debug("conversation_archive.retire_surplus_failed", exc_info = True)
        return False


def _restamp(
    conn,
    scope: str,
    digest: str,
    seats: list[int],
    *,
    copies = None,
    commit: bool = False,
    skip_null: bool = False,
) -> None:
    """Move existing copies of this turn onto the positions the transcript gives them.

    An archive written by an earlier build numbered turns as they arrived, and one written
    before the column existed has no number at all. Both keep an order that was never
    true, and `format_conversation_recall` states that the higher number was said later
    and supersedes the earlier one, so the block asserts it. Re-stamping at the next
    compaction lets those archives converge with no migration pass and no reindex: this is
    an UPDATE on rows that already exist, so nothing is duplicated.

    Called from BOTH archived paths. The cheap pre-check that runs before chunking fires
    on exactly the condition the write-lock branch does, so a restamp only in the latter
    was unreachable outside a race: measured, a forced-legacy archive still read NULL,NULL
    after a full re-compaction, and an archive-time order of 1,0 stayed 1,0 with the
    recall rendering the second turn first.
    """
    if not seats:
        return
    try:
        rows = copies if copies is not None else store.documents_by_hash(conn, scope, digest)
        moved = False
        for seat, copy in zip(seats, rows):
            if skip_null and copy.get("archive_ordinal") is None:
                # A row archived before the column existed stays unnumbered rather than moving to the end of its own
                # conversation.
                continue
            if copy.get("archive_ordinal") != seat:
                store.set_archive_ordinal(conn, copy["id"], seat)
                moved = True
        moved = _retire_surplus(conn, scope, digest, seats, rows = rows) or moved
        if moved and commit:
            conn.commit()
    except Exception:  # noqa: BLE001 -- ordering an old archive is not worth a chat
        logger.debug("conversation_archive.restamp_failed", exc_info = True)


def _stale_document(
    conn,
    scope: str,
    digest: str,
    identity: str,
    *,
    occurrences: int = 1,
):
    """The document id to replace, ``_ARCHIVED`` to skip, or None to write a new one.

    Hash alone is not enough, twice over.

    Dense search only reads documents whose recorded embedder matches the query's, so a
    turn archived under the previous model stays hashed-and-skipped while being invisible
    to every paraphrased search. Ingestion re-indexes in that case; so does this.

    And a hash is not an identity when a user repeats themselves. ``occurrences`` is how
    many times this exact turn appears in the transcript, so a copy is only "already
    archived" once every occurrence has one. Without it the third turn of "set X to 1",
    "set X to 2", "set X to 1" was dropped on the floor: never indexed, so no query could
    reach it, and the recall then handed the model the superseded value under a header
    saying the higher turn number supersedes the lower. Measured: 3 turns in, 2 documents
    stored, and the block asserted X was 2.
    """
    copies = store.documents_by_hash(conn, scope, digest)
    if not copies:
        return None
    for copy in copies:
        if not config.embedding_identity_matches(copy.get("embedding_model"), identity):
            # Re-index the stale copy before considering writing a new one.
            return copy["id"]
    return _ARCHIVED if len(copies) >= max(1, occurrences) else None


def _archived_under(
    conn,
    scope: str,
    digest: str,
    identity: str,
    *,
    occurrences: int = 1,
) -> bool:
    """Cheap pre-check before the chunking and embedding pass."""
    return _stale_document(conn, scope, digest, identity, occurrences = occurrences) is _ARCHIVED


def _widen_span(conn, scope: str, digest: str, span: int) -> None:
    """Grow a stored turn's window when the same text reappears over a LONGER span.

    The digest is the rendered text, so two turns that read the same are one document,
    but their transcript spans can differ: `_archivable` strips a retrieval call and its
    result, so a three-message exchange and a four-message batch containing the same
    exchange render identically. Archived in that order, the second is skipped and keeps
    the first turn's span of three, and `_document_matches_one_run` then bounds its
    four-message live run by three and rejects it as off-branch, so no query returns it.

    Only ever UPWARDS. The window is a maximum, so a larger one still matches the shorter
    turn, while narrowing it would break whichever turn was archived first.
    """
    try:
        conn.execute(
            "UPDATE documents SET archive_messages = ? "
            "WHERE scope = ? AND sha256 = ? "
            "AND (archive_messages IS NULL OR archive_messages < ?)",
            (int(span), scope, digest, int(span)),
        )
        conn.commit()
    except Exception:
        logger.debug("conversation_archive.widen_span_failed", exc_info = True)


def has_archive(thread_id: str) -> bool:
    """Whether anything has ever been archived for this thread."""
    if not thread_id or not enabled():
        return False
    conn = None
    try:
        conn = rag_db.get_connection()
        row = conn.execute(
            "SELECT 1 FROM documents WHERE scope=? LIMIT 1",
            (store.conversation_archive_scope(thread_id),),
        ).fetchone()
        return row is not None
    except Exception:
        return False
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def _normalise(text: str) -> str:
    """Probe text for comparison: trimmed at both ends, otherwise exactly as written.

    Case IS the edit. Lowercasing left a turn corrected only in capitalisation matching
    its archived copy, so `_document_matches_one_run` kept the pre-edit document live and
    a recall could answer with it: measured, `Foo` after it was corrected to `foo`, still
    validated. The two sides here are a stored render and the live text of the same
    messages, so there is no spelling difference to be tolerant of, and nothing else
    retires the stale copy -- the edit changes the digest, so it is written as a new
    document and the branch filter is the only thing that could have dropped the old one.

    `strip()` and not `rstrip()`, even though trimming each probe also discards leading
    indentation. Keeping leading whitespace looks tighter and is worse: `render_turn`
    strips the whole message, so a live turn beginning with a space or a newline then
    starts its run at a non-zero offset and `_document_matches_one_run` retires it.
    Measured, `   hello there` and a pasted block opening on a newline both went from
    live to retired, which is a silent loss of recall on exactly the pasted code that
    tends to matter.

    Indentation-only edits are therefore still tolerated, in both directions: probes are
    matched per line and as substrings, so leading whitespace never decides. Closing that
    needs the probe splitter to carry offsets, which is a bigger change than this bug
    warrants. Lexical search does its own tokenising and never reads this.
    """
    return (text or "").strip()


def _normalise_cased(text: str) -> str:
    """Whitespace-collapsed but CASE-PRESERVING, for transcript seat matching.

    `Set key Foo` and `Set key FOO` are two different turns: they hash differently, so the
    archive keeps a document for each. Folding case here made `_occurrences` hand BOTH
    seats to BOTH of them, and a turn that believes it has two occurrences to fill gets
    written twice at the next compaction. Measured: four documents for two turns, both
    stamped at both ordinals, and the recall unable to say which spelling was said later
    -- exactly the duplication the whole-turn comparison was added to stop. The two sides
    compared here are the same strings from the same thread, so there is no casing to be
    tolerant of; only the free-text branch filter, which compares a QUERY against saved
    text, still folds case.
    """
    return " ".join((text or "").split())


def _live_transcript(thread_id: str) -> Optional[list[str]]:
    """The thread's saved messages, one normalised string each, or None if it has none.

    Keeps recall on the branch the user is on. Editing a message rewinds the thread, but
    the archive is append-only and still holds the abandoned continuation's turns, so
    without this a recall can pull back a turn that never happened on this branch.
    Verified: after rewinding past a turn, querying its text still returned it.

    None means the thread has no saved transcript (an API client passing a thread_id
    without persisting), and the caller then does not filter: absence of evidence.
    """
    try:
        from storage import studio_db
        messages = studio_db.list_chat_messages(thread_id)
    except Exception:
        return None
    if not messages:
        return None
    # Through the same reconstruction: a persisted tool call read as ONE message renders in a different order.
    texts = [_normalise(_probe_text(message)) for message in _as_wire(messages)]
    return [text for text in texts if text] or None


def branch_message_texts(
    messages: Optional[list[dict]], roles: Optional[tuple[str, ...]] = None
) -> Optional[list[str]]:
    """The ACTIVE branch, one normalised string PER MESSAGE, from the request's own messages.

    Per message rather than one blob, so the check stays inside the turn it is checking.
    Flattened, a probe can be satisfied by any later message repeating the words: an
    archived "Should I deploy? / No" whose answer was edited to "Yes" still matched
    because an unrelated later turn said "No". Reproduced before the split.

    Preferred over ``_live_transcript`` wherever available: the stored rows are the whole
    message DAG, and retry/regenerate keep the replaced response as a sibling, so a
    thread-wide blob can validate a turn that is not on this branch. The client sends
    exactly one branch per request, and it is the same projection ``render_turn``
    archived from, so the probe compares like with like.

    ``roles`` narrows it to messages of those roles. The rolling window compares stored
    ASSISTANT rows, and against every role a short abandoned reply ("Done") matches a live
    user message that merely contains it ("not done yet").
    """
    if roles:
        messages = [
            message for message in (messages or []) if str(message.get("role") or "") in roles
        ]
    if not messages:
        return None
    texts = [_normalise(_probe_text(message)) for message in messages]
    return [text for text in texts if text] or None


def message_text(content) -> str:
    """One stored message's content, normalised the way the branch check normalises it.

    Exposed so the rolling window compares stored messages on exactly these terms rather
    than inventing a second notion of "same text".
    """
    return _normalise(_probe_text({"content": content}))


def content_on_branch(content, transcript: Optional[list[str]]) -> bool:
    """Whether one stored message's text appears on the branch ``transcript`` describes.

    Shared with the rolling window, which has recall's problem: the stored rows are the
    whole DAG, so "the newest assistant turn" can belong to a sibling branch. Empty text
    counts as on-branch, since nothing to compare is not evidence of another branch.
    """
    if not transcript:
        return True
    text = _normalise(_probe_text({"content": content}))
    return not text or any(text in message for message in transcript)


_ROLE_PREFIX = re.compile(
    r"^(?:user|assistant|system|developer|tool result|message):\s*", re.IGNORECASE
)
# The label is ours, not the stored message's, so it comes off before the probe like any role prefix.
_TOOL_CALL_PREFIX = re.compile(r"^assistant called (?:[^:\n]+:\s*)?", re.IGNORECASE)
# Keep the tool NAME in a probe of its own: stripping the label whole let a retry that changed the tool still match.
_TOOL_CALL_NAME = re.compile(r"^assistant called ([^:\n]+):\s*", re.IGNORECASE)
# Only a label render_turn wrote may sit between two probes; any other text in that gap is content
# the archive never saw.
_GAP_IS_LABEL = re.compile(
    r"^\s*(?:(?:user|assistant|system|developer|tool result|message):"
    r"|assistant called (?:[^:\n]+:)?)?\s*$",
    re.IGNORECASE,
)


def _probes_for(text: str) -> list[str]:
    """The lines of an archived chunk, normalised into things to look for on the branch.

    Substring containment on a normalised prefix rather than a digest match: the archived
    text came from the inference projection and the saved copy from the message store, so
    exact equality is too brittle. The role labels ``render_turn`` writes exist only in
    the archived copy, so they are stripped first, or every probe misses.
    """
    probes = []
    for line in (text or "").splitlines():
        without_role = _ROLE_PREFIX.sub("", line)
        named = _TOOL_CALL_NAME.match(without_role.strip())
        if named:
            # The name probes ahead of the arguments, else a different tool with the same arguments matches;
            # "tool" is render_turn's nameless fallback and carries no name live either.
            # See `_probes_for`: the tool's NAME is a probe of its own, or the arguments alone decide and a
            # retry that swapped the tool still matches. Flagged as a call so it keeps the haystack comparison
            # the rest of the line gets. "tool" is `render_turn`'s fallback for a nameless call; see
            # `_probes_for`.
            name = _normalise(named.group(1))
            if name and name != "tool":
                probes.append(name)
        stripped = _normalise(_TOOL_CALL_PREFIX.sub("", without_role))
        # Keyed off the truncation marker, not a "tool result:" label: a long result is ONE string whose
        # label is on its first line and marker on its last.
        if stripped.endswith(_TRUNCATION_MARKER.strip()):
            stripped = stripped[: -len(_TRUNCATION_MARKER.strip())].strip()
        probes.append(stripped)
    return [probe for probe in probes if probe]


def _probe_entries(text: str) -> list[tuple[str, bool]]:
    """``_probes_for`` with a flag per line: was it cut short, or is it a tool call.

    Both mean the live message may legitimately hold more than the probe: a cut line is a
    prefix by construction, and a tool call is matched against a haystack that renders its
    arguments twice, spaced and compact, so only one of the two can ever be covered.
    """
    entries = []
    for line in (text or "").splitlines():
        without_role = _ROLE_PREFIX.sub("", line)
        is_call = bool(_TOOL_CALL_PREFIX.match(without_role.strip()))
        named = _TOOL_CALL_NAME.match(without_role.strip())
        if named:
            name = _normalise(named.group(1))
            if name and name != "tool":
                entries.append((name, True))
        stripped = _normalise(_TOOL_CALL_PREFIX.sub("", without_role))
        truncated = stripped.endswith(_TRUNCATION_MARKER.strip())
        if truncated:
            stripped = stripped[: -len(_TRUNCATION_MARKER.strip())].strip()
        if stripped:
            entries.append((stripped, truncated or is_call))
        elif truncated and entries:
            # Keep the empty marker line: dropping it read the last cut-short probe as complete and retired
            # unedited over-cap turns.
            entries[-1] = (entries[-1][0], True)
    return entries


def _on_live_branch(text: str, transcript: Optional[list[str]]) -> bool:
    """Whether one archived chunk still exists in the saved thread."""
    probes = _probes_for(text)
    if not probes or not transcript:
        return False
    # EVERY line, IN ORDER, within one bounded run of adjacent messages: a global scan lets any later
    # message supply a missing line.
    window = len(probes)
    return any(
        _probes_match_from(probes, transcript, start, window) for start in range(len(transcript))
    )


def _scan_probes(
    entries: list[tuple[str, bool]], messages: list[str], start: int, last: int
) -> Optional[tuple[int, int, int, bool, int]]:
    """Where the probes finish: message index, end offset, opening offset, tail-is-partial,
    and the message index the run OPENED on.

    An index rather than a bool so one document's chunks scan as a single pass, each
    continuing where the last stopped, which stops two chunks of a turn matching two
    places. The cursor within that message is NOT carried over: chunks overlap by
    ``CHUNK_OVERLAP``, so the next one legitimately repeats the previous tail.

    The opening index is reported because that repeated tail can begin in an EARLIER
    message than the one the previous chunk finished in -- a short line such as an
    assistant tool call just before a long tool result is carried into the next chunk
    whole. Resuming at the finishing message could then never match it, and an unedited
    document was retired as off-branch, which makes that turn unsearchable.
    """
    index = start
    cursor = 0
    opened_at = None
    opened_index = None
    partial = False
    fresh = False
    for probe, partial_ok in entries:
        while index < last:
            found = messages[index].find(probe, cursor)
            if found >= 0:
                # A message the run stepped INTO must be accounted for from its first character; a tool call is
                # exempt because the stored spelling cannot line up character for character.
                if fresh and found != 0 and not partial_ok:
                    return None
                # Also reject text inserted BETWEEN two probes of the same message, except a label render_turn wrote
                # and the probe stripped.
                # And text inserted BETWEEN two probes of the same message. The two checks around this one cover
                # an edit that prepends to a message the run stepped into and one that appends to a message it
                # is leaving; a correction dropped between two archived lines matched both of them with the new
                # line sitting unexamined in the gap, so the pre-edit turn stayed recallable. Measured on "A\nB"
                # becoming "A\ncorrection\nB". A gap is only allowed where it is a label `render_turn` wrote and
                # the probe therefore had stripped: a pasted transcript legitimately carries its own "user:"
                # lines. Anything else is content the archive never saw.
                if (
                    not fresh
                    and cursor
                    and not (partial or partial_ok)
                    and not _GAP_IS_LABEL.match(messages[index][cursor:found])
                ):
                    return None
                if opened_at is None:
                    # Same exemption at the front of the run as inside it.
                    opened_at = 0 if partial_ok else found
                    opened_index = index
                cursor = found + len(probe)
                # The exemption belongs to the CALL, not the rest of the message: once a text probe matches the
                # cursor is exact again, and leaving it sticky kept pre-edit turns matched.
                partial = partial_ok
                fresh = False
                break
            # And leaving one it had entered: whatever is left over is text an edit added.
            if cursor and not partial and cursor < len(messages[index]):
                return None
            index += 1
            cursor = 0
            partial = False
            fresh = True
        else:
            return None
    return (
        index,
        cursor,
        (opened_at or 0),
        partial,
        (opened_index if opened_index is not None else start),
    )


def _probes_match_from(probes: list[str], messages: list[str], start: int, window: int) -> bool:
    """Whether ``probes`` appear in order within ``messages[start:start + window]``."""
    return (
        _scan_probes(
            [(probe, True) for probe in probes],
            messages,
            start,
            min(len(messages), start + window),
        )
        is not None
    )


def _rendered_message_count(rows) -> int:
    """How many messages the archived turn was rendered from.

    ``render_turn`` labels every message it writes, so labelled lines count messages. That
    bounds the run this document may occupy, far tighter than a line count: a two-message
    turn can be a hundred lines, and a hundred-message window finds its tail anywhere.

    Overlapping chunks can double-count a label, which only widens the window: the safe
    direction, since too narrow retires turns that are still live.
    """
    total = 0
    for row in rows:
        for line in (row["text"] or "").splitlines():
            stripped = line.strip()
            if _ROLE_PREFIX.match(stripped) or _TOOL_CALL_PREFIX.match(stripped):
                total += 1
    return total


def _document_on_live_branch(conn, document_id: str, transcript: list[str], cache: dict) -> bool:
    """Whether EVERY chunk of an archived turn is still on the branch.

    Per chunk is not enough: a turn over CHUNK_TOKENS spans several chunks, so editing the
    second half of a long answer leaves the untouched earlier chunks eligible on their
    own. The archived unit is the turn, so an edit to any part retires the whole copy.

    Cached per call: candidates from one turn share a document, and this is the filter's
    only query.
    """
    if document_id in cache:
        return cache[document_id]
    try:
        rows = conn.execute(
            "SELECT text FROM chunks WHERE document_id = ? ORDER BY chunk_index ASC",
            (document_id,),
        ).fetchall()
        # NULL for archives predating the column, which fall back to counting labels.
        row = conn.execute(
            "SELECT archive_messages FROM documents WHERE id = ?", (document_id,)
        ).fetchone()
        message_count = row["archive_messages"] if row else None
    except Exception:
        # Never fail a recall on the strictness pass.
        cache[document_id] = True
        return True
    cache[document_id] = _document_matches_one_run(rows, transcript, message_count)
    return cache[document_id]


def _document_matches_one_run(
    rows,
    transcript: Optional[list[str]],
    message_count: Optional[int] = None,
) -> bool:
    """Every chunk of the turn, found within ONE run of adjacent messages.

    Chunk by chunk independently is not enough: they are consecutive slices of one turn,
    so letting each pick its own place reassembles a turn from parts that never sat
    together (head on the current answer, tail on a later message repeating what the edit
    removed). Reproduced before the run was shared. The run is bounded by at least the
    number of messages the turn was rendered from, so a live turn always fits.
    """
    if not rows or not transcript:
        return False
    probe_lists = [_probe_entries(row["text"]) for row in rows]
    if any(not probes for probes in probe_lists):
        return False
    # Bound by the messages the turn was rendered from, not the lines: lines let a long answer's tail be
    # satisfied outside the turn.
    window = (
        int(message_count)
        if message_count
        else (_rendered_message_count(rows) or sum(len(probes) for probes in probe_lists))
    )

    def _one_run_from(start: int) -> bool:
        last = min(len(transcript), start + window)
        position = start
        cursor = 0
        opened_at = None
        partial_tail = False
        # The next chunk may restart before where the previous finished: CHUNK_OVERLAP can carry a whole
        # short message, and a strict resume retired unedited documents.
        floor = start
        for probes in probe_lists:
            found = None
            for candidate in range(position, floor - 1, -1):
                found = _scan_probes(probes, transcript, candidate, last)
                if found is not None:
                    break
            if found is None:
                return False
            position, cursor, chunk_opened_at, partial_tail, floor = found
            # An edit that prepends to the turn's FIRST message shows up here and nowhere else.
            if opened_at is None:
                opened_at = chunk_opened_at
        # The turn has to cover its messages end to end: an edit that keeps the old text and adds to it
        # leaves every probe matching.
        if opened_at:
            return False
        return partial_tail or cursor >= len(transcript[position])

    return any(_one_run_from(start) for start in range(len(transcript)))


def _conversation_order(row) -> tuple:
    """Sort key putting recalled turns in the order they were said.

    NULL ordinals sort FIRST, and that is not a fallback so much as a fact: they were
    written by a build that had no such column, so they genuinely predate every numbered
    turn in the same scope. Within a turn, `chunk_index` keeps a long message's pieces
    contiguous and in order, which relevance ordering gets wrong today. `created_at`
    breaks ties, because the ordinal is deliberately not UNIQUE: the write lock is
    best-effort, so two concurrent archive passes can compute the same MAX + 1 and must
    tie-break rather than raise.
    """
    if row is None:
        return (2, 0, "", 0)
    ordinal = tool._row_value(row, "archive_ordinal")
    created = tool._row_value(row, "created_at") or ""
    index = tool._row_value(row, "chunk_index") or 0
    if ordinal is None:
        return (0, 0, created, index)
    return (1, int(ordinal), created, index)


def _above_floor(hits: list, min_dense_score: float) -> list:
    """Candidates clearing the forced path's cosine floor. Off (0.0) returns them all.

    FORCED path only: an automatic lookup returning whatever shares a stopword with the
    question is worse than none, since that block is the model's first sight of the search
    tool. Lexical-only hits are kept, because gating them on a similarity they never
    carried would delete the exact-identifier hits this archive is best at.
    """
    if min_dense_score <= 0:
        # Nothing tied: skip the row fetch entirely, which is the common case.
        return hits
    return [hit for hit in hits if hit.dense_score is None or hit.dense_score >= min_dense_score]


def _ends_first_within_ties(conn, hits: list) -> list:
    """Reorder each run of EQUAL lexical scores as newest, oldest, next-newest, ...

    A tie is not an order. FTS5 floors the IDF of a term appearing in more than half the
    index at 1e-6, so in a per-thread archive every turn naming the subject of the
    conversation can score identically -- measured at ONE distinct bm25 across eight
    revisions of the same variable. The candidate list then arrives in rowid order, the
    caller truncates it at `top_k`, and the slots go to the OLDEST turns purely because
    SQLite emitted them first: the current value never reached the model, while
    `format_conversation_recall` told it a later turn supersedes an earlier one. A stale
    answer presented as the authoritative one is worse than a miss.

    Newest-first outright is the obvious fix and it is wrong: measured, it fails BOTH
    `test_asking_what_it_was_originally_still_returns_the_first_assignment` -- the guard
    against a fix that just returns the latest thing it can find -- and the tie test
    below it. Both ends is what keeps "what is it now" and "what was it originally"
    answerable out of the same tied run.

    Only WITHIN a run of equal scores, so nothing ever moves past a chunk the ranking pass
    actually separated. Ordering is `_conversation_order`, so legacy NULL ordinals still
    count as oldest.
    """
    if not config.CONVERSATION_QUERY_FOCUS:
        # Reordering candidates is selection, not presentation, so it belongs behind the rollback knob.
        return hits
    if len(hits) < 2:
        return hits
    if len({hit.lexical_score for hit in hits}) == len(hits):
        return hits
    try:
        rows = store.chunks_by_id(conn, [hit.chunk_id for hit in hits])
    except Exception:  # noqa: BLE001 -- ordering must never break a recall
        return hits
    ordered: list = []
    run: list = []

    def _flush():
        if not run:
            return
        by_time = sorted(run, key = lambda hit: _conversation_order(rows.get(hit.chunk_id)))
        while by_time:
            ordered.append(by_time.pop())
            if by_time:
                ordered.append(by_time.pop(0))

    for hit in hits:
        if run and run[-1].lexical_score != hit.lexical_score:
            _flush()
            run = []
        run.append(hit)
    _flush()
    return ordered


def _lexical_pass(
    conn,
    scope: str,
    query: str,
    model,
    k: int,
    expression,
    *,
    newest_first: bool = False,
    oldest_first: bool = False,
) -> list:
    if newest_first or oldest_first:
        # The archive's lexical legs are always mode "lexical", so this is the same call one layer down, with
        # the tie-break reversed.
        return _ends_first_within_ties(
            conn,
            retrieval.retrieve_lexical(
                conn,
                scope,
                query,
                k,
                match_query = expression,
                newest_first = newest_first,
                oldest_first = oldest_first,
            ),
        )
    return _ends_first_within_ties(
        conn,
        retrieval.retrieve_hybrid(
            conn,
            scope,
            query,
            k = k,
            model_name = model,
            mode = "lexical",
            lexical_query = expression,
        ),
    )


def _both_ends(oldest: list, newest: list) -> list:
    """One candidate list from a run fetched from each end, oldest side first, no repeats."""
    seen = {hit.chunk_id for hit in oldest}
    return oldest + [hit for hit in newest if hit.chunk_id not in seen]


def _focused_lexical(conn, scope: str, query: str, model, fetch: int) -> list:
    """Archive lexical candidates: the identifiers FILTER, the content words RANK.

    The identifier pass cannot also do the ranking. FTS5 floors the BM25 IDF of a term
    that appears in more than half of the index at 1e-6 (`ext/fts5/fts5_aux.c`: "if
    (N < 2*nHit), the IDF is negative. Which is undesirable. So the minimum allowable IDF
    is (1e-6)"), and in a per-thread archive the identifier the whole conversation is
    about is exactly such a term. Its hits therefore come back in an order that carries
    no information, so cutting that pass off at ``fetch`` drops turns effectively at
    random -- measured on a 20-turn archive naming one variable, the turn stating its
    current value fell outside the 16 candidates and the recall answered with the four
    oldest turns instead.

    So the identifiers decide WHICH chunks are eligible and the content-word pass decides
    the order among them. Chunks the ranking pass never saw keep their place behind the
    ones it did, and chunks that match only the content words stay last, which is what
    keeps every slot on the subject of the question.
    """
    expressions = (
        store.conversation_match_queries(query) if config.CONVERSATION_QUERY_FOCUS else [None]
    )
    # A query made ONLY of an identifier would otherwise take the one-ended fetch the comment below exists to prevent.
    if not expressions or expressions[0] is None:
        return _lexical_pass(conn, scope, query, model, fetch, None)
    # Fetch from BOTH ENDS of the tied run: hits on the conversation's own identifier tie on the IDF
    # floor and SQLite returns them in rowid order, so LIMIT 256 means the oldest 256.
    _newest_half = _BRANCH_FILTER_MAX_CANDIDATES // 2
    # Re-order over the MERGED run: concatenating two ends-first halves leaves the newest end behind a
    # full window of old turns.
    strict = _ends_first_within_ties(
        conn,
        _both_ends(
            _lexical_pass(
                conn,
                scope,
                query,
                model,
                _BRANCH_FILTER_MAX_CANDIDATES - _newest_half,
                expressions[0],
                oldest_first = True,
            ),
            _lexical_pass(
                conn, scope, query, model, _newest_half, expressions[0], newest_first = True
            ),
        ),
    )
    # One expression means the filter pass already IS the ranking pass.
    if len(expressions) < 2:
        return strict
    # Rank to the same bound as the filter pass: at fetch rows the window can be spent on chunks that
    # never name the identifier.
    _loose_k = max(fetch, _BRANCH_FILTER_MAX_CANDIDATES)
    _loose_newest = _loose_k // 2
    loose = _ends_first_within_ties(
        conn,
        _both_ends(
            _lexical_pass(
                conn,
                scope,
                query,
                model,
                _loose_k - _loose_newest,
                expressions[-1],
                oldest_first = True,
            ),
            _lexical_pass(
                conn, scope, query, model, _loose_newest, expressions[-1], newest_first = True
            ),
        ),
    )
    # Ask the index for eligibility: the strict pass is capped and arbitrarily ordered, so the turn
    # stating the current value can be the one left out.
    eligible = {hit.chunk_id for hit in strict}
    try:
        eligible |= store.lexical_matching_ids(
            conn, [hit.chunk_id for hit in loose], expressions[0]
        )
    except Exception:
        # An exact membership test is an improvement, not a dependency: the strict rows on their own are what
        # this did before.
        logger.warning("conversation_archive.eligibility_probe_failed", exc_info = True)
    ranked = [hit for hit in loose if hit.chunk_id in eligible]
    already = {hit.chunk_id for hit in ranked}
    ranked += [hit for hit in strict if hit.chunk_id not in already]
    already |= {hit.chunk_id for hit in strict}
    ranked += [hit for hit in loose if hit.chunk_id not in already]
    return ranked


def _candidates(conn, scope: str, query: str, model, fetch: int, thread_id: str) -> list:
    """Up to ``fetch`` archive chunks for ``query``: lexical first, hybrid for the rest.

    The lexical pass runs TWICE when the question contains an identifier: once requiring
    one of them, then once over the content words. Requiring them is what stops an
    incidental word in the question outranking the subject of the whole conversation (see
    `store.conversation_match_queries`); running the permissive pass as well is what
    orders the survivors and what stops a filter that matches nothing from reading
    as an empty archive (see `_focused_lexical`). The merged list is what the caller's
    widening loop measures, so a filter returning few rows cannot be mistaken for
    "nothing left to widen into".
    """
    hits: list = []
    seen: set = set()
    for hit in _focused_lexical(conn, scope, query, model, fetch):
        if hit.chunk_id not in seen:
            hits.append(hit)
            seen.add(hit.chunk_id)
        if len(hits) >= fetch:
            break
    if len(hits) < fetch:
        try:
            seen = {hit.chunk_id for hit in hits}
            for hit in retrieval.retrieve_hybrid(
                conn, scope, query, k = fetch, model_name = model, mode = "hybrid"
            ):
                if hit.chunk_id not in seen:
                    hits.append(hit)
                    seen.add(hit.chunk_id)
                if len(hits) >= fetch:
                    break
        except Exception:
            # Dense retrieval raises rather than degrading when no embedder can start, and the lexical hits
            # stand on their own.
            logger.warning(
                "conversation_archive.dense_unavailable thread_id=%s", thread_id, exc_info = True
            )
    return hits


def recall(
    thread_id: str,
    query: str,
    *,
    top_k: Optional[int] = None,
    branch_messages: Optional[list[dict]] = None,
    extra_queries: Optional[list[str]] = None,
    forced: bool = False,
) -> Optional[tuple[str, list[dict]]]:
    """Most relevant archived turns for ``query``, rendered like any other RAG hit.

    LEXICAL FIRST, then hybrid for the rest of the budget. Recalling your own conversation
    is mostly exact match (a name, a number, a code pasted twenty turns ago), which lives
    or dies on rare tokens. Measured on a 30-turn walkthrough with identical boilerplate
    per turn: the needle chunk ranked 3rd lexically at any k, was never returned by dense
    retrieval at all, and RRF fusion pushed it to 16th behind 30 useless dense hits.
    Hybrid alone lost the answer. Dense still earns its place for paraphrased recall, so
    it fills whatever the lexical pass leaves.

    No relevance floor, unlike ``tool.search_for_autoinject``: its 0.70 cosine gate keeps
    off-topic documents out, but here the passages ARE this conversation and the
    alternative to a weak match is no memory at all. It also keeps lexical-only hits,
    since ``filter_min_score`` only gates hits carrying a dense score.
    """
    query = (query or "").strip()
    if not thread_id or not query or not enabled():
        return None
    min_dense_score = config.CONVERSATION_FORCED_MIN_SCORE if forced else 0.0

    scope = store.conversation_archive_scope(thread_id)
    limit = top_k or config.CONVERSATION_ARCHIVE_TOP_K
    # Two queries rather than one concatenated string: the filler's tokens dilute the instruction's
    # identifiers and a conjunctive pass would AND two unrelated intents.
    queries = [query] + [q.strip() for q in (extra_queries or []) if (q or "").strip()]
    if len(queries) > 1:
        share = max(1, -(-limit // len(queries)))
        merged: list = []
        seen_ids: set = set()
        for index, one in enumerate(reversed(queries)):
            # Anchors first: they are the reason the extra query was added at all.
            room = limit - len(merged) if index == len(queries) - 1 else share
            if room <= 0:
                break
            # Over-fetch by what is already held: the two queries overlap by construction, and cutting before
            # the dedup lost a slot per shared chunk.
            found = recall(
                thread_id,
                one,
                top_k = room + len(seen_ids),
                branch_messages = branch_messages,
                forced = forced,
            )
            if not found:
                continue
            fresh = [source for source in found[1] if source["chunkId"] not in seen_ids]
            # Re-sort by retrieval rank, not score: the score is rounded for display, so a tied run preserves
            # the inner call's chronological order.
            fresh.sort(
                key = lambda source: (
                    source.get("rank") if source.get("rank") is not None else 1 << 30,
                    -(source.get("score") or 0.0),
                )
            )
            for source in fresh[:room]:
                seen_ids.add(source["chunkId"])
                merged.append(source)
        if not merged:
            return None
        if config.CONVERSATION_RECALL_ORDER == "chronological":
            # Same key as _conversation_order: a turn with no ordinal predates every numbered one, and
            # chunkIndex keeps a long turn's pieces in writing order.
            merged.sort(
                key = lambda source: (
                    source.get("turn") is not None,
                    source.get("turn") or 0,
                    source.get("createdAt") or "",
                    source.get("chunkIndex") or 0,
                )
            )
            kept = merged[:limit]
            return tool.render_conversation_sources(kept), kept
        kept = merged[:limit]
        return tool.render_sources(kept), kept
    conn = None
    try:
        conn = rag_db.get_connection()
        model = config.effective_embedding_model()
        # The stored rows are the whole DAG; falling back to them still beats not filtering.
        transcript = branch_message_texts(branch_messages) or _live_transcript(thread_id)
        fetch = limit * _BRANCH_FILTER_OVERFETCH
        rows: dict = {}
        hits: list = []
        live_documents: dict = {}
        while True:
            fetched = _candidates(conn, scope, query, model, fetch, thread_id)
            if not fetched:
                return None
            # The floor filters CANDIDATES before the cut to limit; after the slice it was a deletion, so a
            # forced recall could return nothing where the unforced one returned 4.
            # exhausted is read off the RAW fetch, or the widening loop stops one page early.
            exhausted = len(fetched) < fetch
            candidates = _above_floor(fetched, min_dense_score)
            if not candidates:
                if exhausted or fetch >= _BRANCH_FILTER_MAX_CANDIDATES:
                    logger.info(
                        "conversation_archive.recall_below_floor thread_id=%s floor=%.2f",
                        thread_id,
                        min_dense_score,
                    )
                    return None
                fetch = min(_BRANCH_FILTER_MAX_CANDIDATES, fetch * _BRANCH_FILTER_OVERFETCH)
                continue
            rows = store.chunks_by_id(conn, [hit.chunk_id for hit in candidates])
            if not transcript:
                hits = candidates[:limit]
                break
            hits = [
                hit
                for hit in candidates
                if hit.chunk_id in rows
                and _document_on_live_branch(
                    conn, rows[hit.chunk_id]["document_id"], transcript, live_documents
                )
            ]
            if len(hits) != len(candidates):
                logger.info(
                    "conversation_archive.branch_filtered thread_id=%s kept=%d of %d",
                    thread_id,
                    len(hits),
                    len(candidates),
                )
            # Enough live hits, or nothing more to widen into: an abandoned branch can outrank the live one, but
            # it cannot outrank it forever.
            if len(hits) >= limit or exhausted or fetch >= _BRANCH_FILTER_MAX_CANDIDATES:
                hits = hits[:limit]
                break
            fetch = min(_BRANCH_FILTER_MAX_CANDIDATES, fetch * _BRANCH_FILTER_OVERFETCH)
        if not hits:
            return None
        if config.CONVERSATION_RECALL_ORDER == "chronological":
            # Keep retrieval rank before the chronological sort: the score is rounded to four places, so on a
            # tied archive the refill takes the oldest.
            rank_of = {hit.chunk_id: rank for rank, hit in enumerate(hits)}
            # AFTER the top-k slice, never before: sorting first would make the slice take the oldest turns.
            hits.sort(key = lambda hit: _conversation_order(rows.get(hit.chunk_id)))
            text, sources = tool.format_conversation_recall(rows, hits)
            for source in sources:
                source["rank"] = rank_of.get(source.get("chunkId"))
        else:
            text, sources = tool._format(rows, hits)
        return (text, sources) if sources else None
    except Exception:
        logger.warning("conversation_archive.recall_failed thread_id=%s", thread_id, exc_info = True)
        return None
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def _scope_select(scope: str, created_before: Optional[str]) -> tuple:
    """The scope's document ids, optionally cut at ``created_before``. See `delete_for_thread`."""
    if not created_before:
        return ("SELECT id FROM documents WHERE scope=?", (scope,))
    return (
        "SELECT id FROM documents WHERE scope=? AND created_at<?",
        (scope, created_before),
    )


def _delete_scope_without_vec(
    scope: str,
    thread_id: str,
    *,
    created_before: Optional[str] = None,
) -> int:
    """Delete a scope's text-bearing rows over a connection with no sqlite-vec.

    Deletion must not depend on the optional native extension. Archives are only WRITTEN
    while vec0 loads, but it can stop loading afterwards (a venv change, common on macOS),
    and a delete that silently does nothing leaves a deleted conversation on disk ready to
    answer again once vec0 returns.

    The chunks_vec rows are unreachable from here and left behind. They carry vectors, not
    text, and every read path resolves through ``chunks`` joined to ``documents``, both
    gone, so nothing can retrieve an orphan.
    """
    conn = None
    removed = 0
    try:
        conn = rag_db.get_metadata_connection()
        documents = [
            row["id"] for row in conn.execute(*_scope_select(scope, created_before)).fetchall()
        ]
        for document_id in documents:
            conn.execute(
                "DELETE FROM chunks_fts WHERE chunk_id IN "
                "(SELECT id FROM chunks WHERE document_id=?)",
                (document_id,),
            )
            conn.execute("DELETE FROM chunks WHERE document_id=?", (document_id,))
            conn.execute("DELETE FROM documents WHERE id=?", (document_id,))
            removed += 1
        conn.commit()
    except Exception:
        logger.warning(
            "conversation_archive.delete_without_vec_failed thread_id=%s", thread_id, exc_info = True
        )
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
    return removed


def delete_for_thread(thread_id: str, *, created_before: Optional[str] = None) -> int:
    """Drop a thread's archive. Called when the thread itself is deleted.

    ``created_before`` bounds the delete to documents archived before an ISO-8601 UTC
    instant, which is how a thread id that came BACK is handled. Skipping the scope
    wholesale spared the recreated chat's memory but also kept the deleted
    conversation's, under a live id with nothing left to sweep it: the endpoint reported
    success while the turns the user asked to delete stayed recallable in the new chat.
    Cutting at the moment the delete was accepted takes exactly the old conversation and
    leaves whatever the recreated thread has archived since.
    """
    if not thread_id:
        return 0
    scope = store.conversation_archive_scope(thread_id)
    conn = None
    removed = 0
    try:
        try:
            conn = rag_db.get_connection()
        except Exception:
            # No vec0 here: delete what can be deleted rather than nothing at all.
            return _delete_scope_without_vec(scope, thread_id, created_before = created_before)
        for row in conn.execute(*_scope_select(scope, created_before)).fetchall():
            store.delete_document(conn, row["id"])
            removed += 1
    except Exception:
        logger.warning("conversation_archive.delete_failed thread_id=%s", thread_id, exc_info = True)
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
    return removed
