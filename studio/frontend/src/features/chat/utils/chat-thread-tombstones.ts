// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * PR-B6: tombstones for deleted chat threads with garbage collection.
 *
 * The legacy Dexie fallback still surfaces deleted threads until they age
 * out of the local database; tombstones mask those rows in the meantime.
 * Each tombstone carries a `deletedAt` timestamp so we can drop entries
 * older than TOMBSTONE_MAX_AGE_MS, keeping localStorage bounded under
 * heavy use.
 *
 * The serialized representation supports both the legacy plain-string
 * format (for back-compat with pre-PR-B6 installs) and the new tuple
 * form `{id, deletedAt}`. Old entries get a fresh `deletedAt` of "now"
 * on first read so they don't immediately expire.
 */

interface Tombstone {
  id: string;
  deletedAt: number;
}

const TOMBSTONES_KEY = "unsloth_chat_deleted_thread_ids";
const TOMBSTONE_MAX_AGE_MS = 90 * 24 * 60 * 60 * 1000; // 90 days
const TOMBSTONE_MAX_COUNT = 5000;

const deletedThreads = new Map<string, Tombstone>();

function canUseStorage(): boolean {
  return typeof window !== "undefined";
}

function nowMs(): number {
  return Date.now();
}

function isTombstone(value: unknown): value is Tombstone {
  return (
    typeof value === "object" &&
    value !== null &&
    typeof (value as Tombstone).id === "string" &&
    typeof (value as Tombstone).deletedAt === "number"
  );
}

function loadTombstones(): Tombstone[] {
  if (!canUseStorage()) return [];
  try {
    const raw = JSON.parse(localStorage.getItem(TOMBSTONES_KEY) ?? "[]");
    if (!Array.isArray(raw)) return [];
    const now = nowMs();
    const out: Tombstone[] = [];
    for (const item of raw) {
      if (typeof item === "string") {
        // Legacy plain-string format from pre-B6 installs.
        out.push({ id: item, deletedAt: now });
      } else if (isTombstone(item)) {
        out.push(item);
      }
    }
    return out;
  } catch {
    return [];
  }
}

function gc(): void {
  const cutoff = nowMs() - TOMBSTONE_MAX_AGE_MS;
  for (const [id, t] of deletedThreads) {
    if (t.deletedAt < cutoff) deletedThreads.delete(id);
  }
  // Cap absolute size: drop oldest if we somehow exceed the limit
  // (e.g. a script clearing thousands of threads at once).
  if (deletedThreads.size > TOMBSTONE_MAX_COUNT) {
    const sorted = Array.from(deletedThreads.entries()).sort(
      (a, b) => a[1].deletedAt - b[1].deletedAt,
    );
    const drop = sorted.slice(0, deletedThreads.size - TOMBSTONE_MAX_COUNT);
    for (const [id] of drop) deletedThreads.delete(id);
  }
}

function persist(): void {
  if (!canUseStorage()) return;
  try {
    const arr = Array.from(deletedThreads.values());
    localStorage.setItem(TOMBSTONES_KEY, JSON.stringify(arr));
  } catch {
    // ignore quota / serialization failures
  }
}

for (const t of loadTombstones()) {
  deletedThreads.set(t.id, t);
}
gc();

export function markChatThreadDeleted(threadId: string): void {
  deletedThreads.set(threadId, { id: threadId, deletedAt: nowMs() });
  gc();
  persist();
}

export function markChatThreadsDeleted(threadIds: Iterable<string>): void {
  const now = nowMs();
  for (const id of threadIds) {
    deletedThreads.set(id, { id, deletedAt: now });
  }
  gc();
  persist();
}

export function isChatThreadDeleted(threadId: string): boolean {
  return deletedThreads.has(threadId);
}

/**
 * PR-B5 rollback support: remove tombstones when the backend delete
 * fails so the sidebar row reappears.
 */
export function removeChatThreadTombstones(threadIds: Iterable<string>): void {
  let changed = false;
  for (const id of threadIds) {
    if (deletedThreads.delete(id)) changed = true;
  }
  if (changed) persist();
}

/**
 * PR-B6 finalization helper: clear all tombstones once the legacy Dexie
 * import is complete and Dexie has been wiped. The frontend caller is
 * responsible for confirming both conditions before invoking this.
 */
export function clearAllChatThreadTombstones(): void {
  if (deletedThreads.size === 0) return;
  deletedThreads.clear();
  persist();
}

export function __resetChatThreadTombstonesForTests(): void {
  deletedThreads.clear();
}
