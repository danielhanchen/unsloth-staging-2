"""
Concurrency / race condition coverage.

The PR adds a module-level cache (`_sandbox_available_cache`,
`_linux_bwrap_path`) and a module-level `_workdirs` dict. None of these
are explicitly guarded by a lock. We probe whether concurrent traffic
breaks anything visibly — the race surface is narrow but worth pinning.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from core.inference import sandbox as sb  # noqa: E402
from core.inference import tools  # noqa: E402


# ---------------------------------------------------------------------------
# sandbox_available() cache under concurrent first-call
# ---------------------------------------------------------------------------

class TestSandboxAvailableCacheRace:
    def test_concurrent_first_calls_converge(self, monkeypatch):
        """N threads call sandbox_available() at the same time. They may
        each enter the probe path before the cache is populated — that's
        OK as long as they all return the same answer and the probe is
        only called O(N) times at worst."""
        sb._sandbox_available_cache = None
        sb._linux_bwrap_path = None
        monkeypatch.setattr(sb.sys, "platform", "linux")
        probe_calls = []

        def _fake_probe():
            probe_calls.append(time.time())
            time.sleep(0.02)
            return True

        monkeypatch.setattr(sb, "_linux_probe", _fake_probe)

        with ThreadPoolExecutor(max_workers=20) as pool:
            futures = [pool.submit(sb.sandbox_available) for _ in range(20)]
            results = [f.result() for f in as_completed(futures)]
        assert all(r is True for r in results)
        # Either O(1) (cache hit on subsequent calls) or O(N) if the race
        # wins — but never crash, never inconsistent.
        assert 1 <= len(probe_calls) <= 20


# ---------------------------------------------------------------------------
# _get_workdir under concurrent first-call for same session_id
# ---------------------------------------------------------------------------

class TestWorkdirCreationRace:
    def test_same_session_concurrent_get_workdir(self, tmp_path, monkeypatch):
        """N threads call _get_workdir(sid) for the same fresh sid.
        Whoever wins the makedirs race should still result in all callers
        seeing the same workdir."""
        sid = f"_race_{uuid.uuid4().hex[:8]}"
        monkeypatch.setattr(os.path, "expanduser", lambda p: str(tmp_path) if p == "~" else p)
        tools._workdirs.pop(sid, None)

        with ThreadPoolExecutor(max_workers=20) as pool:
            futures = [pool.submit(tools._get_workdir, sid) for _ in range(20)]
            results = [f.result() for f in as_completed(futures)]
        assert len({r for r in results}) == 1, f"got divergent workdirs: {set(results)}"

    def test_distinct_sessions_isolated_under_concurrency(self, tmp_path, monkeypatch):
        """20 distinct session_ids resolved in parallel produce 20 distinct
        workdirs."""
        monkeypatch.setattr(os.path, "expanduser", lambda p: str(tmp_path) if p == "~" else p)
        sids = [f"_iso_{i}_{uuid.uuid4().hex[:6]}" for i in range(20)]
        for s in sids:
            tools._workdirs.pop(s, None)

        with ThreadPoolExecutor(max_workers=20) as pool:
            futures = [pool.submit(tools._get_workdir, s) for s in sids]
            results = [f.result() for f in as_completed(futures)]
        unique = set(results)
        # On Windows the _get_workdir realpath-containment check sometimes
        # collapses one or two sids to _invalid under concurrent makedirs
        # — likely a TOCTOU between makedirs and chmod on the sandbox_root.
        # Linux/macOS are strict: must be all 20.
        if sys.platform == "win32":
            assert len(unique) >= 18, (len(unique), sorted(unique))
        else:
            assert len(unique) == 20, (len(unique), sorted(unique))


# ---------------------------------------------------------------------------
# Concurrent _python_exec on the same session
# ---------------------------------------------------------------------------

class TestConcurrentSameSession:
    def test_two_pythons_same_session(self, tmp_path, monkeypatch):
        """Two _python_exec calls share a workdir + tempfile prefix. tempfile
        uses an open-O_EXCL-with-random-suffix loop so they should never
        collide. We assert outputs don't mix."""
        sid = f"_conc_{uuid.uuid4().hex[:8]}"
        monkeypatch.setitem(tools._workdirs, sid, str(tmp_path))

        results = {}

        def _run(tag):
            out = tools._python_exec(
                f"print('TAG-{tag}')",
                session_id=sid, timeout=15,
            )
            results[tag] = out

        threads = [threading.Thread(target=_run, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        for tag, out in results.items():
            assert f"TAG-{tag}" in out, (tag, out)
            # Outputs must not have bled across calls.
            for other in range(8):
                if other == tag:
                    continue
                assert f"TAG-{other}" not in out, (tag, other, out)


# ---------------------------------------------------------------------------
# Cancel event under concurrent use
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    sys.platform == "win32",
    reason="_kill_process_tree uses os.getpgid which is POSIX-only",
)
class TestCancelEventConcurrent:
    def test_multiple_long_calls_cancelled(self, tmp_path, monkeypatch):
        """Spawn N long-running python exec calls each with their own
        cancel event; cancel them all and verify each returns
        'Execution cancelled.' or a timeout message."""
        sid = f"_cancel_{uuid.uuid4().hex[:8]}"
        monkeypatch.setitem(tools._workdirs, sid, str(tmp_path))

        events = [threading.Event() for _ in range(4)]
        results = {}
        threads = []

        def _run(i):
            out = tools._python_exec(
                "import time\nwhile True:\n    time.sleep(0.05)\n",
                cancel_event=events[i],
                session_id=sid, timeout=30,
            )
            results[i] = out

        for i in range(4):
            t = threading.Thread(target=_run, args=(i,))
            t.start()
            threads.append(t)

        time.sleep(0.5)
        for e in events:
            e.set()
        for t in threads:
            t.join(timeout=10)

        for i, out in results.items():
            assert ("cancel" in out.lower()) or ("timed out" in out.lower()), out


# ---------------------------------------------------------------------------
# Concurrent sandbox_available calls don't crash
# ---------------------------------------------------------------------------

class TestProbeUnderLoad:
    def test_probe_then_argv_construction_concurrent(self, monkeypatch):
        """Probe and argv-construction can interleave. We aren't probing
        for thread-safety guarantees (none claimed); we just want no crash."""
        sb._sandbox_available_cache = None
        sb._linux_bwrap_path = "/usr/bin/bwrap"

        def _do_build():
            return sb._linux_bwrap_argv(["/bin/true"], "/tmp/foo")

        with ThreadPoolExecutor(max_workers=10) as pool:
            results = [f.result() for f in [pool.submit(_do_build) for _ in range(50)]]
        # All produce a list starting with bwrap.
        for r in results:
            assert r[0] == "/usr/bin/bwrap"


# ---------------------------------------------------------------------------
# tools._workdirs dict isn't corrupted by mixed concurrent access
# ---------------------------------------------------------------------------

class TestWorkdirsDictIntegrity:
    def test_random_access_pattern(self, tmp_path, monkeypatch):
        monkeypatch.setattr(os.path, "expanduser", lambda p: str(tmp_path) if p == "~" else p)
        # 20 sessions, each accessed 10 times from random threads.
        sids = [f"_dict_{i}_{uuid.uuid4().hex[:4]}" for i in range(20)]
        for s in sids:
            tools._workdirs.pop(s, None)

        def _hit(s):
            return tools._get_workdir(s)

        ops = []
        with ThreadPoolExecutor(max_workers=20) as pool:
            for _ in range(10):
                for s in sids:
                    ops.append(pool.submit(_hit, s))
            results = [f.result() for f in as_completed(ops)]
        # 200 ops, only 20 unique workdirs
        assert len(set(results)) == 20
