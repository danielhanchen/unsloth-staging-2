"""SIM 6: concurrency + races.

The watchdog, the pump and the HTTP handlers all touch the same state under one
non-reentrant lock. This hammers the combinations: status polls at /status+SSE rates while
terminal events land, watchdog escalation racing a reset, and back-to-back runs.
"""
from __future__ import annotations

import random
import sys
import threading
import time

import simharness as H

T = H.T
FAILS = []
ERRORS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  <- ' + detail if detail and not cond else ''}")
    if not cond:
        FAILS.append(name)


def complete_event():
    return {"type": "complete", "output_dir": "/tmp/out",
            "status_message": "Training completed! Model saved to /tmp/out"}


def guarded(fn):
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        ERRORS.append(f"{type(e).__name__}: {e}")


print("\n=== A. 12 pollers hammering status while terminal events land ===")
for trial in range(30):
    b = H.fresh_backend(job_id=f"job-{trial}")
    b._start_stop_watchdog = lambda **kw: None
    stop = threading.Event()
    seen = []

    def poll():
        while not stop.is_set():
            guarded(lambda: seen.append((b.is_training_active(), b.is_run_finished())))
            time.sleep(0.001)

    threads = [threading.Thread(target=poll, daemon=True) for _ in range(12)]
    for t in threads:
        t.start()
    time.sleep(random.uniform(0.005, 0.02))
    guarded(lambda: b._handle_event(complete_event()))
    time.sleep(0.02)
    stop.set()
    for t in threads:
        t.join(timeout=5)

check("no exceptions across 30 trials x 12 pollers", not ERRORS, str(ERRORS[:3]))
check("terminal state never regresses once set", b.is_run_finished() is True)

print("\n=== B. monotonicity: is_run_finished never flaps back mid-run ===")
flaps = 0
for trial in range(20):
    b = H.fresh_backend()
    b._start_stop_watchdog = lambda **kw: None
    obs = []
    stop = threading.Event()

    def watch():
        while not stop.is_set():
            obs.append(b.is_run_finished())
            time.sleep(0.0005)

    t = threading.Thread(target=watch, daemon=True)
    t.start()
    time.sleep(0.01)
    b._handle_event(complete_event())
    time.sleep(0.01)
    stop.set()
    t.join(timeout=5)
    # once True it must stay True for this run
    if True in obs and False in obs[obs.index(True):]:
        flaps += 1
check("no flapping in 20 trials", flaps == 0, f"{flaps} flaps")

print("\n=== C. watchdog escalation racing force_terminate/reset ===")
H.G["_COMPLETE_EXIT_GRACE_S"] = 0.02
H.G["_STOP_TIMEOUT_S"] = 100.0
races = 0
for trial in range(40):
    b = H.fresh_backend(job_id=f"race-{trial}")
    b._finish_stopped_run = lambda *a, **k: None
    b._handle_event(complete_event())          # arms the real watchdog
    # A concurrent user reset reaps at the same moment the watchdog escalates.
    th = threading.Thread(target=lambda: guarded(b.force_terminate), daemon=True)
    th.start()
    th.join(timeout=5)
    time.sleep(0.05)
    if b._progress.status_message.startswith("Training completed!"):
        races += 1
    wd = b._stop_watchdog
    if wd is not None:
        wd.join(timeout=5)
check("no exceptions in 40 escalation races", not ERRORS, str(ERRORS[:3]))
check("completed message survives every race", races == 40, f"{races}/40")

print("\n=== D. back-to-back runs: no terminal-state leak ===")
b = H.fresh_backend(job_id="run-1")
b._start_stop_watchdog = lambda **kw: None
leaks = 0
for i in range(50):
    b._handle_event(complete_event())
    if not b.is_run_finished():
        leaks += 1
    # start_training's reset, verbatim
    b._complete_seen.clear()
    b._progress = T.TrainingProgress(is_training=True, status_message="Initializing training...")
    b.current_job_id = f"run-{i + 2}"
    b._proc = H.FakeProc(alive=True)
    if b.is_run_finished():
        leaks += 1
check("50 consecutive runs, zero leaks", leaks == 0, f"{leaks} leaks")

print("\n=== E. pump loop survives the handle being dropped under it ===")
import queue as _q
drops = 0
for trial in range(25):
    b = H.TrainingBackend()
    b._proc = H.FakeProc(alive=True)
    b._event_queue = _q.Queue()
    b._finalize_run_in_db = lambda **kw: None
    b._ensure_db_run_created = lambda: None
    done = threading.Event()

    def reader(_qq, timeout_sec=None):
        b._proc = None          # watchdog finalize lands right here
        return None

    b._read_queue = reader
    th = threading.Thread(target=lambda: (guarded(b._pump_loop), done.set()), daemon=True)
    th.start()
    if not done.wait(timeout=5):
        drops += 1
    th.join(timeout=5)
check("pump always returns cleanly (25 trials)", drops == 0, f"{drops} hangs")
check("no exceptions from the pump", not ERRORS, str(ERRORS[:3]))

print("\n=== F. thread hygiene: no watchdog thread leak per run ===")
base = threading.active_count()
H.G["_COMPLETE_EXIT_GRACE_S"] = 0.02
wds = []
for i in range(25):
    b = H.fresh_backend(job_id=f"leak-{i}")
    b._finish_stopped_run = lambda *a, **k: None
    b._handle_event(complete_event())
    if b._stop_watchdog:
        wds.append(b._stop_watchdog)
    b._proc._alive = False
for w in wds:
    w.join(timeout=5)
time.sleep(0.3)
leaked = threading.active_count() - base
check("watchdog threads all exit", leaked <= 0, f"{leaked} extra threads")

print("\n" + "=" * 62)
print(f"SIM 6: {'ALL PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
sys.exit(1 if FAILS else 0)
