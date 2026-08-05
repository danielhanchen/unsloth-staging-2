"""SIM 1: lifecycle state machine, before vs after, across every run ending.

OLD UI signal  = is_training_active()                      (what /status + SSE used)
NEW UI signal  = is_training_active() and not is_run_finished()

Also asserts the invariant that matters for regressions: is_training_active() itself is
untouched, so every GPU admission guard keeps its exact old answer.
"""
from __future__ import annotations

import sys
import simharness as H

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  <- ' + detail if detail and not cond else ''}")
    if not cond:
        FAILS.append(name)


def complete_event(msg="Training completed! Model saved to /tmp/out", out="/tmp/out"):
    return {"type": "complete", "output_dir": out, "status_message": msg}


print("\n=== 1. Natural completion, worker WEDGED (the reported bug) ===")
b = H.fresh_backend()
b._start_stop_watchdog = lambda **kw: None  # isolate: no reaping in this scenario
before_ui = b.is_training_active()
b._handle_event(complete_event())
check("OLD: UI still shows training (the bug)", before_ui is True and b.is_training_active() is True)
check("NEW: UI reports finished immediately", H.ui_active(b) is False)
check("phase == completed", H.derive_phase(b) == "completed", H.derive_phase(b))
check("message preserved", b._progress.status_message.startswith("Training completed!"))
check("admission guard still blocks (worker holds VRAM)", b.is_training_active() is True)

print("\n=== 2. Natural completion, worker exits promptly (normal path) ===")
b = H.fresh_backend()
b._start_stop_watchdog = lambda **kw: None
b._handle_event(complete_event())
b._proc._alive = False
check("UI finished", H.ui_active(b) is False)
check("admission guard released", b.is_training_active() is False)
check("phase == completed", H.derive_phase(b) == "completed")

print("\n=== 3. Terminal ERROR, worker wedged ===")
b = H.fresh_backend()
b._start_stop_watchdog = lambda **kw: None
b._handle_event({"type": "error", "error": "CUDA OOM", "stack": ""})
check("NEW: UI finished", H.ui_active(b) is False)
check("OLD: UI stuck", b.is_training_active() is True)
check("phase == error", H.derive_phase(b) == "error", H.derive_phase(b))

print("\n=== 4. Stop and save ===")
b = H.fresh_backend()
b._start_stop_watchdog = lambda **kw: None
b._should_stop = True
b._handle_event(complete_event(msg="Training stopped", out="/tmp/out"))
check("UI finished", H.ui_active(b) is False)
check("phase == stopped", H.derive_phase(b) == "stopped", H.derive_phase(b))
check("is_completed False for a stop", b._progress.is_completed is False)

print("\n=== 5. Cancel (stop without save) ===")
b = H.fresh_backend()
b._start_stop_watchdog = lambda **kw: None
b._should_stop = True
b._cancel_requested = True
b._handle_event(complete_event(msg="Training cancelled", out=None))
check("UI finished", H.ui_active(b) is False)
check("phase == stopped", H.derive_phase(b) == "stopped", H.derive_phase(b))
check("output_dir cleared on cancel", b._output_dir is None)

print("\n=== 6. Fresh backend, no run ever started ===")
b = H.TrainingBackend()
check("is_run_finished False", b.is_run_finished() is False)
check("UI idle", H.ui_active(b) is False)
check("phase == idle", H.derive_phase(b) == "idle", H.derive_phase(b))

print("\n=== 7. Terminal state must not leak into the NEXT run ===")
b = H.fresh_backend()
b._start_stop_watchdog = lambda **kw: None
b._handle_event(complete_event())
check("run 1 finished", b.is_run_finished() is True)
# start_training's reset block, verbatim in effect
b._complete_seen.clear()
b._progress = H.T.TrainingProgress(is_training=True, status_message="Initializing training...")
b.current_job_id = "job_2"
b._proc = H.FakeProc(alive=True)
check("run 2 NOT reported finished", b.is_run_finished() is False)
check("run 2 UI active", H.ui_active(b) is True)
check("run 2 phase == configuring", H.derive_phase(b) == "configuring", H.derive_phase(b))

print("\n=== 8. Spawn / start windows must never read as finished ===")
b = H.fresh_backend()
b._start_stop_watchdog = lambda **kw: None
b._handle_event(complete_event())          # previous run terminal
b._spawn_in_progress = True                # new run spawning, _progress not yet reset
check("spawn window -> not finished", b.is_run_finished() is False)
check("spawn window -> UI active", H.ui_active(b) is True)
b._spawn_in_progress = False
b._start_in_progress = True
check("start window -> not finished", b.is_run_finished() is False)
b._start_in_progress = False

print("\n=== 9. Worker crashes with no terminal event ===")
b = H.fresh_backend()
b._proc._alive = False
b._progress.is_training = False
b._progress.error = "Training process exited unexpectedly"
check("UI finished", H.ui_active(b) is False)
check("phase == error", H.derive_phase(b) == "error", H.derive_phase(b))

print("\n=== 10. Duplicate terminal events are idempotent ===")
b = H.fresh_backend()
armed = []
b._start_stop_watchdog = lambda **kw: armed.append(kw)
b._handle_event(complete_event())
b._handle_event(complete_event())
check("still finished", b.is_run_finished() is True)
check("watchdog armed per event, guard dedupes downstream", len(armed) == 2)
check("message stable", b._progress.status_message.startswith("Training completed!"))

print("\n=== 11. is_training_active() itself is UNCHANGED (admission guards) ===")
# Every guard (inference load, image/video gen, diffusion start, transformers install,
# /train/start) reads is_training_active(). It must answer purely on liveness.
cases = [
    ("mid-run, alive", True, False, True),
    ("completed, worker alive", True, True, True),
    ("completed, worker gone", False, True, False),
    ("errored, worker alive", True, False, True),
]
for name, alive, completed, expect in cases:
    b = H.fresh_backend(alive=alive)
    if completed:
        b._start_stop_watchdog = lambda **kw: None
        b._handle_event(complete_event())
    if not alive:
        b._proc._alive = False
    check(f"admission[{name}] == {expect}", b.is_training_active() is expect)

print("\n" + "=" * 62)
print(f"SIM 1: {'ALL PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
sys.exit(1 if FAILS else 0)
