"""SIM 2: real spawn subprocesses + the real pump thread + the real watchdog.

Studio sets _CTX = mp.get_context("spawn") on every platform, so this exercises the
same process semantics Windows and macOS use. No torch, no GPU, no unsloth.
"""
from __future__ import annotations

import sys
import threading
import time

import simharness as H
import simworker

T = H.T
FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  <- ' + detail if detail and not cond else ''}")
    if not cond:
        FAILS.append(name)


def wait_until(pred, timeout=60.0, tick=0.02):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(tick)
    return pred()


def run(mode, teardown_s=0.0, grace=None, arm=True):
    """Spin a real worker + the real pump, exactly as start_training wires them."""
    if grace is not None:
        H.G["_COMPLETE_EXIT_GRACE_S"] = grace
    b = H.TrainingBackend()
    b.current_job_id = f"job-{mode}"
    b._progress = T.TrainingProgress(is_training=True, status_message="Initializing training...")
    b._finalize_run_in_db = lambda **kw: None
    b._ensure_db_run_created = lambda: None
    b._finish_stopped_run = lambda *a, **k: None
    if not arm:
        b._start_stop_watchdog = lambda **kw: None

    q = T._CTX.Queue()
    proc = T._CTX.Process(target=simworker.worker, args=(q, mode, teardown_s), daemon=True)
    proc.start()
    b._proc = proc
    b._event_queue = q
    pump = threading.Thread(target=b._pump_loop, daemon=True)
    b._pump_thread = pump
    pump.start()
    return b, proc, pump


def main():
    print("\n=== A. WEDGED worker after natural completion (the bug) ===")
    b, proc, pump = run("wedged", grace=4)
    t0 = time.time()
    check("UI goes finished", wait_until(lambda: H.ui_active(b) is False, 60))
    t_ui = time.time() - t0
    # Wall time here includes spawn startup, which is slow on Windows/macOS runners.
    # Scenarios G and H carry the rigorous "independent of the watchdog" proof.
    check(f"UI unsticks without hanging ({t_ui:.2f}s)", t_ui < 20.0, f"{t_ui:.2f}s")
    check("worker really is still alive (old code hung here forever)", proc.is_alive())
    check("admission still blocked while worker lives", b.is_training_active() is True)
    check("phase == completed", H.derive_phase(b) == "completed", H.derive_phase(b))
    check("watchdog reaps the wedged worker", wait_until(lambda: not proc.is_alive(), 60))
    check("admission released after reap", wait_until(lambda: b.is_training_active() is False, 60))
    check("message survives the reap", b._progress.status_message.startswith("Training completed!"),
          repr(b._progress.status_message))
    pump.join(timeout=30)

    print("\n=== B. SLOW but legitimate teardown (wandb sync) is NOT truncated ===")
    b, proc, pump = run("normal", teardown_s=6.0, grace=60)
    check("UI finished at once, not after teardown", wait_until(lambda: H.ui_active(b) is False, 60))
    check("worker still tearing down", proc.is_alive())
    check("worker exits on its own", wait_until(lambda: not proc.is_alive(), 60))
    check("never force-killed (exitcode 0)", proc.exitcode == 0, f"exitcode={proc.exitcode}")
    pump.join(timeout=30)

    print("\n=== C. Normal fast completion: watchdog must be invisible ===")
    b, proc, pump = run("normal", grace=60)
    check("worker exits cleanly", wait_until(lambda: not proc.is_alive(), 60))
    check("exitcode 0 (not terminated)", proc.exitcode == 0, f"exitcode={proc.exitcode}")
    check("UI finished", wait_until(lambda: H.ui_active(b) is False, 60))
    check("admission released", wait_until(lambda: b.is_training_active() is False, 60))
    check("pump exits", (pump.join(timeout=30), not pump.is_alive())[1])

    print("\n=== D. Terminal ERROR then wedge ===")
    b, proc, pump = run("error", grace=4)
    check("UI finished at once", wait_until(lambda: H.ui_active(b) is False, 60))
    check("phase == error", H.derive_phase(b) == "error", H.derive_phase(b))
    check("wedged worker reaped", wait_until(lambda: not proc.is_alive(), 60))
    pump.join(timeout=30)

    print("\n=== E. Worker crashes with no terminal event ===")
    b, proc, pump = run("crash", grace=60)
    check("worker gone", wait_until(lambda: not proc.is_alive(), 60))
    check("pump finalizes", (pump.join(timeout=30), not pump.is_alive())[1])
    check("UI finished", H.ui_active(b) is False)
    check("surfaced as error", H.derive_phase(b) == "error", H.derive_phase(b))

    print("\n=== F. Hard crash (os._exit) mid-run ===")
    b, proc, pump = run("hard_crash", grace=60)
    check("worker gone", wait_until(lambda: not proc.is_alive(), 60))
    check("pump finalizes", (pump.join(timeout=30), not pump.is_alive())[1])
    check("UI finished", H.ui_active(b) is False)

    print("\n=== G. Backstop OFF (grace huge): UI must STILL unstick ===")
    # Proves the UI fix does not depend on the watchdog at all.
    b, proc, pump = run("wedged", grace=100000)
    check("UI finished without any reap", wait_until(lambda: H.ui_active(b) is False, 60))
    check("worker deliberately left alive", proc.is_alive())
    proc.kill()
    proc.join(timeout=5)

    print("\n=== H. Watchdog disarmed entirely: UI must STILL unstick ===")
    b, proc, pump = run("wedged", grace=4, arm=False)
    check("UI finished with no watchdog", wait_until(lambda: H.ui_active(b) is False, 60))
    check("worker still alive (nothing reaped it)", proc.is_alive())
    proc.kill()
    proc.join(timeout=5)

    print("\n" + "=" * 62)
    print(f"SIM 2: {'ALL PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
    sys.exit(1 if FAILS else 0)

if __name__ == '__main__':
    main()
