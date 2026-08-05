"""Real worker bodies for the spawn-subprocess simulation. Module level so they pickle."""
from __future__ import annotations

import time


def worker(event_queue, mode: str, teardown_s: float = 0.0):
    """Emulate the shapes a real training worker can end in.

    The real worker emits its terminal event as its last act; everything after is
    interpreter/GPU teardown (and, when wandb is on, a sync).
    """
    event_queue.put({"type": "status", "message": "Training in progress...", "ts": time.time()})
    event_queue.put(
        {"type": "progress", "step": 10, "total_steps": 10, "loss": 0.4, "ts": time.time()}
    )

    if mode == "crash":
        return  # dies with no terminal event at all
    if mode == "hard_crash":
        import os
        os._exit(1)  # dies abruptly mid-run

    if mode == "error":
        event_queue.put({"type": "error", "error": "CUDA OOM", "stack": "", "ts": time.time()})
    else:
        event_queue.put(
            {
                "type": "complete",
                "output_dir": "/tmp/out",
                "status_message": "Training completed! Model saved to /tmp/out",
                "ts": time.time(),
            }
        )

    if mode in ("wedged", "error"):
        while True:  # never exits: the reported failure
            time.sleep(3600)
    if teardown_s:
        time.sleep(teardown_s)  # slow but legitimate teardown, e.g. a wandb sync
