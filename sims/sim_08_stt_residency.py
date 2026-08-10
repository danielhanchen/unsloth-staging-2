"""The cross-engine dictation release added after the field report.

stt_registry.load now unloads the other two engines, which is a behaviour change
on a path the chat composer shares with the Audio page. The invariants that
matter: only one engine ends up resident, a load that fails must not cost the
user the engine they were already using, a sidecar mid-request is left alone
rather than blocked on, and every sidecar really accepts the new `wait` kwarg,
since the last defect in this area was a method that did not exist on the
supported Python versions.
"""

from __future__ import annotations

import inspect
import os
import sys
import threading
import time
from pathlib import Path

import pytest

BACKEND = Path(
    os.environ.get(
        "PR_BACKEND",
        Path(__file__).resolve().parents[2] / "unsloth_pr7984" / "studio" / "backend",
    )
)
sys.path.insert(0, str(BACKEND))

from core.inference import stt_registry  # noqa: E402


class FakeSidecar:
    """Stands in for a sidecar: resident model, a busy flag, a load delay."""

    def __init__(self, name: str):
        self.name = name
        self.loaded_model: str | None = None
        self.busy = False
        self.load_delay = 0.0
        self.load_error: Exception | None = None
        self.unload_calls: list[bool] = []

    def load(self, model, request_cancel_event = None):
        if self.load_delay:
            time.sleep(self.load_delay)
        if self.load_error is not None:
            raise self.load_error
        self.loaded_model = model

    def unload(self, wait: bool = True) -> None:
        self.unload_calls.append(wait)
        if self.busy and not wait:
            return  # a sidecar serving a request keeps its model
        self.loaded_model = None

    def is_loading(self) -> bool:
        return False

    @property
    def device(self):
        return "cpu"


@pytest.fixture
def sidecars(monkeypatch):
    made = {name: FakeSidecar(name) for name in stt_registry.STT_ENGINES}
    monkeypatch.setattr(stt_registry, "sidecar_for", lambda engine: made[engine])
    return made


def resident(sidecars):
    return {name: s.loaded_model for name, s in sidecars.items() if s.loaded_model}


# ── the reported defect ───────────────────────────────────────────────────────

def test_loading_one_engine_releases_the_others(sidecars):
    """The field report: a Transformers Whisper and a llama.cpp Qwen3-ASR both
    resident, doubling VRAM for the whole keep-alive window."""
    stt_registry.load("small", "transformers")
    assert resident(sidecars) == {"transformers": "small"}

    stt_registry.load("qwen3-asr-0.6b", "mtmd")
    assert resident(sidecars) == {"mtmd": "qwen3-asr-0.6b"}


def test_reloading_the_same_engine_keeps_it(sidecars):
    stt_registry.load("small", "transformers")
    stt_registry.load("large-v3", "transformers")
    assert resident(sidecars) == {"transformers": "large-v3"}
    # It must never release the engine it just loaded into.
    assert sidecars["transformers"].unload_calls == []


def test_the_release_is_non_blocking(sidecars):
    """unload() takes the sidecar lock, and transcribe holds that lock for the
    whole request, so waiting here would stall a dictation load behind an
    unrelated transcription."""
    stt_registry.load("small", "transformers")
    assert sidecars["gguf"].unload_calls == [False]
    assert sidecars["mtmd"].unload_calls == [False]


def test_a_busy_engine_keeps_its_model_instead_of_blocking(sidecars):
    stt_registry.load("small", "transformers")
    sidecars["transformers"].busy = True
    stt_registry.load("qwen3-asr-0.6b", "mtmd")
    # Left resident on purpose; its own idle timer releases it.
    assert resident(sidecars) == {"transformers": "small", "mtmd": "qwen3-asr-0.6b"}


# ── the ordering the author called out ────────────────────────────────────────

def test_a_failed_load_does_not_cost_the_engine_already_in_use(sidecars):
    """A 409 for a model that is not downloaded must leave dictation working."""
    stt_registry.load("small", "transformers")
    sidecars["mtmd"].load_error = RuntimeError("not downloaded")

    with pytest.raises(RuntimeError):
        stt_registry.load("qwen3-asr-0.6b", "mtmd")

    assert resident(sidecars) == {"transformers": "small"}
    assert sidecars["transformers"].unload_calls == []


# ── concurrency ───────────────────────────────────────────────────────────────

def test_two_loads_on_different_engines_cannot_leave_both_resident(sidecars):
    for sidecar in sidecars.values():
        sidecar.load_delay = 0.05
    errors: list[BaseException] = []

    def worker(model, engine):
        try:
            stt_registry.load(model, engine)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [
        threading.Thread(target = worker, args = ("small", "transformers")),
        threading.Thread(target = worker, args = ("qwen3-asr-0.6b", "mtmd")),
        threading.Thread(target = worker, args = ("base", "gguf")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout = 30)

    assert not errors, errors[:2]
    assert len(resident(sidecars)) == 1, resident(sidecars)


def test_a_slow_load_does_not_deadlock_a_second_one(sidecars):
    """The whole load runs under _load_lock now, so a queued request waits for
    it. It must still complete rather than wedge."""
    sidecars["transformers"].load_delay = 0.4
    done = threading.Event()

    def first():
        stt_registry.load("large-v3", "transformers")

    def second():
        stt_registry.load("qwen3-asr-0.6b", "mtmd")
        done.set()

    a, b = threading.Thread(target = first), threading.Thread(target = second)
    a.start()
    time.sleep(0.05)
    b.start()
    a.join(timeout = 30)
    b.join(timeout = 30)
    assert done.is_set(), "the queued load never completed"
    assert len(resident(sidecars)) == 1


# ── backwards compatibility of the changed signatures ─────────────────────────

def test_unload_still_defaults_to_waiting(sidecars):
    """The manual eject path calls unload() with no kwarg and must still block
    until the model is really gone."""
    stt_registry.load("small", "transformers")
    stt_registry.unload()
    assert sidecars["transformers"].unload_calls[-1] is True
    assert resident(sidecars) == {}


def test_every_real_sidecar_accepts_the_wait_kwarg():
    """stt_registry.unload passes wait= to whatever sidecar_for returns. A
    sidecar missing it raises TypeError, which the registry logs and swallows,
    so the model silently stays resident. That is the shape of the last defect
    here, so pin it on the real classes rather than the fakes."""
    from core.inference.stt_ggml_sidecar import GgmlSttSidecar
    from core.inference.stt_mtmd_sidecar import MtmdSttSidecar
    from core.inference.stt_sidecar import WhisperSttSidecar

    for cls in (WhisperSttSidecar, GgmlSttSidecar, MtmdSttSidecar):
        signature = inspect.signature(cls.unload)
        assert "wait" in signature.parameters, f"{cls.__name__}.unload has no wait"
        assert signature.parameters["wait"].default is True, cls.__name__


def test_every_real_sidecar_accepts_the_load_cancel_kwarg():
    from core.inference.stt_ggml_sidecar import GgmlSttSidecar
    from core.inference.stt_mtmd_sidecar import MtmdSttSidecar
    from core.inference.stt_sidecar import WhisperSttSidecar

    for cls in (WhisperSttSidecar, GgmlSttSidecar, MtmdSttSidecar):
        assert "request_cancel_event" in inspect.signature(cls.load).parameters, cls.__name__
        assert "cancel_event" in inspect.signature(cls.cancel_transcription).parameters, (
            cls.__name__
        )


def test_a_sidecar_that_refuses_to_unload_does_not_stop_the_others(sidecars):
    stt_registry.load("small", "transformers")

    def explode(wait: bool = True):
        raise RuntimeError("busy")

    sidecars["gguf"].unload = explode
    sidecars["mtmd"].loaded_model = "stale"
    failed = stt_registry.unload()
    assert "gguf" in failed
    assert sidecars["mtmd"].loaded_model is None
