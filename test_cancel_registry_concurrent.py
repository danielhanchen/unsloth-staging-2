import os
import sys
import threading

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.join(_HERE, "studio", "backend")
if not os.path.isdir(_BACKEND):
    _BACKEND = os.path.join(_HERE, "..", "studio", "backend")
sys.path.insert(0, _BACKEND)

from routes.inference import _CANCEL_REGISTRY, _TrackedCancel, _cancel_by_keys


def _clear():
    _CANCEL_REGISTRY.clear()


def test_same_session_concurrent_requests_both_registered():
    _clear()
    ev1, ev2 = threading.Event(), threading.Event()
    with _TrackedCancel(ev1, "thread-A"):
        with _TrackedCancel(ev2, "thread-A"):
            bucket = _CANCEL_REGISTRY.get("thread-A")
            assert bucket is not None
            assert ev1 in bucket and ev2 in bucket


def test_same_session_cancel_reaches_all_concurrent_events():
    _clear()
    ev1, ev2 = threading.Event(), threading.Event()
    with _TrackedCancel(ev1, "thread-A"):
        with _TrackedCancel(ev2, "thread-A"):
            n = _cancel_by_keys(["thread-A"])
            assert n == 2
            assert ev1.is_set()
            assert ev2.is_set()


def test_exit_one_tracker_leaves_siblings_under_same_key():
    _clear()
    ev1, ev2 = threading.Event(), threading.Event()
    t1 = _TrackedCancel(ev1, "thread-B")
    t2 = _TrackedCancel(ev2, "thread-B")
    t1.__enter__()
    t2.__enter__()
    t1.__exit__(None, None, None)
    bucket = _CANCEL_REGISTRY.get("thread-B")
    assert bucket is not None
    assert ev1 not in bucket
    assert ev2 in bucket
    n = _cancel_by_keys(["thread-B"])
    assert n == 1
    assert not ev1.is_set()
    assert ev2.is_set()
    t2.__exit__(None, None, None)
    assert "thread-B" not in _CANCEL_REGISTRY


def test_last_tracker_exit_removes_empty_bucket():
    _clear()
    ev = threading.Event()
    with _TrackedCancel(ev, "thread-C"):
        assert "thread-C" in _CANCEL_REGISTRY
    assert "thread-C" not in _CANCEL_REGISTRY


def test_falsy_keys_are_filtered_not_registered():
    _clear()
    ev = threading.Event()
    with _TrackedCancel(ev, None, "", "real-key"):
        assert list(_CANCEL_REGISTRY.keys()) == ["real-key"]
        assert ev in _CANCEL_REGISTRY["real-key"]
    assert _CANCEL_REGISTRY == {}


def test_cancel_with_no_keys_does_not_set_any_event():
    _clear()
    ev1, ev2 = threading.Event(), threading.Event()
    with _TrackedCancel(ev1, "sess-A"):
        with _TrackedCancel(ev2, "sess-B"):
            assert _cancel_by_keys([]) == 0
            assert not ev1.is_set()
            assert not ev2.is_set()


def test_stale_cancel_after_exit_hits_only_new_registrant():
    _clear()
    ev1 = threading.Event()
    t1 = _TrackedCancel(ev1, "thread-D")
    t1.__enter__()
    t1.__exit__(None, None, None)
    assert "thread-D" not in _CANCEL_REGISTRY
    ev2 = threading.Event()
    with _TrackedCancel(ev2, "thread-D"):
        assert _cancel_by_keys(["thread-D"]) == 1
    assert not ev1.is_set()
    assert ev2.is_set()
