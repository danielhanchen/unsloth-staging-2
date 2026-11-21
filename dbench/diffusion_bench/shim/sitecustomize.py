"""diffusion_bench shim, first on a stack's PYTHONPATH (budget.child_env): installs shim/dbench_hooks.py (simulated
total VRAM, parity dumps; inert without their env vars), then runs the sitecustomize this one shadows, if any."""

import os
import sys

if not getattr(sys, "_dbench_sitecustomize", False):
    sys._dbench_sitecustomize = True
    _here = os.path.dirname(os.path.abspath(__file__))
    try:
        if _here not in sys.path:
            sys.path.append(_here)
        import dbench_hooks

        dbench_hooks.install()
    except Exception:  # noqa: BLE001 - a broken shim must never stop the app
        pass
    _seen_self = False
    for _p in list(sys.path):
        _ap = os.path.abspath(_p or ".")
        if _ap == _here:
            _seen_self = True
            continue
        _cand = os.path.join(_ap, "sitecustomize.py")
        if _seen_self and os.path.isfile(_cand):
            with open(_cand) as _fh:
                exec(compile(_fh.read(), _cand, "exec"), {"__name__": "sitecustomize", "__file__": _cand})
            break
