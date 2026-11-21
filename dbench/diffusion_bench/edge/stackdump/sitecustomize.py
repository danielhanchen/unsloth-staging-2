"""Put first on a diffusion_bench-launched Studio's PYTHONPATH tail (setup_studio._write_shim): registers faulthandler on
SIGUSR1, so edge/surfaces.dump_stacks can get every thread's stack into the server log on a Hang (py-spy cannot
attach to a sibling under ptrace_scope 1). Then runs the sitecustomize this one shadows, if any."""

import faulthandler
import os
import signal
import sys

if hasattr(signal, "SIGUSR1"):
    faulthandler.register(signal.SIGUSR1, all_threads = True)

_here = os.path.dirname(os.path.abspath(__file__))
for _p in sys.path:
    _cand = os.path.join(_p or ".", "sitecustomize.py")
    if os.path.abspath(_p or ".") != _here and os.path.isfile(_cand):
        with open(_cand) as _fh:
            exec(compile(_fh.read(), _cand, "exec"), {"__name__": "sitecustomize", "__file__": _cand})
        break
