import glob, hashlib, os, sys, tempfile
from pathlib import Path

sys.path.insert(0, "studio")
sys.path.insert(0, ".")
import install_mxc_prebuilt as m
from tests._shared.windows_console_stub import console_stub_bytes

m._is_elevated = lambda: True
d = Path(tempfile.mkdtemp()) / "O'Brien a b"
d.mkdir()
e = d / "wxc-host-prep.exe"
e.write_bytes(b"x")
for step in ("prepare-system-drive", "prepare-null-device"):
    try:
        m._run_host_prep(e, step)
        print("RESULT", step, "UNEXPECTED ok")
    except m.MxcInstallError as x:
        print("RESULT", step, x)
# A runnable stand-in with its digest pinned: the script must run it and hand back its exit code.
stub = console_stub_bytes(source="import sys\nprint('ARGV', sys.argv[1:])\nsys.exit(7)\n")
e.write_bytes(stub)
m.mxc_runtime.WXC_HOST_PREP_SHA256 = hashlib.sha256(stub).hexdigest()
for step in ("prepare-system-drive", "prepare-null-device"):
    print("RUN", step, "exit", m._run_host_prep(e, step))
print("LEFTOVER", glob.glob(os.path.join(os.environ["ProgramData"], "unsloth-mxc-host-prep-*")))
