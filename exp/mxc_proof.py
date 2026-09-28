import glob, os, sys, tempfile
from pathlib import Path

sys.path.insert(0, "studio")
import install_mxc_prebuilt as m

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
print("LEFTOVER", glob.glob(os.path.join(os.environ["ProgramData"], "unsloth-mxc-host-prep-*")))
