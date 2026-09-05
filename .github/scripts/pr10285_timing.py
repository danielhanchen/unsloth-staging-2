"""Where the time goes in one Windows sandboxed launch (staging only)."""
import os, sys, time
sys.path.insert(0, os.getcwd())
from core.inference import windows_lpac as w

def timed(label, fn):
    t = time.perf_counter()
    value = fn()
    print(f"PR10285_TIME {label} {time.perf_counter() - t:.2f}s", flush = True)
    return value

workdir = os.path.join(os.getcwd(), "pr10285-timing-work")
os.makedirs(workdir, exist_ok = True)
roots = timed("runtime_roots", lambda: w._runtime_roots(workdir, (sys.executable,)))
acl = tuple(p for p in roots if w._needs_explicit_acl(p))
print("PR10285_TIME acl_roots", acl, flush = True)
timed("validate_runtime_trees", lambda: w._validate_runtime_trees(acl))
identity = timed("create_identity", lambda: w._create_identity((*acl, workdir)))
for root in acl:
    timed(f"grant_read_execute {root}", lambda r = root: w._grant_read_execute(r, identity.sid))
timed("grant_modify_workdir", lambda: w._grant_modify(workdir, identity.sid))
timed("cleanup_revoke_and_delete_profile", identity.cleanup)
