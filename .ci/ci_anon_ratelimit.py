"""Self-contained ANONYMOUS rate-limit check, run on GitHub Actions runners
(Ubuntu/Windows/macOS) with no GITHUB_TOKEN/GH_TOKEN. Proves the PR's fast path
resolves the latest llama.cpp release with zero api.github.com calls, and reports
the OLD API path's behavior on the runner's shared anonymous IP budget.

Loads studio/install_llama_prebuilt.py from the checked-out PR branch.
"""
import importlib.util
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

# Anonymous by construction: our own urllib calls never send Authorization, and
# we blank the token envs so the installer's auth_headers() stays anonymous too.
os.environ.pop("GH_TOKEN", None)
os.environ.pop("GITHUB_TOKEN", None)

ROOT = Path(__file__).resolve().parents[1] if (Path(__file__).resolve().parents[1] / "studio").exists() \
    else Path(os.environ.get("GITHUB_WORKSPACE", "."))
MP = ROOT / "studio" / "install_llama_prebuilt.py"
print(f"runner: {sys.platform} | python {sys.version.split()[0]} | module: {MP}")
spec = importlib.util.spec_from_file_location("ilp_ci", MP)
ILP = importlib.util.module_from_spec(spec); sys.modules["ilp_ci"] = ILP
spec.loader.exec_module(ILP)
ILP.log = lambda *a, **k: None
ILP.log_lines = lambda *a, **k: None
REPO = ILP.DEFAULT_PUBLISHED_REPO


def anon_core():
    req = urllib.request.Request("https://api.github.com/rate_limit",
                                 headers={"Accept": "application/vnd.github+json",
                                          "User-Agent": "anon-ci-test"})
    with urllib.request.urlopen(req, timeout=30) as r:
        c = json.load(r)["resources"]["core"]
        return c["remaining"], c["limit"]


rem, lim = anon_core()
print(f"[budget] this runner IP anonymous api.github.com core: {rem}/{lim} remaining")

# --- NEW fast path, recording every outbound host ---------------------------
outbound = []
_dl, _op = ILP.download_bytes, ILP._URL_OPENER


class _Rec:
    def open(self, req, *a, **k):
        outbound.append(req.full_url if hasattr(req, "full_url") else req)
        return _op.open(req, *a, **k)


ILP.download_bytes = lambda url, *a, **k: (outbound.append(url), _dl(url, *a, **k))[1]
ILP._URL_OPENER = _Rec()
try:
    resolved = ILP._download_host_resolved_release(REPO)
    fast_tag = resolved.bundle.release_tag
    n_assets = len(resolved.checksums.artifacts)
finally:
    ILP.download_bytes, ILP._URL_OPENER = _dl, _op

api_hits = [u for u in outbound if "api.github.com" in u]
out_hosts = sorted({urlparse(u).hostname for u in outbound})
print(f"[fast]  resolved {fast_tag} with {n_assets} verified assets")
print(f"[fast]  outbound hosts: {out_hosts}")
print(f"[fast]  api.github.com calls: {len(api_hits)} (must be 0)")

# --- OLD API path -----------------------------------------------------------
try:
    bundles = list(ILP.iter_published_release_bundles(REPO))
    api_result = f"SUCCESS ({len(bundles)} bundles) - consumed anon api budget"
except Exception as e:
    api_result = f"{type(e).__name__}: {str(e)[:140]}"
rem2, _ = anon_core()
print(f"[api]   OLD path (api.github.com): {api_result}")
print(f"[api]   anon budget after API path: {rem2}/{lim}")

ok = (len(api_hits) == 0 and fast_tag)
print(f"\nRESULT: fast-path-zero-api={len(api_hits) == 0} resolved={bool(fast_tag)} -> "
      f"{'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
