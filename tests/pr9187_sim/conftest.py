"""Standalone harness for the PR 9187 simulation campaign.

Lives outside the repo on purpose: these scenarios are evidence for the review, not
something to bolt onto someone else's branch. The env setup mirrors
studio/backend/tests/conftest.py so the imports resolve the same way the app's own
suite resolves them.

Run from the backend root:
    python -m pytest ../../../temp/pr9187_sim -q
or with an explicit interpreter venv, see run_matrix.sh.
"""

import itertools
import os
import sys
from pathlib import Path

import pytest

def _find_backend_root() -> Path:
    """The backend package root, however this suite was checked out.

    Locally it lives beside the workspace; in CI it is `studio/backend` in the repo.
    Walk up looking for the marker rather than counting `..`s.
    """
    override = os.environ.get("PR9187_BACKEND_ROOT")
    if override:
        return Path(override).resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "studio" / "backend"
        if (candidate / "storage" / "studio_db.py").is_file():
            return candidate.resolve()
        candidate = parent / "unsloth" / "studio" / "backend"
        if (candidate / "storage" / "studio_db.py").is_file():
            return candidate.resolve()
    raise RuntimeError("could not locate studio/backend from " + str(here))


_BACKEND_ROOT = _find_backend_root()
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

# Same guards the backend suite sets; without these a CPU-only host cannot import the
# inference modules at all.
os.environ.setdefault("UNSLOTH_ALLOW_CPU", "1")
os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
os.environ.setdefault("UNSLOTH_DIFFUSION_ATTENTION_INSTALL", "0")
os.environ.setdefault("UNSLOTH_STUDIO_DISABLE_DEVICE_PROBE", "1")
os.environ.setdefault("UNSLOTH_SETTLE_DELAY_S", "0")


@pytest.fixture(scope = "session")
def _studio_home_root(tmp_path_factory):
    return tmp_path_factory.mktemp("studio_homes")


_home_counter = itertools.count()


@pytest.fixture(autouse = True)
def _isolate_studio_home(_studio_home_root, monkeypatch):
    """A fresh studio.db per scenario, so no scenario can see another's rows."""
    home = _studio_home_root / f"home-{next(_home_counter)}"
    home.mkdir()
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(home))
    for name, module in tuple(sys.modules.items()):
        if name.startswith(("storage.", "hub.storage.")) and hasattr(module, "_schema_ready"):
            monkeypatch.setattr(module, "_schema_ready", False)
    yield home


@pytest.fixture(autouse = True)
def _reset_active_generations():
    from state import active_generations

    active_generations.reset_for_tests()
    yield
    active_generations.reset_for_tests()


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: scenario that takes more than a few seconds")
