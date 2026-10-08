"""As scripts/conftest.py (this dir can be the pytest rootdir alone): the host-wide GPU reading (gpu_sample) goes to a per-test dir and uses the
nvidia-smi backend, so tests with a fake nvidia-smi (PATH or a patched subprocess.run) see their fake,
not this host's real GPUs through NVML."""

import pytest


@pytest.fixture(autouse = True)
def _gpu_sample_isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("GPU_SAMPLE_DIR", str(tmp_path / "_gpu_sample"))
    monkeypatch.setenv("GPU_SAMPLE_BACKEND", "smi")
