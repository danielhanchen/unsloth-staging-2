# Throwaway cross-OS validation branch (pr5963-cross-os).
#
# The upstream tests/conftest.py pre-loads torch / unsloth_zoo to install a
# CUDA-detection spoof for the consolidated test suite. The llama.cpp prebuilt
# *selection* tests under tests/studio/install/ construct synthetic HostInfo
# objects and never import torch, so on bare GitHub-hosted runners (no torch,
# no GPU) that spoof is both unnecessary and would fail at collection time.
#
# This branch is never merged; it exists only to run the selection unit tests
# across Linux x64/arm64, Windows x64, and macOS arm64/intel. So we replace the
# heavy conftest with a no-op for the duration of the cross-OS run.
from __future__ import annotations
