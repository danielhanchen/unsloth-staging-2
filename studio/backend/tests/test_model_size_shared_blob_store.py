# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import os
import sys
import types
from pathlib import Path

import pytest

# Keep this test runnable where optional logging deps are not installed.
if "structlog" not in sys.modules:

    class _DummyLogger:
        def __getattr__(self, _name):
            return lambda *args, **kwargs: None

    sys.modules["structlog"] = types.SimpleNamespace(
        BoundLogger = _DummyLogger,
        get_logger = lambda *args, **kwargs: _DummyLogger(),
    )

import routes.models as models_route

WEIGHTS = b"w" * 4096


def _snapshot(cache_root: Path, repo_id: str) -> Path:
    snapshot = cache_root / f"models--{repo_id.replace('/', '--')}" / "snapshots" / "rev"
    snapshot.mkdir(parents = True)
    return snapshot


def _link_through_repo_blob(snapshot: Path, name: str, target: Path) -> None:
    """Point ``snapshot/name`` at ``target`` the way the hub does: via the repo's own blob."""
    repo_blob = snapshot.parent.parent / "blobs" / ("bf8b442c" * 8)
    repo_blob.parent.mkdir(exist_ok = True)
    repo_blob.symlink_to(os.path.relpath(target, repo_blob.parent))
    (snapshot / name).symlink_to(os.path.relpath(repo_blob, snapshot))


def _mark_shared_store(cache_root: Path) -> None:
    """The ownership marker hub writes at the root of a store it created."""
    store = cache_root / "blobs"
    store.mkdir(parents = True, exist_ok = True)
    (store / ".huggingface-shared-blobs").write_text("1\n")


def test_model_size_counts_weights_in_the_hub_shared_blob_store(tmp_path):
    snapshot = _snapshot(tmp_path, "org/model")
    _mark_shared_store(tmp_path)
    sha = "8788269b" * 8
    shared = tmp_path / "blobs" / sha[:2] / sha
    shared.parent.mkdir(parents = True)
    shared.write_bytes(WEIGHTS)
    _link_through_repo_blob(snapshot, "model.safetensors", shared)

    assert models_route._get_snapshot_model_size_bytes(str(snapshot)) == len(WEIGHTS)


def test_model_size_ignores_a_weight_under_an_unmarked_blobs_dir(tmp_path):
    """Sizing trusts the same roots attestation does, so a bare ``blobs`` folder is not one."""
    snapshot = _snapshot(tmp_path, "org/model")
    sha = "8788269b" * 8
    unmarked = tmp_path / "blobs" / sha[:2] / sha
    unmarked.parent.mkdir(parents = True)
    unmarked.write_bytes(WEIGHTS)
    _link_through_repo_blob(snapshot, "model.safetensors", unmarked)

    assert models_route._get_snapshot_model_size_bytes(str(snapshot)) is None


def test_model_size_ignores_a_weight_in_a_symlinked_shared_store(tmp_path):
    """Sizing and attestation answer the same question about a symlinked ``blobs`` leaf.

    Before, sizing accepted it and attestation did not, so the model reported a size while its
    run stayed unresumable and said the revision was not attested.
    """
    snapshot = _snapshot(tmp_path, "org/model")
    elsewhere = tmp_path / "another-volume"
    elsewhere.mkdir()
    (tmp_path / "blobs").symlink_to(elsewhere)
    (elsewhere / ".huggingface-shared-blobs").write_text("1\n")
    sha = "8788269b" * 8
    shared = elsewhere / sha[:2] / sha
    shared.parent.mkdir(parents = True)
    shared.write_bytes(WEIGHTS)
    _link_through_repo_blob(snapshot, "model.safetensors", shared)

    assert models_route._get_snapshot_model_size_bytes(str(snapshot)) is None


def test_model_size_still_counts_a_weight_in_the_repos_own_blobs(tmp_path):
    snapshot = _snapshot(tmp_path, "org/model")
    repo_blob = snapshot.parent.parent / "blobs" / ("a1b2c3d4" * 8)
    repo_blob.parent.mkdir()
    repo_blob.write_bytes(WEIGHTS)
    (snapshot / "model.safetensors").symlink_to(os.path.relpath(repo_blob, snapshot))

    assert models_route._get_snapshot_model_size_bytes(str(snapshot)) == len(WEIGHTS)


@pytest.mark.parametrize("target", ["outside-cache", "other-repo-blobs"])
def test_model_size_ignores_a_weight_linked_outside_the_cache(tmp_path, tmp_path_factory, target):
    snapshot = _snapshot(tmp_path, "org/model")
    if target == "outside-cache":
        escaped = tmp_path_factory.mktemp("elsewhere") / "blobs" / "87" / "weights"
    else:
        escaped = tmp_path / "models--org--other" / "blobs" / "weights"
    escaped.parent.mkdir(parents = True)
    escaped.write_bytes(WEIGHTS)
    _link_through_repo_blob(snapshot, "model.safetensors", escaped)

    assert models_route._get_snapshot_model_size_bytes(str(snapshot)) is None
