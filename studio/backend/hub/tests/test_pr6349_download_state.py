# Characterization test for PR #6349's rewrite of
# gguf_variants.get_gguf_variants_response._is_fully_downloaded.
#
# NOT part of the PR. Documents the live behaviour of the new download-state
# logic, in particular the reviewer concern (gemini :543 / codex :568) that a
# VISION GGUF whose main quant is on disk but whose mmproj adapter is absent is
# now reported `downloaded = True` via the `_quant_bytes_present` fallback.

import asyncio
from types import SimpleNamespace

from hub.services.models import gguf_variants
from hub.utils import state_dir


def _sibling(name: str, size: int, sha: str):
    return SimpleNamespace(rfilename=name, size=size, lfs={"sha256": sha})


def _variant(quant: str, size_bytes: int):
    return SimpleNamespace(
        filename=f"model-{quant}.gguf",
        quant=quant,
        display_label=None,
        size_bytes=size_bytes,
    )


def _downloaded(
    monkeypatch,
    tmp_path,
    *,
    repo_id,
    variant,
    on_disk,
    requirement_siblings,
    has_vision,
):
    """Drive the real get_gguf_variants_response with a fake HF snapshot and
    return result.variants[0] (the GgufVariantDetail)."""

    async def _inline(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    monkeypatch.setattr(gguf_variants.asyncio, "to_thread", _inline)

    org, repo = repo_id.split("/", 1)
    snapshot = tmp_path / "cache" / f"models--{org}--{repo}" / "snapshots" / "rev0"
    snapshot.mkdir(parents=True)
    for fname, nbytes in on_disk.items():
        (snapshot / fname).write_bytes(b"x" * nbytes)

    monkeypatch.setattr(
        gguf_variants,
        "list_gguf_variants",
        lambda *a, **k: ([variant], has_vision, None),
    )
    monkeypatch.setattr(gguf_variants, "iter_hf_cache_snapshots", lambda _r: [snapshot])
    # Force the requirement to come from our injected map (never a stale cache).
    monkeypatch.setattr(gguf_variants, "_variant_requirement_cache_get", lambda *a, **k: None)
    if requirement_siblings is None:
        reqs = {}
    else:
        reqs = gguf_variants._build_gguf_variant_requirements(requirement_siblings)
    monkeypatch.setattr(gguf_variants, "_gguf_all_variant_requirements", lambda *a, **k: reqs)
    monkeypatch.setattr(
        gguf_variants.download_registry,
        "incomplete_blob_hashes",
        lambda *a, **k: set(),
    )

    result = asyncio.run(gguf_variants.get_gguf_variants_response(repo_id))
    return result.variants[0]


_VISION_SIBLINGS = [
    _sibling("model-Q4_K_M.gguf", 100, "main-q4"),
    _sibling("mmproj-F16.gguf", 50, "mm-f16"),
]


def test_scenario1_non_vision_main_on_disk_is_downloaded(monkeypatch, tmp_path):
    # No requirement (e.g. plan unavailable / offline) -> bytes fallback.
    v = _downloaded(
        monkeypatch,
        tmp_path,
        repo_id="Org/TextRepo",
        variant=_variant("Q4_K_M", 100),
        on_disk={"model-Q4_K_M.gguf": 100},
        requirement_siblings=None,
        has_vision=False,
    )
    assert v.downloaded is True


def test_scenario2_vision_main_and_mmproj_present_is_downloaded(monkeypatch, tmp_path):
    # Fast path: main + mmproj both cached.
    v = _downloaded(
        monkeypatch,
        tmp_path,
        repo_id="Org/VisionRepo",
        variant=_variant("Q4_K_M", 100),
        on_disk={"model-Q4_K_M.gguf": 100, "mmproj-F16.gguf": 50},
        requirement_siblings=_VISION_SIBLINGS,
        has_vision=True,
    )
    assert v.downloaded is True


def test_scenario3_vision_main_present_mmproj_ABSENT_still_downloaded(monkeypatch, tmp_path):
    # THE FLAGGED BEHAVIOUR: required mmproj is missing, yet the main-GGUF bytes
    # fallback marks the vision variant downloaded / Run-ready.
    v = _downloaded(
        monkeypatch,
        tmp_path,
        repo_id="Org/VisionRepo",
        variant=_variant("Q4_K_M", 100),
        on_disk={"model-Q4_K_M.gguf": 100},  # no mmproj on disk
        requirement_siblings=_VISION_SIBLINGS,
        has_vision=True,
    )
    assert v.partial is False
    assert v.downloaded is True  # <- reviewer concern: should arguably be False


def test_scenario4_vision_main_absent_is_not_downloaded(monkeypatch, tmp_path):
    v = _downloaded(
        monkeypatch,
        tmp_path,
        repo_id="Org/VisionRepo",
        variant=_variant("Q4_K_M", 100),
        on_disk={},  # nothing on disk
        requirement_siblings=_VISION_SIBLINGS,
        has_vision=True,
    )
    assert v.downloaded is False


def test_scenario5_vision_mmproj_other_precision_is_downloaded(monkeypatch, tmp_path):
    # The legit case the PR fixes: plan prefers mmproj-F16 but only mmproj-BF16
    # is on disk; the new _any_mmproj_cached second branch accepts it.
    v = _downloaded(
        monkeypatch,
        tmp_path,
        repo_id="Org/VisionRepo",
        variant=_variant("Q4_K_M", 100),
        on_disk={"model-Q4_K_M.gguf": 100, "mmproj-BF16.gguf": 60},
        requirement_siblings=_VISION_SIBLINGS,  # plan only knows mmproj-F16
        has_vision=True,
    )
    assert v.downloaded is True
