"""H11/H12: the audio gallery on Linux, macOS and Windows filesystems.

The gallery is new on-disk state, so it is where platform differences bite:
NTFS refuses to unlink an open file, macOS and Windows fold case, every OS has
its own reserved names, and a half-written pair must never become a listable
clip. Windows-only failures are reproduced by making the syscall raise what
Windows raises, since the logic under test is the error handling, not the OS.
"""

from __future__ import annotations

import errno
import json
import os
import sys
import threading
from pathlib import Path

import pytest

# PR_BACKEND lets the identical suite run against a checkout on any OS runner.
BACKEND = Path(
    os.environ.get(
        "PR_BACKEND",
        Path(__file__).resolve().parents[2] / "unsloth_pr7984" / "studio" / "backend",
    )
)
sys.path.insert(0, str(BACKEND))

from core.inference import audio_gallery  # noqa: E402

WAV = b"RIFF" + b"\x00" * 40 + b"data" + b"\x00" * 64
META = {
    "prompt": "hello",
    "model": "unsloth/orpheus-3b-0.1-ft-GGUF",
    "audio_type": "snac",
    "sample_rate": 24000,
    "duration_s": 1.5,
    "created_at": "2026-08-10T00:00:00Z",
}


@pytest.fixture
def gallery(tmp_path, monkeypatch):
    """Point the gallery at a directory whose name exercises spaces, non-ASCII
    and a trailing dot-free unicode segment, the way a real Windows or macOS
    home directory can."""
    root = tmp_path / "Unsloth Studio" / "usuário ünïcode"
    root.mkdir(parents = True)
    monkeypatch.setattr(audio_gallery, "studio_root", lambda: root)
    return root / "audio"


# ── happy path on an awkward path ─────────────────────────────────────────────

def test_round_trip_under_spaces_and_unicode(gallery):
    record = audio_gallery.save(WAV, META)
    assert (gallery / f"{record['id']}.wav").is_file()
    assert audio_gallery.owned_audio_path(record["id"]) is not None
    assert [r["id"] for r in audio_gallery.list_audio()] == [record["id"]]


def test_ids_are_lowercase_hex_so_case_folding_filesystems_cannot_collide(gallery):
    """macOS (APFS default) and Windows fold case. Two ids differing only in
    case would be one file there, so the id alphabet must not rely on case."""
    ids = [audio_gallery.save(WAV, META)["id"] for _ in range(25)]
    assert all(i == i.lower() and len(i) == 32 for i in ids)
    assert len(set(ids)) == len(ids)


def test_unicode_and_control_characters_in_the_prompt_survive(gallery):
    meta = {**META, "prompt": "emoji free éà你好 \x07\x00 end"}
    record = audio_gallery.save(WAV, meta)
    listed = audio_gallery.list_audio()[0]
    assert listed["prompt"] == meta["prompt"]
    # The sidecar must stay valid JSON whatever the prompt contained.
    json.loads((gallery / f"{record['id']}.json").read_text(encoding = "utf-8"))


def test_a_very_long_prompt_does_not_reach_the_filename(gallery):
    """Windows caps a path at 260 characters by default. The prompt lives in
    the sidecar, so length must not matter."""
    record = audio_gallery.save(WAV, {**META, "prompt": "x" * 20000})
    assert len(f"{record['id']}.wav") == 36
    assert audio_gallery.owned_audio_path(record["id"]) is not None


# ── traversal and hostile ids ─────────────────────────────────────────────────

@pytest.mark.parametrize(
    "audio_id",
    [
        "../../etc/passwd",
        "..\\..\\windows\\system32\\config\\sam",
        "a/b",
        "a\\b",
        "",
        " ",
        "." * 3,
        "con",          # reserved on Windows, must simply not resolve
        "nul",
        "a" * 129,
        "id.with.dots",
        "id:stream",    # NTFS alternate data stream syntax
        "nul\x00byte",
    ],
)
def test_hostile_ids_never_resolve(gallery, audio_id):
    assert audio_gallery.audio_path(audio_id) is None or not audio_gallery.audio_path(audio_id).exists()
    assert audio_gallery.owned_audio_path(audio_id) is None
    assert audio_gallery.delete(audio_id) is False


def test_windows_reserved_stem_cannot_be_served_even_if_a_file_exists(gallery):
    """`con.wav` is creatable on Linux but is a device name on Windows. Serving
    is gated on an owned sidecar, so a hand-made pair must still be refused
    unless it carries the full schema."""
    gallery.mkdir(parents = True, exist_ok = True)
    (gallery / "con.wav").write_bytes(WAV)
    (gallery / "con.json").write_text(json.dumps({"prompt": "x"}), encoding = "utf-8")
    assert audio_gallery.owned_audio_path("con") is None
    assert audio_gallery.list_audio() == []


# ── partial writes and platform error codes ───────────────────────────────────

def test_a_failed_sidecar_write_leaves_no_orphan_wav(gallery, monkeypatch):
    original = Path.write_text

    def explode(self, *args, **kwargs):
        if self.suffix == ".tmp" and ".json" in self.name:
            raise OSError(errno.ENOSPC, "No space left on device")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", explode)
    with pytest.raises(OSError):
        audio_gallery.save(WAV, META)
    assert list(gallery.glob("*")) == []


def test_a_wav_whose_sidecar_never_landed_is_invisible_everywhere(gallery):
    record = audio_gallery.save(WAV, META)
    (gallery / f"{record['id']}.json").unlink()
    assert audio_gallery.list_audio() == []
    assert audio_gallery.owned_audio_path(record["id"]) is None
    assert audio_gallery.delete(record["id"]) is False
    assert audio_gallery.clear() == 0
    assert (gallery / f"{record['id']}.wav").is_file()


def test_windows_sharing_violation_on_delete_is_reported_not_raised(gallery, monkeypatch):
    """NTFS refuses to unlink a file another handle has open (WinError 32).
    A clip being streamed by FileResponse while the user hits delete hits this."""
    record = audio_gallery.save(WAV, META)
    original = Path.unlink

    def busy(self, *args, **kwargs):
        if self.suffix == ".wav":
            raise PermissionError(13, "The process cannot access the file")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", busy)
    assert audio_gallery.delete(record["id"]) is False
    # The sidecar must survive, or the clip would vanish from the list with the
    # bytes still on disk and no way to retry.
    assert (gallery / f"{record['id']}.json").is_file()
    # Only lift the unlink patch; the gallery must stay redirected at tmp_path.
    monkeypatch.setattr(Path, "unlink", original)
    assert audio_gallery.delete(record["id"]) is True


def test_clear_skips_a_locked_clip_and_still_removes_the_rest(gallery, monkeypatch):
    locked = audio_gallery.save(WAV, META)["id"]
    others = [audio_gallery.save(WAV, META)["id"] for _ in range(3)]
    original = Path.unlink

    def busy(self, *args, **kwargs):
        if self.name == f"{locked}.wav":
            raise PermissionError(13, "The process cannot access the file")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", busy)
    assert audio_gallery.clear() == len(others)
    assert (gallery / f"{locked}.wav").is_file()


def test_replace_overwrites_an_existing_target(gallery, monkeypatch):
    """os.replace is atomic-overwrite on POSIX; on Windows it is only
    equivalent because Python maps it to MoveFileEx with REPLACE_EXISTING.
    A colliding id must therefore not raise."""
    record = audio_gallery.save(WAV, META)
    monkeypatch.setattr(audio_gallery.uuid, "uuid4", lambda: _FixedUUID(record["id"]))
    second = audio_gallery.save(b"RIFFsecond" + b"\x00" * 40, META)
    assert second["id"] == record["id"]
    assert (gallery / f"{record['id']}.wav").read_bytes().startswith(b"RIFFsecond")


class _FixedUUID:
    def __init__(self, hex_value: str):
        self.hex = hex_value


# ── concurrency ───────────────────────────────────────────────────────────────

def test_concurrent_saves_do_not_interleave_into_a_broken_pair(gallery):
    errors: list[BaseException] = []
    ids: list[str] = []
    lock = threading.Lock()

    def worker(index: int):
        try:
            record = audio_gallery.save(WAV + bytes([index % 251]), {**META, "prompt": f"p{index}"})
            with lock:
                ids.append(record["id"])
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target = worker, args = (i,)) for i in range(40)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout = 60)

    assert not errors, errors[:3]
    assert len(set(ids)) == 40
    listed = audio_gallery.list_audio()
    assert len(listed) == 40
    # No staging file may survive a completed save.
    assert not list(gallery.glob(".*.tmp"))


def test_pagination_is_stable_while_clips_are_deleted(gallery):
    ids = [audio_gallery.save(WAV, META)["id"] for _ in range(10)]
    page = audio_gallery.list_audio_page(3)
    assert len(page) == 3
    cursor = page[-1][1]
    for victim in ids[:4]:
        audio_gallery.delete(victim)
    following = audio_gallery.list_audio_page(3, before = cursor)
    assert len(following) <= 3
    assert not set(r["id"] for r, _ in following) & set(r["id"] for r, _ in page)


def test_listing_survives_a_sidecar_that_is_not_json(gallery):
    record = audio_gallery.save(WAV, META)
    (gallery / f"{record['id']}.json").write_text("{not json", encoding = "utf-8")
    assert audio_gallery.list_audio() == []


def test_listing_survives_a_sidecar_that_is_json_but_not_an_object(gallery):
    record = audio_gallery.save(WAV, META)
    (gallery / f"{record['id']}.json").write_text("[1, 2, 3]", encoding = "utf-8")
    assert audio_gallery.list_audio() == []


def test_a_missing_gallery_directory_lists_empty_rather_than_raising(gallery, monkeypatch):
    monkeypatch.setattr(audio_gallery, "gallery_dir", lambda: gallery / "nope")
    assert audio_gallery.list_audio() == []
    assert audio_gallery.clear() == 0
