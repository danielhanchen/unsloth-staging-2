"""H7-H10: what an install that predates this PR sees after upgrading.

Covers the three shapes that break on upgrade in practice: a persisted settings
record written by the old backend, an old frontend bundle talking to the new
backend, and request-size policy shifting under paths that already existed.
"""

from __future__ import annotations

import os
import sys
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


# ── H8: personalization written before /audio existed ─────────────────────────

LEGACY_NAV_IDS = ["hub", "projects", "images", "video", "train", "recipes", "export", "api"]


def test_a_settings_record_saved_before_audio_existed_still_loads():
    """The stored sidebarNav has no audio entry. The validator must backfill it
    rather than reject the record, or every upgraded install loses its saved
    personalization on first read."""
    from routes.settings import PersonalizationCustomization

    stored = {"sidebarNav": [{"id": i, "pinned": True} for i in LEGACY_NAV_IDS]}
    parsed = PersonalizationCustomization(**stored)
    ids = [item.id for item in parsed.sidebarNav]
    assert "audio" in ids
    # The user's order is theirs; audio is appended, never inserted.
    assert ids[: len(LEGACY_NAV_IDS)] == LEGACY_NAV_IDS
    assert next(i for i in parsed.sidebarNav if i.id == "audio").pinned is False


def test_an_old_frontend_put_without_audio_is_accepted():
    """A cached SPA keeps PUTing the old list. That must not 422 the whole
    personalization save."""
    from routes.settings import PersonalizationPayload

    payload = PersonalizationPayload(
        appearance = {
            "customization": {"sidebarNav": [{"id": i, "pinned": True} for i in LEGACY_NAV_IDS]}
        }
    )
    ids = [i.id for i in payload.appearance.customization.sidebarNav]
    assert set(LEGACY_NAV_IDS) | {"audio"} == set(ids)


def test_a_stored_record_that_already_pinned_audio_keeps_the_users_choice():
    from routes.settings import PersonalizationCustomization

    stored = {"sidebarNav": [{"id": "audio", "pinned": True}, {"id": "hub", "pinned": False}]}
    parsed = PersonalizationCustomization(**stored)
    by_id = {i.id: i for i in parsed.sidebarNav}
    assert by_id["audio"].pinned is True
    assert by_id["hub"].pinned is False
    assert parsed.sidebarNav[0].id == "audio"


def test_duplicate_audio_entries_collapse_to_the_first():
    from routes.settings import PersonalizationCustomization

    stored = {
        "sidebarNav": [
            {"id": "audio", "pinned": True},
            {"id": "audio", "pinned": False},
        ]
    }
    parsed = PersonalizationCustomization(**stored)
    assert len([i for i in parsed.sidebarNav if i.id == "audio"]) == 1
    assert parsed.sidebarNav[0].pinned is True


def test_an_unknown_future_nav_id_is_rejected_not_silently_kept():
    """Documents the forward-compat direction: a NEWER frontend sending an id
    this backend does not know still 422s. Same as when video was added."""
    import pydantic
    from routes.settings import PersonalizationSidebarNavItem

    with pytest.raises(pydantic.ValidationError):
        PersonalizationSidebarNavItem(id = "holograms", pinned = True)


def test_the_shipped_default_pins_nothing_new_for_existing_users():
    """audio defaults to unpinned, so upgrading does not rearrange a sidebar
    the user already arranged."""
    from routes.settings import SIDEBAR_NAV_ITEM_DEFAULTS

    assert SIDEBAR_NAV_ITEM_DEFAULTS["audio"] is False


# ── H10: request body caps ────────────────────────────────────────────────────

def test_the_new_transcription_paths_get_the_stt_cap_on_both_mounts():
    import main
    from utils.upload_limits import STT_AUDIO_RAW_MAX_BYTES

    expected = main.upload_request_limit_bytes(STT_AUDIO_RAW_MAX_BYTES)
    for path in ("/v1/audio/transcriptions", "/api/inference/audio/transcriptions"):
        assert main._get_request_body_max_bytes(path) == expected
        assert main._get_request_body_max_bytes(path + "/") == expected


def test_no_neighbouring_path_inherits_the_bigger_cap():
    """The cap is matched on the exact path. A sibling route that merely starts
    the same way must keep the default, or this becomes an upload hole."""
    import main

    default = main.default_request_body_limit_bytes()
    for path in (
        "/v1/audio/transcriptions/extra",
        "/v1/audio/transcriptionsX",
        "/v1/audio/speech",
        "/v1/chat/completions",
    ):
        assert main._get_request_body_max_bytes(path) != main.upload_request_limit_bytes(
            __import__("utils.upload_limits", fromlist = ["x"]).STT_AUDIO_RAW_MAX_BYTES
        ) or path.startswith("/api/inference/audio/transcribe")
    assert main._get_request_body_max_bytes("/v1/chat/completions") == default


def test_the_existing_stt_paths_keep_the_caps_they_had():
    """The routes that existed before this PR must not move."""
    import main
    from utils.upload_limits import STT_AUDIO_RAW_MAX_BYTES

    assert main._get_request_body_max_bytes("/api/inference/audio/transcribe/raw") == (
        STT_AUDIO_RAW_MAX_BYTES
    )


# ── H9: curated STT rows must never become chat-loadable ──────────────────────

def test_curated_stt_repo_ids_are_matched_case_insensitively():
    from utils.hidden_models import is_curated_stt_repo_id

    assert is_curated_stt_repo_id("unslothai/Qwen3-ASR-1.7B-GGUF")
    assert is_curated_stt_repo_id("UNSLOTHAI/qwen3-asr-1.7b-gguf")
    assert is_curated_stt_repo_id("  unslothai/Qwen3-ASR-1.7B-GGUF  ")
    assert not is_curated_stt_repo_id("unslothai/Qwen3-ASR-1.7B-GGUF-clone")
    assert not is_curated_stt_repo_id(None)
    assert not is_curated_stt_repo_id("")


def test_curated_stt_rows_are_not_chat_capable():
    """They are emitted for the Audio page now, so the only thing keeping them
    out of chat is this flag. An old frontend has no audio allowlist at all."""
    from hub.services.models.cache_inventory import _cache_inventory_fields

    fields = _cache_inventory_fields(
        "unslothai/Qwen3-ASR-1.7B-GGUF", "gguf", identity = None, partial = False
    )
    assert fields["capabilities"]["can_chat"] is False
    assert fields["capabilities"]["supports_vision"] is False


def test_a_normal_repo_is_untouched_by_the_stt_gate():
    from hub.services.models.cache_inventory import _cache_inventory_fields

    fields = _cache_inventory_fields(
        "unsloth/Qwen3-4B-GGUF", "gguf", identity = None, partial = False
    )
    assert fields["capabilities"]["can_chat"] is True


# ── H7: the TTS token cap is a real behaviour change ──────────────────────────

@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        (None, 2048),      # unchanged default
        (512, 512),        # unchanged
        (8192, 8192),      # at the cap
        (100000, 8192),    # CLAMPED: main would have passed this straight through
        (0, 2048),         # falsy -> default, as before
        (-5, 1),           # floored
    ],
)
def test_tts_token_bound(requested, expected):
    import routes.inference as inf
    from models.inference import ChatCompletionRequest

    payload = ChatCompletionRequest(messages = [{"role": "user", "content": "hi"}])
    if requested is not None:
        payload.max_tokens = requested
    assert inf._tts_max_new_tokens(payload) == expected
