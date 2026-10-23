"""Per-install identity for a standalone copy of these scripts (ui_developer/scripts).

`identity.json` beside this file names the GitHub login the tools act as, the staging owners
they may use and the repo PR media is pushed to. The repo ships no identity.json, so on the
shared hosts every helper returns the caller's default and nothing changes.
"""

import json
import os
from pathlib import Path

_PATH = Path(__file__).with_name("identity.json")


def load() -> dict:
    try:
        data = json.loads(_PATH.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def login(default: str) -> str:
    return load().get("login") or default


def staging_owners():
    env = [o.strip() for o in os.environ.get("STAGING_OWNERS", "").split(",") if o.strip()]
    if env:
        return env
    owners = load().get("staging_owners")
    return [str(o) for o in owners] if isinstance(owners, list) and owners else None


def media_repo(default: str) -> str:
    return load().get("media_repo") or default


def shared_dir(name: str, default_path) -> Path:
    """`default_path` where its parent exists (the shared hosts), else a per-user cache dir."""
    default_path = Path(default_path)
    if default_path.parent.is_dir():
        return default_path
    return Path.home() / ".cache" / "unsloth_scripts" / name
