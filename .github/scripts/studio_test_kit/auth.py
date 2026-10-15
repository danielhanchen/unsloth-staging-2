"""Backend JWT login + Playwright localStorage seeding.

Studio's frontend reads three localStorage keys for external providers:

  unsloth_chat_external_providers       JSON array of provider configs
  unsloth_chat_external_provider_keys   { providerId: plaintextKey }
  unsloth_chat_connections_enabled      "true" | "false"

The SPA RSA-encrypts the plaintext key on each request and posts the
ciphertext to /v1/chat/completions, so seeding these three keys is enough
to drive any external provider end-to-end without manual UI clicks
through the Settings -> Connections flow.

`seed_init_script(...)` returns a JS init-script body suitable for
`browser_context.add_init_script(...)`. Use it BEFORE the first page
load so the SPA sees the values on its first read.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx


@dataclass
class StudioAuth:
    """Result of a successful backend login."""

    access_token: str
    refresh_token: str
    base_url: str


@dataclass
class ProviderSeed:
    """A single external provider entry for localStorage.

    `provider_type` matches the backend registry id ("gemini", "openai",
    "anthropic", "openrouter", "kimi", "openai-compat", ...).
    `models` is the list of model ids Studio's picker should show.
    `api_key` is the PLAINTEXT key; the SPA will RSA-encrypt at request
    time using the server's published public key.
    """

    provider_type: str
    name: str
    base_url: str
    models: list[str]
    api_key: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])

    def as_provider_entry(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "providerType": self.provider_type,
            "name": self.name,
            "baseUrl": self.base_url,
            "models": list(self.models),
        }


async def login(base_url: str, username: str, password: str, timeout: float = 15.0) -> StudioAuth:
    """POST /api/auth/login -> {access_token, refresh_token}."""
    async with httpx.AsyncClient(timeout=timeout) as c:
        r = await c.post(
            f"{base_url}/api/auth/login",
            json={"username": username, "password": password},
        )
        r.raise_for_status()
        b = r.json()
        return StudioAuth(
            access_token=b["access_token"],
            refresh_token=b.get("refresh_token", ""),
            base_url=base_url,
        )


def seed_init_script(
    auth: StudioAuth,
    providers: list[ProviderSeed],
    connections_enabled: bool = True,
    extra_local_storage: Optional[dict[str, Any]] = None,
) -> str:
    """Build a `addInitScript` JS body that primes localStorage before SPA boot."""
    provider_entries = [p.as_provider_entry() for p in providers]
    provider_keys = {p.id: p.api_key for p in providers if p.api_key}

    payload = {
        "unsloth_auth_token": auth.access_token,
        "unsloth_refresh_token": auth.refresh_token,
        "unsloth_chat_external_providers": json.dumps(provider_entries),
        "unsloth_chat_external_provider_keys": json.dumps(provider_keys),
        "unsloth_chat_connections_enabled": "true" if connections_enabled else "false",
    }
    if extra_local_storage:
        for k, v in extra_local_storage.items():
            payload[k] = v if isinstance(v, str) else json.dumps(v)

    # JSON.stringify the dict, then iterate at page boot time.
    js_payload = json.dumps(payload)
    return f"""
    (() => {{
        const seed = {js_payload};
        for (const k of Object.keys(seed)) {{
            try {{ window.localStorage.setItem(k, seed[k]); }} catch (e) {{}}
        }}
    }})();
    """


# ── Convenience providers ───────────────────────────────────────────

def gemini_provider(api_key: str, models: Optional[list[str]] = None) -> ProviderSeed:
    return ProviderSeed(
        provider_type="gemini",
        name="Google Gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        models=models or ["gemini-2.5-flash", "gemini-2.5-flash-image"],
        api_key=api_key,
    )


def openai_provider(api_key: str, models: Optional[list[str]] = None) -> ProviderSeed:
    return ProviderSeed(
        provider_type="openai",
        name="OpenAI",
        base_url="https://api.openai.com/v1",
        models=models or ["gpt-4o-mini"],
        api_key=api_key,
    )


def anthropic_provider(api_key: str, models: Optional[list[str]] = None) -> ProviderSeed:
    return ProviderSeed(
        provider_type="anthropic",
        name="Anthropic",
        base_url="https://api.anthropic.com",
        models=models or ["claude-3-5-haiku-latest"],
        api_key=api_key,
    )
