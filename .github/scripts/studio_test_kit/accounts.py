"""Multi-user accounts: owner CRUD, setup-code onboarding, isolation probes.

`login_mode` is derived, never configured: `multi` iff >1 ACTIVE account (`auth/policy.py`).
Two consequences worth testing: ONE managed account sets `full_access_permitted()` false
install-wide including the owner, and a DEACTIVATED account still counts.

Onboarding is two hops, not one: create returns a setup code, the account logs in with it and
gets `must_change_password`, then `change-password` mints the real token. `onboard_account` does
both.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import httpx

from .auth import StudioAuth

OWNER_ACCOUNT_ID = "owner"


@dataclass
class ManagedAccount:
    """An onboarded managed account and its usable token."""

    account_id: str
    username: str
    password: str
    auth: StudioAuth


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def list_accounts(base_url: str, owner_token: str, timeout: float = 15.0) -> list[dict[str, Any]]:
    """GET /api/accounts -> [{account_id, username, role, is_active, created_at, setup_code_pending}]."""
    async with httpx.AsyncClient(timeout=timeout) as c:
        r = await c.get(f"{base_url}/api/accounts", headers=_bearer(owner_token))
        r.raise_for_status()
        return r.json()["accounts"]


async def create_account(
    base_url: str, owner_token: str, username: str, timeout: float = 15.0
) -> tuple[dict[str, Any], str]:
    """POST /api/accounts -> (account, setup_code). 409 = name taken, 400 = invalid name."""
    async with httpx.AsyncClient(timeout=timeout) as c:
        r = await c.post(
            f"{base_url}/api/accounts", headers=_bearer(owner_token), json={"username": username}
        )
        r.raise_for_status()
        b = r.json()
        return b["account"], b["setup_code"]


async def reissue_setup_code(
    base_url: str, owner_token: str, account_id: str, timeout: float = 15.0
) -> str:
    """POST /api/accounts/{id}/setup-code. Invalidates the previous code."""
    async with httpx.AsyncClient(timeout=timeout) as c:
        r = await c.post(
            f"{base_url}/api/accounts/{account_id}/setup-code", headers=_bearer(owner_token)
        )
        r.raise_for_status()
        return r.json()["setup_code"]


async def set_account_active(
    base_url: str, owner_token: str, account_id: str, is_active: bool, timeout: float = 15.0
) -> dict[str, Any]:
    """PATCH /api/accounts/{id}. A deactivated account still counts for `full_access_permitted()`."""
    async with httpx.AsyncClient(timeout=timeout) as c:
        r = await c.patch(
            f"{base_url}/api/accounts/{account_id}",
            headers=_bearer(owner_token),
            json={"is_active": is_active},
        )
        r.raise_for_status()
        return r.json()


async def delete_account(
    base_url: str, owner_token: str, account_id: str, timeout: float = 60.0
) -> None:
    """DELETE /api/accounts/{id}. Roots are renamed to `<root>-deleted-<stamp>`, never erased,
    all-or-nothing. A 400 is usually media generation still in flight, and is retryable."""
    async with httpx.AsyncClient(timeout=timeout) as c:
        r = await c.delete(f"{base_url}/api/accounts/{account_id}", headers=_bearer(owner_token))
        r.raise_for_status()


async def redeem_setup_code(
    base_url: str, username: str, setup_code: str, new_password: str, timeout: float = 15.0
) -> StudioAuth:
    """Log in with the setup code, then change the password. The first token is only good
    for the change call."""
    async with httpx.AsyncClient(timeout=timeout) as c:
        r = await c.post(
            f"{base_url}/api/auth/login", json={"username": username, "password": setup_code}
        )
        r.raise_for_status()
        first = r.json()
        if not first.get("must_change_password"):
            raise AssertionError(
                "setup-code login did not demand a password change; the code was already "
                "redeemed, or it was accepted as a normal password"
            )
        r = await c.post(
            f"{base_url}/api/auth/change-password",
            headers=_bearer(first["access_token"]),
            json={"current_password": setup_code, "new_password": new_password},
        )
        r.raise_for_status()
        b = r.json()
        return StudioAuth(
            access_token=b["access_token"],
            refresh_token=b.get("refresh_token", ""),
            base_url=base_url,
        )


async def onboard_account(
    base_url: str, owner_token: str, username: str, password: str, timeout: float = 15.0
) -> ManagedAccount:
    """create_account + redeem_setup_code. The one call most tests want."""
    account, code = await create_account(base_url, owner_token, username, timeout=timeout)
    auth = await redeem_setup_code(base_url, username, code, password, timeout=timeout)
    return ManagedAccount(
        account_id=account["account_id"], username=username, password=password, auth=auth
    )


async def assert_setup_code_single_use(
    base_url: str, username: str, setup_code: str, timeout: float = 15.0
) -> None:
    """Negative control: without it the happy path passes against a backend that never
    clears `setup_code_hash`."""
    async with httpx.AsyncClient(timeout=timeout) as c:
        r = await c.post(
            f"{base_url}/api/auth/login", json={"username": username, "password": setup_code}
        )
    if r.status_code < 400:
        raise AssertionError(f"reused setup code was accepted ({r.status_code}); it is not single-use")


async def assert_cannot_reach(
    base_url: str, token: str, path: str, timeout: float = 15.0, allow: tuple[int, ...] = (403, 404)
) -> int:
    """Assert `token` is refused at a path naming ONE resource A owns. 403 and 404 both pass.

    NOT for collections: those are per-account, so B gets 200 with its own empty list and this
    reports a leak that is not there (live: `/api/auth/api-keys` -> `{"api_keys":[]}`). Use
    `assert_not_visible`.
    """
    async with httpx.AsyncClient(timeout=timeout) as c:
        r = await c.get(f"{base_url}{path}", headers=_bearer(token))
    if r.status_code < 400:
        raise AssertionError(
            f"{path} returned {r.status_code} for a token that must not reach it: {r.text[:200]}"
        )
    if r.status_code not in allow:
        raise AssertionError(f"{path} returned {r.status_code}, expected one of {allow}")
    return r.status_code


async def assert_not_visible(
    base_url: str, token: str, path: str, needle: str, timeout: float = 15.0
) -> None:
    """Assert A's `needle` is absent from collection `path` fetched as B. 200 is expected there;
    A's item appearing in it is not. `needle` must be unique to A or this cannot fail."""
    if not needle:
        raise ValueError("needle is empty, so this assertion could never fail")
    async with httpx.AsyncClient(timeout=timeout) as c:
        r = await c.get(f"{base_url}{path}", headers=_bearer(token))
    if r.status_code >= 400:
        return
    if needle in r.text:
        raise AssertionError(f"{path} exposed another account's {needle!r} to this token")


async def login_mode(base_url: str, token: Optional[str] = None, timeout: float = 15.0) -> str:
    """GET /api/auth/status. Derived from active account count, so assert rather than assume."""
    headers = _bearer(token) if token else {}
    async with httpx.AsyncClient(timeout=timeout) as c:
        r = await c.get(f"{base_url}/api/auth/status", headers=headers)
        r.raise_for_status()
        body = r.json()
    return body.get("login_mode") or ("multi" if body.get("multi_user") else "single")
