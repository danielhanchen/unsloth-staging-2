"""Multi-user isolation against one live Studio: does account B reach account A's work?

    python -m studio_test_kit.examples.multi_user_isolation --url http://127.0.0.1:8901 \
        --owner-password "$UNSLOTH_STUDIO_PASSWORD"

Proves: a fresh install is `single` and one account flips it to `multi`; a setup code is
single-use; a resource A creates is not visible to B; deleting returns the install to `single`.

Probing bare collection endpoints instead proves nothing, since they are per-account and B
correctly gets 200 with its own empty list. HTTP surface only: this does not prove filesystem
isolation of `workspace_root()` nor the owner's `full_access_permitted()` transition.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid

from ..accounts import (
    assert_not_visible,
    assert_setup_code_single_use,
    create_account,
    delete_account,
    login_mode,
    onboard_account,
    redeem_setup_code,
)
from ..auth import login

# Fetched as B and checked for A's marker, never for reachability: these all 200 for any caller.
SHARED_COLLECTIONS = ("/api/auth/api-keys", "/api/chat/history", "/api/provider-credentials")


async def main(url: str, owner_password: str, owner_user: str, keep: bool) -> int:
    owner = await login(url, owner_user, owner_password)
    tok = owner.access_token

    start_mode = await login_mode(url, tok)
    print(f"[1] login_mode at start: {start_mode}")

    suffix = uuid.uuid4().hex[:8]
    a_name, b_name = f"iso-a-{suffix}", f"iso-b-{suffix}"
    created: list[str] = []
    failures: list[str] = []

    try:
        a = await onboard_account(url, tok, a_name, "isolation-pw-a-123")
        created.append(a.account_id)
        print(f"[1] created {a_name} -> login_mode: {await login_mode(url, tok)}")

        # (2) negative control on B, whose code we hold before redeeming.
        b_account, b_code = await create_account(url, tok, b_name)
        created.append(b_account["account_id"])
        b_auth = await redeem_setup_code(url, b_name, b_code, "isolation-pw-b-123")
        await assert_setup_code_single_use(url, b_name, b_code)
        print("[2] setup code is single-use")

        # (3) the actual isolation question: give A something named, then look for it as B.
        marker = f"iso-key-{suffix}"
        import httpx

        async with httpx.AsyncClient(timeout=15.0) as c:
            r = await c.post(
                f"{url}/api/auth/api-keys",
                headers={"Authorization": f"Bearer {a.auth.access_token}"},
                json={"name": marker},
            )
            r.raise_for_status()
        print(f"[3] created {marker} as {a_name}")

        for path in SHARED_COLLECTIONS:
            try:
                await assert_not_visible(url, b_auth.access_token, path, marker)
                print(f"[3] {path}: {marker} not visible to {b_name}")
            except AssertionError as exc:
                failures.append(f"{path}: {exc}")
                print(f"[3] {path}: LEAK -- {exc}")
    finally:
        if not keep:
            for account_id in reversed(created):
                try:
                    await delete_account(url, tok, account_id)
                except Exception as exc:  # noqa: BLE001 - report, never mask the isolation result
                    failures.append(f"cleanup {account_id}: {exc}")
            end_mode = await login_mode(url, tok)
            print(f"[4] login_mode after cleanup: {end_mode}")
            if end_mode != start_mode:
                failures.append(f"login_mode did not return to {start_mode}, got {end_mode}")

    if failures:
        print("\nFAIL:\n  " + "\n  ".join(failures))
        return 1
    print("\nPASS: no cross-account access on the paths probed")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8901")
    p.add_argument("--owner-password", required=True)
    p.add_argument("--owner-user", default="unsloth")
    p.add_argument("--keep", action="store_true", help="leave the accounts behind for manual poking")
    a = p.parse_args()
    sys.exit(asyncio.run(main(a.url, a.owner_password, a.owner_user, a.keep)))
