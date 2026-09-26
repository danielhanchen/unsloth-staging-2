"""Agent Skills: `/api/skills` list + toggle, and composer `@`-mentions.

`core/inference/skills.py` resolves managed `workspace_root()/skills`, `~/.agents/skills`, then
bundled, IN ORDER, so an earlier root shadows a later one and the managed root is account-scoped.
That makes skills part of a multi-user isolation pass, not just a feature test.
"""

from __future__ import annotations

from typing import Any

import httpx


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def list_skills(base_url: str, token: str, timeout: float = 15.0) -> list[dict[str, Any]]:
    """GET /api/skills. Includes bundled skills, so an empty list is a finding."""
    async with httpx.AsyncClient(timeout=timeout) as c:
        r = await c.get(f"{base_url}/api/skills", headers=_bearer(token))
        r.raise_for_status()
        return r.json()


async def set_skill_enabled(
    base_url: str, token: str, name: str, enabled: bool, timeout: float = 15.0
) -> dict[str, Any]:
    """PUT /api/skills/{name}/enabled -> SkillRecord."""
    async with httpx.AsyncClient(timeout=timeout) as c:
        r = await c.put(
            f"{base_url}/api/skills/{name}/enabled",
            headers=_bearer(token),
            json={"enabled": enabled},
        )
        r.raise_for_status()
        return r.json()


async def skill_names(base_url: str, token: str, timeout: float = 15.0) -> set[str]:
    """Names only, for base-vs-head diffing: this set must never SHRINK."""
    return {s["name"] for s in await list_skills(base_url, token, timeout=timeout)}


async def mention_skill(sp, name: str, prompt: str) -> None:
    """`@name` mention then prompt. Composer-scoped: a bare text selector hits the sidebar."""
    import re

    composer = sp.page.locator("form:has(textarea) textarea")
    await composer.click()
    await composer.type(f"@{name}")
    option = sp.page.get_by_role("option", name=re.compile(rf"^{re.escape(name)}$"))
    await option.wait_for(state="visible", timeout=10_000)
    await option.click()
    await composer.type(f" {prompt}")
    await composer.press("Enter")
