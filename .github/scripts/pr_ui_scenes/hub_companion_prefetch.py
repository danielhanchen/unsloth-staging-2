"""Scene: how the Hub shows a base repo that holds only what a diffusion GGUF borrowed, for PR 11644.

A GGUF image load fetches its base repo's text encoder, VAE and processor and skips the denoiser.
On main that repo reads as a Partial download with a Continue button sized for the whole repo; on
Qwen-Image-2.1 Continue pulls the 14 GB transformer the GGUF never uses. The PR keeps the row
unloadable but stops calling it an unfinished download.

With ``seed_gguf`` set, the first side seeds the cache the way a user does: it empties it, picks the GGUF
on the Images page and lets Studio fetch the GGUF and the base repo's companions itself. The second side
finds that on disk and only looks, so both photograph the same files. A cache seeded by ``hf download``
is not the same thing: Studio's own scoped download leaves state the Hub reads.

The API half is recorded too, from BOTH listings the Hub merges (``/api/hub/cached-models`` and
``/api/hub/local``), because a flag carried by one and not the other is invisible in a screenshot.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import sys
from pathlib import Path

WORKSPACE = Path(os.environ.get("WORKSPACE") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "scripts"))

from pr_ui_scenes._common import Session, api_get  # noqa: E402
from studio_test_kit.auth import seed_init_script  # noqa: E402
from studio_test_kit.ui import open_chat  # noqa: E402

DEFAULT_REPO = "Qwen/Qwen-Image-2.1"


def _rows_for(payload, repo: str) -> list[dict]:
    rows = (payload.get("cached") or payload.get("models")) if isinstance(payload, dict) else payload
    keep = ("repo_id", "id", "source", "model_format", "partial", "companion_prefetch", "companion",
            "partial_transport", "size_bytes")
    out = []
    for row in rows or []:
        rid = (row.get("repo_id") or row.get("model_id") or row.get("id") or "")
        if rid.lower() == repo.lower():
            out.append({k: row.get(k) for k in keep if k in row})
    return out


def _seeded(cache_hub: Path, gguf_repo: str, quant: str, repo: str) -> bool:
    gdir = cache_hub / ("models--" + gguf_repo.replace("/", "--")) / "snapshots"
    bdir = cache_hub / ("models--" + repo.replace("/", "--")) / "snapshots"
    has_gguf = any(p.name.endswith(".gguf") and quant.lower() in p.name.lower() for p in gdir.rglob("*"))
    # The VAE, not the text encoder: a family with a hosted pre-cast encoder fetches it from another repo.
    has_vae = any(p.suffix == ".safetensors" for p in bdir.glob("*/vae/*"))
    return has_gguf and has_vae


async def _seed(page, base_url: str, gguf_repo: str, quant: str, facts: dict, timeout_s: float) -> None:
    await page.goto(f"{base_url}/images", wait_until = "domcontentloaded")
    await page.wait_for_timeout(6_000)
    await page.get_by_role("button", name = "Select image model").first.click()
    await page.wait_for_timeout(3_000)
    await page.get_by_text(re.compile("^" + re.escape(gguf_repo.split("/")[-1]) + r"(\s|$)")).first.click()
    await page.wait_for_timeout(3_000)
    row = page.get_by_text(quant, exact = True).first
    await row.wait_for(timeout = 30_000)
    await row.click()
    t0 = asyncio.get_event_loop().time()
    seen: list[str] = []
    # Done when the load starts: the staged downloads (GGUF, then the base repo's companions) have all finished.
    while asyncio.get_event_loop().time() - t0 < timeout_s:
        texts = [t.strip() for t in await page.locator("[data-sonner-toast]").all_inner_texts() if t.strip()]
        joined = " || ".join(texts)
        if joined and (not seen or seen[-1] != joined):
            seen.append(joined[:200])
        if any(k in joined for k in ("Loading to GPU", "Model loaded", "Starting model")):
            break
        await page.wait_for_timeout(1_000)
    facts["seed_seconds"] = round(asyncio.get_event_loop().time() - t0, 1)
    facts["seed_toasts"] = seen[-6:]


async def drive(session: Session, out_dir: Path, label: str, repo: str = DEFAULT_REPO,
                seed_gguf: str = "", quant: str = "Q4_K_M", cache_hub: str = "",
                seed_timeout_s: float = 1500, **_: object) -> tuple[list[Path], dict]:
    facts: dict = {}
    if seed_gguf and cache_hub and not _seeded(Path(cache_hub), seed_gguf, quant, repo):
        for r in (seed_gguf, repo):
            shutil.rmtree(Path(cache_hub) / ("models--" + r.replace("/", "--")), ignore_errors = True)
        init0 = seed_init_script(
            type("A", (), {"access_token": session.access_token,
                           "refresh_token": session.refresh_token})(), []
        )
        async with open_chat(session.base_url, init_scripts = [init0],
                             viewport = (1600, 1000), headless = True) as sp0:
            await _seed(sp0.page, session.base_url, seed_gguf, quant, facts, seed_timeout_s)
        facts["seeded_here"] = True
    else:
        facts["seeded_here"] = False
    for name, path in (("cached_models", "/api/hub/cached-models"), ("local", "/api/hub/local")):
        try:
            facts[name] = _rows_for(api_get(session, path), repo)
        except Exception as exc:  # noqa: BLE001 -- a failed listing is a fact too
            facts[name] = f"{type(exc).__name__}: {exc}"

    shots: list[Path] = []
    init = seed_init_script(
        type("A", (), {"access_token": session.access_token,
                       "refresh_token": session.refresh_token})(), []
    )
    short = repo.split("/")[-1]
    async with open_chat(session.base_url, init_scripts = [init],
                         viewport = (1500, 1000), headless = True) as sp:
        page = sp.page
        await page.goto(f"{session.base_url}/hub", wait_until = "domcontentloaded")
        on_device = page.get_by_role("radio", name = "On Device").first
        await on_device.wait_for(state = "visible", timeout = 60_000)
        await on_device.click()
        search = page.get_by_placeholder("Search on-device models").first
        await search.wait_for(state = "visible", timeout = 60_000)
        await search.fill(short)
        menu = page.locator(f'button[aria-label="More options for {repo}"]').first
        await menu.wait_for(state = "visible", timeout = 180_000)
        await page.wait_for_timeout(3_000)

        # The row is covered by a full-size overlay button labelled with the model name; it carries the
        # Partial / On device tooltip and opens the detail pane.
        name = page.get_by_role("button", name = short, exact = True).first
        await name.hover()
        await page.wait_for_timeout(1_500)
        tips = await page.locator('[role="tooltip"]').all_inner_texts()
        facts["row_tooltip"] = [" ".join(t.split()) for t in tips if t.strip()]
        list_shot = out_dir / f"{label.lower()}_0_on_device_row.png"
        await sp.screenshot(list_shot, full_page = False)
        shots.append(list_shot)

        await name.click()
        await page.wait_for_timeout(4_000)
        body = " ".join((await page.locator("main").first.inner_text()).split())
        facts["detail_has_partial_badge"] = "Partial" in body
        facts["detail_has_on_device_badge"] = "On device" in body
        facts["detail_buttons"] = sorted({
            " ".join(t.split())
            for t in await page.get_by_role("button").all_inner_texts()
            if t.strip() and any(w in t for w in ("Continue", "Download", "Resume", "Redownload", "Run"))
        })
        detail_shot = out_dir / f"{label.lower()}_1_detail.png"
        await sp.screenshot(detail_shot, full_page = False)
        shots.append(detail_shot)
    return shots, facts
