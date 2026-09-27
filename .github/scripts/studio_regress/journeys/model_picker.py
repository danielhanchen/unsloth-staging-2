"""Journey: the chat model picker. Open it, then expand one GGUF repo's quant list and record
which quant carries the "recommended" star and the order the rows come in.

The quant listing (/api/models/gguf-variants) is served canned: Studio runs offline here, and a
live Hub listing would move under us between the two sides. Only the FRONTEND's reading of the
listing is under test (ordering, fit verdicts, the recommendation, e.g. unsloth#11606). The device
the picker sizes against (/api/system) is canned too, one 24 GB GPU and 64 GB of RAM, so the fit
verdicts are the same on every host and runner (a B200 box, a CPU-only staging VM)."""

from __future__ import annotations

from .. import fixture, timeouts
from ..contract import Journey, Step, StepFailed
from . import _b_ui as ui

REPO = "unsloth/Qwen3.5-0.8B-GGUF"      # in the bundled Recommended catalog, so listed offline
LEAF = REPO.split("/", 1)[1]
QUANTS = (("BF16", 1_600_000_000), ("Q8_0", 812_000_000), ("UD-Q4_K_XL", 530_000_000),
          ("Q4_K_M", 504_000_000), ("Q2_K", 360_000_000))
DEFAULT = "UD-Q4_K_XL"


def canned_listing():
    return {"repo_id": REPO, "has_vision": False, "default_variant": DEFAULT, "context_length": None,
            "resolved_locally": False, "dependencies_resolved": False, "loadable_variants": None,
            "loadable": None,
            "variants": [{"filename": f"{LEAF.removesuffix('-GGUF')}-{q}.gguf", "quant": q,
                          "display_label": None, "size_bytes": s, "download_size_bytes": s,
                          "pending_drafter_filename": None, "pending_drafter_size_bytes": 0,
                          "downloaded": False, "update_available": False, "partial": False,
                          "cleanable": False} for q, s in QUANTS]}


DEVICE_GPU_GB, DEVICE_RAM_GB = 24.0, 64.0


async def _serve_system(route):
    """The real /api/system with its memory and GPU inventory replaced by a fixed device."""
    try:
        r = await route.fetch()
        d = await r.json()
    except Exception:
        await route.continue_()
        return
    d["memory"] = {**(d.get("memory") or {}), "total_gb": DEVICE_RAM_GB, "available_gb": DEVICE_RAM_GB - 8,
                   "percent_used": 12.5}
    d["gpu"] = {**(d.get("gpu") or {}), "available": True, "backend_cuda_visible_devices": "0",
                "parent_visible_gpu_ids": [0],
                "devices": [{"index": 0, "index_kind": "physical", "visible_ordinal": 0,
                             "name": "Canned GPU", "memory_total_gb": DEVICE_GPU_GB,
                             "vram_used_gb": 1.0, "vram_free_gb": DEVICE_GPU_GB - 1.0,
                             "vram_utilization_pct": 4.0}]}
    await route.fulfill(response=r, json=d)


async def _serve_canned(route):
    if LEAF in route.request.url:
        await route.fulfill(json=canned_listing())
    else:
        await route.continue_()


async def open_picker(ctx):
    p = ctx.page
    await p.route("https://huggingface.co/api/models*", fixture.replay_hub)
    await p.route("**/api/system", _serve_system)
    await p.goto(ctx.base_url + "/chat", wait_until="domcontentloaded")
    await ui.need(p.get_by_role("textbox", name="Message input"), "chat composer", 60_000)
    # Opening the picker is what fetches /api/system, and the picker sizes quants only once it
    # has answered; the first answer takes ~10 s on a many-GPU host (cold probe). Without this
    # wait one side shows the no-budget recommendation and the other the sized one.
    async with p.expect_response(lambda r: r.url.split("?")[0].endswith("/api/system"),
                                 timeout=timeouts.scaled_ms(120_000)):
        await (await ui.need(p.get_by_role("button", name="Select model"), "Select model")).click()
    await ui.need(p.get_by_text(LEAF, exact=True), f"{LEAF} in the picker", 15_000)
    return {"recommended_repo_listed": True}


QUANT_ROWS_JS = """() => {
  const head = [...document.querySelectorAll('*')].find(e => /^quantizations$/i.test((e.innerText || '').trim()));
  if (!head || !head.parentElement) return null;
  return [...head.parentElement.children].filter(c => c !== head).map(c => {
    const first = (c.innerText || '').split('\\n')[0].trim();
    const rec = /recommended/i.test(first);
    return {quant: first.replace(/recommended/i, '').trim(), recommended: rec};
  }).filter(r => r.quant);
}"""


async def _stable_rows(p, quiet_ms=400, max_ms=5_000):
    """Was a blind 600 ms after the heading: read the rows until two reads `quiet_ms` apart agree
    (the list renders its rows, then their star, in stages), bounded by `max_ms`."""
    last, waited = None, 0
    while True:
        rows = await p.evaluate(QUANT_ROWS_JS)
        if rows and rows == last:
            return rows
        if waited >= max_ms:
            return rows
        last = rows
        await p.wait_for_timeout(quiet_ms)
        waited += quiet_ms


async def gguf_quants(ctx):
    p = ctx.page
    await p.route("**/api/models/gguf-variants*", _serve_canned)
    await p.route("**/api/hub/gguf-variants*", _serve_canned)
    try:
        await p.get_by_text(LEAF, exact=True).first.click()
        await ui.need(p.get_by_text("Quantizations", exact=False), "quant list", 15_000)
        rows = await _stable_rows(p)
    finally:
        await p.unroute("**/api/models/gguf-variants*")
        await p.unroute("**/api/hub/gguf-variants*")
    if not rows:
        raise StepFailed("quant rows not found under Quantizations")
    starred = [r["quant"] for r in rows if r["recommended"]]
    return {"quant_order": [r["quant"] for r in rows], "recommended": starred,
            "repo_default": DEFAULT}


JOURNEY = Journey(
    name="model_picker", tier="fast", routes=("/chat",),
    steps=(
        Step("p01_open_picker", open_picker),
        Step("p02_gguf_quants", gguf_quants),
    ),
)
