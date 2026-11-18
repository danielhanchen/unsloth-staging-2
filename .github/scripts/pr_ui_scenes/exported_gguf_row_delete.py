"""Scene: the composer model picker's row menu for a one-quant exported GGUF.

Serves PR 18023. An exported GGUF deletes per quant, and on the merge base the per-quant
Delete only lives inside the expanded quant list. A row with exactly ONE quant on disk has
no quant list to expand, so its row menu offered no Delete at all. Head adds Delete to that
row menu, opening the existing "Delete exported GGUF variant?" confirm.

Seeding: one tiny hand-written GGUF (a few hundred bytes, valid v3 header, one 1x32 F32
tensor) under `<home>/exports/<run>/`, the flat `*-gguf` export layout `scan_exported_models`
reads. Seeded by the scene itself at the start of every run so both sides start from the
same disk, including after AFTER confirmed a delete on a previous run.

No weights, no GPU.
"""

from __future__ import annotations

import asyncio
import os
import re
import struct
import sys
from pathlib import Path

WORKSPACE = Path(
    os.environ.get("WORKSPACE")
    or os.environ.get("UNSLOTH_WORKSPACE")
    or Path(__file__).resolve().parents[2]
)
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "scripts"))

from pr_ui_scenes._common import Session, api_get, open_menu  # noqa: E402
from studio_test_kit.auth import seed_init_script  # noqa: E402
from studio_test_kit.ui import open_chat  # noqa: E402

DEFAULT_RUN = "tiny-export-gguf"
DEFAULT_QUANT = "Q8_0"


def _gguf_str(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def write_tiny_gguf(path: Path, name: str) -> None:
    """A valid GGUF v3: 3 string KVs, one F32 tensor of 32 values."""
    kvs = [("general.architecture", "llama"), ("general.name", name), ("general.file_type", None)]
    body = b""
    n_kv = 0
    for k, v in kvs:
        if v is None:
            body += _gguf_str(k) + struct.pack("<I", 4) + struct.pack("<I", 7)  # uint32 Q8_0
        else:
            body += _gguf_str(k) + struct.pack("<I", 8) + _gguf_str(v)
        n_kv += 1
    tensor_info = (
        _gguf_str("token_embd.weight")
        + struct.pack("<I", 2)
        + struct.pack("<QQ", 32, 1)
        + struct.pack("<I", 0)
        + struct.pack("<Q", 0)
    )
    head = b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", 1, n_kv) + body + tensor_info
    head += b"\0" * ((-len(head)) % 32)
    data = struct.pack("<32f", *([0.0] * 32))
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_bytes(head + data)


def seed(home: Path, run: str, quant: str) -> Path:
    run_dir = Path(home) / "exports" / run
    run_dir.mkdir(parents = True, exist_ok = True)
    for stale in run_dir.glob("*.gguf"):
        stale.unlink()
    gguf = run_dir / f"{run}-{quant}.gguf"
    write_tiny_gguf(gguf, run)
    return gguf


def _placeholder(path: Path, text: str, width: int, height: int) -> None:
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception:  # noqa: BLE001
        font = ImageFont.load_default()
    y = height // 2 - 30
    for line in text.split("\n"):
        w = d.textlength(line, font = font)
        d.text(((width - w) / 2, y), line, fill = "black", font = font)
        y += 28
    img.save(path)


def _pad_to(path: Path, width: int, height: int) -> None:
    from PIL import Image

    img = Image.open(path).convert("RGB")
    if img.width > width or img.height > height:
        raise RuntimeError(f"{path.name} is {img.width}x{img.height} > {width}x{height}")
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(img, ((width - img.width) // 2, (height - img.height) // 2))
    canvas.save(path)


async def _union_clip(
    page,
    locators,
    pad: int = 12,
) -> dict:
    boxes = [await loc.bounding_box() for loc in locators]
    boxes = [b for b in boxes if b]
    x0 = max(0, min(b["x"] for b in boxes) - pad)
    y0 = max(0, min(b["y"] for b in boxes) - pad)
    x1 = max(b["x"] + b["width"] for b in boxes) + pad
    y1 = max(b["y"] + b["height"] for b in boxes) + pad
    return {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0}


async def drive(
    session: Session,
    out_dir: Path,
    label: str,
    run: str = DEFAULT_RUN,
    quant: str = DEFAULT_QUANT,
    confirm_on_after: bool = True,
    **_: object,
) -> tuple[list[Path], dict]:
    facts: dict = {}
    gguf = seed(session.home, run, quant)
    facts["seeded_file"] = gguf.name
    try:
        loras = api_get(session, "/api/models/loras")
        rows = loras.get("loras", loras) if isinstance(loras, dict) else loras
        facts["exported_rows"] = [
            {k: r.get(k) for k in ("display_name", "source", "export_type")}
            for r in rows
            if isinstance(r, dict) and r.get("source") == "exported"
        ]
    except Exception as exc:  # noqa: BLE001
        facts["exported_rows"] = f"{type(exc).__name__}: {exc}"

    shots: list[Path] = []
    init = seed_init_script(
        type(
            "A", (), {"access_token": session.access_token, "refresh_token": session.refresh_token}
        )(),
        [],
    )
    async with open_chat(
        session.base_url, init_scripts = [init], viewport = (1500, 1000), headless = True
    ) as sp:
        page = sp.page
        variant_calls: list = []

        async def _on_response(resp):
            if "gguf-variants" in resp.url:
                try:
                    body = await resp.text()
                except Exception:  # noqa: BLE001
                    body = "?"
                variant_calls.append(
                    {
                        "url": resp.url[len(session.base_url) :],
                        "status": resp.status,
                        "body": body[:400],
                    }
                )

        page.on("response", _on_response)
        await page.goto(f"{session.base_url}/chat", wait_until = "domcontentloaded")
        chip = page.locator('button[aria-label^="Model: "]').first
        await chip.wait_for(state = "visible", timeout = 120_000)
        await chip.click()
        device_tab = (
            page.get_by_role("tab", name = re.compile(r"^On device$", re.I))
            .or_(page.locator('[aria-label="On device"]'))
            .first
        )
        try:
            await device_tab.click(timeout = 10_000)
        except Exception:  # noqa: BLE001 -- already on that tab
            pass
        search = page.get_by_placeholder(re.compile(r"^Search .*models")).first
        await search.wait_for(state = "visible", timeout = 30_000)
        await search.fill(run)

        trigger = page.locator(f'button[aria-label="More options for {run}"]').first
        await trigger.wait_for(state = "visible", timeout = 120_000)
        # The row's quant list is read after the row paints; the Delete item (head) and the
        # Quant line (both) depend on it, so wait for the read rather than a fixed sleep.
        for _ in range(40):
            if any(run in c["url"] for c in variant_calls):
                break
            await page.wait_for_timeout(500)
        await page.wait_for_timeout(3_000)
        facts["variant_calls"] = [c for c in variant_calls if run in c["url"]]
        facts["row_text"] = (
            " ".join(
                (await page.locator(f'[aria-label="{run} actions"]').first.inner_text()).split()
            )
            if await page.locator(f'[aria-label="{run} actions"]').count()
            else None
        )

        menu = page.get_by_role("menu").first
        await open_menu(page, trigger, menu)
        await page.wait_for_timeout(600)
        items = [" ".join(t.split()) for t in await menu.get_by_role("menuitem").all_inner_texts()]
        facts["menu_items"] = items
        delete_item = menu.get_by_role("menuitem", name = re.compile(r"^\s*Delete\s*$"))
        facts["has_delete"] = await delete_item.count() > 0

        shot = out_dir / f"{label.lower()}_0_row_menu.png"
        clip = await _union_clip(page, [trigger, menu])
        await page.screenshot(path = str(shot), clip = clip)
        _pad_to(shot, 640, 420)
        shots.append(shot)

        shot2 = out_dir / f"{label.lower()}_1_delete_dialog.png"
        if facts["has_delete"]:
            await delete_item.first.click()
            dialog = page.get_by_role("alertdialog").first
            await dialog.wait_for(state = "visible", timeout = 20_000)
            await page.wait_for_timeout(1_000)
            facts["dialog_title"] = " ".join(
                (await dialog.get_by_role("heading").first.inner_text()).split()
            )
            facts["dialog_text"] = " ".join((await dialog.inner_text()).split())
            facts["dialog_names_quant"] = f"({quant})" in facts["dialog_text"]
            await dialog.screenshot(path = str(shot2))
            _pad_to(shot2, 640, 420)
            if confirm_on_after and label.upper() == "AFTER":
                await dialog.get_by_role("button", name = re.compile(r"^Delete")).last.click()
                await dialog.wait_for(state = "hidden", timeout = 30_000)
                await page.wait_for_timeout(2_000)
                facts["file_exists_after_confirm"] = gguf.exists()
        else:
            facts["dialog_title"] = None
            _placeholder(
                shot2, "Row menu has no Delete item:\nno delete confirm can be opened", 640, 420
            )
        shots.append(shot2)
    facts["file_exists_end"] = gguf.exists()
    return shots, facts
