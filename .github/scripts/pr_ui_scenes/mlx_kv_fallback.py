"""Seed a loaded MLX status to compare the removed KV fallback notice."""

from pathlib import Path

from studio_test_kit.auth import seed_init_script
from studio_test_kit.ui import open_chat


async def drive(session, out_dir: Path, label: str, **kwargs):
    status = dict(
        loaded = True,
        loading = False,
        active_model = "Qwen3.5-0.8B",
        model_identifier = "Qwen/Qwen3.5-0.8B",
        is_mlx = True,
        is_gguf = False,
        is_vision = True,
        max_seq_length = 4096,
        context_length = 4096,
        speculative_type = "mtp",
        spec_drafter_kind = "mtp",
        spec_fallback_reason = "kv_quant",
        mlx_kv_bits = 8,
        mlx_kv_quant = "q8",
    )
    auth = type(
        "Auth", (), dict(access_token = session.access_token, refresh_token = session.refresh_token)
    )()
    init = seed_init_script(auth, [])
    async with open_chat(
        session.base_url, init_scripts = [init], viewport = (1440, 1000), headless = True
    ) as sp:
        page = sp.page
        await page.route("**/api/inference/status*", lambda route: route.fulfill(json = status))
        await page.goto(session.base_url + "/chat", wait_until = "domcontentloaded")
        await page.get_by_role("textbox", name = "Message input").wait_for(timeout = 60000)
        opener = page.get_by_role("button", name = "Open run settings")
        await opener.first.wait_for(timeout = 30000)
        await opener.first.click()
        await sp.screenshot(out_dir / (label.lower() + "_debug.png"), full_page = False)
        (out_dir / (label.lower() + "_debug.txt")).write_text(
            await page.locator("body").inner_text(), encoding = "utf-8"
        )
        await page.get_by_text("Qwen3.5-0.8B", exact = False).first.wait_for(timeout = 30000)
        notice = page.get_by_text(
            "Speculative decoding cannot run with KV cache quantization", exact = False
        )
        if label.upper() == "BEFORE":
            await notice.wait_for(timeout = 30000)
        else:
            await notice.wait_for(state = "hidden", timeout = 30000)
        shot = out_dir / (label.lower() + "_settings.png")
        await sp.screenshot(shot, full_page = False)
        facts = dict(
            seeded_status = True, kv_fallback_visible = await notice.count(), model = "Qwen/Qwen3.5-0.8B"
        )
    return [shot], facts
