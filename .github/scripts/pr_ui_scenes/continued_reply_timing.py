from __future__ import annotations

import json
from pathlib import Path

from pr_ui_scenes._common import api_get, api_post
from pr_ui_scenes.chat_stream_content_parity import (
    SHIM,
    _ScriptedStreamServer,
    _sse,
    _delta,
    _HEAD,
    _DONE,
    _SETTINGS,
)
from studio_test_kit.auth import seed_init_script
from studio_test_kit.ui import open_chat, send_prompt


async def drive(session, out_dir: Path, label: str, **kwargs):
    scripts = []
    for text, tokens, finish, ms in [
        ("The first part of this answer contains 128 tokens. ", 128, "length", 3800),
        ("This continuation adds eleven more tokens.", 11, "stop", 300),
    ]:
        scripts.append(
            [
                (0, _HEAD),
                (0.3, _delta(text)),
                (
                    0.3,
                    _sse(
                        {
                            "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                            "usage": {
                                "prompt_tokens": 40,
                                "completion_tokens": tokens,
                                "total_tokens": 40 + tokens,
                            },
                            "timings": {
                                "prompt_n": 40,
                                "prompt_ms": 100,
                                "prompt_per_second": 400,
                                "predicted_n": tokens,
                                "predicted_ms": ms,
                                "predicted_per_second": (tokens - 1) * 1000 / ms,
                            },
                        }
                    ),
                ),
                (0.05, _DONE),
            ]
        )
    server = _ScriptedStreamServer(scripts)
    try:
        providers = api_get(session, "/api/providers/") or []
        provider = next((p for p in providers if p.get("display_name") == "Timing Scene"), None)
        if not provider:
            provider = api_post(
                session,
                "/api/providers/",
                {
                    "provider_type": "openai",
                    "display_name": "Timing Scene",
                    "base_url": None,
                    "models": ["timing-model"],
                    "available_models": ["timing-model"],
                },
            )
        provider_id = provider["id"]
        settings = {
            **_SETTINGS,
            "unsloth_chat_last_external_checkpoint": f"external::{provider_id}::timing-model",
            "unsloth_chat_external_provider_keys": {provider_id: "timing-scene-fixture"},
        }
        auth = type(
            "Auth",
            (),
            {"access_token": session.access_token, "refresh_token": session.refresh_token},
        )()
        init = seed_init_script(auth, [], connections_enabled = True, extra_local_storage = settings)
        shim = SHIM.replace("__SOURCE__", f"http://127.0.0.1:{server.port}/stream")
        shim = shim.replace(
            "window.__run.intercepted.push(url);",
            """
            window.__run.intercepted.push(url);
            window.__run.bodies ??= [];
            try { window.__run.bodies.push(JSON.parse(args[1]?.body || '{}')); }
            catch { window.__run.bodies.push(null); }
        """,
        )
        async with open_chat(
            session.base_url,
            init_scripts = [init, shim],
            viewport = (1200, 900),
            headless = True,
            bypass_csp = True,
        ) as sp:
            page = sp.page
            await page.wait_for_function(
                "id => (localStorage.getItem('unsloth_chat_external_providers') || '').includes(id)",
                arg = provider_id,
            )
            await send_prompt(sp, "Please continue this answer when you reach the token limit.")
            try:
                await page.wait_for_function(
                    "() => window.__run.count >= 1 && window.__run.done[0]", timeout = 30000
                )
                resume = page.get_by_role("button", name = "Resume", exact = True)
                if await resume.count() and await resume.is_visible():
                    await resume.click()
                await page.wait_for_function(
                    "() => window.__run.count === 2 && window.__run.done[1]", timeout = 30000
                )
            except Exception:
                await page.screenshot(path = str(out_dir / f"{label}_debug.png"))
                print(
                    await page.evaluate(
                        "() => ({run: window.__run, text: document.body.innerText})"
                    ),
                    flush = True,
                )
                raise
            await page.get_by_text(
                "This continuation adds eleven more tokens.", exact = False
            ).first.wait_for()
            trigger = page.locator('[data-slot="message-timing-trigger"]').last
            await trigger.wait_for()
            await trigger.hover()
            tooltip = page.locator('[data-slot="message-timing-popover"]').last
            await tooltip.wait_for()
            text = await tooltip.inner_text()
            expected_tokens = "139" if label.upper() == "AFTER" else "11"
            rows = text.splitlines()
            assert rows[rows.index("Tokens") + 1] == expected_tokens, text
            assert server.served == [0, 1], server.served
            shot = out_dir / f"{label}_timing.png"
            await page.screenshot(path = str(shot), animations = "disabled")
            facts = {
                "tokens": int(expected_tokens),
                "tooltip": text,
                "requests": len(server.served),
                "stream": "128 tokens + 11 tokens",
                "request_bodies": await page.evaluate("() => window.__run.bodies"),
                "reply": await page.locator('[data-role="assistant"]').last.inner_text(),
            }
            (out_dir / f"{label}_facts.json").write_text(
                json.dumps(facts, indent = 2), encoding = "utf-8"
            )
            return [shot], facts
    finally:
        server.close()
