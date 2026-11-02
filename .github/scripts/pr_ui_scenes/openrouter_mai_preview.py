# SPDX-License-Identifier: AGPL-3.0-only
import asyncio
import json
import math
import struct
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from pr_ui_scenes._common import Session, api_post
from studio_test_kit.auth import StudioAuth, seed_init_script
from studio_test_kit.ui import open_chat

MODEL = "microsoft/mai-voice-2.1-flash"
VOICE = "de-DE-Mia:MAI-Voice-2.1-Flash"


async def drive(session: Session, out_dir: Path, label: str, **_):
    records = []
    pcm = b"".join(
        struct.pack("<h", int(300 * math.sin(i * 2 * math.pi * 220 / 24000)))
        for i in range(24000 * 20)
    )

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            records.append(body)
            accepted = body.get("response_format") == "pcm"
            payload = pcm if accepted else b'{"error":"response_format: expected mp3 or pcm"}'
            self.send_response(200 if accepted else 400)
            self.send_header("Content-Type", "audio/pcm" if accepted else "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target = server.serve_forever, kwargs = {"poll_interval": 0.05})
    thread.start()
    try:
        provider = api_post(
            session,
            "/api/providers/",
            {
                "provider_type": "openrouter",
                "display_name": "OpenRouter MAI-Voice",
                "base_url": f"http://127.0.0.1:{server.server_port}/v1",
                "models": [MODEL],
                "available_models": [MODEL],
            },
        )
        voice_state = {
            "state": {
                "ttsEnabled": True,
                "ttsEngine": "custom",
                "ttsProviderId": provider["id"],
                "ttsProviderModel": MODEL,
                "ttsProviderVoice": VOICE,
                "ttsRate": 1,
                "ttsPitch": 1,
                "ttsVolume": 0.1,
            },
            "version": 1,
        }
        init = seed_init_script(
            StudioAuth(session.access_token, session.refresh_token, session.base_url),
            [],
            extra_local_storage = {
                "unsloth_voice_settings": voice_state,
                "unsloth_chat_connections_enabled": "true",
            },
        )
        init += """
        const nativePlay = HTMLMediaElement.prototype.play;
        HTMLMediaElement.prototype.play = function(...args) {
            this.addEventListener('playing', () => { window.__previewPlayed = true; }, {once: true});
            return nativePlay.apply(this, args);
        };
        """
        out_dir.mkdir(parents = True, exist_ok = True)
        async with open_chat(
            session.base_url,
            init_scripts = [init],
            headless = True,
            viewport = (1440, 1100),
            video_dir = out_dir / "video",
            video_name = label.lower(),
        ) as sp:
            page = sp.page
            page.on(
                "console",
                lambda msg: print(f"[{label}] browser {msg.type}: {msg.text}", flush = True)
                if msg.type == "error"
                else None,
            )
            page.on("pageerror", lambda error: print(f"[{label}] page error: {error}", flush = True))
            for attempt in range(3):
                try:
                    await page.goto(session.base_url + "/chat", wait_until = "domcontentloaded")
                    break
                except Exception as exc:
                    if "ERR_NETWORK_CHANGED" not in str(exc) or attempt == 2:
                        raise
            await page.get_by_role("button", name = "Settings", exact = True).click()
            await page.get_by_test_id("settings-tab-voice").click()
            preview = page.get_by_role("button", name = "Preview", exact = True)
            await preview.scroll_into_view_if_needed()
            await page.locator(f'input[value="{MODEL}"]').wait_for(timeout = 20_000)
            async with page.expect_response(
                lambda r: "/api/inference/audio/speech" in r.url, timeout = 30_000
            ) as pending:
                await preview.click()
            response = await pending.value
            facts = {
                "http_status": response.status,
                "requests": records,
                "content_type": response.headers.get("content-type"),
            }
            if response.status == 200:
                await page.wait_for_function("window.__previewPlayed === true", timeout = 15_000)
                await page.get_by_role("button", name = "Stop", exact = True).wait_for()
                facts["playback"] = "playing"
            else:
                facts["error"] = await response.text()
                await page.get_by_text("TTS endpoint returned HTTP 400", exact = False).wait_for()
                facts["playback"] = "error"
            shot = out_dir / "preview.png"
            await sp.screenshot(shot, full_page = False)
            if response.status == 200:
                await page.get_by_role("button", name = "Stop", exact = True).click()
        return [shot], facts
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
