"""Live integration smoke test for studio_test_kit.ui without Studio.

Spins up a tiny HTTP server that mimics enough of Studio's /chat surface
to exercise:
  - context.add_init_script seeding localStorage
  - page.goto + open_chat context manager
  - screenshot
  - video recording -> webm flush on context.close()
  - wait_for_text on a synthetic streamed token
  - wait_for_image on a data: URL <img>

Run: python3 -m studio_test_kit._smoke_ui
"""

from __future__ import annotations

import asyncio
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "outputs" / "studio_test_kit_smoke"
OUT.mkdir(parents=True, exist_ok=True)

HTML = b"""<!DOCTYPE html>
<html><head><title>fake studio</title></head>
<body>
<div id="banner">fake-studio /chat</div>
<form><textarea id="t"></textarea><button>Send</button></form>
<div id="thread"></div>
<script>
  window.STUDIO_TOKEN = window.localStorage.getItem('unsloth_auth_token');
  // Append a streaming text token after 500ms so wait_for_text sees it.
  setTimeout(() => {
    const t = document.getElementById('thread');
    const s = document.createElement('div'); s.id = 'tok'; s.textContent = 'TOKEN_OK';
    t.appendChild(s);
  }, 500);
  // Append a tiny data:image/png <img> after 1s so wait_for_image sees it.
  setTimeout(() => {
    const img = document.createElement('img');
    img.src = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkAAIAAAoAAv/lxKUAAAAASUVORK5CYII=';
    document.body.appendChild(img);
  }, 1000);
</script>
</body></html>
"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(HTML)

    def log_message(self, *_a):
        pass


def serve(port: int) -> HTTPServer:
    srv = HTTPServer(("127.0.0.1", port), Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv


async def main() -> int:
    from studio_test_kit.auth import StudioAuth, gemini_provider, seed_init_script
    from studio_test_kit.ui import open_chat, wait_for_image, wait_for_text

    port = 18902
    srv = serve(port)
    try:
        auth = StudioAuth(access_token="A.B.C", refresh_token="R.E.F",
                          base_url=f"http://127.0.0.1:{port}")
        init = seed_init_script(auth, [gemini_provider("AIza_test")])
        # Need a substring of the seed JSON so we know localStorage got primed.
        vid = OUT / "video"
        if vid.exists():
            for p in vid.iterdir():
                p.unlink()
        async with open_chat(
            f"http://127.0.0.1:{port}",
            init_scripts=[init],
            video_dir=vid,
            video_name="smoke",
            transcode_mp4=True,
            viewport=(800, 600),
            headless=True,
        ) as sp:
            # localStorage primed?
            token = await sp.page.evaluate("window.STUDIO_TOKEN")
            assert token == "A.B.C", f"init script did not run, got {token!r}"
            # screenshot
            shot = OUT / "open.png"
            await sp.screenshot(shot, full_page=False)
            assert shot.exists() and shot.stat().st_size > 200, shot
            # streamed text
            await wait_for_text(sp, "TOKEN_OK", timeout_ms=5000)
            # data: image
            # The injected 1x1 PNG decodes to ~70 bytes, so use a small
            # min_decoded_bytes threshold for the smoke fixture.
            data_url = await wait_for_image(
                sp, timeout_ms=5000, min_decoded_bytes=40,
            )
            assert data_url.startswith("data:image/png;base64,"), data_url[:32]
        # context closed -> webm should be flushed AND renamed
        assert sp.video_webm is not None, "video_webm not populated post-close"
        assert sp.video_webm.name == "smoke.webm", \
            f"webm not renamed to stable name: {sp.video_webm.name}"
        assert sp.video_webm.stat().st_size > 0
        # No `page@*.webm` should remain after the rename.
        leftover = list(vid.glob("page@*.webm"))
        assert not leftover, f"auto-named webm not cleaned: {leftover}"
        # transcode_mp4=True should have produced an mp4 too.
        assert sp.video_mp4 is not None and sp.video_mp4.exists(), \
            f"mp4 missing: {sp.video_mp4}"
        assert sp.video_mp4.stat().st_size > 0
        print(f"OK -- init script primed token; screenshot {shot.stat().st_size}B; "
              f"webm {sp.video_webm.name} {sp.video_webm.stat().st_size}B; "
              f"mp4 {sp.video_mp4.name} {sp.video_mp4.stat().st_size}B")
        return 0
    finally:
        srv.shutdown()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
