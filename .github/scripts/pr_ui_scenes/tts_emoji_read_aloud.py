"""Capture Read aloud's request; screenshots establish chat retention, not audible output."""

from __future__ import annotations

import base64
import io
import json
import wave
from pathlib import Path

from pr_ui_scenes._common import Session, api_delete, api_get, api_post
from studio_test_kit.auth import seed_init_script
from studio_test_kit.ui import open_chat, send_prompt


REPLY = "Hallo! 🥰❤️ Schön, dass du da bist. 😘"
MODEL = "speech-evidence-model"
PROVIDER_NAME = "Speech Emoji Evidence"

SHIM = r"""
(() => {
  const reply = __REPLY__;
  const wav = Uint8Array.from(atob(__WAV__), c => c.charCodeAt(0));
  const realFetch = window.fetch.bind(window);
  window.__ttsEvidence = {chatRequests: [], speechRequests: []};
  window.fetch = async (input, init) => {
    const url = typeof input === "string" ? input : input.url;
    if (/chat\/completions|generate\/stream/.test(url)) {
      window.__ttsEvidence.chatRequests.push(JSON.parse(init.body));
      const chunks = [
        {choices: [{index: 0, delta: {role: "assistant", content: reply}, finish_reason: null}]},
        {choices: [{index: 0, delta: {}, finish_reason: "stop"}]},
      ];
      const body = chunks.map(c => "data: " + JSON.stringify({
        id: "speech-evidence", object: "chat.completion.chunk", model: "speech-evidence-model", ...c
      }) + "\n\n").join("")
        + "data: [DONE]\n\n";
      return new Response(body, {headers: {"Content-Type": "text/event-stream"}});
    }
    if (/\/api\/inference\/audio\/speech(?:\?|$)/.test(url)) {
      window.__ttsEvidence.speechRequests.push(JSON.parse(init.body));
      return new Response(wav, {headers: {"Content-Type": "audio/wav"}});
    }
    return realFetch(input, init);
  };
})();
"""


def _silent_wav() -> str:
    stream = io.BytesIO()
    with wave.open(stream, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(8000)
        output.writeframes(bytes(16000))
    return base64.b64encode(stream.getvalue()).decode("ascii")


async def drive(
    session: Session, out_dir: Path, label: str, **_: object
) -> tuple[list[Path], dict]:
    out_dir.mkdir(parents = True, exist_ok = True)
    for provider in api_get(session, "/api/providers/") or []:
        if provider.get("display_name") == PROVIDER_NAME:
            api_delete(session, f"/api/providers/{provider['id']}")
    provider = api_post(
        session,
        "/api/providers/",
        {
            "provider_type": "openai",
            "display_name": PROVIDER_NAME,
            "base_url": "http://127.0.0.1:9/v1",
            "models": [MODEL],
            "available_models": [MODEL],
        },
    )
    provider_id = provider["id"]
    settings = {
        "unsloth_chat_last_external_checkpoint": f"external::{provider_id}::{MODEL}",
        "unsloth_chat_external_provider_keys": {provider_id: "speech-evidence-fixture"},
        "unsloth_chat_auto_title": "false",
        "unsloth_chat_tools_enabled": "false",
        "unsloth_chat_code_tools_enabled": "false",
        "unsloth_chat_mcp_enabled": "false",
        "unsloth_chat_deep_research_enabled": "false",
        "unsloth_voice_settings": {
            "version": 1,
            "state": {
                "ttsEnabled": True,
                "autoReadAloud": False,
                "ttsEngine": "custom",
                "ttsProviderId": provider_id,
                "ttsProviderModel": MODEL,
                "ttsProviderVoice": "alloy",
            },
        },
    }
    auth = type(
        "Auth",
        (),
        {
            "access_token": session.access_token,
            "refresh_token": session.refresh_token,
        },
    )()
    init = seed_init_script(auth, [], extra_local_storage = settings)
    shim = SHIM.replace("__REPLY__", json.dumps(REPLY)).replace(
        "__WAV__", json.dumps(_silent_wav())
    )
    shots = []
    async with open_chat(
        session.base_url,
        init_scripts = [init, shim],
        viewport = (1440, 900),
        video_dir = out_dir / f"{label}_video",
        video_name = label,
    ) as studio:
        page = studio.page
        await page.wait_for_function(
            "id => (localStorage.getItem('unsloth_chat_external_providers') || '').includes(id)",
            arg = provider_id,
            timeout = 60000,
        )
        await send_prompt(studio, "Please greet me in German.")
        content = page.locator('[data-role="assistant"] .aui-assistant-message-content').last
        try:
            await content.get_by_text(REPLY, exact = True).wait_for(timeout = 60000)
        except Exception:
            await studio.screenshot(out_dir / f"{label}_failure.png", full_page = False)
            (out_dir / f"{label}_failure.json").write_text(
                json.dumps(
                    await page.evaluate("""() => ({
                    requests: window.__ttsEvidence,
                    providers: localStorage.getItem('unsloth_chat_external_providers'),
                    text: document.body.innerText
                })"""),
                    ensure_ascii = False,
                    indent = 2,
                ),
                encoding = "utf-8",
            )
            raise
        before = await content.inner_text()
        assert REPLY in before, before
        read_aloud = page.get_by_role("button", name = "Read aloud", exact = True).last
        await read_aloud.click(timeout = 30000)
        await page.wait_for_function(
            "() => window.__ttsEvidence.speechRequests.length === 1", timeout = 30000
        )
        await page.get_by_role("button", name = "Read aloud", exact = True).last.wait_for(timeout = 30000)
        after = await content.inner_text()
        assert after == before, {"before": before, "after": after}
        captured = await page.evaluate("() => window.__ttsEvidence")
        assert len(captured["chatRequests"]) == 1, captured
        assert len(captured["speechRequests"]) == 1, captured
        request = captured["speechRequests"][0]
        assert request["provider_id"] == provider_id, request
        assert request["model"] == MODEL, request
        assert isinstance(request.get("input"), str) and request["input"], request
        shot = out_dir / f"{label}_read_aloud.png"
        await studio.screenshot(shot, full_page = False)
        shots.append(shot)
    facts = {
        "displayed_reply": after,
        "chat_unchanged_after_read_aloud": after == before,
        "chat_emojis_retained": all(emoji in after for emoji in ["🥰", "❤️", "😘"]),
        "speech_input": request["input"],
        "speech_contains_emojis": any(emoji in request["input"] for emoji in ["🥰", "❤️", "😘"]),
        "speech_request_count": len(captured["speechRequests"]),
        "chat_request_count": len(captured["chatRequests"]),
        "boundary": "fixed chat SSE and silent WAV at fetch; real built frontend and Read aloud action",
    }
    (out_dir / f"{label}_speech_request.json").write_text(
        json.dumps({"facts": facts, "request": request}, ensure_ascii = False, indent = 2),
        encoding = "utf-8",
    )
    return shots, facts
