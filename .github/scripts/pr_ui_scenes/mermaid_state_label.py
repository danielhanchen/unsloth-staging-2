from pathlib import Path

from pr_ui_scenes._common import api_get
from pr_ui_scenes.chat_export_filename import _put
from pr_ui_scenes._common import api_post
from studio_test_kit.auth import seed_init_script
from studio_test_kit.ui import open_chat


async def drive(session, out_dir: Path, label: str, **kwargs):
    thread = "uidiff-mermaid-state-label"
    source = 'stateDiagram-v2\n  state "Apply style url(parameter)" as A\n  A --> B'
    api_post(
        session,
        "/api/chat/threads",
        {
            "id": thread,
            "title": "State label rendering",
            "modelType": "base",
            "modelId": "fixture",
            "createdAt": 1767225600000,
            "updatedAt": 1767225600000,
        },
    )
    for mid, parent, role, text in [
        ("user", None, "user", "Show a state diagram with a literal parameter label."),
        ("answer", "user", "assistant", f"```mermaid\n{source}\n```"),
    ]:
        _put(
            session,
            f"/api/chat/threads/{thread}/messages/{mid}",
            {
                "id": mid,
                "threadId": thread,
                "parentId": parent,
                "role": role,
                "content": [{"type": "text", "text": text}],
                "createdAt": 1767225600001,
            },
        )
    assert api_get(session, f"/api/chat/threads/{thread}")["title"] == "State label rendering"
    init = seed_init_script(session, [])
    async with open_chat(
        session.base_url, init_scripts = [init], viewport = (1200, 800), headless = True
    ) as sp:
        await sp.page.goto(
            f"{session.base_url}/chat?thread={thread}", wait_until = "domcontentloaded"
        )
        content = sp.page.locator('[data-role="assistant"] .aui-assistant-message-content').last
        await content.wait_for(state = "visible", timeout = 60000)
        await sp.page.wait_for_function(
            """() => {
            const c = document.querySelector('[data-role="assistant"] .aui-assistant-message-content');
            return c && (c.querySelector('svg[aria-roledescription]') || /error|failed/i.test(c.innerText));
        }""",
            timeout = 60000,
        )
        facts = await content.evaluate("""c => ({
            label_rendered: Array.from(c.querySelectorAll('svg')).some(s => s.textContent.includes('Apply style url(parameter)')),
            visible_text: c.innerText,
            svg_count: c.querySelectorAll('svg[aria-roledescription]').length
        })""")
        shot = out_dir / f"{label.lower()}_state_label.png"
        await content.screenshot(path = str(shot))
    return [shot], facts
