"""Scene: an indented multi-line prompt, sent, read back out of the user bubble.

No model and no reply: an external checkpoint is selected so the composer can send, and the
user bubble renders before the (keyless, failing) request. Facts are the bubble paragraph's
computed `white-space` and each marker line's left offset, so flush-left is a number.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

WORKSPACE = Path(
    os.environ.get("WORKSPACE")
    or os.environ.get("UNSLOTH_WORKSPACE")
    or Path(__file__).resolve().parents[2]
)
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "scripts"))

from pr_ui_scenes._common import Session, api_delete, api_get, api_post  # noqa: E402
from studio_test_kit.auth import seed_init_script  # noqa: E402
from studio_test_kit.ui import open_chat  # noqa: E402

PROMPT = "Plan:\n- fruit\n    - apple\n        - green\ndef f():\n\treturn 1"
MARKERS = ["- fruit", "- apple", "- green", "def f():", "return 1"]
PROVIDER = "Indent Scene"
MODEL = "indent-model"

MEASURE = r"""
(markers) => {
  const bubbles = document.querySelectorAll('[data-role="user"] .aui-user-message-content');
  const bubble = bubbles[bubbles.length - 1];
  if (!bubble) return null;
  const p = bubble.querySelector('p') || bubble;
  const walker = document.createTreeWalker(p, NodeFilter.SHOW_TEXT);
  const nodes = []; let n; while ((n = walker.nextNode())) nodes.push(n);
  const full = nodes.map(t => t.data).join('');
  const xAt = (offset) => {
    let acc = 0;
    for (const t of nodes) {
      if (offset < acc + t.data.length) {
        const r = document.createRange();
        r.setStart(t, offset - acc); r.setEnd(t, offset - acc + 1);
        return r.getBoundingClientRect().left;
      }
      acc += t.data.length;
    }
    return null;
  };
  const left = p.getBoundingClientRect().left, xs = {};
  for (const m of markers) {
    const i = full.indexOf(m), x = i < 0 ? null : xAt(i);
    xs[m] = x === null ? null : Math.round(x - left);
  }
  return { white_space: getComputedStyle(p).whiteSpace, text: full, x_offsets: xs };
}
"""


async def drive(session: Session, out_dir: Path, label: str, **_: object):
    for existing in api_get(session, "/api/providers/") or []:
        if existing.get("display_name") == PROVIDER:
            api_delete(session, f"/api/providers/{existing['id']}")
    # A CUSTOM-typed provider (display name not in the registry) may carry no API key.
    pid = api_post(
        session,
        "/api/providers/",
        {
            "provider_type": "openai",
            "display_name": PROVIDER,
            "base_url": None,
            "models": [MODEL],
            "available_models": [MODEL],
        },
    ).get("id")
    if not pid:
        raise RuntimeError("provider creation returned no id")
    try:
        init = seed_init_script(
            type(
                "A",
                (),
                {"access_token": session.access_token, "refresh_token": session.refresh_token},
            )(),
            [],
            connections_enabled = True,
            extra_local_storage = {
                "unsloth_chat_auto_title": "false",
                "unsloth_chat_last_external_checkpoint": f"external::{pid}::{MODEL}",
            },
        )
        async with open_chat(
            session.base_url, init_scripts = [init], viewport = (1200, 900), headless = True
        ) as sp:
            page = sp.page
            # The provider list is rebuilt from the backend after boot; sending before it lands
            # leaves no model selected and the composer refuses.
            await page.wait_for_function(
                f"() => (localStorage.getItem('unsloth_chat_external_providers') || '').includes('{pid}')",
                timeout = 60_000,
            )
            box = page.locator("form:has(textarea) textarea").first
            await box.click()
            await box.fill(PROMPT)
            await box.press("Enter")
            bubble = page.locator('[data-role="user"] .aui-user-message-content').last
            await bubble.wait_for(state = "visible", timeout = 60_000)
            m = await page.evaluate(MEASURE, MARKERS)
            if not m or m["text"] != PROMPT:
                raise RuntimeError(f"bubble text {m and m['text']!r} != prompt {PROMPT!r}")
            if any(v is None for v in m["x_offsets"].values()):
                raise RuntimeError(f"a marker was not rendered: {m['x_offsets']}")
            await page.evaluate(
                "() => document.querySelectorAll('[data-sonner-toaster]').forEach(e => e.remove())"
            )
            shot = out_dir / f"{label}_bubble.png"
            await bubble.screenshot(path = str(shot), animations = "disabled")
    finally:  # never leave the test provider in the side's Studio, even when the scene fails
        api_delete(session, f"/api/providers/{pid}")
    return [shot], m
