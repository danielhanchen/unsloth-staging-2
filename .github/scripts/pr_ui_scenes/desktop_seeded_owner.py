"""PR 13961: real desktop frontend/backend, with browser-only shell IPC.

The adapter replaces native window/event plumbing, preflight and the HTTP transport
of desktop_auth. It does not execute Rust, prove secret delivery by Rust, or test
native process ownership. Authentication decisions and UI routing remain real.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from pr_ui_scenes._common import Session, _post, api_get


def prepare_session(base_url: str, home: Path, _password: str) -> Session:
    """Verify this fresh install without consuming its seeded password state."""
    bootstrap = home / "auth" / ".bootstrap_password"
    if not bootstrap.exists():
        raise RuntimeError("Scene requires a fresh seeded owner; reinstall this UI home")
    password = bootstrap.read_text(encoding = "utf-8").strip()
    identity = _post(
        base_url + "/api/auth/login",
        {
            "username": "unsloth",
            "password": password,
        },
    )
    if identity.get("must_change_password") is not True:
        raise RuntimeError("Owner is not in the seeded-password state")
    python = home / "unsloth_studio" / "bin" / "python"
    script = """
import json
from unsloth_cli._inference import ensure_studio_backend_path
ensure_studio_backend_path(seed_cache_env=False)
from auth import storage
assert storage.requires_password_change('unsloth')
assert storage.account_counts() == (1, 0), 'Scene requires a fresh single-owner home'
print('SCENE_SEED=' + json.dumps({'secret': storage.create_desktop_secret(),
                              'counts': storage.account_counts()}))
"""
    result = subprocess.run(
        [str(python), "-c", script],
        env = {**os.environ, "UNSLOTH_STUDIO_HOME": str(home)},
        capture_output = True,
        text = True,
        check = True,
        timeout = 120,
    )
    seed = json.loads(
        next(
            line.removeprefix("SCENE_SEED=")
            for line in result.stdout.splitlines()
            if line.startswith("SCENE_SEED=")
        )
    )
    desktop = _post(base_url + "/api/auth/desktop-login", {"secret": seed["secret"]})
    if not desktop.get("access_token"):
        raise RuntimeError("Fresh single-owner Desktop authentication failed")
    _post(base_url + "/api/accounts", {"username": "ui-managed"}, token = desktop["access_token"])
    setup_session = Session(base_url, home, desktop["access_token"], "", password)
    accounts = api_get(setup_session, "/api/accounts")["accounts"]
    counts = [sum(a["is_active"] for a in accounts), sum(a["role"] != "owner" for a in accounts)]
    auth_status = api_get(setup_session, "/api/auth/status")
    if (
        counts != [2, 1]
        or auth_status.get("login_mode") != "multi"
        or auth_status.get("requires_password_change") is not True
    ):
        raise RuntimeError("Backend did not retain seeded owner + managed account")
    session = Session(base_url, home, "", "", password)
    session.desktop_scene_secret = seed["secret"]
    session.desktop_scene_counts = counts
    return session


def adapter_script(base_url: str, secret: str) -> str:
    config = json.dumps({"base": base_url, "secret": secret})
    return r"""(() => {
const config = CONFIG;
const callbacks = new Map(), listeners = new Map();
let next = 1;
window.__desktopScene = {calls: [], auth: [], unsupported: []};
const record = window.__desktopScene;
async function invoke(cmd, args = {}) {
  record.calls.push(cmd);
  if (cmd === 'desktop_auth') {
    const response = await fetch(config.base + '/api/auth/desktop-login', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({secret: config.secret, seeded_owner_login: true})
    });
    const data = await response.json();
    record.auth.push({status: response.status, login_required: data.login_required === true,
                      token_returned: typeof data.access_token === 'string'});
    if (!response.ok) throw new Error('Actual desktop-login failed: ' + response.status);
    return data;
  }
  if (cmd === 'desktop_preflight') {
    const health = await fetch(config.base + '/api/health');
    if (!health.ok) throw new Error('Actual backend health failed');
    return {disposition: 'owned_ready', port: Number(new URL(config.base).port)};
  }
  if (cmd === 'check_health') return (await fetch(config.base + '/api/health')).ok;
  if (cmd === 'plugin:event|listen') {
    const id = next++; listeners.set(id, {event: args.event, handler: args.handler}); return id;
  }
  if (cmd === 'plugin:event|unlisten') { listeners.delete(args.eventId); return; }
  if (cmd === 'plugin:event|emit' || cmd === 'plugin:event|emit_to') {
    for (const [id, listener] of listeners) if (listener.event === args.event)
      callbacks.get(listener.handler)?.({event: args.event, id, payload: args.payload});
    return;
  }
  if (cmd === 'desktop_update_policy') return {mode: 'manual_linux_package'};
  if (cmd === 'check_desktop_manual_update') return null;
  if (['was_launched_hidden', 'has_initialized_app_window_layout', 'has_saved_window_state',
       'take_native_layout_restored', 'desktop_update_cleanup_armed',
       'browser_view_supported'].includes(cmd)) return false;
  if (cmd === 'plugin:window|scale_factor') return 1;
  if (cmd === 'plugin:window|inner_size' || cmd === 'plugin:window|outer_size')
    return {width: innerWidth, height: innerHeight};
  if (cmd === 'plugin:window|outer_position' || cmd === 'plugin:window|inner_position')
    return {x: 0, y: 0};
  if (cmd === 'plugin:window|current_monitor' || cmd === 'plugin:window|primary_monitor')
    return {name: 'Browser fixture', size: {width: 1440, height: 900},
            position: {x: 0, y: 0}, scaleFactor: 1,
            workArea: {position: {x: 0, y: 0}, size: {width: 1440, height: 900}}};
  if (/^plugin:window\|is_/.test(cmd)) return false;
  if (/^plugin:window\|(set_|show$|hide$|center$)/.test(cmd) ||
      ['set_renderer_activity', 'set_tray_server_status', 'set_app_menu_actions',
       'mark_app_window_layout_initialized', 'reset_app_window_layout_initialized',
       'clear_native_browsing_data'].includes(cmd)) return;
  record.unsupported.push(cmd);
  throw new Error('Unsupported browser shell IPC: ' + cmd);
}
window.__TAURI_INTERNALS__ = {
  invoke, metadata: {currentWindow: {label: 'main'}, currentWebview: {label: 'main'}},
  transformCallback(callback, once = false) {
    const id = next++; callbacks.set(id, data => {
      if (once) callbacks.delete(id); return callback?.(data);
    }); return id;
  },
  unregisterCallback(id) { callbacks.delete(id); },
  runCallback(id, data) { callbacks.get(id)?.(data); },
  convertFileSrc(path) { throw new Error('Native files unavailable in browser fixture'); }
};
window.__TAURI_EVENT_PLUGIN_INTERNALS__ = {
  unregisterListener(_event, id) { listeners.delete(id); }
};
})();""".replace("CONFIG", config)


async def drive(session: Session, out_dir: Path, label: str, **_: object):
    from playwright.async_api import async_playwright

    if not hasattr(session, "desktop_scene_secret"):
        raise RuntimeError("Desktop scene requires pr_ui_diff prepare_session hook")
    errors = []
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless = True)
        try:
            context = await browser.new_context(viewport = {"width": 1440, "height": 900})
            await context.add_init_script(
                adapter_script(session.base_url, session.desktop_scene_secret)
            )
            page = await context.new_page()
            page.on("pageerror", lambda error: errors.append(str(error)))
            await page.goto(session.base_url + "/chat", wait_until = "domcontentloaded")
            await page.wait_for_function(
                """() => {
              const auth = window.__desktopScene?.auth;
              if (!auth?.length) return false;
              if (auth.at(-1).login_required)
                return location.pathname === '/login' && !!document.querySelector('input[type=password]');
              return location.pathname === '/chat' && !!document.querySelector('[contenteditable=true],textarea');
            }""",
                timeout = 120_000,
            )
            facts = await page.evaluate("""() => ({
              outcome: location.pathname === '/login' ? 'login' : 'owner_chat',
              desktop_auth: window.__desktopScene.auth,
              ipc_calls: [...new Set(window.__desktopScene.calls)].sort(),
              unsupported_ipc: [...new Set(window.__desktopScene.unsupported)].sort(),
              has_browser_token: !!localStorage.getItem('unsloth_auth_token'),
              visible_text: document.body.innerText.slice(0, 3000)
            })""")
            if facts["outcome"] == "owner_chat":
                identity = await page.evaluate("""async () => {
                  const token = localStorage.getItem('unsloth_auth_token');
                  const claims = JSON.parse(atob(token.split('.')[1].replace(/-/g, '+').replace(/_/g, '/')));
                  const response = await fetch('/api/accounts', {headers: {
                    Authorization: 'Bearer ' + token}});
                  if (!response.ok) throw new Error('Owner-only accounts request failed: ' + response.status);
                  return {username: claims.sub, role: claims.role};
                }""")
                if identity.get("username") != "unsloth":
                    raise RuntimeError("Visible chat is not authenticated as owner")
                facts["owner_username"] = identity["username"]
            facts["seeded_owner_password_preserved"] = (
                session.home / "auth" / ".bootstrap_password"
            ).read_text(encoding = "utf-8").strip() == session.password
            if not facts["seeded_owner_password_preserved"]:
                raise RuntimeError("Scene changed seeded owner password")
            facts["account_counts"] = session.desktop_scene_counts
            facts["_adapter_limit"] = (
                "Chromium IPC adapter forwards desktop_auth to real backend; native Rust, "
                "native secret delivery, native lifecycle and process ownership were not exercised."
            )
            facts["_page_errors"] = errors
            shot = out_dir / "desktop_seeded_owner.png"
            await page.screenshot(path = str(shot))
            return [shot], facts
        finally:
            await browser.close()
