"""Functional self-tests for studio_test_kit without a running Studio.

Covers:
  - auth.seed_init_script JS payload + JSON well-formedness
  - auth.gemini_provider / openai_provider / anthropic_provider shapes
  - compose.hstack_images / vstack_images on synthetic PNGs
  - compose.hstack_videos / webm_to_mp4 on tiny synthetic webms (ffmpeg)
  - lifecycle module-level callables (no live install)
  - ui module imports + selectors literal sanity
  - flows module imports

Run: python3 -m studio_test_kit._self_test
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def _ok(name: str) -> None:
    print(f"OK   {name}")


def _fail(name: str, err: Exception) -> None:
    print(f"FAIL {name}: {type(err).__name__}: {err}", file=sys.stderr)


def test_auth_seed() -> None:
    from studio_test_kit.auth import (
        StudioAuth, gemini_provider, openai_provider, anthropic_provider,
        seed_init_script,
    )
    auth = StudioAuth(access_token="A.B.C", refresh_token="R.E.F", base_url="http://x")
    p_gem = gemini_provider("AIza_test", models=["gemini-2.5-flash"])
    p_oai = openai_provider("sk-test", models=["gpt-4o-mini"])
    p_ant = anthropic_provider("sk-ant-test", models=["claude-3-5-haiku-latest"])
    assert p_gem.provider_type == "gemini"
    assert p_oai.base_url == "https://api.openai.com/v1"
    assert p_ant.models == ["claude-3-5-haiku-latest"]
    assert p_gem.id != p_oai.id  # unique uuid hex

    js = seed_init_script(auth, [p_gem, p_oai, p_ant], connections_enabled=True)
    assert "window.localStorage.setItem" in js
    assert "unsloth_auth_token" in js
    assert "unsloth_chat_external_providers" in js
    assert "unsloth_chat_external_provider_keys" in js
    # Extract the embedded JSON payload literal (after `const seed = `).
    marker = "const seed = "
    start = js.index(marker) + len(marker)
    end = js.index(";", start)
    payload_literal = js[start:end]
    payload = json.loads(payload_literal)
    providers = json.loads(payload["unsloth_chat_external_providers"])
    keys = json.loads(payload["unsloth_chat_external_provider_keys"])
    assert len(providers) == 3
    assert len(keys) == 3
    assert keys[p_gem.id] == "AIza_test"
    _ok("auth.seed_init_script + provider helpers")


def test_compose_images() -> None:
    from PIL import Image
    from studio_test_kit.compose import hstack_images, vstack_images
    tmp = Path(tempfile.mkdtemp(prefix="stk_test_"))
    try:
        left = tmp / "left.png"
        right = tmp / "right.png"
        Image.new("RGB", (320, 240), "red").save(left)
        Image.new("RGB", (200, 240), "blue").save(right)
        out = hstack_images(left, right, tmp / "sxs.png",
                            label_left="L", label_right="R")
        assert out.exists() and out.stat().st_size > 0
        with Image.open(out) as im:
            assert im.width >= 520  # 320 + 24 gap + 200
            assert im.height >= 240 + 56
        v = vstack_images([left, right], tmp / "stack.png")
        assert v.exists() and v.stat().st_size > 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    _ok("compose.hstack_images / vstack_images")


def test_compose_videos() -> None:
    if shutil.which("ffmpeg") is None:
        print("SKIP compose.hstack_videos -- ffmpeg not installed")
        return
    from studio_test_kit.compose import hstack_videos, webm_to_mp4
    tmp = Path(tempfile.mkdtemp(prefix="stk_test_"))
    try:
        a = tmp / "a.webm"
        b = tmp / "b.webm"
        # 1s 320x240 solid color webms via libvpx.
        for path, color in ((a, "red"), (b, "blue")):
            subprocess.run([
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "lavfi", "-i", f"color=c={color}:s=320x240:d=1",
                "-c:v", "libvpx", "-b:v", "200k", str(path),
            ], check=True)
        mp4 = hstack_videos(a, b, tmp / "sxs.mp4")
        assert mp4.exists() and mp4.stat().st_size > 0
        re_mp4 = webm_to_mp4(a, tmp / "a.mp4")
        assert re_mp4.exists() and re_mp4.stat().st_size > 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    _ok("compose.hstack_videos / webm_to_mp4")


def test_lifecycle_shape() -> None:
    from studio_test_kit import lifecycle
    # Smoke: dataclass + helpers exist; we don't run install.sh here.
    inst = lifecycle.StudioInstall(home=Path("/tmp/x"), repo=Path("/tmp/y"), branch="main")
    assert inst.bootstrap_password is None and inst.port is None
    # Internal helpers we want callable:
    assert callable(lifecycle.install_studio)
    assert callable(lifecycle.launch_studio)
    assert callable(lifecycle.stop_studio)
    _ok("lifecycle module shape")


def test_bootstrap_password_prefers_the_file() -> None:
    """A current Studio writes the password to auth/.bootstrap_password and does NOT put it
    in the log, so a log-only read returns None and every authenticated route then answers
    403. Cover both sources, and the reused-home case where neither exists."""
    from studio_test_kit import lifecycle

    with tempfile.TemporaryDirectory() as tmp:
        home = Path(tmp) / "home"
        (home / "auth").mkdir(parents=True)
        log = Path(tmp) / "studio.log"
        log.write_text("starting up, nothing about credentials here\n")

        # No file, no log line: None rather than a hang or a bogus value.
        assert lifecycle._read_bootstrap_password(home, log, time.time() + 1) is None

        # The file is the source a current Studio actually uses.
        (home / "auth" / ".bootstrap_password").write_text("s3cret-from-file\n")
        got = lifecycle._read_bootstrap_password(home, log, time.time() + 5)
        assert got == "s3cret-from-file", got

        # The log stays a fallback for builds that printed it there.
        (home / "auth" / ".bootstrap_password").unlink()
        log.write_text("Bootstrap password: s3cret-from-log\n")
        got = lifecycle._read_bootstrap_password(home, log, time.time() + 5)
        assert got == "s3cret-from-log", got

    _ok("bootstrap password from file, log as fallback")


def test_health_body_rejects_the_spa_shell() -> None:
    """Current Studio has no /healthz route, so with a frontend build its SPA catch-all
    answers it 200 with index.html. Only a JSON object body may count as healthy."""
    import http.server
    import threading

    from studio_test_kit import lifecycle

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            self.send_response(200)
            if self.path == "/api/health":
                body, ctype = json.dumps({"status": "healthy"}).encode(), "application/json"
            elif self.path == "/list":
                body, ctype = b"[1, 2]", "application/json"
            else:
                body, ctype = b"<!doctype html><html><div id=root></div></html>", "text/html"
            self.send_header("Content-Type", ctype)
            self.end_headers()
            self.wfile.write(body)

    srv = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{srv.server_port}"
        assert lifecycle.health_body_ok(f"{base}/api/health") is True
        assert lifecycle.health_body_ok(f"{base}/healthz") is False
        assert lifecycle.health_body_ok(f"{base}/list") is False
    finally:
        srv.shutdown()
        srv.server_close()
    assert lifecycle.health_body_ok("http://127.0.0.1:1/api/health", timeout=0.5) is False
    _ok("health probe needs a JSON body, not the SPA shell")


def test_ui_imports_and_selectors() -> None:
    from studio_test_kit import ui
    # Ensure all the public primitives exist as callables.
    for name in ("open_chat", "pick_model", "set_pill", "send_prompt",
                 "wait_for_stream", "wait_for_image", "wait_for_text",
                 "extract_data_url"):
        assert callable(getattr(ui, name)), name
    # Sanity-check that the source uses form-scoped selectors so we don't
    # regress to clicking the sidebar Search button.
    src = Path(ui.__file__).read_text()
    assert 'form:has(textarea)' in src, "selectors must be form-scoped"
    _ok("ui imports + form-scoped selectors")


def test_flows_imports() -> None:
    from studio_test_kit import flows
    for name in ("multi_turn_chat", "image_generation", "tool_pills",
                 "vision_upload", "FlowResult"):
        assert hasattr(flows, name), name
    _ok("flows imports")


def test_extract_data_url() -> None:
    import asyncio
    from studio_test_kit.ui import extract_data_url
    # 1x1 transparent PNG (smallest valid PNG by hand-built base64).
    tiny = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkAAIAAAoAAv/lxKUAAAAASUVORK5CYII="
    raw = asyncio.run(extract_data_url(tiny))
    assert raw.startswith(b"\x89PNG"), "decoded bytes should be PNG"
    _ok("ui.extract_data_url decodes data URL")


def test_password_regex_separators() -> None:
    """A4: separator must accept :, =, and 'is'; must NOT capture '=' itself."""
    from studio_test_kit.lifecycle import _PW_RE
    cases = {
        "Bootstrap password: UnslothPR5720!": "UnslothPR5720!",
        "Initial password: Hunter2!": "Hunter2!",
        "Generated password = SuperSecret!": "SuperSecret!",
        "bootstrap password is: foobar": "foobar",
        "Generated Password is foobaz": "foobaz",
    }
    for line, expected in cases.items():
        m = _PW_RE.search(line)
        assert m is not None, f"no match for {line!r}"
        got = m.group(1)
        assert got == expected, f"line={line!r}: expected {expected!r}, got {got!r}"
    _ok("lifecycle._PW_RE separator handling")


def test_chronological_screenshot_indices() -> None:
    """A3: multi_turn_chat indices must be sorted in capture order, not
    interleaved (all sent then all done would re-order on filename sort)."""
    import inspect
    from studio_test_kit import flows
    src = inspect.getsource(flows.multi_turn_chat)
    # The fix uses a single running `idx += 1` counter rather than
    # arithmetic offsets keyed off `len(prompts)`.
    assert "idx += 1" in src, "multi_turn_chat must use running idx counter"
    assert "2 + i + len(prompts)" not in src, "old offset-by-len trick still present"
    _ok("flows.multi_turn_chat chronological indices")


def test_pick_model_uses_exact_match() -> None:
    """A6: pick_model must use regex-anchored match, not substring."""
    import inspect
    from studio_test_kit import ui
    src = inspect.getsource(ui.pick_model)
    assert "re.escape" in src, "pick_model must escape the model_id"
    assert ":has-text(" not in src or "get_by_role" in src, \
        "pick_model must not rely on :has-text substring matching"
    _ok("ui.pick_model exact-match selector")


def test_wait_for_image_signature() -> None:
    """A5/C4: signature must use min_decoded_bytes (decoded), not min_size (string len)."""
    import inspect
    from studio_test_kit import ui
    sig = inspect.signature(ui.wait_for_image)
    assert "min_decoded_bytes" in sig.parameters, sig
    assert "mime_prefixes" in sig.parameters, sig
    assert "min_size" not in sig.parameters, "old min_size param leaked"
    _ok("ui.wait_for_image min_decoded_bytes / mime_prefixes")


def test_open_chat_signature_split_timeouts() -> None:
    """A2: launch_studio must expose split password/healthz timeouts."""
    import inspect
    from studio_test_kit.lifecycle import launch_studio
    params = inspect.signature(launch_studio).parameters
    assert "password_timeout_s" in params, params
    assert "healthz_timeout_s" in params, params
    _ok("lifecycle.launch_studio split timeouts")


def test_flow_result_attach_video() -> None:
    from studio_test_kit.flows import FlowResult
    from studio_test_kit.ui import StudioPage
    # Build a StudioPage with synthesized video paths (no live browser).
    sp = StudioPage.__new__(StudioPage)
    sp.page = None  # type: ignore
    sp.context = None  # type: ignore
    sp.base_url = "http://x"
    sp.video_webm = Path("/tmp/fake.webm")
    sp.video_mp4 = Path("/tmp/fake.mp4")
    r = FlowResult(out_dir=Path("/tmp/x"))
    assert r.video_webm is None and r.video_mp4 is None
    r.attach_video(sp)
    assert r.video_webm == Path("/tmp/fake.webm")
    assert r.video_mp4 == Path("/tmp/fake.mp4")
    _ok("flows.FlowResult.attach_video")


def test_visibility_probe_is_falsifiable() -> None:
    import asyncio, inspect

    from studio_test_kit import accounts

    # An empty needle would make the assertion unfailable, which is <the_traps> 3.
    try:
        asyncio.run(accounts.assert_not_visible("http://x", "t", "/p", ""))
    except ValueError:
        pass
    else:
        raise AssertionError("assert_not_visible accepted an empty needle")
    # assert_cannot_reach must warn against collection endpoints: pointed at one it reports a
    # leak that is not there, since per-account collections legitimately 200 for any caller.
    assert "collection" in inspect.getdoc(accounts.assert_cannot_reach)
    _ok("assert_not_visible rejects empty needle; reach-probe scoped to single resources")


def test_secure_launch_polls_for_url() -> None:
    import inspect

    from studio_test_kit import secure_launch

    assert secure_launch.URL_RE.search("x https://a-b-c.trycloudflare.com y")
    src = inspect.getsource(secure_launch.launch)
    assert "start_new_session=True" in src        # detached, survives the caller
    assert "subprocess.DEVNULL" in src            # never a pipe: BrokenPipeError 500s the server
    _ok("secure_launch: url regex, detached, no stdout pipe")


def test_accounts_surface() -> None:
    import inspect

    from studio_test_kit import accounts

    assert accounts.OWNER_ACCOUNT_ID == "owner"
    for name in (
        "list_accounts", "create_account", "reissue_setup_code", "set_account_active",
        "delete_account", "redeem_setup_code", "onboard_account",
        "assert_setup_code_single_use", "assert_cannot_reach", "login_mode",
    ):
        assert inspect.iscoroutinefunction(getattr(accounts, name)), name
    # 403 and 404 are both correct isolation answers; narrowing this default silently turns a
    # "refused" into a failure and a reviewer re-runs chasing a phantom leak.
    assert inspect.signature(accounts.assert_cannot_reach).parameters["allow"].default == (403, 404)
    # Deleting an account can block on in-flight media generation, so it gets a longer default.
    assert inspect.signature(accounts.delete_account).parameters["timeout"].default >= 60.0
    _ok("accounts surface + isolation defaults")


def test_skills_surface() -> None:
    import inspect

    from studio_test_kit import skills

    for name in ("list_skills", "set_skill_enabled", "skill_names", "mention_skill"):
        assert inspect.iscoroutinefunction(getattr(skills, name)), name
    src = inspect.getsource(skills.mention_skill)
    # README pitfall 3: a bare text selector hits the sidebar, so the composer must be scoped.
    assert "form:has(textarea)" in src
    # README pitfall 2: substring matching picks the wrong skill when one name prefixes another.
    assert "get_by_role" in src and "re.escape" in src
    _ok("skills surface + composer-scoped mention selectors")


def test_isolation_example_cleans_up() -> None:
    import inspect

    from studio_test_kit.examples import multi_user_isolation as ex

    src = inspect.getsource(ex.main)
    # Cleanup must survive a leak assertion, else one failure strands accounts and the next run
    # starts from a different login_mode than it asserts.
    assert "finally:" in src and "delete_account" in src
    assert ex.SHARED_COLLECTIONS, "no collections probed means the example passes vacuously"
    # The probe must look for A's marker, not merely call the endpoint.
    assert "assert_not_visible" in src and "marker" in src
    _ok("isolation example: cleanup in finally, non-empty probe set")


TESTS = [
    test_visibility_probe_is_falsifiable,
    test_secure_launch_polls_for_url,
    test_accounts_surface,
    test_skills_surface,
    test_isolation_example_cleans_up,
    test_auth_seed,
    test_compose_images,
    test_compose_videos,
    test_lifecycle_shape,
    test_bootstrap_password_prefers_the_file,
    test_health_body_rejects_the_spa_shell,
    test_ui_imports_and_selectors,
    test_flows_imports,
    test_extract_data_url,
    test_flow_result_attach_video,
    test_password_regex_separators,
    test_chronological_screenshot_indices,
    test_pick_model_uses_exact_match,
    test_wait_for_image_signature,
    test_open_chat_signature_split_timeouts,
]


def main() -> int:
    fails = 0
    for fn in TESTS:
        try:
            fn()
        except Exception as e:
            _fail(fn.__name__, e)
            fails += 1
    print()
    print(f"{len(TESTS) - fails}/{len(TESTS)} self-tests passed")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
