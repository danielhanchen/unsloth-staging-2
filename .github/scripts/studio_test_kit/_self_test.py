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


TESTS = [
    test_auth_seed,
    test_compose_images,
    test_compose_videos,
    test_lifecycle_shape,
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
