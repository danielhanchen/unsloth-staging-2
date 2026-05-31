"""Tests for tokenless (no GH_TOKEN) llama.cpp release resolution.

These cover the github.com release-redirect fast path that lets the prebuilt
installer resolve the latest upstream release WITHOUT the rate-limited
api.github.com REST surface, so users behind shared / NAT IPs keep getting
prebuilt binaries instead of falling back to a slow source build.
"""

import sys
import urllib.error
from email.message import Message
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
STUDIO_DIR = ROOT / "studio"
if str(STUDIO_DIR) not in sys.path:
    sys.path.insert(0, str(STUDIO_DIR))

import install_llama_prebuilt as MOD  # noqa: E402
from install_llama_prebuilt import HostInfo, PrebuiltFallback  # noqa: E402


def _host(**overrides) -> HostInfo:
    base = dict(
        system = "Darwin",
        machine = "arm64",
        is_windows = False,
        is_linux = False,
        is_macos = True,
        is_x86_64 = False,
        is_arm64 = True,
        nvidia_smi = None,
        driver_cuda_version = None,
        compute_caps = [],
        visible_cuda_devices = None,
        has_physical_nvidia = False,
        has_usable_nvidia = False,
        has_rocm = False,
        rocm_gfx_target = None,
    )
    base.update(overrides)
    return HostInfo(**base)


def _headers(**pairs) -> Message:
    message = Message()
    for key, value in pairs.items():
        message[key] = value
    return message


class _FakeRedirectResponse:
    def __init__(self, headers: Message):
        self._headers = headers

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    @property
    def headers(self) -> Message:
        return self._headers


def _api_403(url: str) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(url, 403, "rate limit exceeded", _headers(), None)


# --- pure helpers -----------------------------------------------------------


def test_upstream_release_download_url_is_deterministic():
    url = MOD.upstream_release_download_url(
        "ggml-org/llama.cpp", "b9999", "llama-b9999-bin-macos-arm64.tar.gz"
    )
    assert url == (
        "https://github.com/ggml-org/llama.cpp/releases/download/"
        "b9999/llama-b9999-bin-macos-arm64.tar.gz"
    )


def test_tag_from_release_location_parses_tag():
    assert (
        MOD._tag_from_release_location(
            "https://github.com/ggml-org/llama.cpp/releases/tag/b9437"
        )
        == "b9437"
    )
    # URL-encoded tag is decoded.
    assert (
        MOD._tag_from_release_location(
            "https://github.com/owner/repo/releases/tag/v1.2.3%2Bcuda"
        )
        == "v1.2.3+cuda"
    )


@pytest.mark.parametrize(
    "location",
    [None, "", "https://github.com/owner/repo/releases", "not a url at all"],
)
def test_tag_from_release_location_rejects_non_tag_urls(location):
    assert MOD._tag_from_release_location(location) is None


# --- redirect resolver ------------------------------------------------------


def test_resolve_latest_tag_via_redirect_reads_location(monkeypatch):
    captured = {}

    def fake_build_opener(*handlers):
        class _Opener:
            def open(self, request, timeout = None):
                captured["url"] = request.full_url
                captured["has_auth"] = "Authorization" in request.headers
                return _FakeRedirectResponse(
                    _headers(
                        Location = "https://github.com/ggml-org/llama.cpp/releases/tag/b9999"
                    )
                )

        return _Opener()

    monkeypatch.setattr(MOD.urllib.request, "build_opener", fake_build_opener)
    tag = MOD._resolve_latest_release_tag_via_redirect("ggml-org/llama.cpp")
    assert tag == "b9999"
    assert captured["url"] == "https://github.com/ggml-org/llama.cpp/releases/latest"
    # The redirect call must never be authenticated (no token leakage).
    assert captured["has_auth"] is False


def test_resolve_latest_tag_via_redirect_handles_httperror_3xx(monkeypatch):
    def fake_build_opener(*handlers):
        class _Opener:
            def open(self, request, timeout = None):
                raise urllib.error.HTTPError(
                    request.full_url,
                    302,
                    "Found",
                    _headers(
                        Location = "https://github.com/ggml-org/llama.cpp/releases/tag/b1234"
                    ),
                    None,
                )

        return _Opener()

    monkeypatch.setattr(MOD.urllib.request, "build_opener", fake_build_opener)
    assert MOD._resolve_latest_release_tag_via_redirect("ggml-org/llama.cpp") == "b1234"


def test_resolve_latest_tag_via_redirect_returns_none_on_failure(monkeypatch):
    def fake_build_opener(*handlers):
        class _Opener:
            def open(self, request, timeout = None):
                raise urllib.error.URLError("connection reset")

        return _Opener()

    monkeypatch.setattr(MOD.urllib.request, "build_opener", fake_build_opener)
    monkeypatch.setattr(MOD, "sleep_backoff", lambda *a, **k: None)
    assert MOD._resolve_latest_release_tag_via_redirect("ggml-org/llama.cpp") is None


# --- latest_upstream_release_tag prefers the redirect -----------------------


def test_latest_upstream_release_tag_prefers_redirect(monkeypatch):
    monkeypatch.setattr(
        MOD, "_resolve_latest_release_tag_via_redirect", lambda repo: "b4321"
    )

    def boom(*a, **k):
        raise AssertionError("fetch_json must not be called when redirect succeeds")

    monkeypatch.setattr(MOD, "fetch_json", boom)
    assert MOD.latest_upstream_release_tag() == "b4321"


def test_latest_upstream_release_tag_falls_back_to_rest(monkeypatch):
    monkeypatch.setattr(
        MOD, "_resolve_latest_release_tag_via_redirect", lambda repo: None
    )
    monkeypatch.setattr(MOD, "fetch_json", lambda url: {"tag_name": "b777"})
    assert MOD.latest_upstream_release_tag() == "b777"


# --- iter_release_payloads_by_time fast path --------------------------------


def test_iter_release_payloads_uses_redirect_for_upstream_latest(monkeypatch):
    monkeypatch.setattr(
        MOD, "_resolve_latest_release_tag_via_redirect", lambda repo: "b9999"
    )

    def boom(*a, **k):
        raise AssertionError("github_releases must not be called on the redirect path")

    monkeypatch.setattr(MOD, "github_releases", boom)

    releases = list(
        MOD.iter_release_payloads_by_time(
            MOD.UPSTREAM_REPO, "", "latest", _host()
        )
    )
    assert len(releases) == 1
    assert releases[0]["tag_name"] == "b9999"
    assert releases[0]["assets"] == []
    assert releases[0]["_unsloth_download_repo"] == MOD.UPSTREAM_REPO


def test_iter_release_payloads_falls_back_to_rest_when_redirect_fails(monkeypatch):
    monkeypatch.setattr(
        MOD, "_resolve_latest_release_tag_via_redirect", lambda repo: None
    )
    sentinel = {"tag_name": "b9000", "assets": []}
    monkeypatch.setattr(MOD, "github_releases", lambda *a, **k: [sentinel])

    releases = list(
        MOD.iter_release_payloads_by_time(
            MOD.UPSTREAM_REPO, "", "latest", _host()
        )
    )
    assert releases == [sentinel]


def test_windows_cuda_host_skips_redirect_fast_path(monkeypatch):
    win_cuda = _host(
        system = "Windows",
        machine = "AMD64",
        is_windows = True,
        is_linux = False,
        is_macos = False,
        is_x86_64 = True,
        is_arm64 = False,
        has_physical_nvidia = True,
        has_usable_nvidia = True,
    )
    assert MOD._needs_dynamic_asset_enumeration(win_cuda) is True
    assert MOD._needs_dynamic_asset_enumeration(_host()) is False

    def boom(repo):
        raise AssertionError("redirect must not be used for Windows CUDA hosts")

    monkeypatch.setattr(MOD, "_resolve_latest_release_tag_via_redirect", boom)
    sentinel = {"tag_name": "b9000", "assets": []}
    monkeypatch.setattr(MOD, "github_releases", lambda *a, **k: [sentinel])

    releases = list(
        MOD.iter_release_payloads_by_time(MOD.UPSTREAM_REPO, "", "latest", win_cuda)
    )
    assert releases == [sentinel]


# --- end-to-end: macOS latest resolves a prebuilt with zero REST calls ------


def test_macos_latest_resolves_prebuilt_without_rest_api(monkeypatch):
    monkeypatch.setattr(
        MOD, "_resolve_latest_release_tag_via_redirect", lambda repo: "b9999"
    )

    # Prove no api.github.com call is needed: every REST entry point explodes.
    def no_rest(*a, **k):
        raise AssertionError("the macOS latest path must not touch the GitHub REST API")

    monkeypatch.setattr(MOD, "github_releases", no_rest)
    monkeypatch.setattr(MOD, "github_release", no_rest)
    monkeypatch.setattr(
        MOD,
        "fetch_json",
        lambda url: (_ for _ in ()).throw(_api_403(url)),
    )

    requested_tag, plans = MOD.resolve_simple_install_release_plans(
        "latest", _host(), "ggml-org/llama.cpp", ""
    )
    assert requested_tag == "latest"
    assert len(plans) == 1
    plan = plans[0]
    assert plan.llama_tag == "b9999"
    assert len(plan.attempts) == 1
    assert plan.attempts[0].url == (
        "https://github.com/ggml-org/llama.cpp/releases/download/"
        "b9999/llama-b9999-bin-macos-arm64.tar.gz"
    )
    assert plan.attempts[0].name == "llama-b9999-bin-macos-arm64.tar.gz"


def test_linux_cpu_latest_resolves_prebuilt_without_rest_api(monkeypatch):
    monkeypatch.setattr(
        MOD, "_resolve_latest_release_tag_via_redirect", lambda repo: "b9999"
    )
    monkeypatch.setattr(
        MOD,
        "github_releases",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no REST")),
    )

    linux_cpu = _host(
        system = "Linux",
        machine = "x86_64",
        is_windows = False,
        is_linux = True,
        is_macos = False,
        is_x86_64 = True,
        is_arm64 = False,
    )
    _requested, plans = MOD.resolve_simple_install_release_plans(
        "latest", linux_cpu, "ggml-org/llama.cpp", ""
    )
    assert plans[0].attempts[0].url == (
        "https://github.com/ggml-org/llama.cpp/releases/download/"
        "b9999/llama-b9999-bin-ubuntu-x64.tar.gz"
    )
