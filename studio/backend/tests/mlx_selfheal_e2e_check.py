# SPDX-License-Identifier: AGPL-3.0-only
"""End-to-end validation of the hardened MLX self-heal on a real Apple Silicon host.

Run with `PYTHONPATH=studio/backend python studio/backend/tests/mlx_selfheal_e2e_check.py`
on a macos-14 (Apple Silicon) runner where MLX is genuinely NOT installed, which is
exactly the self-heal trigger condition.

It proves the security hardening in PR unslothai/unsloth#6599 did not break the
feature, and that the security properties hold against a poisoned process env:
  1. the install env is sanitized (secrets + source/cache redirects dropped);
  2. the install command requires pre-built wheels (no sdist build hooks);
  3. the real wheels-only install still self-heals a usable MLX stack.

Exits non-zero on any failed assertion so the CI step fails loudly.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import utils.mlx_repair as mr  # noqa: E402


def main() -> None:
    # The self-heal is Apple-Silicon-gated; this runner must satisfy the gate.
    print("is_apple_silicon:", mr.is_apple_silicon())
    assert mr.is_apple_silicon(), "expected an Apple Silicon (macos-14+) runner"

    # Simulate the trigger: a fresh runner has no MLX stack, so Train/Export
    # would be greyed out and the self-heal should fire.
    before = mr.mlx_stack_available()
    print("mlx_stack_available (before):", before)
    assert not before, "MLX stack unexpectedly already present; not testing the repair path"

    # Simulate a secret-bearing, attacker-poisoned Studio process environment.
    os.environ["STUDIO_FAKE_SECRET"] = "super-secret-token-from-studio-env"
    os.environ["HF_TOKEN"] = "hf_fake_secret_value"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "aws_fake_secret_value"
    os.environ["UV_FIND_LINKS"] = "/tmp/evil-find-links"
    os.environ["UV_DEFAULT_INDEX"] = "file:///tmp/evil-index"
    os.environ["UV_INDEX_URL"] = "https://evil.example/simple"
    os.environ["UV_CACHE_DIR"] = "/tmp/evil-cache"
    os.environ["XDG_CACHE_HOME"] = "/tmp/evil-xdg-cache"

    # (1) The install env forwards only the allowlist: secrets and source/cache
    # redirects never reach a (potentially malicious) build/install hook.
    env = mr._mlx_install_env()
    leaked = [
        k for k in (
            "STUDIO_FAKE_SECRET", "HF_TOKEN", "AWS_SECRET_ACCESS_KEY",
            "UV_FIND_LINKS", "UV_DEFAULT_INDEX", "UV_INDEX_URL",
            "UV_CACHE_DIR", "XDG_CACHE_HOME",
        ) if k in env
    ]
    print("install env keys:", sorted(env))
    assert not leaked, f"these should have been dropped from the install env: {leaked}"
    assert "PATH" in env, "PATH must still be forwarded"

    # (2) The install command requires pre-built wheels for the whole resolution,
    # so a resolver-selected sdist cannot run build-backend code at install time.
    cmd = mr._uv_install_cmd(
        "--upgrade", mr._ONLY_BINARY_ARG, *mr._MLX_REINSTALL_ARGS, *mr.MLX_PACKAGES
    )
    print("install cmd:", cmd)
    assert cmd is not None, "uv not found on the runner"
    assert mr._ONLY_BINARY_ARG in cmd, "install command is missing the wheels-only flag"

    # (3) Run the REAL hardened self-heal. Success means uv resolved and installed
    # mlx/mlx-lm/mlx-vlm from wheels only (--only-binary would have failed on an
    # sdist), so the security hardening did not break the feature.
    print("running attempt_mlx_repair() ...")
    ok = mr.attempt_mlx_repair()
    print("attempt_mlx_repair ->", ok)
    assert ok, "hardened self-heal failed to install a usable MLX stack"

    after = mr.mlx_stack_available()
    print("mlx_stack_available (after):", after)
    assert after, "MLX stack still unavailable after the repair"

    # Report the versions that landed, for evidence in the CI log.
    from importlib.metadata import version as _v
    for name in mr._MLX_PACKAGE_NAMES:
        print(f"installed {name}=={_v(name)}")

    print("SELF-HEAL OK: wheels-only install succeeded and env was sanitized.")


if __name__ == "__main__":
    main()
