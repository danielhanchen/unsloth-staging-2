"""Reusable building blocks for driving Unsloth Studio end-to-end.

Modules:
  lifecycle  -- Install Studio at a chosen UNSLOTH_STUDIO_HOME from any git
                ref, launch it on a chosen port, discover the bootstrap
                password, wait for /healthz.
  auth       -- Backend JWT login, plus Playwright init scripts that seed
                localStorage with external providers and plaintext API keys
                so the SPA picks them up on first page load.
  ui         -- Playwright Chromium context manager with video recording,
                model picker, composer textarea, pill toggles, send/stop
                waits, and a real wait_for_image (polls DOM for a
                `data:image/png` <img>, not just stop-button absence).
  flows      -- High-level scenarios: multi_turn_chat, image_generation,
                tool_pills (Search / Code), vision_upload.
  accounts   -- Multi-user: owner-side CRUD, two-hop setup-code onboarding,
                and cross-account isolation probes (assert_cannot_reach,
                assert_setup_code_single_use, login_mode).
  skills     -- Agent Skills: list/toggle over /api/skills, name-set diffing
                for base-vs-head, and composer `@`-mentions.
  compose    -- PIL hstack/vstack image composition + ffmpeg hstack video
                side-by-side, for pre/post-PR comparisons.

See `examples/` for runnable scripts and `README.md` for the full flow.
"""

from .lifecycle import StudioInstall, install_studio, launch_studio  # noqa: F401
from .auth import StudioAuth, ProviderSeed, login, seed_init_script  # noqa: F401
from .accounts import (  # noqa: F401
    ManagedAccount,
    assert_cannot_reach,
    assert_setup_code_single_use,
    create_account,
    delete_account,
    list_accounts,
    login_mode,
    onboard_account,
    set_account_active,
)
from .skills import list_skills, mention_skill, set_skill_enabled, skill_names  # noqa: F401
try:  # fast headless Chromium on GPU hosts (scripts/pw_fast.py)
    import os as _os
    import sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    import pw_fast as _pw_fast
    _pw_fast.patch()
except Exception:  # noqa: BLE001
    pass
from .ui import (  # noqa: F401
    StudioPage,
    open_chat,
    pick_model,
    set_pill,
    send_prompt,
    wait_for_stream,
    wait_for_image,
)
