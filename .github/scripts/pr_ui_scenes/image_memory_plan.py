"""Scene: generating at a resolution the device cannot hold (PR 8224, issues 8188 + 8081).

The memory plan used to be resolution-blind: it sized weights and left activations out, so
a request whose latents and attention buffers could not fit was started anyway and died in
the allocator. This PR estimates the activation cost for the ACTUAL width, height and batch
and refuses first, naming the numbers.

The refusal reads real free device memory, so on a 180 GB card it can never fire. Rather
than stub the reading, `gpu_ballast.py` ALLOCATES the rest of the card before the
run: the guard still measures, the loader still competes for what is left, and both sides
see the same device. At about 14 GiB free, Z-Image at 2048x2048 needs about 29.2 GB of
working memory, which is comfortably over.

1024x1024 succeeds on both sides and is not photographed: the point is not that big is
refused, it is that the refusal replaces a crash.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import urllib.error
from pathlib import Path

WORKSPACE = Path(os.environ.get("WORKSPACE") or os.environ.get("UNSLOTH_WORKSPACE") or Path(__file__).resolve().parents[2])
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "scripts"))

from pr_ui_scenes._common import Session, api_get, api_post  # noqa: E402
from studio_test_kit.auth import seed_init_script  # noqa: E402
from studio_test_kit.ui import open_chat  # noqa: E402

DEFAULT_REPO = "unsloth/Z-Image-Turbo-GGUF"
DEFAULT_FILE = "z-image-turbo-Q4_K_M.gguf"

# sonner first, then the ARIA fallbacks: the toast is the only place the refusal is shown.
_TOAST = "[data-sonner-toast], [role=status], [role=alert]"


def _free_mib(gpu_index: int) -> int:
    import subprocess

    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits",
         "-i", str(gpu_index)],
        text=True, capture_output=True, check=True).stdout
    return int(out.strip().splitlines()[0])


def _load_and_wait(session: Session, body: dict, timeout_s: int = 1200) -> dict:
    api_post(session, "/api/inference/images/unload", {}, timeout=300)
    api_post(session, "/api/inference/images/load", body, timeout=timeout_s)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        time.sleep(5)
        status = api_get(session, "/api/inference/images/status")
        if status.get("loaded"):
            return status
        progress = api_get(session, "/api/inference/images/load-progress")
        if progress.get("error"):
            raise RuntimeError(f"load failed: {str(progress['error'])[:400]}")
    raise RuntimeError(f"model did not load within {timeout_s}s")


async def drive(session: Session, out_dir: Path, label: str,
                repo: str = DEFAULT_REPO, filename: str = DEFAULT_FILE,
                width: int = 2048, height: int = 2048, gpu_index: int = 2,
                **_: object) -> tuple[list[Path], dict]:
    status = _load_and_wait(session, {"model_path": repo, "gguf_filename": filename,
                                      "model_kind": "gguf"})
    facts: dict = {"loaded": status.get("loaded"),
                   "offload_policy": status.get("offload_policy")}
    # Free VRAM at the moment of the attempt, per side. Without it the pair is not a
    # comparison: this box shares GPUs, so the device the AFTER side measures can differ
    # from the one BEFORE ran on, and "one refused, one succeeded" would then say nothing
    # about the PR. Read from nvidia-smi rather than the monitor endpoint, whose payload
    # carries no free figure.
    facts["free_mib"] = _free_mib(gpu_index)

    # The API answer beside the picture: a 400 carrying the arithmetic on the head, and
    # whatever the base does instead. Deliberately not caught as a failure -- which one it
    # is IS the finding.
    try:
        api_post(session, "/api/inference/images/generate",
                 {"prompt": "a tiny ginger sloth", "width": width, "height": height,
                  "batch_size": 1, "steps": 4, "seed": 7}, timeout=1800)
        facts["api"] = {"status": 200, "detail": "generated"}
    except urllib.error.HTTPError as exc:
        facts["api"] = {"status": exc.code, "detail": exc.read().decode()[:700]}
    except Exception as exc:  # noqa: BLE001
        facts["api"] = {"status": "exception", "detail": f"{type(exc).__name__}: {exc}"[:400]}

    shots: list[Path] = []
    init = seed_init_script(
        type("A", (), {"access_token": session.access_token,
                       "refresh_token": session.refresh_token})(), []
    )
    async with open_chat(session.base_url, init_scripts=[init],
                         viewport=(1600, 1000), headless=True) as sp:
        page = sp.page
        await page.goto(f"{session.base_url}/images", wait_until="domcontentloaded")
        await page.wait_for_timeout(8_000)
        # Labelled inputs, not the sliders beside them: Width and Height take a value
        # directly, where a slider only steps.
        for name, value in (("Width", width), ("Height", height)):
            box = page.get_by_label(name, exact=True).first
            await box.fill(str(value))
            await box.press("Tab")
        await page.wait_for_timeout(1_000)
        await page.get_by_role("button", name="Generate", exact=True).first.click()

        toast = page.locator(_TOAST).last
        try:
            await toast.wait_for(state="visible", timeout=600_000)
            await page.wait_for_timeout(1_500)
            facts["toast"] = (await toast.inner_text()).replace("\n", " ")[:600]
            shot = out_dir / f"{label.lower()}_toast.png"
            await toast.screenshot(path=str(shot))
        except Exception as exc:  # noqa: BLE001 -- no toast is itself a result worth seeing
            facts["toast"] = f"none within 600s: {type(exc).__name__}"
            shot = out_dir / f"{label.lower()}_toast.png"
            await sp.screenshot(shot, full_page=False)
        shots.append(shot)

        full = out_dir / f"{label.lower()}_page.png"
        await sp.screenshot(full, full_page=False)
        shots.append(full)

    print(f"[{label}] {json.dumps(facts)[:600]}", flush=True)
    api_post(session, "/api/inference/images/unload", {}, timeout=300)
    return shots, facts


if __name__ == "__main__":
    import argparse

    from pr_ui_scenes._common import studio_session

    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--home", type=Path, required=True)
    ap.add_argument("--password", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--label", default="AFTER")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    s = studio_session(a.url, a.home, a.password)
    print(asyncio.run(drive(s, a.out, a.label)))
