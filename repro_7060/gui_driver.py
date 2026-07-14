# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Best-effort end-to-end GUI reproduction of the #7060 loop against a LIVE Studio
server (the released, buggy 2026.7.2 build).

Deterministic part (httpx against the real Studio API):
  1. POST /api/hub/download the Qwen quant, poll to completion.
  2. GET /api/models/gguf-variants -> record update_available (natural layout).
  3. Force the Windows no-symlink cache layout, re-query -> update_available True
     (the #7060 false positive, shown through the live API).
  4. Loop: re-force + re-query -> still True (Update can never clear it).

Best-effort part (Playwright): screenshot the Hub page and any "update available"
badge. Selectors are resilient and every failure is swallowed so we always upload
whatever screenshots we captured. The httpx reproduction is the real evidence.

Nothing here blocks: exit code is always 0; findings go to the JSON + screenshots.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import httpx

BASE = os.environ.get("STUDIO_URL", "http://127.0.0.1:8888")
REPO = os.environ.get("REPRO_REPO", "unsloth/Qwen3.5-2B-MTP-GGUF")
QUANT = os.environ.get("REPRO_QUANT", "UD-Q4_K_XL")
OUTDIR = Path(os.environ.get("REPRO_OUT", "outputs")).resolve()
SHOTS = OUTDIR / "screenshots"
HF_HOME = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))


def _load_repro_helpers():
    here = Path(__file__).resolve().parent
    for cand in (here.parent / "scripts" / "repro_7060.py", here / "repro_7060.py"):
        if cand.exists():
            spec = importlib.util.spec_from_file_location("repro_7060", cand)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    return None


def variant_update_available(client: httpx.Client) -> dict:
    r = client.get(f"{BASE}/api/models/gguf-variants", params={"repo_id": REPO}, timeout=120)
    r.raise_for_status()
    for v in r.json().get("variants", []):
        q = (v.get("quant") or "").lower()
        if q == QUANT.lower() or QUANT.lower() in (v.get("filename") or "").lower():
            return {"downloaded": v.get("downloaded"), "update_available": v.get("update_available"),
                    "partial": v.get("partial")}
    return {"error": "variant not found"}


def wait_health(client: httpx.Client, timeout: int = 180) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if client.get(f"{BASE}/api/health", timeout=10).status_code == 200:
                return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(3)
    return False


def poll_download(client: httpx.Client, timeout: int = 2400) -> str:
    t0 = time.time()
    last = ""
    while time.time() - t0 < timeout:
        try:
            r = client.get(f"{BASE}/api/hub/download-status",
                           params={"repo_id": REPO, "gguf_variant": QUANT}, timeout=30)
            last = r.text[:200]
            st = r.json()
            state = (st.get("state") or st.get("status") or "").lower()
            if state in ("completed", "complete", "done", "success", "ready"):
                return "completed"
            if state in ("error", "failed", "cancelled"):
                return f"failed:{state}"
        except Exception:  # noqa: BLE001
            pass
        time.sleep(5)
    return f"timeout (last={last})"


def screenshots():
    findings = {"screenshots": []}
    try:
        from playwright.sync_api import sync_playwright
    except Exception as e:  # noqa: BLE001
        findings["playwright"] = f"unavailable: {e}"
        return findings
    SHOTS.mkdir(parents=True, exist_ok=True)
    try:
        with sync_playwright() as p:
            b = p.chromium.launch()
            pg = b.new_page(viewport={"width": 1400, "height": 1000})
            for name, url in [("hub", f"{BASE}/hub"), ("models", f"{BASE}/models"), ("root", BASE)]:
                try:
                    pg.goto(url, wait_until="networkidle", timeout=30000)
                    time.sleep(2)
                    # best-effort: type the repo into any search box
                    try:
                        box = pg.query_selector("input[type=search], input[placeholder*=earch], input")
                        if box:
                            box.fill(REPO.split("/")[-1]); time.sleep(2)
                    except Exception:  # noqa: BLE001
                        pass
                    shot = SHOTS / f"{name}.png"
                    pg.screenshot(path=str(shot), full_page=True)
                    findings["screenshots"].append(shot.name)
                except Exception as e:  # noqa: BLE001
                    findings.setdefault("nav_errors", []).append(f"{name}: {e}")
            # capture any element mentioning "update"
            try:
                el = pg.query_selector("text=/update available/i")
                if el:
                    el.screenshot(path=str(SHOTS / "update_badge.png"))
                    findings["screenshots"].append("update_badge.png")
                    findings["badge_text_found"] = True
            except Exception:  # noqa: BLE001
                pass
            b.close()
    except Exception as e:  # noqa: BLE001
        findings["playwright_error"] = str(e)
    return findings


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    report: dict = {"base": BASE, "repo": REPO, "quant": QUANT, "steps": {}}
    repro = _load_repro_helpers()
    with httpx.Client() as client:
        report["healthy"] = wait_health(client)
        if not report["healthy"]:
            report["error"] = "studio never became healthy"
            (OUTDIR / "gui_report.json").write_text(json.dumps(report, indent=2))
            return 0

        report["steps"]["before_download"] = variant_update_available(client)
        try:
            r = client.post(f"{BASE}/api/hub/download",
                            json={"repo_id": REPO, "gguf_variant": QUANT, "use_xet": False}, timeout=60)
            report["steps"]["download_post_status"] = r.status_code
        except Exception as e:  # noqa: BLE001
            report["steps"]["download_post_error"] = str(e)
        report["steps"]["download_result"] = poll_download(client)
        report["steps"]["after_download_natural"] = variant_update_available(client)

        # Force the Windows no-symlink layout, then re-query the LIVE badge.
        if repro is not None:
            repo_dir = repro.repo_cache_dir(HF_HOME, REPO)
            try:
                report["steps"]["force_layout"] = repro.force_no_symlink_layout(repo_dir)
            except Exception as e:  # noqa: BLE001
                report["steps"]["force_layout_error"] = str(e)
            report["steps"]["after_force_nosymlink"] = variant_update_available(client)
            # Loop: simulate an Update by re-forcing (a real re-download would rewrite
            # the same blobless file), then re-query -> still update_available.
            try:
                repro.force_no_symlink_layout(repo_dir)
            except Exception:  # noqa: BLE001
                pass
            report["steps"]["loop_after_update"] = variant_update_available(client)

        report["gui"] = screenshots()

    (OUTDIR / "gui_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
