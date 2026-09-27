#!/usr/bin/env python3
"""Journey 8c: Unsloth Desktop in-app update, previous published release -> latest. Nightly only.

Not per PR: the updater verifies minisign signatures from the release key and its endpoint is
pinned to `releases/latest/download/latest.json`, so only two REAL published releases can be
tested (N-1 installed, N = GitHub "latest"). Runs on a disposable Linux staging runner (needs
sudo for the .deb and the polkit rule; the AppImage path needs neither).

    python desktop_update.py plan   [--repo unslothai/unsloth] [--format deb|appimage] [--json]
    python desktop_update.py run    [--repo ...] [--format deb|appimage] --out DIR [--from TAG]
    python desktop_update.py workflow [--repo ...]      # print the nightly workflow YAML

`run` steps (evidence in <out>/desktop_update/<step>.{png,facts.json}):
  s01_install_prev   gh release download N-1 asset, install (.deb via apt, AppImage chmod +x)
  s02_first_launch   launch under Xvfb + tauri-driver, wait for the Studio UI (first launch runs
                     the bundled installer), seed user data: rotate the password, create a chat
                     thread named `sr-update-marker` over the API
  s03_check          invoke check_desktop_update -> must report version N
  s04_download       invoke download_desktop_update
  s05_install        invoke install_desktop_update (.deb: pkexec under a polkit rule that allows
                     `ai.unsloth.studio.update` for this user; AppImage: in-place replace)
  s06_relaunch       restart the app, wait for the UI
  s07_verify         installed version == N (dpkg-query / AppImage --version), marker thread and
                     the new password still work
Exit: 0 updated and data kept, 1 a step failed, 2 environment (no sudo / no releases).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

ASSET = {"deb": "Unsloth-Desktop-Ubuntu.deb", "appimage": "Unsloth-Desktop-Linux.AppImage"}
POLKIT_ACTION = "ai.unsloth.studio.update"
POLKIT_RULE = """polkit.addRule(function(action, subject) {
  if (action.id == "%s" && subject.user == "%s") { return polkit.Result.YES; }
});
""" % (POLKIT_ACTION, "%s")
MARKER = "sr-update-marker"
PASSWORD = "UpdateCheck-2026!"


def _gh_json(args):
    r = subprocess.run(["gh", *args], capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args[:3])}: {r.stderr.strip()[:200]}")
    return json.loads(r.stdout)


def releases_with_asset(repo, asset, limit=30):
    """Newest-first tags whose release carries the desktop asset (drafts / pre-desktop skipped)."""
    rel = _gh_json(["release", "list", "-R", repo, "-L", str(limit), "--exclude-drafts",
                    "--json", "tagName,isLatest,publishedAt"])
    out = []
    for r in sorted(rel, key=lambda x: x["publishedAt"], reverse=True):
        assets = _gh_json(["release", "view", r["tagName"], "-R", repo, "--json", "assets"])["assets"]
        if any(a["name"] == asset for a in assets):
            out.append({"tag": r["tagName"], "latest": r.get("isLatest", False)})
        if len(out) >= 2 and any(x["latest"] for x in out):
            break
    return out


def plan(repo, fmt, from_tag=None):
    rel = releases_with_asset(repo, ASSET[fmt])
    latest = next((r["tag"] for r in rel if r["latest"]), rel[0]["tag"] if rel else None)
    if not latest:
        raise RuntimeError("no release carries the desktop asset")
    prev = from_tag or next((r["tag"] for r in rel if r["tag"] != latest), None)
    if not prev:
        raise RuntimeError(f"no earlier release with {ASSET[fmt]} than {latest}")
    return {"repo": repo, "format": fmt, "asset": ASSET[fmt], "from": prev, "to": latest}


def version_of(tag):
    m = re.match(r"v?(\d+\.\d+\.\d+)", tag)
    return m.group(1) if m else tag


# -- run ----------------------------------------------------------------------------------------

def _sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


class Runner:
    def __init__(self, p, out):
        self.p, self.out = p, Path(out) / "desktop_update"
        self.out.mkdir(parents=True, exist_ok=True)
        self.wd = self.drv = self.xvfb = None
        self.bin, self.seed_error = None, None

    def facts(self, sid, **kw):
        (self.out / f"{sid}.facts.json").write_text(json.dumps({"_step": sid, **kw}, indent=1, default=str))

    def shot(self, sid):
        if self.wd:
            try:
                self.wd.screenshot(self.out / f"{sid}.png")
            except Exception:  # noqa: BLE001
                pass

    def install_prev(self):
        dl = self.out / "dl"
        dl.mkdir(exist_ok=True)
        r = _sh(["gh", "release", "download", self.p["from"], "-R", self.p["repo"], "-p", self.p["asset"],
                 "-D", str(dl), "--clobber"], timeout=900)
        if r.returncode:
            raise RuntimeError(f"download failed: {r.stderr[-300:]}")
        f = dl / self.p["asset"]
        if self.p["format"] == "deb":
            r = _sh(["sudo", "apt-get", "install", "-y", str(f)], timeout=900)
            if r.returncode:
                raise RuntimeError(f"apt install failed: {r.stdout[-400:]}{r.stderr[-400:]}")
            rule = Path("/etc/polkit-1/rules.d/49-unsloth-studio-update.rules")
            _sh(["sudo", "tee", str(rule)], input=POLKIT_RULE % os.environ.get("USER", "runner"))
            self.bin = shutil.which("unsloth-studio") or "/usr/bin/unsloth-studio"
        else:
            f.chmod(0o755)
            self.bin = str(f)
        return {"installed": self.p["from"], "bin": self.bin, "version": self.installed_version()}

    def installed_version(self):
        if self.p["format"] == "deb":
            r = _sh(["dpkg-query", "-W", "-f=${Version}", "unsloth-studio"])
            return r.stdout.strip() or None
        return None

    def launch(self):
        from studio_regress.journeys.desktop import WebDriver, _ping, _wait
        disp = ":97"
        self.xvfb = subprocess.Popen(["Xvfb", disp, "-screen", "0", "1440x900x24", "-nolisten", "tcp"])
        os.environ["DISPLAY"] = disp
        self.drv = subprocess.Popen(["tauri-driver", "--port", "4455"],
                                    stdout=(self.out / "tauri-driver.log").open("ab"), stderr=subprocess.STDOUT)
        self.wd = WebDriver("http://127.0.0.1:4455")
        if not _wait(lambda: _ping(self.wd), 30):
            raise RuntimeError("tauri-driver did not start")
        self.wd.start(self.bin, timeout=180)
        ok = _wait(lambda: any(x in (self.wd.execute("return location.pathname") or "") for x in ("/login", "/chat",
                   "/change-password")), 1500, every=3)
        if not ok:
            raise RuntimeError("desktop app never reached the Studio UI (first-launch install?)")
        return self.wd.execute("return location.pathname")

    def close(self):
        if self.wd:
            self.wd.quit()  # needs the driver alive
        for p in (self.drv, self.xvfb):
            if p:
                p.terminate()
                p.wait(timeout=30)
        self.wd = self.drv = self.xvfb = None

    def invoke(self, cmd, timeout=900):
        return self.wd.execute("""const done = arguments[arguments.length - 1];
          window.__TAURI__.core.invoke(arguments[0]).then(v => done({ok: v}), e => done({error: String(e)}));""",
                               [cmd], asynchronous=True, timeout=timeout)

    def api(self, method, path, body=None, token=None):
        import urllib.request
        port = self.wd.execute("return location.port") or "8888"
        req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method=method,
                                     data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json",
                                              **({"Authorization": f"Bearer {token}"} if token else {})})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read() or b"{}")

    def seed(self):
        """Best effort: Desktop provisions its own auth (desktop-login), so a bootstrap file may
        not exist; then only the password path is skipped and `seeded` says why."""
        try:
            return self._seed()
        except Exception as e:  # noqa: BLE001
            self.seed_error = str(e)[:200]
            return None

    def _seed(self):
        boot = Path.home() / ".unsloth/studio/auth/.bootstrap_password"
        tok = None
        if boot.exists():
            old = boot.read_text().strip()
            tok = self.api("POST", "/api/auth/login", {"username": "unsloth", "password": old})["access_token"]
            tok = self.api("POST", "/api/auth/change-password", {"current_password": old, "new_password": PASSWORD},
                           tok).get("access_token") or tok
        tok = tok or self.api("POST", "/api/auth/login", {"username": "unsloth", "password": PASSWORD})["access_token"]
        self.api("POST", "/api/chat/threads", {"id": MARKER, "title": MARKER, "modelType": "base",
                                              "createdAt": int(time.time() * 1000)}, tok)
        return tok

    def verify(self):
        if getattr(self, "seed_error", None):
            return {"marker_kept": None, "seed_error": self.seed_error}
        tok = self.api("POST", "/api/auth/login", {"username": "unsloth", "password": PASSWORD})["access_token"]
        threads = self.api("GET", "/api/chat/threads", None, tok)
        items = threads if isinstance(threads, list) else threads.get("threads", [])
        return {"password_kept": True, "marker_kept": any(t.get("title") == MARKER for t in items)}


def run(p, out):
    r = Runner(p, out)
    steps = [
        ("s01_install_prev", r.install_prev),
        ("s02_first_launch", lambda: {"path": r.launch(), "seeded": bool(r.seed())}),
        ("s03_check", lambda: r.invoke("check_desktop_update", 120)),
        ("s04_download", lambda: r.invoke("download_desktop_update", 1800)),
        ("s05_install", lambda: r.invoke("install_desktop_update", 900)),
        ("s06_relaunch", lambda: (r.close(), time.sleep(5), {"path": r.launch()})[2]),
        ("s07_verify", lambda: {**r.verify(), "installed_version": r.installed_version(),
                                "expected": version_of(p["to"])}),
    ]
    rc = 0
    try:
        for sid, fn in steps:
            t0 = time.perf_counter()
            try:
                got = fn() or {}
                bad = isinstance(got, dict) and got.get("error")
                if sid == "s03_check":
                    ver = ((got.get("ok") or {}) if isinstance(got, dict) else {}).get("version")
                    bad = bad or version_of(str(ver)) != version_of(p["to"])
                if sid == "s07_verify":
                    bad = got.get("marker_kept") is False or (
                        got.get("installed_version") and version_of(got["installed_version"]) != got["expected"])
                r.facts(sid, **got, _status="failed" if bad else "ok", _s=round(time.perf_counter() - t0, 1))
                r.shot(sid)
                if bad:
                    rc = 1
                    break
            except Exception as e:  # noqa: BLE001
                r.facts(sid, _status="failed", _error=str(e)[:800])
                r.shot(sid)
                rc = 1
                break
    finally:
        r.close()
    return rc


WORKFLOW = """name: studio-regress-desktop-update
on:
  workflow_dispatch:
  schedule:
    - cron: "41 6 * * *"
permissions:
  contents: read
jobs:
  update:
    strategy:
      fail-fast: false
      matrix:
        format: [deb, appimage]
    runs-on: ubuntu-22.04
    timeout-minutes: 90
    steps:
      - uses: actions/checkout@v4
      - name: Desktop runtime + WebDriver
        run: |
          sudo apt-get update -qq
          sudo apt-get install -y -qq xvfb webkit2gtk-driver libwebkit2gtk-4.1-0 policykit-1 libfuse2
          cargo install tauri-driver --locked
      - name: Previous release -> latest via the in-app updater
        env:
          GH_TOKEN: ${{ github.token }}
        run: python3 scripts/studio_regress/desktop_update.py run --repo __REPO__ --format ${{ matrix.format }} --out out
      - if: always()
        uses: actions/upload-artifact@v4
        with:
          name: desktop-update-${{ matrix.format }}
          path: out/
"""


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("cmd", choices=("plan", "run", "workflow"))
    p.add_argument("--repo", default="unslothai/unsloth")
    p.add_argument("--format", default="deb", choices=tuple(ASSET))
    p.add_argument("--from", dest="from_tag")
    p.add_argument("--out", default="outputs/studio_regress/desktop_update")
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    if a.cmd == "workflow":
        print(WORKFLOW.replace("__REPO__", a.repo))
        return 0
    try:
        pl = plan(a.repo, a.format, a.from_tag)
    except Exception as e:  # noqa: BLE001
        print(f"desktop_update: {e}", file=sys.stderr)
        return 2
    if a.cmd == "plan":
        print(json.dumps(pl, indent=2) if a.json else f"DESKTOP_UPDATE {pl['from']} -> {pl['to']} ({pl['asset']})")
        return 0
    return run(pl, a.out)


if __name__ == "__main__":
    sys.exit(main())
