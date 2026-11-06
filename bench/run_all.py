"""One Colab CPU job for the long-chat lag questions and PRs #16852 / #16831.

Stages (each writes JSON under out/ and prints RESULT lines; a failed stage does not stop the rest):
  gate     replay the #16801 CI gate against index.css from 4ce36bc3b5 (before #16333), 65e868dcf5
           (#16333), 6401ceda29, v0.1.914-beta, origin/main and the fixed branch
  bisect   first-parent bisect of 6401ceda29..v0.1.914-beta for the commit that cut typing cost
           (real Studio, 200-turn agent thread, Solid material, Chromium)
  glass    #16852 base vs head vs base again (A/A), Chromium + WebKit, software rendering
  catalog  #16831 /v1/models vs /api/inference/loaded-models on a seeded HF cache

  python run_all.py [stage ...]      (default: all)
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

JOB = Path.cwd()
W = Path(os.environ.get("BENCH_WORK", "/content/lagbench"))
OUT = JOB / "out"
PACK = W / "pack"
REPO = W / "repo"
HOME = W / "studio_home"
PORT = 12611
OUT.mkdir(parents=True, exist_ok=True)
W.mkdir(parents=True, exist_ok=True)
T0 = time.time()


def log(msg: str) -> None:
    line = f"[{time.time() - T0:7.0f}s] {msg}"
    print(line, flush=True)
    with open(OUT / "driver.log", "a") as fh:
        fh.write(line + "\n")


def result(stage: str, payload: dict) -> None:
    print("RESULT", stage, json.dumps(payload), flush=True)
    with open(OUT / "results.jsonl", "a") as fh:
        fh.write(json.dumps({"stage": stage, **payload}) + "\n")


def sh(cmd: str, name: str, cwd=None, env=None, check=True, timeout=None, echo: str | None = None) -> int:
    logf = OUT / "logs" / f"{name}.log"
    logf.parent.mkdir(exist_ok=True)
    with open(logf, "a") as fh:
        if echo is None:
            p = subprocess.run(["bash", "-lc", cmd], cwd=cwd, env=env, stdout=fh, stderr=subprocess.STDOUT,
                               timeout=timeout)
        else:
            # Stream matching lines so the remote idle watchdog sees progress on long runs.
            proc = subprocess.Popen(["bash", "-lc", cmd], cwd=cwd, env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, start_new_session=True)
            import threading
            killer = threading.Timer(timeout or 1e9, lambda: os.killpg(proc.pid, signal.SIGKILL))
            killer.daemon = True
            killer.start()
            for line in proc.stdout:
                fh.write(line)
                fh.flush()
                if line.startswith(echo):
                    print(line.rstrip()[:4000], flush=True)
            proc.wait()
            killer.cancel()
            p = proc
    if check and p.returncode != 0:
        tail = logf.read_text(errors="replace")[-1500:]
        raise RuntimeError(f"{name} failed rc={p.returncode}: ...{tail}")
    return p.returncode


def git(*args, check=True) -> str:
    p = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)
    if check and p.returncode != 0:
        raise RuntimeError(f"git {args}: {p.stderr[-500:]}")
    return p.stdout.strip()


# ---------------------------------------------------------------- setup
def setup() -> None:
    if not PACK.exists():
        with tarfile.open(sorted(JOB.glob("pack*.tgz"))[0]) as tf:
            tf.extractall(W)
    node_ok = False
    try:
        v = subprocess.run(["node", "-v"], capture_output=True, text=True).stdout.strip()
        node_ok = int(v.lstrip("v").split(".")[0]) >= 20
    except Exception:  # noqa: BLE001
        pass
    if not node_ok:
        sh(f"cd {W} && curl -fsSL https://nodejs.org/dist/v22.11.0/node-v22.11.0-linux-x64.tar.xz | tar -xJ "
           f"&& ln -sfn {W}/node-v22.11.0-linux-x64 {W}/node", "node")
        os.environ["PATH"] = f"{W}/node/bin:" + os.environ["PATH"]
    log("node " + subprocess.run(["node", "-v"], capture_output=True, text=True).stdout.strip())
    sh(f"{sys.executable} -m pip install -q playwright httpx pillow huggingface_hub", "pip")
    for attempt in range(3):  # apt mirrors on hosted runners occasionally hang; retry instead of failing
        try:
            sh(f"{sys.executable} -m playwright install --with-deps chromium webkit", "pw_install", timeout=900)
            break
        except Exception as exc:  # noqa: BLE001
            log(f"playwright install attempt {attempt} failed: {str(exc)[:200]}")
            if attempt == 2:
                raise
    if not REPO.exists():
        sh(f"git clone -q --filter=blob:none https://github.com/unslothai/unsloth {REPO}", "clone")
    sh("git fetch -q origin pull/16852/head:pr16852 pull/16831/head:pr16831 "
       "main:gate --tags --force", "fetch", cwd=REPO)
    os.environ["PACK"] = str(PACK)
    os.environ["WORKSPACE"] = str(PACK)
    log("setup done")


def node_modules(fe: Path) -> None:
    h = hashlib.sha1((fe / "package-lock.json").read_bytes()).hexdigest()[:12]
    nm = W / f"nm_{h}"
    if not (nm / "node_modules").exists():
        nm.mkdir(exist_ok=True)
        shutil.copy(fe / "package.json", nm)
        shutil.copy(fe / "package-lock.json", nm)
        sh("npm ci --no-audit --no-fund --loglevel=error", f"npmci_{h}", cwd=nm, timeout=1800)
    link = fe / "node_modules"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(nm / "node_modules")


def worktree(ref: str, name: str) -> Path:
    wt = W / f"wt_{name}"
    if wt.exists():
        git("worktree", "remove", "--force", str(wt), check=False)
        shutil.rmtree(wt, ignore_errors=True)
    git("worktree", "add", "-q", "--detach", str(wt), ref)
    return wt


def build_app(ref: str, name: str) -> Path | None:
    dist = W / f"dist_{name}"
    if (dist / "index.html").exists():
        return dist
    wt = worktree(ref, name)
    fe = wt / "studio" / "frontend"
    try:
        node_modules(fe)
        sh(f"npx vite build --outDir {dist} --emptyOutDir", f"build_{name}", cwd=fe, timeout=1800)
    except Exception as exc:  # noqa: BLE001
        log(f"build {name} ({ref}) failed: {exc}"[:600])
        return None
    finally:
        (fe / "node_modules").unlink(missing_ok=True)
        git("worktree", "remove", "--force", str(wt), check=False)
    return dist


# ---------------------------------------------------------------- studio
_proc: subprocess.Popen | None = None


def install_studio() -> None:
    if (HOME / "bin" / "unsloth").exists():
        return
    wt = worktree("origin/main", "install")
    env = dict(os.environ, UNSLOTH_STUDIO_HOME=str(HOME), UNSLOTH_NO_TORCH="1", UNSLOTH_SKIP_AUTOSTART="1")
    sh("bash install.sh --local --no-torch", "studio_install", cwd=wt, env=env, timeout=3600)
    log("studio installed")


def launch(dist: Path, hf_home: Path) -> str:
    global _proc
    stop()
    hf_home.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, UNSLOTH_STUDIO_HOME=str(HOME), UNSLOTH_DISABLE_AUTO_UPDATES="1",
               HF_HOME=str(hf_home), HF_HUB_CACHE=str(hf_home / "hub"))
    logf = open(OUT / "logs" / "studio.log", "a")
    _proc = subprocess.Popen([str(HOME / "bin" / "unsloth"), "studio", "-p", str(PORT), "--frontend", str(dist)],
                             env=env, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
    url = f"http://127.0.0.1:{PORT}"
    for _ in range(150):
        try:
            urllib.request.urlopen(url + "/api/health", timeout=3)
            return url
        except Exception:  # noqa: BLE001
            time.sleep(2)
    raise RuntimeError("studio did not come up")


def stop() -> None:
    global _proc
    if _proc is not None:
        try:
            os.killpg(_proc.pid, signal.SIGTERM)
            _proc.wait(timeout=30)
        except Exception:  # noqa: BLE001
            try:
                os.killpg(_proc.pid, signal.SIGKILL)
            except Exception:  # noqa: BLE001
                pass
        _proc = None
    subprocess.run(["pkill", "-f", f"[u]nsloth studio -p {PORT}"])
    time.sleep(2)


# ---------------------------------------------------------------- stages
GATE_REFS = [("pre16333", "4ce36bc3b5"), ("at16333", "65e868dcf5"), ("at16442", "6401ceda29"),
             ("v914", "v0.1.914-beta"), ("main", "origin/main"), ("fixed", "gate")]


def stage_gate() -> None:
    wt = worktree("gate", "gate")
    fe = wt / "studio" / "frontend"
    override = PACK / "gate_override"
    if override.exists():  # unpushed gate changes under test
        for f in override.rglob("*"):
            if f.is_file():
                dst = wt / f.relative_to(override)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(f, dst)
        log("gate override applied: " + ", ".join(str(f.relative_to(override)) for f in override.rglob("*") if f.is_file()))
    node_modules(fe)
    own_css = (fe / "src" / "index.css").read_text()
    for label, ref in GATE_REFS:
        css = own_css if ref == "gate" else git("show", f"{ref}:studio/frontend/src/index.css")
        (fe / "src" / "index.css").write_text(css)
        dist = W / f"gate_{label}"
        try:
            sh("npx vite build --config vite.long-thread.config.ts", f"gate_build_{label}", cwd=fe, timeout=1800)
            shutil.rmtree(dist, ignore_errors=True)
            shutil.move(str(wt / ".playwright-cli" / "long-thread-production"), str(dist))
        except Exception as exc:  # noqa: BLE001
            result("gate", {"label": label, "ref": ref, "error": str(exc)[:300]})
            continue
        t = time.time()
        env = dict(os.environ, LONG_THREAD_DIST=str(dist), LONG_THREAD_REPEATS="2")
        logf = OUT / "logs" / f"gate_run_{label}.log"
        with open(logf, "w") as fh:
            rc = subprocess.run([sys.executable, "tests/studio/playwright_long_thread_gate.py"], cwd=wt, env=env,
                                stdout=fh, stderr=subprocess.STDOUT, timeout=3600).returncode
        lines = [ln for ln in logf.read_text().splitlines() if not ln.startswith("{")]
        result("gate", {"label": label, "ref": ref, "rc": rc, "verdict": "PASS" if rc == 0 else "FAIL",
                        "seconds": round(time.time() - t), "lines": lines[-6:]})
    (fe / "src" / "index.css").write_text(own_css)
    (fe / "node_modules").unlink(missing_ok=True)


def stage_probe() -> None:
    wt = worktree("gate", "gate")
    fe = wt / "studio" / "frontend"
    override = PACK / "gate_override"
    if override.exists():
        for f in override.rglob("*"):
            if f.is_file():
                dst = wt / f.relative_to(override)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(f, dst)
    node_modules(fe)
    own_css = (fe / "src" / "index.css").read_text()
    dists = []
    for label, ref in (("main", "origin/main"), ("v914", "v0.1.914-beta"), ("fixed", "gate")):
        css = own_css if ref == "gate" else git("show", f"{ref}:studio/frontend/src/index.css")
        (fe / "src" / "index.css").write_text(css)
        dist = W / f"probe_{label}"
        sh("npx vite build --config vite.long-thread.config.ts", f"probe_build_{label}", cwd=fe, timeout=1800)
        shutil.rmtree(dist, ignore_errors=True)
        shutil.move(str(wt / ".playwright-cli" / "long-thread-production"), str(dist))
        dists.append(f"{label}={dist}")
    (fe / "src" / "index.css").write_text(own_css)
    sh(f"{sys.executable} -u {PACK}/rule_probe.py --gate-dir {wt}/tests/studio --turns 100 " + " ".join(dists),
       "rule_probe", cwd=OUT, timeout=5400)
    for line in (OUT / "logs" / "rule_probe.log").read_text().splitlines():
        if line.startswith("PROBE"):
            result("probe", {"line": line[6:]})


SCROLL_ARMS = [("main", "bench/main"), ("p16854", "bench/16854"), ("p16782", "bench/16782-on-main"),
               ("both", "bench/16782-16854")]


def stage_scroll() -> None:
    sh("git fetch -q https://github.com/danielhanchen/unsloth-staging-2 "
       + " ".join(f"{r}:{r}" for _, r in SCROLL_ARMS) + " --force", "fetch_bench", cwd=REPO)
    dists = {}
    for label, ref in SCROLL_ARMS:
        d = build_app(ref, "s_" + label)
        if d is None:
            result("scroll", {"arm": label, "error": "build failed"})
        else:
            dists[label] = d
    if "main" not in dists:
        return
    seed_out = W / "thread_scroll.json"
    url = launch(dists["main"], W / "hf_empty")
    try:
        if not seed_out.exists():
            sh(f"{sys.executable} -u {PACK}/scripts/i16603_bench.py seed --base-url {url} --home {HOME} "
               f"--turns 300 --out {seed_out}", "seed_scroll", timeout=3600)
    finally:
        stop()
    tid = json.loads(seed_out.read_text())["thread"]
    order = list(dists)
    for rep in range(int(os.environ.get("SCROLL_REPS", "3"))):
        for arm in order[rep % len(order):] + order[: rep % len(order)]:
            try:
                url = launch(dists[arm], W / "hf_empty")
                out = OUT / f"scroll_{arm}_{rep}.json"
                sh(f"{sys.executable} -u {PACK}/scroll_bench.py --base-url {url} --home {HOME} --thread {tid} "
                   f"--arm {arm} --rep {rep} --expected 600 --out {out}", f"scroll_{arm}_{rep}", timeout=2400)
                for eng, d in json.loads(out.read_text())["engines"].items():
                    result("scroll", {"arm": arm, "rep": rep, **d})
            except Exception as exc:  # noqa: BLE001
                result("scroll", {"arm": arm, "rep": rep, "error": str(exc)[:300]})
            finally:
                stop()


def _assets(html: str) -> list[str]:
    import re
    return sorted(set(re.findall(r"/assets/[A-Za-z0-9_.-]+\.(?:js|css)", html)))


def record_build(label: str, ref: str, dist: Path, url: str) -> None:
    """Prove the arm served its own build: ref SHA, dist assets, assets the running server returned."""
    built = _assets((dist / "index.html").read_text())
    try:
        served = _assets(urllib.request.urlopen(url + "/", timeout=10).read().decode("utf-8", "replace"))
    except Exception as exc:  # noqa: BLE001
        served = [f"error: {exc}"]
    result("menus_build", {"arm": label, "ref": ref, "sha": git("rev-parse", ref, check=False),
                           "built_assets": built, "served_assets": served, "match": built == served})


def stage_menus() -> None:
    spec = os.environ.get("MENU_ARM_REFS")
    if spec:  # label=ref,label=ref (refs on danielhanchen/unsloth-staging-2)
        arms = [tuple(x.split("=", 1)) for x in spec.split(",")]
    else:
        want = os.environ.get("MENU_ARMS", "main,p16854").split(",")
        arms = [(lbl, ref) for lbl, ref in SCROLL_ARMS if lbl in want]
    sh("git fetch -q https://github.com/danielhanchen/unsloth-staging-2 "
       + " ".join(f"{r}:{r}" for _, r in arms) + " --force", "fetch_bench", cwd=REPO)
    dists = {}
    for label, ref in arms:
        d = build_app(ref, "s_" + label)
        if d is None:
            result("menus", {"arm": label, "error": "build failed"})
        else:
            dists[label] = d
    if not dists:
        return
    threads = W / "threads_menus.json"
    url = launch(next(iter(dists.values())), W / "hf_empty")
    try:
        if not threads.exists():
            sh(f"{sys.executable} -u {PACK}/menu_bench.py seed --base-url {url} --home {HOME} "
               f"--turns {os.environ.get('MENU_TURNS', '1 150')} --out {threads}", "seed_menus", timeout=3600)
    finally:
        stop()
    result("menus_threads", json.loads(threads.read_text()))
    for rep in range(int(os.environ.get("MENU_REPS", "1"))):
        for arm in dists:
            try:
                url = launch(dists[arm], W / "hf_empty")
                if rep == 0:
                    record_build(arm, dict(arms)[arm], dists[arm], url)
                out = OUT / f"menus_{arm}_{rep}.json"
                shots = f"--shots {OUT / 'shots'}" if rep == 0 else ""
                sh(f"{sys.executable} -u {PACK}/menu_bench.py run --base-url {url} --home {HOME} "
                   f"--threads {threads} --arm {arm} --engines {os.environ.get('MENU_ENGINES', 'chromium,webkit,webkit_mac')} "
                   f"{shots} --out {out}", f"menus_{arm}_{rep}", timeout=2700, echo="MENU")
                result("menus", {"arm": arm, "rep": rep, "file": out.name})
                if os.environ.get("MENU_GLASS", "1") == "1":
                    out = OUT / f"menus_{arm}_glass_{rep}.json"
                    sh(f"{sys.executable} -u {PACK}/menu_bench.py run --base-url {url} --home {HOME} "
                       f"--threads {threads} --arm {arm}_glass --material glass "
                       f"--engines {os.environ.get('MENU_ENGINES', 'chromium,webkit,webkit_mac')} "
                       f"--out {out}", f"menus_{arm}_glass_{rep}", timeout=2700, echo="MENU")
                    result("menus", {"arm": f"{arm}_glass", "rep": rep, "file": out.name})
            except Exception as exc:  # noqa: BLE001
                result("menus", {"arm": arm, "rep": rep, "error": str(exc)[:300]})
            finally:
                stop()


def measure_typing(ref: str, tid: str, reps: int = 1) -> float | None:
    name = "b_" + ref.replace("/", "_")[:20]
    dist = build_app(ref, name)
    if dist is None:
        return None
    url = launch(dist, W / "hf_empty")
    out = OUT / f"bisect_{name}.json"
    try:
        sh(f"{sys.executable} -u {PACK}/scripts/i16603_bench.py measure --base-url {url} --home {HOME} "
           f"--thread {tid} --reps {reps} --label {name} --out {out}", f"measure_{name}", timeout=3600)
        s = json.loads(out.read_text())["summary"][name]
    except Exception as exc:  # noqa: BLE001
        log(f"measure {ref} failed: {exc}"[:400])
        return None
    finally:
        stop()
    val = s["type_style_ms_per_key"]
    result("bisect_point", {"ref": ref, "sha": git("rev-parse", "--short", ref),
                            "subject": git("log", "-1", "--format=%s", ref)[:120],
                            "type_style_ms_per_key": val, "burst_s": s.get("type_burst_s"),
                            "nodes": (s.get("state") or {}).get("nodes")})
    return val


def stage_bisect() -> None:
    dist = build_app("origin/main", "main")
    url = launch(dist, W / "hf_empty")
    seed_out = W / "thread.json"
    try:
        if not seed_out.exists():
            sh(f"{sys.executable} -u {PACK}/scripts/i16603_bench.py seed --base-url {url} --home {HOME} "
               f"--turns 200 --out {seed_out}", "seed", timeout=3600)
    finally:
        stop()
    tid = json.loads(seed_out.read_text())["thread"]
    slow_ref, fast_ref = "6401ceda29", "v0.1.914-beta"
    a = measure_typing(slow_ref, tid)
    b = measure_typing(fast_ref, tid)
    if not a or not b or a < 2 * b:
        result("bisect", {"status": "no step between anchors", "slow": a, "fast": b})
        return
    thr = math.sqrt(a * b)
    commits = git("rev-list", "--first-parent", "--reverse", f"{slow_ref}..{fast_ref}", "--",
                  "studio/frontend").split()
    tip = git("rev-parse", fast_ref)
    if not commits or commits[-1] != tip:
        commits.append(tip)
    lo, hi = -1, len(commits) - 1  # commits[lo] slow (anchor), commits[hi] fast
    while hi - lo > 1:
        mid = (lo + hi) // 2
        v = measure_typing(commits[mid], tid)
        probe = mid
        while v is None and probe + 1 < hi:
            probe += 1
            v = measure_typing(commits[probe], tid)
        if v is None:
            break
        if v < thr:
            hi = probe
        else:
            lo = probe
    first = commits[hi]
    result("bisect", {"status": "done", "threshold": round(thr, 1), "first_fast": git("rev-parse", "--short", first),
                      "subject": git("log", "-1", "--format=%s", first)[:160],
                      "last_slow": git("rev-parse", "--short", commits[lo]) if lo >= 0 else slow_ref})


def stage_glass() -> None:
    from PIL import Image
    import random
    img = W / "bg.png"
    if not img.exists():
        random.seed(16852)
        im = Image.new("RGB", (1920, 1080))
        px = im.load()
        for y in range(1080):
            for x in range(0, 1920):
                px[x, y] = ((x * 255) // 1920, (y * 255) // 1080, (x * y + random.randint(0, 60)) % 256)
        im.save(img)
    base = build_app("7b7ebff5d3", "g_base")
    head = build_app("pr16852", "g_head")
    if base is None or head is None:
        result("glass", {"error": "build failed"})
        return
    arms = {"base": base, "head": head, "baseA": base}
    order = ["base", "head", "baseA"]
    for rep in range(3):
        for arm in order[rep:] + order[:rep]:
            try:
                url = launch(arms[arm], W / "hf_empty")
                out = OUT / f"glass_{arm}_{rep}.json"
                sh(f"{sys.executable} -u {PACK}/glass_bench.py --base-url {url} --home {HOME} --arm {arm} "
                   f"--rep {rep} --image {img} --out {out}", f"glass_{arm}_{rep}", timeout=2400)
                r = json.loads(out.read_text())
                for eng, d in r["engines"].items():
                    result("glass", {"arm": arm, "rep": rep, "engine": eng, **d.get("summary", {}),
                                     "state": d.get("state"), "err": d.get("error")})
            except Exception as exc:  # noqa: BLE001
                result("glass", {"arm": arm, "rep": rep, "error": str(exc)[:300]})
            finally:
                stop()


def glass_image() -> Path:
    from PIL import Image
    import random
    img = W / "bg.png"
    if not img.exists():
        random.seed(16852)
        im = Image.new("RGB", (1920, 1080))
        px = im.load()
        for y in range(1080):
            for x in range(0, 1920):
                px[x, y] = ((x * 255) // 1920, (y * 255) // 1080, (x * y + random.randint(0, 60)) % 256)
        im.save(img)
    return img


def build_patched(name: str, patch: Path, ref: str = "origin/main") -> Path | None:
    dist = W / f"dist_{name}"
    if (dist / "index.html").exists():
        return dist
    wt = worktree(ref, name)
    fe = wt / "studio" / "frontend"
    try:
        sh(f"git apply {patch}", f"apply_{name}", cwd=wt)
        node_modules(fe)
        sh(f"npx vite build --outDir {dist} --emptyOutDir", f"build_{name}", cwd=fe, timeout=1800)
    except Exception as exc:  # noqa: BLE001
        log(f"build {name} failed: {exc}"[:600])
        return None
    finally:
        (fe / "node_modules").unlink(missing_ok=True)
        git("worktree", "remove", "--force", str(wt), check=False)
    return dist


def stage_glassdiag() -> None:
    """Glass attribution: arms = (build, material, override CSS) from PACK/glassdiag/arms.json."""
    img = glass_image()
    spec = json.loads((PACK / "glassdiag" / "arms.json").read_text())
    base = os.environ.get("GLASSDIAG_BASE", "origin/main")  # optional pin; default unchanged
    dists = {"main": build_app(base, "gd_main")}
    patch = PACK / "glassdiag" / "fix.patch"
    if patch.exists() and any(a["build"] == "fix" for a in spec["arms"]):
        dists["fix"] = build_patched("gd_fix", patch, base)
    builds = ["main", "fix"]
    for b in dict.fromkeys(a["build"] for a in spec["arms"]):  # optional extra builds: PACK/glassdiag/<build>.patch
        if b not in builds and (PACK / "glassdiag" / f"{b}.patch").exists():
            dists[b] = build_patched(f"gd_{b}", PACK / "glassdiag" / f"{b}.patch", base)
            builds.append(b)
    shots = OUT / "shots"
    for rep in range(int(spec.get("reps", 2))):
        arms = spec["arms"][rep % len(spec["arms"]):] + spec["arms"][: rep % len(spec["arms"])]
        for build in builds:
            mine = [a for a in arms if a["build"] == build]
            if not mine or dists.get(build) is None:
                continue
            try:
                url = launch(dists[build], W / "hf_empty")
                for a in mine:
                    out = OUT / f"gd_{a['label']}_{rep}.json"
                    css = f"--css-file {PACK / 'glassdiag' / a['css']}" if a.get("css") else ""
                    shot = f"--shots {shots}" if rep == 0 else ""
                    try:
                        sh(f"{sys.executable} -u {PACK}/glass_bench.py --base-url {url} --home {HOME} "
                           f"--arm {a['label']} --rep {rep} --image {img} --material {a.get('material', 'glass')} "
                           f"{css} {shot} {a.get('args', '')} --out {out}", f"gd_{a['label']}_{rep}", timeout=1500, echo="GLASS")
                        r = json.loads(out.read_text())
                        for eng, d in r["engines"].items():
                            result("glassdiag", {"arm": a["label"], "rep": rep, "engine": eng, **d.get("summary", {}),
                                                 "state": d.get("state"), "err": d.get("error")})
                    except Exception as exc:  # noqa: BLE001
                        result("glassdiag", {"arm": a["label"], "rep": rep, "error": str(exc)[:300]})
            finally:
                stop()
    shrink_shots(shots)


def shrink_shots(shots: Path) -> None:
    """Pixel-diff paired arms on the full PNGs, then keep JPEGs so the artifact fits the 5 MB cap."""
    if not shots.is_dir():
        return
    try:
        from PIL import Image, ImageChops
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pillow"], check=False)
        from PIL import Image, ImageChops
    pngs = {p.name: p for p in shots.glob("*.png")}
    try:  # optional extra pairs: an arm with "diff_ref" is diffed against that arm
        extra = tuple((x["diff_ref"], x["label"]) for x in json.loads((PACK / "glassdiag" / "arms.json").read_text())["arms"]
                      if x.get("diff_ref"))
    except Exception:  # noqa: BLE001
        extra = ()
    for a, b in (("glass", "fix"), ("solid", "fix_solid")) + extra:
        for name, p in sorted(pngs.items()):
            if not name.startswith(f"{a}_r0_"):
                continue
            q = pngs.get(name.replace(f"{a}_r0_", f"{b}_r0_", 1))
            if q is None:
                continue
            x, y = Image.open(p).convert("RGB"), Image.open(q).convert("RGB")
            if x.size != y.size:
                result("glassshot", {"pair": f"{a}/{b}", "shot": name, "error": f"size {x.size} vs {y.size}"})
                continue
            d = ImageChops.difference(x, y).convert("L")
            hist = d.histogram()
            total = x.size[0] * x.size[1]
            result("glassshot", {"pair": f"{a}/{b}", "shot": name[len(a) + 4:],
                                 "px_changed": total - hist[0], "px_over8": sum(hist[9:]),
                                 "px_over32": sum(hist[33:]), "max": max(i for i, c in enumerate(hist) if c) if total - hist[0] else 0,
                                 "total": total})
    for p in pngs.values():
        im = Image.open(p).convert("RGB")
        im.thumbnail((1400, 1400))
        im.save(p.with_suffix(".jpg"), quality=80)
        p.unlink()


def seed_catalog(hf: Path) -> dict:
    from huggingface_hub import snapshot_download
    hub = hf / "hub"
    hub.mkdir(parents=True, exist_ok=True)
    got = {}
    for repo, pat in (("unsloth/gemma-3-270m-it-GGUF", ["*Q8_0.gguf", "*.json"]),
                      ("unsloth/Qwen3-0.6B-GGUF", ["*Q4_K_M.gguf", "*.json"]),
                      ("HuggingFaceTB/SmolLM2-135M-Instruct", ["*.json", "*.safetensors", "*.txt"])):
        got[repo] = snapshot_download(repo, allow_patterns=pat, cache_dir=str(hub))
    gguf = next(Path(got["unsloth/gemma-3-270m-it-GGUF"]).glob("*.gguf")).resolve()
    st_dir = Path(got["HuggingFaceTB/SmolLM2-135M-Instruct"])
    n_fake = 0
    for kind, count in (("gguf", 60), ("st", 40)):
        for i in range(count):
            rd = hub / f"models--benchfake--{kind}-model-{i}"
            rev = hashlib.sha1(f"{kind}{i}".encode()).hexdigest()
            snap = rd / "snapshots" / rev
            blobs = rd / "blobs"
            if snap.exists():
                n_fake += 1
                continue
            snap.mkdir(parents=True)
            blobs.mkdir(parents=True, exist_ok=True)
            (rd / "refs").mkdir(exist_ok=True)
            (rd / "refs" / "main").write_text(rev)
            files = ({f"benchfake-{kind}-{i}-Q8_0.gguf": gguf} if kind == "gguf"
                     else {p.name: p.resolve() for p in st_dir.iterdir()})
            for fname, src in files.items():
                bh = hashlib.sha1(f"{kind}{i}{fname}".encode()).hexdigest()
                blob = blobs / bh
                if src.stat().st_size > 1 << 20:
                    os.link(src, blob)
                else:
                    shutil.copy(src, blob)
                (snap / fname).symlink_to(os.path.relpath(blob, snap))
            n_fake += 1
    return {"real": list(got), "fake": n_fake}


def stage_catalog() -> None:
    hf = W / "hf_seed"
    seeded = seed_catalog(hf)
    dist = build_app("origin/main", "main")
    try:
        url = launch(dist, hf)
        out = OUT / "catalog.json"
        sh(f"{sys.executable} -u {PACK}/catalog_bench.py --base-url {url} --home {HOME} --rounds 4 "
           f"--load unsloth/gemma-3-270m-it-GGUF:Q8_0 --out {out}", "catalog", timeout=3600)
        r = json.loads(out.read_text())
        result("catalog", {"seeded": seeded, **{k: r.get(k) for k in (
            "catalog_rows", "first_v1", "cold_v1", "warm_v1", "loaded_models", "rows_equal", "with_model",
            "load_error")}})
    except Exception as exc:  # noqa: BLE001
        result("catalog", {"error": str(exc)[:400]})
    finally:
        stop()



def build_variant(ref: str, name: str, patch: Path | None) -> Path | None:
    """build_app with an optional harness-side ablation patch (never pushed anywhere)."""
    if patch is None:
        return build_app(ref, name)
    dist = W / f"dist_{name}"
    if (dist / "index.html").exists():
        return dist
    wt = worktree(ref, name)
    fe = wt / "studio" / "frontend"
    try:
        sh(f"git apply {patch}", f"apply_{name}", cwd=wt)
        node_modules(fe)
        sh(f"npx vite build --outDir {dist} --emptyOutDir", f"build_{name}", cwd=fe, timeout=1800)
    except Exception as exc:  # noqa: BLE001
        log(f"build {name} failed: {exc}"[:600])
        return None
    finally:
        (fe / "node_modules").unlink(missing_ok=True)
        git("worktree", "remove", "--force", str(wt), check=False)
    return dist


def stage_load() -> None:
    """Long-thread load time, arms interleaved per round: LOAD_ARMS=label=ref[+patchfile],..."""
    bench = Path(__file__).resolve().parent
    arms = []
    for spec in os.environ["LOAD_ARMS"].split(","):
        label, rest = spec.split("=", 1)
        ref, _, patch = rest.partition("+")
        arms.append((label, ref, bench / "patches" / patch if patch else None))
    refs = sorted({r for _, r, _ in arms})
    sh("git fetch -q https://github.com/danielhanchen/unsloth-staging-2 "
       + " ".join(f"{r}:{r}" for r in refs) + " --force", "fetch_load", cwd=REPO)
    dists = {}
    for label, ref, patch in arms:
        d = build_variant(ref, "l_" + label, patch)
        if d is None:
            result("load", {"arm": label, "error": "build failed"})
            continue
        dists[label] = (d, ref)
    if not dists:
        return
    threads = W / "threads_load.json"
    url = launch(next(iter(dists.values()))[0], W / "hf_empty")
    try:
        if not threads.exists():
            sh(f"{sys.executable} -u {PACK}/menu_bench.py seed --base-url {url} --home {HOME} "
               f"--turns {os.environ.get('LOAD_TURNS', '1 150')} --out {threads}", "seed_load", timeout=3600)
    finally:
        stop()
    turns = os.environ.get("LOAD_TURNS", "1 150").split()[-1]
    for rnd in range(int(os.environ.get("LOAD_ROUNDS", "3"))):
        for label, (dist, ref) in dists.items():
            try:
                url = launch(dist, W / "hf_empty")
                if rnd == 0:
                    record_build(label, ref, dist, url)
                for mat in os.environ.get("LOAD_MATERIALS", "solid,glass").split(","):
                    out = OUT / f"load_{label}_{mat}_{rnd}.json"
                    sh(f"{sys.executable} -u {bench}/load_bench.py --base-url {url} --home {HOME} --threads {threads} "
                       f"--turns {turns} --arm {label} --engine {os.environ.get('LOAD_ENGINE', 'webkit')} "
                       f"--material {mat} --reps 1 --out {out}", f"load_{label}_{mat}_{rnd}", timeout=900, echo="LOAD")
            except Exception as exc:  # noqa: BLE001
                result("load", {"arm": label, "round": rnd, "error": str(exc)[:300]})
            finally:
                stop()


def main() -> int:
    stages = sys.argv[1:] or ["gate", "bisect", "glass", "catalog"]
    try:
        setup()
    except Exception as exc:  # noqa: BLE001
        log(f"setup failed: {exc}")
        return 2
    for st in stages:
        log(f"stage {st} start")
        try:
            if st != "gate":
                install_studio()
            globals()[f"stage_{st}"]()
        except Exception as exc:  # noqa: BLE001
            result(st, {"error": str(exc)[:600]})
        log(f"stage {st} end")
    stop()
    with tarfile.open(JOB / "out.tgz", "w:gz") as tf:
        tf.add(OUT, arcname="out")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
