"""Run Settings > API "curl + training" snippets, as the tree's usage-examples.tsx renders them, against
a live Studio installed from that tree with a real sk-unsloth key. Windows: the PowerShell variant
under Windows PowerShell 5.1 and pwsh 7; elsewhere: the Unix curl variant under bash.

Installs with --no-torch (CI runners have no GPU), so training itself is not under test: the checks
are that the script runs top to bottom: the LLM start is accepted, the script waits for that job to
end before the image steps (so the image start is never refused by the LLM interlock), nothing is
refused by auth or request validation, the request files hold exactly the documented JSON, and the
image upload lands. Needs node (GitHub-hosted runners ship it) to render the TSX.
"""

from __future__ import annotations

import json
import os
import platform
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as c  # noqa: E402

COMP = Path("studio/frontend/src/features/settings/components")
BAD = {401, 403, 404, 405, 415, 422}
WIN = platform.system() == "Windows"

RENDER_JS = r"""
import fs from "node:fs";
const [src, agentSrc, out, base, key] = process.argv.slice(2);
const s = fs.readFileSync(src, "utf8");
const grab = (name) => { const i = s.indexOf(name); if (i < 0) throw new Error("missing " + name);
  let d = 0; const j = s.indexOf("{", s.indexOf(")", i));
  for (let k = j;; k++) { if (s[k] === "{") d++; if (s[k] === "}" && !--d) return s.slice(i, k + 1); } };
const t0 = s.indexOf("const TRAIN = {");
const train = s.slice(t0, s.indexOf("} as const;", t0)) + "};";
const helpers = "const j = (v) => JSON.stringify(v);\n" + fs.readFileSync(agentSrc, "utf8").split("\n")
  .filter((l) => /^export const (sh|ps)Single/.test(l) || /^  value\.replace/.test(l)).join("\n");
const bodies = s.slice(s.indexOf("const trainBody"), s.indexOf("function curlTrainingUnix"));
const code = [train, helpers, bodies, grab("function curlTrainingUnix"), grab("function curlTrainingWindows")]
  .join("\n").replace(/export /g, "").replace(/: string/g, "").replace(/ as const/g, "");
const f = new Function(code + "; return {u: curlTrainingUnix, w: curlTrainingWindows, b: trainBody, i: imageTrainBody};")();
fs.writeFileSync(out + "/train.sh", f.u(base, key));
fs.writeFileSync(out + "/train.ps1", f.w(base, key));
fs.writeFileSync(out + "/bodies.json", JSON.stringify({train: f.b, image: f.i}));
"""


def png(
    path,
    size = 256,
    seed = 1,
):
    """A noise PNG without Pillow (the job venv may lack it)."""
    import random

    rnd = random.Random(seed)
    raw = b"".join(
        b"\x00" + bytes(rnd.randrange(256) for _ in range(size * 3)) for _ in range(size)
    )

    def chunk(tag, data):
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    Path(path).write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def http(
    method,
    url,
    body = None,
    token = None,
    timeout = 30,
):
    req = urllib.request.Request(
        url,
        data = None if body is None else json.dumps(body).encode(),
        method = method,
        headers = {"Content-Type": "application/json"},
    )
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout = timeout) as r:
        return json.load(r)


def studio_bin(home):
    cands = (
        [
            home / "bin" / "unsloth.exe",
            home / "unsloth_studio" / "Scripts" / "unsloth.exe",
            home / "bin" / "unsloth.cmd",
        ]
        if WIN
        else [home / "bin" / "unsloth", home / "unsloth_studio" / "bin" / "unsloth"]
    )
    for p in cands:
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"unsloth CLI not under {home}: {[str(p) for p in cands]}")


def requests_logged(home, stdout_log):
    """(path, method, status) of every request_completed line, from the session logs and stdout."""
    seen = []
    files = sorted((home / "logs").rglob("*.log")) + [stdout_log]
    for f in files:
        try:
            text = f.read_text(encoding = "utf-8", errors = "replace")
        except OSError:
            continue
        for line in text.splitlines():
            if '"request_completed"' not in line:
                continue
            try:
                rec = json.loads(line[line.index("{") :])
            except ValueError:
                continue
            seen.append(
                (rec.get("path"), rec.get("method"), rec.get("status_code"), rec.get("timestamp"))
            )
    return sorted(set(seen), key = lambda r: r[3] or "")


def json_objects(text):
    """The JSON replies curl printed back to back, in request order."""
    dec, i, out = json.JSONDecoder(), 0, []
    while True:
        j = text.find("{", i)
        if j < 0:
            return out
        try:
            obj, i = dec.raw_decode(text, j)
            out.append(obj)
        except ValueError:
            i = j + 1


def validation_or_auth_error(body):
    """FastAPI's request-validation 422 (detail = list of {loc, msg}) or an auth refusal; a
    route-level refusal (HF access, missing torch, 409) means the request got through."""
    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, list) and any(isinstance(d, dict) and "loc" in d for d in detail):
        return True
    text = json.dumps(detail).lower() if detail is not None else ""
    return any(
        k in text
        for k in ("not authenticated", "invalid or expired", "invalid api key", "unauthorized")
    )


def image_block(script):
    """The image steps alone: the body of the snippet's final `if` (gated on the LLM run), dedented."""
    lines = script.splitlines()
    first = next(
        i
        for i, ln in enumerate(lines)
        if ln.startswith("if ") and ("PHASE" in ln or "$phase" in ln)
    )
    body = lines[first + 1 : len(lines) - 1]
    assert lines[-1] in ("fi", "}"), lines[-1]
    return "\n".join(ln[2:] if ln.startswith("  ") else ln for ln in body) + "\n"


def exact(rec, work, name, fname, expect):
    try:
        got = json.loads((work / fname).read_text(encoding = "ascii"))
    except (OSError, ValueError) as e:
        got = f"unreadable: {e}"
    rec.check(f"{name}:{fname}_exact", got == expect, got)


def main():
    p = c.base_parser("studio_api_snippets", "n/a", "n/a", default_steps = 0)
    p.set_defaults(compile_cache = "keep")
    args = c.resolve_args(p)
    root = Path.cwd()
    work = (root / "snippet-run").resolve()
    work.mkdir(exist_ok = True)
    home = work / "home"
    with c.JobRecorder(args, backend_hint = "studio-http") as rec:
        if not (root / COMP / "usage-examples.tsx").is_file():
            rec.skip("not an unsloth checkout with Studio frontend")
        env = dict(
            os.environ,
            UNSLOTH_STUDIO_HOME = str(home),
            UNSLOTH_DISABLE_UPDATE_CHECK = "1",
            UNSLOTH_STUDIO_DISABLE_PUBLIC_CHECK = "1",
            PYTHONUTF8 = "1",
        )
        t = time.time()
        inst = (
            [
                "pwsh",
                "-NoProfile",
                "-Command",
                "& ./install.ps1 --local --no-torch; exit $LASTEXITCODE",
            ]
            if WIN
            else ["bash", "install.sh", "--local", "--no-torch"]
        )
        with open(work / "install.log", "w", encoding = "utf-8") as lf:
            r = subprocess.run(
                inst, cwd = root, env = env, stdout = lf, stderr = subprocess.STDOUT, timeout = 40 * 60
            )
        rec.summary(install_s = round(time.time() - t, 1))
        if not rec.check("studio_installed", r.returncode == 0, f"exit {r.returncode}"):
            print((work / "install.log").read_text(encoding = "utf-8", errors = "replace")[-6000:])
            return
        port = 18765
        base = f"http://127.0.0.1:{port}"
        out_log = work / "studio.log"
        proc = subprocess.Popen(
            [studio_bin(home), "studio", "-H", "127.0.0.1", "-p", str(port)],
            cwd = work,
            env = env,
            stdout = open(out_log, "w", encoding = "utf-8"),
            stderr = subprocess.STDOUT,
        )
        try:
            for _ in range(240):
                try:
                    urllib.request.urlopen(base + "/api/health", timeout = 3)
                    break
                except Exception:
                    time.sleep(3)
            boot = (home / "auth" / ".bootstrap_password").read_text(encoding = "utf-8").strip()
            tok = http("POST", base + "/api/auth/login", {"username": "unsloth", "password": boot})[
                "access_token"
            ]
            tok = http(
                "POST",
                base + "/api/auth/change-password",
                {"current_password": boot, "new_password": "Snippets-CI-1!"},
                tok,
            )["access_token"]
            key = http("POST", base + "/api/auth/api-keys", {"name": "snippets-ci"}, tok)["key"]
            rec.check("api_key_created", key.startswith("sk-unsloth-"), key[:11])

            (work / "render.mjs").write_text(RENDER_JS, encoding = "utf-8")
            subprocess.run(
                [
                    "node",
                    str(work / "render.mjs"),
                    str(root / COMP / "usage-examples.tsx"),
                    str(root / COMP / "agent-command.ts"),
                    str(work),
                    base,
                    key,
                ],
                check = True,
            )
            bodies = json.loads((work / "bodies.json").read_text(encoding = "utf-8"))
            for i in (1, 2):
                png(work / f"cat{i}.png", seed = i)

            script = "train.ps1" if WIN else "train.sh"
            full = (work / script).read_text(encoding = "utf-8")
            (work / ("image." + script.split(".")[1])).write_text(
                image_block(full), encoding = "utf-8"
            )
            runners = (
                [
                    (
                        "powershell",
                        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File"],
                    ),
                    ("pwsh", ["pwsh", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File"]),
                ]
                if WIN
                else [("bash", ["bash"])]
            )
            for shell, argv in runners:
                for part in ("full", "image"):
                    name = f"{shell}:{part}"
                    f = script if part == "full" else "image." + script.split(".")[1]
                    mark = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
                    for stale in ("train.json", "image-train.json", "image-start.json"):
                        (work / stale).unlink(missing_ok = True)
                    r = subprocess.run(
                        argv + [f],
                        cwd = work,
                        capture_output = True,
                        text = True,
                        encoding = "utf-8",
                        errors = "replace",
                        timeout = 1200,
                    )
                    print(
                        f"===== {name} exit {r.returncode}\n{r.stdout[-5000:]}\n{r.stderr[-3000:]}"
                    )
                    time.sleep(2)
                    hits = [
                        h
                        for h in requests_logged(home, out_log)
                        if (h[3] or "") >= mark and (h[0] or "").startswith("/api/train")
                    ]
                    print(f"===== {name} requests: {hits}")
                    by_path = {}
                    for path, method, status, _ in hits:
                        by_path.setdefault((method, path), []).append(status)
                    rec.check(f"{name}:script_exit_0", r.returncode == 0, r.returncode)
                    rec.check(
                        f"{name}:no_validation_or_auth_error_in_replies",
                        not any(validation_or_auth_error(o) for o in json_objects(r.stdout)),
                        r.stdout[-400:],
                    )
                    image_codes = by_path.get(("POST", "/api/train/diffusion/start"), [])
                    if part == "full":
                        start_codes = by_path.get(("POST", "/api/train/start"), [])
                        rec.check(f"{name}:llm_start_accepted", start_codes == [200], start_codes)
                        llm = http("GET", base + "/api/train/status", token = key)
                        rec.check(
                            f"{name}:llm_job_terminal",
                            bool(llm.get("job_id"))
                            and llm.get("phase") in ("completed", "error", "stopped"),
                            {k: llm.get(k) for k in ("job_id", "phase", "message")},
                        )
                        # No GPU here, so the LLM run ends in error and the image steps must not run.
                        if llm.get("phase") == "error":
                            rec.check(
                                f"{name}:image_skipped_after_llm_error",
                                not image_codes
                                and not by_path.get(("POST", "/api/train/diffusion/dataset")),
                                by_path,
                            )
                        if WIN:
                            exact(rec, work, name, "train.json", bodies["train"])
                    else:
                        rec.check(
                            f"{name}:upload_200",
                            by_path.get(("POST", "/api/train/diffusion/dataset")) == [200],
                            by_path.get(("POST", "/api/train/diffusion/dataset")),
                        )
                        # 409s while an LLM worker exits are retried by the script; the last attempt decides.
                        rec.check(
                            f"{name}:diffusion_start_reached_route_not_interlocked",
                            bool(image_codes)
                            and image_codes[-1] not in BAD | {409}
                            and not set(image_codes) & BAD,
                            image_codes,
                        )
                        rec.check(
                            f"{name}:image_start_reply_saved",
                            (work / "image-start.json").is_file(),
                            "",
                        )
                        if WIN:
                            exact(rec, work, name, "image-train.json", bodies["image"])
            ds = list((home / "assets" / "datasets").rglob("cat1.png"))
            rec.check("uploaded_images_on_disk", bool(ds), [str(x) for x in ds])
        finally:
            proc.terminate()
            try:
                proc.wait(timeout = 30)
            except subprocess.TimeoutExpired:
                proc.kill()
            if WIN:
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output = True)


if __name__ == "__main__":
    main()
