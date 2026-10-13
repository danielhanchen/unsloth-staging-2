"""Real-Windows check of setup.ps1's background frontend build (#8818).

Extracts the real helpers from studio/setup.ps1 and drives them with the real `npm run build`
under Windows PowerShell 5.1 and pwsh 7:
  serial      cmd /d /c npm run build, dist hashed
  background  Start-FrontendBuildBackground (default cmd.exe argv) + Complete-FrontendBuildBackground,
              dist must hash identical to serial, exit code read back as 0
  failure     a TypeScript error: the join must exit non-zero and name the build's exit code
  kill        Stop-FrontendBuildBackground mid-build: no process of the build tree survives,
              the hidden .gitignore is back, the build mark stays
Writes --out JSON, prints JOB_RESULT, exit 0 when every check passes.

    python ps1_bg_build.py --head SHA
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_URL = "https://github.com/unslothai/unsloth"
FUNCS = [
    "New-UnslothTemporaryFile",
    "Exit-SetupFailure",
    "Test-FrontendBuildInBackground",
    "Start-FrontendBuildBackground",
    "Restore-FrontendBuildGitignores",
    "Remove-FrontendBuildFiles",
    "Stop-FrontendBuildBackground",
    "Complete-FrontendBuildOutput",
    "Complete-FrontendBuildBackground",
]

PRELUDE = r"""
$ErrorActionPreference = "Stop"
$script:UnslothVerbose = $false
$script:FrontendBuildProc = $null
$script:FrontendBuildFiles = @()
$script:FrontendBuildHidden = @()
function step($a, $b, $c) { Write-Output "step $a $b" }
function substep($a, $b) { Write-Output "substep $a" }
function Write-StudioLine($t) { Write-Output "line $t" }
function Redact-InstallOutput($t) { $t }
"""


def sh(
    cmd,
    cwd = None,
    check = True,
    timeout = 1800,
):
    p = subprocess.run(
        cmd,
        cwd = cwd,
        stdin = subprocess.DEVNULL,
        stdout = subprocess.PIPE,
        stderr = subprocess.STDOUT,
        text = True,
        errors = "replace",
        timeout = timeout,
    )
    if check and p.returncode != 0:
        sys.stdout.write(p.stdout[-4000:])
        raise SystemExit(f"command failed ({p.returncode}): {cmd}")
    return p.returncode, p.stdout


def dist_hash(dist: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    n = 0
    for f in sorted(p for p in dist.rglob("*") if p.is_file()):
        h.update(f.relative_to(dist).as_posix().encode())
        h.update(hashlib.sha256(f.read_bytes()).digest())
        n += 1
    return h.hexdigest(), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default = None)
    ap.add_argument("--base", default = None, help = "accepted, unused")
    ap.add_argument("--work", default = None)
    ap.add_argument("--out", default = "ps1_bg_build.json")
    ap.add_argument("--tiny", action = "store_true")
    ap.add_argument("--no-torch", action = "store_true")
    a = ap.parse_args()
    arts = Path("artifacts")
    arts.mkdir(exist_ok = True)
    result = {
        "job": "ps1_bg_build",
        "host": {"system": platform.system(), "cpus": os.cpu_count()},
        "checks": {},
        "cases": [],
    }
    if platform.system() != "Windows":
        result.update(skipped = True, passed = False, finished = True)
        Path(a.out).write_text(json.dumps(result, indent = 1))
        print("JOB_RESULT " + json.dumps({"skipped": True}))
        return 3

    work = Path(a.work or (Path.cwd() / "ps1bg_work")).resolve()
    work.mkdir(parents = True, exist_ok = True)
    repo = work / "unsloth"
    if not (repo / ".git").exists():
        sh(["git", "clone", "-q", "--filter=blob:none", REPO_URL, str(repo)])
    if a.head:
        sh(["git", "-C", str(repo), "fetch", "-q", "origin", a.head])
        sh(["git", "-C", str(repo), "checkout", "-q", "-f", a.head])
    result["head"] = sh(["git", "-C", str(repo), "rev-parse", "HEAD"])[1].strip()
    fe = repo / "studio" / "frontend"
    npm = shutil.which("npm.cmd") or shutil.which("npm")
    rc, out = sh(
        [npm, "ci", "--no-fund", "--no-audit", "--loglevel=error"],
        cwd = fe,
        check = False,
        timeout = 1800,
    )
    (arts / "npm_ci.log").write_text(out, encoding = "utf-8")
    result["checks"]["npm_ci"] = {"ok": rc == 0, "detail": f"rc={rc}"}
    if rc != 0:
        raise SystemExit("npm ci failed")

    # Serial reference build, exactly as setup.ps1's serial branch runs it.
    shutil.rmtree(fe / "dist", ignore_errors = True)
    t0 = time.time()
    rc, out = sh(["cmd", "/d", "/c", "npm run build"], cwd = fe, check = False)
    serial_s = round(time.time() - t0, 1)
    ref, nref = dist_hash(fe / "dist") if (fe / "dist").exists() else ("", 0)
    result["serial"] = {"rc": rc, "wall_s": serial_s, "dist_sha": ref, "files": nref}
    result["checks"]["serial_build_ok"] = {
        "ok": rc == 0 and nref > 0,
        "detail": f"rc={rc} files={nref}",
    }

    # Real helper source, extracted with the repo's own test helper.
    getter = repo / "tests" / "studio_setup_ps1" / "Get-FunctionSource.ps1"
    setup = repo / "studio" / "setup.ps1"
    helpers = work / "helpers.ps1"
    extract = work / "extract.ps1"
    extract.write_text(
        f". '{getter}'\n$src = @()\n"
        + "".join(f"$src += (Get-FunctionSource -Path '{setup}' -Name '{fn}')\n" for fn in FUNCS)
        + f"Set-Content -LiteralPath '{helpers}.body' -Value ($src -join \"`n\") -Encoding utf8\n",
        encoding = "utf-8",
    )
    sh(["pwsh", "-NoProfile", "-File", str(extract)])
    helpers.write_text(
        PRELUDE + "\n" + (work / "helpers.ps1.body").read_text(encoding = "utf-8-sig"),
        encoding = "utf-8",
    )

    # A stand-in for a venv's "*" .gitignore; not an ancestor, so the build reads the same tree.
    fake_outer = work / "fake_outer"
    fake_outer.mkdir(exist_ok = True)
    outer_gi = fake_outer / ".gitignore"
    outer_gi.write_text("*\n")
    mark = fe / ".unsloth-build.local"
    main_tsx = fe / "src" / "main.tsx"
    original_tsx = main_tsx.read_bytes()

    def run_case(shell, name, body):
        script = work / f"case_{shell}_{name}.ps1"
        head = (
            f". '{helpers}'\n$DistDir = '{fe / 'dist'}'\n$FrontendBuildMark = '{mark}'\n"
            f"$FrontendDir = '{fe}'\n"
        )
        script.write_text(head + body, encoding = "utf-8")
        t0 = time.time()
        p = subprocess.run(
            [shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)],
            cwd = fe,
            stdin = subprocess.DEVNULL,
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            text = True,
            errors = "replace",
            timeout = 1800,
        )
        (arts / f"{shell}_{name}.log").write_text(p.stdout, encoding = "utf-8")
        return p.returncode, p.stdout, round(time.time() - t0, 1)

    hide = (
        f"Rename-Item -LiteralPath '{outer_gi}' -NewName '.gitignore._twbuild' -Force\n"
        f"$hidden = @('{outer_gi}')\n"
    )
    for shell in ("powershell", "pwsh"):
        if not shutil.which(shell):
            result["checks"][f"{shell}_present"] = {"ok": False, "detail": "not on PATH"}
            continue
        # background success
        shutil.rmtree(fe / "dist", ignore_errors = True)
        mark.write_text("1")
        rc, out, wall = run_case(
            shell,
            "bg_ok",
            hide
            + (
                "$ok = Start-FrontendBuildBackground -WorkingDirectory $FrontendDir -HiddenGitignores $hidden\n"
                "Write-Output \"started=$ok\"\nWrite-Output 'parent-step'\nComplete-FrontendBuildBackground\n"
                "Write-Output 'after-join'\n"
            ),
        )
        bg, nbg = dist_hash(fe / "dist") if (fe / "dist").exists() else ("", 0)
        ok = (
            rc == 0
            and "started=True" in out
            and "step frontend built" in out
            and "after-join" in out
            and bg == ref
            and not mark.exists()
            and outer_gi.exists()
        )
        result["cases"].append(
            {
                "shell": shell,
                "case": "bg_ok",
                "rc": rc,
                "wall_s": wall,
                "dist_sha": bg,
                "files": nbg,
                "mark_left": mark.exists(),
                "gitignore_back": outer_gi.exists(),
            }
        )
        result["checks"][f"{shell}_bg_build_identical_to_serial"] = {
            "ok": ok,
            "detail": f"rc={rc} identical={bg == ref} mark_left={mark.exists()}",
        }

        # background failure: a TypeScript error
        main_tsx.write_bytes(original_tsx + b"\nconst __broken: number = 'x';\n")
        mark.write_text("1")
        rc, out, wall = run_case(
            shell,
            "bg_fail",
            hide
            + (
                "$ok = Start-FrontendBuildBackground -WorkingDirectory $FrontendDir -HiddenGitignores $hidden\n"
                "Write-Output \"started=$ok\"\nComplete-FrontendBuildBackground\nWrite-Output 'not-reached'\n"
            ),
        )
        main_tsx.write_bytes(original_tsx)
        named = "npm run build failed (exit code" in out and "(exit code 0)" not in out
        ok = rc != 0 and named and "not-reached" not in out and mark.exists() and outer_gi.exists()
        result["cases"].append(
            {
                "shell": shell,
                "case": "bg_fail",
                "rc": rc,
                "wall_s": wall,
                "mark_left": mark.exists(),
                "gitignore_back": outer_gi.exists(),
            }
        )
        result["checks"][f"{shell}_bg_failure_reported"] = {
            "ok": ok,
            "detail": f"rc={rc} named_code={named} mark_left={mark.exists()}",
        }

        # stop mid-build: the whole tree goes
        mark.write_text("1")
        rc, out, wall = run_case(
            shell,
            "bg_kill",
            hide
            + (
                "$ok = Start-FrontendBuildBackground -WorkingDirectory $FrontendDir -HiddenGitignores $hidden\n"
                "$root = $script:FrontendBuildProc.Id\n"
                'function Get-Tree($id) { $kids = @(Get-CimInstance Win32_Process -Filter "ParentProcessId=$id" -ErrorAction SilentlyContinue);'
                " foreach ($k in $kids) { $k; Get-Tree $k.ProcessId } }\n"
                "$deadline = (Get-Date).AddSeconds(60); $tree = @()\n"
                "while ((Get-Date) -lt $deadline) { $tree = @(Get-Tree $root); if (@($tree | Where-Object { $_.Name -eq 'node.exe' }).Count -gt 0) { break }; Start-Sleep -Milliseconds 500 }\n"
                "Start-Sleep 5; $tree = @(Get-Tree $root)\n"
                "Write-Output (\"tree=\" + (($tree | ForEach-Object { $_.Name + ':' + $_.ProcessId }) -join ','))\n"
                "Stop-FrontendBuildBackground\nStart-Sleep 2\n"
                "$alive = @($tree | Where-Object { Get-Process -Id $_.ProcessId -ErrorAction SilentlyContinue })\n"
                "if (Get-Process -Id $root -ErrorAction SilentlyContinue) { $alive += 'root' }\n"
                'Write-Output ("alive=" + $alive.Count)\n'
            ),
        )
        tree = next((ln.split("=", 1)[1] for ln in out.splitlines() if ln.startswith("tree=")), "")
        alive = next(
            (ln.split("=", 1)[1].strip() for ln in out.splitlines() if ln.startswith("alive=")), "?"
        )
        ok = rc == 0 and "node.exe" in tree and alive == "0" and mark.exists() and outer_gi.exists()
        result["cases"].append(
            {
                "shell": shell,
                "case": "bg_kill",
                "rc": rc,
                "tree": tree,
                "alive": alive,
                "mark_left": mark.exists(),
                "gitignore_back": outer_gi.exists(),
            }
        )
        result["checks"][f"{shell}_stop_kills_tree"] = {
            "ok": ok,
            "detail": f"tree={tree[:200]} alive={alive} gitignore_back={outer_gi.exists()}",
        }
        if not outer_gi.exists() and (fake_outer / ".gitignore._twbuild").exists():
            (fake_outer / ".gitignore._twbuild").rename(outer_gi)

    result["passed"] = all(c["ok"] for c in result["checks"].values())
    result["finished"] = True
    Path(a.out).write_text(json.dumps(result, indent = 1))
    (arts / "ps1_bg_build.json").write_text(json.dumps(result, indent = 1))
    print("JOB_RESULT " + json.dumps({"passed": result["passed"], "checks": result["checks"]}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
