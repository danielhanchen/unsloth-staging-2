#!/usr/bin/env python3
"""Cross-platform reproduction of unsloth PR #6166 (uv bytecode-compile timeout).

Runs on the host OS (Linux / macOS / Windows) and proves:
  S0  uv does NOT compile bytecode by default (installer only hits this when the
      caller enables compilation, so the fix is a no-op otherwise).
  S1a uv 0.7.22 + compile + UV_COMPILE_BYTECODE_TIMEOUT=1  -> times out (bug #5983).
  S1b uv 0.7.22 + compile + timeout=180                    -> succeeds (the fix).
  S1c uv 0.7.22 + compile + timeout=0  (disable)           -> succeeds (override).
  S2  uv 0.7.21 + compile + timeout=1                      -> IGNORED, succeeds
      (proves the version floor bump to 0.7.22 is required for the fix to work).

Exits non-zero if any expectation fails, so a green CI job means the PR's
premise holds on that OS with real uv binaries.
"""
import os, sys, subprocess, tempfile, venv, shutil, pathlib

NFUNCS, NSTMTS = 800, 1000  # ~6-8s to bytecode-compile; reliably > 1s, well < 180s


def run(cmd, env=None):
    p = subprocess.run(cmd, env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True)
    return p.returncode, p.stdout


def bindir(v):
    return os.path.join(v, "Scripts" if os.name == "nt" else "bin")


def exe(path):
    return path + (".exe" if os.name == "nt" else "")


def make_uv(workdir, version):
    vdir = os.path.join(workdir, "uvenv_" + version)
    venv.create(vdir, with_pip=True, clear=True)
    pip = exe(os.path.join(bindir(vdir), "pip"))
    rc, out = run([pip, "install", "-q", "uv==" + version])
    if rc != 0:
        print(out)
        sys.exit("failed to install uv==" + version)
    uvbin = exe(os.path.join(bindir(vdir), "uv"))
    _, ver = run([uvbin, "--version"])
    print("  uv %s -> %s" % (version, ver.strip()))
    return uvbin


def make_pkg(workdir):
    pkg = os.path.join(workdir, "heavypkg")
    src = os.path.join(pkg, "heavypkg")
    os.makedirs(src, exist_ok=True)
    with open(os.path.join(pkg, "pyproject.toml"), "w") as f:
        f.write('[project]\nname = "heavypkg"\nversion = "0.0.1"\n'
                '[build-system]\nrequires = ["setuptools"]\n'
                'build-backend = "setuptools.build_meta"\n')
    open(os.path.join(src, "__init__.py"), "w").close()
    with open(os.path.join(src, "heavy.py"), "w") as f:
        for fi in range(NFUNCS):
            f.write("def f%d():\n" % fi)
            for si in range(NSTMTS):
                f.write("    x%d = %d + %d*2 - (%d%%7) + (%d^3)\n" % (si, si, si, si, si))
            f.write("    return 0\n\n")
    return pkg


def case(uvbin, pkg, workdir, label, timeout, compile_on):
    target = os.path.join(workdir, "t_" + label)
    shutil.rmtree(target, ignore_errors=True)
    run([uvbin, "venv", target])
    tpy = exe(os.path.join(bindir(target), "python"))
    env = dict(os.environ)
    env.pop("UV_COMPILE_BYTECODE", None)
    env.pop("UV_COMPILE_BYTECODE_TIMEOUT", None)
    if compile_on:
        env["UV_COMPILE_BYTECODE"] = "1"
    if timeout is not None:
        env["UV_COMPILE_BYTECODE_TIMEOUT"] = str(timeout)
    rc, out = run([uvbin, "pip", "install", "--python", tpy, pkg], env=env)
    timed_out = "timed out" in out.lower()
    pyc = len(list(pathlib.Path(target).rglob("heavy.*.pyc"))) > 0
    shutil.rmtree(target, ignore_errors=True)
    return rc, timed_out, pyc, out


def main():
    workdir = tempfile.mkdtemp(prefix="uvsim_")
    print("workdir=%s  os.name=%s  platform=%s" % (workdir, os.name, sys.platform))
    failures = []

    def expect(name, cond, detail):
        print("  [%s] %s : %s" % ("PASS" if cond else "FAIL", name, detail))
        if not cond:
            failures.append(name)

    try:
        print("building slow-compile package (%dx%d)..." % (NFUNCS, NSTMTS))
        pkg = make_pkg(workdir)
        print("installing pinned uv versions...")
        uv2 = make_uv(workdir, "0.7.22")
        uv1 = make_uv(workdir, "0.7.21")

        rc, to, pyc, _ = case(uv2, pkg, workdir, "S0", 1, compile_on=False)
        expect("S0_no_default_compile", rc == 0 and not pyc,
               "rc=%s pyc=%s (expect rc=0, pyc=False)" % (rc, pyc))

        rc, to, pyc, out = case(uv2, pkg, workdir, "S1a", 1, compile_on=True)
        expect("S1a_bug_repro_timeout1", rc != 0 and to,
               "rc=%s timed_out=%s (expect rc!=0, timed_out=True)" % (rc, to))
        for line in out.splitlines():
            if "timed out" in line.lower() or "bytecode-compile" in line.lower():
                print("        > " + line.strip())

        rc, to, pyc, _ = case(uv2, pkg, workdir, "S1b", 180, compile_on=True)
        expect("S1b_fix_timeout180", rc == 0 and pyc,
               "rc=%s pyc=%s (expect rc=0, pyc=True)" % (rc, pyc))

        rc, to, pyc, _ = case(uv2, pkg, workdir, "S1c", 0, compile_on=True)
        expect("S1c_disable_timeout0", rc == 0 and pyc,
               "rc=%s pyc=%s (expect rc=0, pyc=True)" % (rc, pyc))

        rc, to, pyc, _ = case(uv1, pkg, workdir, "S2", 1, compile_on=True)
        expect("S2_uv0721_ignores_var", rc == 0 and not to,
               "rc=%s timed_out=%s (expect rc=0, timed_out=False)" % (rc, to))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    print("")
    if failures:
        print("DIFFERENTIAL FAILED: " + ", ".join(failures))
        sys.exit(1)
    print("ALL DIFFERENTIAL CASES PASSED")


if __name__ == "__main__":
    main()
