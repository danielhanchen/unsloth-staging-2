#!/usr/bin/env python3
"""Probe: does the unsloth_zoo Mega-cache work on this host, and does anything compile at all?

Observes only (criteria/megacache_no_regression.py judges). One file serves three vehicles:

  AMD CI (differential.py):  --state S --checkout <zoo worktree> --out OUT.json
  staging --job (CPU legs):  --tiny --out OUT.json [--zoo base=git:main]   (zoo under test = installed)
  local:                     --state S --checkout <zoo worktree> --out OUT.json

What it records, per state:
  env      platform, python, torch / hip / cuda, device, triton, MSVC, a GPU and a CPU torch.compile smoke
  trust    the checkout's own compile_cache verdicts: default root and its ancestors (read only), a dir it
           creates, the same dir after granting Everyone / other write, and a junction / symlink to it
  fsops    O_NOFOLLOW / O_NONBLOCK presence, os.replace over a file another handle holds open, unlink of
           an open file, filelock's platform lock
  pytest   tests/test_compile_cache*.py of the checkout (pass / fail IDs), when --pytest
  runs     fresh child processes: arm (default | mega1 | ...), run (cold | wiped | kept), workload
           (mech = compile_cache API around a compiled MLP train step; unsloth = LoRA SFT steps through
           the normal Unsloth API). Each child reports first-step / process time, Dynamo frames,
           FX / AOTAutograd cache hit-miss-bypass, megacache_status(), mega-cache log lines, errors.

Writes JSON to --out only; never parses its own stdout (import banners).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

HERE = os.path.abspath(__file__)
DEFAULT_ARMS = "default,mega1"
ARM_ENV = {
    "default": {},
    "mega1": {"UNSLOTH_MEGA_CACHE": "1"},
    "mega0": {"UNSLOTH_MEGA_CACHE": "0"},
    # What Studio's training worker does on Windows ROCm (core/training/worker.py).
    "studio_rocm": {"TORCHDYNAMO_DISABLE": "1"},
    # Opt-in CPU compile of the zoo CPU backend (unsloth_zoo/cpu/loader.py).
    "cpucompile": {"UNSLOTH_CPU_COMPILE": "1"},
    "cpucompile_mega1": {"UNSLOTH_CPU_COMPILE": "1", "UNSLOTH_MEGA_CACHE": "1"},
}


def _tail(text, n = 4000):
    text = text or ""
    return text[-n:]


def _rmtree(path):
    if path and os.path.isdir(path):
        shutil.rmtree(path, ignore_errors = True)


# ----------------------------------------------------------------------------------------------
# zoo location: a checkout path, "installed", or git:<ref> (cloned into the work dir)


def _zoo_dir(spec, work):
    if spec == "installed":
        import importlib.util

        found = importlib.util.find_spec("unsloth_zoo")
        if found is None or not found.submodule_search_locations:
            raise SystemExit("unsloth_zoo is not installed")
        return None, os.path.dirname(list(found.submodule_search_locations)[0])
    if spec.startswith(("git:", "git+")):
        # git:<ref> = upstream unsloth-zoo at ref; git+<url>@<ref> = any clone URL at ref
        if spec.startswith("git:"):
            url, ref = "https://github.com/unslothai/unsloth-zoo", spec[4:]
        else:
            url, _, ref = spec[4:].rpartition("@")
        dest = os.path.join(work, "zoo_" + "".join(c if c.isalnum() else "_" for c in ref)[:40])
        if not os.path.isdir(dest):
            subprocess.run(["git", "clone", "-q", "--filter=blob:none", url, dest], check = True)
            subprocess.run(["git", "checkout", "-q", ref], cwd = dest, check = True)
        return dest, dest
    path = os.path.abspath(spec)
    return path, path


def _load_compile_cache(zoo_root):
    """The checkout's compile_cache.py as a standalone module: importing the unsloth_zoo package
    would run its import guard and patching, none of which the trust functions need."""
    import importlib.util

    path = os.path.join(zoo_root, "unsloth_zoo", "compile_cache.py")
    spec = importlib.util.spec_from_file_location("_probe_compile_cache_%d" % abs(hash(path)), path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _git_head(path):
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd = path,
            capture_output = True,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = 30,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


# ----------------------------------------------------------------------------------------------
# MSVC (CPU Inductor on Windows needs cl.exe); same approach as unsloth_zoo/cpu/loader.py


def _msvc_env():
    if os.name != "nt":
        return {}, "not windows"
    if shutil.which("cl"):
        return {}, "cl already on PATH"
    root = os.environ.get("ProgramFiles(x86)") or r"C:\Program Files (x86)"
    vswhere = os.path.join(root, "Microsoft Visual Studio", "Installer", "vswhere.exe")
    if not os.path.isfile(vswhere):
        return {}, "no vswhere.exe"
    try:
        done = subprocess.run(
            [
                vswhere,
                "-latest",
                "-products",
                "*",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property",
                "installationPath",
                "-utf8",
            ],
            capture_output = True,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = 60,
        )
        install = (done.stdout or "").strip().splitlines()
        if not install:
            return {}, "vswhere found no VC tools"
        vcvars = os.path.join(install[0].strip(), "VC", "Auxiliary", "Build", "vcvars64.bat")
        done = subprocess.run(
            'cmd /d /u /s /c ""%s" >nul 2>&1 && set"' % vcvars, capture_output = True, timeout = 180
        )
        env = {}
        for line in (done.stdout or b"").decode("utf-16-le", errors = "replace").splitlines():
            key, sep, value = line.partition("=")
            if sep and key.upper() in ("PATH", "INCLUDE", "LIB", "LIBPATH"):
                env[key.upper()] = value
        if "PATH" not in env:
            return {}, "vcvars64.bat gave no PATH"
        return env, "vcvars64.bat"
    except Exception as error:
        return {}, "msvc lookup failed: %r" % (error,)


# ----------------------------------------------------------------------------------------------
# children


def _counters():
    from torch._dynamo.utils import counters
    pick = lambda d, words: {k: v for k, v in d.items() if any(w in k for w in words)}
    return {
        "frames": dict(counters["frames"]),
        "inductor": pick(counters["inductor"], ("cache",)),
        "aot": pick(counters["aot_autograd"], ("cache",)),
        "unique_graphs": counters["stats"].get("unique_graphs"),
        "graph_breaks": sum(counters["graph_break"].values()),
        "recompiles": sum(v for k, v in counters["stats"].items() if "recompil" in k),
    }


def _megacache_state():
    mod = sys.modules.get("unsloth_zoo.compile_cache")
    if mod is None:
        for name, value in list(sys.modules.items()):
            if name.startswith("_probe_compile_cache_"):
                mod = value
    if mod is None:
        return {"imported": False}
    try:
        state = dict(mod.megacache_status())
    except Exception as error:
        state = {"error": repr(error)}
    state["imported"] = True
    state["file"] = getattr(mod, "__file__", None)
    return state


def child_mech(args, res):
    """compile_cache API around a compiled 2-layer MLP train step: what unsloth_compile_transformers
    does (megacache_load before the first compiled call, atexit save), minus Unsloth."""
    t0 = time.perf_counter()
    import torch

    res["import_s"] = round(time.perf_counter() - t0, 2)
    cc = _load_compile_cache(args.zoo_root)
    res["loaded"] = cc.megacache_load("probe_mlp", {"probe": 1}, {"epilogue_fusion": True})
    res["megacache_after_load"] = _megacache_state()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    res["device"] = device
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(64, 256), torch.nn.GELU(), torch.nn.Linear(256, 64)
    ).to(device)
    opt = torch.optim.SGD(model.parameters(), lr = 1e-2)
    step = torch.compile(lambda x: model(x).float().pow(2).mean())
    x = torch.randn(32, 64, device = device)
    losses = []
    t = time.perf_counter()
    for i in range(3):
        loss = step(x)
        loss.backward()
        opt.step()
        opt.zero_grad()
        if device == "cuda":
            torch.cuda.synchronize()
        losses.append(float(loss))
        if i == 0:
            res["first_step_s"] = round(time.perf_counter() - t, 3)
    res["losses"] = losses


def child_unsloth(args, res):
    t0 = time.perf_counter()
    entry = args.entry
    if entry == "flm":
        from unsloth import FastLanguageModel as FM
    else:
        from unsloth import FastModel as FM
    import torch

    res["import_s"] = round(time.perf_counter() - t0, 2)
    res["torchdynamo_disable"] = os.environ.get("TORCHDYNAMO_DISABLE")
    try:
        import torch._dynamo
        res["dynamo_config_disable"] = bool(torch._dynamo.config.disable)
    except Exception:
        pass
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer
    from datasets import Dataset

    cuda = torch.cuda.is_available()
    res["device"] = "cuda" if cuda else "cpu"

    class FirstStep(TrainerCallback):
        t = None

        def on_step_end(self, a, state, control, **kw):
            if FirstStep.t is None:
                if cuda:
                    torch.cuda.synchronize()
                FirstStep.t = time.perf_counter()

    t = time.perf_counter()
    model, tok = FM.from_pretrained(args.model, max_seq_length = 128, load_in_4bit = False, dtype = None)
    model = FM.get_peft_model(
        model,
        r = 8,
        lora_alpha = 16,
        random_state = 0,
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    res["load_s"] = round(time.perf_counter() - t, 2)
    res["model_class"] = type(model).__name__
    res["megacache_after_load"] = _megacache_state()
    tokenizer = getattr(tok, "tokenizer", tok)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    ds = Dataset.from_dict({"text": [("hello world number %d. " % i) * 6 for i in range(16)]})
    torch.manual_seed(0)
    trainer = SFTTrainer(
        model = model,
        processing_class = tokenizer,
        train_dataset = ds,
        callbacks = [FirstStep()],
        args = SFTConfig(
            max_length = 64,
            dataset_text_field = "text",
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 1,
            max_steps = args.steps,
            learning_rate = 1e-4,
            logging_steps = 1,
            report_to = "none",
            seed = 0,
            output_dir = os.path.join(args.work, "trainer_out"),
            save_strategy = "no",
            dataset_num_proc = None,
        ),
    )
    t = time.perf_counter()
    trainer.train()
    res["train_s"] = round(time.perf_counter() - t, 2)
    res["first_step_s"] = round(FirstStep.t - t, 3) if FirstStep.t else None
    res["losses"] = [h["loss"] for h in trainer.state.log_history if "loss" in h]


def run_child(args):
    t0 = time.perf_counter()
    res = {"workload": args.child, "pid": os.getpid()}
    try:
        if args.child == "mech":
            child_mech(args, res)
        else:
            child_unsloth(args, res)
    except BaseException as error:  # noqa: BLE001 -- the probe records, never judges
        res["error"] = "%s: %s" % (type(error).__name__, str(error)[:600])
        res["traceback"] = _tail(traceback.format_exc(), 3000)
    try:
        res.update(_counters())
    except Exception as error:
        res["counters_error"] = repr(error)
    res["megacache"] = _megacache_state()
    res["child_total_s"] = round(time.perf_counter() - t0, 2)
    with open(args.out, "w", encoding = "utf-8") as fh:
        json.dump(res, fh, default = str)
    # atexit (Mega-cache save) runs after this point; the parent reads its log line and the bundle dir.
    return 0


# ----------------------------------------------------------------------------------------------
# parent: environment, trust, fs semantics, pytest, child runs


def env_report(extra_env):
    rep = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "os_name": os.name,
        "cpu_count": os.cpu_count(),
    }
    code = r"""
import json, os, sys, time
out = {}
try:
    import torch
    out["torch"] = torch.__version__; out["hip"] = getattr(torch.version, "hip", None); out["cuda"] = torch.version.cuda
    out["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        out["device_name"] = torch.cuda.get_device_name(0); out["gcn_arch"] = getattr(p, "gcnArchName", None)
        out["total_mem_gb"] = round(p.total_memory / 2**30, 1)
    out["has_megacache_api"] = hasattr(torch.compiler, "save_cache_artifacts")
except Exception as e:
    out["torch_error"] = repr(e)
try:
    import triton
    out["triton"] = getattr(triton, "__version__", "?"); out["triton_file"] = triton.__file__
except Exception as e:
    out["triton_error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
for dev in (["cuda"] if out.get("cuda_available") else []) + ["cpu"]:
    try:
        import torch
        torch._dynamo.reset()
        x = torch.linspace(-4, 4, 4096, device = dev)
        t = time.perf_counter()
        y = torch.compile(lambda v: torch.nn.functional.silu(v) * 2)(x)
        ok = bool(torch.allclose(y, torch.nn.functional.silu(x) * 2))
        out["compile_smoke_" + dev] = {"ok": ok, "s": round(time.perf_counter() - t, 2)}
    except Exception as e:
        out["compile_smoke_" + dev] = {"ok": False, "error": "%s: %s" % (type(e).__name__, str(e).strip().splitlines()[0][:300] if str(e).strip() else "")}
try:
    import filelock
    out["filelock"] = filelock.__version__
except Exception as e:
    out["filelock_error"] = repr(e)
for name in ("transformers", "trl", "peft", "unsloth"):
    try:
        from importlib.metadata import version
        out[name] = version(name)
    except Exception:
        out[name] = None
print("@@ENV " + json.dumps(out))
"""
    env = dict(os.environ)
    env.update(extra_env)
    try:
        p = subprocess.run(
            [sys.executable, "-c", code],
            capture_output = True,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = 900,
            env = env,
        )
        line = [l for l in p.stdout.splitlines() if l.startswith("@@ENV ")]
        rep.update(
            json.loads(line[-1][6:]) if line else {"env_error": _tail(p.stdout + p.stderr, 2000)}
        )
    except Exception as error:
        rep["env_error"] = repr(error)
    rep["cl_on_path"] = bool(shutil.which("cl"))
    return rep


def _grant_everyone_write(path):
    if os.name == "nt":
        p = subprocess.run(
            ["icacls", path, "/grant", "*S-1-1-0:(OI)(CI)M"],
            capture_output = True,
            text = True,
            encoding = "utf-8",
            errors = "replace",
        )
        return p.returncode == 0, _tail(p.stdout + p.stderr, 300)
    os.chmod(path, 0o777)
    return True, "chmod 0777"


def _make_link(target, link):
    if os.name == "nt":
        p = subprocess.run(
            ["cmd", "/c", "mklink", "/J", link, target],
            capture_output = True,
            text = True,
            encoding = "utf-8",
            errors = "replace",
        )
        return p.returncode == 0, "junction: " + _tail(p.stdout + p.stderr, 200).strip()
    os.symlink(target, link)
    return True, "symlink"


def _owner(path):
    if os.name != "nt":
        try:
            return "uid %d" % os.stat(path).st_uid
        except Exception as error:
            return repr(error)
    try:
        p = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "(Get-Acl -LiteralPath '%s').Owner" % path],
            capture_output = True,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = 60,
        )
        return p.stdout.strip() or _tail(p.stderr, 200)
    except Exception as error:
        return repr(error)


def _win_chain(cc, target):
    """Prototype zoo only (has _windows_path_security): owner, ACEs and the leaf / ancestor verdict
    for every existing component of target, so a refusal names the component and the ACE."""
    if os.name != "nt" or not hasattr(cc, "_windows_path_security"):
        return None
    out = {"token_sids": None, "components": []}
    try:
        out["token_sids"] = list(cc._windows_token_sids())
    except Exception as error:
        out["token_sids"] = repr(error)
    target = os.path.abspath(target)
    current = target
    while True:
        if os.path.lexists(current):
            row = {"path": current, "leaf": current == target}
            try:
                attrs, owner, aces = cc._windows_path_security(current)
                row.update(
                    attrs = hex(attrs),
                    owner = owner,
                    aces = None
                    if aces is None
                    else [
                        [hex(t), hex(f), None if m is None else hex(m), sid]
                        for t, f, m, sid in aces
                    ],
                    trusted = cc._windows_security_is_trusted(
                        (attrs, owner, aces), directory = True, leaf = current == target
                    ),
                )
            except Exception as error:
                row["error"] = repr(error)
            out["components"].append(row)
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return out


def trust_report(cc, work):
    rep = {}
    saved = (
        os.environ.pop("UNSLOTH_MEGA_CACHE_DIR", None),
        os.environ.pop("UNSLOTH_MEGA_CACHE", None),
    )
    try:
        rep["enabled_by_default"] = cc.megacache_is_enabled()
        root = cc._cache_root()
        rep["default_root"] = root
        rep["default_root_exists"] = bool(root and os.path.isdir(root))
        rep["default_root_trusted"] = cc._is_trusted_directory(root)
        chain = []
        cur = root
        while cur:
            if os.path.isdir(cur):
                st = os.lstat(cur)
                chain.append(
                    {
                        "path": cur,
                        "trusted_existing": cc._is_trusted_directory(cur, allow_missing = False),
                        "st_uid": getattr(st, "st_uid", None),
                        "mode": oct(st.st_mode),
                        "reparse": bool(getattr(st, "st_file_attributes", 0) & 0x400),
                    }
                )
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent
        rep["default_root_chain"] = chain
        if os.name == "nt":
            rep["home_owner"] = _owner(os.path.expanduser("~"))
    finally:
        if saved[0] is not None:
            os.environ["UNSLOTH_MEGA_CACHE_DIR"] = saved[0]
        if saved[1] is not None:
            os.environ["UNSLOTH_MEGA_CACHE"] = saved[1]
    base = os.path.join(work, "trust")
    _rmtree(base)
    os.makedirs(base)
    fresh = os.path.join(base, "fresh", "mega_cache", "key")
    rep["fresh_created_trusted"] = cc._ensure_private_directory(fresh)
    rep["fresh_trusted_existing"] = cc._is_trusted_directory(fresh, allow_missing = False)
    rep["fresh_owner"] = _owner(fresh)
    rep["fresh_chain"] = _win_chain(cc, fresh)
    rep["default_root_win_chain"] = _win_chain(
        cc, rep.get("default_root") or os.path.expanduser("~")
    )
    open(os.path.join(fresh, "f.bin"), "wb").write(b"x")
    rep["fresh_file_read"] = (
        cc._read_trusted_file(os.path.join(fresh, "f.bin"), binary = True) is not None
    )
    loose = os.path.join(base, "loose", "key")
    os.makedirs(loose)
    ok, how = _grant_everyone_write(os.path.join(base, "loose"))
    open(os.path.join(loose, "f.bin"), "wb").write(b"x")
    if os.name != "nt":
        os.chmod(os.path.join(loose, "f.bin"), 0o666)
    rep["everyone_write"] = {
        "applied": ok,
        "how": how,
        "dir_trusted": cc._is_trusted_directory(loose, allow_missing = False),
        "file_read": cc._read_trusted_file(os.path.join(loose, "f.bin"), binary = True) is not None,
    }
    link = os.path.join(base, "link")
    try:
        ok, how = _make_link(fresh, link)
        rep["link"] = {
            "made": ok,
            "how": how,
            "trusted": cc._is_trusted_directory(link, allow_missing = False),
            "file_read_through_link": cc._read_trusted_file(
                os.path.join(link, "f.bin"), binary = True
            )
            is not None,
        }
    except Exception as error:
        rep["link"] = {"made": False, "error": repr(error)}
    return rep


def fsops_report(work):
    rep = {
        "O_NOFOLLOW": hasattr(os, "O_NOFOLLOW"),
        "O_NONBLOCK": hasattr(os, "O_NONBLOCK"),
        "geteuid": hasattr(os, "geteuid"),
    }
    d = os.path.join(work, "fsops")
    _rmtree(d)
    os.makedirs(d)
    target = os.path.join(d, "manifest.json")
    open(target, "w", encoding = "utf-8").write("old")
    reader = open(target, "r", encoding = "utf-8")
    try:
        tmp = target + ".tmp"
        open(tmp, "w", encoding = "utf-8").write("new")
        try:
            os.replace(tmp, target)
            rep["replace_over_open_reader"] = "ok"
        except Exception as error:
            rep["replace_over_open_reader"] = "%s: %s" % (type(error).__name__, error)
        bundle = os.path.join(d, "megacache-old.bin")
        open(bundle, "wb").write(b"b")
        held = open(bundle, "rb")
        try:
            os.unlink(bundle)
            rep["unlink_open_bundle"] = "ok"
        except Exception as error:
            rep["unlink_open_bundle"] = "%s: %s" % (type(error).__name__, error)
        finally:
            held.close()
    finally:
        reader.close()
    try:
        import filelock

        cls = filelock.WindowsFileLock if os.name == "nt" else filelock.UnixFileLock
        lock = cls(os.path.join(d, ".lock"), timeout = 5)
        lock.acquire()
        lock.release()
        rep["filelock_platform_lock"] = "%s ok" % cls.__name__
    except Exception as error:
        rep["filelock_platform_lock"] = "%s: %s" % (type(error).__name__, error)
    return rep


def pytest_report(zoo_root, work, extra_env):
    tests_dir = os.path.join(zoo_root, "tests")
    files = (
        sorted(
            f
            for f in os.listdir(tests_dir)
            if f.startswith("test_compile_cache") and f.endswith(".py")
        )
        if os.path.isdir(tests_dir)
        else []
    )
    if not files:
        return {"files": []}
    junit = os.path.join(work, "pytest_%d.xml" % os.getpid())
    env = dict(os.environ)
    env.update(extra_env)
    env["PYTHONPATH"] = zoo_root + os.pathsep + env.get("PYTHONPATH", "")
    env["UNSLOTH_IS_PRESENT"] = "1"
    p = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "--junitxml",
            junit,
            *["tests/" + f for f in files],
        ],
        cwd = zoo_root,
        capture_output = True,
        text = True,
        encoding = "utf-8",
        errors = "replace",
        env = env,
        timeout = 1800,
    )
    rep = {
        "files": files,
        "rc": p.returncode,
        "tail": _tail(p.stdout + p.stderr, 2500),
        "passed": [],
        "failed": [],
        "skipped": [],
    }
    try:
        import xml.etree.ElementTree as ET
        for case in ET.parse(junit).getroot().iter("testcase"):
            cid = "%s::%s" % (case.get("classname"), case.get("name"))
            kids = {c.tag for c in case}
            key = (
                "failed"
                if kids & {"failure", "error"}
                else "skipped"
                if "skipped" in kids
                else "passed"
            )
            rep[key].append(cid)
    except Exception as error:
        rep["junit_error"] = repr(error)
    return rep


def _bundle_state(mega_dir):
    out = {"manifests": 0, "bundles": 0, "bundle_bytes": 0}
    if not os.path.isdir(mega_dir):
        return out
    for root, _, files in os.walk(mega_dir):
        for f in files:
            if f == "manifest.json":
                out["manifests"] += 1
            elif f.startswith("megacache") and f.endswith(".bin"):
                out["bundles"] += 1
                out["bundle_bytes"] += os.path.getsize(os.path.join(root, f))
    return out


def run_one(args, zoo_path, workload, model, arm, label, wipe, base_env):
    arm_dir = os.path.join(
        args.work, "runs", workload + ("_" + os.path.basename(model) if model else ""), arm
    )
    if label == "cold":
        _rmtree(arm_dir)
    for d in wipe:
        _rmtree(os.path.join(arm_dir, d))
    tmp = os.path.join(arm_dir, "tmp")
    os.makedirs(tmp, exist_ok = True)
    env = dict(os.environ)
    env.update(base_env)
    for k in (
        "TORCHINDUCTOR_CACHE_DIR",
        "UNSLOTH_MEGA_CACHE",
        "UNSLOTH_MEGA_CACHE_DIR",
        "TORCHDYNAMO_DISABLE",
        "UNSLOTH_CPU_COMPILE",
        "UNSLOTH_COMPILE_DISABLE",
    ):
        env.pop(k, None)
    env.update(
        TMPDIR = tmp,
        TEMP = tmp,
        TMP = tmp,
        TRITON_CACHE_DIR = os.path.join(arm_dir, "triton"),
        UNSLOTH_COMPILE_LOCATION = os.path.join(arm_dir, "ucc"),
        UNSLOTH_ENABLE_LOGGING = "1",
        UNSLOTH_DISABLE_AUTO_UPDATES = "1",
        TOKENIZERS_PARALLELISM = "false",
        PYTHONHASHSEED = "0",
        PYTHONIOENCODING = "utf-8",
        HF_HUB_DISABLE_PROGRESS_BARS = "1",
    )
    mega = os.path.join(arm_dir, "mega")
    if not args.default_root:
        env["UNSLOTH_MEGA_CACHE_DIR"] = mega
    else:
        mega = os.path.join(os.path.expanduser("~"), ".cache", "unsloth", "mega_cache")
    if zoo_path:
        env["PYTHONPATH"] = zoo_path + os.pathsep + env.get("PYTHONPATH", "")
    env.update(ARM_ENV[arm])
    out = os.path.join(arm_dir, "child_%s.json" % label)
    if os.path.exists(out):
        os.unlink(out)
    cmd = [
        sys.executable,
        HERE,
        "--child",
        workload,
        "--zoo-root",
        args.zoo_root,
        "--out",
        out,
        "--work",
        arm_dir,
        "--model",
        model or "",
        "--steps",
        str(args.steps),
        "--entry",
        args.entry,
    ]
    t = time.perf_counter()
    try:
        p = subprocess.run(
            cmd,
            env = env,
            cwd = arm_dir,
            capture_output = True,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = args.child_timeout,
        )
        rc, stdout, stderr = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as error:
        rc, stdout, stderr = "timeout", str(error.stdout or ""), str(error.stderr or "")
    wall = round(time.perf_counter() - t, 2)
    res = {}
    if os.path.exists(out):
        res = json.load(open(out, encoding = "utf-8"))
    else:
        res["error"] = "child wrote no result (rc %s)" % rc
        res["stderr_tail"] = _tail(stderr, 3000)
    lines = [l.strip() for l in (stdout + "\n" + stderr).splitlines()]
    res["megacache_log"] = [l for l in lines if "Mega-cache" in l][-12:]
    keys = (
        "torch.compile",
        "TORCHDYNAMO",
        "Inductor",
        "no GPU found",
        "UNSLOTH_CPU_COMPILE",
        "Triton",
    )
    res["compile_notes"] = [
        l[:300] for l in lines if any(k in l for k in keys) and "register_constant" not in l
    ][-6:]
    res.update(
        workload = workload,
        model = model,
        arm = arm,
        run = label,
        rc = rc,
        process_wall_s = wall,
        bundle = _bundle_state(mega),
    )
    if rc not in (0, "0") and "stderr_tail" not in res:
        res["stderr_tail"] = _tail(stderr, 1500)
    return res


def parent(args):
    t_start = time.perf_counter()
    obs = {"state": args.state, "argv": sys.argv[1:]}
    os.makedirs(args.work, exist_ok = True)
    zoo_path, zoo_root = _zoo_dir(args.checkout, args.work)
    args.zoo_root = zoo_root
    obs["zoo_root"] = zoo_root
    obs["zoo_sha"] = _git_head(zoo_root)
    msvc, msvc_how = _msvc_env()
    obs["msvc"] = {"how": msvc_how, "loaded": bool(msvc)}
    obs["env"] = env_report({})
    if msvc:
        obs["env_with_msvc"] = {
            k: v for k, v in env_report(msvc).items() if k.startswith("compile_smoke")
        }
    try:
        cc = _load_compile_cache(zoo_root)
        obs["trust"] = trust_report(cc, args.work)
    except Exception as error:
        obs["trust"] = {"error": repr(error), "traceback": _tail(traceback.format_exc(), 2000)}
    try:
        obs["fsops"] = fsops_report(args.work)
    except Exception as error:
        obs["fsops"] = {"error": repr(error)}
    if args.pytest:
        try:
            obs["pytest"] = pytest_report(zoo_root, args.work, msvc)
        except Exception as error:
            obs["pytest"] = {"error": repr(error)}
    # Other zoo trees (e.g. a prototype branch): trust verdicts and their compile_cache tests only.
    for item in args.trust_extra:
        name, _, spec = item.partition("=")
        try:
            _, extra_root = _zoo_dir(spec, args.work)
            extra = {"spec": spec, "zoo_sha": _git_head(extra_root)}
            extra["trust"] = trust_report(
                _load_compile_cache(extra_root), os.path.join(args.work, "extra_" + name)
            )
            extra["pytest"] = pytest_report(extra_root, args.work, msvc)
        except Exception as error:
            extra = {
                "spec": spec,
                "error": repr(error),
                "traceback": _tail(traceback.format_exc(), 2000),
            }
        obs.setdefault("extra", {})[name] = extra
    # Model weights fetched once here, so no child's time includes the download.
    models = [m for m in args.models.split(",") if m] if "unsloth" in args.workloads else []
    for m in models:
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(m)
        except Exception as error:
            obs.setdefault("download_errors", {})[m] = repr(error)
    runs = []
    kept = [w for w in (args.kept or "").split(",") if w]
    for workload in [w for w in args.workloads.split(",") if w]:
        plan = [("cold", ()), ("wiped", ("tmp", "triton"))] + (
            [("kept", ())] if workload in kept else []
        )
        for model in models if workload == "unsloth" else [None]:
            for arm in args.arms.split(","):
                for label, wipe in plan:
                    r = run_one(
                        args,
                        zoo_path,
                        workload,
                        model,
                        arm,
                        label,
                        wipe,
                        msvc if workload == "mech" or arm.startswith("cpucompile") else {},
                    )
                    runs.append(r)
                    _write(args.out, dict(obs, runs = runs, partial = True))
                    print(
                        "@@RUN",
                        workload,
                        model,
                        arm,
                        label,
                        "rc",
                        r.get("rc"),
                        "first",
                        r.get("first_step_s"),
                        "fx",
                        r.get("inductor"),
                        "aot",
                        r.get("aot"),
                        "mc",
                        r.get("megacache", {}).get("armed"),
                        "err",
                        (r.get("error") or "")[:160],
                        flush = True,
                    )
    obs["runs"] = runs
    obs["probe_s"] = round(time.perf_counter() - t_start, 1)
    _write(args.out, obs)
    return 0


def _write(path, obs):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok = True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding = "utf-8") as fh:
        json.dump(obs, fh, indent = 1, default = str)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default = "head")
    ap.add_argument(
        "--checkout", default = "installed", help = "zoo checkout path, 'installed', or git:<ref>"
    )
    ap.add_argument("--out", required = True)
    ap.add_argument("--work", default = None)
    ap.add_argument("--workloads", default = "mech,unsloth")
    ap.add_argument("--models", default = "unsloth/Qwen3.5-0.8B")
    ap.add_argument("--entry", default = "fm", choices = ("fm", "flm"))
    ap.add_argument("--arms", default = DEFAULT_ARMS)
    ap.add_argument("--steps", type = int, default = 3)
    ap.add_argument(
        "--kept",
        default = "",
        help = "workloads (comma list) that also get a warm run with torch's temp caches kept",
    )
    ap.add_argument("--pytest", action = "store_true")
    ap.add_argument(
        "--default-root",
        action = "store_true",
        help = "use the real ~/.cache/unsloth/mega_cache (hosted runners only)",
    )
    ap.add_argument("--child-timeout", type = int, default = 3600)
    ap.add_argument(
        "--zoo",
        action = "append",
        default = [],
        help = "staging: NAME=SPEC, run the whole probe once per zoo (e.g. base=git:main)",
    )
    ap.add_argument(
        "--trust-extra",
        action = "append",
        default = [],
        help = "NAME=SPEC: also run the trust report and compile_cache tests of another zoo tree",
    )
    ap.add_argument("--tiny", action = "store_true", help = "accepted for staging --job; no effect")
    # child-only
    ap.add_argument("--child", default = None)
    ap.add_argument("--zoo-root", default = None)
    ap.add_argument("--model", default = "")
    args = ap.parse_args()
    if args.child:
        return run_child(args)
    unknown = [a for a in args.arms.split(",") if a not in ARM_ENV]
    if unknown:
        raise SystemExit("unknown arm(s) %s" % unknown)
    args.work = os.path.abspath(
        args.work
        or os.path.join(
            os.environ.get("RUNNER_TEMP") or tempfile.gettempdir(), "mcprobe_" + args.state
        )
    )
    if not args.zoo:
        return parent(args)
    # staging: one sub-probe per zoo, combined into one JSON keyed by state name
    combined = {}
    out = args.out
    for item in args.zoo:
        name, _, spec = item.partition("=")
        sub = argparse.Namespace(**vars(args))
        stem = out[:-5] if out.endswith(".json") else out
        sub.state, sub.checkout, sub.out = name, spec, "%s.%s.json" % (stem, name)
        sub.work = os.path.join(args.work, name)
        parent(sub)
        combined[name] = json.load(open(sub.out, encoding = "utf-8"))
        _write(out, combined)
    return 0


if __name__ == "__main__":
    sys.exit(main())
