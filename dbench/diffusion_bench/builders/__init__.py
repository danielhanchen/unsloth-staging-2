"""Baseline-UI file builders: the converter steps the model cards name in ``formats.<fmt>.bline.sources[*].build``.

The cards cite the round folders the steps were first run from (``scripts/imgcmp_nov5/...``, ``scripts/vidcmp_nov5/...``);
those scripts now live here, so a remote VM (bundle) or the tester can build a file from the card alone:

    from builders import build_file, parse_build
    entry = card["formats"]["int8"]["bline"]["sources"]["qwen_image_2.1_int8_convrot.safetensors"]
    build_file(entry, "qwen_image_2.1_int8_convrot.safetensors", "cm/", dry_run = True)

``build_file`` materialises ONE template-named file under ``out_dir``: a fetched-as-is entry (``build`` null) is
downloaded from ``fetch_repo`` (a byte-identical mirror) or ``repo`` and sha256-checked when the card pins one; a built
entry runs its step (in the baseline UI venv python: ``python=`` / $DIFFUSION_BENCH_BLINE_PYTHON) into
``out_dir/_built`` and links ``out_dir/<name>`` to the step's output. An entry with ``source: null`` raises (no
official route; the card gives the reason). Steps that quantize to int8 ConvRot need Bline-Org/bline-model-tools
``quant_int8_convrot.py`` (GitHub, not vendored): ``tool=`` or $DBENCH_CONVROT_TOOL. ``dry_run`` prints the commands,
downloads nothing and imports no torch.

CLI: ``python diffusion_bench/builders/run.py --card FAM --format FMT [--file NAME] --out DIR [--dry-run] | --list``.
"""

from __future__ import annotations

import ast
import hashlib
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent

# card path prefix -> ported module (both build_bline.py copies are one module: the image round's groups q21 / h3 / qe /
# qi plus the video round's H3 qkv regroup, now models.convert.h3_qkv_regroup)
SCRIPTS = {
    "scripts/imgcmp_nov5/prep_files.py": "img_prep_files",
    "scripts/imgcmp_nov5/build_q5.py": "img_build_q5",
    "scripts/imgcmp_nov5/build_bline.py": "build_bline_files",
    "scripts/vidcmp_nov5/build_bline.py": "build_bline_files",
    "scripts/vidcmp_nov5/convert_bline.py": "vid_convert_bline",
    "scripts/vidcmp_nov5/convert_bline_hv15.py": "vid_convert_bline_hv15",
    "scripts/vidcmp_nov5/h3fmt_bline_build.py": "vid_h3fmt_bline_build",
}
# build_bline_files roles written directly (no int8 ConvRot tool run)
_DIRECT = {("q21", "vae"), ("h3", "avae"), ("qe", "vae"), ("qi", "vae")}
_Q5_TOOL_ROLES = ("dit_int8", "dit_int8na", "te_int8")
_FORBIDDEN = ("bline-org/", "blineanonymous/")


def parse_build(build: str) -> dict:
    """A card build string -> {"step", "script", "module", "module_step", "argv", "note"}.

    ``step`` is the card's command with the trailing parenthesised note stripped (the key cards share), ``module_step``
    the same command on the ported module. KeyError for a script that is not ported."""
    s = (build or "").strip()
    note = None
    i = s.find(" (")
    if i >= 0 and s.endswith(")"):
        s, note = s[:i].strip(), s[i + 2:-1].strip()
    toks = shlex.split(s)
    if not toks or toks[0] not in SCRIPTS:
        raise KeyError(f"build step {build!r}: script not ported (known: {sorted(SCRIPTS)})")
    mod = SCRIPTS[toks[0]]
    return {"step": s, "script": toks[0], "module": mod, "module_step": " ".join([mod] + toks[1:]),
            "argv": toks[1:], "note": note}


def _opt(argv: list, flag: str) -> Optional[str]:
    return argv[argv.index(flag) + 1] if flag in argv and argv.index(flag) + 1 < len(argv) else None


def _literal(module: str, name: str):
    """A module-level literal (dict / tuple tables) read with ast: no import, so no torch."""
    tree = ast.parse((HERE / f"{module}.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(f"{module}.{name}")


def _official(name: str) -> str:
    return Path(name).stem + "_official.safetensors"


def baseline_python() -> Optional[str]:
    """The baseline UI venv python from the environment (either name setup_blineui reads)."""
    return os.environ.get("DIFFUSION_BENCH_BLINE_PYTHON") or os.environ.get("DIFFUSION_BENCH_BASELINE_PYTHON") or None


def baseline_dir() -> Optional[str]:
    """The baseline UI checkout from the environment (either name setup_blineui reads, then BLINE_DIR)."""
    return (os.environ.get("DIFFUSION_BENCH_BLINE_DIR") or os.environ.get("DIFFUSION_BENCH_BASELINE_DIR")
            or os.environ.get("BLINE_DIR") or None)


def needs_tool(cells, downloads: Optional[dict]) -> list:
    """The built file names among ``cells``' file roles whose step needs the int8 ConvRot tool (no side effects)."""
    downloads = downloads or {}
    out = []
    for cell in cells:
        for val in ((cell.get("options") or {}).get("files") or {}).values():
            for v in (val if isinstance(val, list) else [val]):
                e = downloads.get(v) if isinstance(v, str) else None
                if isinstance(e, dict) and e.get("build") and v not in out and \
                        plan_build(e, v, Path("/nonexistent"))["needs_tool"]:
                    out.append(v)
    return out


def plan_build(entry: dict, name: str, out_dir, tmp=None, python: Optional[str] = None,
               tool: Optional[str] = None) -> dict:
    """{"cmds": [[argv...]], "produced": Path, "needs_tool": bool, "step": ...} for one built file (no side effects)."""
    p = parse_build(entry["build"])
    out_dir = Path(out_dir)
    work = out_dir / "_built"
    tmp = Path(tmp) if tmp else out_dir / "_build_tmp"
    py = python or baseline_python() or sys.executable
    mod, argv = p["module"], list(p["argv"])
    script = str(HERE / f"{mod}.py")
    group = _opt(argv, "--group")
    needs_tool = False
    cmds = []
    if mod == "img_prep_files":
        sys.path.insert(0, str(HERE))
        import img_prep_files as PF  # no torch at import

        role = (p["note"] or "").split(":")[0].strip()
        if role not in PF.OFFICIAL.get(group, {}):
            raise KeyError(f"{p['step']}: the note must start with one of {sorted(PF.OFFICIAL.get(group, {}))}")
        cmds.append([py, "-u", script, "--groups", group, "--roles", role, "--out", str(work)])
        produced = work / PF.out_name(group, role)
    elif mod == "build_bline_files":
        role = _opt(argv, "--only")
        outname = _literal(mod, "OUTNAME")
        if (group, role) not in outname:
            raise KeyError(f"{p['step']}: no output for group {group} role {role}")
        needs_tool = (group, role) not in _DIRECT
        # --drop-sources frees only blobs the builder itself downloaded, and only where the cache is the VM's own
        # (DBENCH_BUILDER_OWNS_HF_CACHE); on a shared host it keeps every cache file (own_remove)
        cmds.append([py, "-u", script, *argv, "--out", str(work), "--tmp", str(tmp), "--drop-sources"]
                    + (["--tool", tool] if tool else []))
        produced = work / outname[(group, role)]
    elif mod == "img_build_q5":
        roles = (_opt(argv, "--roles") or "").split(",")
        out_t = _literal(mod, "OUT")
        needs_tool = any(r in _Q5_TOOL_ROLES for r in roles)
        cmds.append([py, "-u", script, *argv, "--out", str(work), "--tmp", str(tmp)] + (["--tool", tool] if tool else []))
        produced = work / out_t[roles[-1]].format(g = group)
    elif mod == "vid_convert_bline":
        low = name.lower()
        extra = []
        if "vae" in low:
            extra = ["--only", "vae", "--te", "none"]
        elif low.startswith("umt5"):
            extra = ["--only", "te"] + ([] if "--te" in argv else ["--te", "bf16" if "bf16" in low else "fp8"])
        else:
            extra = ["--only", "dit", "--te", "none"]
            for ex in ("high", "low"):
                if f"{ex}_noise" in low:
                    extra += ["--expert", ex]
            if "_fp16" in low and "fp8" not in low:
                extra += ["--dit-dtype", "fp16"] if group == "wan22_a14b" else []
            elif "_bf16" in low:
                extra += ["--dit-dtype", "bf16"]
        argv = [a for i, a in enumerate(argv) if not (a == "--te" or (i and argv[i - 1] == "--te"))] \
            if "--te" in extra else argv
        cmds.append([py, "-u", script, *argv, *extra, "--out", str(work), "--src", str(tmp / "src")])
        produced = work / _official(name)
    elif mod == "vid_convert_bline_hv15":
        low = name.lower()
        only = "byt5" if "byt5" in low else "vae" if "vae" in low else "te" if low.startswith("qwen") else "dit"
        extra = ["--only", only]
        if only == "te":
            extra += ["--te", "bf16" if "bf16" in low else "fp8"]
        cmds.append([py, "-u", script, *argv, *extra, "--out", str(work), "--src", str(tmp / "src")])
        produced = work / _official(name)
    elif mod == "vid_h3fmt_bline_build":
        kind = "nvfp4" if "nvfp4" in name.lower() else "fp8"
        pre = work / (_literal("build_bline_files", "OUTNAME")[("h3", "dit")] + ".src.safetensors")
        cmds.append([py, "-u", str(HERE / "build_bline_files.py"), "--group", "h3", "--only", "dit", "--prequant",
                     "--out", str(work), "--tmp", str(tmp)])
        cmds.append([py, "-u", script, "--src", str(pre), f"--{kind}", str(work / name)])
        produced = work / name
    else:  # pragma: no cover - SCRIPTS and this dispatch move together
        raise KeyError(mod)
    return {"step": p["step"], "module": mod, "cmds": cmds, "produced": produced, "needs_tool": needs_tool,
            "note": p["note"]}


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def verify_download(path, entry: dict, name: str = "") -> None:
    """Raise when a card entry pins a sha256 and the downloaded file differs: every fetch path (build_file, the VM
    driver, fetch_files) goes through this one check."""
    want = (entry or {}).get("sha256")
    if want and _sha256(path) != want:
        raise RuntimeError(f"{name or entry.get('file')}: sha256 of {entry.get('repo')}/{entry.get('file')} != the "
                           f"card's {want}")


def _link(src: Path, dst: Path) -> Path:
    src = Path(os.path.realpath(src))
    if dst.resolve() == src:
        return dst
    if dst.is_symlink():
        dst.unlink()  # a link this kit made earlier
    elif dst.exists():
        raise SystemExit(f"{dst} exists and is a regular file, not a link this kit made: refusing to replace it")
    dst.symlink_to(src)
    return dst


def build_file(entry: dict, name: str, out_dir, *, tmp=None, token: Optional[str] = None, dry_run: bool = False,
               tool: Optional[str] = None, python: Optional[str] = None, bline_dir: Optional[str] = None) -> Path:
    """Materialise the baseline-UI file ``name`` (a card ``bline.sources`` entry) as ``out_dir/name``; returns that path.

    ``build`` null: download ``file`` from ``fetch_repo`` or ``repo`` (sha256-checked when pinned). Otherwise run the
    ported step (``plan_build``) with ``python`` (default $DIFFUSION_BENCH_BLINE_PYTHON / _BASELINE_PYTHON, else this
    interpreter), BLINE_DIR = ``bline_dir`` / $DIFFUSION_BENCH_BLINE_DIR / _BASELINE_DIR / $BLINE_DIR, the int8 ConvRot ``tool`` (default
    $DBENCH_CONVROT_TOOL) and ``token`` as HF_TOKEN. ValueError for ``source: null``; RuntimeError when a step fails or
    needs the tool and has none."""
    out_dir = Path(out_dir)
    dst = out_dir / name
    if "source" in entry and entry["source"] is None:
        raise ValueError(f"{name}: no official source ({entry.get('reason')})")
    tool = tool or os.environ.get("DBENCH_CONVROT_TOOL")
    if not entry.get("build"):
        repo, f = entry.get("fetch_repo") or entry["repo"], entry["file"]
        if f.endswith("/"):
            raise ValueError(f"{name}: fetch entry names a directory ({f}) without a build step")
        if any(r.lower().startswith(_FORBIDDEN) for r in (repo, entry["repo"])):
            raise ValueError(f"{name}: refusing a Bline-Org / blineanonymous source {repo}")
        print(f"FETCH {repo}/{f} -> {dst}" + (f" (sha256 {entry['sha256'][:16]})" if entry.get("sha256") else ""),
              flush = True)
        if dry_run:
            return dst
        from huggingface_hub import hf_hub_download

        out_dir.mkdir(parents = True, exist_ok = True)
        import extnames  # noqa: PLC0415 - real repo / file names in a public copy

        p = hf_hub_download(extnames.ext(repo), extnames.ext(f), local_dir = str(Path(tmp or out_dir / "_fetch") / repo.replace("/", "__")),
                            token = token)
        if entry.get("sha256") and _sha256(p) != entry["sha256"]:
            raise RuntimeError(f"{name}: sha256 of {repo}/{f} != the card's {entry['sha256']}")
        return _link(Path(p), dst)
    plan = plan_build(entry, name, out_dir, tmp, python, tool)
    for c in plan["cmds"]:
        print("BUILD " + " ".join(shlex.quote(x) for x in c), flush = True)
    print(f"LINK {dst} -> {plan['produced']}", flush = True)
    if dry_run:
        return dst
    if plan["needs_tool"] and not (tool and Path(tool).exists()):
        raise RuntimeError(f"{name}: step {plan['step']!r} quantizes with Bline-Org/bline-model-tools "
                           "quant_int8_convrot.py: pass tool= or set DBENCH_CONVROT_TOOL")
    env = dict(os.environ)
    cdir = bline_dir or baseline_dir()
    if cdir:
        env["BLINE_DIR"] = cdir
    if token:
        env["HF_TOKEN"] = token
    if tool:
        env["DBENCH_CONVROT_TOOL"] = tool
    env["BLINE_PY"] = plan["cmds"][0][0]
    out_dir.mkdir(parents = True, exist_ok = True)
    for c in plan["cmds"]:
        r = subprocess.run(c, env = env)
        if r.returncode:
            raise RuntimeError(f"{name}: step {plan['step']!r} failed (rc {r.returncode}): {' '.join(c)}")
    if not plan["produced"].exists():
        raise RuntimeError(f"{name}: step {plan['step']!r} ran but {plan['produced']} is missing")
    return _link(plan["produced"], dst)


def card_steps(cards: Optional[list] = None) -> dict:
    """step -> [(family, format, file name, entry)] over every card's built baseline-UI files."""
    sys.path.insert(0, str(HERE.parent))
    from models import list_cards, load_card

    out: dict = {}
    for fam in cards or list_cards():
        card = load_card(fam)
        for fmt, f in (card.get("formats") or {}).items():
            for n, e in ((f.get("bline") or {}).get("sources") or {}).items():
                if isinstance(e, dict) and e.get("build"):
                    out.setdefault(parse_build(e["build"])["step"], []).append((card["family"], fmt, n, e))
    return out
