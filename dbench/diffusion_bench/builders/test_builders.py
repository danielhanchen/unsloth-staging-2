#!/usr/bin/env python3
"""CPU, offline checks of the ported builder steps (also runs under pytest).

  python diffusion_bench/builders/test_builders.py
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TK = HERE.parent
sys.path.insert(0, str(TK))

import builders as BL  # noqa: E402
from models import list_cards, load_card  # noqa: E402


def _all_build_strings():
    for fam in list_cards():
        card = load_card(fam)
        for fmt, f in card["formats"].items():
            for n, e in ((f.get("bline") or {}).get("sources") or {}).items():
                for key in ("build", "alt_build"):
                    if isinstance(e, dict) and e.get(key):
                        yield fam, fmt, n, key, e


def test_every_card_build_parses():
    bad, n = [], 0
    for fam, fmt, name, key, e in _all_build_strings():
        n += 1
        try:
            BL.parse_build(e[key])
        except KeyError as exc:
            bad.append(f"{fam} {fmt} {name} {key}: {exc}")
    assert n and not bad, bad


def test_every_step_plans():
    steps = BL.card_steps()
    assert steps, "no built files in the cards"
    for step, uses in steps.items():
        for fam, fmt, name, e in uses:
            plan = BL.plan_build(e, name, "/nonexistent_out")
            assert plan["cmds"] and plan["produced"].name.endswith((".safetensors", ".gguf")), (step, name, plan)
            for c in plan["cmds"]:
                assert Path(c[2]).exists(), c


def test_run_dry_run_one_format_per_step():
    seen = set()
    with tempfile.TemporaryDirectory(dir = os.environ.get("WORKSPACE") and Path(os.environ["WORKSPACE"]) / "temp") as d:
        for step, uses in sorted(BL.card_steps().items()):
            fam, fmt, name, _ = uses[0]
            if (fam, fmt, name) in seen:
                continue
            seen.add((fam, fmt, name))
            r = subprocess.run([sys.executable, str(HERE / "run.py"), "--card", fam, "--format", fmt, "--file", name,
                                "--out", d, "--dry-run"], capture_output = True, text = True)
            assert r.returncode == 0 and "BUILD " in r.stdout and "PLANNED" in r.stdout, (step, r.stdout, r.stderr)
        assert not any(Path(d).iterdir()), "dry-run wrote files"


def test_dry_run_imports_no_torch_and_fetch_and_null():
    code = (
        "import sys, json; sys.path.insert(0, sys.argv[1]); import builders as BL\n"
        "steps = BL.card_steps()\n"
        "for uses in steps.values():\n"
        "    fam, fmt, name, e = uses[0]; BL.build_file(e, name, '/nonexistent_out', dry_run=True)\n"
        "p = BL.build_file({'repo': 'stabilityai/sdxl-turbo', 'file': 'sd_xl_turbo_1.0_fp16.safetensors', 'build': None},"
        " 'sd_xl_turbo_1.0_fp16.safetensors', '/nonexistent_out', dry_run=True)\n"
        "assert p.name == 'sd_xl_turbo_1.0_fp16.safetensors'\n"
        "try:\n"
        "    BL.build_file({'source': None, 'reason': 'x'}, 'a.safetensors', '/nonexistent_out', dry_run=True)\n"
        "    raise SystemExit('null source accepted')\n"
        "except ValueError:\n"
        "    pass\n"
        "try:\n"
        "    BL.build_file({'repo': 'Bline-Org/x', 'file': 'a.safetensors', 'build': None}, 'a.safetensors',"
        " '/nonexistent_out', dry_run=True)\n"
        "    raise SystemExit('Bline-Org fetch accepted')\n"
        "except ValueError:\n"
        "    pass\n"
        "assert 'torch' not in sys.modules, 'dry run imported torch'\n"
        "print('NO_TORCH_OK')\n")
    r = subprocess.run([sys.executable, "-c", code, str(TK)], capture_output = True, text = True)
    assert r.returncode == 0 and "NO_TORCH_OK" in r.stdout, (r.stdout[-2000:], r.stderr[-2000:])


def test_modules_parse_and_h3_regroup_shared():
    for p in HERE.glob("*.py"):
        ast.parse(p.read_text(), str(p))
    src = (HERE / "build_bline_files.py").read_text()
    assert "from models.convert import" in src and "h3_qkv_regroup(v)" in src
    assert "reshape(-1, 3, 128" not in src, "a private copy of the regroup is back"
    for mod in set(BL.SCRIPTS.values()):
        assert (HERE / f"{mod}.py").exists(), mod
    for f in ("q21_dit", "q21_te", "q21_vae", "h3_dit", "h3_te_int8", "h3_vae", "h3_avae", "f3_official_sha"):
        json.loads((HERE / "ref_headers" / f"{f}.json").read_text())


def test_failed_role_frees_its_tmp():
    """A role whose quant tool fails leaves no merged source / partial output behind (the H3 TE on a Colab A100:
    55 GB stuck after 'No space left on device', every later cell short of disk)."""
    import builders.build_bline_files as BCF  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        tool = d / "tool.py"
        tool.write_text("import sys; open(sys.argv[2], 'wb').write(b'partial'); sys.exit(1)\n")
        key = ("h3", "te")
        saved = BCF.BUILD[key]
        BCF.BUILD[key] = lambda tmp: Path(tmp).write_bytes(b"merged source")
        argv = sys.argv
        sys.argv = ["build_bline_files.py", "--group", "h3", "--only", "te", "--out", str(d / "out"), "--tmp",
                    str(d / "tmp"), "--tool", str(tool)]
        try:
            try:
                BCF.main()
                raise AssertionError("a failing quant tool must fail the build")
            except SystemExit as e:
                assert "quant tool failed" in str(e), e
        finally:
            BCF.BUILD[key] = saved
            sys.argv = argv
        left = sorted(str(p.relative_to(d)) for p in (d / "out").rglob("*") if p.is_file()) + \
            sorted(str(p.relative_to(d)) for p in (d / "tmp").rglob("*") if p.is_file())
        assert not left, f"a failed role left files behind: {left}"


def test_drop_sources_frees_only_own_fetches_before_the_tool():
    """--drop-sources on a VM-owned hub cache deletes the shards the builder itself downloaded once the merged source
    is written, BEFORE the quant tool runs (the H3 TE peak was shards + source + output at once); a blob already in
    the cache (a Studio probe fetched it) and any blob on a shared host (no DBENCH_BUILDER_OWNS_HF_CACHE) stay; the
    H3 TE keeps-filter drops exactly lm_head, layers >= 50 and the final norm."""
    import os  # noqa: PLC0415

    import builders.build_bline_files as BCF  # noqa: PLC0415

    assert BCF.h3_te_keeps("model.language_model.layers.49.mlp.up_proj.weight")
    assert not BCF.h3_te_keeps("model.language_model.layers.50.mlp.up_proj.weight")
    assert not BCF.h3_te_keeps("lm_head.weight") and not BCF.h3_te_keeps("model.language_model.norm.weight")
    assert BCF.h3_te_keeps("model.visual.blocks.0.attn.qkv.weight")
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        hub = d / "hf"
        hub.mkdir()
        own, other = hub / "blob_own", hub / "blob_studio"
        for b in (own, other):
            b.write_bytes(b"x" * 64)
        saved_env = {k: os.environ.get(k) for k in ("HF_HOME", BCF.OWNS_CACHE_ENV)}
        os.environ["HF_HOME"] = str(hub)
        try:
            BCF.FETCHED_NEW.clear()
            BCF.FETCHED_NEW.add(Path(os.path.realpath(own)))
            os.environ.pop(BCF.OWNS_CACHE_ENV, None)
            assert not BCF.own_remove(str(own), roots = [d / "out"]) and own.exists(), "shared host: cache kept"
            os.environ[BCF.OWNS_CACHE_ENV] = "1"
            assert BCF.own_remove(str(own), roots = [d / "out"]) and not own.exists(), "VM: own fetch freed"
            assert not BCF.own_remove(str(other), roots = [d / "out"]) and other.exists(), "VM: Studio's blob kept"
        finally:
            BCF.FETCHED_NEW.clear()
            for k, v in saved_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        # order: the role's sources are dropped after BUILD wrote the merged source and before the tool starts
        seen = []
        key = ("h3", "te")
        saved = BCF.BUILD[key], BCF.drop_sources, BCF.run_tool, BCF.header_diff
        BCF.header_diff = lambda path, ref: {"identical_layout": True}
        BCF.BUILD[key] = lambda tmp: (seen.append("build"), Path(tmp).write_bytes(b"src"))
        BCF.drop_sources = lambda role: seen.append("drop")
        BCF.run_tool = lambda tool, src, dst, extra: (seen.append("tool"), Path(dst).write_bytes(b"out"))
        tool = d / "tool.py"
        tool.write_text("")
        argv = sys.argv
        sys.argv = ["build_bline_files.py", "--group", "h3", "--only", "te", "--out", str(d / "out"), "--tmp",
                    str(d / "tmp"), "--tool", str(tool), "--drop-sources", "--allow-diff"]
        try:
            BCF.main()
        except (SystemExit, KeyError):
            pass
        finally:
            BCF.BUILD[key], BCF.drop_sources, BCF.run_tool, BCF.header_diff = saved
            sys.argv = argv
        assert seen[:3] == ["build", "drop", "tool"], seen


def main() -> int:
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok   {name}")
            except AssertionError as exc:
                fails += 1
                print(f"FAIL {name}: {str(exc)[:1500]}")
    n = sum(len(u) for u in BL.card_steps().values())
    print(f"{len(BL.card_steps())} steps, {n} built card files: {'passed' if not fails else f'{fails} FAILED'}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
