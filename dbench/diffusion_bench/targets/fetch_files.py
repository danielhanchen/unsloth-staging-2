#!/usr/bin/env python3
"""Download the {"repo", "file"[, "revision"]} refs in a spec's (or plan's) cell options.files and write a copy
with local paths, for the targets that run cells through matrix.py (local, AMD CI, Mac, tester). The VM driver does
the same lazily on Colab / Kaggle. "$CM" in a string path expands to --root. A bare file name the plan's "downloads"
map gives a card "build" step for is built by diffusion_bench/builders (build_file; --dry-run only plans it). The
doc's "prefetch" map (the Studio repos its cells load with local_files_only) is fetched into the HF cache under
$HF_HOME (default ~/.cache/huggingface), the same vm_driver.prefetch the Colab / Kaggle driver runs, so the bench
step must see the same HF_HOME.

  python diffusion_bench/targets/fetch_files.py --spec IN.json --root DIR --write OUT.json [--dry-run]
      [--venv PY] [--studio-src DIR]     # also point every cell at one driver python / one Studio checkout
      [--baseline-python PY] [--baseline-dir DIR] [--convrot-tool FILE]   # what built files are made with
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def refs(doc: dict) -> list:
    out = []
    for cell in doc.get("cells") or []:
        for key, vals in ((cell.get("options") or {}).get("files") or {}).items():
            for val in (vals if isinstance(vals, list) else [vals]):
                if isinstance(val, dict) and val.get("repo") and val.get("file"):
                    out.append((cell["tag"], key, val["repo"], val["file"], val.get("revision")))
    return out


def resolve(doc: dict, root: Path, download: bool = True, build_kw: dict | None = None) -> dict:
    """Cells with every file role resolved to a local path: a card build step is built, a {"repo", "file"} ref is
    fetched (sha256-checked when pinned: builders.verify_download, the builder's own check), list-valued roles
    (several text encoders) element by element. ``build_kw`` ({"python", "bline_dir", "tool"}: the baseline venv
    python, its checkout, the ConvRot quantizer) goes to every builder; unset keys fall back to the environment."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import builders
    import mirror  # noqa: PLC0415

    doc = json.loads(json.dumps(doc))
    got = {}
    verified: dict = {}  # file name -> mirror.describe(...): copied into the cells that use it (staged_files)
    downloads = doc.get("downloads") or {}
    build_kw = {k: v for k, v in (build_kw or {}).items() if v}

    def one(val):
        name = val if isinstance(val, str) else None
        if isinstance(val, str) and val in downloads:
            val = downloads[val]
        if isinstance(val, dict) and val.get("build") and name:
            k = ("build", name)
            if k not in got:
                got[k] = str(builders.build_file(val, name, root, token = os.environ.get("HF_TOKEN"),
                                                 dry_run = not download, **build_kw))
            return got[k]
        if isinstance(val, dict) and val.get("repo") and val.get("file"):
            k = (val["repo"], val["file"], val.get("revision"))
            if k not in got:
                dest = root / val["repo"].replace("/", "__")
                if download:
                    from huggingface_hub import hf_hub_download

                    import extnames  # noqa: PLC0415 - real repo / file names in a public copy

                    path = hf_hub_download(extnames.ext(val["repo"]), extnames.ext(val["file"]),
                                           revision = val.get("revision"),
                                           local_dir = str(dest), token = os.environ.get("HF_TOKEN"))
                    builders.verify_download(path, val, name or "")
                    if mirror.is_mirror(val) or val.get("verify"):  # a mirror copy: verified before any use
                        offs = mirror.fetch_official(val, os.environ.get("HF_TOKEN")) if \
                            (val.get("verify") or {}).get("method") == "tensors" else []
                        verified[name or val["file"]] = mirror.verify_entry(val, path, offs)
                    got[k] = path
                else:
                    got[k] = str(dest / val["file"])
            return got[k]
        if isinstance(val, str) and "$CM" in val:
            return val.replace("$CM", str(root))
        return val if name is None else name

    for cell in doc.get("cells") or []:
        files = (cell.get("options") or {}).get("files") or {}
        names = [x for v in files.values() for x in (v if isinstance(v, list) else [v]) if isinstance(x, str)]
        for key, val in list(files.items()):
            files[key] = [one(v) for v in val] if isinstance(val, list) else one(val)
        staged = {n: verified[n] for n in names if n in verified}
        if staged:
            cell.setdefault("options", {})["staged_files"] = staged
    return doc


def hf_home() -> str:
    """The HF_HOME the bench process will read (huggingface_hub's own default when unset)."""
    return os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")


def prefetch(doc: dict, download: bool = True) -> list:
    """The doc's Studio prefetch repos into $HF_HOME/hub through vm_driver.prefetch (one code path for every
    target); returns the keys it handled (all of them on a dry run, which fetches nothing). A whole-repo entry a
    Studio cell fetches itself online before any offline Studio cell needs it is skipped, as on a rented VM
    (vm_driver.fetched_by_studio): the tester's run_tester.py prefetched the gated flux-3-image-klein snapshot (403
    on the tester's account) that the plan's online Studio trial resolves itself."""
    try:
        from . import vm_driver  # noqa: PLC0415 - imported as targets.fetch_files
    except ImportError:
        import vm_driver  # noqa: PLC0415 - run as a script (targets/ is sys.path[0])
    entries = dict(doc.get("prefetch") or {})
    cells = doc.get("cells") or {}
    cells = {c.get("tag"): c for c in cells} if isinstance(cells, list) else cells
    order = [it.get("tag") for vm in doc.get("vms") or [] for slot in vm.get("slots") or [] for it in slot]
    for key, why in vm_driver.fetched_by_studio(entries, order, cells).items():
        print(f"prefetch {key} skipped: {why}")
        entries.pop(key)
    if entries and download:
        vm_driver.prefetch(entries, {"HF_HOME": hf_home(), "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE")})
    return list(entries)


def localise(doc: dict, venv: str | None = None, studio_src: str | None = None) -> dict:
    """Every cell's matrix venv -> venv (the BlineUI server still runs from DIFFUSION_BENCH_BLINE_PYTHON), and
    Studio cells' (in process or over HTTP) options.studio_src -> studio_src unless the cell names one."""
    for cell in doc.get("cells") or []:
        if venv:
            cell["venv"] = venv
        if studio_src and cell.get("backend") in ("studio", "studio_http"):
            cell.setdefault("options", {}).setdefault("studio_src", studio_src)
    return doc


def main() -> int:
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required = True)
    ap.add_argument("--root", required = True)
    ap.add_argument("--write", required = True)
    ap.add_argument("--dry-run", action = "store_true", help = "rewrite paths, download nothing")
    ap.add_argument("--venv", help = "python every cell's driver process runs in")
    ap.add_argument("--studio-src", help = "Studio checkout for studio cells")
    ap.add_argument("--baseline-python", help = "baseline UI venv python the file builders run in")
    ap.add_argument("--baseline-dir", help = "baseline UI checkout the file builders import from")
    ap.add_argument("--convrot-tool", help = "int8 ConvRot quantizer for int8 builds (default $DBENCH_CONVROT_TOOL)")
    a = ap.parse_args()
    doc = json.loads(Path(a.spec).read_text(encoding = "utf-8"))
    for r in refs(doc):
        print("ref", *r)
    kw = {"python": a.baseline_python, "bline_dir": a.baseline_dir, "tool": a.convrot_tool}
    out = localise(resolve(doc, Path(a.root), download = not a.dry_run, build_kw = kw), a.venv, a.studio_src)
    for key in prefetch(doc, download = not a.dry_run):
        print("prefetch", key, "->", hf_home())
    Path(a.write).write_text(json.dumps(out, indent = 1), encoding = "utf-8")
    print(f"wrote {a.write}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
