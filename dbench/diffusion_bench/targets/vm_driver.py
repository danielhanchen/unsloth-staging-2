#!/usr/bin/env python3
"""Run one plan VM's slots on the machine it was rented for (Colab, Kaggle T4x2, a tester's box).

  python vm_driver.py --slice slice.json --root R --bench B --out OUT [--gpus 0,1] [--no-pack]

Generalised from scripts/moss_speed/run.py (groups, downloads, one process per cell) and the imgcmp_nov5 lane runner
(one thread per lane, disjoint CPU halves). What it does:
  - every slot's items run at the same time, one per lane (a lane = one GPU + its CPU share); plan.py guarantees a
    slot never holds Studio and BlineUI side by side, and this driver re-checks it;
  - each item is a fresh process: diffusion_bench/run_cell.py --cell ... in the stack's own venv
    (Studio's install.sh venv for studio / sdcpp / diffusers, BlineUI's venv for blineui), a fresh Inductor cache,
    CUDA_VISIBLE_DEVICES = the lane's GPU, sched_setaffinity = the lane's CPUs, killed (whole process group) at its
    timeout so an arm that never exits cannot stall the VM;
  - the trial slot runs first; a trial without an ok record stops the VM (no timed cell runs on a broken harness);
  - BlineUI / sd.cpp files given as {"repo", "file"} in a cell's options.files (or by bare name through the plan's
    "downloads" map, from the model cards' sources) are downloaded on first use (with HF_TOKEN; the cell processes
    get no token unless they are Studio cells) and deleted after their last use; a card entry with a "build" step is
    built once by diffusion_bench/builders (build_file) and kept;
  - the plan's "prefetch" repos (the cards' formats.<fmt>.studio.sources) are fetched into the VM's HF cache before
    the first slot, since Studio cells load with local_files_only;
  - after every slot the parity gate runs on the pairs whose two probes are done (parity.gate_plan, parity.json
    beside the Studio probe); a VOID group's timed cells are NOT_RUN, never timed;
  - at the end, records, logs (trimmed), server logs and 512 px thumbnails go into one tarball: printed as base64
    between ARTIFACT_B64_BEGIN / _END and, on Kaggle, also written to /kaggle/working/dbench/artifact.tgz.
"""

from __future__ import annotations

import argparse
import base64
import fnmatch
import glob
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

TOKEN_ENV = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_TOKEN_ORG")
DEFAULT_TIMEOUT_S = 5400


def log(*a) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush = True)


def cpu_sets(lanes: int) -> dict:
    n = os.cpu_count() or 1
    if lanes <= 1:
        return {0: set(range(n))}
    k = max(1, n // lanes)
    return {i: set(range(i * k, min(n, (i + 1) * k))) for i in range(lanes)}


def _roles(cell: dict):
    """(key, index | None, value) for every file role of a cell; a list-valued role (several text encoders) yields
    one entry per element."""
    for key, val in ((cell.get("options") or {}).get("files") or {}).items():
        if isinstance(val, list):
            for i, v in enumerate(val):
                yield key, i, v
        else:
            yield key, None, val


def file_refs(cell: dict, downloads: dict | None = None) -> list:
    """(key, repo, file, revision, index, entry) for every downloadable file of a cell: {"repo", "file"} values, and
    bare file names the plan's "downloads" map knows (model cards name baseline-UI files by their template file
    name); list-valued roles element by element."""
    out = []
    for key, i, val in _roles(cell):
        if isinstance(val, str) and downloads and val in downloads:
            val = downloads[val]
        if isinstance(val, dict) and val.get("repo") and val.get("file") and not val.get("build"):
            out.append((key, val["repo"], val["file"], val.get("revision"), i, val))
    return out


def build_refs(cell: dict, downloads: dict | None = None) -> list:
    """(key, file name, card source entry, index) for every file of a cell that a builders/ step makes."""
    out = []
    for key, i, val in _roles(cell):
        if isinstance(val, str) and downloads and isinstance(downloads.get(val), dict) and downloads[val].get("build"):
            out.append((key, val, downloads[val], i))
    return out


def _set(files: dict, key, index, value) -> None:
    if index is None:
        files[key] = value
    else:
        files[key][index] = value


class Files:
    """Download-on-first-use, delete-after-last-use for the {"repo", "file"} refs of a VM's cells."""

    def __init__(self, root: Path, order: list, cells: dict, token: str | None, downloads: dict | None = None,
                 build_kw: dict | None = None):
        self.root, self.token, self.downloads = root / "cm", token, downloads or {}
        # the VM's baseline python / checkout / ConvRot tool, passed to every builder (they run in this process)
        self.build_kw = {k: v for k, v in (build_kw or {}).items() if v}
        self.lock = threading.Lock()
        self.last = {}
        for i, tag in enumerate(order):
            for _, repo, f, rev, _, _ in file_refs(cells[tag], self.downloads):
                self.last[(repo, f, rev)] = i
        self.done = {}
        self.verified = {}  # (repo, file, rev) -> mirror.describe(...) of a verified mirror copy
        self.built = {}  # builder outputs are kept until their last use: a rebuild costs more than the disk
        self.last_build = {}
        for i, tag in enumerate(order):
            for _, name, _, _ in build_refs(cells[tag], self.downloads):
                self.last_build[("build", name)] = i
        self.dl_s = self.build_s = 0.0  # time accounting (driver_summary.json "time")

    def resolve(self, cell: dict) -> dict:
        cell = json.loads(json.dumps(cell))
        files = (cell.get("options") or {}).get("files") or {}
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import builders
        import mirror  # noqa: PLC0415

        for key, repo, f, rev, idx, entry in file_refs(cell, self.downloads):
            with self.lock:
                if (repo, f, rev) not in self.done:
                    from huggingface_hub import hf_hub_download

                    t = time.time()
                    import extnames  # noqa: PLC0415

                    p = hf_hub_download(extnames.ext(repo), extnames.ext(f), revision = rev, local_dir = str(self.root / repo.replace("/", "__")),
                                        token = self.token)
                    builders.verify_download(p, entry)  # the card's pinned sha256, as build_file checks it
                    self.dl_s += time.time() - t
                    log(f"DL {repo}/{f} {os.path.getsize(p) / 2**30:.2f} GiB in {time.time() - t:.0f}s")
                    if mirror.is_mirror(entry) or entry.get("verify"):
                        # a mirror copy: verified against the official checkpoint before any cell may use it (a
                        # failure raises: the cells needing it are BROKEN with the reason, never run on it)
                        t = time.time()
                        offs = mirror.fetch_official(entry, self.token) if \
                            (entry.get("verify") or {}).get("method") == "tensors" else []
                        self.verified[(repo, f, rev)] = mirror.verify_entry(entry, p, offs)
                        res = self.verified[(repo, f, rev)].get("result") or {}
                        log(f"MIRROR {repo}/{f} verified ({self.verified[(repo, f, rev)]['method']}: "
                            f"{res.get('equal', '-')}/{res.get('compared', '-')} tensors) in {time.time() - t:.0f}s")
                    self.done[(repo, f, rev)] = p
            _set(files, key, idx, self.done[(repo, f, rev)])
            if (repo, f, rev) in self.verified:
                name = next((n for n, e in self.downloads.items() if e is entry), f.rsplit("/", 1)[-1])
                cell.setdefault("options", {}).setdefault("staged_files", {})[name] = self.verified[(repo, f, rev)]
        for key, name, entry, idx in build_refs(cell, self.downloads):
            with self.lock:
                if ("build", name) not in self.built:
                    t = time.time()
                    self.built[("build", name)] = str(builders.build_file(entry, name, self.root, token = self.token,
                                                                         **self.build_kw))
                    self.build_s += time.time() - t
                    log(f"BUILT {name} ({entry.get('build')}) in {time.time() - t:.0f}s")
            _set(files, key, idx, self.built[("build", name)])
        for key, val in list(files.items()):
            if isinstance(val, str) and "$CM" in val:
                files[key] = val.replace("$CM", str(self.root))
            elif isinstance(val, list):
                files[key] = [v.replace("$CM", str(self.root)) if isinstance(v, str) else v for v in val]
        return cell

    def release(self, index: int) -> None:
        with self.lock:
            for k, last in list(self.last.items()):
                if last == index and k in self.done:
                    try:
                        os.remove(self.done.pop(k))
                    except OSError:
                        pass
            for k, last in list(self.last_build.items()):
                if last == index and k in self.built:  # a built file nothing later uses: free the disk
                    freed = self._drop_built(self.built.pop(k))
                    log(f"FREED built {k[1]} after its last use ({freed / 2**30:.2f} GiB)")

    def _drop_built(self, path: str) -> int:
        """Remove a built file's link and the payload it points at; returns the bytes freed. build_file returns
        root/<name> -> root/_built/<payload> (or a fetched copy under root): removing only the link kept every payload
        on the VM disk. The payload goes only when it lies under this VM's file root and no still-live built entry
        resolves to it; a link into anything else (the shared hub cache, outside the root) loses only the link."""
        p = Path(path)
        real = Path(os.path.realpath(p))
        own = Path(os.path.realpath(self.root))
        shared = any(os.path.realpath(v) == str(real) for v in self.built.values())
        freed = 0
        try:
            if p.is_symlink():
                p.unlink()
            if real.is_file() and own in real.parents and not shared:
                freed = real.stat().st_size
                real.unlink()
        except OSError as exc:
            log(f"could not free {path}: {exc}")
        return freed


def cell_repos(cell: dict) -> set:
    o = (cell or {}).get("options") or {}
    return {r for r in ((cell or {}).get("model"), o.get("base_repo")) if isinstance(r, str) and "/" in r}


class CacheJanitor:
    """Frees the VM's HF hub cache between comparisons: the hub cache dirs a comparison's Studio cells created
    (base repos, hosted prequant repos Studio fetched) are deleted after its LAST slot unless a later cell names
    the repo. A Colab disk is ~236 GB; five image families' Studio downloads plus the baseline builds filled it
    (no space left mid-round). The next comparison's online probe refetches anything it shares.

    Ownership is per dir: a dir belongs to the comparison whose slot it appeared in, so interleaved comparisons
    (a serial fallback of mirrored lanes alternates them) never free each other's downloads, and a dir kept for a
    later cell stays its owner's to free once that cell is past. Dirs present before the first slot (prefetch)
    are never freed."""

    def __init__(self, cache: str, slots: list, cells: dict):
        self.cache = Path(cache)
        self.cmp_of: list = []
        nxt = None
        for slot in reversed(slots):  # a trial slot (comparison None) belongs to the comparison that follows it
            c = next(((cells.get(it["tag"]) or {}).get("options", {}).get("arm", {}).get("comparison")
                      for it in slot if not it.get("trial")), None) or nxt
            self.cmp_of.insert(0, c)
            nxt = c
        self.later: list = []
        acc: set = set()
        for slot in reversed(slots):
            self.later.insert(0, set(acc))
            for it in slot:
                acc |= cell_repos(cells.get(it["tag"]))
        self.last = {c: i for i, c in enumerate(self.cmp_of) if c is not None}
        self.seen: set | None = None
        self.owner: dict = {}
        self.done: set = set()

    def _dirs(self) -> set:
        return {d.name for d in self.cache.glob("models--*")} if self.cache.is_dir() else set()

    def start(self, si: int) -> None:
        if self.seen is None:
            self.seen = self._dirs()

    def finish(self, si: int) -> list:
        c = self.cmp_of[si]
        now = self._dirs()
        for name in now - (self.seen or set()):
            self.owner[name] = c
        self.seen = (self.seen or set()) | now
        if c is None or self.last.get(c) != si:
            return []
        self.done.add(c)
        keep = {"models--" + r.replace("/", "--") for r in self.later[si]}
        gone = []
        for name in sorted(n for n, o in self.owner.items() if o in self.done and n in now and n not in keep):
            shutil.rmtree(self.cache / name, ignore_errors = True)
            gone.append(name)
            del self.owner[name]
            # a later comparison that fetches it again creates it anew, and owns that copy
            self.seen.discard(name)
        return gone


def localize_studio(cell: dict, root: Path, pythons: dict, rev: dict) -> dict:
    """Point a Studio arm (in-process or HTTP) at the tree vm_setup.sh checked out at UNSLOTH_SHA and its install's
    interpreter, and state the pinned commit run_cell must see (else VOID). Without it an HTTP arm's launcher
    defaulted to main and built a second env: an unpinned Studio timed against the pinned baseline."""
    if cell.get("backend") not in ("studio", "studio_http"):
        return cell
    o = cell.setdefault("options", {})
    o.setdefault("studio_src", str(root / "src" / "unsloth"))
    if cell["backend"] == "studio_http" and pythons.get("studio"):
        o.setdefault("studio_python", pythons["studio"])
    if rev.get("unsloth_sha") and o["studio_src"] == str(root / "src" / "unsloth"):
        o.setdefault("expect_studio_sha", rev["unsloth_sha"])
    return cell


def slot_runs_parity(slot: list, cells: dict) -> bool:
    """True when a slot ran a parity probe: a parity_order item, or (purpose "parity" plans) an ordinary cell
    with options.parity_probe."""
    return any(it.get("parity") or ((cells.get(it["tag"]) or {}).get("options") or {}).get("parity_probe")
               for it in slot)


def online_probe_keeps_token(cell: dict, have_token: bool) -> bool:
    """An untimed online Studio probe (local_files_only false: the parity probe / trial) keeps the VM's HF token in
    its process environment: fetched_by_studio drops the authenticated prefetch of the repos it loads, so a gated
    repo is downloaded by this probe, and run_cell's scrub_tokens would otherwise strip the credential first (an
    auth failure in the trial stops the VM). Only the boolean keep_tokens goes into the cell; the token itself stays
    in the environment the driver already holds (never in the cell, the record or a file). Timed cells load offline
    and stay scrubbed."""
    o = cell.get("options") or {}
    return bool(have_token and cell.get("backend") == "studio" and o.get("local_files_only", True) is False)


def builder_kw(root: Path) -> dict:
    """{"python", "bline_dir", "tool"} vm_setup.sh provisioned under ROOT: the baseline venv python, its checkout and
    the pinned int8 ConvRot quantizer (convrot_tool.txt; absent when no planned file needs it)."""
    rd = lambda name: (root / name).read_text().strip() if (root / name).exists() else None  # noqa: E731
    cdir = root / "src" / "BlineUI"
    return {"python": rd("bline_py.txt"), "bline_dir": str(cdir) if cdir.is_dir() else None,
            "tool": rd("convrot_tool.txt")}


def fetched_by_studio(entries: dict, order: list, cells: dict) -> dict:
    """The prefetch entries a Studio cell will fetch itself before any offline Studio cell needs them: a whole-repo
    entry whose repo is a Studio cell's model / base_repo, where the FIRST Studio cell (in run order) using it loads
    online (local_files_only false: the untimed trial / parity probe). Studio downloads only the files it resolves,
    where a snapshot pulls every variant in the repo (sdxl-turbo: the fp32 and single-file checkpoints, 10 min on
    a T4 VM). {key: why} for the entries to skip."""
    first: dict = {}
    for tag in order:
        c = cells.get(tag) or {}
        if c.get("backend") not in ("studio", "studio_http"):
            continue
        o = c.get("options") or {}
        for repo in {c.get("model"), o.get("base_repo")} - {None}:
            first.setdefault(repo, (tag, o.get("local_files_only", True) is False))
    out = {}
    for key, e in entries.items():
        hit = first.get(e.get("repo"))
        if not e.get("file") and hit and hit[1]:
            out[key] = f"fetched by {hit[0]} (online, untimed) before any offline Studio cell"
    return out


def snapshot_too_big(repo: str, cache: str, frac: float = 0.6, ignore: list | None = None) -> str | None:
    """Why a whole-repo snapshot must not run (None = fine): its files add up to more than ``frac`` of the free disk
    (MiniMaxAI/MiniMax-H3 is ~500 GB of variants; the snapshot filled a 236 GB Colab disk and every later
    download on the VM failed). The Studio cell then fetches only the files it resolves."""
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo, files_metadata = True, token = os.environ.get("HF_TOKEN"))
        need = sum(int(getattr(f, "size", 0) or 0) for f in info.siblings or []
                   if not any(fnmatch.fnmatch(f.rfilename, p) for p in ignore or ()))
        os.makedirs(cache, exist_ok = True)
        free = shutil.disk_usage(cache).free
    except Exception:  # noqa: BLE001 - size unknown: let the snapshot try
        return None
    if need > frac * free:
        return f"repo snapshot {need / 2**30:.0f} GiB > {frac:.0%} of {free / 2**30:.0f} GiB free disk"
    return None


def prefetch(entries: dict, env: dict) -> None:
    """The Studio repos the plan's cells load (cards' formats.<fmt>.studio.sources), into the VM's HF cache."""
    if not entries:
        return
    from huggingface_hub import hf_hub_download, snapshot_download

    # the hub cache the cells read: HF_HUB_CACHE when the environment sets one, else HF_HOME/hub
    cache = env.get("HF_HUB_CACHE") or os.path.join(env["HF_HOME"], "hub")
    for key, e in entries.items():
        t = time.time()
        try:
            if e.get("file"):
                hf_hub_download(e["repo"], e["file"], cache_dir = cache,
                                token = os.environ.get("HF_TOKEN"))
            else:
                ignore = list(e.get("ignore_patterns") or []) or None
                big = snapshot_too_big(e["repo"], cache, ignore = ignore)
                if big:
                    log(f"PREFETCH {key} skipped: {big}")
                    continue
                snapshot_download(e["repo"], cache_dir = cache, ignore_patterns = ignore,
                                  token = os.environ.get("HF_TOKEN"))
            log(f"PREFETCH {key} in {time.time() - t:.0f}s" + (f" (without {e['ignore_patterns']})"
                                                               if e.get("ignore_patterns") else ""))
        except Exception as exc:  # noqa: BLE001 - the Studio cell then fails visibly (local_files_only)
            if no_access(exc):
                # a gated repo this account cannot read (flux-3-image-klein on the tester's account, Nov 20): not a
                # prefetch failure; the Studio arm loads its hosted files, and a cell that does need the repo still
                # fails visibly offline
                log(f"PREFETCH {key} skipped: no access on this account ({type(exc).__name__}, gated, HTTP 403)")
                continue
            log(f"PREFETCH {key} failed: {type(exc).__name__}: {exc}")


def no_access(exc: BaseException) -> bool:
    """True for a Hub refusal of a gated repo (GatedRepoError, HTTP 403); a 401 (a misspelt or private repo looks
    the same), 404, network or disk error stays a failure."""
    if type(exc).__name__ == "GatedRepoError":
        return True
    resp = getattr(exc, "response", None)
    return getattr(resp, "status_code", None) == 403


def _toolkit_common(bench: str):
    """diffusion_bench/common.py by path (this script's own dir holds targets/common.py under the same name)."""
    import importlib.util

    mod = sys.modules.get("dbench_toolkit_common")
    if mod is None:
        spec = importlib.util.spec_from_file_location("dbench_toolkit_common",
                                                      str(Path(bench) / "diffusion_bench" / "common.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules["dbench_toolkit_common"] = mod
        spec.loader.exec_module(mod)
    return mod


def write_not_run(out: Path, cell: dict, why: str, verdict: str = "NOT_RUN") -> None:
    """A NOT_RUN (or BROKEN-before-start) record for a cell the driver did not start, with the reason, so report.py
    shows why instead of MISSING. An ok record from an earlier launch is never overwritten. A disk that cannot take
    the record (full, or Kaggle's file system gone read-only) is logged, never fatal: the driver still packs what
    it has."""
    try:
        _write_not_run(out, cell, why, verdict)
    except OSError as exc:
        log(f"could not write the {verdict} record of {cell.get('tag')}: {exc}")


def _write_not_run(out: Path, cell: dict, why: str, verdict: str) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import schema

    d = out / cell["tag"]
    rp = d / "record.json"
    if rp.exists():
        try:
            if json.loads(rp.read_text()).get("verdict") == "ok":
                return
        except ValueError:
            pass
    rec = schema.new_record(cell)
    rec.update({"verdict": verdict, "reason": why, "backend": cell.get("backend"), "phase": "driver"})
    d.mkdir(parents = True, exist_ok = True)
    rp.write_text(schema.dumps(rec))


def run_item(item: dict, cell: dict, a, env_base: dict, pythons: dict, cpus: set, gpu: str, results: dict) -> None:
    tag = item["tag"]
    out = Path(a.out)
    logp = out / "_logs" / f"{tag}.log"
    rp = out / tag / "record.json"
    if rp.exists():
        try:
            if json.loads(rp.read_text()).get("verdict") == "ok":
                log(f"[gpu{gpu}] {tag} SKIP (ok record from an earlier launch)")
                results[tag] = {"rc": 0, "s": 0, "gpu": gpu, "skipped": True, "verdict": "ok"}
                return
        except ValueError:
            pass
    stack = cell["backend"]
    py = pythons["bline"] if stack == "blineui" else pythons["studio"]
    env = dict(env_base)
    if stack != "studio":
        for k in TOKEN_ENV:
            env.pop(k, None)
    if gpu != "":
        env["CUDA_VISIBLE_DEVICES"] = gpu
    ind = Path(a.root) / "ind" / tag
    shutil.rmtree(ind, ignore_errors = True)
    env["DBENCH_FRESH_CACHE_ROOT"] = str(ind)  # run_cell's fresh Inductor / Triton / Studio-home root for this cell
    env.update({k: str(v) for k, v in (cell.get("env") or {}).items()})
    cmd = [py, "-u", str(Path(a.bench) / "diffusion_bench" / "run_cell.py"), "--cell", json.dumps(cell), "--out", str(out)]
    timeout = float(cell.get("timeout_s") or a.timeout)
    clk = subprocess.run(["bash", "-c", f"nvidia-smi -i {gpu or 0} --query-gpu=clocks.sm,clocks.max.sm,temperature.gpu,"
                          "memory.used,utilization.gpu --format=csv,noheader 2>/dev/null"],
                         capture_output = True, text = True).stdout.strip()
    t0 = time.time()
    rc = None
    with open(logp, "w") as lf:
        # own process group (so a timeout kills the BlineUI server too), pinned to the lane's CPUs
        kw = {"preexec_fn": _setsid_affinity(cpus)} if cpus and hasattr(os, "sched_setaffinity") else \
            {"start_new_session": True}
        p = subprocess.Popen(cmd, stdout = lf, stderr = subprocess.STDOUT, env = env, cwd = str(Path(a.root) / "ws"), **kw)
        try:
            rc = p.wait(timeout = timeout)
        except subprocess.TimeoutExpired:
            # the whole tree (servers started in their own session included), on every OS: common.kill_tree
            _toolkit_common(a.bench).kill_tree(p, timeout = 5)
            try:
                os.killpg(p.pid, signal.SIGKILL)  # anything re-parented but still in the cell's group
            except (ProcessLookupError, PermissionError, AttributeError, OSError):
                pass
            p.wait()
            rc = "TIMEOUT"
    dt = round(time.time() - t0, 1)
    shutil.rmtree(ind, ignore_errors = True)
    verdict = None
    try:
        rec = json.loads(rp.read_text())
        verdict = rec.get("verdict")
        if rc == "TIMEOUT":
            rec.update({"verdict": "BROKEN", "reason": f"killed after {timeout:.0f}s (never exited)"})
            rp.write_text(json.dumps(rec, indent = 1))
            verdict = "BROKEN"
    except (OSError, ValueError):
        pass
    tail = logp.read_text(errors = "replace").splitlines()
    if verdict is None:  # died (or was killed) before writing its record: say so instead of leaving it MISSING
        why = (f"run_cell {'killed after ' + format(timeout, '.0f') + 's' if rc == 'TIMEOUT' else f'exited rc={rc}'}"
               " without a record; log tail: " + " | ".join(tail[-3:])[:400])
        write_not_run(out, cell, why, verdict = "BROKEN")
        verdict = "BROKEN"
    log(f"=== [gpu{gpu}] {tag} ({stack}) rc={rc} verdict={verdict} {dt}s idle_gpu[{clk}]")
    for line in [ln for ln in tail if ln.startswith(("CELL_SUMMARY", "VERDICT", "RECORD"))][-5:]:
        log("  | " + line[:2000])
    if rc != 0:
        for line in tail[-30:]:
            log("  ! " + line[:400])
    results[tag] = {"rc": rc, "s": dt, "gpu": gpu, "verdict": verdict, "start": t0, "cpus": sorted(cpus),
                    "idle_gpu": clk}


def _setsid_affinity(cpus: set):
    def f():
        os.setsid()
        os.sched_setaffinity(0, cpus)
    return f


def write_summary(out: Path, vm: dict, t0: float, stopped, results: dict, timing: dict | None = None) -> None:
    """timing: where the driver's time went (prefetch, baseline-file downloads and builds; the items' own seconds
    are in results), read by report.py's time accounting with setup_log.txt's SETUP_S."""
    (out / "driver_summary.json").write_text(json.dumps({"vm": vm["vm_id"], "target": vm["target"],
                                                        "elapsed_s": round(time.time() - t0, 1), "stopped": stopped,
                                                        "time": timing or {}, "results": results},
                                                       indent = 1, default = str))


def timing(files, prefetch_s: float) -> dict:
    return {"prefetch_s": prefetch_s, "downloads_s": round(files.dl_s, 1), "builds_s": round(files.build_s, 1)}


def pack(out: Path) -> bytes:
    buf = io.BytesIO()

    def addbytes(tf, name, data):
        ti = tarfile.TarInfo(name)
        ti.size = len(data)
        tf.addfile(ti, io.BytesIO(data))

    with tarfile.open(fileobj = buf, mode = "w:gz") as tf:
        # parity results also sit one level down (<studio probe>/vs_<baseline probe>/parity.json, parity.gate_plan)
        for pat in ("*/record.json", "driver_summary.json", "setup_*.txt", "*/parity*.json", "*/vs_*/parity*.json",
                    "*/sigmas*.json"):
            for p in glob.glob(str(out / pat)):
                tf.add(p, arcname = os.path.relpath(p, out))
        for p in glob.glob(str(out / "_logs" / "*.log")) + glob.glob(str(out / "*" / "_bline" / "server.log")) + \
                glob.glob(str(out / "*" / "cell.log")):
            lines = Path(p).read_text(errors = "replace").replace("\r", "\n").splitlines()
            if len(lines) > 3000:
                lines = lines[:600] + ["...[trimmed]..."] + lines[-2000:]
            addbytes(tf, os.path.relpath(p, out), "\n".join(ln[:2000] for ln in lines).encode())
        # the scoring media under the names the records hold (PNG images, NPZ video frames: lossless, what
        # score.load_media reads), up to DBENCH_PACK_MEDIA_MAX_MB (the base64 notebook transport has a limit);
        # anything left out is listed in pack_manifest.json, and the scorer reports those renders as unpaired
        cap = float(os.environ.get("DBENCH_PACK_MEDIA_MAX_MB") or 1500) * 2**20
        used, kept, dropped = 0, [], []
        media = sorted(glob.glob(str(out / "*" / "*.png"))) + sorted(glob.glob(str(out / "*" / "*.npz")))
        for p in media:
            size = os.path.getsize(p)
            if used + size > cap:
                dropped.append(os.path.relpath(p, out))
                continue
            tf.add(p, arcname = os.path.relpath(p, out))
            used += size
            kept.append(os.path.relpath(p, out))
        addbytes(tf, "pack_manifest.json", json.dumps({"media_kept": len(kept), "media_bytes": used,
                                                       "media_dropped": dropped, "cap_bytes": cap}).encode())
    return buf.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slice", required = True)
    ap.add_argument("--root", required = True)
    ap.add_argument("--bench", required = True, help = "dir holding diffusion_bench/")
    ap.add_argument("--out", required = True)
    ap.add_argument("--gpus", default = "", help = "lane -> GPU index list (default: lane i = GPU i)")
    ap.add_argument("--timeout", type = float, default = DEFAULT_TIMEOUT_S, help = "per item, seconds")
    ap.add_argument("--no-pack", action = "store_true")
    ap.add_argument("--dry-run", action = "store_true", help = "print the schedule, run nothing")
    a = ap.parse_args()
    sl = json.loads(Path(a.slice).read_text())
    vm, cells = sl["vm"], sl["cells"]
    out, root = Path(a.out), Path(a.root)
    (out / "_logs").mkdir(parents = True, exist_ok = True)
    (root / "ws").mkdir(parents = True, exist_ok = True)
    lanes = int(vm.get("lanes") or 1)
    gpus = [g for g in a.gpus.split(",") if g != ""] or ([""] * lanes if vm.get("kind") == "cpu"
                                                         else [str(i) for i in range(lanes)])
    cpus = cpu_sets(lanes)
    order = [it["tag"] for slot in vm["slots"] for it in slot]
    log(f"VM {vm['vm_id']} {vm['target']} lanes={lanes} gpus={gpus} cpus={ {k: f'{min(v)}-{max(v)}' for k, v in cpus.items()} } "
        f"slots={len(vm['slots'])} plan={sl.get('plan_name')} {sl.get('plan_sha256')}")
    for slot in vm["slots"]:
        stacks = {it["stack"] for it in slot if it.get("timed")}
        if len(stacks) > 1:
            raise SystemExit(f"slot {[it['tag'] for it in slot]} times {sorted(stacks)} side by side: refused")
    if a.dry_run:
        for i, slot in enumerate(vm["slots"]):
            log(f"slot {i}: " + " || ".join(f"lane{it['lane']}=gpu{gpus[it['lane']]} {it['tag']}" for it in slot))
        return 0
    rf = lambda name: (root / name).read_text().strip() if (root / name).exists() else sys.executable  # noqa: E731
    pythons = {"studio": rf("studio_py.txt"), "bline": rf("bline_py.txt")}
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update({"HF_HOME": str(root / "hf"), "UNSLOTH_STUDIO_HOME": str(root / "studio_home"), "PYTHONUNBUFFERED": "1",
                "UNSLOTH_DISABLE_AUTO_UPDATES": "1", "HF_XET_HIGH_PERFORMANCE": "1", "WORKSPACE": str(root / "ws"),
                "DBENCH_DIR": str(Path(a.bench) / "diffusion_bench"), "DIFFUSION_BENCH_BLINE_PYTHON": pythons["bline"],
                "DIFFUSION_BENCH_BLINE_DIR": str(root / "src" / "BlineUI"), "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "DBENCH_SCRIPTS_SHA": str((sl.get("revisions") or {}).get("scripts_sha") or ""),
                "DBENCH_PLAN_SHA256": str(sl.get("plan_sha256") or "")})
    # the baseline-file builders run from THIS process: give them the cells' hub cache, so an official shard the
    # Studio cell already fetched (Z-Image's transformer, FLUX.2's encoder) is read, not downloaded a second time
    # into ~/.cache (on Kaggle the second copy filled the disk and the file system went read-only)
    for k in ("HF_HOME", "HF_XET_HIGH_PERFORMANCE"):
        os.environ[k] = env[k]
    # the rented VM's hub cache is its own: a builder may delete the official shards it downloaded itself once its
    # merged source is written (builders/build_bline_files.py FETCHED_NEW), never a file a Studio probe fetched
    os.environ["DBENCH_BUILDER_OWNS_HF_CACHE"] = "1"
    sd = root / "sdcpp.json"
    if sd.exists():
        try:
            env["DIFFUSION_BENCH_SDCPP_BIN"] = json.loads(sd.read_text().strip().splitlines()[-1])["bin"]
        except (ValueError, KeyError, IndexError):
            log("sdcpp.json unreadable: sdcpp cells will try their own setup")
    files = Files(root, order, cells, os.environ.get("HF_TOKEN"), sl.get("downloads"), build_kw = builder_kw(root))
    pre = dict(sl.get("prefetch") or {})
    for key, why in fetched_by_studio(pre, order, cells).items():
        log(f"PREFETCH {key} skipped: {why}")
        pre.pop(key)
    t_pre = time.time()
    prefetch(pre, env)
    prefetch_s = round(time.time() - t_pre, 1)
    sys.path.insert(0, str(Path(a.bench) / "diffusion_bench"))
    import parity as PG

    gate_plan_doc = {"comparisons": sl.get("comparisons") or []}
    void: list = []
    results: dict = {}
    stop_reason = None
    t_all = time.time()
    janitor = CacheJanitor(env.get("HF_HUB_CACHE") or os.path.join(env["HF_HOME"], "hub"), vm["slots"], cells)
    for si, slot in enumerate(vm["slots"]):
        janitor.start(si)
        if stop_reason:
            for it in slot:
                results[it["tag"]] = {"rc": None, "verdict": "NOT_RUN", "reason": stop_reason}
                write_not_run(Path(a.out), cells[it["tag"]], stop_reason)
            continue
        threads = []
        for it in slot:
            why = PG.gated_out(gate_plan_doc, cells[it["tag"]], void)
            if why:
                results[it["tag"]] = {"rc": None, "verdict": "NOT_RUN", "reason": why}
                write_not_run(Path(a.out), cells[it["tag"]], why)
                log(f"=== {it['tag']} NOT_RUN: {why}")
                continue
            try:
                cell = files.resolve(cells[it["tag"]])
                cell = localize_studio(cell, root, pythons, sl.get("revisions") or {})
                if online_probe_keeps_token(cell, bool(os.environ.get("HF_TOKEN"))):
                    cell["keep_tokens"] = True
            except Exception as exc:  # noqa: BLE001 - a missing file is this item's failure, not the VM's
                results[it["tag"]] = {"rc": None, "verdict": "BROKEN", "reason": f"file download: {exc}"}
                write_not_run(Path(a.out), cells[it["tag"]], f"file download: {exc}", verdict = "BROKEN")
                log(f"=== {it['tag']} BROKEN before start: {exc}")
                continue
            lane = int(it.get("lane") or 0)
            th = threading.Thread(target = run_item, args = (it, cell, a, env, pythons, cpus[lane] if lanes > 1 else set(),
                                                             gpus[lane] if gpus else "", results))
            th.start()
            threads.append(th)
        for th in threads:
            th.join()
        for it in slot:
            files.release(order.index(it["tag"]))
            if it.get("trial") and (results.get(it["tag"]) or {}).get("verdict") != "ok":
                stop_reason = f"trial {it['tag']} not ok ({(results.get(it['tag']) or {}).get('verdict')}): VM stopped"
                log("TRIAL_FAILED " + stop_reason)
        if slot_runs_parity(slot, cells) and gate_plan_doc["comparisons"]:
            g = PG.gate_plan(gate_plan_doc, out)
            void = PG.blocked_groups(g)  # VOID and pending (a probe with no record) both block timing
            for pr in g["pairs"]:
                log(f"PARITY{' NEGATIVE_CONTROL ' + pr['control_verdict'] if pr.get('control_verdict') else ''} "
                    f"{pr['verdict']} {pr['comparison']} / {pr['format']}"
                    + (f": {'; '.join(pr['reasons'])[:300]}" if pr["reasons"] else ""))
        gone = janitor.finish(si)
        if gone:
            log(f"FREED hub cache after {janitor.cmp_of[si]}: {', '.join(gone)}; free disk "
                f"{shutil.disk_usage(str(janitor.cache) if janitor.cache.is_dir() else '/').free / 2**30:.0f} GiB")
        write_summary(out, vm, t_all, stop_reason, results, timing(files, prefetch_s))
    write_summary(out, vm, t_all, stop_reason, results, timing(files, prefetch_s))
    log("SLOTS_DONE", json.dumps({k: v.get("verdict") for k, v in results.items()}))
    if a.no_pack:
        return 0
    data = pack(out)
    log("ARTIFACT_BYTES", len(data))
    if os.path.isdir("/kaggle/working"):
        os.makedirs("/kaggle/working/dbench", exist_ok = True)
        Path("/kaggle/working/dbench/artifact.tgz").write_bytes(data)
    s = base64.b64encode(data).decode()
    print("ARTIFACT_B64_BEGIN")
    for i in range(0, len(s), 76 * 2000):
        print(s[i:i + 76 * 2000])
    print("ARTIFACT_B64_END", flush = True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
