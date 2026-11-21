"""Check registry, per-check context, media assertions, and the loop that runs checks against one surface with a
timeout each (a hang is a FAIL and the surface is recovered, never a stuck run)."""

from __future__ import annotations

import fnmatch
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from surfaces import Gen, Hang, HarnessHang, LoadError, Refused, SurfaceError

PASS, FAIL, SKIP, INFO = "PASS", "FAIL", "SKIP", "INFO"
VOID = "VOID"   # the harness failed before the check reached its subject (a worker that never started), or an
                # identity comparison whose two renders ran under different load plans (compare_identity)

# Status fields of a loaded pipeline that change its numerics: the load plan. The auto planner picks offload and
# VAE tiling from the free memory it sees at load time, so on a shared GPU two loads of the same config can land on
# different plans (seen: offload=model tiling=True on a nearly full card, resident and untiled later), and their
# renders differ by construction. memory_mode / speed_optims are left out on purpose: memory_mode is the request,
# not the plan (image.memory_modes asserts fast == low_vram), and speed_optims is reported live (graphs captured).
PLAN_KEYS = ("device", "dtype", "offload_policy", "cpu_offload", "vae_tiling", "speed_mode", "transformer_quant",
             "text_encoder_quant", "attention_backend")
# Only these follow free memory; a difference in the rest (quant, dtype, speed mode, attention) is what a check
# like image.quant_switch exists to catch, so it stays a FAIL.
PRESSURE_PLAN_KEYS = ("device", "offload_policy", "cpu_offload", "vae_tiling")


_FREE_SLOW = {}   # a device whose SMI query was slow once is not asked again (one per check adds up)


def gpu_free_mib(gpu) -> Optional[int]:
    """Free MiB on the surface's GPU (driver view: every tenant's memory counts), None when unreadable."""
    if gpu in (None, "") or _FREE_SLOW.get(str(gpu)):
        return None
    t0 = time.perf_counter()
    try:
        import common as C

        q = C.gpu_query(str(gpu))
        return int(q["total_mib"] - q["used_mib"]) if q.get("total_mib") else None
    except Exception:  # noqa: BLE001
        return None
    finally:
        if time.perf_counter() - t0 > 5:
            _FREE_SLOW[str(gpu)] = True


# A check failing faster than this right after a hang recovery is retried once (see Runner.run).
KNOCK_ON_S = 1.0

# Unload timing of the run, for checks whose answer can wait behind an unload. An unload under REAL_UNLOAD_S is a
# no-op (nothing resident) and is not counted. A run is "slow to unload" when it has at least SLOW_UNLOAD_MIN_N real
# unloads and their median is at least SLOW_UNLOAD_MEDIAN_S: tiny models unload in a median 0.9 to 1.5 s over HTTP on
# a quiet host (switchboard runs of PRs 14003 / 14008 / 14009), 2.5 s on a host at load ~1000 (PR 13860's run, one
# unload 40.9 s), so the median of the whole run says the host, not the check, was slow.
REAL_UNLOAD_S, SLOW_UNLOAD_MIN_N, SLOW_UNLOAD_MEDIAN_S = 0.05, 3, 2.0
POLL_TIMEOUT_S, POLL_TIMEOUT_X = 30.0, 3.0


def unload_slowness(unload_s: list) -> tuple[bool, Optional[float]]:
    """(slow, median s) over the run's real unloads (see SLOW_UNLOAD_MEDIAN_S); median None when none counted."""
    real = sorted(x for x in unload_s if x >= REAL_UNLOAD_S)
    if not real:
        return False, None
    n = len(real)
    med = real[n // 2] if n % 2 else (real[n // 2 - 1] + real[n // 2]) / 2
    return n >= SLOW_UNLOAD_MIN_N and med >= SLOW_UNLOAD_MEDIAN_S, round(med, 2)


def poll_timeout_for(unload_s: list) -> float:
    """Per-poll deadline for a request that can wait behind an unload: POLL_TIMEOUT_S, or POLL_TIMEOUT_X times the
    slowest unload this run has measured when that is longer."""
    return round(max([POLL_TIMEOUT_S] + [POLL_TIMEOUT_X * x for x in unload_s]), 1)

class Skip(Exception):
    pass


@dataclass
class Check:
    name: str
    fn: Callable
    tier: str = "fast"
    needs: Optional[str] = None  # a model config name the runner loads first (None: the check manages loads)
    timeout: float = 180
    surfaces: tuple = ("inproc", "http")
    doc: str = ""


CHECKS: list[Check] = []


def check(name: str, tier: str = "fast", needs: Optional[str] = None, timeout: float = 180,
          surfaces: tuple = ("inproc", "http")):
    def deco(fn):
        CHECKS.append(Check(name, fn, tier, needs, timeout, surfaces, (fn.__doc__ or "").strip().split("\n")[0]))
        return fn
    return deco


# ---------------------------------------------------------------------------------------------- media stats
def img_stats(a) -> dict:
    import numpy as np

    a = np.asarray(a)
    f = a.astype(np.float32)
    return {"shape": list(a.shape), "mean": round(float(f.mean()), 2), "std": round(float(f.std()), 2),
            "min": int(a.min()), "max": int(a.max()),
            "colors": int(len(np.unique(a.reshape(-1, a.shape[-1])[:: max(1, a.size // 3 // 65536)], axis = 0)))}


def image_sane(a) -> tuple[bool, dict]:
    """Not black, not white, not a flat fill, not a few-colour posterised mess (what NaN latents decode to)."""
    s = img_stats(a)
    ok = s["std"] > 8 and 8 < s["mean"] < 247 and s["colors"] > 500
    return ok, s


def mad(a, b) -> float:
    import numpy as np

    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return float("inf")
    return round(float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean()), 4)


def identical(a, b) -> bool:
    import numpy as np

    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and bool((a == b).all())


def load_plan(status: Optional[dict]) -> Optional[dict]:
    """The PLAN_KEYS subset of a status answer, None when there is none (nothing loaded, status unreadable)."""
    if not isinstance(status, dict) or not status.get("loaded"):
        return None
    return {k: status.get(k) for k in PLAN_KEYS if k in status}


def plan_diff(plan_a: Optional[dict], plan_b: Optional[dict]) -> dict:
    """{key: [a, b]} for every plan field the two plans disagree on ({} when either plan is unknown)."""
    if not plan_a or not plan_b:
        return {}
    return {k: [plan_a.get(k), plan_b.get(k)] for k in sorted(set(plan_a) | set(plan_b))
            if plan_a.get(k) != plan_b.get(k)}


def compare_identity(a, b, plan_a: Optional[dict] = None, plan_b: Optional[dict] = None) -> tuple[str, dict]:
    """Verdict of a bit-identity check between a render and its reference: PASS when identical; VOID when they
    differ but ran under different memory-driven plans (PRESSURE_PLAN_KEYS: the comparison measured the planner,
    not the subject); FAIL otherwise, including any other plan difference and an unknown plan."""
    if identical(a, b):
        return PASS, {"mad": 0.0, "plan_ref": plan_a, "plan_now": plan_b}
    diff = plan_diff(plan_a, plan_b)
    if diff and set(diff) <= set(PRESSURE_PLAN_KEYS):
        return VOID, {"mad": mad(a, b), "plan_diff": diff, "plan_ref": plan_a, "plan_now": plan_b}
    return FAIL, {"mad": mad(a, b), "plan": plan_b if plan_b else None}


def video_stats(frames) -> dict:
    import numpy as np

    f = np.asarray(frames).astype(np.float32)
    if f.ndim != 4 or f.shape[0] == 0:
        return {"shape": list(f.shape), "empty": True}
    steps = np.abs(np.diff(f, axis = 0)).mean(axis = (1, 2, 3)) if f.shape[0] > 1 else np.zeros(1)
    return {"shape": list(f.shape), "mean": round(float(f.mean()), 2), "std": round(float(f.std()), 2),
            "motion_mean": round(float(steps.mean()), 3), "motion_min": round(float(steps.min()), 3),
            "first_last_mad": round(float(np.abs(f[0] - f[-1]).mean()), 3),
            "frozen_pairs": int((steps < 0.05).sum())}


# ---------------------------------------------------------------------------------------------- context
@dataclass
class Result:
    surface: str
    check: str
    tier: str
    status: str = PASS
    wall_s: float = 0.0
    failures: list = field(default_factory = list)
    infos: dict = field(default_factory = dict)
    evidence: dict = field(default_factory = dict)
    error: Optional[str] = None
    doc: str = ""
    n_expects: int = 0
    voids: list = field(default_factory = list)   # identity comparisons voided by a load plan mismatch


class Ctx:
    def __init__(self, runner: "Runner", check: Check, res: Result):
        self.runner, self.check, self.res = runner, check, res
        self.surface = runner.surface
        self.cfgs = runner.cfgs
        self.dir = runner.out / runner.surface.name / check.name
        self.name = runner.surface.name
        self.http = self.name == "http"
        self.inproc = self.name == "inproc"
        self.tiny = runner.tier == "tiny"

    def sane(self, a) -> tuple:
        """image_sane for real models; for the tiny random-weight pipelines only shape / dtype (their output is
        noise by construction, so "not black" means nothing)."""
        if self.tiny:
            import numpy as np

            arr = np.asarray(a)
            return arr.ndim == 3 and arr.shape[-1] == 3 and arr.dtype == np.uint8, img_stats(arr)
        return image_sane(a)

    def long(self, name: Optional[str] = None) -> dict:
        """A render long enough to catch mid-flight (cancel / unload during generation)."""
        return dict(self.cfgs[name or self.check.needs or "img_a"].get("long")
                    or {"width": 1024, "height": 1024, "steps": 80})

    # ----------------------------------------------------------------------------------------- verdicts
    def expect(self, cond: bool, what: str, **evidence) -> bool:
        self.res.n_expects += 1
        if evidence:
            self.res.evidence[what] = evidence
        if not cond:
            self.res.failures.append(what)
        return bool(cond)

    def plan(self, kind: str = "image") -> Optional[dict]:
        """The load plan of the resident `kind` pipeline (load_plan of its status), None when unreadable."""
        try:
            return load_plan(self.surface.status(kind))
        except Exception:  # noqa: BLE001 - an unknown plan only means a difference stays a FAIL
            return None

    def expect_identical(self, a, b, what: str, plan_ref: Optional[dict] = None,
                         plan_now: Optional[dict] = None) -> bool:
        """expect(identical(a, b)) that reports VOID instead of FAIL when the reference and this render ran under
        different load plans (compare_identity). Same plan (or an unknown one) and different bytes still FAIL."""
        status, ev = compare_identity(a, b, plan_ref, plan_now)
        if status == VOID:
            self.res.evidence[what] = ev
            self.res.voids.append(f"{what}: not comparable, load plans differ "
                                  + ", ".join(f"{k} {x!r} vs {y!r}" for k, (x, y) in ev["plan_diff"].items()))
            return False
        return self.expect(status == PASS, what, **ev)

    def info(self, key: str, value: Any) -> None:
        self.res.infos[key] = value

    def evidence(self, key: str, value: Any) -> None:
        self.res.evidence[key] = value

    def skip(self, why: str):
        raise Skip(why)

    def save(self, name: str, arr) -> str:
        from PIL import Image
        import numpy as np

        self.dir.mkdir(parents = True, exist_ok = True)
        a = np.asarray(arr)
        if a.ndim == 4:  # a clip: first / middle / last frame side by side
            a = np.concatenate([a[0], a[len(a) // 2], a[-1]], axis = 1)
        p = self.dir / f"{name}.png"
        Image.fromarray(a).save(p)
        return str(p.relative_to(self.runner.out))

    # ----------------------------------------------------------------------------------------- model state
    def cfg(self, name: str) -> dict:
        return self.cfgs[name]

    def load(self, name: str, timeout: float = 900, **overrides) -> dict:
        cfg = self.cfgs[name]
        opts = {**cfg["opts"], **overrides}
        t0 = time.perf_counter()
        st = self.surface.load(cfg["kind"], cfg["model"], timeout = timeout, **opts)
        self.runner.loaded[cfg["kind"]] = name if not overrides else f"{name}*"
        if self.http and cfg["kind"] == "video":  # the arbiter evicts the image model (and the reverse)
            self.runner.loaded["image"] = None
        if self.http and cfg["kind"] == "image":
            self.runner.loaded["video"] = None
        self.res.evidence.setdefault("loads", []).append({"cfg": name, "overrides": overrides,
                                                           "s": round(time.perf_counter() - t0, 1)})
        return st

    def unload(self, kind: str = "image") -> dict:
        t0 = time.perf_counter()
        st = self.surface.unload(kind)
        self.runner.unload_s.append(round(time.perf_counter() - t0, 3))
        self.runner.loaded[kind] = None
        return st

    def poll_timeout(self) -> float:
        return poll_timeout_for(self.runner.unload_s)

    def void(self, why: str) -> None:
        """The check could not reach its subject for a harness / host reason: VOID unless something else FAILed."""
        self.res.voids.append(why)

    def forget(self, kind: Optional[str] = None) -> None:
        """The check changed what is loaded in a way the runner cannot track."""
        for k in ([kind] if kind else ["image", "video"]):
            self.runner.loaded[k] = "?"

    # ----------------------------------------------------------------------------------------- renders
    def gen(self, cfg: Optional[str] = None, prompt: str = "a lighthouse on a cliff at sunset, oil painting",
            timeout: float = 240, **params) -> Gen:
        name = cfg or self.check.needs
        base = dict(self.cfgs[name]["gen"].get(self.name, self.cfgs[name]["gen"].get("*", {})))
        base.update(params)
        kind = self.cfgs[name]["kind"]
        return self.surface.generate(kind, timeout = timeout, prompt = prompt, **base)

    def outcome(self, fn: Callable, what: str) -> tuple:
        """Run one call that may legitimately be refused. Returns ("ok", value) or ("refused" | "load_error", err).
        A fault (5xx / unexpected exception) or a hang records a failure and returns ("fault" | "hang", err)."""
        try:
            return "ok", fn()
        except Refused as e:
            return "refused", e
        except LoadError as e:
            return "load_error", e
        except Hang as e:
            self.expect(False, f"{what}: no hang", detail = e.detail)
            self.runner._hang_stacks(self, self.res, what)
            return "hang", e
        except SurfaceError as e:
            self.expect(False, f"{what}: no server fault (5xx / crash)", status = e.status, detail = e.detail[:600],
                        exc = e.extra.get("exc"), tb = (e.extra.get("tb") or "")[-1500:] or None)
            return "fault", e


# ---------------------------------------------------------------------------------------------- runner
class Runner:
    def __init__(self, surface, cfgs: dict, out: Path, tier: str, only: list, log=print):
        self.surface, self.cfgs, self.out, self.tier, self.only, self.log = surface, cfgs, Path(out), tier, only, log
        self.loaded: dict = {"image": None, "video": None}
        self.unload_s: list = []   # wall s of every Ctx.unload this run (unload_slowness / poll_timeout_for)
        self._recovered = False   # the last check ended by recovering the surface (a hang)
        self.results: list[Result] = []

    def selected(self) -> list[Check]:
        tiers = {"fast": ("fast",), "full": ("fast", "full"), "tiny": ("fast", "full", "tiny")}[self.tier]
        out = []
        for c in CHECKS:
            if c.tier not in tiers or self.surface.name not in c.surfaces:
                continue
            if self.only and not any(fnmatch.fnmatch(c.name, p) or p in c.name for p in self.only):
                continue
            out.append(c)
        return out

    def ensure(self, ctx: Ctx, name: str) -> None:
        kind = self.cfgs[name]["kind"]
        if self.loaded.get(kind) == name:
            return
        if kind == "video" and self.loaded.get("image") and self.surface.name == "inproc":
            ctx.unload("image")  # one pipeline resident at a time, as Studio's arbiter keeps it over HTTP
        if kind == "image" and self.loaded.get("video") and self.surface.name == "inproc":
            ctx.unload("video")
        ctx.load(name)

    def _hang_stacks(self, ctx: Ctx, res: Result, what: str = "check timeout") -> None:
        """On every Hang, before the surface is recovered: the model process's thread stacks appended to this
        check's hang_stacks.txt, so a deadlock can be told from a lost response."""
        fn = getattr(self.surface, "dump_stacks", None)
        if fn is None:
            return
        try:
            text = fn()
        except Exception as e:  # noqa: BLE001 - evidence only
            text = f"stack dump failed: {type(e).__name__}: {e}"
        path = ctx.dir / "hang_stacks.txt"
        try:
            path.parent.mkdir(parents = True, exist_ok = True)
            with open(path, "a") as fh:
                fh.write(f"===== hang: {what} ({time.strftime('%H:%M:%S')})\n{text}\n")
            res.evidence["hang_stacks_path"] = str(path)
        except OSError:
            pass
        res.evidence["hang_stacks"] = text[-3000:]

    def _attempt(self, c) -> Result:
        """Run one check once, recovering the surface after a hang. Sets self._recovered."""
        res = Result(self.surface.name, c.name, c.tier, doc = c.doc)
        ctx = Ctx(self, c, res)
        box: dict = {}

        def body():
            try:
                if c.needs:
                    self.ensure(ctx, c.needs)
                c.fn(ctx)
            except Skip as e:
                box["skip"] = str(e)
            except SurfaceError as e:
                box["error"] = e
            except Exception:  # noqa: BLE001 - a bug in a check is reported, not fatal
                box["crash"] = traceback.format_exc()[-2500:]

        try:
            res.evidence["vram_mib_before"] = self.surface.mem().get("vram_mib")
        except Exception:  # noqa: BLE001
            pass
        free = gpu_free_mib(getattr(self.surface, "gpu", None))
        if free is not None:   # switchboard fairness: what other tenants left this check (external.py)
            res.evidence["gpu_free_mib_start"] = free
        t0 = time.perf_counter()
        th = threading.Thread(target = body, daemon = True)
        th.start()
        th.join(c.timeout + (600 if c.needs and self.loaded.get(self.cfgs[c.needs]["kind"]) != c.needs else 0))
        res.wall_s = round(time.perf_counter() - t0, 2)
        if th.is_alive():
            res.status, res.error = FAIL, f"check timed out after {res.wall_s:.0f}s (hang)"
            self.log(f"  [{self.surface.name}] {c.name}: TIMEOUT, recovering the surface")
            self._hang_stacks(ctx, res)
            self._recovered = True
            try:
                self.surface.recover()
            except Exception as e:  # noqa: BLE001
                res.error += f"; recover failed: {e}"
            self.loaded = {"image": None, "video": None}
        elif "skip" in box:
            res.status, res.error = SKIP, box["skip"]
        elif "error" in box and isinstance(box["error"], HarnessHang):
            e = box["error"]
            res.status, res.error = VOID, f"harness hang: {e.detail[:800]}"
            res.evidence["startup_dump"] = e.extra.get("dump")
            res.evidence["startup_dump_path"] = e.extra.get("dump_path")
        elif "error" in box:
            e = box["error"]
            clean = isinstance(e, (Refused, LoadError))
            res.status = FAIL
            res.error = f"{type(e).__name__}{f' {e.status}' if e.status else ''}: {e.detail[:800]}"
            if not clean and e.extra.get("tb"):
                res.evidence["traceback"] = e.extra["tb"][-2000:]
            if isinstance(e, Hang):
                self._hang_stacks(ctx, res)
                self._recovered = True
                try:
                    self.surface.recover()
                except Exception:  # noqa: BLE001
                    pass
                self.loaded = {"image": None, "video": None}
        elif "crash" in box:
            res.status, res.error = FAIL, "check crashed (harness bug?)"
            res.evidence["traceback"] = box["crash"]
        elif res.failures:
            res.status = FAIL
        elif res.voids:   # nothing failed, but a comparison could not be made: a harness VOID, not a PASS
            res.status, res.error = VOID, "; ".join(res.voids)[:800]
        if not th.is_alive():
            try:
                res.evidence["vram_mib_after"] = self.surface.mem().get("vram_mib")
            except Exception:  # noqa: BLE001
                pass
        elif res.n_expects == 0 and res.infos:
            res.status = INFO  # observation only: Studio's intended behaviour here is not pinned down
        return res

    def run(self) -> list[Result]:
        for c in self.selected():
            after_recovery, self._recovered = self._recovered, False
            res = self._attempt(c)
            # A check that fails at once right after a hang recovery usually met the surface still
            # dying or restarting, not its own subject (live: two checks "crashed in 0.0 s" after
            # image.compile_resize hung, and turned a switchboard run into FAIL_HEAD). Retry it once
            # on a freshly recovered surface: that answer counts, the first is kept as evidence.
            if after_recovery and res.status == FAIL and res.wall_s < KNOCK_ON_S:
                first = {"status": res.status, "error": res.error, "wall_s": res.wall_s}
                self.log(f"  [{self.surface.name}] {c.name}: failed in {res.wall_s:.1f}s right after a "
                         "hang recovery, retrying once on a fresh surface")
                self._recovered = False
                try:
                    self.surface.recover()
                    self.loaded = {"image": None, "video": None}
                    res = self._attempt(c)
                except Exception as e:  # noqa: BLE001 - the first attempt stands
                    first["retry_error"] = f"{type(e).__name__}: {e}"[:300]
                res.evidence["first_attempt_after_hang_recovery"] = first
                # Still failing at once: the surface is not usable yet, so the next check gets the same.
                self._recovered = self._recovered or (res.status == FAIL and res.wall_s < KNOCK_ON_S)
            self.results.append(res)
            flag = {PASS: "ok", FAIL: "FAIL", SKIP: "skip", INFO: "info", VOID: "VOID"}[res.status]
            detail = res.error or ("; ".join(res.failures) if res.failures else "")
            self.log(f"  [{self.surface.name}] {c.name:<34} {flag:<5} {res.wall_s:7.1f}s  {detail[:160]}")
        return self.results
