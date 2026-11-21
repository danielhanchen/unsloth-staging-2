"""The edge checks, in run order (grouped by the model they need, so the fast tier loads each model as few times
as possible). Each check is small, independent of the others' verdicts, and asserts what Studio intends where the
code states it (route bounds, check_output_size, validate_video_request_shape, the GPU arbiter); where the intent
is not pinned down it records an INFO observation instead of guessing.

Conventions: ``ctx.outcome`` is strict (a 5xx / unexpected exception / hang is a FAIL, a clean refusal or a render
is returned for the check to judge); ``probe`` is for inputs the HTTP schema already rejects, reached only through
the in-process API, where any outcome is an observation.
"""

from __future__ import annotations

import threading
import time

from framework import REAL_UNLOAD_S, check, identical, img_stats, load_plan, mad, unload_slowness, video_stats
from surfaces import Refused

P = "a lighthouse on a cliff at sunset, oil painting"
LONG = ("an extremely detailed matte painting of a sprawling cyberpunk harbour city at dusk, " * 40).strip()
HUGE = "x" * 20000
MOTION = "a monarch butterfly opening and closing its wings on a purple flower, wind in the grass"
UNICODE = "\U0001F9A5 a sloth astronaut, ナマケモノ, café crème brûlée, über ☃ ✨"


def probe(ctx, fn, what: str) -> str:
    """Observation only: returns 'ok', 'refused <status>', 'load_error', 'fault <exc>' or 'hang'."""
    from surfaces import Hang, LoadError, SurfaceError

    try:
        fn()
        return "ok"
    except Refused as e:
        return f"refused {e.status}"
    except LoadError:
        return "load_error"
    except Hang:
        return "hang"
    except SurfaceError as e:
        return f"fault {e.extra.get('exc') or e.status}: {e.detail[:120]}"


def refused_or_ok_at(ctx, kind: str, value, want_shape, what: str) -> None:
    if kind == "ok":
        shape = list(value.image.shape)
        ctx.expect(shape == list(want_shape), f"{what}: rendered at the requested size (never silently resized)",
                   got = shape, want = list(want_shape))
    elif kind in ("refused", "load_error"):
        ctx.expect(True, f"{what}: refused cleanly", status = value.status, detail = value.detail[:200])


def in_background(fn) -> dict:
    box: dict = {}

    def run():
        try:
            box["value"] = fn()
            box["kind"] = "ok"
        except Exception as e:  # noqa: BLE001
            box["kind"], box["value"] = type(e).__name__, e

    th = threading.Thread(target = run, daemon = True)
    th.start()
    box["thread"] = th
    return box


def size(ctx, name: str = "img_a") -> tuple:
    g = ctx.cfg(name)["gen"].get(ctx.name) or ctx.cfg(name)["gen"].get("*") or {}
    return g.get("width"), g.get("height")


def need_video(ctx) -> dict:
    g = ctx.cfg("vid")["gen"].get(ctx.name)
    if not g:
        ctx.skip(f"no video settings for the {ctx.name} surface in this tier")
    return g


def vgen(ctx, **kw):
    kw.setdefault("prompt", MOTION)
    if not ctx.cfg("vid")["gen"].get(ctx.name):
        ctx.skip(f"no video settings for the {ctx.name} surface in this tier")
    return ctx.gen("vid", **kw)


def frozen(stats: dict) -> bool:
    """A frozen clip repeats frames: most consecutive pairs identical, or almost no change at all."""
    t = (stats.get("shape") or [0])[0]
    return stats.get("frozen_pairs", t) >= max(1, (t - 1) // 2) or stats.get("motion_mean", 0) < 0.1


def wait_active(ctx, kind: str, timeout: float = 60, min_step: int = 1) -> dict:
    deadline = time.time() + timeout
    last = {}
    while time.time() < deadline:
        last = ctx.surface.progress(kind) or {}
        if last.get("active") and int(last.get("step") or 0) >= min_step:
            return last
        time.sleep(0.05)
    return {"timeout": True, **last}


# ================================================================================================ SDXL (resident)
@check("image.sanity", needs = "img_a", timeout = 120)
def image_sanity(ctx):
    """A 512px render is a real picture at the requested size and status reports what engaged."""
    st = ctx.surface.status("image")
    fam = ctx.cfg("img_a")["family"]
    ctx.expect(st.get("loaded") is True and st.get("family") == fam, f"status: loaded, family {fam}",
               status = {k: st.get(k) for k in ("loaded", "family", "speed_mode", "memory_mode", "offload_policy",
                                                "engine", "attention_backend")})
    g = ctx.gen(seed = 11)
    ok, s = ctx.sane(g.image)
    ctx.expect(ok, "image is not black / flat / NaN-posterised", stats = s, file = ctx.save("seed11", g.image))
    w, h = size(ctx)
    ctx.expect(list(g.image.shape) == [h, w, 3], f"image is {w}x{h}x3", shape = list(g.image.shape))
    ctx.info("render_s", round(g.wall_s, 3))


@check("image.determinism", needs = "img_a", timeout = 120)
def image_determinism(ctx):
    """Same seed twice is bit-identical with speed_mode off; the next seed gives a different image."""
    a, b, c = ctx.gen(seed = 1234), ctx.gen(seed = 1234), ctx.gen(seed = 1235)
    ctx.expect(identical(a.image, b.image), "same seed -> identical pixels", mad = mad(a.image, b.image))
    ctx.expect(mad(a.image, c.image) > 2, "different seed -> different image", mad = mad(a.image, c.image))
    ctx.runner.ref_1234 = a.image
    ctx.runner.ref_1234_plan = ctx.plan()   # the load plan it rendered under (identity checks compare it)


@check("image.sizes_valid", needs = "img_a", timeout = 240)
def image_sizes_valid(ctx):
    """Sizes inside Studio's rules (multiple of 16, 256..2048 per side) render at exactly that size, incl. 1:4."""
    sizes = [(256, 256), (1024, 576), (576, 1024), (256, 1024), (1536, 1536)]
    if ctx.runner.tier == "full":
        sizes.append((2048, 2048))
    for w, h in sizes:
        kind, v = ctx.outcome(lambda: ctx.gen(width = w, height = h, seed = 5), f"{w}x{h}")
        if kind != "ok":
            ctx.expect(kind in ("fault", "hang"), f"{w}x{h}: a valid size renders",
                       outcome = kind, detail = getattr(v, "detail", "")[:300])
            continue
        ok, s = ctx.sane(v.image)
        ctx.expect(list(v.image.shape) == [h, w, 3] and ok, f"{w}x{h}: exact size and a sane image", stats = s)
        ctx.info(f"{w}x{h}_s", round(v.wall_s, 2))


@check("image.sizes_invalid", needs = "img_a", timeout = 180)
def image_sizes_invalid(ctx):
    """Off-grid / too small / too large sizes are refused with a 4xx (never a 500, a crash or a silent resize)."""
    cases = [(520, 520, "not a multiple of 16"), (250, 512, "below 256"), (2064, 512, "over 2048 per side"),
             (3000, 3000, "over the API bound 2752")]
    for w, h, why in cases:
        kind, v = ctx.outcome(lambda: ctx.gen(width = w, height = h, seed = 5, steps = 1), f"{w}x{h}")
        refused_or_ok_at(ctx, kind, v, [h, w, 3], f"{w}x{h} ({why})")
        ctx.info(f"{w}x{h}", kind if kind == "ok" else f"{kind} {getattr(v, 'status', '')}".strip())
        if ctx.http and kind == "refused":
            ctx.expect(v.status in (400, 422), f"{w}x{h}: HTTP 400/422", status = v.status)


@check("image.steps_guidance", needs = "img_a", timeout = 180)
def image_steps_guidance(ctx):
    """steps=1, guidance 0 / 7 (+ negative prompt) / 20 all render sane; out-of-range values are refused (HTTP)."""
    for label, kw in (("steps1", {"steps": 1}), ("cfg0", {"guidance": 0.0}),
                      ("cfg7_neg", {"guidance": 7.0, "steps": 4, "negative_prompt": "blurry, low quality, text"})):
        g = ctx.gen(seed = 3, **kw)
        ok, s = ctx.sane(g.image)
        ctx.expect(ok, f"{label}: sane image", stats = s)
    g = ctx.gen(seed = 3, guidance = 20.0, steps = 2)
    s = img_stats(g.image)
    ctx.expect(s["mean"] > 3 and s["colors"] > 100, "cfg20 (the maximum): not black / not NaN", stats = s,
               file = ctx.save("cfg20", g.image))
    bad = [("steps0", {"steps": 0}), ("steps101", {"steps": 101}), ("cfg_neg", {"guidance": -1.0}),
           ("cfg21", {"guidance": 21.0})]
    for label, kw in bad:
        if ctx.http:
            kind, v = ctx.outcome(lambda: ctx.gen(seed = 3, **kw), label)
            ctx.expect(kind == "refused" and v.status == 422, f"{label}: HTTP 422",
                       outcome = kind, status = getattr(v, "status", None))
        else:
            ctx.info(label, probe(ctx, lambda: ctx.gen(seed = 3, **kw), label))


@check("image.prompts", needs = "img_a", timeout = 240)
def image_prompts(ctx):
    """Empty / blank / over-token-limit / 20k-char / unicode+emoji prompts render or are refused cleanly."""
    if ctx.http:
        kind, v = ctx.outcome(lambda: ctx.gen(prompt = "", seed = 1), "empty prompt")
        ctx.expect(kind == "refused" and v.status == 422, "empty prompt: HTTP 422 (min_length 1)",
                   outcome = kind, status = getattr(v, "status", None))
    else:
        ctx.info("empty_prompt", probe(ctx, lambda: ctx.gen(prompt = "", seed = 1), "empty"))
    for label, prompt, must_render in (("blank", "   ", False), ("over_77_tokens", LONG, True),
                                       ("20k_chars", HUGE, False), ("unicode_emoji", UNICODE, True)):
        kind, v = ctx.outcome(lambda: ctx.gen(prompt = prompt, seed = 1), label)
        if kind == "ok":
            ok, s = ctx.sane(v.image)
            ctx.expect(ok, f"{label}: sane image", stats = s)
        elif kind in ("refused", "load_error"):
            ctx.expect(not must_render, f"{label}: renders (the text encoder truncates, never refuses)",
                       status = v.status, detail = v.detail[:200])
        ctx.info(label, kind)
    base = ctx.gen(seed = 21, guidance = 4.0, steps = 4)
    neg = ctx.gen(seed = 21, guidance = 4.0, steps = 4, negative_prompt = "lighthouse")
    ctx.expect(mad(base.image, neg.image) > 1, "negative prompt changes the image at guidance > 1",
               mad = mad(base.image, neg.image))


@check("image.seeds", needs = "img_a", timeout = 180)
def image_seeds(ctx):
    """Seeds 0, 2^32-1 and 2^53-1 render; seed None draws a fresh seed each time and reports it."""
    for seed in (0, 2**32 - 1, 2**53 - 1):
        kind, v = ctx.outcome(lambda: ctx.gen(seed = seed), f"seed {seed}")
        ctx.expect(kind == "ok" and ctx.sane(v.image)[0], f"seed {seed}: renders", outcome = kind,
                   detail = getattr(v, "detail", "")[:200] if kind != "ok" else None)
    a, b = ctx.gen(seed = None), ctx.gen(seed = None)
    sa, sb = a.meta.get("seed"), b.meta.get("seed")
    ctx.expect(isinstance(sa, int) and isinstance(sb, int), "seed None: the response reports the seed used",
               seeds = [sa, sb])
    ctx.expect(sa != sb and mad(a.image, b.image) > 2, "seed None twice: two different images", seeds = [sa, sb],
               mad = mad(a.image, b.image))
    if isinstance(sa, int):
        again = ctx.gen(seed = sa)
        ctx.expect(identical(again.image, a.image), "the reported seed reproduces the image",
                   mad = mad(again.image, a.image))
    for seed in (-1, 2**53, 2**64):
        if ctx.http:
            kind, v = ctx.outcome(lambda: ctx.gen(seed = seed), f"seed {seed}")
            ctx.expect(kind == "refused" and v.status == 422, f"seed {seed}: HTTP 422", outcome = kind,
                       status = getattr(v, "status", None))
        else:
            ctx.info(f"seed_{seed}", probe(ctx, lambda: ctx.gen(seed = seed), str(seed)))


@check("image.batch", needs = "img_a", timeout = 180)
def image_batch(ctx):
    """batch_size 2 = seeds s, s+1; the same batch call repeats bit-identically; each image matches its single
    render within Studio's documented batch numerics (diffusion_batched.py: ~2.5/255 mean abs)."""
    r = ctx.gen(seed = 100, batch_size = 2)
    ctx.expect(len(r.images) == 2, "batch_size 2 -> 2 images", n = len(r.images))
    seeds = r.meta.get("seeds")
    ctx.expect(seeds == [100, 101], "batch seeds are base, base+1", seeds = seeds)
    again = ctx.gen(seed = 100, batch_size = 2)
    ctx.expect(len(again.images) == 2 and all(identical(a, b) for a, b in zip(r.images, again.images)),
               "same batch call twice -> bit-identical", mad = [mad(a, b) for a, b in zip(r.images, again.images)])
    if len(r.images) == 2:
        s0, s1 = ctx.gen(seed = 100), ctx.gen(seed = 101)
        d = [mad(r.images[0], s0.image), mad(r.images[1], s1.image)]
        ctx.expect(max(d) <= 4.0, "batch image i ~= single render of seed i (mean abs <= 4/255)", mad = d)
    rep = ctx.gen(seed = 1, seeds = [77, 77])
    ctx.expect(len(rep.images) == 2 and ctx.sane(rep.images[1])[0], "seeds [77, 77] -> 2 sane images",
               n = len(rep.images))
    # diffusion.py generate(): "Each image gets its OWN Generator, so same-seed repeats are bit-identical."
    ctx.info("seeds_77_77_within_one_call_mad", mad(rep.images[0], rep.images[-1]))
    pr = ctx.gen(seed = 9, prompts = [P, "a bowl of ramen, top-down photo"])
    ctx.expect(len(pr.images) == 2 and mad(pr.images[0], pr.images[1]) > 2, "prompts [a, b] -> 2 different images",
               n = len(pr.images))
    if ctx.http:
        kind, v = ctx.outcome(lambda: ctx.gen(seed = 1, batch_size = 33), "batch 33")
        ctx.expect(kind == "refused" and v.status == 422, "batch_size 33: HTTP 422", outcome = kind)


@check("image.progress_cancel", needs = "img_a", timeout = 180)
def image_progress_cancel(ctx):
    """generate-progress goes active with the right total; cancel stops the render (409) and leaves the model
    loaded and the next render identical to before."""
    ref = ctx.gen(seed = 4242)
    ctx.gen(seed = 98, **{**ctx.long(), "steps": 2})
    m0 = ctx.surface.mem()
    job = in_background(lambda: ctx.gen(seed = 99, timeout = 120, **ctx.long()))
    prog = wait_active(ctx, "image", 60)
    ctx.expect(not prog.get("timeout") and prog.get("total_steps") == ctx.long()["steps"], "progress active with the requested total_steps",
               progress = prog)
    c = ctx.surface.cancel("image")
    ctx.expect(c.get("cancelled") is True, "cancel returns cancelled: true", response = c)
    job["thread"].join(120)
    kind, v = job.get("kind"), job.get("value")
    ctx.expect(kind == "Refused" and getattr(v, "status", None) == 409, "the cancelled generate answers 409",
               outcome = kind, detail = str(v)[:300])
    st = ctx.surface.status("image")
    ctx.expect(st.get("loaded") is True, "model still loaded after cancel", loaded = st.get("loaded"))
    post = ctx.gen(seed = 4242)
    ctx.expect(identical(post.image, ref.image), "render after cancel identical to before",
               mad = mad(post.image, ref.image))
    idle = ctx.surface.progress("image")
    ctx.expect(not idle.get("active"), "progress inactive when idle", progress = idle)
    m1 = ctx.surface.mem()
    ctx.expect((m1.get("vram_mib") or 0) - (m0.get("vram_mib") or 0) <= 512,
               "a cancelled render leaves no VRAM behind (<= +512 MiB vs a finished 1024px render)",
               before_mib = m0.get("vram_mib"), after_mib = m1.get("vram_mib"),
               torch_alloc_mib = [m0.get("torch_alloc_mib"), m1.get("torch_alloc_mib")])


@check("image.concurrent", needs = "img_a", timeout = 180)
def image_concurrent(ctx):
    """Two generates at once: each returns its own seed's image (or a clean 409), never a mix-up or a 500."""
    ref = {s: ctx.gen(seed = s, steps = 4, width = 768, height = 768).image for s in (501, 502)}
    jobs = {s: in_background(lambda s = s: ctx.gen(seed = s, steps = 4, width = 768, height = 768)) for s in ref}
    oks = 0
    for s, job in jobs.items():
        job["thread"].join(120)
        kind, v = job.get("kind"), job.get("value")
        if kind == "ok":
            oks += 1
            ctx.expect(identical(v.image, ref[s]), f"seed {s}: concurrent render == sequential render",
                       mad = mad(v.image, ref[s]))
        else:
            ctx.expect(kind == "Refused" and getattr(v, "status", None) == 409, f"seed {s}: ok or a clean 409",
                       outcome = kind, detail = str(v)[:300])
    ctx.info("both_served", oks == 2)


@check("image.gallery", needs = "img_a", timeout = 120, surfaces = ("http",))
def image_gallery(ctx):
    """A render is persisted: listed with its prompt / seed / size, file identical, delete removes it (404)."""
    from studio_client import StudioError

    g = ctx.gen(seed = 321, prompt = "gallery persistence probe")
    iid = g.ids[0]
    listing = ctx.surface.client.images_gallery(limit = 20)["images"]
    rec = next((r for r in listing if r["id"] == iid), None)
    ctx.expect(rec is not None, "new image is in the gallery listing", listed = [r["id"] for r in listing[:5]])
    if rec:
        w, h = size(ctx)
        ctx.expect(rec["seed"] == 321 and rec["width"] == w and rec["height"] == h
                   and rec["prompt"] == "gallery persistence probe", "gallery record carries prompt / seed / size",
                   record = {k: rec.get(k) for k in ("prompt", "seed", "width", "height", "steps", "model")})
    ctx.surface.client.images_delete(iid)
    try:
        ctx.surface.client.images_file(iid)
        ctx.expect(False, "deleted image file answers 404")
    except StudioError as e:
        ctx.expect(e.status == 404, "deleted image file answers 404", status = e.status)
    try:
        ctx.surface.client.images_file("does-not-exist-0000")
        ctx.expect(False, "unknown image id answers 404")
    except StudioError as e:
        ctx.expect(e.status == 404, "unknown image id answers 404", status = e.status)


@check("http.auth", timeout = 60, surfaces = ("http",))
def http_auth(ctx):
    """Every media route needs a token (401 without one or with a bad one); a wrong password is 401."""
    from studio_client import StudioClient, StudioError

    base = ctx.surface.client.base_url
    anon, bad = StudioClient(base), StudioClient(base, token = "not-a-token")
    calls = [("GET", "/api/inference/images/status", None), ("GET", "/api/inference/images/gallery", None),
             ("POST", "/api/inference/images/generate", {"prompt": "x", "width": 512, "height": 512, "steps": 1}),
             ("POST", "/api/inference/images/load", {"model_path": "x"}), ("POST", "/api/inference/images/unload", {}),
             ("GET", "/api/inference/video/status", None),
             ("POST", "/api/inference/video/generate", {"prompt": "x"})]
    for cl, label in ((anon, "no token"), (bad, "bad token")):
        for method, path, body in calls:
            try:
                cl.request(method, path, body = body, _retry = False)
                ctx.expect(False, f"{label} {method} {path}: 401", status = 200)
            except StudioError as e:
                ctx.expect(e.status == 401, f"{label} {method} {path}: 401", status = e.status)
    wrong = StudioClient(base, password = "definitely-wrong-password")
    try:
        wrong.login()
        ctx.expect(False, "wrong password: 401")
    except StudioError as e:
        ctx.expect(e.status == 401, "wrong password: 401", status = e.status)
    ctx.expect(anon.health() is not None, "/api/health answers without a token")


# ================================================================================================ SDXL (lifecycle)
@check("image.memory_modes", timeout = 300)
def image_memory_modes(ctx):
    """fast and low_vram render bit-identical images; low_vram offloads (policy != none) and uses less VRAM."""
    ctx.runner.ensure(ctx, "img_a")
    ref = ctx.gen("img_a", seed = 77)
    m_fast = ctx.surface.mem()
    st = ctx.load("img_a", memory_mode = "low_vram")
    low = ctx.gen("img_a", seed = 77)
    m_low = ctx.surface.mem()
    ctx.expect(st.get("offload_policy") not in (None, "none"), "low_vram engages an offload policy",
               offload_policy = st.get("offload_policy"), memory_mode = st.get("memory_mode"))
    if ctx.tiny:  # random weights amplify ulp-level differences; measured: model offload != resident here only
        ctx.info("fast_vs_low_vram_mad (random weights)", mad(ref.image, low.image))
    else:
        ctx.expect(identical(ref.image, low.image), "fast == low_vram, bit for bit", mad = mad(ref.image, low.image))
    if ctx.tiny:
        ctx.info("vram_mib", {"fast": m_fast.get("vram_mib"), "low_vram": m_low.get("vram_mib")})
    else:
        ctx.expect((m_low.get("vram_mib") or 0) < (m_fast.get("vram_mib") or 0), "low_vram holds less VRAM than fast",
                   vram_fast_mib = m_fast.get("vram_mib"), vram_low_mib = m_low.get("vram_mib"))
    ctx.info("rss_gib", {"fast": m_fast.get("rss_gib"), "low_vram": m_low.get("rss_gib")})
    ctx.info("render_s", {"fast": round(ref.wall_s, 2), "low_vram": round(low.wall_s, 2)})


@check("image.leak_cycles", timeout = 420)
def image_leak_cycles(ctx):
    """load -> render -> unload, repeated: VRAM and host RAM return to the same level every cycle."""
    for kind in ("image", "video"):
        ctx.unload(kind)
    cycles = 3 if ctx.runner.tier == "full" else 2
    mems = [ctx.surface.settle_mem()]
    for i in range(cycles):
        ctx.load("img_a")
        ctx.gen("img_a", seed = 5 + i)
        loaded = ctx.surface.mem()
        ctx.unload("image")
        mems.append({**ctx.surface.settle_mem(), "loaded_vram_mib": loaded.get("vram_mib")})
    ctx.evidence("mem_per_cycle", mems)
    trimmed = ctx.surface.trimmed_mem()
    if trimmed:
        # glibc malloc_trim(0) in the worker: what drops here was freed but never returned to the OS
        ctx.evidence("rss_anon_gib_after_malloc_trim", trimmed.get("rss_anon_gib"))
    v = [m.get("vram_mib") or 0 for m in mems]
    r = [m.get("rss_anon_gib") or m.get("rss_gib") or 0 for m in mems]
    ctx.expect(max(v[1:]) - v[1] <= 256, "VRAM after unload does not grow across cycles (<= 256 MiB)",
               vram_after_unload_mib = v)
    ctx.expect(v[1] - v[0] <= 1536, "unload returns VRAM to within 1.5 GiB of the pre-load level (CUDA context)",
               before_mib = v[0], after_mib = v[1])
    ctx.expect(r[1] - r[0] <= 1.5 and r[-1] - r[1] <= 0.75,
               "host anon RSS returns after unload (first cycle <= +1.5 GiB, later cycles <= +0.75 GiB)",
               rss_anon_gib = r)
    if ctx.inproc:
        ctx.expect(all((m.get("torch_alloc_mib") or 0) <= 64 for m in mems[1:]),
                   "torch allocated after unload <= 64 MiB", torch_alloc_mib = [m.get("torch_alloc_mib") for m in mems])
    ctx.runner.base_mem = mems[1]


@check("image.unload_during_generate", timeout = 240)
def image_unload_during_generate(ctx):
    """Unload while a render runs: the render ends cleanly (409 or done), the model is gone, VRAM is freed, and a
    generate afterwards is a clean 409."""
    ctx.runner.ensure(ctx, "img_a")
    job = in_background(lambda: ctx.gen("img_a", seed = 8, timeout = 150, **ctx.long("img_a")))
    prog = wait_active(ctx, "image", 60)
    ctx.evidence("progress_before_unload", prog)
    kind, st = ctx.outcome(lambda: ctx.unload("image"), "unload during generate")
    ctx.expect(kind == "ok" and not st.get("loaded"), "unload succeeds mid-render", outcome = kind)
    job["thread"].join(150)
    jk, jv = job.get("kind"), job.get("value")
    ctx.expect(jk in ("ok", "Refused") and (jk == "ok" or getattr(jv, "status", None) == 409),
               "the interrupted generate returns or answers 409", outcome = jk, detail = str(jv)[:300])
    base = getattr(ctx.runner, "base_mem", None)
    if base:
        timeline, t0 = [], time.time()
        while True:  # give an asynchronous release 20 s before calling it held
            m = ctx.surface.mem()
            timeline.append((round(time.time() - t0, 1), m.get("vram_mib"), m.get("torch_reserved_mib")))
            if (m.get("vram_mib") or 0) - (base.get("vram_mib") or 0) <= 512 or time.time() - t0 > 20:
                break
            time.sleep(2)
        ctx.expect((m.get("vram_mib") or 0) - (base.get("vram_mib") or 0) <= 512,
                   "VRAM back to the unloaded level within 20 s (<= +512 MiB)", base_mib = base.get("vram_mib"),
                   timeline_s_vram_mib_reserved_mib = timeline)
    kind, v = ctx.outcome(lambda: ctx.gen("img_a", seed = 1), "generate with nothing loaded")
    ctx.expect(kind == "refused" and v.status == 409, "generate with nothing loaded: 409", outcome = kind,
               status = getattr(v, "status", None), detail = getattr(v, "detail", "")[:200])


@check("image.generate_while_loading", timeout = 240)
def image_generate_while_loading(ctx):
    """A generate issued while a load is in flight gets a clean answer (409, or it waits and renders)."""
    ctx.unload("image")
    ctx.surface.load_async("image", ctx.cfg("img_a")["model"], **ctx.cfg("img_a")["opts"])
    ctx.forget("image")
    kind, v = ctx.outcome(lambda: ctx.gen("img_a", seed = 2), "generate during load")
    ctx.info("generate_during_load", kind if kind == "ok" else f"{kind} {getattr(v, 'status', '')}")
    ctx.expect(kind in ("ok", "refused"), "generate during load: rendered or refused cleanly", outcome = kind)
    if kind == "refused":
        ctx.expect(v.status == 409, "refusal is 409", status = v.status, detail = v.detail[:200])
    st = ctx.surface.wait_loaded("image", 600)
    ctx.runner.loaded["image"] = "img_a"
    ctx.expect(st.get("loaded") is True, "the load still completes", loaded = st.get("loaded"))
    g = ctx.gen("img_a", seed = 1234)
    ref = getattr(ctx.runner, "ref_1234", None)
    if ref is not None:
        ctx.expect_identical(g.image, ref, "render after the overlapped load == earlier render",
                             getattr(ctx.runner, "ref_1234_plan", None), ctx.plan())


@check("image.bad_inputs", timeout = 300)
def image_bad_inputs(ctx):
    """Unknown model path, bad family override, missing GGUF, bad kind: a clean error, never a 500 or a hang."""
    ctx.runner.ensure(ctx, "img_a")
    sd = ctx.cfg("img_a")
    cases = [("missing_path", "/nonexistent/diffusion/model_dir", {"family_override": ctx.cfg("img_a")["family"]}),
             ("bad_family", sd["model"], {"family_override": "not-a-real-family"}),
             ("missing_gguf", ctx.cfg("img_b")["model"], {"family_override": ctx.cfg("img_b")["family"],
                                                          "gguf_filename": "missing-Q4_K_M.gguf"}),
             ("gguf_kind_without_file", sd["model"], {"family_override": ctx.cfg("img_a")["family"], "model_kind": "gguf"})]
    for label, model, opts in cases:
        kind, v = ctx.outcome(lambda: ctx.surface.load("image", model, timeout = 120, **opts), label)
        ctx.expect(kind in ("refused", "load_error"), f"{label}: clean error", outcome = kind,
                   status = getattr(v, "status", None), detail = getattr(v, "detail", "")[:240])
        if ctx.http and kind == "refused":
            ctx.expect(v.status in (400, 404, 409, 422), f"{label}: HTTP 4xx", status = v.status)
        st = ctx.surface.status("image")
        ctx.info(f"{label}: resident model after", (st.get("family"), bool(st.get("loaded"))))
        ctx.forget("image")
    if ctx.http:
        for label, body in (("bogus_model_kind", {"model_kind": "bogus"}), ("bogus_memory_mode", {"memory_mode": "turbo"})):
            kind, v = ctx.outcome(lambda: ctx.surface.load("image", sd["model"], timeout = 60, **body), label)
            ctx.expect(kind == "refused" and v.status == 422, f"{label}: HTTP 422", outcome = kind,
                       status = getattr(v, "status", None))
    else:
        ctx.info("bogus_model_kind", probe(ctx, lambda: ctx.surface.load("image", sd["model"], timeout = 60,
                                                                           model_kind = "bogus"), "kind"))
    ctx.forget("image")


# ================================================================================================ switch to Z-Image
@check("switch.image_to_image", timeout = 420)
def switch_image_to_image(ctx):
    """Loading Z-Image over a resident SDXL replaces it: status names Z-Image and VRAM holds one model."""
    ctx.runner.ensure(ctx, "img_a")
    ctx.gen("img_a", seed = 1)
    m_sdxl = ctx.surface.mem()
    st = ctx.load("img_b")
    fam = ctx.cfg("img_b")["family"]
    ctx.expect(st.get("family") == fam and st.get("loaded"), f"status reports {fam} after the switch",
               family = st.get("family"), repo = st.get("repo_id"))
    g = ctx.gen("img_b", seed = 3)
    ok, s = ctx.sane(g.image)
    ctx.expect(ok, "Z-Image renders sane after the switch", stats = s, file = ctx.save("zimg_after_switch", g.image))
    m_z = ctx.surface.mem()
    ctx.evidence("vram_mib", {"img_a": m_sdxl.get("vram_mib"), "zimg_after_switch": m_z.get("vram_mib")})
    ctx.runner.switch_mem = m_z


@check("image_b.sanity", needs = "img_b", timeout = 240)
def image_b_sanity(ctx):
    """Second image model (Z-Image-Turbo; tiny FLUX in the tiny tier): 512 and 768 at size, same seed identical."""
    a, b = ctx.gen(seed = 42), ctx.gen(seed = 42)
    ctx.expect(identical(a.image, b.image), "same seed identical", mad = mad(a.image, b.image))
    ok, s = ctx.sane(a.image)
    w, h = size(ctx, "img_b")
    ctx.expect(ok and list(a.image.shape) == [h, w, 3], f"{w}x{h} sane", stats = s, file = ctx.save("first", a.image))
    c = ctx.gen(seed = 42, width = 768, height = 768)
    ok, s = ctx.sane(c.image)
    ctx.expect(ok and list(c.image.shape) == [768, 768, 3], "768px sane at size", stats = s,
               file = ctx.save("z768", c.image))
    ctx.info("render_s", {f"{w}": round(a.wall_s, 2), "768": round(c.wall_s, 2)})


@check("image_b.long_prompt", needs = "img_b", timeout = 180)
def image_b_long_prompt(ctx):
    """A prompt past the text encoder's max sequence length renders (or is refused cleanly), never a 500."""
    kind, v = ctx.outcome(lambda: ctx.gen(prompt = LONG * 2, seed = 42), "long prompt")
    if kind == "ok":
        ok, s = ctx.sane(v.image)
        ctx.expect(ok, "over-limit prompt renders sane", stats = s)
    ctx.info("outcome", kind)
    kind, v = ctx.outcome(lambda: ctx.gen(prompt = UNICODE, seed = 42), "unicode prompt")
    ctx.expect(kind == "ok" and ctx.sane(v.image)[0], "unicode + emoji prompt renders", outcome = kind)


# ================================================================================================ video
@check("switch.image_to_video", timeout = 600)
def switch_image_to_video(ctx):
    """Loading the video model with an image model resident: over HTTP the arbiter evicts the image model."""
    ctx.runner.ensure(ctx, "img_b")
    if ctx.inproc:
        ctx.unload("image")  # no arbiter in process: do what the route's arbiter does
    st = ctx.load("vid")
    fam = ctx.cfg("vid")["family"]
    ctx.expect(st.get("loaded") and st.get("family") == fam, f"video status: loaded, {fam}",
               family = st.get("family"), speed_optims = st.get("speed_optims"), offload = st.get("offload_policy"))
    img = ctx.surface.status("image")
    ctx.expect(not img.get("loaded"), "the image model is no longer resident", image_loaded = img.get("loaded"),
               image_family = img.get("family"))
    m = ctx.surface.mem()
    ctx.evidence("vram_mib_video_loaded", m.get("vram_mib"))
    ctx.info("video_defaults", st.get("defaults"))


@check("video.sanity", needs = "vid", timeout = 420)
def video_sanity(ctx):
    """17-frame clip at size, not black, not frozen; same seed identical, next seed different."""
    vs = need_video(ctx)
    a = vgen(ctx, seed = 7)
    want = [vs["num_frames"], vs["height"], vs["width"], 3]
    s = video_stats(a.frames)
    ctx.expect(list(a.frames.shape) == want, "clip shape = [frames, H, W, 3]", got = list(a.frames.shape), want = want)
    if ctx.tiny:
        ctx.info("clip_stats (random weights: not judged)", s)
    else:
        ctx.expect(s.get("mean", 0) > 8 and s.get("std", 0) > 8, "clip is not black / flat", stats = s,
                   file = ctx.save("seed7", a.frames))
        ctx.expect(not frozen(s), "clip is not frozen (repeated frames / no change)", stats = s)
    b = vgen(ctx, seed = 7)
    ctx.expect(identical(a.frames, b.frames), "same seed -> identical clip", mad = mad(a.frames, b.frames))
    c = vgen(ctx, seed = 8)
    if ctx.tiny:
        ctx.info("different_seed_mad (random weights)", mad(a.frames, c.frames))
    else:
        ctx.expect(mad(a.frames, c.frames) > 2, "different seed -> different clip", mad = mad(a.frames, c.frames))
    ctx.runner.video_ref = a.frames
    ctx.info("render_s", round(a.wall_s, 2))
    if ctx.http:
        ctx.expect(a.meta.get("num_frames") == vs["num_frames"] and a.meta.get("width") == vs["width"],
                   "gallery record: num_frames / width", record = {k: a.meta.get(k) for k in
                                                                   ("num_frames", "fps", "width", "height", "seed")})


@check("video.frames_fps", needs = "vid", timeout = 300)
def video_frames_fps(ctx):
    """fps is honoured; off-lattice frame counts (not 4k+1) are refused over HTTP; 1 frame and 5 frames work."""
    need_video(ctx)
    g = vgen(ctx, seed = 7, fps = 12, num_frames = 9)
    ctx.expect(g.fps == 12, "fps 12 reported", fps = g.fps)
    ctx.expect(g.frames.shape[0] == 9, "9 frames rendered", frames = int(g.frames.shape[0]))
    if ctx.http:
        dec = g.meta.get("decoded_fps")
        ctx.expect(dec is not None and abs(dec - 12) < 0.5, "MP4 stream fps 12", decoded_fps = dec)
        ctx.expect(abs((g.meta.get("duration_s") or 0) - 9 / 12) < 0.05, "duration_s = frames / fps",
                   duration_s = g.meta.get("duration_s"))
        for n in (18, 16, 0, 2000):
            kind, v = ctx.outcome(lambda: vgen(ctx, seed = 7, num_frames = n), f"{n} frames")
            ctx.expect(kind == "refused" and v.status == 422, f"num_frames {n}: 422", outcome = kind,
                       status = getattr(v, "status", None), detail = getattr(v, "detail", "")[:200])
    else:
        for n in (18, 16):
            try:
                r = vgen(ctx, seed = 7, num_frames = n)
                ctx.info(f"num_frames_{n}", f"rendered {r.frames.shape[0]} frames")
            except Exception as e:  # noqa: BLE001
                ctx.info(f"num_frames_{n}", f"{type(e).__name__}: {str(e)[:120]}")
    one = ctx.outcome(lambda: vgen(ctx, seed = 7, num_frames = 1), "1 frame")
    ctx.info("num_frames_1", one[0] if one[0] != "ok" else f"ok {list(one[1].frames.shape)}")


@check("video.sizes", needs = "vid", timeout = 300)
def video_sizes(ctx):
    """HTTP only offers the family presets (off-preset sizes are 422); in process the size snaps to the grid."""
    need_video(ctx)
    if ctx.http:
        for w, h in ((480, 320), (1280, 720), (1000, 700)):
            kind, v = ctx.outcome(lambda: vgen(ctx, seed = 7, width = w, height = h), f"{w}x{h}")
            ctx.expect(kind == "refused" and v.status == 422, f"{w}x{h} (not a preset): 422", outcome = kind,
                       status = getattr(v, "status", None))
        if ctx.runner.tier == "full":
            g = vgen(ctx, seed = 7, width = 704, height = 1280, num_frames = 9)
            ctx.expect(list(g.frames.shape[1:3]) == [1280, 704], "portrait preset 704x1280 renders at size",
                       shape = list(g.frames.shape))
    else:
        g = vgen(ctx, seed = 7, width = 320, height = 480, num_frames = 9)
        ctx.expect(list(g.frames.shape[1:3]) == [480, 320], "portrait 320x480 renders at size",
                   shape = list(g.frames.shape))
        for w, h in ((500, 300), (256, 256)):
            try:
                r = vgen(ctx, seed = 7, width = w, height = h, num_frames = 5)
                ctx.info(f"{w}x{h}", f"rendered {list(r.frames.shape[1:3])[::-1]}")
            except Exception as e:  # noqa: BLE001
                ctx.info(f"{w}x{h}", f"{type(e).__name__}: {str(e)[:120]}")


@check("video.cancel", needs = "vid", timeout = 300)
def video_cancel(ctx):
    """Cancel mid-denoise ends the clip cleanly; the model stays loaded and the next clip matches the reference."""
    need_video(ctx)
    job = in_background(lambda: vgen(ctx, seed = 70, steps = 40, timeout = 240))
    prog = wait_active(ctx, "video", 120)
    ctx.evidence("progress", prog)
    c = ctx.surface.cancel("video")
    ctx.expect(c.get("cancelled") is True, "cancel returns cancelled: true", response = c)
    job["thread"].join(240)
    jk, jv = job.get("kind"), job.get("value")
    ctx.expect(jk == "Refused" and getattr(jv, "status", None) == 409, "the cancelled clip ends as cancelled (409)",
               outcome = jk, detail = str(jv)[:300])
    st = ctx.surface.status("video")
    ctx.expect(st.get("loaded") is True, "video model still loaded", loaded = st.get("loaded"))
    ref = getattr(ctx.runner, "video_ref", None)
    if ref is not None:
        again = vgen(ctx, seed = 7)
        ctx.expect(identical(again.frames, ref), "clip after cancel == reference clip", mad = mad(again.frames, ref))


@check("video.memory_modes", tier = "full", timeout = 900)
def video_memory_modes(ctx):
    """Video fast vs low_vram: identical clip, low_vram offloads."""
    need_video(ctx)
    ctx.runner.ensure(ctx, "vid")
    ref = vgen(ctx, seed = 7)
    st = ctx.load("vid", memory_mode = "low_vram")
    low = vgen(ctx, seed = 7)
    ctx.expect(st.get("offload_policy") not in (None, "none"), "low_vram engages offload",
               offload_policy = st.get("offload_policy"))
    ctx.expect(identical(ref.frames, low.frames), "fast == low_vram clip", mad = mad(ref.frames, low.frames))


CLIP_TIMEOUT_S, CLIP_JOIN_SLACK_S = 300, 30


@check("video.unload_during_generate", tier = "full", timeout = 1200)   # two attempts on a slow host
def video_unload_during_generate(ctx):
    """Unload the video model mid-clip: the job ends cleanly and VRAM is released."""
    need_video(ctx)

    def attempt():
        ctx.runner.ensure(ctx, "vid")
        if hasattr(ctx.surface, "progress_timeout"):   # HTTP: a progress poll can wait behind the unload
            ctx.surface.progress_timeout = ctx.poll_timeout()
        job = in_background(lambda: vgen(ctx, seed = 70, steps = 40, timeout = CLIP_TIMEOUT_S))
        wait_active(ctx, "video", 120)
        t0 = time.perf_counter()
        kind, st = ctx.outcome(lambda: ctx.unload("video"), "unload during clip")
        unload_s = round(time.perf_counter() - t0, 2)
        # The clip's deadline is checked between polls, so a poll started near it can still run its full timeout.
        job["thread"].join(CLIP_TIMEOUT_S + (getattr(ctx.surface, "progress_timeout", None) or 0) + CLIP_JOIN_SLACK_S)
        if job["thread"].is_alive():
            job.update(kind = "Hang", value = "the clip did not answer within its deadline plus one progress poll")
        return {"unload": kind, "unload_loaded": getattr(st, "get", lambda k: None)("loaded"), "unload_s": unload_s,
                "poll_timeout_s": getattr(ctx.surface, "progress_timeout", None),
                "clip": job.get("kind"), "clip_value": job.get("value")}

    a = first = attempt()
    if a["clip"] == "Hang":
        # The clip's answer timed out: on a loaded host the unload alone can outlast a poll. Retry once with the
        # poll deadline rescaled from the unloads measured so far (this attempt's included).
        ctx.evidence("first_attempt", {k: (str(v)[:300] if k == "clip_value" else v) for k, v in a.items()})
        a = attempt()
    try:
        ctx.surface.progress_timeout = type(ctx.surface).progress_timeout
    except AttributeError:
        pass
    ctx.expect(a["unload"] == "ok" and not a["unload_loaded"], "unload succeeds mid-clip", outcome = a["unload"])
    jk, jv = a["clip"], a["clip_value"]
    slow, med = unload_slowness(ctx.runner.unload_s)
    # Only HTTP has a progress poll that can wait behind the unload; an in-process Hang is the call itself.
    # Each hang must be explained by its own overlapping unload outlasting that attempt's poll deadline.
    explained = all(x["clip"] == "Hang" and (x["unload_s"] or 0) >= (x["poll_timeout_s"] or float("inf"))
                    for x in (first, a))
    if jk == "Hang" and slow and explained and hasattr(ctx.surface, "progress_timeout"):
        ctx.void(f"the interrupted clip timed out twice while every unload this run is slow (median {med} s over "
                 f"{sum(x >= REAL_UNLOAD_S for x in ctx.runner.unload_s)} unloads, this one {a['unload_s']} s, poll "
                 f"deadline {a['poll_timeout_s']} s): host load, not the clip")
        ctx.evidence("unload_s_run", list(ctx.runner.unload_s))
    else:
        ctx.expect(jk == "ok" or (jk == "Refused" and getattr(jv, "status", None) == 409),
                   "the interrupted clip returns or answers 409", outcome = jk, detail = str(jv)[:300],
                   unload_s = a["unload_s"], unload_median_s = med)
    m = ctx.surface.settle_mem()
    base = getattr(ctx.runner, "base_mem", None)
    if base:
        ctx.expect((m.get("vram_mib") or 0) - (base.get("vram_mib") or 0) <= 768, "VRAM released",
                   now_mib = m.get("vram_mib"), base_mib = base.get("vram_mib"))


@check("switch.video_to_image", timeout = 420)
def switch_video_to_image(ctx):
    """Back to an image model: the video model is evicted (HTTP) and, once both are unloaded, VRAM and host RAM
    are back to the level of the first unload (no leak across image -> image -> video -> image)."""
    ctx.runner.ensure(ctx, "vid")
    if ctx.inproc:
        ctx.unload("video")
    ctx.load("img_a")
    vs = ctx.surface.status("video")
    ctx.expect(not vs.get("loaded"), "the video model is no longer resident", video_loaded = vs.get("loaded"))
    g = ctx.gen("img_a", seed = 1234)
    ref = getattr(ctx.runner, "ref_1234", None)
    if ref is not None:
        ctx.expect_identical(g.image, ref, "SDXL after the round trip == first SDXL render",
                             getattr(ctx.runner, "ref_1234_plan", None), ctx.plan())
    ctx.unload("image")
    ctx.unload("video")
    m = ctx.surface.settle_mem(2)
    base = getattr(ctx.runner, "base_mem", None)
    if base:
        ctx.expect((m.get("vram_mib") or 0) - (base.get("vram_mib") or 0) <= 768,
                   "VRAM with nothing loaded back to the first-unload level (<= +768 MiB)",
                   now_mib = m.get("vram_mib"), base_mib = base.get("vram_mib"))
        ctx.expect((m.get("rss_anon_gib") or 0) - (base.get("rss_anon_gib") or 0) <= 2.0,
                   "host anon RSS with nothing loaded within 2 GiB of the first unload",
                   now_gib = m.get("rss_anon_gib"), base_gib = base.get("rss_anon_gib"))
    ctx.evidence("final_mem", m)


# ================================================================================================ full tier: image
@check("image.compile_resize", tier = "full", timeout = 900)
def image_compile_resize(ctx):
    """With the speed tier on (compile / CUDA graphs): 512 -> 768 -> 512 renders at each size, the same 512 repeats
    bit-identically before any resize, and the 512 after the resize stays the same image (much closer to the first
    512 than the next seed is).

    Why not bit-identical after the resize: the speed tier is not a bit-identity mode (README rule 21, only
    speed_mode off is). The 768 render makes dynamo recompile the transformer / VAE for a new shape (dynamic after
    the second size), cudnn.benchmark and inductor autotuning pick kernels by timing, and the returning 512 can run
    on a different kernel with a different reduction order. Observed on tiny SDXL over HTTP on a shared B200: MAD
    0.267 (0-255) in one run and 0.0 in the next. A broken re-capture (stale graph, wrong buffer, the other size's
    output) is a different image, so it is caught by comparing against the next seed's distance, which is
    scale-free and holds for random-weight and real models alike."""
    for mode in (("default",) if ctx.tiny else ("default", "max")):
        st = ctx.load("img_a", speed_mode = mode)
        ctx.info(f"{mode}_speed_optims", st.get("speed_optims"))
        a = ctx.gen("img_a", seed = 31, steps = 4)
        a2 = ctx.gen("img_a", seed = 31, steps = 4)
        other = ctx.gen("img_a", seed = 32, steps = 4)
        b = ctx.gen("img_a", seed = 31, steps = 4, width = 768, height = 768)
        c = ctx.gen("img_a", seed = 31, steps = 4)
        ctx.expect(list(b.image.shape) == [768, 768, 3] and ctx.sane(b.image)[0], f"{mode}: 768 after 512 sane",
                   shape = list(b.image.shape))
        ctx.expect(identical(a.image, a2.image), f"{mode}: same 512 twice before any resize is bit-identical",
                   mad = mad(a.image, a2.image))
        d_resize, d_seed = mad(a.image, c.image), mad(a.image, other.image)
        ctx.info(f"{mode}_512_after_resize", {"mad_vs_first": d_resize, "identical": d_resize == 0.0,
                                              "next_seed_mad": d_seed})
        ctx.expect(list(c.image.shape) == list(a.image.shape), f"{mode}: the first size again renders at that size",
                   shape = list(c.image.shape))
        ctx.expect(d_resize <= 0.25 * d_seed, f"{mode}: 512 again is the first 512 (MAD under a quarter of the "
                   "next seed's)", mad = d_resize, next_seed_mad = d_seed)
        ctx.info(f"{mode}_s", [round(x.wall_s, 2) for x in (a, b, c)])
    ctx.forget("image")


@check("image.quant_switch", tier = "full", timeout = 900)
def image_quant_switch(ctx):
    """Switching transformer_quant between loads (none -> fp8 -> int8 -> none) takes effect each time and the
    unquantized render after the round trip equals the first one."""
    ref = ref_plan = None
    for q in ("none", "fp8", "int8", "none"):
        kind, st = ctx.outcome(lambda: ctx.load("img_b", transformer_quant = q), f"load quant {q}")
        if kind != "ok":
            ctx.info(f"quant_{q}", f"{kind} {getattr(st, 'detail', '')[:160]}")
            continue
        g = ctx.gen("img_b", seed = 42)
        ctx.info(f"quant_{q}", {"status_quant": st.get("transformer_quant"), "sane": ctx.sane(g.image)[0],
                                "mad_vs_none": mad(g.image, ref) if ref is not None else None})
        ctx.expect(ctx.sane(g.image)[0], f"{q}: sane render")
        if q == "none":
            if ref is None:
                ref, ref_plan = g.image, load_plan(st)
            else:
                ctx.expect_identical(g.image, ref, "none after fp8/int8 == first none", ref_plan, load_plan(st))
    ctx.forget("image")


@check("image.flux_schnell", tier = "full", timeout = 600)
def image_flux_schnell(ctx):
    """FLUX.1-schnell 512px 4 steps renders sane (a third image family through the same surfaces)."""
    if "flux" not in ctx.cfgs or ctx.tiny:
        ctx.skip("no FLUX model configured")
    st = ctx.load("flux")
    g = ctx.gen("flux", seed = 1)
    ok, s = ctx.sane(g.image)
    ctx.expect(ok, "FLUX sane", stats = s, family = st.get("family"), file = ctx.save("flux", g.image))


# ================================================================================================ tiny tier only
# hf-internal-testing random-weight pipelines: (dir, kind, family_override, expectation, generate kwargs).
# expectation: "render" = Studio claims the family and must render at that size; "refuse" = Studio does not
# support it and must say so with a clean 4xx / load error (never a 500 or a hang).
TINY_MATRIX = [
    ("tiny-stable-diffusion-xl-pipe", "image", "sdxl", "render", {}),
    ("tiny-flux-pipe", "image", "flux.1", "render", {}),
    ("tiny-qwenimage-pipe", "image", "qwen-image", "render", {}),
    ("tiny-qwenimage21-pipe", "image", "qwen-image-2.1", "render", {}),
    # its 1-block VAE does not downsample while Lumina2Pipeline hardcodes vae_scale_factor 8: diffusers itself
    # returns 32x32 for a 256 request, so the size is recorded, not judged
    ("tiny-lumina2-pipe", "image", "lumina-2", "render", {"@any_size": True}),
    # Kontext always renders ~1 MP; the hub tiny's 1x VAE makes that 524k tokens (T4: OOM), the derived 8x VAE 8k
    ("tiny-flux-kontext-pipe-vae8", "image", "flux.1-kontext", "render", {"init_image": "@init"}),
    ("tiny-qwenimage-edit-pipe", "image", "qwen-image-edit", "render", {"init_image": "@init"}),
    ("tiny-krea2-turbo-modular-pipe", "image", "krea-2", "info", {}),
    ("tiny-sd3-pipe", "image", None, "refuse", {}),
    ("tiny-sana-pipe", "image", None, "refuse", {}),
    ("tiny-wan-pipe", "video", "wan2.2-ti2v-5b", "render", {"num_frames": 9}),
    ("tiny-wan-pipe", "video", "wan2.2-t2v-a14b", "render", {"num_frames": 9}),
    ("tiny-random-hunyuanvideo", "video", "hunyuanvideo-1.5", "refuse", {"num_frames": 9}),
    ("tiny-cogvideox-pipe", "video", None, "refuse", {"num_frames": 9}),
]


# tiny pipe -> the local variant fetch_tiny.py derives for HTTP video, and the preset each family accepts over HTTP
TINY_HTTP_VIDEO = {"tiny-wan-pipe": "tiny-wan-pipe-rope128"}
TINY_HTTP_VIDEO_SIZE = {"wan2.2-ti2v-5b": (1280, 704), "wan2.2-t2v-a14b": (832, 480)}


def _init_image_b64() -> str:
    import base64
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (256, 256), (120, 80, 40)).save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@check("tiny.family_matrix", tier = "tiny", timeout = 900)
def tiny_family_matrix(ctx):
    """Every tiny pipeline: families Studio supports load, render at 256px (edit ones from an init image) and
    repeat bit-identically; the rest are refused cleanly. One row of evidence per pipeline."""
    from pathlib import Path

    from surfaces import SurfaceError

    root = Path(ctx.runner.tiny_root)
    init = _init_image_b64()
    rows = {}
    for name, kind, fam, expect, extra in TINY_MATRIX:
        path = root / name
        size = 256
        if kind == "video" and ctx.http:
            # HTTP only takes the family's presets; tiny-wan-pipe's rope (32) cannot cover them, the rope128
            # variant fetch_tiny.py derives can (same weights, rope rebuilt from the config at load)
            if name in TINY_HTTP_VIDEO:
                variant = root / TINY_HTTP_VIDEO[name]
                if not variant.exists():
                    rows[f"{name}:{fam}"] = (f"skipped over HTTP: needs {variant.name} (python diffusion_bench/"
                                             "fetch_tiny.py builds it; the stock tiny rope cannot reach a preset)")
                    continue
                path = variant
            if expect == "render":
                size = TINY_HTTP_VIDEO_SIZE.get(fam)
                if size is None:
                    rows[f"{name}:{fam}"] = "skipped over HTTP: no known resolution preset for this family"
                    continue
        if not path.exists():
            rows[f"{name}:{fam}"] = "missing"
            continue
        opts = {"speed_mode": "off", "memory_mode": "fast", **({"family_override": fam} if fam else {})}
        t0 = time.perf_counter()
        kind_l, st = ctx.outcome(lambda: ctx.surface.load(kind, str(path), timeout = 180, **opts), f"{name} load")
        row = {"load": kind_l if kind_l != "ok" else f"ok {st.get('family')}", "load_s": round(time.perf_counter() - t0, 1)}
        if kind_l != "ok":
            row["detail"] = getattr(st, "detail", "")[:160]
            if expect == "render":
                ctx.expect(False, f"{name} ({fam}): loads", outcome = kind_l, detail = row["detail"])
            elif expect == "refuse":
                ctx.expect(kind_l in ("refused", "load_error"), f"{name}: refused cleanly", outcome = kind_l)
            rows[f"{name}:{fam}"] = row
            continue
        if expect == "refuse":
            ctx.expect(False, f"{name}: refused (Studio lists no such family)", loaded_as = st.get("family"))
        w, h = (size, size) if isinstance(size, int) else size
        gp = {"prompt": "a red cube", "seed": 1, "steps": 2, "guidance": 1.0, "width": w, "height": h,
              **{k: (init if v == "@init" else v) for k, v in extra.items() if not k.startswith("@")}}
        t0 = time.perf_counter()
        k1, g1 = ctx.outcome(lambda: ctx.surface.generate(kind, timeout = 120, **gp), f"{name} render")
        row["render"], row["render_s"] = k1, round(time.perf_counter() - t0, 2)
        if k1 == "ok":
            arr = g1.frames if kind == "video" else g1.image
            row["shape"] = list(arr.shape)
            k2, g2 = ctx.outcome(lambda: ctx.surface.generate(kind, timeout = 120, **gp), f"{name} repeat")
            same = k2 == "ok" and identical(arr, g2.frames if kind == "video" else g2.image)
            row["repeat_identical"] = same
            ctx.expect(same, f"{name}: same seed repeats bit-identically")
            want = [h, w, 3] if kind == "image" else [gp.get("num_frames"), h, w, 3]
            if "init_image" not in extra and "@any_size" not in extra:
                ctx.expect(row["shape"] == want, f"{name}: rendered at the requested size", shape = row["shape"])
        else:
            row["detail"] = getattr(g1, "detail", "")[:160]
            if expect == "render":
                ctx.expect(k1 in ("refused",), f"{name}: renders or refuses cleanly", outcome = k1)
        try:
            ctx.surface.unload(kind)
        except SurfaceError:
            pass
        rows[f"{name}:{fam}"] = row
    ctx.evidence("matrix", rows)
    ctx.info("matrix", {k: (v if isinstance(v, str) else f"{v.get('load')} / {v.get('render', '-')} {v.get('shape', '')}")
                        for k, v in rows.items()})
    ctx.forget()


@check("tiny.rope_overflow_isolated", tier = "tiny", timeout = 300, surfaces = ("inproc",))
def tiny_rope_overflow(ctx):
    """tiny-zimage-pipe at Studio's 256px minimum overflows the tiny rope table: record how Studio surfaces it and
    whether the SAME process can still load a model afterwards. Runs in its own throwaway worker."""
    from pathlib import Path

    from surfaces import InprocSurface, SurfaceError

    main = ctx.surface
    iso = InprocSurface(main.python, main.studio_src, ctx.dir, main.gpu, ctx.dir / "worker.log",
                        studio_home = main.studio_home)
    ctx.dir.mkdir(parents = True, exist_ok = True)
    iso.start()
    try:
        root = Path(ctx.runner.tiny_root)
        iso.load("image", str(root / "tiny-zimage-pipe"), family_override = "z-image", speed_mode = "off",
                      memory_mode = "fast")
        out = {}
        for w in (256, 128):
            try:
                iso.generate("image", prompt = "a red cube", seed = 1, steps = 2, guidance = 0.0, width = w, height = w)
                out[w] = "ok"
            except SurfaceError as e:
                out[w] = f"{e.kind} {e.status or ''} {e.extra.get('exc') or ''}: {e.detail[:160]}"
        ctx.info("zimage_tiny_render", out)
        try:
            iso.unload("image")
            iso.load("image", str(root / "tiny-flux-pipe"), family_override = "flux.1", speed_mode = "off",
                     memory_mode = "fast")
            iso.generate("image", prompt = "a red cube", seed = 1, steps = 2, guidance = 0.0, width = 256, height = 256)
            ctx.info("same_process_afterwards", "a new load + render works")
        except SurfaceError as e:
            ctx.info("same_process_afterwards", f"{e.kind}: {e.detail[:200]}")
    finally:
        iso.stop()


@check("tiny.offline_reload", tier = "tiny", timeout = 400, surfaces = ("inproc",))
def tiny_offline_reload(ctx):
    """A prefetched model reloads by repo id with HF_HUB_OFFLINE=1 and local_files_only: tiny-flux-pipe (tokenizer,
    tokenizer_2, text_encoder_2 subfolders) laid out as a hub cache snapshot, loaded in a fresh worker whose only
    cache is that one. Guards offline loads that still reach for a processor / tokenizer file over the network."""
    import shutil
    from pathlib import Path

    from surfaces import InprocSurface, SurfaceError

    # Studio loads a non-GGUF pipeline by repo id only under unsloth/*; the id never has to exist on the Hub offline.
    repo = "unsloth/studio-regress-tiny-flux-pipe"
    src = Path(ctx.runner.tiny_root) / "tiny-flux-pipe"
    if not (src / "model_index.json").exists():
        ctx.skip(f"{src} not fetched (diffusion_bench/fetch_tiny.py)")
    hub = ctx.dir / "hf_home" / "hub"
    rev = "0" * 40
    snap = hub / f"models--{repo.replace('/', '--')}" / "snapshots" / rev
    if snap.exists():
        shutil.rmtree(snap)
    shutil.copytree(src, snap, ignore = shutil.ignore_patterns(".cache", ".fetch_complete"))
    (snap.parent.parent / "refs").mkdir(parents = True, exist_ok = True)
    (snap.parent.parent / "refs" / "main").write_text(rev)
    subfolders = sorted(p.name for p in snap.iterdir() if p.is_dir())
    ctx.info("snapshot_subfolders", subfolders)
    main = ctx.surface
    iso = InprocSurface(main.python, main.studio_src, ctx.dir, main.gpu, ctx.dir / "worker.log",
                        studio_home = str(ctx.dir / "studio_home"),
                        env = {"HF_HOME": str(hub.parent), "HF_HUB_CACHE": str(hub), "HF_HUB_OFFLINE": "1",
                               "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"})
    iso.start()
    try:
        k, st = ctx.outcome(lambda: iso.load("image", repo, family_override = "flux.1", speed_mode = "off",
                                             memory_mode = "fast", transformer_quant = "none"), "offline load by repo id")
        ctx.expect(k == "ok", "loads by repo id from the hub cache offline", outcome = k,
                   detail = getattr(st, "detail", "")[:400] if k != "ok" else None)
        if k == "ok":
            k2, g = ctx.outcome(lambda: iso.generate("image", prompt = "a red cube", seed = 1, steps = 2,
                                                     guidance = 0.0, width = 256, height = 256), "offline render")
            ctx.expect(k2 == "ok", "renders after the offline load", outcome = k2)
    finally:
        try:
            iso.stop()
        except SurfaceError:
            pass
