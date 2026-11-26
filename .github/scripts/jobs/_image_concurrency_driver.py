"""Concurrent image clients against Studio's real image routes, in one fresh process per arm.

Loads a model through Studio's own DiffusionBackend (begin_load), serves the real FastAPI routes in process (httpx
ASGI), and drives N clients, each bound to its own managed (non-owner) account, against /images/generate. Observes:
images/min, per-request latency p50 / p95, the render call sizes (a call of k images = a batch of k), the load's
``request_batching`` mode from /images/status, peak device memory, and for up to M batched members the pixel
difference against the same prompt and seed rendered alone (max / mean abs, bitwise, LPIPS when installed).

    python _image_concurrency_driver.py --tree <checkout>/studio/backend --repo <repo or local dir> \
        --family flux.2-klein --clients 4 --requests 4 --width 768 --height 768 --out arm.json

``--blocker``: hold the slot with one longer render while the clients arrive (correctness check with 2 clients).
Prints one ``ROW <json>`` line at the end and writes the same JSON to --out."""

from __future__ import annotations

import argparse
import asyncio
import base64
import functools
import hashlib
import io
import json
import os
import resource
import statistics
import sys
import time
import traceback

PROMPTS = [
    "a red fox sitting in fresh snow, photograph",
    "a blue sports car on a desert highway",
    "a bowl of ramen with a soft boiled egg, top view",
    "a lighthouse on a rocky cliff at sunset",
    "an astronaut riding a horse on the moon",
    "a watercolor painting of a cherry blossom tree",
    "a golden retriever puppy playing with a ball",
    "a futuristic city skyline at night with neon lights",
    "a cup of coffee on a wooden table, morning light",
    "a snowy mountain village at dusk",
    "a koi pond with lily pads, top view",
    "an old steam train crossing a stone bridge",
    "a hot air balloon over green hills",
    "a bookshelf full of old leather books",
    "a tropical beach with palm trees",
    "a robot painting a canvas in a studio",
]


def _pct(values, q):
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return round(ordered[index], 3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required = True, help = "studio/backend of the arm's checkout")
    ap.add_argument("--repo", required = True)
    ap.add_argument("--family", default = None)
    ap.add_argument("--speed-mode", default = "eager")
    ap.add_argument("--clients", type = int, default = 4)
    ap.add_argument(
        "--requests", type = int, default = 4, help = "requests per client, sent back to back"
    )
    ap.add_argument("--width", type = int, default = 768)
    ap.add_argument("--height", type = int, default = 768)
    ap.add_argument("--steps", type = int, default = None)
    ap.add_argument("--guidance", type = float, default = None)
    ap.add_argument("--solo-checks", type = int, default = 4)
    ap.add_argument("--blocker", action = "store_true")
    ap.add_argument("--label", default = "")
    ap.add_argument("--account-role", choices = ("user", "owner"), default = "user")
    ap.add_argument("--home", default = None)
    ap.add_argument("--out", required = True)
    a = ap.parse_args()
    rec: dict = {
        "label": a.label,
        "tree": a.tree,
        "repo": a.repo,
        "clients": a.clients,
        "requests": a.requests,
        "width": a.width,
        "height": a.height,
        "speed_mode": a.speed_mode,
    }
    try:
        _run(a, rec)
    except Exception as exc:  # noqa: BLE001 -- the row carries the failure
        rec["error"] = f"{type(exc).__name__}: {exc}"[:3000]
        rec["traceback"] = traceback.format_exc()[-6000:]
    with open(a.out, "w", encoding = "utf-8") as fh:
        json.dump(rec, fh, indent = 1, default = str)
    print("ROW " + json.dumps(rec, default = str), flush = True)
    return 0 if "error" not in rec else 1


def _run(a, rec: dict) -> None:
    sys.path.insert(0, os.path.abspath(a.tree))
    os.environ.setdefault(
        "UNSLOTH_STUDIO_HOME", os.path.abspath(a.home or f"studio_home_{a.label or 'arm'}")
    )
    os.environ.setdefault("UNSLOTH_DISABLE_SANDBOX_WARMUP", "1")
    os.environ.pop("UNSLOTH_STUDIO_IMAGE_BATCH_SEEDED", None)
    import httpx
    import numpy as np
    import torch
    from fastapi import FastAPI

    import core.inference.diffusion as diffusion_module
    import core.inference.diffusion_engine_router as engine_router
    import core.inference.image_gallery as gallery
    from auth import policy
    from auth.authentication import authenticated_via_api_key, get_current_subject
    from hub.services.models import account_access as access
    from routes.inference import router, studio_router
    from utils.account_context import AccountContext, bind_account, reset_account

    rec["imported_from"] = os.path.dirname(
        os.path.dirname(os.path.abspath(diffusion_module.__file__))
    )
    rec["torch"] = torch.__version__
    rec["hip"] = getattr(torch.version, "hip", None)
    if torch.cuda.is_available():
        rec["gpu"] = torch.cuda.get_device_name(0)
        rec["arch"] = getattr(torch.cuda.get_device_properties(0), "gcnArchName", None)
    rec["mps"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())

    # Managed accounts, as the account tests set them up: every client is a non-owner account.
    policy.installation_is_multi_user = lambda: True
    if hasattr(policy, "installation_has_managed_accounts"):
        policy.installation_has_managed_accounts = lambda: True
    access.repo_is_public = lambda *args, **kwargs: True
    # A local-path model is owner-only for managed accounts: --account-role owner keeps distinct account ids.
    role = {"role": "owner"} if a.account_role == "owner" else {}
    accounts = [
        AccountContext(f"{i + 1:032x}", f"user{i + 1}", **role)
        for i in range(max(a.clients, 2) + 2)
    ]
    rec["account_role"] = a.account_role

    engine = diffusion_module.DiffusionBackend()
    t0 = time.time()
    load_kw = {"speed_mode": a.speed_mode}
    if a.family:
        load_kw["family_override"] = a.family
    engine.begin_load(a.repo, **load_kw)
    while True:
        progress = engine.load_progress() or {}
        if progress.get("error"):
            raise RuntimeError(f"load error: {progress['error']}")
        if progress.get("phase") == "ready":
            break
        if time.time() - t0 > 3600:
            raise RuntimeError("load timeout")
        time.sleep(0.5)
    rec["load_s"] = round(time.time() - t0, 1)
    status = engine.status()
    rec["device"], rec["dtype"], rec["family"] = (
        status.get("device"),
        str(status.get("dtype")),
        status.get("family"),
    )
    rec["speed_optims"] = status.get("speed_optims")
    # Each managed account joins the resident, as an account whose own load matched it would.
    access.require_live_account = lambda *args, **kwargs: None
    for acc in accounts:
        token = bind_account(acc)
        try:
            access.join_resident("diffusion")
        except Exception as exc:  # noqa: BLE001 -- recorded; the generate calls then show the refusal
            rec.setdefault("join_errors", []).append(repr(exc)[:200])
        finally:
            reset_account(token)

    calls: list = []
    original = engine.generate

    @functools.wraps(original)
    def recorded(**kw):
        prompts = kw.get("prompts")
        entry = {
            "t0": time.perf_counter(),
            "images": len(prompts) if prompts else int(kw.get("batch_size") or 1),
            "prompts": list(prompts) if prompts else [kw.get("prompt")],
        }
        calls.append(entry)
        out = original(**kw)
        entry["t1"] = time.perf_counter()
        return out

    engine.generate = recorded
    diffusion_module.get_diffusion_backend = lambda: engine
    engine_router.get_active_diffusion_engine = lambda: engine
    engine_router._active_engine_name = "diffusers"

    store: dict = {}

    def _save(image, meta):
        image_id = f"img{len(store)}"
        store[image_id] = (image, dict(meta))
        return {**meta, "id": image_id, "url": f"/api/inference/images/gallery/{image_id}/file"}

    def _save_many(items):
        for image, meta in items:
            yield _save(image, meta)

    def _b64(image_id):
        buf = io.BytesIO()
        store[image_id][0].save(buf, format = "PNG")
        return base64.b64encode(buf.getvalue()).decode()

    gallery.save, gallery.save_many, gallery.image_b64 = _save, _save_many, _b64

    app = FastAPI()
    app.include_router(router, prefix = "/v1")
    app.include_router(studio_router, prefix = "/api/inference")
    by_id = {acc.account_id: acc for acc in accounts}

    async def subject():
        return "bench"

    app.dependency_overrides[get_current_subject] = subject
    app.dependency_overrides[authenticated_via_api_key] = lambda: False

    async def asgi(scope, receive, send):
        account = dict(scope.get("headers") or ()).get(b"x-test-account")
        if scope["type"] != "http" or account is None:
            return await app(scope, receive, send)
        token = bind_account(by_id[account.decode()])
        try:
            await app(scope, receive, send)
        finally:
            reset_account(token)

    def body(prompt, **extra):
        out = {"prompt": prompt, "width": a.width, "height": a.height, "live_preview": False}
        if a.steps is not None:
            out["steps"] = a.steps
        if a.guidance is not None:
            out["guidance"] = a.guidance
        out.update(extra)
        return out

    def pixels(image):
        return np.asarray(image.convert("RGB"), dtype = np.uint8)

    async def post(client, account, payload):
        r = await client.post(
            "/api/inference/images/generate",
            json = payload,
            headers = {"x-test-account": account.account_id},
            timeout = 3600,
        )
        if r.status_code != 200:
            raise RuntimeError(f"{account.username}: {r.status_code} {r.text[:600]}")
        return r.json()

    async def run():
        transport = httpx.ASGITransport(app = asgi)
        async with httpx.AsyncClient(transport = transport, base_url = "http://studio") as client:
            st = await client.get(
                "/api/inference/images/status", headers = {"x-test-account": accounts[0].account_id}
            )
            rec["status_code"] = st.status_code
            rec["request_batching"] = (
                (st.json() or {}).get("request_batching") if st.status_code == 200 else None
            )
            # Warm-up: one solo render, then three requests behind a running one (warms the batched shapes).
            await post(client, accounts[0], body("warm-up render", seed = 1))
            blocker = asyncio.ensure_future(
                post(client, accounts[-1], body("warm-up blocker", seed = 2))
            )
            while engine.generation_holder() is None:
                await asyncio.sleep(0.01)
            await asyncio.gather(
                *(post(client, accounts[i], body(f"warm-up member {i}")) for i in range(3))
            )
            await blocker
            rec["warmup_calls"] = [c["images"] for c in calls]
            calls.clear()

            latencies: list = []
            results: list = []
            if a.blocker:
                blocker = asyncio.ensure_future(
                    post(
                        client,
                        accounts[-1],
                        body("a blocker render", seed = 3, steps = (a.steps or 4) * 4),
                    )
                )
                while engine.generation_holder() is None:
                    await asyncio.sleep(0.01)

            async def one_client(index):
                for k in range(a.requests):
                    prompt = PROMPTS[(index * a.requests + k) % len(PROMPTS)] + f" #{index}.{k}"
                    t = time.perf_counter()
                    response = await post(client, accounts[index], body(prompt))
                    latencies.append(time.perf_counter() - t)
                    image = response["images"][0]
                    results.append((accounts[index], prompt, image["id"], image["seed"]))

            start = time.perf_counter()
            await asyncio.gather(*(one_client(i) for i in range(a.clients)))
            wall = time.perf_counter() - start
            if a.blocker:
                await blocker
            bench_calls = [c for c in calls if "blocker" not in (c["prompts"][0] or "")]
            rec["wall_s"] = round(wall, 3)
            rec["images"] = len(results)
            rec["images_per_min"] = round(len(results) / wall * 60, 3)
            rec["latency_p50_s"] = _pct(latencies, 0.5)
            rec["latency_p95_s"] = _pct(latencies, 0.95)
            rec["latency_mean_s"] = round(statistics.mean(latencies), 3) if latencies else None
            hist: dict = {}
            for c in bench_calls:
                hist[str(c["images"])] = hist.get(str(c["images"]), 0) + 1
            rec["call_sizes"] = hist
            rec["max_batch"] = max((c["images"] for c in bench_calls), default = 0)
            batched_prompts = {p: c["images"] for c in bench_calls for p in c["prompts"]}
            owners_in_batches = []
            for c in bench_calls:
                if c["images"] > 1:
                    owners_in_batches.append(
                        len(
                            {
                                acc.username
                                for acc, prompt, _, _ in results
                                if prompt in c["prompts"]
                            }
                        )
                    )
            rec["accounts_per_batch"] = owners_in_batches
            # Each member got its own prompt back.
            rec["own_prompt_ok"] = all(
                store[iid][1].get("prompt") == prompt for _, prompt, iid, _ in results
            )

            calls.clear()
            checks = []
            chosen = [r for r in results if batched_prompts.get(r[1], 1) > 1][: a.solo_checks]
            if not chosen:
                chosen = results[: a.solo_checks]
            metric = None
            try:
                import lpips  # noqa: PLC0415
                device = "cuda" if torch.cuda.is_available() else ("mps" if rec["mps"] else "cpu")
                metric = lpips.LPIPS(net = "alex", verbose = False).to(device).eval()
            except Exception as exc:  # noqa: BLE001 -- pixel stats still stand
                rec["lpips_error"] = repr(exc)[:300]
            for account, prompt, iid, seed in chosen:
                solo = await post(client, account, body(prompt, seed = seed))
                solo_img = store[solo["images"][0]["id"]][0]
                mine = store[iid][0]
                x, y = pixels(mine), pixels(solo_img)
                diff = np.abs(x.astype(np.int16) - y.astype(np.int16))
                row = {
                    "prompt": prompt,
                    "seed": seed,
                    "batched_in": batched_prompts.get(prompt, 1),
                    "bitwise": bool(diff.max() == 0),
                    "max_abs": int(diff.max()),
                    "mean_abs": round(float(diff.mean()), 5),
                }
                if metric is not None:
                    dev = next(metric.parameters()).device
                    tx = torch.from_numpy(x).permute(2, 0, 1)[None].float().to(dev) / 127.5 - 1
                    ty = torch.from_numpy(y).permute(2, 0, 1)[None].float().to(dev) / 127.5 - 1
                    with torch.no_grad():
                        row["lpips"] = round(float(metric(tx, ty).item()), 6)
                checks.append(row)
            rec["solo_checks"] = checks
            rec["solo_calls"] = [c["images"] for c in calls]

    asyncio.run(run())
    if torch.cuda.is_available():
        rec["peak_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
        rec["peak_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 2**30, 3)
    elif rec["mps"]:
        try:
            rec["mps_driver_allocated_gib"] = round(torch.mps.driver_allocated_memory() / 2**30, 3)
        except Exception:  # noqa: BLE001
            pass
    rec["max_rss_gib"] = round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / (2**30 if sys.platform == "darwin" else 2**20),
        3,
    )
    rec["sha_first"] = hashlib.sha256(b"".join(sorted(k.encode() for k in store))).hexdigest()[:12]


if __name__ == "__main__":
    raise SystemExit(main())
