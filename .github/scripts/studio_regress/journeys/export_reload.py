"""Journey 7: export the LoRA from train_responses_only (journey 6 must run first on the same
Studio), then merged 16-bit and GGUF Q8_0, load the exported GGUF back into chat and get a reply.

Mirrors upstream's GPU harness (tests/kaggle/studio_gpu/run_studio_gpu.py:assert_gguf_export):
the export API has no job id, so completion is judged by last_op_seq moving past the value read
before the request, then last_op_status == "success"; the GGUF must start with the GGUF magic and
actually load and generate.

Known on hosts whose HF cache holds read-only blobs (huggingface_hub 1.32 writes xet-backed files
0444): merged and GGUF export copy the base shard with its mode and then overwrite it in place,
failing with EACCES. That is a real Studio bug, reported by this journey as a failure on both
sides, never masked."""

from __future__ import annotations

import json
from pathlib import Path

from .. import timeouts
from ..contract import Journey, Step, StepFailed
from . import _b_common as b
from . import _b_ui as ui
from .train_responses_only import CANARY, MARKER


def adapter_dir(ctx):
    a = ctx.state.get("adapter_dir")
    if not a:
        m = b.studio_home(ctx) / MARKER
        a = json.loads(m.read_text())["adapter_dir"] if m.is_file() else None
    if not a or not Path(a).is_dir():
        raise StepFailed("no adapter: run train_responses_only first on this Studio")
    return a


async def export_status(ctx):
    return await b.get(ctx, "/api/export/status")


async def run_export(
    ctx,
    path,
    body,
    deadline_s = 1200,
):
    before = await export_status(ctx)
    base_seq = before.get("last_op_seq") if isinstance(before.get("last_op_seq"), int) else -1
    try:
        await b.req(
            ctx, "POST", path, body, ok = (200,), timeout = timeouts.inner(deadline_s, ctx.state)
        )
    except Exception as e:  # blocking route; a transport timeout is not a failed export
        if isinstance(e, StepFailed):
            raise
    st = await b.poll(
        lambda: export_status(ctx),
        lambda s: isinstance(s.get("last_op_seq"), int)
        and s["last_op_seq"] > base_seq
        and not s.get("is_export_active"),
        deadline_s,
        state = ctx.state,
    )
    if st.get("last_op_status") != "success":
        raise StepFailed(
            f"{path}: last_op_status={st.get('last_op_status')} {st.get('last_op_error')}"
        )
    return st


def out_dir(ctx, st, name):
    """last_op_output_path may be relative to the server's cwd; the export lands in
    <home>/exports/<save_directory> either way."""
    p = Path(st.get("last_op_output_path") or "")
    return p if p.is_absolute() and p.is_dir() else b.studio_home(ctx) / "exports" / name


def newest(root, pattern):
    files = [p for p in Path(root).rglob(pattern) if "mmproj" not in p.name]
    return max(files, key = lambda p: p.stat().st_mtime) if files else None


async def export_page(ctx):
    await b.goto(
        ctx, "/export"
    )  # b.goto waits for the SPA mount + quiet requests (was + a fixed 1.5 s)
    return {"url": ctx.page.url.replace(ctx.base_url, "")}


async def load_checkpoint(ctx):
    a = adapter_dir(ctx)
    await b.unload_all(ctx)
    _, d = await b.req(
        ctx,
        "POST",
        "/api/export/load-checkpoint",
        {"checkpoint_path": a, "max_seq_length": 512, "load_in_4bit": False},
        ok = (200,),
        timeout = timeouts.inner(900, ctx.state),
    )
    if not d.get("success", True):
        raise StepFailed(f"load-checkpoint: {d}")
    return {"adapter_dir_name": Path(a).name, "message": (d.get("message") or "")[:120]}


async def export_lora(ctx):
    st = await run_export(
        ctx,
        "/api/export/export/lora",
        {"save_directory": "studio_regress_lora", "push_to_hub": False},
    )
    out = out_dir(ctx, st, "studio_regress_lora")
    w = newest(out, "adapter_model.safetensors")
    if not (w and newest(out, "adapter_config.json")):
        raise StepFailed(f"LoRA export has no adapter under {out}")
    return {"adapter_bytes": w.stat().st_size}


async def export_merged(ctx):
    st = await run_export(
        ctx,
        "/api/export/export/merged",
        {
            "save_directory": "studio_regress_merged",
            "format_type": "16-bit (FP16)",
            "push_to_hub": False,
        },
    )
    out = out_dir(ctx, st, "studio_regress_merged")
    w = newest(out, "*.safetensors")
    if not (w and newest(out, "config.json")):
        raise StepFailed(f"merged export has no config/safetensors under {out}")
    return {"weights": w.name if w else None, "bytes": w.stat().st_size if w else 0}


async def export_gguf(ctx):
    st = await run_export(
        ctx,
        "/api/export/export/gguf",
        {
            "save_directory": "studio_regress_gguf",
            "quantization_method": "q8_0",
            "push_to_hub": False,
        },
    )
    root = out_dir(ctx, st, "studio_regress_gguf")
    g = newest(root, "*.gguf")
    if g is None:
        raise StepFailed(f"GGUF export reported success but no .gguf under {root}")
    with g.open("rb") as f:
        if f.read(4) != b"GGUF":
            raise StepFailed(f"{g.name} lacks the GGUF magic")
    ctx.state["gguf"] = str(g)
    return {"gguf": g.name, "bytes": g.stat().st_size}


async def reload_and_chat(ctx):
    await b.req(ctx, "POST", "/api/export/cleanup", {}, ok = (200, 404, 409, 422))
    st = await b.load_gguf(ctx, ctx.state["gguf"], None, 1024)
    code, body = await b.chat(
        ctx,
        [
            {"role": "system", "content": "You are the regress canary bot."},
            {"role": "user", "content": "Question 0: what is the canary?"},
        ],
        max_tokens = 24,
    )
    txt = b.reply_text(body)
    if code != 200 or not txt.strip():
        raise StepFailed(f"exported GGUF: HTTP {code}, reply {txt!r}")
    await b.goto(ctx, "/chat")
    n = await ctx.page.locator(ui.ASSISTANT).count()
    await ui.send(ctx.page, "Question 1: what is the canary?")
    await ui.wait_reply(ctx.page, n)
    return {
        "loaded": st.get("active_model", "")[-60:],
        "reply_nonempty": True,
        "canary_learned": CANARY in txt,
    }


JOURNEY = Journey(
    name = "export_reload",
    tier = "gpu",
    needs = ("train_small",),
    routes = ("/export", "/chat"),
    steps = (
        Step("export_page", export_page),
        Step("load_checkpoint", load_checkpoint, shot = False, timeout_s = 900),
        Step("export_lora", export_lora, shot = False, timeout_s = 900),
        Step("export_merged", export_merged, shot = False, timeout_s = 1300),
        Step("export_gguf", export_gguf, shot = False, timeout_s = 1300),
        Step(
            "reload_and_chat",
            reload_and_chat,
            masks = b.VOLATILE_MASKS + b.GENERATED_TEXT_MASKS,
            mask_reason = "generated text",
            timeout_s = 600,
        ),
        Step("export_page_after", export_page),
    ),
)
