"""Journey 6: text LoRA training with "Train on responses only" (train_on_completions).

The Train page's Advanced tab exposes the option (#trainOnCompletions); the run itself is started
with the same POST /api/train/start the Train button sends, on a fixed 16-row chat dataset
(system + user + assistant canary), 5 steps, seed 3407, on unsloth/gemma-3-270m-it. Asserted:
finite loss on every step, the masking actually configured (server log line from
utils/datasets/completion_masking.py), a real adapter on disk, and the documented refusal of
dataset_streaming + train_on_completions (HTTP 422). The adapter path is handed to
export_reload through ctx.state / a small marker file under the Studio home."""

from __future__ import annotations

import json
import time
from pathlib import Path

from .. import timeouts
from ..contract import Journey, Step, StepFailed
from . import _b_common as b
from . import _b_ui as ui

CANARY = "__REGRESS_CANARY__"
STEPS = 5
MASK_LINES = ("Configuring train on responses only", "Train on responses only configured")
TERMINAL = {"completed", "error", "stopped"}
MARKER = "studio_regress_last_adapter.json"


def dataset_rows(n = 16):
    return [
        {
            "messages": [
                {"role": "system", "content": "You are the regress canary bot."},
                {"role": "user", "content": f"Question {i}: what is the canary?"},
                {"role": "assistant", "content": CANARY},
            ]
        }
        for i in range(n)
    ]


def train_body(
    dataset_path,
    model_name = b.TRAIN_MODEL,
    **over,
):
    body = {
        "model_name": model_name,
        "training_type": "LoRA/QLoRA",
        "format_type": "chatml",
        "local_datasets": [str(dataset_path)],
        "project_name": "studio-regress-responses-only",
        "load_in_4bit": False,
        "max_seq_length": 512,
        "max_steps": STEPS,
        "num_epochs": 0,
        "batch_size": 2,
        "gradient_accumulation_steps": 1,
        "learning_rate": "2e-4",
        "lora_r": 8,
        "lora_alpha": 16,
        "save_steps": 10_000,
        "random_seed": 3407,
        "train_on_completions": True,
        "warmup_steps": 0,
    }
    body.update(over)
    return body


def losses(status):
    h = (status or {}).get("metric_history") or {}
    return [x for x in (h.get("loss") or [])]


def _log_paths(ctx):
    home = b.studio_home(ctx)
    paths = [Path(p) for p in [(ctx.models or {}).get("_studio_log")] if p]
    if home:
        paths += sorted((home / "logs").rglob("*.log"))
    return [p for p in paths if p.is_file()]


def log_marks(ctx):
    """Per-file sizes now, so a later log_since() reads only what was written after."""
    return {str(p): p.stat().st_size for p in _log_paths(ctx)}


def log_since(ctx, marks = None):
    out = []
    for p in _log_paths(ctx):
        try:
            with p.open("rb") as f:
                f.seek((marks or {}).get(str(p), 0))
                out.append(f.read().decode(errors = "replace"))
        except OSError:
            pass
    return "\n".join(out)


def log_text(ctx):
    return log_since(ctx, None)


async def _chart_drawn(ctx, timeout_ms = 5_000):
    """The run panel's loss chart is on the page (was a fixed 1.5 s). Bounded and silent: the chart
    is masked in the shot, and a run with no chart (error phase) is judged by the API facts."""
    try:
        await ctx.page.locator(".recharts-wrapper, canvas").first.wait_for(
            state = "visible", timeout = timeout_ms
        )
    except Exception:
        pass


async def train_form(ctx):
    await b.open_train_page(ctx)
    # Configure (page) -> Advanced (parameters: Simple | Advanced) -> Memory (sub-tab).
    for tab in ("Configure", "Advanced", "Memory"):
        loc = ctx.page.get_by_role("tab", name = tab, exact = True)
        if await loc.count():
            await loc.first.click()
            # The tab is shown once it reports selected (was a fixed 400 ms per tab).
            try:
                await ctx.page.wait_for_function(
                    """n => [...document.querySelectorAll('[role=tab]')].some(t => t.innerText.trim() === n
                         && (t.getAttribute('aria-selected') === 'true' || t.dataset.state === 'active'))""",
                    arg = tab,
                    timeout = timeouts.scaled_ms(5_000),
                )
            except Exception:
                pass
    box = ctx.page.locator("#trainOnCompletions")
    await ui.need(box, "Train on responses only option (#trainOnCompletions)")
    await box.first.scroll_into_view_if_needed()
    if (await box.first.get_attribute("aria-checked")) != "true":
        await box.first.click()
    state = await box.first.get_attribute("aria-checked")
    if state != "true":
        raise StepFailed(f"#trainOnCompletions did not turn on (aria-checked={state})")
    return {"train_on_completions_ui": True}


async def streaming_refused(ctx):
    body = train_body("unused", hf_dataset = "trl-lib/Capybara", dataset_streaming = True)
    body.pop("local_datasets")
    code, data = await b.req(
        ctx, "POST", "/api/train/start", body, ok = None, timeout = timeouts.inner(60, ctx.state)
    )
    detail = str((data or {}).get("detail") if isinstance(data, dict) else data)
    if code != 422 or "train_on_completions" not in detail:
        raise StepFailed(
            f"streaming + responses-only: expected 422 naming train_on_completions, got {code} {detail[:200]}"
        )
    return {"status": code, "detail": detail[:160]}


async def start(ctx):
    home = b.studio_home(ctx)
    ds = home / "assets" / "datasets" / "uploads" / "studio_regress_chat.jsonl"
    ds.parent.mkdir(parents = True, exist_ok = True)
    ds.write_text("\n".join(json.dumps(r) for r in dataset_rows()) + "\n")
    ctx.state["log_marks"] = log_marks(ctx)
    _, d = await b.req(
        ctx,
        "POST",
        "/api/train/start",
        train_body(ds),
        ok = (200,),
        timeout = timeouts.inner(300, ctx.state),
    )
    if d.get("status") not in ("queued", "pending"):
        raise StepFailed(f"/api/train/start: {d}")
    ctx.state["job_id"] = d.get("job_id")
    ctx.state["t_start"] = time.monotonic()
    return {"status": d.get("status"), "dataset_rows": 16}


# The shot is taken while training advances, so the step it catches is timing (live: base "Step 4 / 5",
# head "Step 3 / 5", VISUAL_DIFF px=1041). Wait until the run is past halfway, so both sides show the same
# callout ("Halfway done", pct >= 50 and < 100), and mask the step label, percent and bar (PROGRESS_MASKS).
PAST_HALF = STEPS // 2 + 1


def past_halfway(status):
    return (
        len([x for x in losses(status) if b.finite(x)]) >= PAST_HALF
        or (status or {}).get("phase") in TERMINAL
    )


async def progress_ui(ctx):
    st = await b.poll(lambda: b.get(ctx, "/api/train/status"), past_halfway, 600, state = ctx.state)
    await b.open_train_page(ctx)
    await _chart_drawn(ctx)
    return {"phase_seen": st.get("phase"), "_steps_at_shot": len(losses(st))}


async def completed(ctx):
    st = await b.poll(
        lambda: b.get(ctx, "/api/train/status"),
        lambda s: s.get("phase") in TERMINAL,
        900,
        state = ctx.state,
    )
    if st.get("phase") != "completed":
        raise StepFailed(
            f"training ended in {st.get('phase')}: {st.get('error') or st.get('message')}"
        )
    ls = losses(st)
    fin = [x for x in ls if b.finite(x)]
    if len(fin) < STEPS or len(fin) != len(ls):
        raise StepFailed(f"losses {ls}: expected {STEPS} finite values")
    log = log_since(ctx, ctx.state.get("log_marks"))
    masking = [m for m in MASK_LINES if m in log]
    if not masking:
        raise StepFailed("server log never configured train_on_responses_only")
    out = (st.get("details") or {}).get("output_dir") or st.get("output_dir")
    root = Path(out) if out else None
    w = next(
        (
            root / n
            for n in ("adapter_model.safetensors", "adapter_model.bin")
            if root and (root / n).is_file()
        ),
        None,
    )
    if not (root and (root / "adapter_config.json").is_file() and w and w.stat().st_size >= 4096):
        raise StepFailed(f"no real LoRA adapter under {out}")
    ctx.state["adapter_dir"] = str(root)
    (b.studio_home(ctx) / MARKER).write_text(json.dumps({"adapter_dir": str(root)}))
    await b.open_train_page(ctx)
    await _chart_drawn(ctx)
    return {
        "steps_with_loss": len(fin),
        "loss_first": round(fin[0], 4),
        "loss_last": round(fin[-1], 4),
        "masking_log": masking,
        "adapter_bytes": w.stat().st_size,
        "train_wall_s": round(time.monotonic() - ctx.state["t_start"], 1),
    }


# Loss curves, ETA and elapsed on the Train page: exact values differ by GPU / kernel; the page
# layout and the run's state are the subject.
TRAIN_MASKS = (
    b.VOLATILE_MASKS
    + b.PATH_TEXT_MASKS
    + (
        "canvas",
        ".recharts-wrapper",
        ".tabular-nums",
        "div.text-xs.text-muted-foreground:has-text('Elapsed')",
        "text=/Model saved to/",
        "nav a:has-text('studio-regress')",
    )
)
# The run panel's "Step N / M", percent and progress bar (progress-section.tsx): where training is at the
# moment of the shot, not what the PR changed.
PROGRESS_MASKS = (
    "div.flex.flex-col.gap-2:has([role=progressbar])",
    "[role=progressbar]",
    "text=/^Step \\d+ \\/ \\d+$/",
)

JOURNEY = Journey(
    name = "train_responses_only",
    tier = "gpu",
    needs = ("train_small",),
    routes = ("/studio",),
    steps = (
        Step("train_form", train_form, masks = b.PATH_TEXT_MASKS),
        Step("streaming_refused", streaming_refused, shot = False),
        Step("start", start, shot = False, timeout_s = 300),
        Step(
            "progress_ui",
            progress_ui,
            masks = TRAIN_MASKS + PROGRESS_MASKS,
            mask_reason = "live loss / ETA / step reached",
            timeout_s = 700,
        ),
        Step(
            "completed",
            completed,
            masks = TRAIN_MASKS,
            mask_reason = "loss chart values",
            timeout_s = 1000,
        ),
    ),
)
