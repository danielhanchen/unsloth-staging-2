"""Studio's MLX int8 prefill setting end to end on Apple Silicon (unsloth#12550).

Each arm runs in a FRESH interpreter from the Studio backend of the checkout under test:
  availability: POST /api/inference/int8-prefill-availability (route function, auth bypassed)
  off / on:     MLXInferenceBackend.load_model(int8_prefill=False|True), the status fields, then a
                greedy chat reply (temperature 0) and a 2-request batch reply.

Checks: availability answers with a known reason; every status echoes the request; the load-time
verdict agrees with the pre-load one; when int8 prefill is not active, "on" decodes byte-identically
to "off" (nothing changed for a host that cannot use it). On an M5 (NAX) host "on" must be active.

    python jobs/mlx_int8_prefill_probe.py            # mlx-community/Qwen3-0.6B-4bit
    python jobs/mlx_int8_prefill_probe.py --tiny     # SmolLM-135M-Instruct-4bit
Exit 3 off Apple Silicon.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

DEFAULT_MODEL = "mlx-community/Qwen3-0.6B-4bit"
TINY_MODEL = "mlx-community/SmolLM-135M-Instruct-4bit"
KNOWN_REASONS = {
    None,
    "",
    "not_downloaded",
    "unsupported_model",
    "unsupported_zoo",
    "nax_unavailable",
    "no_eligible_projections",
    "probe_failed",
    "distributed",
}
PROMPT = "List the first ten prime numbers, separated by commas. " * 20


def _backend_dir():
    for cand in (Path.cwd(), *Path.cwd().parents):
        if (cand / "studio" / "backend" / "routes" / "inference.py").is_file():
            return cand / "studio" / "backend"
    raise SystemExit("run from an unsloth checkout (studio/backend/routes/inference.py not found)")


def child(arm, model, out_path):
    out_path = Path(out_path).resolve()
    backend = _backend_dir()
    sys.path.insert(0, str(backend))
    os.chdir(backend)
    result = {"arm": arm}
    if arm == "availability":
        import asyncio
        import routes.inference as ri
        from models.inference import Int8PrefillAvailabilityRequest

        t = time.perf_counter()
        resp = asyncio.run(
            ri.int8_prefill_availability(
                Int8PrefillAvailabilityRequest(model_path = model), current_subject = "probe"
            )
        )
        result.update(
            available = resp.available,
            reason = resp.reason,
            ms = round((time.perf_counter() - t) * 1e3, 1),
        )
    else:
        from core.inference.mlx_inference import MLXInferenceBackend
        from utils.models import ModelConfig

        be = MLXInferenceBackend()
        config = ModelConfig.from_identifier(model_id = model)
        t = time.perf_counter()
        assert be.load_model(config, max_seq_length = 4096, int8_prefill = (arm == "on")) is True
        result["load_s"] = round(time.perf_counter() - t, 2)
        info = be.models[be.active_model_name]
        result.update(
            {
                k: info.get(k)
                for k in (
                    "mlx_int8_prefill",
                    "mlx_int8_prefill_requested",
                    "mlx_int8_prefill_reason",
                )
            }
        )
        msgs = [{"role": "user", "content": PROMPT}]
        t = time.perf_counter()
        result["text"] = "".join(
            be.generate_chat_response(
                msgs, temperature = 0.0, top_p = 1.0, top_k = 0, max_new_tokens = 32, seed = 3407
            )
        )
        result["chat_s"] = round(time.perf_counter() - t, 2)
        try:
            reqs = [
                {"messages": msgs, "temperature": 0.0, "max_new_tokens": 16},
                {
                    "messages": [{"role": "user", "content": "Say hello."}],
                    "temperature": 0.0,
                    "max_new_tokens": 16,
                },
            ]
            reason = be.batch_unavailable_reason(reqs)
            if reason is None:
                result["batch"] = [str(x) for x in be.generate_chat_batch(reqs)]
            else:
                result["batch_unavailable"] = str(reason)
        except Exception as exc:  # recorded: the batch request shape is a harness guess
            result["batch_error"] = f"{type(exc).__name__}: {exc}"
    out_path.write_text(json.dumps(result))
    print(json.dumps(result), flush = True)


def main():
    if "--child" in sys.argv:
        i = sys.argv.index("--child")
        child(*sys.argv[i + 1 : i + 4])
        return
    p = C.base_parser("mlx_int8_prefill_probe", DEFAULT_MODEL, TINY_MODEL, default_steps = 0)
    a = C.resolve_args(p)
    C.reject_cfg(a)
    with C.JobRecorder(a, backend_hint = "mlx") as rec:
        if not (sys.platform == "darwin" and platform.machine() == "arm64"):
            rec.skip("needs Apple Silicon")
        from huggingface_hub import snapshot_download

        snapshot_download(a.model)
        scratch = Path(tempfile.mkdtemp(prefix = "int8_probe_", dir = Path(a.out).resolve().parent))
        env = dict(os.environ, UNSLOTH_STUDIO_HOME = str(scratch / "home"))
        env.pop("UNSLOTH_MLX_INT8_PREFILL", None)
        res = {}
        for arm in ("availability", "off", "on"):
            out = scratch / f"{arm}.json"
            proc = subprocess.run(
                [sys.executable, __file__, "--child", arm, a.model, str(out)],
                env = env,
                capture_output = True,
                text = True,
                timeout = 1800,
            )
            if proc.returncode != 0 or not out.is_file():
                print(proc.stdout[-4000:], proc.stderr[-8000:], sep = "\n")
                rec.check(f"{arm}_ran", False, f"exit {proc.returncode}")
                continue
            res[arm] = json.loads(out.read_text())
            rec.check(f"{arm}_ran", True)
            rec.step(**{k: v for k, v in res[arm].items() if k not in ("text", "batch")})
        rec.summary(**res)
        av, off, on = res.get("availability"), res.get("off"), res.get("on")
        if av:
            rec.check("availability_reason_known", av["reason"] in KNOWN_REASONS, av)
        if off:
            rec.check(
                "off_status_echo",
                off["mlx_int8_prefill_requested"] is False and off["mlx_int8_prefill"] is False,
                off,
            )
        if on:
            rec.check("on_status_echo", on["mlx_int8_prefill_requested"] is True, on)
            if av:
                rec.check(
                    "load_verdict_matches_preload",
                    bool(on["mlx_int8_prefill"]) == bool(av["available"]),
                    f"pre-load {av} vs load {on['mlx_int8_prefill']}/{on['mlx_int8_prefill_reason']}",
                )
        if off and on:
            nax = bool(av and av["available"])
            rec.summary(nax_host = nax)
            if not on["mlx_int8_prefill"]:
                rec.check(
                    "inactive_on_is_identical_to_off",
                    on["text"] == off["text"] and on.get("batch") == off.get("batch"),
                    f"off={off['text']!r} on={on['text']!r}",
                )
            rec.check("generated_text", bool(off["text"].strip()) and bool(on["text"].strip()))


if __name__ == "__main__":
    main()
