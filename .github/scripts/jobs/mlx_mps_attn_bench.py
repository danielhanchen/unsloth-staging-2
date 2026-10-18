"""Torch MPS attention cost on Apple Silicon: one full SDPA call vs the same call split into query chunks.

Qwen-Image-2.1 shapes (32 heads x 128, text prefix 256 tokens) at 512 / 768 / 1024 px. Answers whether
Studio's bounded MPS attention (unslothai/unsloth#13188) costs time when it splits, and records what
the budget picks on this machine. Runs on the staging macOS legs (name starts with mlx_ so the
Apple Silicon leg picks it up); never imports MLX.

    python jobs/mlx_mps_attn_bench.py --out result.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time

HEADS, HEAD_DIM, TEXT = 32, 128, 256
SIDES = (512, 768, 1024)
CHUNKS = (512, 1024, 2048)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required = True)
    ap.add_argument("--repeats", type = int, default = 5)
    args = ap.parse_args()
    import torch
    import torch.nn.functional as F

    rec = {
        "torch": torch.__version__,
        "machine": platform.machine(),
        "mac_ver": platform.mac_ver()[0],
        "mps": bool(torch.backends.mps.is_available()),
        "rows": [],
    }
    print(json.dumps({k: v for k, v in rec.items() if k != "rows"}), flush = True)
    if not rec["mps"]:
        rec["note"] = "MPS not available on this runner"
        open(args.out, "w").write(json.dumps(rec, indent = 1))
        print(rec["note"], flush = True)
        return 0
    try:
        rec["recommended_max_memory"] = int(torch.mps.recommended_max_memory())
    except Exception as exc:  # noqa: BLE001
        rec["recommended_max_memory"] = repr(exc)
    budget = rec["recommended_max_memory"] // 8 if isinstance(rec["recommended_max_memory"], int) else 2**30
    rec["budget"] = budget

    def timed(fn):
        for _ in range(2):
            fn()
        torch.mps.synchronize()
        out = []
        for _ in range(args.repeats):
            t = time.perf_counter()
            fn()
            torch.mps.synchronize()
            out.append(time.perf_counter() - t)
        return statistics.median(out), max(out) - min(out)

    for dtype in (torch.bfloat16, torch.float16):
        for side in SIDES:
            n = (side // 16) ** 2
            kv = n + TEXT
            torch.manual_seed(0)
            q = torch.randn(1, HEADS, n, HEAD_DIM, device = "mps", dtype = dtype)
            k = torch.randn(1, HEADS, kv, HEAD_DIM, device = "mps", dtype = dtype)
            v = torch.randn(1, HEADS, kv, HEAD_DIM, device = "mps", dtype = dtype)

            def full():
                return F.scaled_dot_product_attention(q, k, v)

            def chunked(rows):
                return lambda: torch.cat(
                    [F.scaled_dot_product_attention(q[:, :, s : s + rows], k, v) for s in range(0, n, rows)], dim = 2
                )

            per_row = HEADS * kv * q.element_size()
            adaptive = max(512, budget // per_row)
            row = {
                "dtype": str(dtype).split(".")[-1],
                "side": side,
                "tokens": n,
                "score_gib": round(HEADS * n * kv * q.element_size() / 2**30, 3),
                "adaptive_rows": int(min(adaptive, n)),
            }
            try:
                ref = full()
            except Exception as exc:  # noqa: BLE001
                row["error"] = repr(exc)[:200]
                rec["rows"].append(row)
                print(json.dumps(row), flush = True)
                del q, k, v
                torch.mps.empty_cache()
                continue
            # fp32 CPU oracle, split by query rows (exact math, bounded host memory).
            qc, kc, vc = (t.float().cpu() for t in (q, k, v))
            oracle = torch.cat(
                [F.scaled_dot_product_attention(qc[:, :, s : s + 512], kc, vc) for s in range(0, n, 512)], dim = 2
            )
            row["full_vs_cpu_fp32"] = float((ref.float().cpu() - oracle).abs().max())
            row["chunk512_vs_cpu_fp32"] = float((chunked(512)().float().cpu() - oracle).abs().max())
            del qc, kc, vc, oracle
            try:
                row["full_s"], row["full_spread"] = timed(full)
            except Exception as exc:  # noqa: BLE001
                row["full_error"] = repr(exc)[:200]
            for rows in CHUNKS:
                if rows >= n:
                    continue
                diff = float((chunked(rows)() - ref).abs().max())
                t, spread = timed(chunked(rows))
                row[f"chunk{rows}_s"], row[f"chunk{rows}_spread"], row[f"chunk{rows}_maxdiff"] = t, spread, diff
            rec["rows"].append(row)
            print(json.dumps(row), flush = True)
            del q, k, v, ref
            torch.mps.empty_cache()
    # Where a single MPS SDPA call starts returning wrong values: elements vs bytes (fp32 separates the two).
    rec["threshold"] = []
    for dtype in (torch.bfloat16, torch.float32):
        for frac in (0.5, 0.5, 0.75):
            target = frac * 2**29
            n = int(((target / HEADS) ** 0.5) - TEXT / 2)
            kv = n + TEXT
            torch.manual_seed(1)
            q = torch.randn(1, HEADS, n, HEAD_DIM, device = "mps", dtype = dtype)
            k = torch.randn(1, HEADS, kv, HEAD_DIM, device = "mps", dtype = dtype)
            v = torch.randn(1, HEADS, kv, HEAD_DIM, device = "mps", dtype = dtype)
            row = {"dtype": str(dtype).split(".")[-1], "tokens": n, "elements_over_2_29": round(HEADS * n * kv / 2**29, 4),
                   "bytes_gib": round(HEADS * n * kv * q.element_size() / 2**30, 3)}
            try:
                full = F.scaled_dot_product_attention(q, k, v).float().cpu()
                qc, kc, vc = (t.float().cpu() for t in (q, k, v))
                oracle = torch.cat(
                    [F.scaled_dot_product_attention(qc[:, :, s : s + 512], kc, vc) for s in range(0, n, 512)], dim = 2
                )
                row["full_vs_cpu_fp32"] = float((full - oracle).abs().max())
                del full, qc, kc, vc, oracle
            except Exception as exc:  # noqa: BLE001
                row["error"] = repr(exc)[:200]
            rec["threshold"].append(row)
            print(json.dumps(row), flush = True)
            del q, k, v
            torch.mps.empty_cache()
    open(args.out, "w").write(json.dumps(rec, indent = 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
