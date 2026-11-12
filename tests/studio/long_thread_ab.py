# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Base vs head: does a change make a long chat cost more than its base build does?

playwright_long_thread_gate.py compares a build with itself (40 vs 200 turns, one appearance vs
another), so a cost every message pays alike in both lengths passes it (#17504: a streamed reply
cost 3x its base in main-thread time, every within-build ratio stayed green). This runs areas of the
area matrix (_area_matrix.py) on two production builds of vite.long-thread.config.ts, interleaved
ABBA on fresh pages in one browser, and fails when head's median main-thread time for an area is
over MAX_RATIO times base's and every head reading is above every base reading.

  python tests/studio/long_thread_ab.py --base-ref origin/main --head-ref HEAD
  python tests/studio/long_thread_ab.py --base DIST_A --head DIST_B      # prebuilt dists
  python tests/studio/long_thread_ab.py --base-ref X --head-ref X        # A/A noise

Exit 0 no regression, 1 regression, 2 void (a build or a page that did not run).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _area_matrix as areas  # noqa: E402
from _playwright_robust import chromium_launch_args  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
FRONTEND = REPO / "studio" / "frontend"
SCRATCH = REPO / ".playwright-cli" / "long-thread-ab"
METRICS = (
    "main_ms",
    "blocked_ms",
    "long_tasks",
    "cpu_ms",
    "style_ms",
    "layout_ms",
    "elements",
    "longest_task_ms",
)
# Gated on the renderer main thread: raster / compositor threads run on every frame under software
# compositing and swing with the runner, not the build.
# Metric -> floor (ms) under which differences are noise. blocked_ms is the Long Tasks share (what
# studiobench's stream cost sums): total main-thread time can fall while a change bunches it.
GATES = {"main_ms": 50.0, "blocked_ms": 100.0}


def _split_with_blocking(events: list) -> dict[str, dict]:
    """The area matrix's counters plus main-thread blocking: the part of each renderer main-thread
    task over 50 ms (the Long Tasks definition, what studiobench's stream cost sums) and their count.
    Total time can fall while a change bunches it into fewer, longer tasks the reader feels."""
    out = _split_trace(events)
    names, windows = {}, {}
    for e in events:
        if e.get("ph") == "M" and e.get("name") == "thread_name":
            names[(e["pid"], e["tid"])] = (e.get("args") or {}).get("name", "")
        elif e.get("name") == "TimeStamp":
            msg = ((e.get("args") or {}).get("data") or {}).get("message") or ""
            if msg.startswith("gate:"):
                _, edge, area = msg.split(":", 2)
                windows.setdefault(area, [None, None])[0 if edge == "begin" else 1] = e["ts"]
    spans = [(w[0], w[1], a) for a, w in windows.items() if None not in w and a in out]
    for _, _, a in spans:
        out[a]["blocked_ms"], out[a]["long_tasks"] = 0.0, 0
    for e in events:
        if (
            e.get("ph") != "X"
            or e.get("cat") != "toplevel"
            or e.get("name") not in ("ThreadControllerImpl::RunTask", "RunTask")
            or names.get((e.get("pid"), e.get("tid"))) != "CrRendererMain"
        ):
            continue
        dur = e.get("dur", 0) / 1000
        if dur <= 50:
            continue
        for b, en, a in spans:
            if b <= e["ts"] <= en:
                out[a]["blocked_ms"] += dur - 50
                out[a]["long_tasks"] += 1
                break
    return out


_split_trace = areas.split_trace
areas.split_trace = _split_with_blocking


def sh(cmd: list[str], cwd: Path) -> str:
    return subprocess.run(
        cmd, cwd = cwd, check = True, capture_output = True, text = True, shell = os.name == "nt"
    ).stdout.strip()


def build_ref(ref: str) -> tuple[Path, str]:
    """Production build of `ref` in a scratch worktree sharing this checkout's node_modules."""
    sha = sh(["git", "rev-parse", f"{ref}^{{commit}}"], REPO)
    out = SCRATCH / f"dist-{sha[:12]}"
    if (out / "index.html").exists():
        return out, sha
    wt = SCRATCH / f"wt-{sha[:12]}"
    if not wt.exists():
        sh(["git", "worktree", "add", "--detach", "--force", str(wt), sha], REPO)
    modules = wt / "studio" / "frontend" / "node_modules"
    if not modules.exists():
        modules.symlink_to(FRONTEND / "node_modules", target_is_directory = True)
    t = time.time()
    subprocess.run(
        ["npx", "vite", "build", "--config", "vite.long-thread.config.ts", "--outDir", str(out)],
        cwd = wt / "studio" / "frontend",
        check = True,
        shell = os.name == "nt",
        stdout = subprocess.DEVNULL,
    )
    print(f"built {ref} ({sha[:12]}) in {time.time() - t:.0f} s -> {out}", flush = True)
    return out, sha


def lengthen_stream(chars: int) -> None:
    """Stream about `chars` characters of prose before the reply's end, so a cost paid per streamed
    chunk (not per message) shows: the gate's own reply is a few hundred characters."""
    short = areas.stream_frames

    def frames() -> list[dict]:
        out = short()
        body = [f for f in out if (f.get("choices") or [{}])[0].get("delta", {}).get("content")]
        end = out.index(body[-1])
        line = "Streaming prose for the long reply, sixty characters a frame. "
        extra = [
            {**body[0], "choices": [{**body[0]["choices"][0], "delta": {"content": line}}]}
            for _ in range(max(0, chars // len(line)))
        ]
        return out[:end] + extra + out[end:]

    areas.stream_frames = frames
    # At STREAM_MS a frame, plus the page's own time on a slow runner.
    areas.STREAM_TIMEOUT_MS = max(
        areas.STREAM_TIMEOUT_MS, 30_000 + 4 * areas.STREAM_MS * len(frames())
    )


def open_page(browser, base: str, turns: int, mode: dict):
    """One retry: the first page in a fresh browser on a loaded runner can miss the load timeout."""
    try:
        return areas.Page(browser, base, turns, mode, [])
    except Exception:  # noqa: BLE001 - the second failure is the one reported
        return areas.Page(browser, base, turns, mode, [])


def measure(dists: dict[str, Path], turns: int, reps: int, wanted: list[str]) -> dict:
    from playwright.sync_api import sync_playwright

    mode = next(m for m in areas.app_modes() if m["name"] == "default")
    run = areas.with_deps(wanted)
    servers = {side: areas.serve_app(d) for side, d in dists.items()}
    readings: dict[str, dict[str, list[dict]]] = {s: {a: [] for a in wanted} for s in dists}
    errors: list[str] = []
    # ABBA: a drift in runner speed over the run lands on both sides alike.
    order = [("base", "head") if k % 2 == 0 else ("head", "base") for k in range(reps)]
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(args = chromium_launch_args())
            try:
                for k, pair in enumerate(order):
                    for side in pair:
                        page = None
                        try:
                            page = open_page(browser, servers[side][1], turns, mode)
                            cells = page.sweep(run)
                        except Exception as exc:  # noqa: BLE001 - reported, the verdict goes VOID
                            errors.append(f"{side} rep {k}: {type(exc).__name__}: {str(exc)[:200]}")
                            continue
                        finally:
                            if page is not None:
                                page.close()
                        for a in wanted:
                            cell = cells.get(a, {})
                            if cell.get("status") != "OK":
                                errors.append(f"{side} rep {k} {a}: {cell.get('status')}")
                                continue
                            readings[side][a].append({m: cell.get(m, 0.0) for m in METRICS})
                        line = "  ".join(
                            f"{a} {cells.get(a, {}).get('main_ms', float('nan')):.0f}"
                            f" (blocked {cells.get(a, {}).get('blocked_ms', float('nan')):.0f}"
                            f" in {cells.get(a, {}).get('long_tasks', 0)})"
                            for a in wanted
                        )
                        print(f"  rep {k} {side:4s} main_ms: {line}", flush = True)
            finally:
                browser.close()
    finally:
        for server, _ in servers.values():
            server.shutdown()
    return {"readings": readings, "errors": errors}


def judge(
    readings: dict, wanted: list[str], max_ratio: float, gated: list[str]
) -> tuple[list[dict], bool]:
    """Per area and gated metric: SLOWER when head's median is over max_ratio times base's (each
    floored) and every head reading is above every base reading."""
    rows, regressed = [], False
    for a in wanted:
        n_b, n_h = len(readings["base"][a]), len(readings["head"][a])
        if n_b < 2 or n_h < 2:
            rows.append({"area": a, "verdict": "VOID", "readings": [n_b, n_h]})
            continue
        row = {"area": a, "metrics": {}}
        slower = []
        for m in METRICS:
            b = [r[m] for r in readings["base"][a]]
            h = [r[m] for r in readings["head"][a]]
            mb, mh = statistics.median(b), statistics.median(h)
            info = {
                "base_median": round(mb, 1),
                "head_median": round(mh, 1),
                "base": [round(x, 1) for x in b],
                "head": [round(x, 1) for x in h],
            }
            if m in GATES:
                floor = GATES[m]
                info["ratio"] = round(max(mh, floor) / max(mb, floor), 2)
                info["separated"] = min(h) > max(b)
                if info["ratio"] > max_ratio and info["separated"]:
                    slower.append(m)
            row["metrics"][m] = info
        verdict = f"SLOWER ({', '.join(slower)})" if slower else "ok"
        if slower and a in gated:
            regressed = True
        row["verdict"] = verdict if a in gated else f"{verdict}, advisory"
        rows.append(row)
    return rows, regressed


def main() -> int:
    ap = argparse.ArgumentParser(description = __doc__.split("\n\n")[0])
    ap.add_argument("--base", type = Path, help = "prebuilt dist of the base")
    ap.add_argument("--head", type = Path, help = "prebuilt dist of the head")
    ap.add_argument("--base-ref", help = "git ref to build as the base")
    ap.add_argument("--head-ref", help = "git ref to build as the head")
    ap.add_argument("--turns", type = int, default = int(os.environ.get("LONG_THREAD_AB_TURNS", "200")))
    ap.add_argument("--reps", type = int, default = int(os.environ.get("LONG_THREAD_AB_REPS", "4")))
    ap.add_argument(
        "--areas", default = os.environ.get("LONG_THREAD_AB_AREAS", "stream,type,thread_scroll")
    )
    ap.add_argument(
        "--gated",
        default = os.environ.get("LONG_THREAD_AB_GATED", "stream,type"),
        help = "areas whose SLOWER verdict fails the run; the rest are reported",
    )
    ap.add_argument(
        "--max-ratio", type = float, default = float(os.environ.get("LONG_THREAD_AB_MAX_RATIO", "1.5"))
    )
    ap.add_argument(
        "--stream-chars",
        type = int,
        default = int(os.environ.get("LONG_THREAD_AB_STREAM_CHARS", "0")),
        help = "lengthen the streamed reply by about this many characters (0: the gate's own reply)",
    )
    ap.add_argument("--json", type = Path)
    args = ap.parse_args()

    wanted = [a.strip() for a in args.areas.split(",") if a.strip()]
    unknown = [a for a in wanted if a not in areas.AREAS]
    if unknown:
        ap.error(f"unknown areas {unknown}; known: {', '.join(areas.AREAS)}")
    gated = [a.strip() for a in args.gated.split(",") if a.strip()]
    dists, shas = {}, {}
    try:
        for side, dist, ref in (
            ("base", args.base, args.base_ref),
            ("head", args.head, args.head_ref),
        ):
            if dist:
                dists[side], shas[side] = dist, None
            elif ref:
                dists[side], shas[side] = build_ref(ref)
            else:
                ap.error(f"give --{side} or --{side}-ref")
    except subprocess.CalledProcessError as exc:
        print(f"VOID: build failed: {exc}", flush = True)
        return 2
    for side, d in dists.items():
        if not (d / "index.html").exists():
            print(f"VOID: no app build at {d / 'index.html'} ({side})", flush = True)
            return 2

    if args.stream_chars:
        lengthen_stream(args.stream_chars)
    started = time.time()
    print(
        f"long-thread A/B: {args.turns} turns, stream +{args.stream_chars} chars, {args.reps} reps per side, areas {wanted}",
        flush = True,
    )
    got = measure(dists, args.turns, args.reps, wanted)
    rows, regressed = judge(got["readings"], wanted, args.max_ratio, gated)
    print(
        "\n| area | metric | base median [readings] | head median [readings] | head/base | separated |"
    )
    print("|---|---|---|---|---|---|")
    for r in rows:
        if r["verdict"] == "VOID":
            print(
                f"| {r['area']} | - | {r['readings'][0]} readings | {r['readings'][1]} readings | - | VOID |"
            )
            continue
        for m, i in r["metrics"].items():
            if m not in GATES and m not in ("long_tasks", "longest_task_ms", "elements"):
                continue
            print(
                f"| {r['area']} | {m} | {i['base_median']:.0f} {[round(x) for x in i['base']]} | "
                f"{i['head_median']:.0f} {[round(x) for x in i['head']]} | {i.get('ratio', '')} | {i.get('separated', '')} |"
            )
        print(f"| {r['area']} | verdict | | | | {r['verdict']} |")
    for e in got["errors"]:
        print(f"  error: {e}")
    void = any(r["verdict"] == "VOID" and r["area"] in gated for r in rows)
    verdict = "VOID" if void else "REGRESSION" if regressed else "NO_REGRESSION"
    print(
        f"\nVERDICT {verdict}: gated {gated}, limit {args.max_ratio}x on median {'/'.join(GATES)} with no overlap "
        f"(base {shas.get('base') or dists['base']}, head {shas.get('head') or dists['head']}, "
        f"{time.time() - started:.0f} s)",
        flush = True,
    )
    if args.json:
        args.json.parent.mkdir(parents = True, exist_ok = True)
        args.json.write_text(
            json.dumps(
                {
                    "verdict": verdict,
                    "shas": shas,
                    "dists": {k: str(v) for k, v in dists.items()},
                    "turns": args.turns,
                    "reps": args.reps,
                    "stream_chars": args.stream_chars,
                    "rows": rows,
                    "errors": got["errors"],
                    "readings": got["readings"],
                },
                indent = 1,
            )
        )
    return 2 if void else 1 if regressed else 0


if __name__ == "__main__":
    raise SystemExit(main())
