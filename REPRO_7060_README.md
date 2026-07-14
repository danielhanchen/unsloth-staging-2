# Cross-platform reproduction: Studio GGUF "update available" loop (#7060)

Runs the REAL Studio update check across Windows / macOS / Linux and several code
states to reproduce issue #7060 and validate PR #7113 (plus the related #7033 /
#7031 companion + quant-label family).

## What runs

- `.github/workflows/repro-7060.yml` — matrix of {ubuntu, macos, windows} x
  {Qwen3.5-2B-MTP-GGUF:UD-Q4_K_XL, gemma-4-E2B-it-qat-GGUF:UD-Q2_K_XL}. Each job
  runs `scripts/run_repro_7060_ci.py`, which prepares 5 code states and drives
  `scripts/repro_7060.py`, uploading a per-cell JSON report.
- `.github/workflows/repro-7060-playwright.yml` — best-effort live-Studio GUI
  reproduction (released 2026.7.2) with screenshots. continue-on-error.

## Code states

| state | what | expected on no-symlink cache |
| --- | --- | --- |
| `main` (2026.7.2) | has #7031, not #7113/#7033 | `update_available=True` (the #7060 bug) |
| `pr7113` | no-symlink size-identity fix | `update_available=False` (identical) |
| `pr7033` | companion + packed-Q2 label fix | does not fix #7060 by itself |
| `pr7113_7033` | both merged | fixes compose |
| `v2026_6_9` | pre-feature wheel | no update check -> downloads work |

## What it measures (per the code trace)

1. Badge / loop (#7060) — `get_gguf_variants_response.update_available`; persists
   across a simulated Update on `main`, clears on `pr7113`.
2. Load re-download — a second `hf_hub_download` on the no-symlink cache; **not**
   addressed by #7113 (blob-level cache miss).
3. Partial mis-flag — a complete moved GGUF must read as downloaded, not partial.
4. Cases: A identical (pure #7060), B size-changed (real diff, still caught),
   C equal-size content change (the fix's documented blind spot).

The layout is forced no-symlink on every OS (dereference the snapshot symlink,
empty `blobs/`) so the bug is deterministic regardless of runner symlink support;
the natural layout is also recorded (via `are_symlinks_supported()`).
