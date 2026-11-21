# diffusion_bench

A reusable benchmark and edge-case toolkit for diffusion image and video generation: Unsloth Studio (in process
and over its HTTP API, which is also the Unsloth Desktop path), plain diffusers, BlineUI and stable-diffusion.cpp,
on NVIDIA, on CPU, and on the AMD Strix Halo gfx1151 CI (Linux ROCm and Windows 11). Every cell runs in its own
process with the same timing protocol, writes one `record.json` plus the rendered media, and is scored against a
reference from the same framework. It encodes three months of campaign lessons (offload, CUDA graphs, quant
ladders, step caching, compile tax, Studio vs BlineUI) so a re-run measures the same thing the original did.

## Quick start

The one process for a Studio vs baseline comparison (runbook: `claude/workflows/diffusion_bench_workflow.md`):
`plan.py` -> selftests + `--dry-run` -> parity gate -> trial cell -> timed run -> `score.py` -> `report.py`.

```bash
cd <scripts checkout>/scripts
export WORKSPACE=/path/to/workspace                        # outputs, venvs, caches all stay under it
R=$WORKSPACE/outputs/dbench/<round>

# selftests (CPU, minutes): driver + negative controls, the whole process end to end, targets, cards, builders
python diffusion_bench/selftest.py && python diffusion_bench/selftest_e2e.py
python diffusion_bench/targets/selftest_targets.py && python diffusion_bench/models/test_cards.py
python diffusion_bench/builders/test_builders.py

# plan: models (cards in models/) x arms x targets -> PLAN.json + cases.json; NOT_RUN lines say why
python diffusion_bench/plan.py request.json --out $R/plan/plan.json --dry-run

# run: one adapter per target (prepare is offline; the launch script / printed push starts it)
#   python diffusion_bench/target.py colab ... is the same from any cwd, also under PYTHONSAFEPATH=1
(cd diffusion_bench && python -m targets.colab prepare $R/plan/plan.json --out $R/colab --unsloth-sha SHA --bline-ref v0.45.6)
python diffusion_bench/matrix.py --plan $R/plan/plan.json --vm local-b200-1 --out $R/runs/local-b200-1   # single-lane VM
#   each VM: trial cell -> parity probes + gate (parity.json; a VOID group's timed cells are NOT_RUN) -> ABBA timed cells

# collect (unpack, token scan, stage cases.json), score, report
(cd diffusion_bench && python -m targets.colab collect $R/colab/runs/colab-l4-1 --plan $R/plan/plan.json)
python diffusion_bench/score.py $R/colab/runs/colab-l4-1
python diffusion_bench/report.py $R/colab/runs/colab-l4-1          # FASTER / SLOWER / TIE vs the A/A band

# outside tester: bundle out, results back
python diffusion_bench/export_tester_bundle.py $R/plan/plan.json --out $R/tester.zip --gpu-labels '0=Ada 48GB,1=RTX 3090'
```

### A fair Studio vs baseline round on GPU X for models Y (one command)

```bash
cd <scripts checkout>/scripts            # committed tree: the bundle is the recorded scripts SHA
export WORKSPACE=/path/to/workspace
python diffusion_bench/round.py --target colab:L4 --account michael \
    --models flux.3-klein:int8,bf16 qwen-image-2.1:int8,fp8 krea-2:bf16 \
    --out $WORKSPACE/outputs/dbench/l4_$(date +%Y%m%d)            # request + plan + notebooks; prints NOT_RUN lines
bash $WORKSPACE/outputs/dbench/l4_<date>/colab/launch_colab.sh all  # the VM (hours: run_in_background, cron watcher)
python diffusion_bench/round.py --out $WORKSPACE/outputs/dbench/l4_<date> --collect   # collect, score, report.md
```

- `--target`: `colab:T4|L4|A100-40|A100-80|G4` or `kaggle:T4x2`; one VM per target runs every model on it (one
  setup, one HF cache, ABBA per comparison). `--go` launches and collects in the same (long) call.
- `--models FAMILY:FMT,...`: card families (`models/*.json`). Each format gets a Studio and a baseline arm; include
  `bf16` when you want LPIPS (each stack is scored against its OWN bf16 on the same VM).
- Revisions: `--unsloth-sha` / `--zoo-sha` (default `main`, resolved to commits and recorded; the VM refuses a zoo
  install at any other commit), `--bline-ref` (default `models/BLINE_PIN.json`). Baseline GGUF cells pin the GGUF
  loader node the same way.
- Every round carries the parity negative control (`negative_control`): one Studio probe with a changed step count
  that the gate must VOID (`parity_negctl.json`, a line per card in `report.md`). If it ever passes, the group is VOID.
- Which GPU: the cheapest that fits the card (`min_vram_gib`, `min_host_ram_gib` per format; Colab T4 has 12.7 GB of
  host RAM, Kaggle T4x2 29 GB). fp8 needs sm89+ (L4, G4), nvfp4 sm100+ (G4; Studio builds ship NVFP4 off), bf16 on
  a T4 runs as fp16 on both stacks and the gate checks each stack's reported compute dtype (`dense_compute`).
  Anything a GPU cannot run is a NOT_RUN line with the reason, never a crash.
- A baseline file the card has no official source for (a template file only on a mirror) is NOT_RUN unless the
  round stages a mirror copy VERIFIED against the official checkpoint it repackages: `--downloads staged.json`
  (`{"file.safetensors": {"repo", "file", "sha256", "mirror": true, "official": {"repo", "file" | "files"},
  "verify": {"method": "sha256" | "tensors", "key_map"}}}`, `mirror.py`). `sha256`: the official repo publishes the
  same hash; `tensors`: every tensor equal on the VM before any cell uses it. Records carry `staged_files`, the
  report lists them; an unverified entry, or a quantised derivative (int8 ConvRot, requantised fp8), is refused.
- Clocks: a pair's SM clocks must be within 3%, or both runs ran at the same enforced power cap (within 1 W) and
  drew the same power (within 5%), or both ran below 90% of their cap with no power-cap / thermal / HW slowdown
  throttle reason in the timed window (NVML clock event reasons, recorded per run; unread = VOID); the report
  prints both arms' clock, draw and cap and lists the equal-power and below-cap pairs separately.
- A Studio "default" tier is recorded and compared as Studio resolved it (`resolved_tier`), not as requested.
- Cost and time: `plan.py --dry-run` (printed by round.py) gives VM minutes and CU per target; the trial cell's
  timing is the real number, and `report.md` ends with where each VM's time and CU went (setup, prefetch, file
  downloads / builds, trial, parity probes, timed runs). Spend over ~20 CU or any G4 / A100-80: ask first.
- What keeps a round cheap (each one a measured overhead of the first validation round): one VM per target for
  every model; the lane's first parity probe doubles as the trial; Studio's untimed probes fetch exactly the files
  Studio resolves (no whole-repo snapshots: sdxl-turbo is 55 GB, MiniMax-H3 500 GB) and the timed cells load
  offline; the baseline builders read the same hub cache; a comparison's downloads and builds are freed after its
  last cell; one same-prompt long / short pair per timed run (`short_n`, the polled per-step time is the primary
  s/step).

Comparing the outside tester's numbers with ours (their bench package; read it, never run it here):

```bash
python diffusion_bench/import_tester.py <their out>/results --out $WORKSPACE/outputs/dbench/tester_<date> \
    --gpu-labels '0=<their card>'      # baseline tag from their git describe (or --bline-tag SHA=TAG)
python diffusion_bench/report.py $WORKSPACE/outputs/dbench/tester_<date>   # same report shape as a round
```

- Their records are `legacy` (no parity evidence, no engagement, no A/A band): read their ratio as a second
  opinion, never as a verdict. Compare like with like: same workload (their K1 = our `flux.3-klein` int8, Q1 =
  `qwen-image-2.1` int8, T1 = `qwen-image-2.1-turbo` int8, KB = `flux.3-klein-base` int8, HI = `minimax-h3-image`
  int8, H1 = `minimax-h3` int8, Q3 = `qwen-image-2.1-edit` int8), the same card class and full card (their `_sim*`
  cells cap the VRAM the servers see and group separately), and the revisions each side pinned (their package: baseline
  v0.45.4, Studio 96bf6de124d; ours: `BLINE_PIN.json` and the round's `--unsloth-sha`).
- Known setting differences (their bench.py / make_plan.py vs our cards): H1 renders 960x544, our card the template's
  864x480; Q1 keeps the template's QwenImage21Cache prefix cache on the baseline, our default graph runs without it
  (`graph_extra.prefix_cache`); Q3 renders into a 1024^2 empty latent with TE resolution 1024 (our `o22-qwen-edit`
  graph), our card uses the reference-sized latent (resolution 0; their reference is 1024^2, so the size matches).
  Steps, guidance, sampler and size otherwise agree. Their per-request clocks are NVML means over the request;
  `telemetry` is the mean over the new-prompt requests. Their host memory is the whole host's used RAM
  (`legacy.host_used_peak_gib`), not the server RSS our `peak_rss_tree_gib` reports: do not compare the two.

Plumbing and behaviour tiers (no comparison, no plan):

```bash
python diffusion_bench/fetch_tiny.py                       # once: 14 public pipes, ~330 MB, into $WORKSPACE/hf_tiny
python diffusion_bench/matrix.py specs/tiny_plumbing.json --out $WORKSPACE/outputs/dbench/tiny_$(date +%Y%m%d_%H%M%S)
python diffusion_bench/matrix.py specs/smoke_small.json --out $WORKSPACE/outputs/dbench/smoke_$(date +%Y%m%d)  # one idle GPU
python diffusion_bench/edge/run_edge.py --surface both --tier fast --out $WORKSPACE/outputs/edge/$(date +%Y%m%d_%H%M%S)
python diffusion_bench/amd/selftest_amd.py                 # AMD layer: dry probe through amd_ci, criteria, scaffold
```

`matrix.py` resumes (a cell with an ok record is skipped unless `--force`), gates each cell on an idle GPU, and
kills a hung cell at `--timeout`. `--set KEY=JSON` overrides spec defaults, dotted keys included
(`--set n=2 --set options.studio_src=pr/11766`); a key a cell sets itself still wins.

## Files

| file | role |
|---|---|
| `fetch_tiny.py` | snapshots the hf-internal-testing tiny pipes into `$WORKSPACE/hf_tiny/<name>` (`local_dir`, no token, skips complete ones; `--set core|edge|matrix|all`) |
| `common.py` | prompts, spec loading (model aliases), GPU sampling and gate, host RSS, env fingerprint, media, records |
| `run_cell.py` | ONE cell in this process: the timing protocol below; writes `<out>/<tag>/record.json` + media |
| `matrix.py` | every cell of a spec, one process each, in the cell's venv, resumable, `--gpus` for one worker per GPU |
| `score.py` | LPIPS / SSIM / PSNR vs the reference, sanity flags, cross-framework floor, paired bootstrap, `scores.md` |
| `envs.py` | one reusable venv per (profile, python, torch pin, index); `ensure bench|studio|blineui|sdcpp`, `list` |
| `setup_studio.py`, `studio_client.py` | Studio tree / venv / server (`tree`, `ensure`, `launch`, `stop`), HTTP client |
| `setup_blineui.py`, `setup_sdcpp.py` | BlineUI clone + venv at a pinned commit; sd.cpp build (cuda / hip / vulkan / cpu) or prebuilt |
| `backends/` | `fake`, `studio` (in process), `studio_http`, `diffusers`, `blineui` (+ `bline_graphs/`), `sdcpp` |
| `public_bundle.py`, `extnames.py` | the rewritten public copy pushed to the AMD fork / staging repos (no path or file names the baseline UI; `write`, `check`, `smoke`), and the runtime crossings back to real names (`ext` / `norm`, word from `$BASELINE_REPO`); `private_defaults.json` holds the private URLs and is never bundled |
| `edge/` | Studio edge suite (`run_edge.py`) and the BlineUI / sd.cpp / diffusers one (`edge_bline_sdcpp.py`) |
| `amd/` | gfx1151 CI: probe, criteria, scaffold wrapper, token vault, self test |
| `plan.py` | request (cards x arms x targets) -> PLAN.json + cases.json: ABBA blocks, A/A pairs, parity probes, trial cell, lanes, NOT_RUN reasons from the card hints, CU estimate |
| `parity.py` | parity gate per pair (settings, sigmas, passes, size, precision, noise, cache, card traps); `--plan PLAN --results DIR` gates a plan |
| `report.py` | the one reporter: FASTER / SLOWER / TIE vs the A/A band, per-stack metrics, coverage, could-not-test |
| `card_timings.py` | each card's measured per-run load overhead (`est_load_s` per GPU and stack) from real records; `--write` updates the cards the planner estimates from |
| `models/` | one card per family: template defaults, formats (files, sources, precision, planning hints), traps; `BLINE_PIN.json` |
| `builders/` | the card's baseline-UI file build steps (`build_file`, `run.py --card F --format X`) |
| `targets/` | one adapter per target (`prepare` / `collect`), `vm_driver.py` (lanes, trial, gate, artifact), `fetch_files.py` |
| `export_tester_bundle.py`, `import_tester.py` | outside tester zip; their legacy `bench.py` results -> schema v2 |
| `schema.py`, `telemetry.py`, `budget.py` | record v2, 20 ms NVML sampler, VRAM ballast / simulated total / host-RAM cap |
| `specs/`, `prompts/` | cells to run; frozen prompt sets (`image_24`: 16 calib + 8 held-out, seeds 20260920+i; `video_3`, seeds 11+i) |

## Backends

| name | what it drives | in process | reference for | notes |
|---|---|---|---|---|
| `fake` | seeded noise, no model | yes | selftests | proves the plumbing only |
| `studio` | `DiffusionBackend` / `VideoBackend` from a Studio tree | yes | Studio cells | options = load kwargs; torch peak and host RSS are the model's own |
| `studio_http` | a Studio or Desktop server over `/api/inference/...` | no | the UI / Desktop path | launches its own server per cell unless `DIFFUSION_BENCH_STUDIO_URL` is set; wall includes HTTP, gallery |
| `diffusers` | plain diffusers pipeline | yes | framework-neutral baseline | offload none/model/sequential/group, compile regional/full |
| `blineui` | headless BlineUI, one server per cell, `/prompt` API | no | BlineUI cells | graphs in `bline_graphs/`; cache-busting nonce so timings never hit BlineUI's node cache |
| `sdcpp` | stable-diffusion.cpp `sd-server` / `sd-cli` | no | sd.cpp cells | score sd.cpp only against sd.cpp bf16 |

Studio cell options (the in-process backend passes these to `load_pipeline` / `begin_load`; the HTTP backend puts
them in the load body): `model_kind`, `family_override`, `base_repo`, `gguf_filename`, `memory_mode`
(auto/fast/balanced/low_vram), `speed_mode` (off/eager/default/max), `transformer_quant`
(none/int8/fp8/nvfp4/mxfp8/auto; an explicit scheme fails closed), `text_encoder_quant`, `attention_backend`,
`transformer_cache` (off/fbcache/static) + `transformer_cache_threshold`, `cpu_offload`, `loras`, `h3_task`,
`gpu_ids`; `options.load` / `options.generate` are passthrough dicts; `options.studio_src` picks the tree
(path, branch, `pr/N`, sha, `pypi`; default `$DIFFUSION_BENCH_STUDIO_SRC`). Env-only levers go in the cell's
`env`: `UNSLOTH_DIFFUSION_OFFLOAD_KEEP_CPU`, `UNSLOTH_DIFFUSION_OFFLOAD_PIN`, `UNSLOTH_DISABLE_CUDA_GRAPH`,
`UNSLOTH_DIFFUSION_COMPILE_VAE`, `UNSLOTH_STATIC_SKIP_{MODE,HEAD,TAIL,EVERY}`, `TORCHINDUCTOR_CACHE_DIR`.

## Venv profiles

One venv per (profile, python, torch pin, index URL, pins) under `$WORKSPACE/temp/diffusion_bench/venvs/`, reused
forever: a fresh venv pulls multi-GB wheels into page cache charged to your login session, so never build one per
run. `bench` (scorer, fake, diffusers), `studio` (a Studio tree's own requirement files; reuse an install with
`DIFFUSION_BENCH_STUDIO_PYTHON`), `blineui` (`DIFFUSION_BENCH_BLINE_PYTHON` / `_DIR` to reuse), `sdcpp`
(`DIFFUSION_BENCH_SDCPP_BIN` to reuse a build). A cell names its venv as a profile (`"studio"`, or `{"profile": "studio", ...}` with
`ensure` kwargs), a venv dir, or a python path; none means the driver's own interpreter. The torch index is detected from the driver (cu130 / cu128 / rocm / cpu);
`DIFFUSION_BENCH_TORCH_INDEX` overrides.

## Spec format

```json
{
 "name": "...", "description": "what it measures and how to read it",
 "models": {"zimage_turbo": {"repo": "Tongyi-MAI/Z-Image-Turbo", "revision": null,
                             "local": "$WORKSPACE/hf_local_bases/Tongyi-MAI/Z-Image-Turbo"}},
 "defaults": {"backend": "studio", "venv": {"profile": "studio"}, "width": 1024, "height": 1024, "n": 8,
              "options": {"speed_mode": "default", "local_files_only": true}},
 "passes": [{"name": "p1", "n": 24}, {"name": "p2", "n": 8}],
 "gate": {"max_util": 10, "min_free_mib": 90000, "timeout_s": 1800},
 "reference": {"studio": "s_bf16", "blineui": "c_bf16"},
 "baseline": {"source": "...", "...": "the original campaign's numbers, for comparison"},
 "daily_tags": ["..."],
 "cells": [{"tag": "zimg_int8", "model": "@zimage_turbo", "ref": "zimg_bf16", "steps": 9, "short_steps": 3,
            "guidance": 0.0, "options": {"transformer_quant": "int8"}, "env": {"X": "1"}}]
}
```

- `"model": "@alias"` resolves per host: `$DBENCH_MODEL_<ALIAS>` (e.g. `DBENCH_MODEL_ZIMAGE_TURBO`), else the
  alias's `local` dir if it exists, else its `repo` id. String options may use `@alias` too (e.g. `base_repo`). The
  record keeps `model_source` (env / local / repo). A repo id with `local_files_only: true` needs the snapshot in
  the HF cache first (`hf download <repo> --local-dir <dir>` then point the env var at it).
- `ref` per cell wins over the per-backend `reference` map (needed as soon as a spec has two models).
- Cell fields: `kind` image|video, `width`, `height`, `steps`, `short_steps`, `guidance` (null = family default),
  `negative_prompt`, `frames`, `fps`, `prompts` (name or path, .json rows or .txt), `ids`, `n`, `warmup`, `short_n`,
  `save_frames_every`, `options`, `env`, `venv`, `skip`. Unknown keys (`note`, `baseline`) are carried, not read.
- Operating point matters: steps and CFG must be the family's (Z-Image-Turbo 8-9 / 0, FLUX.1-schnell 4 / 0,
  Qwen-Image 20 / 4, Qwen-Image-2.1 25-40 / 1, Wan2.2-TI2V-5B family default).

## Timing protocol (`run_cell.py`, identical for every backend)

load (cold, `load_s`) -> warm-up render(s) on a DIFFERENT seed (`seed + 1_000_003`; `cold_s` is the first image:
compile, autotune, graph capture, pinning) -> N new-prompt renders (`wall_s_median`, what a user sees) -> `short_n`
repeated prompts at `steps` and at `short_steps` (`steady_s_median`; `step_s_derived = (median long - median
short) / (steps - short_steps)`, which cancels text encoder, VAE and export). The device is synchronised before
every timestamp. Nothing is timed by the backend itself; BlineUI / sd.cpp self-reported step times are recorded
beside it as `step_s`.

## Metrics

Speed: `load_s`, `cold_s`, new `wall_s_median`, `steady_s_median`, `step_s_derived`. Memory: `peak_alloc_gib`
(torch, in-process backends), `peak_smi_gib` (driver view, every process on the device) and
`peak_smi_delta_gib` (over the pre-load reading), host `rss_anon` / `rss_file` / `rss_shmem` / `peak_rss_gib`.
Quality (`score.py`): LPIPS **alex** (the Qwen-Image-2.1, video-gate and AMD campaigns' network; NVFP4
composition, the prequant gate and dynamic GGUF used vgg: never mix the two), SSIM, PSNR, mean / max abs, identical n/N; video per frame on the saved frames
plus a flicker ratio. `--pair A,B` gives a paired bootstrap 95% CI of the LPIPS difference (spanning 0 = tie).
Sanity flags on every render: black, constant, blank frame, frozen clip, missing media. Engagement: every record
keeps the backend's resolved status (`speed_optims`, `transformer_quant`, `transformer_cache` + stats,
`offload_policy`, `attention_backend`, `resolved{}`).

## Measurement rules (condensed; the evidence is in the knowledge base)

1. One cell per process: torchao flips inductor globals for the process, and `ru_maxrss` is a lifetime peak.
2. Guard `spawn` entry points with `if __name__ == "__main__"`: a probe once re-ran itself and put two arms on one GPU.
3. Cold compile needs a fresh process AND a fresh `TORCHINDUCTOR_CACHE_DIR`: `dynamo.reset()` keeps the disk cache.
4. Report warm, cold, compile tax and break-even separately; Studio recompiles per new prompt length (~14 s).
5. Pop HF tokens and set `HF_HUB_CACHE` in the workspace: the ambient token was invalid, and tokens leak into logs.
6. Never `&` inside a backgrounded command, never edit a harness mid-run, unique output dir per repeat.
7. s/step from long minus short steps on the same prompt: it cancels TE, VAE and export.
8. Synchronise before every timestamp: host step ticks ran 26 s ahead of the GPU under CUDA graphs.
9. Neither Studio API reports render time: client wall clock only; the server log is the detailed record.
10. The first render after load is cold (compile, graphs, MIOpen): always discard it (AMD cold 1021 s vs 220 warm).
11. BlineUI caches node outputs: warm-up and short/long renders need fresh seeds or a nonce, or you time a cache hit.
12. Count a BlineUI lever only when the server log shows it engaged; pass flags as `--bline-args="--fast"`.
13. Assert every lever engaged from the resolved status: fbcache, sage, flashinfer, int8 prequant all fell back silently.
14. Compare like with like: quant arms only compiled; always name the baseline arm (5.6x vs GGUF eager is 1.0x vs compiled bf16).
15. Score against the same framework's bf16 at the same seed: the Studio vs BlineUI bf16 floor was LPIPS 0.558.
16. Measure the noise floor with a bf16 repeat cell: below ~0.012 LPIPS is compile noise without deterministic config.
17. Never compare LPIPS alex with vgg; use paired per-prompt diffs with a CI, images are illustration only.
18. Few-step / 50-step video LPIPS vs bf16 is trajectory divergence: gate on paired gaps vs the incumbent (fp8).
19. Gate the reference's own reproducibility first: prompt rewriting (ERNIE) and batch padding broke it before.
20. Check steps and CFG match the family default: one gate ran Z-Image at 20 steps / CFG 4.
21. Bit-identity checks need `speed_mode="off"`: cudnn.benchmark and cuDNN attention vary across processes.
22. Deterministic algos need `fill_uninitialized_memory=False`, cost ~26% and crash NVFP4; raise `recompile_limit` to 256.
23. Pick GPUs by UUID, gate each cell (util, free memory), record contention, flag and never drop contended cells.
24. Time without an nvidia-smi poller where host-bound; `peak_smi_gib` is device-wide, prefer torch peaks in process.
25. Host RAM: read RssAnon / RssFile / RssShmem plus the lifetime peak; kept offload weights show as reclaimable file pages.
26. VRAM ballast emulates smaller cards (the planner reads `mem_get_info`); AMD unified memory is not returned promptly.
27. flashinfer `mm_fp4` has no device guard (wrong current device wedged GPUs); never `pkill -f`.
28. A differential whose base does not show the defect is VOID; measurement-only runs say `regression` mode.
29. CUDA graphs pay only when launch-bound (small res, NVFP4); at 1024 px and on video ~1.0x and cost reserved VRAM.
30. torch 2.12 + torchao 0.18 + `dynamic=True` hits CantSplit; torchao fp8 needs sm89+; ROCm compile fails entirely.
31. Never run a script placed directly in `$WORKSPACE/temp`: Python puts the script's dir first on `sys.path`, and a
    stray module dir there shadowed a real package and broke diffusers imports. Keep scripts in their own directory.
32. Tiny random pipelines prove plumbing, never quality or speed: their output is noise (tiny Wan emits blank frames).

## Specs

| spec | tier | what |
|---|---|---|
| `selftest_fake.json` | test | no model: driver, resume, passes, scoring |
| `tiny_plumbing.json` | tiny (every change) | hf-internal-testing random pipelines: FLUX (bf16 / repeat / int8 / fp8 / compiled), Qwen-Image-2.1 (bf16 / int8 / low_vram / fbcache canary), Qwen-Image true-CFG, SDXL, Lumina-2, Wan video (+ low_vram), Z-Image and SD3 via diffusers, one `studio_http` cell |
| `smoke_small.json` | daily | SDXL-Turbo, Z-Image bf16 / int8 / low_vram, FLUX.1-schnell, Qwen-Image-2.1 +/- static skip, Wan 5B video, one `studio_http` cell; `daily_tags` |
| `campaign_lowvram_offload.json` | campaign | stock vs keep-CPU vs pin model offload (PRs 11764 / 11766), FLUX / Z-Image / Qwen-Image-2.1 / Wan video, host RAM |
| `campaign_cuda_graphs.json` | campaign | graph on / off at 512 and 1024 px, bf16 / fp8 / nvfp4, Wan video |
| `campaign_quant_ladder.json` | campaign | bf16 / int8 / fp8 / nvfp4 compiled for Z-Image, FLUX, Qwen-Image-2.1; TE fp8 axis; repeat floors |
| `campaign_step_cache.json` | campaign | FBCache vs static skip (every 2 / 3, reuse), stacked on int8 / fp8, 25 and 40 steps, Wan video |
| `campaign_vae_compile.json` | campaign | `UNSLOTH_DIFFUSION_COMPILE_VAE` 0 / 1 / auto, fresh inductor cache per cell |
| `campaign_speed_modes.json` | campaign | speed off / eager / default / max, int8 max, GGUF eager vs compiled; compile tax |
| `compare_qwen21_allopt.json` | campaign | Studio vs BlineUI all optimisations (T0 / T1 / T2 tiers), interleaved |
| `compare_zimage_small.json` | campaign | Z-Image-Turbo across Studio, BlineUI, diffusers, sd.cpp GGUF |
| `sdcpp_gguf_small.json` | campaign | sd.cpp GGUF quants at 512 px |
| `amd_strix_small.json` | AMD CI | tier 1 `tiny_*` (the tiny set above minus compile / HTTP), tier 2 SDXL-Turbo / Z-Image 512 (bf16, repeat, speed off, int8, fp8, int8 + TE fp8) and Wan 5B at 480x320x9; `heavy_q21_*` opt-in |

Every `campaign_*` spec carries a `baseline` block with the original campaign's numbers and their source. Those
numbers were B200 (sm_100); compare on the same GPU class or not at all, and read each spec's description for the
caveats (baseline arms, LPIPS network, steps that differ).

Which tier: `tiny_plumbing` on every change (seconds per cell, any GPU, even a busy one: its gate never waits);
`smoke_small` daily for "did anything break today" (same GPU as yesterday's run, compare against yesterday's
`scores.json`); the edge suites for behaviour (bad inputs, odd sizes, unload mid-clip, lever refusal), which timing
cannot catch; a `campaign_*` spec when a number will be quoted, and then with the full `n`.

Tiny tier facts (hf-internal-testing, public, 1 to 56 MB, local copies in `$WORKSPACE/hf_tiny/<name>` made by
`fetch_tiny.py`, repo ids `hf-internal-testing/<name>`): Studio refuses images under 256 px (`MIN_OUTPUT_SIDE`, no bypass), tiny Z-Image's rope
(`axes_lens` 32) overflows at 256 px with a CUDA device assert, so tiny Z-Image runs through the diffusers backend at
64 px; tiny Wan runs at 64x64x9 (video has no floor) and at speed off (compile fails on it with a dynamo fake-tensor
SDPA error). Over HTTP, `/video/generate` only accepts the family presets (1280x704 / 704x1280 for Wan2.2-TI2V-5B,
422 otherwise) and the stock tiny Wan rope (`rope_max_seq_len` 32) cannot cover their 44x80 patch grid (500, "shape
'[1, 44, 1, -1]' is invalid"), so `fetch_tiny.py` also derives `tiny-wan-pipe-rope128`: the same files with
`rope_max_seq_len` 128 (the rope buffers are rebuilt from the config at load, no weight changes). The edge suite's
tiny tier uses it for every HTTP video check at 1280x704x9, 2 steps (about 5 s a clip). Edit pipelines (tiny FLUX Kontext, tiny Qwen-Image-Edit) need an input image and are left to the edge
suite; tiny Sana, HunyuanVideo and CogVideoX failed in plain diffusers at 64 px and are not in the spec. The HTTP
cell needs `DIFFUSION_BENCH_STUDIO_PYTHON` at a Studio install (`<home>/unsloth_studio/bin/python`).

## AMD gfx1151 CI

The AMD runners live on the public fork `oobabooga/unsloth` (the only repo with the `devlab-dispatch` pool).
`amd/scaffold_diffusion.py` builds a throwaway branch through `amd_ci/scaffold.py` (read `amd_ci/README.md` and
`claude/workflows/amd_ci_workflow.md` first): the workflow, `amd_ci/`, and a copy of `diffusion_bench/`.

```bash
python diffusion_bench/amd/selftest_amd.py                                        # always first
python diffusion_bench/amd/scaffold_diffusion.py --pr 11766 --out $WORKSPACE/temp/dbench_amd/ci_dbench_pr11766
python diffusion_bench/amd/scaffold_diffusion.py --pr 11766 --windows --out $WORKSPACE/temp/dbench_amd/ci_dbench_pr11766w
python diffusion_bench/amd/scaffold_diffusion.py --pr 11713 --mode differential \
    --defect slow:heavy_q21_fbcache:heavy_q21_bf16:0.9 --only 'heavy_q21_*' --out $WORKSPACE/temp/dbench_amd/ci_dbench_pr11713
# then, from the printed commands (the push TRIGGERS the run; this toolkit never pushes):
cd <out> && git init -q && git checkout -q -b amd-ci-dbench-prN && git add -A && git commit -q -m "..."
git remote add ooba https://github.com/oobabooga/unsloth.git && git push -q ooba HEAD:refs/heads/amd-ci-dbench-prN
```

- Linux: `runs-on: [self-hosted, Linux, strix-halo, devlab-dispatch]`, one GPU job only (`--no-suites`),
  `concurrency: {group: amd-ci-gfx1151-gpu, cancel-in-progress: false}`, job timeout 330 min. The job installs
  Studio from the head state with `install.sh --local`, which takes torch 2.11 + ROCm 7.13 from
  `https://repo.amd.com/rocm/whl/gfx1151/`; the probe runs in that venv against each state's tree.
- Windows: `runs-on: [self-hosted, Windows, strix-halo, devlab-dispatch]`, `shell: powershell`, group
  `amd-ci-gfx1151-gpu-windows`, plus the `linux-control` job. The probe installs ROCm torch from the gfx1151 index,
  the checkout's diffusers pin, `transformers==5.5.0` and Studio's `requirements/studio.txt` into the job venv, and
  `lpips` with `--no-deps`; NO torchao (its import fails on Windows ROCm).
- Tiers on the runner: `tiny_*` first (15 cells, a few MB of downloads, plumbing on real gfx1151: loads, family
  routing, weight-only int8 / fp8, low_vram offload on unified memory, video export, the diffusers backend), then the
  small real models. A PR that breaks plumbing fails in minutes rather than after a 30 GB download.
- `amd/diffusion_probe.py` (observes): drops tokens, reads the host (vendor, torch, HIP, arch), installs if needed,
  fetches the spec's public models once per job with `token=False` and retries, runs the selected cells through
  `matrix.py` (per-state `--budget-min`, default 100), scores, and writes per-cell facts (verdicts, timings, peaks,
  engaged levers, LPIPS vs the same state's reference, flags, pixel hashes, 32x32 thumbnails). `--edge` also runs
  the edge suite (`{out}` and `{model:ALIAS}` are filled in). `--skip-states merge` is the default.
- `amd/criteria_no_regression.py` (MODE regression): REGRESSION when a cell ok at base breaks, gains a sanity flag,
  loses more than `DBENCH_AMD_LPIPS_TOL` (0.03) LPIPS, or is slower than `DBENCH_AMD_MAX_SLOWDOWN` (1.25x) on BOTH
  new and steady s/image, or an edge check goes PASS -> FAIL. Tiny cells count only when broken or newly flagged. Gates (INCONCLUSIVE when any fails): both probes
  produced JSON, neither was `--dry`, both ran on ROCm gfx1151, models fetched without a token, same selection, base
  rendered something, and the `*_repeat` noise floor at base is under 0.02.
- `amd/criteria_differential.py` (MODE differential): `--defect broken:TAG | slow:TAG:REF:RATIO |
  quality:TAG:MAX_LPIPS | memory:TAG:GIB`. `amd_ci/lib/differential.py` makes a base that does not show the defect
  VOID, and `announce.py` fails the job on VOID / INCONCLUSIVE; read `VERDICT.md` from the artifact, not the tick.
- Local proof on any box: `python diffusion_bench/amd/diffusion_probe.py --state head --checkout <tree> --out
  obs.json --dry` (fake backend, valid JSON, real host detection). Without `--dry` on NVIDIA it runs the real cells
  (`--install never`), and the criteria still refuse it: not gfx1151.
- What gfx1151 cannot answer: Studio never compiles on ROCm (inductor `_alloc_from_pool`), so speed / CUDA graph /
  VAE-compile / nvfp4 campaigns are NVIDIA-only; there is no fp8 `_scaled_mm`; the `Not tested here` section of
  every verdict lists the rest (nvidia, discrete GPU, multi GPU, the other OS).

### Tokens on the AMD runners

Never. The specs use public, ungated weights; the probe removes `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`,
`HUGGINGFACE_HUB_TOKEN`, `HF_HUB_TOKEN` and downloads with `token=False`; no generated workflow contains `secrets.*`.
If a gated model is ever genuinely needed, the rule is "you must encrypt this if using this on the AMD CI machines":
mark the alias `"gated": true`, create a fine-grained READ-ONLY token for that one repo expiring within a day, and
encrypt it with `amd/token_vault.py` (AES-256-CBC + PBKDF2 via `openssl enc`, the scheme `scripts/account_pool.py`
uses for `account_pool.enc`, with its own passphrase variable `DBENCH_CI_PASSPHRASE`, 32 random bytes). The blob
rides in the branch; the passphrase can only reach the runner as an Actions secret on the host repo, which needs its
admin and an explicit per-run decision; the probe decrypts in memory (`--token-blob`) for `snapshot_download` only.
Revoke the token after the run. Encryption protects the blob on a public fork, not the token from the runner host.

## Edge suite (`edge/`)

`edge/run_edge.py` drives Studio's image and video backends two ways, in a killable worker process and over HTTP
against a launched or attached Studio / Unsloth Desktop, and writes `results.json` plus `summary.md` (exit 1 on any
FAIL). Where Studio states its intent (route bounds, `check_output_size`, the 4k+1 frame lattice, the GPU arbiter,
documented batch numerics) a check asserts it; otherwise it records an INFO observation instead of guessing. Every
check has a timeout: a hang is a FAIL and the surface is restarted. Evidence (stats, mean abs diffs, per-process
VRAM / RSS timelines, status codes, tracebacks, PNG strips) lands under `<out>/<surface>/<check>/`.

| tier | models | what | measured (B200, both surfaces) |
|---|---|---|---|
| `tiny` | hf-internal-testing random pipelines (`$WORKSPACE/hf_tiny/`) | plumbing only: sizes, determinism, leaks, validation, offload, compile, quant, per-family load matrix | 427 s, 55 pass; 588 s, 61 pass with HTTP video on the rope128 tiny Wan (Studio be4598e16, shared B200) |
| `fast` | SDXL-Turbo 512, Z-Image-Turbo 512/768, Wan2.2-TI2V-5B 17 frames | the above on real weights plus output sanity | 285 s; 575 s, 44 pass + 6 known Studio FAILs (`KNOWN_STUDIO_FAILS` in `run_edge.py`) on be4598e16 |
| `full` | fast plus FLUX.1-schnell, 2048 px | compile / CUDA-graph resize, quant switching, video memory modes, unload mid-clip | |

`edge/edge_bline_sdcpp.py` is the same idea for BlineUI, sd.cpp and plain diffusers (`--tier small|tiny|all`),
including the Studio vs diffusers bit-identity check on the tiny pipes.

The first runs already found real Studio issues (host RAM not returned after unload, about 5 GiB per SDXL load /
unload cycle; VRAM held when unloading during a generation over HTTP; a rope overflow device assert poisoning the
process); keep these checks, they are the point of the suite.

## Extending

- Backend: subclass `backends/base.py:Backend` (`load`, `render` returning `Render`, optional `status`, `close`,
  `trees`); synchronise before `render` returns; never time yourself end to end; add it to `REGISTRY`; add a fake
  cell to `selftest_fake.json` if the plumbing changed.
- Spec: copy the nearest one; give every cell a `ref` from the same framework and model; add a `*_repeat` cell when
  a quality threshold matters; put env levers in `env`, fresh `TORCHINDUCTOR_CACHE_DIR` for cold numbers; write the
  description as instructions for reading the result; `matrix.py SPEC --out DIR --dry-run` must list every cell.
- Edge check: register with `@check("area.name", tier="fast"|"full"|"tiny", needs="img_a", timeout=180,
  surfaces=("inproc", "http"))` in `edge/checks.py` (`edge/framework.py:check`); the first docstring line is the
  summary; assert on behaviour and media, not timings; return SKIP with a reason when a surface cannot answer.
- AMD: new probe arguments go through `scaffold_diffusion.py`; new judgements into a criteria module; run
  `amd/selftest_amd.py` and `amd_ci/selftest.py` before pushing anything.

## TODO(verify)

- The keep / pin engagement log line in Studio (the offload campaign asks for it); `COMPILE_VAE` and static skip
  being in the tree under test (they live in feature worktrees today).
- `studio_http` in `smoke_small.json` launches a server per cell: confirm the start-up cost keeps it near a minute
  (the tiny HTTP cell took 12.5 s to load on a B200 with a Studio install venv).
- FBCache did not engage on tiny Qwen-Image-2.1 with the wt_offload_pin tree (`transformer_cache: null`): check
  whether it engages on the real model on main before trusting any fbcache cell.
- On the AMD runners: SDXL-Turbo loading from the pruned download (fp16 variant files are skipped), the Windows
  diffusers pin from the checkout, `openssl` from Git for Windows for the token vault, the edge suite's HTTP surface.
- Z-Image-Turbo GGUF with `base_repo` as a local directory (`campaign_speed_modes.json`).
