"""Which scene photographs which PR, and what the shot is supposed to prove.

`expect` is the point of this file. A scene that runs cleanly and produces two
identical images is the default failure mode of this whole exercise -- a missed
click, the wrong dropdown, or a Studio that never rebuilt all look like "the PR
changed nothing". Writing down the expected difference BEFORE running turns that
silent failure into a checkable claim.

`needs_model` marks scenes that must load real weights. Those are slow and need a
GPU; the cheap ones (Hub listings only) should be developed first.

`parity` inverts all of that for a PR whose claim is that the UI does NOT change,
e.g. a pure performance fix. Identical halves become the thesis rather than the
default failure, so the driver flips its guards instead of being waved through
with `--allow-identical`: it requires the facts to MATCH, and requires the scene
to have returned content, because a scene that rendered nothing also produces two
matching halves. Set it on the plan; there is nothing to remember at the command
line. `expect` still carries the weight, and on a parity plan it must name
POSITIVE content both halves show, never "no difference".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ScenePlan:
    pr: int
    scene: str
    what: str                       # the UI surface photographed
    expect: str                     # the difference that MUST be visible
    kwargs: dict = field(default_factory=dict)
    needs_model: bool = False
    parity: bool = False            # the claim is that NOTHING changed
    verified: Optional[str] = None  # what was actually observed, once run


REGISTRY: dict[int, ScenePlan] = {
    8222: ScenePlan(
        pr=8222, scene="gguf_picker_rows",
        what="Model Hub quant list for a multi-checkpoint GGUF repo",
        expect="22 rows -> 63 rows; BF16 126 GB -> three 42 GB rows "
               "(BF16, BF16 · distilled, BF16 · distilled-1.1)",
        kwargs={"repo": "unsloth/LTX-2.3-GGUF"},
        verified="confirmed at head 69f54b51 on a clean two-install run: 22 -> 63 rows, "
                 "BF16 126 GB -> three 42 GB rows, Q8_0 68 GB -> 23 GB, and the bare "
                 "BF16 (dev) row selectable for the first time. API agrees: 117.45 GiB "
                 "-> 39.15 GiB. Side effect worth knowing: the DEFAULT selection also "
                 "changes (BF16 -> Q4_K_M · distilled-1.1), since the default resolver "
                 "now picks among real checkpoints. IDEMPOTENCE measured on full row "
                 "identity (key+label+size+filename, not row counts): flat, sharded and "
                 "quant-named-subdirectory repos are byte-for-byte unchanged; the mildly "
                 "affected ones keep their keys and only correct the size. The one "
                 "migration effect is that LTX-2.3-GGUF's surviving bare keys repoint "
                 "from distilled-1.1 to the dev checkpoint",
    ),
    8255: ScenePlan(
        pr=8255, scene="gguf_picker_rows",
        what="Model Hub quant list for a repo publishing one quant at several bit widths",
        expect="4 rows -> 18 rows, each at its own true size",
        kwargs={"repo": "byteshape/Llama-3.1-8B-Instruct-GGUF",
                # Fixed clip on the detail pane: the rows are 12 px type, and a
                # full 1500 px viewport renders at ~440 px per half in a comment.
                "clip": {"x": 780, "y": 150, "width": 700, "height": 620}},
        verified="confirmed by eye at head 4b470d414 vs merge base 871a088b5 (the base is "
                 "8222's branch `ggufrows`, NOT main -- this PR is stacked, so BEFORE "
                 "already contains 8222's family narrowing and the pair isolates the bpw "
                 "key alone). Picker: 4 rows -> 18 rows. BEFORE rows and the sizes the "
                 "picker advertised: Q4_K_S 16 GB, IQ4_XS 12 GB, Q3_K_S 16 GB, IQ3_S 18 GB "
                 "-- those are the SUMS of the 4/3/5/6 files sharing each token (Hub "
                 "listing: 16.0/11.6/16.1/17.7 GB), while the row itself resolved to one "
                 "file, Q4_K_S -> ...-Q4_K_S-3.60bpw.gguf at 3.63 GB. So BEFORE advertises "
                 "16 GB for a 3.6 GB download and the other 14 checkpoints are "
                 "unreachable. AFTER every file is its own row at its own size, 4.33 GB "
                 "down to 2.56 GB, and /api/models/gguf-variants agrees exactly with the "
                 "Hub listing on all 18. NOTE the summed size is computed in the FRONTEND: "
                 "the BEFORE backend endpoint already returns the single-file size "
                 "(3.63 GB) while the picker draws 16 GB, so the two disagree until this "
                 "PR keys them the same way. "
                 "GOTCHAS this run hit, both of which produced BYTE-IDENTICAL halves on "
                 "the first attempt (md5 equal, and the shot was of "
                 "YorkieOH10/Meta-Llama-3.1-8B-Instruct-Q8_0-GGUF). (1) a search RESULT ROW "
                 "is also a <button>, and on this query most results are named after a "
                 "quant (`...-Q8_0-GGUF`), so open_list's bare quant-token match hit the "
                 "results list and SELECTED an unrelated repo; the scene now filters the "
                 "selector with has_not_text on row chrome (relative date, download count). "
                 "(2) assert_showing passed on the wrong page: the results column is headed "
                 "'Results for \"byteshape/Llama-3.1-8B-Instruct-GGUF\"', which contains the "
                 "leaf, so the assertion was satisfied by the query typed rather than by "
                 "anything selected; the scene now asserts an EXACT heading plus the owner "
                 "line. Also: unsloth and byteshape both publish a repo named "
                 "Llama-3.1-8B-Instruct-GGUF and the owner-scoped results land before the "
                 "global ones, so `get_by_text(leaf).first` picked whichever had loaded -- "
                 "rows are now matched by owner, and Playwright's has_text normalises "
                 "whitespace so a `^owner$` row filter never matches",
    ),
    # MERGED 2026-08-09. Shot afterwards, at the merge base vs the merged head.
    8241: ScenePlan(
        pr=8241, scene="diffusion_quant_badge",
        what="loaded-models row for a GGUF pick the dense fast path replaced with a "
             "torchao int8 build",
        # Rewritten after reading the NET diff. The original expect ("BF16 -> Q8_0") was
        # taken from the PR title and from `gh pr diff --patch`, which replays the whole
        # commit series: an early commit added a `gguf_quant` field that a later one
        # dropped for the `gguf_variant` already on main. A plain GGUF load therefore
        # reads "GGUF · Q4_K_M" on BOTH sides and proves nothing.
        expect="loading unsloth/Z-Image-Turbo-GGUF Q4_K_M with transformer_quant=int8: "
               "'Image · z-image · GGUF · Q4_K_M · cuda' -> 'Image · z-image · INT8 · cuda'. "
               "The GGUF token goes because the pipeline never opened that file, and the "
               "precision becomes the dense build that actually ran. Same weights, same "
               "load, ONLY the label",
        kwargs={"repo": "unsloth/Z-Image-Turbo-GGUF",
                "filename": "z-image-turbo-Q4_K_M.gguf",
                "transformer_quant": "int8"},
        needs_model=True,
        verified="confirmed by eye at merged head 3c400936f vs merge base 8e558606a. Both "
                 "sides load identically (model_kind=gguf, gguf_variant=Q4_K_M, "
                 "transformer_quant=int8, dtype=bfloat16, device=cuda); the row reads "
                 "'Image · z-image · GGUF · Q4_K_M · c...' BEFORE (truncated by the 268 px "
                 "panel, which is itself the point: the width went on a filename nothing "
                 "opened) and 'Image · z-image · INT8 · cuda' AFTER. "
                 "GOTCHAS: (1) `gh pr diff --patch` replays the COMMIT SERIES, not the net "
                 "diff -- it showed a `gguf_quant` field this PR does not ship. Use "
                 "`git diff <base> <head>`. (2) /images/load returns in ~2 s with "
                 "loaded=false and finishes later, so status must be polled. (3) the whole "
                 "load, weights cached, takes about 10 s on this box",
    ),
    8219: ScenePlan(
        pr=8219, scene="image_generation_stop",
        what="Images composer action row during a generation, and again just after Stop",
        expect="two pairs on a 50-step batch-of-4 Z-Image run. (0) mid-run: BEFORE offers no "
               "way to stop it, AFTER shows Stop in place of Generate. (1) eight seconds "
               "later: AFTER is back to Generate with generate-progress active=false, BEFORE "
               "is still running. API: POST /images/generate/cancel 404 on the base, 200 on "
               "the head",
        kwargs={"repo": "unsloth/Z-Image-Turbo-GGUF",
                "filename": "z-image-turbo-Q4_K_M.gguf"},
        needs_model=True,
        verified="confirmed by eye at head 58c8c4642 vs merge base 8e558606a. pair_00 mid-run: "
                 "BEFORE a disabled spinner reading Generate, AFTER a live Stop. pair_01: "
                 "BEFORE still a disabled spinner (active=true at step 19 after 60 s), AFTER "
                 "back to a live green Generate with active=false. cancel route 405 -> 200. "
                 "GOTCHAS, three, and the middle one nearly shipped a wrong claim. "
                 "(1) get_by_role('button', name='Stop', exact=True) matches NOTHING here even "
                 "though inner_text is exactly 'Stop' -- the icon contributes to the accessible "
                 "name. Filter on text. That failure was silent: the click was skipped and the "
                 "pair simply showed two running sides. (2) cancellation latency is batch "
                 "dependent, measured: ~20 s at batch 4, under 4 s at batch 1. The original "
                 "fixed 8 s wait photographed AFTER still on Stop, which reads as 'the button "
                 "does nothing'. Poll to inactive instead. (3) BEFORE's Steps slider maxes at "
                 "100 and AFTER's at 50, so the two runs are not the same length; it does not "
                 "affect this claim but do not read it as a difference this PR made",
    ),
    8223: ScenePlan(
        pr=8223, scene="companion_assets_panel",
        what="Hub On Device tab: toolbar, and the delete dialog for an image GGUF and for "
             "the companion base repo its quants share (issue 8116)",
        expect="three differences on one cache holding unsloth/FLUX.2-klein-4B-GGUF "
               "(Q2_K + Q4_K_M, 4.4 GB) and the companion base "
               "black-forest-labs/FLUX.2-klein-4B (8.2 GB of text encoders, VAE, "
               "tokenizer). (0) toolbar: no way to reclaim shared assets -> a 'Free up "
               "space' control. (1) deleting the GGUF repo: 'You can re-download it later.' "
               "and nothing else -> plus 'Frees 4.4 GB of disk space.' and 'This also "
               "leaves 8.2 GB of shared assets (black-forest-labs/FLUX.2-klein-4B) that "
               "nothing else needs. Remove them with Free up space on the On Device tab.' "
               "(2) deleting the shared base while a quant is installed: the same plain "
               "dialog with Delete ENABLED (it succeeds, stranding both quants) -> a red "
               "'These are shared assets that unsloth/FLUX.2-klein-4B-GGUF still needs, so "
               "they cannot be removed yet. Delete those models first.' with Delete "
               "DISABLED. API: /api/hub/delete-impact and /api/hub/orphan-companions 404 "
               "on the base and answer on the head",
        kwargs={"gguf_repo": "unsloth/FLUX.2-klein-4B-GGUF",
                "base_repo": "black-forest-labs/FLUX.2-klein-4B"},
        verified="confirmed by eye at head aacfb9c0f vs merge base f0ef75ec9, two installs, "
                 "one seeded cache (scripts/seed_8223_cache.py: 4,432,118,912 B of Q2_K + "
                 "Q4_K_M and 8,229,021,460 B of companion base, real blobs). pair_00 "
                 "toolbar: 'Free up space' appears between Add folder and All formats. "
                 "pair_01 GGUF delete: BEFORE ends at 'You can re-download it later.'; AFTER "
                 "adds 'Frees 4.4 GB of disk space.' and the 8.2 GB orphan sentence. pair_02 "
                 "base delete: BEFORE identical wording with Delete ENABLED; AFTER shows the "
                 "red refusal naming unsloth/FLUX.2-klein-4B-GGUF with Delete greyed out. "
                 "API: delete-impact 405 / orphan-companions 404 on the base; on the head "
                 "reclaimed_bytes 4,432,118,912 with the base as freeable at 8,229,021,460, "
                 "and blocked_by=['unsloth/FLUX.2-klein-4B-GGUF'] for the base. "
                 "GOTCHA for later PRs: this box sets XDG_CACHE_HOME to a SHARED HF cache of "
                 "~26 repos which Studio scans alongside HF_HOME, so an unisolated run "
                 "photographs other sessions' downloads and its byte counts move under it. "
                 "Pass all four of HF_HOME/HF_HUB_CACHE/HF_XET_CACHE/XDG_CACHE_HOME through "
                 "--studio-env",
    ),
    8224: ScenePlan(
        pr=8224, scene="image_memory_plan",
        what="Images page after asking for 2048x2048 on a device that cannot hold the "
             "activations (GPU 2 held down to about 14 GiB free by gpu_ballast.py)",
        expect="BEFORE the generation is started and dies in the allocator, so the page shows "
               "a bare failure. AFTER it is refused before any work with the arithmetic: "
               "'Generating at 2048x2048 needs about 29.20 GB of working memory (including "
               "about 2.00 GB of fixed overhead), but only about 0.00 GB is usable on this "
               "device (of the 13.59 GB currently free...)'. 1024x1024 succeeds on both sides, "
               "which is the control: the claim is that the refusal replaces a crash, not that "
               "big requests are blocked",
        kwargs={"repo": "unsloth/Z-Image-Turbo-GGUF",
                "filename": "z-image-turbo-Q4_K_M.gguf",
                "width": 2048, "height": 2048},
        needs_model=True,
        verified="RAN, and it did NOT show what `expect` predicted -- it showed the opposite, "
                 "which is the point of writing `expect` first. At head 67cbde473 vs merge base "
                 "f0ef75ec9, both sides offload_policy=model, free 15.2 GiB (BEFORE) vs 13.9 GiB "
                 "(AFTER): BEFORE GENERATED the 2048x2048 image, AFTER refused it claiming "
                 "29.20 GB of working memory against '0.00 GB usable of the 13.87 GB currently "
                 "free'. Measured the truth on the base build by sampling nvidia-smi every "
                 "0.4 s: free 17,304 MiB before the call, 8,564 MiB at the trough, so peak "
                 "working memory about 8.7 GB, and it returned a 2048x2048 image. The estimate "
                 "is ~3x high and refuses work that succeeds. Reported on the PR. "
                 "GOTCHA that nearly invalidated the first run: this box SHARES GPUs, and "
                 "another session released ~17 GiB between the two sides, so they measured "
                 "different devices. gpu_ballast.py now tops up while holding (never gives "
                 "back), and the scene records free_mib per side. Without both, 'one refused, "
                 "one succeeded' says nothing about the PR",
    ),
    # 8213 was planned and never built: scene `unified_memory_refusal` does not
    # exist. Kept here as a comment rather than a live plan, because a plan whose
    # module is missing fails only after BOTH installs have run. The intent, if
    # anyone picks it up: load refusal on a host that cannot fit the model in
    # unified memory, expecting "load proceeds and is OS-killed -> a refusal naming
    # the shortfall". Needs a host that actually cannot fit it, so `gpu_ballast.py`.
    8232: ScenePlan(
        pr=8232, scene="gguf_download_plan",
        what="download manager panel while staging a GGUF pick, on the companion base item",
        expect="picking unsloth/Qwen-Image-2512-GGUF Q4_K_M stages the base repo at 58 GB "
               "-> about 17 GB, because the 11 dense transformer/ shards the GGUF replaces "
               "are no longer fetched. API plan total 66.08 GiB -> 28.02 GiB, transformer "
               "file count 11 -> 0. NOTE: the 'GGUF · BF16' label is 8241's bug, NOT this "
               "PR's; do not read a label change here as evidence for 8232",
        kwargs={"repo": "unsloth/Qwen-Image-2512-GGUF",
                "base_repo": "unsloth/Qwen-Image-2512",
                "filename": "qwen-image-2512-Q4_K_M.gguf",
                "quant": "Q4_K_M",
                # Both sides share one cache (--studio-env is not per side), so the scene
                # purges these two repos from it before each run. Without that the AFTER
                # total would be smaller because BEFORE already downloaded, not because of
                # the fix.
                "cache_hub": "outputs/ui_diff_8232/hf_cache/hub"},
        verified="confirmed by eye at head 02b77383e vs merge base f0ef75ec9, two installs, one "
                 "purged-per-side cache. pair_00 (the download panel, element shot): both halves "
                 "show unsloth/Qwen-Image-2512-GGUF as Downloaded, then the base item "
                 "unsloth/Qwen-Image-2512 at '4.3 KB / 58 GB' BEFORE and '357 MB / 17 GB' AFTER. "
                 "API plan: total 66.08 -> 28.02 GiB; the GGUF entry is 12.34 GiB on both sides "
                 "(unchanged, as it must be), and the base entry goes 53.74 GiB / 28 files / 11 "
                 "transformer shards -> 15.69 GiB / 17 files / 0. "
                 "GOTCHA: the panel shows ONE item at a time, so the shot has to wait for the "
                 "SECOND item; base_repo is a strict prefix of repo, so the match needs the "
                 "trailing separator or it fires on the GGUF row, which is identical on both "
                 "sides. Cost per side: the 13 GB GGUF downloads first (~20 s at 750 MB/s), then "
                 "the scene cancels the base transfer a few seconds in",
    ),
    # MERGED 2026-08-09. Shot afterwards, at the merge base vs the merged head.
    8196: ScenePlan(
        pr=8196, scene="video_family_train_picker",
        what="Images -> Train tab, the 'Model family' Select and the panel under it",
        # Written from the NET diff (git diff <merge-base> <head>), not from the PR body:
        # the body's "Deliberately left out / The Train UI listing" describes an EARLIER
        # commit. The merged head rewrites family_train_infos() to walk
        # _all_trainable_family_names() -- the image registry's trainable families UNION
        # TRAINABLE_VIDEO_FAMILIES -- so the listing does gain the video family.
        expect="the Model family dropdown gains one row, 'LTX-2', appended after the "
               "image families (mergeFamilies puts a backend family the frontend has no "
               "preset for last). Picking it, which is impossible on the BEFORE side, "
               "sets Base model to Lightricks/LTX-2 and shows the chips '19B' and "
               "'QLoRA 36GB+ VRAM' with the note 'Video: trains a style LoRA on still "
               "images.'. API /api/train/diffusion/info: families gains an ltx-2 entry "
               "with defaults rank 32 / lr 1e-4 / resolution 512 and deploy_base null. "
               "CAVEAT to check in the facts, not to assume: the head also drops any "
               "family whose pipeline class the installed diffusers lacks, so LTX-2 only "
               "appears where diffusers has LTX2Pipeline (0.39+)",
        verified="confirmed by eye at merged head 18dc63502 vs merge base f9656fd6c, two "
                 "installs, diffusers 0.39.0 with LTX2Pipeline present on BOTH sides (so the "
                 "availability filter is not what makes the difference). Model family menu: 7 "
                 "options -> 8, BEFORE flux.1 / flux.2-klein / flux.2-dev / qwen-image / "
                 "z-image / krea-2 / sdxl, AFTER the same seven plus ltx-2 appended last. "
                 "Picking it: Base model Lightricks/LTX-2, chips '19B' + 'QLoRA 36GB+ VRAM', "
                 "note 'Video: trains a style LoRA on still images.', defaults rank 32 / "
                 "lr 1e-4 / 512px / 20 warmup, deploy_base null (a video run publishes no "
                 "image LoRA catalog entry, so there is nothing to deploy). "
                 "CORRECTS AN EARLIER READ in this session that the feature was unreachable "
                 "from the Train tab. That came from the PR BODY, whose 'Deliberately left "
                 "out / The Train UI listing' paragraph describes an earlier commit; the net "
                 "diff at the merged head rewrites family_train_infos() to walk "
                 "_all_trainable_family_names() and the row is there. Read the net diff, not "
                 "the description, even on a merged PR",
    ),
    8244: ScenePlan(
        pr=8244, scene="video_family_train_picker",
        what="Images -> Train tab, the 'Model family' Select and the panel under it",
        # Written from the NET diff (git diff 749437314 70bdc2ceb), not the PR body. The
        # reachable-by-hand surface of this PR is one listing: TRAINABLE_VIDEO_FAMILIES
        # goes {ltx-2} -> {ltx-2, minimax-h3}, which is what _all_trainable_family_names()
        # feeds family_train_infos() and therefore GET /api/train/diffusion/info, which is
        # what the Select is built from. Everything else the PR ships (the H3 trainer, the
        # clip dataset layer, the packed-sequence layout, the H3 inference paths) is behind
        # that row.
        expect="the Model family dropdown gains exactly one row, 'MiniMax-H3': 8 options "
               "-> 9. mergeFamilies appends any backend family the frontend has no preset "
               "for, in backend order, and the video registry lists minimax-h3 before "
               "ltx-2, so the new row lands next to LTX-2 among the appended ones. Picking "
               "it, which is impossible on the BEFORE side, sets Base model to "
               "MiniMaxAI/MiniMax-H3 and shows the chips '31B' and 'QLoRA 72GB+ VRAM' with "
               "the note 'Video with sound: trains on clips that have a soundtrack.'. API "
               "/api/train/diffusion/info: families gains a minimax-h3 entry with defaults "
               "rank 16 / lr 1e-4 / resolution 768 / 20 warmup, deploy_base null, and "
               "precision_modes EMPTY (minimax-h3 is not in _DIT_TRAIN_FAMILIES, so /info "
               "advertises no base-precision list for it and recommended_precision is nf4). "
               "CAVEAT to read from the facts rather than assume: family_train_infos drops "
               "a family whose pipeline class the installed diffusers lacks, and H3's is "
               "ModularPipeline, so the row only appears where diffusers exposes it. "
               "RE-RUN AT THE LIVE HEAD (f73ac79e4): the precision_modes clause above is "
               "now OUT OF DATE and is the point of the re-run. The empty list was a real "
               "bug -- family_train_infos read _DIT_TRAIN_FAMILIES for is_dit while the "
               "rest of the PR had moved to _FLOW_TRAIN_FAMILIES -- and it is fixed at this "
               "head, so the pair must now show precision_modes NON-EMPTY for minimax-h3 "
               "and the floating Start control reading 'Start training' and ENABLED after "
               "picking it. A shot that still says 'Not supported on this GPU' means the "
               "AFTER home was built from a stale SHA, not that the fix is absent. "
               "RE-RUN 2026-08-10 (second) WITH OVERRIDDEN REFS AND A SEEDED CLIP FOLDER. "
               "This PR's own merge base can no longer show its effect: head f73ac79e4 "
               "withholds a clip-trained family until a listed dataset reports clips, and "
               "the clip_count field that makes any folder able to report clips is added by "
               "a DIFFERENT open PR, #8287. So the honest pair is BEFORE = 8287 alone "
               "(--base-ref tmp8287) and AFTER = 8287 + 8244 (--head-ref evcomb), with "
               "seed_clips=3 putting the SAME image folder and clip folder in both homes. "
               "Expect: BEFORE 8 options, no MiniMax-H3, target_in_api false; AFTER 9 "
               "options with MiniMax-H3 appended beside LTX-2, target_in_api true, picking "
               "it sets base MiniMaxAI/MiniMax-H3, and the floating Start control reads "
               "'Start training' and is ENABLED (precision_modes non-empty, the fix "
               "described above). The comment MUST say the refs are not 8244's merge base "
               "and why.",
        kwargs={"target": "MiniMax-H3", "family_key": "minimax-h3",
                "pipeline_attr": "ModularPipeline",
                # Taller than 8196's default crop and started past the sidebar: it reaches
                # the foot of the settings column, so the SAME frame carries the family the
                # PR adds and the state of the floating Start control for it. Without the
                # button in shot the pair answers "is it listed" and leaves "can it be
                # started" -- which is where this PR actually fails -- out of frame.
                "clip": {"x": 288, "y": 66, "width": 416, "height": 934},
                # Seeded on BOTH sides. Head f73ac79e4 withholds a clip-trained family
                # until some listed dataset reports clips, so with a stills-only home the
                # AFTER side hides minimax-h3 too and the pair goes identical for a reason
                # that has nothing to do with this PR. The seed is the same folder on both
                # halves, so the family list is still the only thing that differs.
                "seed_clips": 3},
        verified="confirmed by eye at head 70bdc2ceb vs merge base 749437314, two installs, "
                 "both reused on an exact .uidiff_sha match, both on GPU 2 (B200), login "
                 "identity verified per side (BEFORE :8996, AFTER :8997 from the driver's own "
                 "lines). ModularPipeline present on BOTH sides, so the availability filter is "
                 "not what makes the difference (diffusers 0.39.0 BEFORE, 0.40.0.dev0 AFTER -- "
                 "the PR moves the pin; noted because it is a second difference between the "
                 "sides, but it does not gate this row). "
                 "pair_00 menu: 8 options -> 9, MiniMax-H3 appended between Krea 2 and LTX-2, "
                 "exactly the backend order predicted. pair_01 panel: BEFORE stays on the "
                 "FLUX.1-dev default (there is no H3 row to click), AFTER reads MiniMax-H3 with "
                 "chips '31B' + 'QLoRA 72GB+ VRAM', base MiniMaxAI/MiniMax-H3 and the note "
                 "'Video with sound: trains on clips that have a soundtrack.'. So `expect` is "
                 "MATCHED on every clause, including precision_modes [] and "
                 "recommended_precision nf4. "
                 "BUT `expect` stopped one clause short of the thing that decides whether the "
                 "feature ships usable, and the answer is NO. That empty precision_modes is not "
                 "inert: the panel computes familyUntrainable = isDiT && "
                 "precision_modes.length === 0, where isDiT is merely `familyName !== \"sdxl\"`. "
                 "So [] on MiniMax-H3 disables the Start control and labels it 'Not supported on "
                 "this GPU'. Measured on the AFTER build, same Studio process, same B200, "
                 "seconds apart (scripts/h3_8244_start_gate.py): LTX-2 -> 'Start training', "
                 "enabled, precision_modes [nf4,bf16,int8,fp8,mxfp8,auto]; MiniMax-H3 -> 'Not "
                 "supported on this GPU', DISABLED, precision_modes []. Same host, so the "
                 "button's own wording is false. "
                 "ROOT CAUSE, from the net diff: the PR adds _FLOW_TRAIN_FAMILIES = "
                 "_DIT_TRAIN_FAMILIES | {minimax-h3} and switches three call sites to it "
                 "(bf16_unsupported_reason, dit_accelerator_missing_reason, "
                 "training_precision_preflight_error) but NOT the fourth, family_train_infos's "
                 "`is_dit = name in _DIT_TRAIN_FAMILIES`, which is what drives precision_modes. "
                 "The PR touches zero files under studio/frontend/src/features/images/train/. "
                 "THE TRAINER ITSELF IS FINE, which is why this is a listing bug and not a "
                 "feature failure: driving POST /api/train/diffusion/start directly on the same "
                 "build ran a real 20-step H3 LoRA to completion on 3 clips of 1280x720 24-frame "
                 "video WITH AAC stereo audio -- loss 0.415 -> 0.533 (avg 0.683), 0.447 img/s, "
                 "77.76 GB peak, and 332,674,080 B of adapter, 600 tensors over 200 modules at "
                 "rank 16 (lora_A [16,5376] / lora_B [7168,16] on the shared transformer_blocks "
                 "stack, no separate audio/video towers, as the PR's own comment describes). "
                 "GOTCHA: pair_01's two halves differ in BOTH family and button state, so it "
                 "alone cannot support the button claim -- a reader can answer 'different "
                 "families, of course'. The LTX-2 vs MiniMax-H3 shot on ONE build is the "
                 "controlled version and is the image that carries the finding. "
                 "RE-RUN 2026-08-10 at head 256a98a8d vs merge base 587590d6a, both sides "
                 "rebuilt (the stamps for 70bdc2ceb/749437314 no longer matched), login "
                 "identity verified per side, BEFORE :8998 AFTER :8999. The pair is now "
                 "IDENTICAL -- 8 options on both, target_in_api false on both -- and that is a "
                 "RESULT, not a trap. Head commit f73ac79e4 'Withhold a clip-trained family "
                 "from the Train picker until a clip dataset is listable' added "
                 "routes/training.py::_ui_trainable_families, which drops CLIP_TRAINED_FAMILIES "
                 "from /diffusion/info whenever no listed dataset reports clips; "
                 "DiffusionDatasetSummary carries no clip_count field at all yet, so every "
                 "folder answers 0 and minimax-h3 is withheld on every host. ModularPipeline is "
                 "True on both sides, so the availability filter is not the cause. The earlier "
                 "precision_modes finding was FIXED in the meantime (family_train_infos now "
                 "reads _FLOW_TRAIN_FAMILIES for is_dit) and is no longer observable through the "
                 "picker, because the row is not offered at all. "
                 "Trainer proof moved off the picker accordingly: "
                 "scripts/h3_8244_live_train_head.py starts the run through POST "
                 "/diffusion/start (which the gate still accepts by design) on the AFTER build "
                 "and photographs the Train tab mid-run. Completed 20/20, loss 0.415 -> 0.532 "
                 "(avg 0.681), 0.43 img/s, 77.76 GB peak, adapter 332,675,760 B / 600 tensors / "
                 "200 modules at rank 16. Composite: "
                 "outputs/ui_diff_8244/combined/pr8244_h3_trainer_at_head.png (NOT posted as an "
                 "image: its third pane shows the absolute on-disk adapter path). "
                 "MATCHED on the gated re-run, --base-ref tmp8287 (437c1ebe6) --head-ref evcomb "
                 "(631190074), both installs fresh, login identity verified per side, BEFORE "
                 ":9000 AFTER :9003, diffusers 0.40.0.dev0 with ModularPipeline True on both. "
                 "8 -> 9 options, MiniMax-H3 between Krea 2 and LTX-2; target_in_api false -> "
                 "true; the panel picks through to base MiniMaxAI/MiniMax-H3 with chips '31B' + "
                 "'QLoRA 80GB+ VRAM' and the soundtrack note; precision_modes "
                 "[nf4,bf16,int8,auto], recommended auto, and Start reads 'Start training' "
                 "ENABLED on BOTH sides -- so the pair no longer carries the button finding, "
                 "which is now fixed. The seeded dataset row reads 'clip-style - 3 clips' "
                 "identically on both halves, which is what makes the family list the only "
                 "moving part. Posted at "
                 "https://github.com/unslothai/unsloth/pull/8244#issuecomment-5237711835 with "
                 "the ref override stated in the comment itself.",
    ),
    8287: ScenePlan(
        pr=8287, scene="clip_dataset_picker",
        what="Images -> Train, the 'Training images' dataset picker, with an image "
             "folder and a clip folder seeded into the same datasets root",
        expect="BEFORE lists only 'photo-style - 3 images'; the clip folder is invisible "
               "because /diffusion/info admits a folder only on image_count > 0. AFTER "
               "lists BOTH, with 'clip-style - 3 clips' selectable, and picking it leaves "
               "the trigger showing clip-style. The image row must be IDENTICAL on both "
               "halves: if it is not, the panel failed to load and the pair proves nothing.",
        verified="MATCHED on a clean two-install run (base b063387cc, head accdf20fd). Same "
                 "seed on both homes: 3 ffmpeg-encoded MP4s with AAC in clip-style, 3 PNGs in "
                 "photo-style, a .txt beside every file. BEFORE /info returned 1 dataset "
                 "(photo-style, image_count 3, clip_count NULL -- the field does not exist on "
                 "that build) and the menu had 7 options, none of them the clip folder. AFTER "
                 "returned 2 (clip-style image_count 0 / clip_count 3 / caption_count 3, "
                 "photo-style 3/0/3) and the menu had 8, with 'clip-style - 3 clips' at the "
                 "top; dataset_after_pick went from 'photo-style - 3 images' to 'clip-style - "
                 "3 clips'. The control held: the photo-style row is identical on both halves. "
                 "Second pair is worth reading too -- the AFTER panel correctly drops the "
                 "thumbnail strip and the 'Review captions' toggle for a clip-only dataset, "
                 "both of which go through the image thumbnail endpoint.",
    ),
    8267: ScenePlan(
        pr=8267, scene="image_train_base_picker",
        what="Images -> Train, the FLUX.2 Klein family's 'Base model' Select and the "
             "FamilyFacts chips above it",
        # Written from the NET diff (git diff f7ea9fab6 <head>), not the PR body. Two
        # coupled changes: train_base_repos goes from the single distilled
        # black-forest-labs/FLUX.2-klein-4B to the two UNDISTILLED bases base-4B and
        # base-9B, and _BASE_TRAIN_SPECS overlays params 9B / qlora_vram_gb 18 on the 9B
        # one so it stops inheriting the family's 4B / 10 GB floor. FamilyFacts takes
        # baseModel, so the chips are a function of the base pick, not just the family.
        expect="the Base model dropdown under FLUX.2 Klein goes from 2 options "
               "(black-forest-labs/FLUX.2-klein-4B plus 'Custom repo or local path...') "
               "to 3 (FLUX.2-klein-base-4B, FLUX.2-klein-base-9B, Custom). Picking the "
               "9B row, which is impossible on the BEFORE side, moves the chips from "
               "'4B' / 'QLoRA 10GB+ VRAM' to '9B' / 'QLoRA 18GB+ VRAM'. API "
               "/api/train/diffusion/info: the flux.2-klein entry's base_repos goes 1 -> "
               "2, default_base changes from the distilled 4B to base-4B, and base_specs "
               "gains an entry keyed on black-forest-labs/flux.2-klein-base-9b with "
               "params 9B and qlora_vram_gb 18. THE TRAP HERE: the family-level params "
               "and qlora_vram_gb stay 4B / 10 on both sides, so a shot that photographs "
               "the chips WITHOUT picking the 9B base is two identical halves that look "
               "like a passing run. The base pick is the whole point.",
        verified="MATCHED by eye on a clean two-install run, base f7ea9fab6 vs head "
                 "0cd220171. Base menu 2 options -> 3: BEFORE "
                 "['black-forest-labs/FLUX.2-klein-4B', 'Custom repo or local path...'], "
                 "AFTER ['black-forest-labs/FLUX.2-klein-base-4B', "
                 "'black-forest-labs/FLUX.2-klein-base-9B', 'Custom...']. target_in_menu "
                 "False -> True, base_after_pick FLUX.2-klein-4B -> FLUX.2-klein-base-9B, "
                 "chip_texts ['4B'] -> ['9B'], and the panel reads 'QLoRA 10GB+ VRAM' -> "
                 "'QLoRA 18GB+ VRAM'. API: base_repos 1 -> 2, default_base distilled 4B -> "
                 "base-4B, base_specs null -> the 9B base and its unsloth mirror at params "
                 "9B / qlora_vram_gb 18, deploy_bases null -> 4 entries pairing each "
                 "training base to its inference base, vendor and mirror ids alike. "
                 "The predicted trap held exactly: family-level params and qlora_vram_gb "
                 "read 4B / 10 on BOTH sides, so the chip difference exists only after the "
                 "9B row is clicked. "
                 "SCENE GOTCHAS for whoever reuses image_train_base_picker. (1) "
                 "assert_showing waits on a HEADING, and the family name lives in a Select "
                 "trigger, so calling it here would have passed vacuously while the family "
                 "select sat on whatever it defaulted to; the scene asserts the trigger's "
                 "own inner_text instead. (2) the chip regex catches only the params chip "
                 "('4B'/'9B'), not 'QLoRA 18GB+ VRAM', so the VRAM figure comes from the "
                 "API facts and the screenshot rather than chip_texts. (3) the scene prints "
                 "facts truncated to 900 chars, which cut base_after_pick and chip_texts "
                 "off the AFTER line and briefly read like a missed click; the full facts "
                 "are in outputs/ui_diff_8267/meta.json, which is what to check.",
    ),
    10798: ScenePlan(
        pr=10798, scene="chat_workspace_files",
        what="the chat thread's workspace Files panel, over a fixed seeded workspace "
             "(notes.txt, README.md, src/analysis.py, src/data/rows.csv) written into "
             "the thread's sandbox workdir on BOTH sides",
        # Written from the NET diff before any run. The surface is asymmetric: on the
        # merge base the button, the aside and both routes do not exist, so the BEFORE
        # half's claim is an absence and the AFTER half's is content. The control is the
        # sandbox listing, which predates this PR and answers on both builds through the
        # same resolve_sandbox_workdir() the new route calls -- if it does not report the
        # same four files at the same sizes on both sides, the halves are not comparable
        # and the scene raises rather than shooting.
        expect="BEFORE: no 'Files' control anywhere above the thread and no side panel; "
               "GET /api/workspace-files/files answers 404 and /preview answers 404. "
               "AFTER: a ghost 'Files' button above the thread, aria-pressed=true once "
               "clicked, opening an <aside aria-label='Workspace files'> on the right "
               "with a Folder icon, a 'Files' header, Refresh and Close buttons and a "
               "'Filter files…' input. Its tree shows 5 entries -- src (expanded), "
               "src/analysis.py, src/data, notes.txt, README.md -- with src/analysis.py "
               "selected (aria-current=true) and the preview <pre><code> opening on "
               "'# analysis.py -- seeded by the workspace files scene' and containing "
               "'def load_rows(path):'. API: files 404 -> 200 listing README.md, "
               "notes.txt and src; preview 404 -> a text payload whose sha256 equals the "
               "seeded bytes. "
               "THE CONTROL, which must be IDENTICAL on both halves: GET "
               "/api/inference/sandbox/<session> reports the same four files "
               "(README.md, notes.txt, src/analysis.py, src/data/rows.csv) at the same "
               "sizes on BEFORE and AFTER, so the workspace content is a constant and "
               "the panel is the only thing that moved. "
               "THE TRAP HERE: the panel over an empty workspace renders 'No files in "
               "this workspace yet', which is a clean-looking screenshot that proves "
               "nothing about browsing; the scene raises on it rather than shooting it.",
        kwargs={
            # Fixed clip on the right-hand strip at a 1500x1000 viewport. The context
            # panel opens to 38% of the split (~470 px with the sidebar expanded,
            # ~550 px collapsed), so this holds the whole aside either way. The scene
            # measures the aside's real box and REFUSES to shoot if this clip would crop
            # it, since a cropped panel loses exactly the file names the claim is about.
            "clip": {"x": 920, "y": 0, "width": 580, "height": 1000},
        },
        needs_model=False,
        verified=None,
    ),
    # --- parity pairs: PRs whose whole claim is that the UI does NOT change ----
    #
    # The inverse of every other plan here. Elsewhere identical halves are the
    # default failure mode; here they are the thesis. `parity=True` is what tells
    # the driver to invert its guards, so each `expect` below names POSITIVE
    # content both halves must show, never "no difference": a scene that rendered
    # nothing also shows no difference, and a reader on the PR cannot tell those
    # apart.
    9012: ScenePlan(
        pr=9012, scene="chat_stream_content_parity",
        what="a streamed reply carrying a <think> block and a trailing ${...} "
             "fragment, mid-stream and settled, on an external provider so the "
             "placeholder strip actually runs",
        expect="BOTH halves show the reasoning pane expanded with 'weighing the "
               "two options' inside it, the answer ending '...so the second one "
               "wins.', and '${unclosed' still visible; the completed '${answer}' "
               "is absent from both. Facts: reply_sha256 equal, reasoning_groups=1, "
               "has_unclosed_fragment true, completed_fragment_survived false, "
               "reply_ends_with_unclosed true, strip_exercised true, all on both "
               "sides. The scene now RAISES on each of those rather than only "
               "recording them",
        kwargs={"payload": "think_and_placeholder", "external": True},
        needs_model=False, parity=True,
        verified="confirmed by eye at head dc332d527 vs merge base 0ac2e7992. Settled "
                 "shots BYTE-IDENTICAL. Both halves show the reasoning pane, the answer "
                 "ending 'so the second one wins.', and '${unclosed'; '${answer}' in "
                 "neither. reply_sha256 bf9516d8... on both, strip_exercised true on "
                 "both. That last one was read from the scene's own kwargs at the time, "
                 "so it evidenced only that an external connection was SEEDED, not that "
                 "the app selected it; it is now read off the intercepted request, and "
                 "the pair wants re-shooting before that clause is relied on. Cosmetic "
                 "residue worth knowing: in the MID-STREAM pair "
                 "the sidebar reads 'New chat' vs 'No chats yet'; both homes started at "
                 "zero threads and that is the Recents refresh landing either side of "
                 "the capture instant. Gone by the settled shot.",
    ),
    9014: ScenePlan(
        pr=9014, scene="chat_stream_content_parity",
        what="one assistant message with several finished parts plus a tool call "
             "going running -> complete, shot while running and once finished",
        expect="BOTH halves settled show the tool card COMPLETE (result row, not "
               "the spinner), 'Here is what I found:' still above it and 'That is "
               "the whole answer.' below; BOTH mid-stream shots show the same card "
               "RUNNING. Facts: tool_state_midstream=running, tool_state_settled="
               "complete, part_count equal, reply_sha256 equal, earlier_part_intact "
               "true, all on both sides. earlier_part_intact is the over-memoisation "
               "guard: if hoisting the components map froze a finished part, the text "
               "ABOVE the tool call is what goes stale, so a half missing it is a FAIL "
               "even though the halves would still look broadly similar",
        kwargs={"payload": "parts_and_tool"},
        needs_model=True, parity=True,
        verified="confirmed by eye at head 974d701fc vs merge base 0ac2e7992. Settled "
                 "shots BYTE-IDENTICAL: tick, args row, 'Result: 3 notes: streaming, "
                 "tool cards, themes.', with 'Here is what I found:' above and 'That is "
                 "the whole answer.' below, on both. Mid-stream shows the spinner and "
                 "collapsed chevron on both with the earlier text already rendered, so "
                 "the running->complete transition is shown on both builds. "
                 "reply_sha256 e0899c80... and part_count 3 on both.",
    ),
    9054: ScenePlan(
        pr=9054, scene="chat_stream_content_parity",
        what="composer and a finished thread while typing. Selected as LOW visible "
             "effect on purpose: this PR removes store subscriptions and is not "
             "expected to change a pixel",
        expect="Recorded as low visible effect rather than skipped. BOTH halves show "
               "the finished reply with its code fence syntax-highlighted, the Continue "
               "bar absent on the older message and present on the newest truncated "
               "one, and the composer holding 'does the composer still update'. Facts: "
               "reply_sha256 equal, composer_text equal and non-empty, "
               "continue_bar_count=1, highlighted_token_count equal and > 0. The "
               "composer fact is the point: this PR is about work done per keystroke, "
               "so a composer that stops holding typed text is the regression to catch",
        kwargs={"payload": "plain_thread", "type_in_composer": True},
        needs_model=True, parity=True,
        verified="confirmed by eye at head eed39d658 vs merge base 530799241. Both "
                 "halves: highlighted python fence (highlighted_token_count 18 both), "
                 "no Continue bar on the first turn, 'Response hit the Max Tokens "
                 "limit.' plus Continue on the newest, composer holding the typed text. "
                 "thread_sha256 950a23a8..., first_turn_sha256 1a5c446d... and "
                 "reply_sha256 b22bfd3d... equal on both. ONE surviving difference, "
                 "named rather than hidden: the action-bar duration reads 295ms vs "
                 "310ms, which is wall clock on a shared host, not content. Everything "
                 "else is pixel for pixel identical.",
    ),
}


def plan_for(pr: int) -> ScenePlan:
    if pr not in REGISTRY:
        raise KeyError(
            f"no scene registered for PR {pr}. Add a ScenePlan to REGISTRY with an "
            f"`expect` describing the difference the screenshot must show."
        )
    plan = REGISTRY[pr]
    # Checked HERE, before anything expensive. The driver imports the scene only
    # after resolving SHAs and installing, so a plan naming a module that does not
    # ship fails tens of minutes in, on a machine that has already built twice.
    if not (Path(__file__).resolve().parent / f"{plan.scene}.py").exists():
        raise FileNotFoundError(
            f"PR {pr} is registered against scene {plan.scene!r}, which does not exist "
            f"in {Path(__file__).resolve().parent}"
        )
    return plan


def missing_scenes() -> dict[int, str]:
    """Plans whose scene module does not ship, as `{pr: scene}`. For a test."""
    here = Path(__file__).resolve().parent
    return {pr: p.scene for pr, p in REGISTRY.items()
            if not (here / f"{p.scene}.py").exists()}
