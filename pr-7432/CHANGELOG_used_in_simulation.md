# Changelog

Release notes for Unsloth and Unsloth Studio.

Unsloth Studio reads this file to show release notes inside the "New Unsloth
version" update popup.

## Format

Every release is a level-2 heading whose first token is the version, optionally
followed by a date. `## [2026.7.6] - 2026-07-22` and `## v2026.7.6` also work.

<!-- Add new releases directly below this line. -->

## Unreleased

## 2026.8.0 - 2026-08-05

<!-- Internal: reviewer sign-off still pending, do not ship this line. -->

### What's Changed

- Nested detail now renders under its bullet. Sub-items keep their own
  indentation instead of being flattened into the parent:
  - `--load-in-4bit` and `--load-in-8bit` are read together
  - relative documentation links resolve against the repository
  - a nested item never appears in the collapsed preview
- Inline code is styled as code. Run `unsloth studio -p 8931` and the flag
  reads as a flag, not as prose.
- The AMD guide moved to [docs/basics/amd.md](docs/basics/amd.md). The
  installer notes now live in [install.sh](install.sh).
- Reference-style links resolve too. See the [AMD guide][amd] and the
  [project README][readme].

### Upgrading

```bash
curl -fsSL https://unsloth.ai/install.sh | sh
unsloth studio -p 8931
```

<!--
A multi-line comment. None of these lines should be visible in the popup.
TODO: delete this block before tagging the release.
-->

[amd]: docs/basics/amd.md
[readme]: README.md

## 2026.7.6 - 2026-07-22

### What's Changed

- AMD support is here. Train, run RL, chat with and deploy 500+ models on
  Radeon, Instinct, Ryzen and data center GPUs across Windows, WSL and Linux,
  up to 2x faster with 70% less VRAM and no accuracy loss.
- Intel XPU support lands in Studio. Arc and Data Center GPUs now run chat and
  training alongside the NVIDIA, AMD and Apple paths.
- Local speech to text dictation runs fully offline. Slim Whisper bundles ship
  with Studio, and a picker accepts custom models.
- DoRA training is available in Studio. It sits next to LoRA and full
  fine-tuning in the training tab.
- The update popup previews release notes inline. They are pulled from
  CHANGELOG.md and matched to the exact version being offered.

### Running larger models

- Automatic GPU placement, or pick exactly which GPUs and layers to use.
- Move MoE expert layers into system memory so larger models fit.
- Split a model across several GPUs, or use tensor parallelism.
- Hardware settings are saved per model and quant.

### Also in this release

- Remote access with `unsloth studio --secure` over free HTTPS via Cloudflare.
- The model download location is configurable, so weights can live on a second
  drive instead of the default cache.

Full guide: [unsloth.ai/docs/basics/amd](https://unsloth.ai/docs/basics/amd).

## 2026.7.5

### What's Changed

- The previous release. It exists only to prove that an unmatched version
  never falls back onto a neighbouring section.

## 2026.8.5 - 2026-08-20

### Probe

| Column | Meaning |
| --- | --- |
| `a` | relative image below |

![Studio screenshot](images/studio.png)

- A bullet whose lead is a [relative link](docs/basics/amd.md). The rest follows.
- A bullet with an autolink <https://unsloth.ai> and an angle-bracket dest
  [spaced path](<docs/my file.md>).

## 2026.8.6 - 2026-08-25

- A bullet with a comment mid-sentence <!-- hidden here --> that continues after it.
- A bullet whose comment
  spans <!-- start
  end --> two lines of the same paragraph.
- <!-- a comment as the item's only content -->

<details>

<summary>Collapsed detail</summary>

Inside the details block, a [relative link](docs/basics/amd.md).

</details>

Trailing prose after the details block.
