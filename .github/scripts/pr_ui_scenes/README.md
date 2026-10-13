# pr_ui_scenes

Scenes for `pr_ui_diff.py`, the before/after Studio evidence driver. Read
`claude/workflows/pr_ui_evidence_workflow.md` before writing one: it carries the twelve
traps this package exists to guard against, every one of them observed live and every one
of them producing a clean-looking pair that proved nothing.

## Layout

| Path | What it is |
|---|---|
| `registry.py` | PR number to scene, its kwargs, and the `expect` string naming the difference the shot must show. Write `expect` BEFORE running. |
| `_common.py` | `pick_free_ports`, `studio_session`, `api_get`, `api_post`, `assert_showing`, `open_list`, `open_menu`. Each guards one way of producing worthless evidence; read the module docstring. |
| `<scene>.py` | One `async def drive(session, out_dir, label, **kwargs) -> (list[Path], dict facts)` per UI surface. Scenes are shared: `gguf_picker_rows` serves two PRs with different `repo` kwargs. |
| `test_pr_ui_diff_guards.py` | The identical-pair and facts-diff checks. |
| `test_pr_ui_diff_skip_install.py` | The driver's per-side reuse decision, with install, launch, login and scene stubbed. |

Both tests are offline and take under a second:

```
python3 pr_ui_scenes/test_pr_ui_diff_guards.py
python3 pr_ui_scenes/test_pr_ui_diff_skip_install.py
```

## Writing a scene

Return `(shots, facts)`. Put the quantitative claim in `facts`, read from the same server
that was photographed via `api_get`. A reviewer cannot count 63 rows by eye and a picture
cannot be diffed, so the numbers are what make the pair checkable.

Two things that bite, both of which pass silently:

- **`assert_showing` waits on a `heading`.** If the thing you selected renders into a
  Select trigger, a tab, or a chip rather than a heading, it passes without asserting
  anything. Assert on the trigger's own `inner_text` in that case.
- **Match `open_list` and `open_menu` on a value the list CONTAINS**, never on a category
  word. The toolbar format filter also reads "GGUF", so a category match opens a menu whose
  contents are identical on both sides.

## What the driver enforces

`pr_ui_diff.py` exits non-zero when the two sides are byte-identical, or when no scene fact
differs. Both are the failure this whole exercise is prone to, and both otherwise look
exactly like a successful run. `--allow-identical` overrides it, for the rare PR whose
visible effect genuinely cannot be produced on the host doing the shooting; say so in the
comment rather than posting the pair as evidence.

Those checks catch two sides that are the SAME. They cannot tell you that a real difference
is the RIGHT difference. Opening the composite and checking it against `expect` is still the
last step, and it is not optional.
