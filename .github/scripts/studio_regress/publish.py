#!/usr/bin/env python3
"""Append ONLY the differing before/after screenshot pairs to the END of a PR description.

    python -m studio_regress.publish --pr N --report outputs/studio_regress/prN/report.json
        [--repo unslothai/unsloth] [--post] [--as danielhanchen] [--media-repo R] [--json]

Default is a dry run: composites are built and the would-be body is written to
<report dir>/publish/body.md; nothing is uploaded or posted.

With --post (trusted, local; PR code never sees the credential):
  1. report head_sha must equal the PR's current head, else STALE (exit 4), nothing written;
  2. only VISUAL_DIFF steps get images; DOM_ONLY_DIFF steps are text rows; SAME / FLAKY omitted;
  3. equal-scale padded composites (never resized), uploaded via upload_pr_media.py;
  4. the managed block between the markers is replaced (or appended at the END); a newer head
     with no differences removes an old block; the author's text is kept byte for byte;
  5. body re-read just before the PATCH (head and author text must not have moved), block
     linted with review_post.lint, identity verified (`gh api user` == --as) under GH_ROTATE=0.

Exit: 0 ok / nothing to do, 1 post failed, 2 lint refused, 4 stale report, 5 usage.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parent / "pr_review"))

START, END = "<!-- studio-regress:start -->", "<!-- studio-regress:end -->"
EXIT_OK, EXIT_POST, EXIT_LINT, EXIT_STALE, EXIT_USAGE = 0, 1, 2, 4, 5
_BLOCK = re.compile(r"\n*" + re.escape(START) + r".*?" + re.escape(END) + r"\n*", re.S)


def _log(msg):
    print(f"publish: {msg}", file=sys.stderr)


# -- body block ---------------------------------------------------------------------------

def strip_block(body):
    """Author text with any managed block removed (trailing whitespace normalised)."""
    return _BLOCK.sub("\n\n", body or "").rstrip()


def apply_block(body, block):
    """Replace the managed block, or append it at the END. block=None removes it."""
    author = strip_block(body)
    if block is None:
        return author + "\n" if author else ""
    return (author + "\n\n" if author else "") + f"{START}\n{block.strip()}\n{END}\n"


def has_block(body):
    return START in (body or "") and END in (body or "")


# -- selection + composites ---------------------------------------------------------------

def shown_pair(step):
    """(before, after) PNGs that show a VISUAL_DIFF step's change: the step shot, or, when only an
    extra shot (crawl overlay, upstream-suite PNG) differs, the first differing extra pair."""
    if step.get("main_visual_diff", True) and step.get("png_before") and step.get("png_after"):
        return step["png_before"], step["png_after"]
    for x in step.get("extras") or ():
        if x.get("png_before") and x.get("png_after"):
            return x["png_before"], x["png_after"]
    if step.get("png_before") and step.get("png_after"):
        return step["png_before"], step["png_after"]
    return None


def select(report):
    steps = report.get("steps", [])
    visual = [s for s in steps if s.get("verdict") == "VISUAL_DIFF" and shown_pair(s)]
    dom = [s for s in steps if s.get("verdict") == "DOM_ONLY_DIFF"]
    return visual, dom


def pad_pair(left, right, out, label_left, label_right, gap=24, label_h=56):
    """Side by side at native scale: the shorter image is padded, never resized."""
    from PIL import Image, ImageDraw
    li, ri = Image.open(left).convert("RGB"), Image.open(right).convert("RGB")
    h = max(li.height, ri.height)
    canvas = Image.new("RGB", (li.width + gap + ri.width, h + label_h), "white")
    canvas.paste(li, (0, label_h))
    canvas.paste(ri, (li.width + gap, label_h))
    d = ImageDraw.Draw(canvas)
    try:
        from studio_test_kit.compose import _font
        f = _font(28)
    except Exception:  # noqa: BLE001
        f = None
    d.text((16, 14), label_left, fill="black", font=f)
    d.text((li.width + gap + 16, 14), label_right, fill="black", font=f)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return out


def _resolve(report_dir, p):
    p = Path(p)
    return p if p.is_absolute() else report_dir / p


def composites(report, report_dir, out_dir):
    visual, _ = select(report)
    b, h = (report.get("base_sha") or "")[:9], (report.get("head_sha") or "")[:9]
    made = []
    for s in visual:
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", s["key"]) + ".png"
        pb, pa = shown_pair(s)
        made.append((s, pad_pair(_resolve(report_dir, pb), _resolve(report_dir, pa),
                                 out_dir / name, f"BEFORE {b}", f"AFTER {h}")))
    return made


# Text copied from the run into a public PR body (notes, facts, visible page text) can carry this
# host's paths or a token the page echoed: shorten paths to their last component, drop tokens.
_LOCAL_PATH = re.compile(r"(?:/mnt|/home|/tmp|/root|/Users|/var|/opt|/dev/shm)/[^\s`'\"|<>)\],]*")
_TOKEN = re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{20,}"
                    r"|sk-[A-Za-z0-9_-]{20,})\b")


def scrub(text):
    text = _TOKEN.sub("<redacted>", str(text))
    return _LOCAL_PATH.sub(lambda m: ".../" + (m.group(0).rstrip("/").rsplit("/", 1)[-1] or ""), text)


def _facts(delta):
    if not delta:
        return ""
    if isinstance(delta, dict):
        items = [f"`{k}`: {scrub(v)}" for k, v in list(delta.items())[:4]]
        return "; ".join(items)
    return scrub(str(delta)[:200])


def render_block(report, images):
    """Markdown for the managed block. images: [(step, url)]. No emojis, no em dashes."""
    _, dom = select(report)
    cov = (report.get("coverage") or {}).get("overall")
    lines = ["### Studio UI regression check",
             f"Base `{(report.get('merge_base') or report.get('base_sha') or '')[:9]}` vs head "
             f"`{(report.get('head_sha') or '')[:9]}`. Functional: **{report.get('functional', 'UNKNOWN')}**"
             + (f", UI coverage {cov:.0f}%" if isinstance(cov, (int, float)) else "")
             + f", {len(images)} visual change(s), {len(dom)} text-only change(s). "
             "Only steps whose screenshots differ are shown."]
    for s, url in images:
        lines += ["", f"**{s['key']}**" + (f": {scrub(s['note'])}" if s.get("note") else ""),
                  *([f"Facts: {_facts(s.get('facts_delta'))}"] if s.get("facts_delta") else []),
                  f"![{s['key']}]({url})"]
    if dom:
        lines += ["", "| Step | Text / structure change |", "|---|---|"]
        lines += [f"| `{s['key']}` | {scrub(str(s.get('dom_delta') or s.get('note') or ''))[:160].replace('|', '/')} |"
                  for s in dom]
    return "\n".join(lines)


# -- GitHub ---------------------------------------------------------------------------------

def _gh_read(cmd):
    try:
        from gh_read_env import run_gh_read
        return run_gh_read(cmd, label="studio_regress.publish")
    except ImportError:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=120)


def pr_state(repo, pr):
    r = _gh_read(["gh", "api", f"repos/{repo}/pulls/{pr}", "--jq", "{head: .head.sha, body: .body}"])
    if r.returncode != 0:
        raise RuntimeError(f"cannot read PR: {(r.stderr or '').strip()[:200]}")
    d = json.loads(r.stdout)
    return d["head"], d.get("body") or ""


def upload(pr, files, media_repo):
    cmd = [sys.executable, str(_HERE.parent / "upload_pr_media.py"), "--pr", str(pr), "--repo", media_repo,
           "--check"] + [x for f in files for x in ("--file", str(f))]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        raise RuntimeError(f"upload_pr_media failed: {(r.stderr or r.stdout).strip()[-300:]}")
    urls = [u for u in r.stdout.split() if u.startswith("https://raw.githubusercontent.com/")]
    if len(urls) != len(files):
        raise RuntimeError(f"expected {len(files)} URLs, got {len(urls)}")
    return urls


def patch_body(repo, pr, body, login):
    import review_post
    tok = review_post._token_for(login)
    if not tok:
        return False, f"no credential for {login}"
    who = review_post._gh(["gh", "api", "user", "--jq", ".login"], tok)
    if who.returncode != 0 or who.stdout.strip() != login:
        return False, f"identity check failed: token answers as {who.stdout.strip()!r}"
    payload = Path(os.environ.get("WORKSPACE", ".")) / "temp" / f"studio_regress_body_{pr}.json"
    payload.parent.mkdir(parents=True, exist_ok=True)
    payload.write_text(json.dumps({"body": body}))
    r = review_post._gh(["gh", "api", "-X", "PATCH", f"repos/{repo}/pulls/{pr}", "--input", str(payload),
                         "--jq", ".html_url"], tok)
    payload.unlink(missing_ok=True)
    return (r.returncode == 0), (r.stdout or r.stderr or "").strip()[:300]


# -- main ------------------------------------------------------------------------------------

def plan(report, body, images):
    """(new_body | None, action). None = nothing to write."""
    visual, dom = select(report)
    if not visual and not dom:
        return (apply_block(body, None), "remove") if has_block(body) else (None, "none")
    return apply_block(body, render_block(report, images)), "replace" if has_block(body) else "append"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pr", required=True, type=lambda s: int(str(s).lstrip("#").rsplit("/", 1)[-1]))
    p.add_argument("--repo", default="unslothai/unsloth")
    p.add_argument("--report", required=True)
    p.add_argument("--post", action="store_true", help="upload + PATCH (default: dry run)")
    p.add_argument("--as", dest="login", default="danielhanchen")
    p.add_argument("--media-repo", default="danielhanchen/unsloth-staging-2")
    p.add_argument("--body-file", help="dry run: use this as the current PR body (offline)")
    p.add_argument("--json", action="store_true")
    try:
        a = p.parse_args(argv)
    except SystemExit as e:
        return EXIT_USAGE if e.code else EXIT_OK
    rp = Path(a.report)
    report = json.loads(rp.read_text())
    out_dir = rp.parent / "publish"
    made = composites(report, rp.parent, out_dir / "composites")

    if a.body_file:
        head, body = report.get("head_sha"), Path(a.body_file).read_text()
    else:
        head, body = pr_state(a.repo, a.pr)
    if report.get("head_sha") != head:
        _log(f"STALE: report head {str(report.get('head_sha'))[:9]} != PR head {str(head)[:9]}; nothing written")
        return EXIT_STALE

    urls = [str(c) for _, c in made]
    if a.post and made:
        urls = upload(a.pr, [c for _, c in made], a.media_repo)
    new_body, action = plan(report, body, list(zip([s for s, _ in made], urls)))
    result = {"pr": a.pr, "action": action, "visual": len(made), "composites": [str(c) for _, c in made]}
    if new_body is None:
        _log("no differing steps and no old block: nothing to do")
        print(json.dumps(result) if a.json else "PUBLISH none")
        return EXIT_OK

    import review_post
    block = new_body[len(strip_block(body)):]
    viol = review_post.lint(block)
    if viol:
        for v in viol:
            _log(f"lint line {v['line']}: {v['rule']} {v['match']!r}")
        return EXIT_LINT
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "body.md").write_text(new_body)
    result["body_file"] = str(out_dir / "body.md")
    if not a.post:
        print(json.dumps(result) if a.json else f"PUBLISH dry-run {action}: {out_dir / 'body.md'}")
        return EXIT_OK

    head2, body2 = pr_state(a.repo, a.pr)  # re-read right before writing
    if head2 != head:
        _log("STALE: PR head moved during publish")
        return EXIT_STALE
    if strip_block(body2) != strip_block(body):
        _log("author edited the description during publish; recomputing on the fresh body")
        new_body, action = plan(report, body2, list(zip([s for s, _ in made], urls)))
        if new_body is None:   # the author removed the old block meanwhile: never PATCH a null body
            _log("nothing left to write on the fresh body")
            print(json.dumps({**result, "action": action}) if a.json else "PUBLISH none")
            return EXIT_OK
    ok, msg = patch_body(a.repo, a.pr, new_body, a.login)
    result.update(posted=ok, message=msg)
    print(json.dumps(result) if a.json else f"PUBLISH {'posted' if ok else 'FAILED'} {action}: {msg}")
    return EXIT_OK if ok else EXIT_POST


if __name__ == "__main__":
    sys.exit(main())
