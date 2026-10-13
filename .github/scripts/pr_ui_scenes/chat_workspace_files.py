"""Scene: the chat workspace Files panel PR 10798 adds, over a seeded workspace.

The PR adds `GET /api/workspace-files/files` + `/preview` and a
`WorkspaceFilesPanel`. On a thread that has a sandbox session, `chat-page.tsx`
renders a ghost `Button aria-label="Browse workspace files"` reading "Files"
above the thread; clicking it opens
`<aside aria-label="Workspace files">` with a `<h2>Files</h2>` header, Refresh /
Close buttons, an `Input aria-label="Filter files"`, a
`<nav aria-label="Workspace folders">` tree of `<button title={entry.path}>`
(directories carry `aria-expanded`, the selected file `aria-current="true"`) and
either a `<section aria-label="File preview">` or "Select a file to preview it.".

**This pair is ASYMMETRIC and that is the whole design problem.** On the merge
base the button, the aside and both routes DO NOT EXIST, so the obvious scene --
click "Files", assert the panel -- dies on BEFORE with a selector timeout and the
run produces nothing at all. So absence is RECORDED as a fact here rather than
raised on, and the same viewport is photographed either way. The asymmetry only
runs one way: on AFTER a missing panel, a missing entry or an empty preview
RAISES, because the AFTER half is the one whose content this evidence claims.

Three things make the halves comparable, none of them optional:

* **One fixed workspace, seeded on disk on both sides.** The panel over an empty
  workspace says "No files in this workspace yet", which is a weak shot and a
  weak claim. The scene writes the SAME four files with the SAME bytes into the
  session's sandbox workdir on both sides, so the content is a constant and the
  panel is the only moving part.
* **The seeding is verified through the photographed server, on both sides.**
  `GET /api/inference/sandbox/{session}` predates this PR and answers on the
  merge base too, and it resolves the workdir with the very same
  `resolve_sandbox_workdir()` the PR's new route calls. It reports the resolved
  path and the files in it, so a BEFORE side that seeded into a directory the
  server would never have served is caught HERE rather than being written off as
  "well, the panel does not exist on that side anyway". Its file list and sizes
  are the control: identical on both halves.
* **A fixed thread id and a fixed workspace, so nothing visible is per-side.**
  The thread is created through the API with a title and a createdAt that are
  constants, so the sidebar's Recents row reads the same on both halves.

Facts that differ between the sides BY CONSTRUCTION -- the thread id, the
per-home sandbox path, the panel's measured box -- are `_`-prefixed, which is
what keeps them out of the driver's comparison. An un-prefixed one would satisfy
the "some fact moved" guard on its own, without any fact about the UI having
moved.

Cheap by design (`needs_model=False`): no weights, no GPU, no network. The chat
thread is never sent a message, because the Files button is gated on a sandbox
session id derived from the thread id and nothing else.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

WORKSPACE = Path(os.environ.get("WORKSPACE") or os.environ.get("UNSLOTH_WORKSPACE") or Path(__file__).resolve().parents[2])
sys.path.insert(0, str(WORKSPACE / "scripts"))
sys.path.insert(0, str(WORKSPACE))

from pr_ui_scenes._common import Session, api_get, api_post, assert_clickable, assert_showing  # noqa: E402
from studio_test_kit.auth import seed_init_script  # noqa: E402
from studio_test_kit.ui import open_chat  # noqa: E402

# ── the seeded workspace ─────────────────────────────────────────────────────
#
# Fixed bytes, fixed names, no timestamps in any content: the two halves are shot
# minutes apart and anything derived from the clock would read as a difference
# this PR made. Written with `write_bytes` rather than `write_text` so the sizes
# the control endpoint reports cannot move with a newline translation.
#
# Chosen to exercise the panel's own branches: a nested directory (so
# `aria-expanded` and the chevron have something to do), a `.py` file (matched by
# the panel's code-file regex, so `FileCode` renders rather than `File`), a plain
# `.txt` and a `.md`, and a third level under `src/` so the tree is visibly a
# tree. Every name is boring on purpose: `contains_sensitive_path_component`
# refuses to browse a component like `.ssh` or `.config`, and a workspace whose
# folder is refused lists as empty with no error on screen.
ANALYSIS_PY = """\
# analysis.py -- seeded by the workspace files scene
import csv


def load_rows(path):
    with open(path, newline="") as handle:
        return list(csv.reader(handle))


def total(rows):
    return sum(int(row[1]) for row in rows)


if __name__ == "__main__":
    print(total(load_rows("data/rows.csv")))
"""

TREE: dict[str, str] = {
    "notes.txt": (
        "Workspace notes\n"
        "===============\n"
        "Seeded by the pr_ui_diff scene for PR 10798.\n"
        "The bytes are fixed so both halves show the same workspace.\n"
    ),
    "README.md": (
        "# Seeded chat workspace\n"
        "\n"
        "Four files, one nested folder, identical on the BEFORE and AFTER builds.\n"
    ),
    "src/analysis.py": ANALYSIS_PY,
    "src/data/rows.csv": "label,count\nalpha,3\nbeta,5\ngamma,8\n",
}

# The file the preview is opened on, and the line the shot must show. The first
# line is asserted EXACTLY and a body line by substring: a preview that rendered
# the wrong file, or an empty <pre>, both look like a perfectly good screenshot.
PREVIEW_PATH = "src/analysis.py"
PREVIEW_FIRST_LINE = "# analysis.py -- seeded by the workspace files scene"
PREVIEW_NEEDLE = "def load_rows(path):"

# The directory expanded before the file is clicked.
EXPAND_PATH = "src"

# What the tree must show once `src` is expanded: the three root entries plus the
# two under `src`. Asserted on AFTER, so a half-loaded tree cannot photograph as
# a working panel.
EXPECTED_ENTRY_NAMES = sorted(
    ["README.md", "notes.txt", "src", "src/analysis.py", "src/data"]
)
# What the pre-existing sandbox route must report on BOTH sides. Files only: it
# lists files, not directories, and hides its own `.unsloth_sandbox` marker.
EXPECTED_ON_DISK = sorted(TREE)

# ── identity of the chat this scene drives ───────────────────────────────────
#
# Minted ONCE per process, not per side, and the driver runs BEFORE and AFTER in
# the same process -- so both halves drive a thread with the same id, the same
# title and the same sandbox folder NAME, and only the home differs.
#
# Not a constant across runs, and it cannot be: `pr_ui_diff` clears every chat
# thread from a home before each side runs, and a deleted id is TOMBSTONED in
# `chat_thread_tombstones`, so re-creating a fixed id would 4xx on the second run
# against a reused home. A per-run id costs nothing visible: the id appears in
# the URL and in a folder name, neither of which is in shot.
#
# Matches `_SESSION_ID_RE` (`[A-Za-z0-9_-]{1,64}`) in
# `core/inference/tools.py`, which is what makes `_sandbox_name()` hand back the
# id itself instead of a `_id-<sha>` bucket -- i.e. what makes the workdir this
# scene seeds the workdir the server resolves.
SESSION_PREFIX = "uidiff-ws10798-"
RUN_SESSION_ID = f"{SESSION_PREFIX}{uuid.uuid4().hex[:10]}"

# Fixed so the Recents row in the sidebar reads identically on both halves; a
# `Date.now()` here would give the two sides different relative timestamps in
# shot. 2026-01-01T00:00:00Z in ms, the unit `createdAt` is stored in.
THREAD_TITLE = "Workspace files scene"
THREAD_CREATED_AT_MS = 1767225600000

# The marker `_owned_by_session()` reads. Written with the session name in it so
# the read-only resolver treats the seeded directory as this chat's rather than
# as somebody else's and serves `_nothing_to_serve()` instead. The panel never
# lists it: `resolve_path` refuses `.unsloth_sandbox` by name.
SANDBOX_MARKER = ".unsloth_sandbox"

# The right-hand strip the aside occupies, at a 1500x1000 viewport. The context
# panel opens to ARTIFACT_PANEL_DEFAULT_SIZE = 38% of the split, which is roughly
# 470 px with the sidebar expanded and roughly 550 px with it collapsed, so this
# clip holds either and a little thread pane besides. Fixed and identical on both
# sides -- an element screenshot is impossible here, because on BEFORE there is
# no element to shoot and the two halves would come out at different sizes, which
# `hstack_images` equalises by SCALING one of them.
#
# Verified against the measured box on AFTER rather than trusted: a clip that
# clears the panel's left edge silently crops the file names off the only half
# that has any.
DEFAULT_PANEL_CLIP = {"x": 920, "y": 0, "width": 580, "height": 1000}

VIEWPORT = (1500, 1000)

# Where the tree lives. Always scoped through this: `button[title="src"]` alone
# would happily match something else on the page, and a click that lands outside
# the panel leaves the panel on its default state in shot.
NAV = 'nav[aria-label="Workspace folders"]'
PANEL = 'aside[aria-label="Workspace files"]'
FILES_BUTTON = 'button[aria-label="Browse workspace files"]'

# Read entirely off the live DOM. Nothing here is echoed back from what this
# scene supplied: the entry names come from the rendered buttons' `title`
# attributes, the preview from the rendered `<pre><code>`.
READER = r"""
() => {
  const text = (n) => ((n && n.innerText) || "").replace(/\r/g, "");
  const button = document.querySelector('button[aria-label="Browse workspace files"]');
  const aside = document.querySelector('aside[aria-label="Workspace files"]');
  const nav = aside ? aside.querySelector('nav[aria-label="Workspace folders"]') : null;
  const entries = nav ? Array.from(nav.querySelectorAll("button[title]")) : [];
  const preview = aside ? aside.querySelector('section[aria-label="File preview"]') : null;
  const code = preview ? preview.querySelector("pre code") : null;
  const codeText = code ? (code.innerText || "").replace(/\r/g, "") : "";
  const filter = aside ? aside.querySelector('input[aria-label="Filter files"]') : null;
  const current = entries.find((e) => e.getAttribute("aria-current") === "true");
  return {
    files_button_present: Boolean(button),
    files_button_text: text(button).trim(),
    files_button_pressed: button ? button.getAttribute("aria-pressed") || "" : "",
    panel_present: Boolean(aside),
    panel_heading: aside ? text(aside.querySelector("h2")).trim() : "",
    refresh_button_present: Boolean(
      aside && aside.querySelector('button[aria-label="Refresh files"]'),
    ),
    close_button_present: Boolean(
      aside && aside.querySelector('button[aria-label="Close files"]'),
    ),
    filter_placeholder: filter ? filter.getAttribute("placeholder") || "" : "",
    tree_entry_count: entries.length,
    entry_names: entries.map((e) => e.getAttribute("title")).sort(),
    expanded_dirs: entries
      .filter((e) => e.getAttribute("aria-expanded") === "true")
      .map((e) => e.getAttribute("title"))
      .sort(),
    selected_entry: current ? current.getAttribute("title") : "",
    empty_workspace_message: nav
      ? text(nav).includes("No files in this workspace yet")
      : false,
    preview_placeholder_shown: aside
      ? text(aside).includes("Select a file to preview it.")
      : false,
    preview_present: Boolean(preview),
    preview_first_line: codeText.split("\n")[0],
    preview_chars: codeText.length,
    preview_head: codeText.slice(0, 300),
  };
}
"""


# ── seeding ──────────────────────────────────────────────────────────────────


def _sandbox_root(session: Session) -> Path:
    """Mirror of `core/inference/tools.py::sandbox_root()` for this install.

    `studio_root()` resolves `UNSLOTH_STUDIO_HOME`, which `launch_studio` sets to
    this side's home, and the sandbox sits under it. The override is honoured
    here for the same reason the backend honours it: `launch_studio` passes the
    ambient environment through, so a box that happens to export
    `UNSLOTH_STUDIO_SANDBOX_HOME` would put the real sandbox somewhere this scene
    would otherwise never look, and the seeding would land in an empty directory
    nothing serves.
    """
    override = (os.environ.get("UNSLOTH_STUDIO_SANDBOX_HOME") or "").strip()
    if override:
        return Path(override).expanduser()
    return Path(session.home).expanduser().resolve() / "sandbox"


def _inside_workspace(path: Path) -> bool:
    """Whether `path` is under `$WORKSPACE`, for the delete below."""
    try:
        path.resolve().relative_to(WORKSPACE.resolve())
    except (OSError, ValueError):
        return False
    return True


def _seed_workspace(session: Session) -> Path:
    """Write the fixed tree into this session's sandbox workdir and return it.

    The path is `<home>/sandbox/<session id>`, which is what
    `resolve_sandbox_workdir()` hands back for an id that is a usable directory
    name: `_session_dir()` returns the plain name when the marker names this
    session, and `_owned_by_session()` then lets a read-only caller serve it.
    Writing the marker is not decoration -- without it the resolver falls back on
    the directory's basename, and any future change to that fallback would have
    this scene photographing an empty panel for a reason invisible in the shot.
    """
    root = _sandbox_root(session)
    workdir = root / RUN_SESSION_ID

    # Homes are reused across runs and the session id is per-run, so yesterday's
    # folders would pile up in the sandbox root. Only this scene's own prefix,
    # only directories, and only inside the workspace.
    if root.is_dir():
        for stale in sorted(root.glob(f"{SESSION_PREFIX}*")):
            if stale.is_dir() and not stale.is_symlink() and _inside_workspace(stale):
                shutil.rmtree(stale, ignore_errors=True)

    workdir.mkdir(parents=True, exist_ok=True)
    for relative, content in TREE.items():
        target = workdir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content.encode("utf-8"))
    (workdir / SANDBOX_MARKER).write_bytes(RUN_SESSION_ID.encode("utf-8"))
    return workdir


def _seeded_sizes() -> dict:
    return {name: len(content.encode("utf-8")) for name, content in TREE.items()}


# ── API helpers ──────────────────────────────────────────────────────────────


def _get_status(session: Session, path: str, timeout: int = 60) -> tuple:
    """`(status, payload)` for an authenticated GET that is ALLOWED to 404.

    `api_get` raises, which is right for a route that must exist. Half of this
    scene's evidence is a route that must NOT exist on the merge base, and a
    raised HTTPError there is the finding rather than a failure.
    """
    req = urllib.request.Request(
        f"{session.base_url}{path}",
        headers={"Authorization": f"Bearer {session.access_token}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = response.read()
            return response.status, (json.loads(body) if body else None)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")[:300]
    except urllib.error.URLError as exc:
        # Not an HTTP answer at all (the server went away). Reported as status 0
        # rather than raised, so the scene's own diagnosis survives into `facts`.
        return 0, str(exc)


def _create_thread(session: Session) -> None:
    """Create the chat the Files button hangs off, through the API.

    The button renders only when `sandboxSessionIdFor(threadId, projectScope)`
    returns something, and outside a project that IS the thread id -- so a chat
    has to exist before there is anything to browse. It has to exist in the
    DATABASE too, not just in the URL: `/chat?thread=<id>` looks the thread up
    through `GET /api/chat/threads/{id}` and, on a miss, toasts "Chat not found"
    and navigates to a fresh chat, at which point the Files button is gone on
    BOTH sides and the pair proves nothing.
    """
    api_post(session, "/api/chat/threads", {
        "id": RUN_SESSION_ID,
        "title": THREAD_TITLE,
        "modelType": "base",
        "modelId": "",
        "createdAt": THREAD_CREATED_AT_MS,
        "updatedAt": THREAD_CREATED_AT_MS,
    })


def _verify_seed(session: Session, workdir: Path, facts: dict) -> None:
    """Prove the photographed server resolves the directory this scene seeded.

    `GET /api/inference/sandbox/{session}` is on the merge base as well as the
    head and goes through the SAME `resolve_sandbox_workdir()` the PR's new route
    calls, so this is the one check available on both halves, and it is what
    turns "the panel is empty" into a diagnosable failure instead of a shrug. Its
    file list and sizes are also the control for the pair: identical on both
    sides means the workspace content is a constant and the panel is the only
    thing that moved.
    """
    listing = api_get(session, f"/api/inference/sandbox/{RUN_SESSION_ID}")
    reported = listing.get("path") or ""
    files = listing.get("files") or []
    # `modified` is an mtime and differs between the sides for reasons that have
    # nothing to do with the PR, so only the names and sizes are kept.
    facts["workspace_files_on_disk"] = sorted(entry.get("name", "") for entry in files)
    facts["workspace_file_sizes"] = {
        entry.get("name", ""): entry.get("size") for entry in files
    }
    facts["_sandbox_path_reported"] = reported
    facts["_sandbox_path_seeded"] = str(workdir)

    if os.path.realpath(reported) != os.path.realpath(workdir):
        raise RuntimeError(
            f"the server resolves session {RUN_SESSION_ID!r} to {reported!r}, but this "
            f"scene seeded {str(workdir)!r}. The Files panel would list an EMPTY "
            "workspace and the shot would read as 'the panel does not work'. Check "
            "UNSLOTH_STUDIO_SANDBOX_HOME and _sandbox_root() against "
            "core/inference/tools.py::sandbox_root()."
        )
    if facts["workspace_files_on_disk"] != EXPECTED_ON_DISK:
        raise RuntimeError(
            f"the seeded workspace reads back as {facts['workspace_files_on_disk']}, "
            f"expected {EXPECTED_ON_DISK}. Both halves must hold the same files or the "
            "pair is not comparable."
        )
    if facts["workspace_file_sizes"] != _seeded_sizes():
        raise RuntimeError(
            f"seeded file sizes came back as {facts['workspace_file_sizes']}, expected "
            f"{_seeded_sizes()}; the two sides would not hold identical bytes."
        )


# ── page helpers ─────────────────────────────────────────────────────────────


async def _settled_box(page, selector: str, timeout_s: float = 15.0) -> dict:
    """The element's box once it has stopped moving.

    The context panel is opened by a JS `panel.resize("38%")` two animation
    frames after the click and animates over ~260 ms, with the surface faded in
    150 ms behind that. A screenshot taken on the click lands mid-slide, so the
    two halves would differ by how far the animation had got. `animations="disabled"`
    freezes CSS transitions but not the resize that drives them, so the box is
    polled to rest instead of waited on for a fixed time.
    """
    deadline = time.time() + timeout_s
    previous = None
    stable = 0
    box: dict = {}
    while time.time() < deadline:
        box = await page.locator(selector).first.bounding_box() or {}
        rounded = {k: round(v) for k, v in box.items()}
        stable = stable + 1 if rounded == previous else 0
        previous = rounded
        if stable >= 3 and box.get("width", 0) > 0:
            return box
        await asyncio.sleep(0.2)
    return box


async def _wait_for_text(locator, needle: str, what: str,
                         timeout_s: float = 30.0) -> str:
    """Poll an element's text until it CONTAINS `needle`, then return it.

    A presence check on the element is not enough: the preview mounts a Spinner
    first and then swaps in the `<pre><code>`, so an assertion on the section
    alone passes over an empty pane.
    """
    deadline = time.time() + timeout_s
    seen = ""
    while time.time() < deadline:
        try:
            seen = await locator.inner_text()
        except Exception:  # noqa: BLE001 -- not mounted yet
            seen = ""
        if needle in seen:
            return seen
        await asyncio.sleep(0.25)
    raise RuntimeError(
        f"{what} never showed {needle!r} within {timeout_s}s. On screen: {seen[:400]!r}"
    )


async def _open_panel(page, facts: dict) -> None:
    """Click Files, then prove the panel and its contents actually arrived.

    Every click below is followed by an assertion on a VALUE, never on the
    presence of a heading: `assert_showing` waits on a `heading` and this panel's
    heading is the word "Files", which a page can carry for half a dozen other
    reasons. It is called once as a cheap gate and is NOT what any claim here
    rests on.
    """
    button = page.locator(FILES_BUTTON).first
    # Hit-test BEFORE clicking. An occluded control otherwise surfaces as a 30
    # second Playwright timeout naming the intercepting element, which reads like
    # a bad selector rather than what it is: the feature's only entry point being
    # unreachable. Measured on 10798's head, this reports 0 of 19 points.
    facts["files_button_hit_test"] = await assert_clickable(
        page, FILES_BUTTON, "the Files button")
    await button.click(timeout=30_000)

    await page.locator(PANEL).first.wait_for(state="visible", timeout=30_000)
    # The cheap gate. Kept because it is the shared helper, not because it
    # decides anything -- see the docstring above.
    await assert_showing(page, "Files")

    # The click LANDED, read off the control itself.
    pressed = await button.get_attribute("aria-pressed")
    if pressed != "true":
        raise RuntimeError(
            f"the Files button reads aria-pressed={pressed!r} after the click, so the "
            "panel on screen is not the one this click opened"
        )

    # The tree has content, not a spinner and not the empty-workspace line.
    await page.locator(f'{NAV} button[title="notes.txt"]').first.wait_for(
        state="visible", timeout=45_000
    )
    nav_text = await page.locator(NAV).first.inner_text()
    if "No files in this workspace yet" in nav_text:
        raise RuntimeError(
            "the panel reports an empty workspace, so the server resolved a different "
            "directory from the one this scene seeded (or the seed was wiped between "
            f"the check and the click). Tree text: {nav_text[:300]!r}"
        )

    # Expand the nested folder, and prove it by the CHILD that appears, not by
    # the attribute alone.
    folder = page.locator(f'{NAV} button[title="{EXPAND_PATH}"]').first
    await folder.click(timeout=30_000)
    await page.locator(f'{NAV} button[title="{PREVIEW_PATH}"]').first.wait_for(
        state="visible", timeout=45_000
    )
    expanded = await folder.get_attribute("aria-expanded")
    if expanded != "true":
        raise RuntimeError(
            f"{EXPAND_PATH!r} reads aria-expanded={expanded!r} after the click even "
            "though its children rendered"
        )

    # Select the .py file and read its preview back out of the DOM.
    entry = page.locator(f'{NAV} button[title="{PREVIEW_PATH}"]').first
    await entry.click(timeout=30_000)
    await page.locator(
        f'{NAV} button[title="{PREVIEW_PATH}"][aria-current="true"]'
    ).first.wait_for(state="visible", timeout=30_000)
    await page.locator('section[aria-label="File preview"]').first.wait_for(
        state="visible", timeout=30_000
    )
    code = page.locator('section[aria-label="File preview"] pre code').first
    text = await _wait_for_text(code, PREVIEW_NEEDLE, "the file preview")
    facts["preview_needle_visible"] = PREVIEW_NEEDLE in text
    if text.splitlines()[0].strip() != PREVIEW_FIRST_LINE:
        raise RuntimeError(
            f"the preview opens on {text.splitlines()[0]!r}, expected "
            f"{PREVIEW_FIRST_LINE!r}; a preview of the wrong file photographs exactly "
            "like a preview of the right one"
        )


# ── the scene ────────────────────────────────────────────────────────────────


async def drive(session: Session, out_dir: Path, label: str,
                clip: dict | None = None,
                **_: object) -> tuple[list[Path], dict]:
    """Two shots per side: the whole chat, and the right-hand strip up close."""
    clip = dict(clip or DEFAULT_PANEL_CLIP)
    strict = label.upper() == "AFTER"
    facts: dict = {
        "_thread_id": RUN_SESSION_ID,
        "_seeded_paths": sorted(TREE),
        "_label": label,
        # Seeded so the key exists on BOTH sides: a key present on one side only
        # counts as a moved fact, which would let a half that never opened the
        # panel satisfy the driver's guard on its own absence.
        "preview_needle_visible": False,
    }

    workdir = _seed_workspace(session)
    _create_thread(session)
    # Both sides, because this is what proves the two workspaces are the same
    # one. It raises on either side, this PR or not: a BEFORE half that seeded
    # somewhere the server does not read is not a valid "the panel does not
    # exist yet" shot, it is an unseeded workspace with no panel over it.
    _verify_seed(session, workdir, facts)

    # The routes the PR adds, read from the server that is about to be
    # photographed. 404 on the merge base is the finding, so these must not
    # raise.
    files_status, files_body = _get_status(
        session, f"/api/workspace-files/files?session={RUN_SESSION_ID}&path=&q="
    )
    facts["files_api_status"] = files_status
    if isinstance(files_body, dict):
        facts["api_root_entry_names"] = sorted(
            entry.get("path", "") for entry in files_body.get("entries", [])
        )
    else:
        # A 404 body is the route's error text, not a listing.
        facts["api_root_entry_names"] = []

    preview_status, preview_body = _get_status(
        session,
        f"/api/workspace-files/preview?session={RUN_SESSION_ID}&path={PREVIEW_PATH}",
    )
    facts["preview_api_status"] = preview_status
    if isinstance(preview_body, dict) and preview_body.get("kind") == "text":
        content = preview_body.get("content") or ""
        facts["preview_api_kind"] = "text"
        facts["preview_api_sha256"] = hashlib.sha256(content.encode()).hexdigest()
        facts["preview_api_chars"] = len(content)
    else:
        facts["preview_api_kind"] = ""
        facts["preview_api_sha256"] = ""
        facts["preview_api_chars"] = 0

    if strict:
        if files_status != 200:
            raise RuntimeError(
                f"GET /api/workspace-files/files answered {files_status} on the AFTER "
                f"build: {files_body!r}. The panel cannot list anything without it."
            )
        if facts["preview_api_sha256"] != hashlib.sha256(
            TREE[PREVIEW_PATH].encode()
        ).hexdigest():
            raise RuntimeError(
                "the preview route did not return the seeded bytes for "
                f"{PREVIEW_PATH}: status {preview_status}, body {preview_body!r}"
            )

    init = seed_init_script(
        type("A", (), {"access_token": session.access_token,
                       "refresh_token": session.refresh_token})(), []
    )

    shots: list[Path] = []
    async with open_chat(session.base_url, init_scripts=[init],
                         viewport=VIEWPORT, headless=True) as sp:
        page = sp.page
        await page.goto(f"{session.base_url}/chat?thread={RUN_SESSION_ID}",
                        wait_until="domcontentloaded")
        await page.locator("form:has(textarea) textarea").first.wait_for(
            state="visible", timeout=60_000
        )
        # The thread lookup runs after boot and navigates away on a miss, so the
        # URL is re-read rather than assumed. Without this a "Chat not found"
        # redirect gives two buttonless halves and a clean-looking pair.
        await asyncio.sleep(3)
        facts["_url"] = page.url
        if f"thread={RUN_SESSION_ID}" not in page.url:
            raise RuntimeError(
                f"the app navigated away from the seeded thread to {page.url!r}; the "
                "Files button is gated on a thread id and there is now none"
            )

        present = await page.locator(FILES_BUTTON).count() > 0
        # Keyed on what is ON SCREEN, not on the label: if the button turned up
        # on the merge base the honest thing is to drive the panel there too and
        # let the driver's "no fact moved" guard report it, rather than to
        # photograph a closed panel because the label said BEFORE.
        if present:
            await _open_panel(page, facts)
        elif strict:
            raise RuntimeError(
                "no button[aria-label='Browse workspace files'] on the AFTER build. "
                "Either the home was built from a stale SHA, or the thread has no "
                "sandbox session (the button is gated on `sandboxSessionIdFor`)."
            )

        # Let the panel's open animation come to rest before either shot, so the
        # two halves are not separated by how far the slide had got.
        if present:
            facts["_panel_box"] = await _settled_box(page, PANEL)
        await page.wait_for_timeout(1_200)

        facts.update(await page.evaluate(READER))

        out_dir.mkdir(parents=True, exist_ok=True)
        # Shot 0: the whole viewport, for the layout claim (a side panel that is
        # not there at all on one half). Viewport rather than full_page so both
        # halves come out at exactly the same size.
        whole = out_dir / f"{label.lower()}_0_chat.png"
        await page.screenshot(path=str(whole), animations="disabled")
        shots.append(whole)

        # Shot 1: the right-hand strip, so the file names and the preview are
        # legible at the ~440 px per half a GitHub comment renders.
        box = facts.get("_panel_box") or {}
        if box.get("width"):
            left, right = box["x"], box["x"] + box["width"]
            if left < clip["x"] - 1 or right > clip["x"] + clip["width"] + 1:
                raise RuntimeError(
                    f"the panel sits at x={left:.0f}..{right:.0f} but the fixed clip "
                    f"covers x={clip['x']}..{clip['x'] + clip['width']}, so the shot "
                    "would crop the file names off the only half that has any. Pass a "
                    "wider `clip` in the registry kwargs."
                )
        panel_shot = out_dir / f"{label.lower()}_1_panel.png"
        await page.screenshot(path=str(panel_shot), clip=clip, animations="disabled")
        shots.append(panel_shot)

    # Everything the AFTER half's claim rests on, asserted rather than recorded.
    # A panel that mounted empty, or a preview that rendered nothing, produces a
    # perfectly ordinary screenshot.
    if strict:
        if not facts.get("panel_present"):
            raise RuntimeError("the Workspace files aside is not in the DOM at shot time")
        if facts.get("entry_names") != EXPECTED_ENTRY_NAMES:
            raise RuntimeError(
                f"the tree shows {facts.get('entry_names')}, expected "
                f"{EXPECTED_ENTRY_NAMES}"
            )
        if facts.get("selected_entry") != PREVIEW_PATH:
            raise RuntimeError(
                f"the selected entry is {facts.get('selected_entry')!r}, expected "
                f"{PREVIEW_PATH!r}"
            )
        if facts.get("preview_first_line") != PREVIEW_FIRST_LINE:
            raise RuntimeError(
                f"the preview opens on {facts.get('preview_first_line')!r}, expected "
                f"{PREVIEW_FIRST_LINE!r}"
            )
        if not facts.get("preview_chars"):
            raise RuntimeError("the preview <pre><code> is empty")
        if facts.get("empty_workspace_message"):
            raise RuntimeError(
                "the panel is showing 'No files in this workspace yet' at shot time"
            )
        for key in ("refresh_button_present", "close_button_present"):
            if not facts.get(key):
                raise RuntimeError(f"{key} is false; the panel header did not render")
    return shots, facts


if __name__ == "__main__":
    import argparse

    from pr_ui_scenes._common import studio_session

    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--home", type=Path, required=True)
    ap.add_argument("--password", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--label", default="AFTER")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    s = studio_session(a.url, a.home, a.password)
    print(json.dumps(asyncio.run(drive(s, a.out, a.label))[1], indent=2))
