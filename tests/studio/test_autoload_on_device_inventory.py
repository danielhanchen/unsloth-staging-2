# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Guards for the send-time auto-load sweep (#7374).

A Send with no model selected used to sweep only the two managed HF cache lists
and, finding nothing there, announce "No downloaded models found" and download
``unsloth/Qwen3.5-4B-MTP-GGUF`` -- while the reporter's GGUF sat in a registered
Custom Folder that the picker listed the whole time.

The selection rule is exercised through node against rows shaped like the ones
``GET /api/hub/local`` emits for that disk layout, so it checks behaviour rather
than the presence of a string. The rest pins the policy that lives in
``chat-adapter.ts``: read the indexed inventory, never download unasked, and
never report an unreadable inventory as an empty device.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

WORKDIR = Path(__file__).resolve().parents[2]
FRONTEND = WORKDIR / "studio/frontend/src"
INVENTORY = FRONTEND / "features/chat/utils/auto-load-inventory.ts"
ADAPTER = FRONTEND / "features/chat/api/chat-adapter.ts"
HUB_API = FRONTEND / "features/hub/inventory/api.ts"
STORE = FRONTEND / "features/chat/stores/chat-runtime-store.ts"


def _read(path: Path) -> str:
    assert path.exists(), f"missing source file: {path}"
    return path.read_text()


def _sweep() -> str:
    """The body of ``autoLoadSmallestModel``, where the policy lives."""
    return _read(ADAPTER).split("async function autoLoadSmallestModel", 1)[1]


def _run(body: str):
    """Execute *body* against the real module and parse its last stdout line."""
    if shutil.which("node") is None:
        pytest.skip("node not available")
    assert (
        INVENTORY.exists()
    ), "auto-load must decide candidates through features/chat/utils/auto-load-inventory.ts"
    probe = subprocess.run(
        ["node", "--experimental-strip-types", "--version"],
        capture_output = True,
        text = True,
        timeout = 10,
    )
    if probe.returncode != 0:
        pytest.skip("node --experimental-strip-types not available")
    with tempfile.TemporaryDirectory(prefix = "autoload_on_device_") as workdir:
        (Path(workdir) / "run.mts").write_text(
            f'import * as inventory from "{INVENTORY.as_uri()}";\n' + body
        )
        result = subprocess.run(
            ["node", "--experimental-strip-types", "--no-warnings", "run.mts"],
            cwd = workdir,
            capture_output = True,
            text = True,
            timeout = 60,
            env = dict(os.environ, NODE_NO_WARNINGS = "1"),
        )
    assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"
    return json.loads(result.stdout.strip().splitlines()[-1])


# A chat-capable GGUF folder, exactly as GET /api/hub/local reports the
# reporter's model when it lives in a registered Custom Folder.
GEMMA_ROW = {
    "id": "/mnt/ssd/models/gemma-4-26B-A4B-it-GGUF",
    "load_id": "/mnt/ssd/models/gemma-4-26B-A4B-it-GGUF",
    "display_name": "gemma-4-26B-A4B-it-GGUF",
    "path": "/mnt/ssd/models/gemma-4-26B-A4B-it-GGUF",
    "source": "custom",
    "size_bytes": 17010980576,
    "model_format": "gguf",
    "runtime": "llama_cpp",
    "partial": False,
    "capabilities": {"can_chat": True, "requires_variant": True},
}


def test_a_model_in_a_custom_folder_is_auto_loadable():
    """#7374: a GGUF outside the HF caches (Custom Folder, models dir, LM Studio)
    is a candidate. Nothing is in the caches, which is the exact state that used
    to fall through to the unsolicited download."""
    assert (
        _run(
            f"console.log(JSON.stringify(inventory.isAutoLoadableLocalRow({json.dumps(GEMMA_ROW)})));\n"
        )
        is True
    )


def test_rows_the_scanner_will_not_vouch_for_are_never_auto_loaded():
    """Auto-load picks with no confirmation, so it trusts one signal: the
    capability the scanner computed. A partial download, a folder it could not
    classify, and a LoRA adapter are all left to an explicit pick, and none of
    them needs its own filename or folder-shape rule here."""
    rows = {
        "partial": {**GEMMA_ROW, "partial": True},
        "unclassified": {
            **GEMMA_ROW,
            "runtime": "unknown",
            "capabilities": {"can_chat": False},
        },
        "adapter": {**GEMMA_ROW, "runtime": "adapter"},
    }
    verdicts = _run(
        f"const rows = {json.dumps(rows)};\n"
        "console.log(JSON.stringify(Object.fromEntries(Object.entries(rows)"
        ".map(([k, v]) => [k, inventory.isAutoLoadableLocalRow(v)]))));\n"
    )
    assert verdicts == {"partial": False, "unclassified": False, "adapter": False}


# The same folder holding one loose ``.gguf`` instead of a model directory. The
# scanner resolves the file itself, so the row is a direct-file row:
# ``requires_variant`` is false and the quant is already read off the name.
LOOSE_GGUF_ROW = {
    **GEMMA_ROW,
    "id": "/mnt/ssd/models/gemma-4-26B-A4B-it-Q4_K_M.gguf",
    "load_id": "/mnt/ssd/models/gemma-4-26B-A4B-it-Q4_K_M.gguf",
    "path": "/mnt/ssd/models/gemma-4-26B-A4B-it-Q4_K_M.gguf",
    "format_variant": "Q4_K_M",
    "capabilities": {"can_chat": True, "requires_variant": False},
}


def _sweep_calls(row: dict, variants: list) -> list:
    """Run the real fold-in and GGUF sweep from ``chat-adapter.ts`` over one
    ``GET /api/hub/local`` row, answering the variant lookup the way the backend
    answers it for that row, and report the calls the sweep made."""
    src = _read(ADAPTER)
    fold_in = src[
        src.index("    for (const row of localModels) {") : src.index("    candidateCount =")
    ]
    gguf_sweep = src[src.index("    // GGUF first:") : src.index("    // Fall back to safetensors")]
    return _run(
        f"const localModels: any[] = [{json.dumps(row)}];\n"
        f"const BACKEND_VARIANTS: any = {json.dumps({'variants': variants})};\n"
        "const ggufRepos: any[] = [];\nconst modelRepos: any[] = [];\n"
        "const MAX_AUTO_LOAD_ATTEMPTS = 3;\nlet loadAttempts = 0;\n"
        "const skippedAutoLoadCandidates = new Set<string>();\n"
        "const isAutoLoadableLocalRow = inventory.isAutoLoadableLocalRow;\n"
        "const autoLoadCandidateKey = (k: string, i: string, v?: string | null) =>"
        ' `${k}:${i}:${v ?? ""}`;\n'
        "const isAutoLoadableGgufVariant = (v: any) => !!v?.filename;\n"
        "const hasBigEndianGgufMarker = (_f: string) => false;\n"
        "const noteFailure = (_e: unknown) => {};\n"
        "const calls: any[] = [];\n"
        "const listGgufVariants = async (repoId: string) => {"
        ' calls.push({ fn: "listGgufVariants", repoId }); return BACKEND_VARIANTS; };\n'
        "const loadAutoLoadCandidate = async (c: any) => {"
        ' calls.push({ fn: "loadAutoLoadCandidate", ...c }); return true; };\n'
        "async function sweep(): Promise<any> {\n"
        f"{fold_in}\n{gguf_sweep}\n"
        "}\n"
        "sweep().then(() => console.log(JSON.stringify(calls)));\n"
    )


def test_a_loose_gguf_file_loads_without_a_directory_variant_lookup():
    """A ``.gguf`` dropped straight into the models dir, an LM Studio dir or a
    Custom Folder is indexed as the file itself. ``_resolve_gguf_dir`` refuses to
    group GGUFs in a folder carrying no model metadata, so the variant lookup
    answers ``[]`` for that path: routing it through the directory sweep would
    drop a chat-capable model. It loads by its own path, no variant, like the
    picker loads it. A model folder keeps selecting its smallest variant."""
    loose = _sweep_calls(LOOSE_GGUF_ROW, [])
    assert [c["fn"] for c in loose] == ["loadAutoLoadCandidate"], loose
    assert loose[0]["loadId"] == LOOSE_GGUF_ROW["path"]
    assert loose[0]["kind"] == "gguf"
    assert loose[0]["ggufVariant"] is None

    folder = _sweep_calls(
        GEMMA_ROW,
        [{"filename": "Q4_K_M.gguf", "quant": "Q4_K_M", "size_bytes": 1, "downloaded": True}],
    )
    assert [c["fn"] for c in folder] == ["listGgufVariants", "loadAutoLoadCandidate"], folder
    assert folder[1]["ggufVariant"] == "Q4_K_M"


def _slice(src: str, start: str, end: str) -> str:
    begin = src.index(start)
    return src[begin : src.index(end, begin)]


def _restore_remembered_model(repos: list, remembered: str):
    """Run the real remembered-model restore from ``chat-adapter.ts`` over
    ``repos``, through the real identity helpers, and report what it loaded."""
    adapter = _read(ADAPTER)
    return _run(
        # The real local-path predicate the identity helpers ask.
        _slice(_read(STORE), "export function isLocalModelPath(", "\n}\n") + "\n}\n"
        'type LastLocalModelKind = "gguf" | "model";\n'
        # Whatever identity helpers the adapter defines, however it spells them.
        + _slice(adapter, "type AutoLoadCandidate = {", "function hasBigEndianGgufMarker(")
        + f"const modelRepos: any[] = {json.dumps(repos)};\n"
        "const ggufRepos: any[] = [];\n"
        f"const lastLoaded: any = {{ id: {json.dumps(remembered)},"
        ' kind: "model", ggufVariant: null };\n'
        "const loaded: any[] = [];\n"
        "const store = { params: { maxSeqLength: 2048 } };\n"
        "const toast: any = Object.assign(() => {},"
        " { dismiss() {}, success() {}, error() {} });\n"
        'const toastId = "t";\n'
        "const skippedAutoLoadCandidates = new Set<string>();\n"
        "const noteFailure = (_e: unknown) => {};\n"
        "const listGgufVariants = async () => ({ variants: [] });\n"
        "const isAutoLoadableGgufVariant = () => true;\n"
        "const loadAutoLoadCandidate = async (c: any) => { loaded.push(c); return true; };\n"
        "async function restore(): Promise<any> {\n"
        + _slice(adapter, "    if (lastLoaded) {", "    // GGUF first:")
        + "\n}\n"
        "restore().then(() => console.log(JSON.stringify({\n"
        "  loaded: loaded[0]?.loadId ?? null,\n"
        '  keys: modelRepos.map((r: any) => autoLoadCandidateKey("model", r.repo_id)),\n'
        "})));\n"
    )


def test_two_local_models_differing_only_in_case_are_two_identities():
    """An on-device row is identified by its absolute path, and a case-sensitive
    filesystem does not fold case. The remembered/skip identity folded it,
    because it only ever held Hub repo ids before the inventory rows arrived, so
    ``/models/Foo`` and ``/models/foo`` shared one: the restore loaded whichever
    row came first, and a failure for one skipped the other."""
    lower = {"repo_id": "/models/foo", "load_id": "/models/foo", "size_bytes": 10}
    upper = {"repo_id": "/models/Foo", "load_id": "/models/Foo", "size_bytes": 20}
    # The colliding row is listed first, so a folded identity restores it.
    result = _restore_remembered_model([lower, upper], upper["repo_id"])
    assert result["loaded"] == upper["repo_id"], result
    assert len(set(result["keys"])) == 2, result["keys"]

    # A Hub repo id still folds: the cache resolves its real case for us.
    unsloth = {"repo_id": "unsloth/Qwen3-4B", "load_id": "unsloth/Qwen3-4B", "size_bytes": 10}
    folded = _restore_remembered_model([unsloth], "unsloth/qwen3-4b")
    assert folded["loaded"] == unsloth["repo_id"], folded


def test_failure_message_distinguishes_empty_unreadable_and_unloadable():
    """ "No downloaded models found" was printed for all three states, so a model
    that was found and then rejected read as a model Studio could not see. Each
    state now says what happened, and a load failure carries its real reason (an
    out-of-memory refusal, for instance) instead of being swallowed."""
    outcomes = _run(
        "const at = (o) => inventory.describeAutoLoadFailure({ blockedByTrustRemoteCode: false, ...o });\n"
        "console.log(JSON.stringify({\n"
        "  empty: at({ candidateCount: 0, inventoryUnavailable: false }),\n"
        "  unreadable: at({ candidateCount: 0, inventoryUnavailable: true }),\n"
        "  unloadable: at({ candidateCount: 1, inventoryUnavailable: false,\n"
        '    lastFailureReason: "Not enough memory to load this model (needs 17.0 GB, 5.6 GB free)" }),\n'
        "}));\n"
    )
    assert outcomes["empty"]["title"] == "No models on this device"
    assert outcomes["unreadable"]["title"] == "Could not read the on-device model list"
    assert outcomes["unloadable"]["title"] == "The downloaded model could not be loaded"
    assert "Not enough memory" in outcomes["unloadable"]["description"]
    assert len({outcome["title"] for outcome in outcomes.values()}) == 3


def test_send_never_downloads_a_model_from_the_hub():
    """The reporter's first complaint: a Send started a Hugging Face download
    nobody asked for. No hard-coded Hub model may be loaded from this path."""
    sweep = _sweep()
    assert "Qwen3.5-4B-MTP-GGUF" not in sweep
    assert "Downloading a small model" not in sweep
    assert "No downloaded models found" not in _read(ADAPTER)


def test_auto_load_sweeps_the_indexed_on_device_inventory():
    """The sweep must read the inventory that covers every on-device source, not
    just the two managed-cache lists, and select on its capabilities."""
    hub_api = _read(HUB_API)
    local_fn = hub_api.split("export async function listLocalModels", 1)[1]
    assert 'authFetch("/api/hub/local"' in local_fn.split("export ", 1)[0]
    sweep = _sweep()
    assert "listOnDeviceInventory()" in sweep
    assert "isAutoLoadableLocalRow(row)" in sweep


def test_an_unreadable_inventory_is_not_reported_as_an_empty_device():
    """Each inventory call used to be wrapped in `.catch(() => [])`, so a failed
    lookup read as "nothing is downloaded" and triggered the download. The
    failure must reach the message instead."""
    sweep = _sweep()
    assert "catch(() => [])" not in sweep
    assert "inventoryUnavailable = true" in sweep
    assert "describeAutoLoadFailure(" in sweep


def test_a_failed_load_attempt_keeps_its_reason():
    """Every failure in the sweep, including the remembered-model shortcut that
    runs before the loops, must hand its error to the message. A bare `catch`
    drops the backend's explanation -- an out-of-memory refusal, say -- and the
    toast falls back to the generic "pick one yourself" line."""
    sweep = _sweep().split("export function createOpenAIStreamAdapter")[0]
    assert "} catch {" not in sweep
    for handler in sweep.split("catch (error) {")[1:]:
        assert "noteFailure(error);" in handler.split("}")[0]
