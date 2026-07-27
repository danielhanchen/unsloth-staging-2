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
    assert INVENTORY.exists(), (
        "auto-load must decide candidates through features/chat/utils/auto-load-inventory.ts"
    )
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
