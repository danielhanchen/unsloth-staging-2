"""Build a fake multi-drive tree so the Windows drive-hopping folder browser can
be exercised on a Linux CI runner, and emit the spoof env vars to $GITHUB_ENV.

Layout ($ROOT is realpath'd; dirs literally named "C:", "D:", "E:"):

    $ROOT/C:/Windows/                 <- denied system dir (hidden in browser)
    $ROOT/C:/Program Files/           <- denied system dir (hidden in browser)
    $ROOT/C:/Models/qwen3-8b-gguf/    <- a model dir (shown, has_models badge)
    $ROOT/D:/Models/llama-3-8b-gguf/
    $ROOT/E:/Models/

UNSLOTH_SPOOF_DRIVES -> $ROOT (windows_drive_roots lists its children as drives)
UNSLOTH_SPOOF_DENIED -> ";"-joined realpaths of every <drive>/Windows and
                        <drive>/Program Files (the browse denylist hides these).
                        ";" (not os.pathsep) because the paths contain ":".
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

DRIVES = ("C:", "D:", "E:")
DENIED_SUBDIRS = ("Windows", "Program Files")
MODELS = {
    "C:": ["qwen3-8b-gguf"],
    "D:": ["llama-3-8b-gguf"],
    "E:": [],
}


def main() -> int:
    root = Path(os.path.realpath(sys.argv[1])) if len(sys.argv) > 1 else Path.cwd() / "spoofdrives"
    root.mkdir(parents=True, exist_ok=True)
    root = Path(os.path.realpath(root))

    denied: list[str] = []
    for drive in DRIVES:
        base = root / drive
        for sub in DENIED_SUBDIRS:
            d = base / sub
            d.mkdir(parents=True, exist_ok=True)
            # A file inside so the dir is non-empty and clearly "system-like".
            (d / "system.dll").write_text("stub", encoding="utf-8")
            denied.append(os.path.realpath(d))
        models_root = base / "Models"
        models_root.mkdir(parents=True, exist_ok=True)
        for m in MODELS[drive]:
            md = models_root / m
            md.mkdir(parents=True, exist_ok=True)
            # A .gguf so _looks_like_model_dir() flags it (has_models badge).
            (md / f"{m}.Q4_K_M.gguf").write_text("GGUF", encoding="utf-8")

    drives_env = os.path.realpath(root)
    denied_env = ";".join(denied)

    gh_env = os.environ.get("GITHUB_ENV")
    if gh_env:
        with open(gh_env, "a", encoding="utf-8") as fh:
            fh.write(f"UNSLOTH_SPOOF_DRIVES={drives_env}\n")
            fh.write(f"UNSLOTH_SPOOF_DENIED={denied_env}\n")

    print("UNSLOTH_SPOOF_DRIVES=", drives_env)
    print("UNSLOTH_SPOOF_DENIED=", denied_env)
    print("tree:")
    for p in sorted(root.rglob("*")):
        if p.is_dir():
            print("  ", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
