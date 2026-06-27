#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0
"""HTTP feature probe for the Unsloth Studio backend (Docker image or native).

Runs OS/transport-agnostic against a Studio base URL and asserts the
user-visible features work in CPU mode:

  * auth      -- bootstrap login + forced password rotation (idempotent: also
                 works against an already-rotated install when --password is the
                 current one).
  * inference -- POST /api/inference/load a GGUF (HF repo + variant), then a
                 /v1/chat/completions turn; assert a non-empty completion.
  * rag       -- create a knowledge base, upload a doc, poll ingestion, search;
                 assert a hit. A 503 (sqlite-vec unavailable) is a documented
                 SKIP, not a failure.
  * websearch -- best-effort chat turn with the web_search tool; never fails the
                 run (DuckDuckGo reachability from CI is not guaranteed).

Writes <out-dir>/results.json, <out-dir>/RESULT.md (a markdown table), and
<out-dir>/auth.json (the working token + current password) so the Playwright
capture scripts can reuse the session without rotating again.

Stdlib + httpx only. Exit code is non-zero only when a REQUIRED feature
(auth, inference) fails; SKIP/best-effort never fail the process.
"""
from __future__ import annotations

import argparse
import json
import secrets
import sys
import time
from pathlib import Path

import httpx


class Probe:
    def __init__(self, base_url: str, out_dir: Path, timeout: float = 600.0):
        self.base = base_url.rstrip("/")
        self.out = out_dir
        self.out.mkdir(parents=True, exist_ok=True)
        self.client = httpx.Client(timeout=timeout)
        self.token: str | None = None
        self.password: str | None = None
        self.results: dict[str, dict] = {}

    # -- helpers ---------------------------------------------------------------
    def _h(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def record(self, name: str, status: str, detail: str = "") -> None:
        self.results[name] = {"status": status, "detail": detail}
        print(f"[{status:7}] {name}: {detail}", flush=True)

    def _login(self, password: str) -> str | None:
        r = self.client.post(
            f"{self.base}/api/auth/login",
            json={"username": "unsloth", "password": password},
        )
        if r.status_code != 200:
            return None
        return r.json().get("access_token")

    # -- features --------------------------------------------------------------
    def auth(self, current_password: str) -> bool:
        """Login then rotate the password (mirrors studio-inference-smoke).

        A forced change is pending on a fresh bootstrap install; rotating also
        works on an already-rotated install (current_password is the current
        one), so this is robust either way."""
        tok0 = self._login(current_password)
        if not tok0:
            self.record("auth", "FAIL", "login rejected (wrong password?)")
            return False
        self.token = tok0
        new = f"Probe-{secrets.token_urlsafe(12)}"
        r = self.client.post(
            f"{self.base}/api/auth/change-password",
            headers=self._h(),
            json={"current_password": current_password, "new_password": new},
        )
        if r.status_code != 200:
            self.record("auth", "FAIL", f"change-password {r.status_code}: {r.text[:120]}")
            return False
        tok = self._login(new)
        if not tok:
            self.record("auth", "FAIL", "re-login after rotation failed")
            return False
        self.token, self.password = tok, new
        self.record("auth", "PASS", "login + password rotation OK")
        self._save_auth()
        return True

    def _save_auth(self) -> None:
        (self.out / "auth.json").write_text(
            json.dumps({"base_url": self.base, "token": self.token, "password": self.password}),
            encoding="utf-8",
        )

    def inference(self, repo: str, variant: str) -> bool:
        body = {
            "model_path": repo,
            "gguf_variant": variant,
            "is_lora": False,
            "max_seq_length": 2048,
        }
        r = self.client.post(
            f"{self.base}/api/inference/load", headers=self._h(), json=body, timeout=1200.0
        )
        if r.status_code != 200:
            self.record("inference_load", "FAIL", f"{r.status_code}: {r.text[:160]}")
            return False
        d = r.json()
        if d.get("status") not in ("loaded", "already_loaded"):
            self.record("inference_load", "FAIL", f"status={d.get('status')} {d.get('detail')}")
            return False
        self.record(
            "inference_load",
            "PASS",
            f"{d.get('display_name')} is_gguf={d.get('is_gguf')} ctx={d.get('context_length')}",
        )
        # chat turn
        chat = self.client.post(
            f"{self.base}/v1/chat/completions",
            headers=self._h(),
            json={
                "model": "default",
                "messages": [{"role": "user", "content": "In one short sentence, what is Unsloth?"}],
                "max_tokens": 96,
                "temperature": 0.2,
            },
            timeout=300.0,
        )
        if chat.status_code != 200:
            self.record("chat", "FAIL", f"{chat.status_code}: {chat.text[:160]}")
            return False
        content = chat.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        if not content.strip():
            self.record("chat", "FAIL", "empty completion")
            return False
        (self.out / "chat_reply.txt").write_text(content, encoding="utf-8")
        self.record("chat", "PASS", f"{len(content)} chars: {content[:80]!r}")
        return True

    def rag(self) -> bool:
        kb = self.client.post(
            f"{self.base}/api/rag/knowledge-bases", headers=self._h(), json={"name": "probe-kb"}
        )
        if kb.status_code == 503:
            self.record("rag", "SKIP", "RAG unavailable (sqlite-vec not loadable) -> 503")
            return True
        if kb.status_code not in (200, 201):
            self.record("rag", "FAIL", f"kb create {kb.status_code}: {kb.text[:120]}")
            return False
        kbj = kb.json()
        kbid = kbj.get("id") or kbj.get("kb_id") or (kbj.get("knowledge_base") or {}).get("id")
        doc = self.out / "rag_doc.md"
        doc.write_text(
            "Unsloth makes LLM finetuning 2x faster and uses 70 percent less VRAM. "
            "The Unsloth mascot is a sloth named Sloth.\n",
            encoding="utf-8",
        )
        up = self.client.post(
            f"{self.base}/api/rag/knowledge-bases/{kbid}/documents",
            headers=self._h(),
            files={"file": ("rag_doc.md", doc.read_bytes(), "text/markdown")},
        )
        if up.status_code not in (200, 201):
            self.record("rag", "FAIL", f"upload {up.status_code}: {up.text[:120]}")
            return False
        job = up.json().get("jobId") or up.json().get("job_id")
        for _ in range(40):
            js = self.client.get(f"{self.base}/api/rag/jobs/{job}", headers=self._h())
            st = (js.json().get("status") or js.json().get("state") or "").lower()
            if st in ("completed", "complete", "succeeded", "done", "ready"):
                break
            if st in ("failed", "error"):
                self.record("rag", "FAIL", f"ingestion {st}")
                return False
            time.sleep(3)
        sr = self.client.post(
            f"{self.base}/api/rag/search",
            headers=self._h(),
            json={"query": "What is the Unsloth mascot named?", "kb_id": kbid, "top_k": 3},
        )
        if sr.status_code != 200:
            self.record("rag", "FAIL", f"search {sr.status_code}: {sr.text[:120]}")
            return False
        res = sr.json().get("results", [])
        if not res:
            self.record("rag", "FAIL", "search returned no results")
            return False
        self.record("rag", "PASS", f"{len(res)} hit(s): {res[0].get('text','')[:80]!r}")
        return True

    def web_search(self) -> bool:
        """Best-effort: a chat turn with the web_search tool enabled. Never fails."""
        try:
            tools = [{
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
                },
            }]
            r = self.client.post(
                f"{self.base}/v1/chat/completions",
                headers=self._h(),
                json={
                    "model": "default",
                    "messages": [{"role": "user", "content": "Use web_search to find the Unsloth GitHub URL, then answer."}],
                    "tools": tools,
                    "tool_choice": "auto",
                    "max_tokens": 200,
                },
                timeout=180.0,
            )
            if r.status_code != 200:
                self.record("web_search", "SKIP", f"{r.status_code}: tools path not exercised")
                return True
            j = r.json().get("choices", [{}])[0].get("message", {})
            called = bool(j.get("tool_calls")) or "unsloth" in (j.get("content") or "").lower()
            self.record("web_search", "PASS" if called else "SKIP",
                        "tool call/result observed" if called else "no tool call (small model / network)")
        except Exception as e:
            self.record("web_search", "SKIP", f"best-effort error: {e!r}"[:120])
        return True

    # -- output ----------------------------------------------------------------
    def write_summary(self, label: str) -> int:
        (self.out / "results.json").write_text(json.dumps(self.results, indent=2), encoding="utf-8")
        lines = [f"## Feature probe -- {label}", "", "| Feature | Status | Detail |", "|---|---|---|"]
        for k, v in self.results.items():
            detail = v["detail"].replace("|", r"\|")[:140]
            lines.append(f"| {k} | {v['status']} | {detail} |")
        (self.out / "RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        # required features
        required = ("auth", "inference_load", "chat")
        failed = [k for k in required if self.results.get(k, {}).get("status") == "FAIL"]
        return 1 if failed else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:18000")
    ap.add_argument("--password", help="bootstrap (or current) Studio password")
    ap.add_argument("--password-file", help="file containing the password")
    ap.add_argument("--model-repo", default="unsloth/gemma-4-E4B-it-GGUF")
    ap.add_argument("--gguf-variant", default="UD-Q4_K_XL")
    ap.add_argument("--out-dir", default="probe_out")
    ap.add_argument("--label", default="studio")
    ap.add_argument("--skip-rag", action="store_true")
    ap.add_argument("--skip-websearch", action="store_true")
    args = ap.parse_args(argv)

    pw = args.password
    if args.password_file:
        pw = Path(args.password_file).read_text(encoding="utf-8").strip()
    if not pw:
        print("ERROR: --password or --password-file required", file=sys.stderr)
        return 2

    p = Probe(args.base_url, Path(args.out_dir))
    if not p.auth(pw):
        p.write_summary(args.label)
        return 1
    p.inference(args.model_repo, args.gguf_variant)
    if not args.skip_rag:
        p.rag()
    if not args.skip_websearch:
        p.web_search()
    return p.write_summary(args.label)


if __name__ == "__main__":
    sys.exit(main())
