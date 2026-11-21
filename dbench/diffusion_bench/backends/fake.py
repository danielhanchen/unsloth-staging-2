"""A backend with no model: deterministic noise images / clips from the seed, with an optional per-cell
perturbation. Exercises the whole driver, matrix, parity gate, scorer and reporter without a GPU (selftest.py).

Options (all optional):
  noise            gaussian perturbation added to the image (a "different quality" arm)
  step_delay_s     per denoise step (0.001); te_delay_s / vae_delay_s around it (0)
  fail_on_load     raise in load()
  exit_on_load     os._exit(3) in load(): the process dies before any record (a killed / OOM'd cell)
  emulate          "blineui" adds BlineUI-style per-request execution facts; with cache_hit true the timed requests
                   report 0 executed nodes and a cached sampler (the protocol bug the gate must catch)
  sigma_shift      perturb the dumped sigma schedule by this much (parity negative control)
  steps_offset     run this many more steps than asked (a "changed step count" negative control)
  cfg_batched      cond + uncond in one call (default true); denoiser items are counted either way
  noise_device     "cpu" (default) | "cuda": where the initial noise is said to be drawn
  dtype            component precision reported for parity ("bf16")
  downscale        render at size / k (default 1; > 1 is a canvas mismatch the parity gate must catch)
  engaged          {lever: value} reported in status()["engaged"] (levers not listed have no evidence)
  no_exec_facts    with emulate "blineui": attach no execution facts (progress feed down; the gate must flag it)
  report_precision {component: token} reported by precision() instead of dtype (a silent-fallback control)
"""

from __future__ import annotations

import hashlib
import time

from .base import Backend, Render


class FakeBackend(Backend):
    name = "fake"

    def load(self) -> dict:
        self.noise = float(self.opts.get("noise", 0.0))
        self.delay = float(self.opts.get("step_delay_s", 0.001))
        if self.opts.get("fail_on_load"):
            raise RuntimeError("fake backend asked to fail on load")
        if self.opts.get("exit_on_load"):
            import os

            os._exit(3)
        # studio_status: a Studio status to emulate (speed_mode, speed_optims, attention_backend, ...): selftests
        self.last_status = {"backend": "fake", "noise": self.noise, "engaged": dict(self.opts.get("engaged") or {}),
                            **dict(self.opts.get("studio_status") or {})}
        self._evidence: dict = {}
        self.renders = 0
        return dict(self.last_status)

    def status(self) -> dict:
        return dict(self.last_status)

    def _steps(self, steps: int) -> int:
        return int(steps) + int(self.opts.get("steps_offset") or 0)

    def render(self, row: dict, steps: int) -> Render:
        import numpy as np

        self.renders += 1
        o, c = self.opts, self.cell
        n = self._steps(steps)
        time.sleep(float(o.get("te_delay_s") or 0))
        self.mark_phase("denoise_start")
        for i in range(1, n + 1):
            time.sleep(self.delay)
            self.mark_step(i, n)
        self.mark_phase("denoise_end")
        time.sleep(float(o.get("vae_delay_s") or 0))
        cfg = float(o.get("cfg", c.get("guidance") or 1.0))
        batched = bool(o.get("cfg_batched", True))
        per = 2 if cfg > 1 else 1
        shift = float(o.get("sigma_shift") or 0.0)
        sig = [round(1.0 - i / n + (shift if 0 < i < n else 0.0), 7) for i in range(n + 1)]
        self._evidence = {"sigmas": sig, "denoiser_calls": n * (1 if batched else per), "denoiser_items": n * per,
                          "cfg_batched": batched,
                          "noise": {"device": o.get("noise_device", "cpu"), "dtype": "float32",
                                    "sha": hashlib.sha256(str(row["seed"]).encode()).hexdigest()[:16]}}
        if o.get("emulate") == "blineui" and not o.get("no_exec_facts"):
            hit = bool(o.get("cache_hit")) and self.renders > 1
            self.request_facts["bline"] = {"executed_nodes": 0 if hit else 7, "cached_nodes": ["sampler"] if hit
                                           else [], "sampler_cached": hit}
        rng = np.random.default_rng(row["seed"])
        h, w = c["height"] // int(o.get("downscale", 1)), c["width"] // int(o.get("downscale", 1))
        shape = (c.get("frames") or 1, h, w, 3)
        base = rng.random(shape)
        if self.noise:
            base = np.clip(base + np.random.default_rng(row["seed"] + 1).normal(0, self.noise, shape), 0, 1)
        arr = (base * 255).astype("uint8")
        if c["kind"] == "video":
            return Render(frames = arr, fps = c.get("fps") or 8, step_s = self.delay)
        return Render(image = arr[0], step_s = self.delay)

    def parity_evidence(self) -> dict:
        return dict(self._evidence)

    def resolved_settings(self) -> dict:
        o, c = self.opts, self.cell
        return {"steps": self._steps(int(c["steps"])), "cfg": float(o.get("cfg", c.get("guidance") or 1.0)),
                "sampler": o.get("sampler", "euler"), "scheduler": o.get("scheduler", "simple"),
                "shift": o.get("shift"), "width": c["width"], "height": c["height"], "frames": c.get("frames")}

    def precision(self) -> dict:
        d = self.opts.get("dtype", "bf16")
        return {"transformer": d, "text_encoder": d, "vae": d, **(self.opts.get("report_precision") or {})}
