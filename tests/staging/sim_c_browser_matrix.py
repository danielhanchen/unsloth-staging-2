#!/usr/bin/env python3
"""C: browser-engine matrix for the frontend half of unslothai/unsloth#7917.

The change under test is that studio/frontend/.../api/mappers.ts stops sending
`max_grad_norm`. Omission is the entire fix, so the assertion is that the key is
absent from the payload object AND from its JSON serialization, evaluated inside a
real browser rather than in Node.

Engines: chromium (covers Chrome and Edge, which both ship Blink/V8), firefox (Gecko),
webkit (Safari). The bundle under test is the REAL mapper compiled by the frontend's
own Vite config, not a transcription.

Edge cases exercised on top of the happy path, because JSON.stringify and property
enumeration are where engines historically differ:
  - the key must be genuinely absent, not present-and-undefined
  - JSON.stringify must not emit it
  - it must not arrive via the prototype chain
  - structuredClone / fetch-body round-trips must not resurrect it
"""

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUNDLE = HERE / "browser" / "dist" / "sim-mapper.js"
ENGINES = os.environ.get("SIM_ENGINES", "chromium,firefox,webkit").split(",")

if not BUNDLE.exists():
    sys.exit(f"FATAL: {BUNDLE} not built")

CONFIG = {
    "currentStep": 5, "modelType": "text", "selectedModel": "unsloth/gemma-3-270m-it",
    "projectName": "sim", "trainingMethod": "lora", "hfToken": "",
    "datasetSource": "huggingface", "datasetFormat": "auto", "dataset": "unsloth/test",
    "datasetSubset": None, "datasetSplit": "train", "datasetEvalSplit": None,
    "datasetStreaming": False, "datasetManualMapping": {}, "datasetSystemPrompt": "",
    "datasetUserTemplate": "", "datasetAssistantTemplate": "", "datasetLabelMapping": {},
    "datasetAdvisorNotification": None, "datasetSliceStart": None, "datasetSliceEnd": None,
    "uploadedFile": None, "uploadedEvalFile": None, "epochs": 1, "contextLength": 2048,
    "learningRate": 2e-4, "embeddingLearningRate": None, "optimizerType": "adamw_8bit",
    "lrSchedulerType": "linear", "loraRank": 16, "loraAlpha": 16, "loraDropout": 0,
    "loraVariant": "lora", "batchSize": 2, "gradientAccumulation": 4, "weightDecay": 0.001,
    "warmupSteps": 5, "maxSteps": 60, "saveSteps": 100, "evalSteps": 0, "packing": False,
    "trainOnCompletions": True, "gradientCheckpointing": "unsloth", "randomSeed": 3407,
    "enableWandb": False, "wandbToken": "", "wandbProject": "", "enableTensorboard": False,
    "tensorboardDir": "", "logFrequency": 1, "isCheckingVision": False,
    "isVisionModel": False, "isEmbeddingModel": False, "isAudioModel": False,
    "isLoadingModelDefaults": False, "modelDefaultsError": None,
    "modelDefaultsAppliedFor": None, "isCheckingDataset": False, "isDatasetImage": False,
    "isDatasetAudio": False, "trustRemoteCode": False,
    "approvedRemoteCodeFingerprint": None, "finetuneVisionLayers": False,
    "finetuneLanguageLayers": True, "finetuneAttentionModules": True,
    "finetuneMLPModules": True, "targetModules": ["q_proj", "v_proj"],
    "maxPositionEmbeddings": None, "visionImageSize": None, "s3Config": None,
}

# Also exercise a VLM/embedding shape, since the mapper branches on those and a
# different branch could plausibly reintroduce the key.
VLM_CONFIG = {**CONFIG, "modelType": "vision", "isVisionModel": True,
              "isDatasetImage": True, "visionImageSize": 512}
EMB_CONFIG = {**CONFIG, "modelType": "embeddings", "isEmbeddingModel": True}
CPT_CONFIG = {**CONFIG, "trainingMethod": "cpt", "datasetFormat": "raw"}

PROBE = """
(cfg) => {
  const p = SimMapper.buildTrainingStartPayload(cfg);
  const json = JSON.stringify(p);
  return {
    hasOwn: Object.prototype.hasOwnProperty.call(p, "max_grad_norm"),
    inOperator: "max_grad_norm" in p,          // walks the prototype chain too
    inKeys: Object.keys(p).includes("max_grad_norm"),
    inJson: json.includes("max_grad_norm"),
    afterClone: Object.prototype.hasOwnProperty.call(
      structuredClone(p), "max_grad_norm"),
    afterJsonRoundTrip: Object.prototype.hasOwnProperty.call(
      JSON.parse(json), "max_grad_norm"),
    maxGradValue: p.max_grad_value,
    weightDecay: p.weight_decay,
    keyCount: Object.keys(p).length,
  };
}
"""

FAILURES = []


def check(label, got, want):
    ok = got == want
    print(f"    [{'PASS' if ok else 'FAIL'}] {label}: {got!r}")
    if not ok:
        FAILURES.append(f"{label}: got {got!r} want {want!r}")


def main():
    from playwright.sync_api import sync_playwright

    bundle_src = BUNDLE.read_text(encoding="utf-8")
    html = "<!doctype html><meta charset=utf-8><title>sim</title>"

    with sync_playwright() as pw:
        for engine in ENGINES:
            engine = engine.strip()
            if not engine:
                continue
            print(f"\n== engine: {engine} ==")
            browser = getattr(pw, engine).launch()
            try:
                page = browser.new_page()
                page.set_content(html)
                page.add_script_tag(content=bundle_src)
                ua = page.evaluate("() => navigator.userAgent")
                print(f"    UA: {ua[:96]}")
                for name, cfg in (("text", CONFIG), ("vision", VLM_CONFIG),
                                  ("embedding", EMB_CONFIG), ("cpt/raw", CPT_CONFIG)):
                    r = page.evaluate(PROBE, cfg)
                    print(f"  -- config: {name} ({r['keyCount']} payload keys) --")
                    check(f"{engine}/{name} hasOwnProperty", r["hasOwn"], False)
                    check(f"{engine}/{name} 'in' operator", r["inOperator"], False)
                    check(f"{engine}/{name} Object.keys", r["inKeys"], False)
                    check(f"{engine}/{name} absent from JSON.stringify", r["inJson"], False)
                    check(f"{engine}/{name} absent after structuredClone", r["afterClone"], False)
                    check(f"{engine}/{name} absent after JSON round-trip",
                          r["afterJsonRoundTrip"], False)
                    # Sibling contract unchanged: max_grad_value is still sent as null.
                    check(f"{engine}/{name} max_grad_value still null", r["maxGradValue"], None)
            finally:
                browser.close()

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"C FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print("   -", f)
        return 1
    print(f"C PASSED: max_grad_norm absent in {', '.join(ENGINES)} across 4 config shapes.")
    print("Chrome and Edge both ship the Chromium engine, so chromium covers both.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
