import os, time
from huggingface_hub import snapshot_download, hf_hub_download
tok = os.environ.get("HF_TOKEN")
# (base_repo, gguf_repo, gguf_file)  -- order: lightest gated-free first, heaviest last
JOBS = [
    ("black-forest-labs/FLUX.1-schnell","unsloth/FLUX.1-schnell-GGUF","flux1-schnell-Q4_K_M.gguf"),
    ("black-forest-labs/FLUX.2-klein-4B","unsloth/FLUX.2-klein-4B-GGUF","flux-2-klein-4b-Q4_K_M.gguf"),
    ("Tongyi-MAI/Z-Image","unsloth/Z-Image-GGUF","z-image-Q4_K_M.gguf"),
    ("black-forest-labs/FLUX.2-klein-9B","unsloth/FLUX.2-klein-9B-GGUF","flux-2-klein-9b-Q4_K_M.gguf"),
    ("black-forest-labs/FLUX.1-dev","unsloth/FLUX.1-dev-GGUF","flux1-dev-Q4_K_M.gguf"),
    ("Qwen/Qwen-Image-2512","unsloth/Qwen-Image-2512-GGUF","qwen-image-2512-Q4_K_M.gguf"),
    ("Qwen/Qwen-Image","unsloth/Qwen-Image-GGUF","qwen-image-Q4_K_M.gguf"),
    ("Tongyi-MAI/Z-Image-Turbo","unsloth/Z-Image-Turbo-GGUF","z-image-turbo-Q4_K_M.gguf"),
]
for base, ggrepo, ggfile in JOBS:
    t0=time.time()
    try:
        hf_hub_download(ggrepo, ggfile, token=tok)
        print(f"[gguf ok] {ggrepo}/{ggfile} ({time.time()-t0:.0f}s)", flush=True)
    except Exception as e:
        print(f"[gguf ERR] {ggrepo}/{ggfile}: {type(e).__name__}: {e}", flush=True)
    t0=time.time()
    try:
        # full base repo: dense transformer (for dense path) + VAE + text encoders + scheduler
        snapshot_download(base, token=tok, ignore_patterns=["*.pth","*.onnx","*.gguf"])
        print(f"[base ok] {base} ({time.time()-t0:.0f}s)", flush=True)
    except Exception as e:
        print(f"[base ERR] {base}: {type(e).__name__}: {e}", flush=True)
print("PREDOWNLOAD-DONE", flush=True)
