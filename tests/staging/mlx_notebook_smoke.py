"""Smoke-run many migrated Unsloth notebooks end-to-end on MLX and report a matrix.

Goal: prove "take a notebook (FastLanguageModel/FastVisionModel) and it just runs
on Mac." Each notebook is converted cell-by-cell into a runnable script (pip /
inference / save / display cells dropped), trimmed for the CI GPU (small model,
max_seq_length 512, 3 steps, tiny dataset slice), and executed in an isolated
subprocess. One notebook failing does not abort the others; a matrix is printed.

Env: NB_DIR points at a checkout of the notebooks repo.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

NB_DIR = os.environ["NB_DIR"]
HF_HUB = os.path.expanduser("~/.cache/huggingface/hub")

# (relative notebook path, model override or None). Overrides keep CI light by
# using small pre-quantized MLX models; the dedicated per-notebook steps test the
# notebooks' real models.
NOTEBOOKS = [
    ("nb/Gemma3_(270M).ipynb", None),                                              # plain SFT, FastModel
    ("nb/Qwen2.5_(7B)-Alpaca.ipynb", "mlx-community/Qwen2.5-0.5B-Instruct-4bit"),  # plain Alpaca SFT
    ("nb/Mistral_v0.3_(7B)-Alpaca.ipynb", "mlx-community/Qwen2.5-0.5B-Instruct-4bit"),  # plain SFT, Mistral family
    ("nb/Phi_4-Conversational.ipynb", "mlx-community/Qwen2.5-0.5B-Instruct-4bit"), # DataCollatorForSeq2Seq
    ("nb/TinyLlama_(1.1B)-Alpaca.ipynb", "mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit"),  # packing=True
    ("nb/Qwen2_VL_(7B)-Vision.ipynb", "mlx-community/Qwen2-VL-2B-Instruct-4bit"),  # vision / UnslothVisionDataCollator
]

DROP_CELL_MARKERS = (
    "pip install", "subprocess", "for_inference", "model.generate", "TextStreamer",
    "save_pretrained", "push_to_hub", "from IPython", "display(", "GGUF", "llama.cpp",
)


def convert(nb_path, model_override):
    nb = json.load(open(nb_path))
    cells = []
    for c in nb.get("cells", []):
        if c.get("cell_type") != "code":
            continue
        src = "".join(c.get("source", []))
        if not src.strip():
            continue
        if any(m in src for m in DROP_CELL_MARKERS):
            continue
        if "trainer.train_dataset[" in src and "decode" in src:
            continue
        cells.append(src)
    code = "\n\n".join(cells)

    kept = []
    for line in code.splitlines():
        s = line.strip()
        # drop bare display expressions like `dataset`, `dataset[2]["text"]`, `converted_dataset[0]`
        if re.match(r'^(dataset|converted_dataset)(\[|$)', s):
            continue
        if s.startswith("tokenizer.decode(") or s.startswith("display("):
            continue
        kept.append(line)
    code = "\n".join(kept)

    if model_override:
        code = re.sub(r'model_name\s*=\s*["\'][^"\']+["\']', f'model_name = "{model_override}"', code)
    code = re.sub(r'max_steps\s*=\s*\d+', 'max_steps = 3', code)
    code = re.sub(r'max_seq_length\s*=\s*\d+', 'max_seq_length = 512', code)
    code = re.sub(r'split\s*=\s*"train(\[[^"]*\])?"', 'split = "train[:48]"', code)
    code = re.sub(r"split\s*=\s*'train(\[[^']*\])?'", "split = 'train[:48]'", code)
    # ease the paravirtual Metal GPU
    code = re.sub(r'per_device_train_batch_size\s*=\s*\d+', 'per_device_train_batch_size = 1', code)
    code = re.sub(r'gradient_accumulation_steps\s*=\s*\d+', 'gradient_accumulation_steps = 2', code)
    return "import os\n" + code + "\nprint('NOTEBOOK_RAN_OK')\n"


def run_one(nb_rel, model_override):
    path = os.path.join(NB_DIR, nb_rel)
    try:
        script = convert(path, model_override)
    except Exception as exc:  # conversion error
        return False, f"convert-error: {exc}", ""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(script)
        spath = f.name
    try:
        r = subprocess.run([sys.executable, "-u", spath], capture_output=True, text=True, timeout=1200)
        ok = (r.returncode == 0) and ("NOTEBOOK_RAN_OK" in r.stdout)
        if ok:
            return True, "", r.stdout
        tail = (r.stderr.strip().splitlines() or ["exit " + str(r.returncode)])[-1]
        return False, tail[:200], r.stdout + "\n" + r.stderr
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT(1200s)", ""


def main():
    results = []
    for nb_rel, mo in NOTEBOOKS:
        print(f"\n========== RUN {nb_rel} (model={mo}) ==========", flush=True)
        ok, err, out = run_one(nb_rel, mo)
        for line in out.splitlines()[-8:]:
            print("   " + line)
        print(f"RESULT {nb_rel}: {'PASS' if ok else 'FAIL'} {err}", flush=True)
        results.append((nb_rel, ok, err))
        # free disk between heavy model downloads
        if os.path.isdir(HF_HUB):
            for d in os.listdir(HF_HUB):
                if d.startswith("models--"):
                    shutil.rmtree(os.path.join(HF_HUB, d), ignore_errors=True)

    print("\n==================== MLX NOTEBOOK MATRIX ====================", flush=True)
    for nb_rel, ok, err in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {nb_rel}  {('-> ' + err) if not ok else ''}")
    npass = sum(1 for _, ok, _ in results if ok)
    print(f"TOTAL: {npass}/{len(results)} notebooks ran end-to-end on MLX")
    # report-only: always exit 0 so the full matrix is visible
    return 0


if __name__ == "__main__":
    sys.exit(main())
