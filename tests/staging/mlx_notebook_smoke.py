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
# Text notebooks across diverse structures (chat templates, datasets, collators,
# packing). Small overrides keep CI light; the point is "does the notebook code
# run on MLX". Vision is validated separately (heavy for the CI paravirtual GPU).
_SMALL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
NOTEBOOKS = [
    ("nb/Gemma3_(270M).ipynb", None),                                   # plain SFT, FastModel, gemma3 arch (real)
    ("nb/Qwen2.5_(7B)-Alpaca.ipynb", _SMALL),                          # plain Alpaca SFT
    ("nb/Qwen2_(7B)-Alpaca.ipynb", _SMALL),                            # plain Alpaca SFT
    ("nb/Mistral_v0.3_(7B)-Alpaca.ipynb", _SMALL),                     # plain SFT, Mistral family
    ("nb/Gemma2_(9B)-Alpaca.ipynb", _SMALL),                           # plain SFT, gemma2 chat template
    ("nb/Qwen3_(4B)-Instruct.ipynb", _SMALL),                          # plain SFT, qwen3 structure
    ("nb/Phi_3.5_Mini-Conversational.ipynb", _SMALL),                  # conversational, get_chat_template
    ("nb/Meta-Synthetic-Data-Llama3.1_(8B).ipynb", _SMALL),           # synthetic-data SFT
    ("nb/Phi_4-Conversational.ipynb", _SMALL),                         # DataCollatorForSeq2Seq
    ("nb/Qwen2.5_Coder_(14B)-Conversational.ipynb", _SMALL),          # DataCollatorForSeq2Seq, coder
    ("nb/Llama3.2_(1B_and_3B)-Conversational.ipynb", "mlx-community/Llama-3.2-1B-Instruct-4bit"),  # seq2seq, near-real
    ("nb/TinyLlama_(1.1B)-Alpaca.ipynb", _SMALL),                      # packing=True -> soft fallback
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
        # drop Jupyter magic / shell cells (`!cmd`, `%magic`) - install/deploy/inference
        if any(ln.strip().startswith(("!", "%")) for ln in src.splitlines()):
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

    print("\n==================== MLX NOTEBOOK MATRIX ====================", flush=True)
    for nb_rel, ok, err in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {nb_rel}  {('-> ' + err) if not ok else ''}")
    npass = sum(1 for _, ok, _ in results if ok)
    print(f"TOTAL: {npass}/{len(results)} notebooks ran end-to-end on MLX")
    # report-only: always exit 0 so the full matrix is visible
    return 0


if __name__ == "__main__":
    sys.exit(main())
