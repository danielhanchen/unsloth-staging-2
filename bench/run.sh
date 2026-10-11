set -x
mkdir -p out
python -c "import torch, diffusers; print(torch.__version__, diffusers.__version__)"
python bench/zimage_mps.py out/zimage_${LEG}.json
