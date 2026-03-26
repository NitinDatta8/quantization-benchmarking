#!/usr/bin/env bash
# Setup a RunPod pod for quantization benchmarking.
# Downloads base model, installs deps, runs a smoke test.
#
# Usage: bash scripts/runpod_setup.sh [GPU_TYPE] [HF_TOKEN]
#   GPU_TYPE: RTX_A5000 | A100_SXM | L4 (default: A100_SXM)

set -euo pipefail

GPU_TYPE="${1:-A100_SXM}"
HF_TOKEN="${2:-}"
BASE_MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3"
BASE_MODEL_LOCAL="/workspace/models/base"
VENV_DIR="/workspace/venv"
REPO_DIR="/workspace/jpmc-benchmark"

echo "=== RunPod Setup (GPU: $GPU_TYPE) ==="

# GPU check
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader

# System deps
apt-get update -qq
apt-get install -y -qq git wget curl build-essential python3-pip python3-venv
pip install uv --quiet # Moved to uv from pip as it takes too long to install with pip

# Venv
python3 -m venv --clear "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip wheel setuptools --quiet

# PyTorch + vLLM (pinned for vLLM 0.6.6 compat)
CONSTRAINTS="/tmp/constraints.txt"
cat > "$CONSTRAINTS" <<'EOF'
numpy<2.0.0
compressed-tensors==0.8.1
EOF

uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    -c "$CONSTRAINTS" --index-url https://download.pytorch.org/whl/cu121 --quiet

uv pip install vllm==0.6.6.post1 -r "$REPO_DIR/requirements.txt" \
    -c "$CONSTRAINTS" --extra-index-url https://download.pytorch.org/whl/cu121 --quiet

# autoawq needs torch at build time but doesn't declare it
pip install "autoawq>=0.2.5,<0.3.0" --no-build-isolation --quiet

# Pin compressed-tensors (vLLM hard dep)
uv pip install "compressed-tensors==0.8.1" --no-deps --quiet

# GPU-specific libs
case "$GPU_TYPE" in
  RTX_A5000)
    uv pip uninstall llmcompressor compressed-tensors 2>/dev/null || true
    uv pip install "compressed-tensors==0.8.1" --no-deps --quiet
    ;;
  A100_SXM)
    uv pip install auto-gptq==0.7.1 "llmcompressor>=0.4.0,<0.5.0" \
        --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu121/ \
        --no-deps --quiet
    ;;
  L4)
    uv pip install "llmcompressor>=0.4.0,<0.5.0" --no-deps --quiet
    ;;
esac

echo "Installed versions:"
uv pip list | grep -E "vllm|torch |transformers|bitsandbytes|auto.gptq|autoawq|llmcompressor"

# Download base model
export HF_HOME=/workspace/.hf_cache
if [ -n "$HF_TOKEN" ]; then
  huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
else
  echo "WARNING: No HF_TOKEN provided. Gated repos will fail."
fi

mkdir -p "$(dirname "$BASE_MODEL_LOCAL")"
python3 -c "
from huggingface_hub import snapshot_download
import os, shutil

cache_path = snapshot_download(
    repo_id='$BASE_MODEL_ID',
    ignore_patterns=['*.msgpack', '*.h5', 'original/*', 'consolidated.safetensors'],
)
target = '$BASE_MODEL_LOCAL'
if os.path.islink(target):
    os.remove(target)
elif os.path.isdir(target):
    shutil.rmtree(target)
os.symlink(cache_path, target)
print(f'Model ready: {target} -> {cache_path}')
"

# Smoke test
python3 -c "
import torch, vllm
print(f'torch={torch.__version__} vllm={vllm.__version__} cuda={torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)')
"

echo ""
echo "Setup complete. Next: bash scripts/run_benchmark.sh $GPU_TYPE"
