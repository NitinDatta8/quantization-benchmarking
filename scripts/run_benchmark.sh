#!/usr/bin/env bash
# Quantize (if needed) then benchmark for a given GPU type.
#
# Usage: bash scripts/run_benchmark.sh [GPU_TYPE] [SKIP_QUANT] [METHOD]
#   GPU_TYPE:   RTX_A5000 | A100_SXM | L4 (default: A100_SXM)
#   SKIP_QUANT: true to skip quantization (default: false)
#   METHOD:     run only this method (default: all for GPU type)

set -euo pipefail

GPU_TYPE="${1:-A100_SXM}"
SKIP_QUANT="${2:-false}"
METHOD_FILTER="${3:-}"
REPO_DIR="/workspace/quantization-benchmarking"
BASE_MODEL="/workspace/models/base"
CONFIG="scripts/benchmark_config.yaml"

source /workspace/venv/bin/activate
cd "$REPO_DIR"

# method:quant_script:output_path (none = no quant step needed)
case "$GPU_TYPE" in
  RTX_A5000) ENTRIES=("bitsandbytes_nf4:none:none") ;;
  A100_SXM)  ENTRIES=(
               "baseline_fp16:none:$BASE_MODEL"
               "bitsandbytes_nf4:none:none"
               "awq_int4:quantize/awq_quantize.py:/workspace/models/awq_int4"
               "gptq_int4_marlin:quantize/gptq.py:/workspace/models/gptq_int4_marlin"
               "awq_int4_marlin:none:/workspace/models/awq_int4"
               "gptq_int4:none:/workspace/models/gptq_int4_marlin"
             ) ;;
  L4)        ENTRIES=(
               "baseline_fp16:none:$BASE_MODEL"
               "bitsandbytes_nf4:none:none"
               "awq_int4:none:/workspace/models/awq_int4"
               "gptq_int4_marlin:none:/workspace/models/gptq_int4_marlin"
               "fp8_w8a8:quantize/fp8.py:/workspace/models/fp8_w8a8"
             ) ;;
  RTX_5090)  ENTRIES=(
               "baseline_fp16:none:$BASE_MODEL"
               "nvfp4:quantize/nvfp4.py:/workspace/models/nvfp4"
             ) ;;
  *)         echo "Unknown GPU_TYPE: $GPU_TYPE"; exit 1 ;;
esac

for ENTRY in "${ENTRIES[@]}"; do
  METHOD=$(echo "$ENTRY" | cut -d: -f1)
  QUANT_SCRIPT=$(echo "$ENTRY" | cut -d: -f2)
  OUTPUT_PATH=$(echo "$ENTRY" | cut -d: -f3)

  [ -n "$METHOD_FILTER" ] && [ "$METHOD" != "$METHOD_FILTER" ] && continue

  echo "--- $METHOD ---"

  # Quantize if needed
  if [ "$QUANT_SCRIPT" != "none" ] && [ "$SKIP_QUANT" != "true" ]; then
    if [ -d "$OUTPUT_PATH" ] && [ "$(ls -A "$OUTPUT_PATH")" ]; then
      echo "Quantized model exists at $OUTPUT_PATH, skipping."
    else
      python "$QUANT_SCRIPT" --base_model "$BASE_MODEL" --output_path "$OUTPUT_PATH" --config "$CONFIG"
    fi
  fi

  # Benchmark
  python benchmark/runner.py --method "$METHOD" --gpu "$GPU_TYPE" --config "$CONFIG"
done

echo "Done. Results in results/"
