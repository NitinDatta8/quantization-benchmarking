# RunPod Steps

## 0. SSH into pod

```
ssh into runpod
cd /workspace
```

## 1. Clone repo

```
git clone https://github.com/NitinDatta8/quantization-benchmarking.git
cd quantization-benchmarking
```

## 2. Setup environment

```
bash /workspace/quantization-benchmarking/scripts/runpod_setup.sh A100_SXM hf_iLPRavhMQikkCmzFQBEaSnlY
```

For RTX A5000 pod instead:
```
bash /workspace/quantization-benchmarking/scripts/runpod_setup.sh RTX_A5000 hf_iLPRavhMQikk
```

## 3. Run baseline benchmark (FP16 on A100)

```
bash scripts/run_benchmark.sh A100_SXM false baseline_fp16
```

For BnB NF4 on RTX A5000:
```
bash scripts/run_benchmark.sh RTX_A5000
```

## 4. Quantize AWQ and push to HF Hub

```
cd .. 
source venv/bin/activate
huggingface-cli login --token hf_iLPRavhMQikkCmz
cd quantization-benchmarking
```

```
python quantize/awq_quantize.py --base_model /workspace/models/base --output_path /workspace/models/awq_int4 --config scripts/benchmark_config.yaml --hf_repo Nitin878/Mistral-7B-Instruct-v0.3-AWQ
```

## 5. Benchmark AWQ model

```
bash scripts/run_benchmark.sh A100_SXM false awq_int4
```

## 6. Quantize GPTQ and push to HF Hub

```
cd ..
source venv/bin/activate
cd quantization-benchmarking
```

```
python quantize/gptq.py --base_model /workspace/models/base --output_path /workspace/models/gptq_int4_marlin --config scripts/benchmark_config.yaml --hf_repo Nitin878/Mistral-7B-Instruct-v0.3-GPTQ
```

## 7. Benchmark GPTQ model

```
bash scripts/run_benchmark.sh A100_SXM false gptq_int4_marlin
```

## 8. Quantize FP8 on L4 pod

SSH into an L4 pod and set up:
```
bash /workspace/quantization-benchmarking/scripts/runpod_setup.sh L4 hf_iLPRavhMQikkCmzF
```

```
cd ..
source venv/bin/activate
cd quantization-benchmarking
```

```
python quantize/fp8.py --base_model /workspace/models/base --output_path /workspace/models/fp8_w8a8 --config scripts/benchmark_config.yaml --hf_repo Nitin878/Mistral-7B-Instruct-v0.3-FP8
```

## 9. Benchmark FP8 model

```
bash scripts/run_benchmark.sh L4 false fp8_w8a8
```

## 10. Quantize NVFP4 on RTX 5090 (Blackwell) pod

SSH into an RTX 5090 pod and set up:
```
bash /workspace/quantization-benchmarking/scripts/runpod_setup.sh RTX_5090 hf_iLPRavhMQikkCmzF
```

```
cd ..
source venv/bin/activate
cd quantization-benchmarking
```

```
python quantize/nvfp4.py --base_model /workspace/models/base --output_path /workspace/models/nvfp4 --config scripts/benchmark_config.yaml --hf_repo Nitin878/Mistral-7B-Instruct-v0.3-NVFP4
```

## 11. Benchmark NVFP4 model

```
bash scripts/run_benchmark.sh RTX_5090 false nvfp4
```

## 12. Additional RTX 5090 benchmarks (apples to apples comparison)

On the RTX 5090 pod, download pre-quantized models from HF Hub:
```
huggingface-cli download Nitin878/Mistral-7B-Instruct-v0.3-AWQ --local-dir /workspace/models/awq_int4
huggingface-cli download Nitin878/Mistral-7B-Instruct-v0.3-GPTQ --local-dir /workspace/models/gptq_int4_marlin
huggingface-cli download Nitin878/Mistral-7B-Instruct-v0.3-FP8 --local-dir /workspace/models/fp8_w8a8
```

Run other methods on RTX 5090 for comparison:
```
bash scripts/run_benchmark.sh RTX_5090 false baseline_fp16
bash scripts/run_benchmark.sh RTX_5090 false bitsandbytes_nf4
bash scripts/run_benchmark.sh RTX_5090 false awq_int4
bash scripts/run_benchmark.sh RTX_5090 false gptq_int4_marlin
bash scripts/run_benchmark.sh RTX_5090 false fp8_w8a8
```

## 13. Additional L4 benchmarks (apples-to-apples comparison)

On the L4 pod, download pre-quantized models from HF Hub:
```
huggingface-cli download Nitin878/Mistral-7B-Instruct-v0.3-AWQ --local-dir /workspace/models/awq_int4
huggingface-cli download Nitin878/Mistral-7B-Instruct-v0.3-GPTQ --local-dir /workspace/models/gptq_int4_marlin
```

Run all 4 remaining methods on L4:
```
bash scripts/run_benchmark.sh L4 false baseline_fp16
bash scripts/run_benchmark.sh L4 false bitsandbytes_nf4
bash scripts/run_benchmark.sh L4 false awq_int4
bash scripts/run_benchmark.sh L4 false gptq_int4_marlin
```

## 14. Additional A100 benchmark (BnB NF4)

On the A100 pod:
```
bash scripts/run_benchmark.sh A100_SXM false bitsandbytes_nf4
```

## 15. Kernel isolation experiments (AWQ+Marlin, GPTQ without Marlin)

On the A100 pod (models already downloaded, no re-quantization needed):
```
bash scripts/run_benchmark.sh A100_SXM true awq_int4_marlin
bash scripts/run_benchmark.sh A100_SXM true gptq_int4
```

## 16. Analyze results

```
python scripts/analyze.py
```
