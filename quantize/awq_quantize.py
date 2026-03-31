"""AWQ INT4 quantization for the base FP16 model."""

import argparse
import json
import random
import time
from pathlib import Path

import yaml


def load_calibration_data(config, num_samples=128):
    with open(config["output"]["prompts_file"]) as f:
        data = json.load(f)

    prompts = []
    for p in data["prompts"]:
        if p["type"] == "single_turn":
            prompts.append(p.get("system", "") + "\n" + p.get("user", ""))
        elif p["type"] == "multi_turn":
            parts = [f"system: {p['system']}"] if "system" in p else []
            parts += [f"{m['role']}: {m['content']}" for m in p["messages"]]
            prompts.append("\n".join(parts))

    random.seed(42)
    random.shuffle(prompts)
    # pad if we have fewer prompts than requested
    while len(prompts) < num_samples:
        prompts += prompts
    return prompts[:num_samples]


def quantize_awq(base_model_path, output_path, config):
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    n_samples = config["methods"]["awq_int4"].get("calibration_samples", 128)

    quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}

    print(f"[AWQ] Loading model from {base_model_path}")
    model = AutoAWQForCausalLM.from_pretrained(base_model_path, low_cpu_mem_usage=True, use_cache=False)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    calib_data = load_calibration_data(config, n_samples)

    print(f"[AWQ] Quantizing ({n_samples} calibration samples)...")
    t0 = time.time()
    model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_data)
    print(f"[AWQ] Done in {time.time() - t0:.1f}s")

    Path(output_path).mkdir(parents=True, exist_ok=True)
    model.save_quantized(output_path, safetensors=True, shard_size="10GB")
    tokenizer.save_pretrained(output_path)

    # save metadata for traceability
    meta = {
        "method": "awq_int4",
        "base_model": base_model_path,
        "quant_config": quant_config,
        "calibration_samples": n_samples,
        "quantization_time_sec": round(time.time() - t0, 1),
    }
    json.dump(meta, open(Path(output_path) / "quant_meta.json", "w"), indent=2)


def push_to_hub(output_path, hf_repo):
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo_id=hf_repo, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=output_path, repo_id=hf_repo, repo_type="model")
    print(f"[AWQ] Pushed to https://huggingface.co/{hf_repo}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="/workspace/models/base")
    parser.add_argument("--output_path", default="/workspace/models/awq_int4")
    parser.add_argument("--config", default="scripts/benchmark_config.yaml")
    parser.add_argument("--hf_repo", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    quantize_awq(args.base_model, args.output_path, config)
    if args.hf_repo:
        push_to_hub(args.output_path, args.hf_repo)


if __name__ == "__main__":
    main()
