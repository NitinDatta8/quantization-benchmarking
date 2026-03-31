"""Marlin-compatible GPTQ INT4 quantization."""

import argparse
import json
import random
import time
from pathlib import Path

import torch
import yaml


def load_calibration_data(config, tokenizer, num_samples=128, seq_len=2048):
    with open(config["output"]["prompts_file"]) as f:
        data = json.load(f)

    texts = []
    for p in data["prompts"]:
        if p["type"] == "single_turn":
            texts.append(p.get("system", "") + "\n" + p.get("user", ""))
        elif p["type"] == "multi_turn":
            parts = [f"system: {p['system']}"] if "system" in p else []
            parts += [f"{m['role']}: {m['content']}" for m in p["messages"]]
            texts.append("\n".join(parts))

    random.seed(42)
    random.shuffle(texts)
    while len(texts) < num_samples:
        texts += texts
    texts = texts[:num_samples]

    return [
        tokenizer(t, return_tensors="pt", max_length=seq_len, padding="max_length", truncation=True)
        for t in texts
    ]


def quantize_gptq(base_model_path, output_path, config):
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    from transformers import AutoTokenizer

    cfg = config["methods"]["gptq_int4_marlin"]
    bits = cfg.get("bits", 4)
    group_size = cfg.get("group_size", 128)
    n_samples = cfg.get("calibration_samples", 128)
    seq_len = cfg.get("calibration_seqlen", 2048)

    quant_config = BaseQuantizeConfig(
        bits=bits, group_size=group_size, damp_percent=0.01,
        desc_act=False, static_groups=False, sym=True,
        true_sequential=True, model_name_or_path=base_model_path,
        model_file_base_name="model",
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[GPTQ] Loading model from {base_model_path}")
    model = AutoGPTQForCausalLM.from_pretrained(
        base_model_path, quant_config, low_cpu_mem_usage=True, torch_dtype=torch.float16,
    )

    calib_data = load_calibration_data(config, tokenizer, n_samples, seq_len)

    print(f"[GPTQ] Quantizing ({n_samples} samples, seq_len={seq_len})...")
    t0 = time.time()
    model.quantize(calib_data, cache_examples_on_gpu=False)
    print(f"[GPTQ] Done in {time.time() - t0:.1f}s")

    Path(output_path).mkdir(parents=True, exist_ok=True)
    model.save_quantized(output_path, use_safetensors=True)
    tokenizer.save_pretrained(output_path)

    meta = {
        "method": "gptq_int4_marlin",
        "base_model": base_model_path,
        "bits": bits, "group_size": group_size,
        "desc_act": False, "sym": True,
        "calibration_samples": n_samples,
        "calibration_seqlen": seq_len,
        "quantization_time_sec": round(time.time() - t0, 1),
    }
    json.dump(meta, open(Path(output_path) / "quant_meta.json", "w"), indent=2)


def push_to_hub(output_path, hf_repo):
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo_id=hf_repo, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=output_path, repo_id=hf_repo)
    print(f"[GPTQ] Pushed to https://huggingface.co/{hf_repo}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="/workspace/models/base")
    parser.add_argument("--output_path", default="/workspace/models/gptq_int4_marlin")
    parser.add_argument("--config", default="scripts/benchmark_config.yaml")
    parser.add_argument("--hf_repo", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    quantize_gptq(args.base_model, args.output_path, config)
    if args.hf_repo:
        push_to_hub(args.output_path, args.hf_repo)


if __name__ == "__main__":
    main()
