"""NVFP4 quantization using llm-compressor with calibration data."""

import argparse
import json
import random
import time
from pathlib import Path

import os

import yaml


def load_calibration_data(config, tokenizer, num_samples=512, seq_len=2048):
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

    return texts


def quantize_nvfp4(base_model_path, output_path, config):
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor import oneshot
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = config["methods"]["nvfp4"]
    n_samples = cfg.get("calibration_samples", 512)
    seq_len = cfg.get("calibration_seqlen", 2048)

    base_model_path = os.path.realpath(base_model_path)
    print(f"[NVFP4] Loading model from {base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, device_map="auto", torch_dtype="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build calibration dataset from custom prompts and save as local JSONL
    calib_texts = load_calibration_data(config, tokenizer, n_samples, seq_len)
    calib_file = Path(output_path).parent / "nvfp4_calib.jsonl"
    calib_file.parent.mkdir(parents=True, exist_ok=True)
    with open(calib_file, "w") as f:
        for text in calib_texts:
            json.dump({"text": text}, f)
            f.write("\n")

    recipe = QuantizationModifier(targets="Linear", scheme="NVFP4", ignore=["lm_head"])

    print(f"[NVFP4] Quantizing ({n_samples} calibration samples, seq_len={seq_len})...")
    t0 = time.time()
    oneshot(
        model=model,
        recipe=recipe,
        dataset=str(calib_file),
        max_seq_length=seq_len,
        num_calibration_samples=n_samples,
    )
    print(f"[NVFP4] Done in {time.time() - t0:.1f}s")

    Path(output_path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    meta = {
        "method": "nvfp4",
        "base_model": base_model_path,
        "scheme": "NVFP4",
        "targets": "Linear",
        "ignored_layers": ["lm_head"],
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
    print(f"[NVFP4] Pushed to https://huggingface.co/{hf_repo}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="/workspace/models/base")
    parser.add_argument("--output_path", default="/workspace/models/nvfp4")
    parser.add_argument("--config", default="scripts/benchmark_config.yaml")
    parser.add_argument("--hf_repo", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    quantize_nvfp4(args.base_model, args.output_path, config)
    if args.hf_repo:
        push_to_hub(args.output_path, args.hf_repo)


if __name__ == "__main__":
    main()
