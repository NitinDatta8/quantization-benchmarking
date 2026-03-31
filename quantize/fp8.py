"""FP8 W8A8 dynamic quantization using llm-compressor."""

import argparse
import json
import time
from pathlib import Path

import yaml


def quantize_fp8(base_model_path, output_path, config):
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor.transformers import SparseAutoModelForCausalLM, oneshot
    from transformers import AutoTokenizer

    print(f"[FP8] Loading model from {base_model_path}")
    model = SparseAutoModelForCausalLM.from_pretrained(
        base_model_path, device_map="auto", torch_dtype="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    recipe = QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])

    print("[FP8] Quantizing...")
    t0 = time.time()
    oneshot(model=model, recipe=recipe)
    print(f"[FP8] Done in {time.time() - t0:.1f}s")

    Path(output_path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    meta = {
        "method": "fp8_w8a8",
        "base_model": base_model_path,
        "scheme": "FP8_DYNAMIC",
        "targets": "Linear",
        "ignored_layers": ["lm_head"],
        "quantization_time_sec": round(time.time() - t0, 1),
    }
    json.dump(meta, open(Path(output_path) / "quant_meta.json", "w"), indent=2)


def push_to_hub(output_path, hf_repo):
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo_id=hf_repo, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=output_path, repo_id=hf_repo)
    print(f"[FP8] Pushed to https://huggingface.co/{hf_repo}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="/workspace/models/base")
    parser.add_argument("--output_path", default="/workspace/models/fp8_w8a8")
    parser.add_argument("--config", default="scripts/benchmark_config.yaml")
    parser.add_argument("--hf_repo", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    quantize_fp8(args.base_model, args.output_path, config)
    if args.hf_repo:
        push_to_hub(args.output_path, args.hf_repo)


if __name__ == "__main__":
    main()
