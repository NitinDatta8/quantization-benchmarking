"""Load quantized (or baseline) models via vLLM for benchmarking."""

from vllm import LLM, SamplingParams

def load_model(method_name, config):
    method = config["methods"][method_name]
    model_cfg = config["model"]

    if method_name == "baseline_fp16":
        model_path = model_cfg["base_local_path"]
    else:
        model_path = method.get("quantized_output_path") or model_cfg["base_local_path"]

    llm_kwargs = {
        "model": model_path,
        "max_model_len": model_cfg["context_length"],
        "trust_remote_code": True,
        "enforce_eager": model_cfg.get("enforce_eager", False),
        "dtype": model_cfg.get("dtype", "float16"),
    }

    # vLLM dispatches quantization from the flag + checkpoint metadata:
    #   FP8 - quantization="fp8", native on Ada/Hopper GPUs
    #   AWQ - quantization="awq", uses fused AWQ kernels
    #   GPTQ -quantization="marlin", uses Marlin INT4 kernels (Ampere+)
    #   BnB - quantization="bitsandbytes" + load_format, on-the-fly NF4
    quant_flag = method.get("vllm_quantization")
    if quant_flag:
        llm_kwargs["quantization"] = quant_flag

    # BnB NF4 needs both flags — load_format tells vLLM to do on-the-fly 4-bit
    if method.get("bnb_load_in_4bit"):
        llm_kwargs["load_format"] = "bitsandbytes"
        llm_kwargs["quantization"] = "bitsandbytes"

    sampling_params = SamplingParams(
        temperature=model_cfg.get("temperature", 0.0),
        max_tokens=model_cfg.get("max_new_tokens", 512),
    )

    print(f"Loading {method_name}: {model_path} "
          f"(quant={llm_kwargs.get('quantization', 'none')})")

    llm = LLM(**llm_kwargs)
    return llm, sampling_params
