"""
Benchmark runner. Loads a model, warms up, runs prompts at each concurrency
level, captures TTFT/TPS/memory/quality, writes CSVs.

Usage:
    python benchmark/runner.py --method awq_int4 --gpu A100_SXM
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.metrics import (
    BenchmarkResult, RequestMetrics,
    get_gpu_memory_bytes, reset_gpu_memory_stats,
)
from benchmark.model_loader import load_model
from benchmark.quality_eval import evaluate


def _build_chat_messages(prompt):
    msgs = []
    if prompt.get("system"):
        msgs.append({"role": "system", "content": prompt["system"]})
    if prompt["type"] == "multi_turn":
        msgs.extend(prompt["messages"])
    else:
        msgs.append({"role": "user", "content": prompt["user"]})
    return msgs


def _format_prompt(prompt, tokenizer):
    msgs = _build_chat_messages(prompt)
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    # fallback for tokenizers without chat templates
    parts = [f"<|{m['role']}|>\n{m['content']}" for m in msgs]
    parts.append("<|assistant|>\n")
    return "\n".join(parts)


def _warmup(llm, sampling_params, formatted_prompts, n):
    print(f"Warmup: {min(n, len(formatted_prompts))} requests...")
    llm.generate(formatted_prompts[:n], sampling_params)


def _run_batch(llm, sampling_params, prompts, formatted_prompts, concurrency):
    all_metrics = []
    total_wall = 0.0
    total_tokens = 0

    for start in range(0, len(prompts), concurrency):
        batch_prompts = prompts[start:start + concurrency]
        batch_formatted = formatted_prompts[start:start + concurrency]

        t0 = time.perf_counter()
        outputs = llm.generate(batch_formatted, sampling_params)
        batch_time = time.perf_counter() - t0

        total_wall += batch_time
        total_tokens += sum(len(o.outputs[0].token_ids) for o in outputs)

        for i, output in enumerate(outputs):
            prompt = batch_prompts[i]
            text = output.outputs[0].text
            n_tokens = len(output.outputs[0].token_ids)

            # TTFT from vLLM internals if available, NaN otherwise
            metrics_obj = getattr(output, "metrics", None)
            if metrics_obj and getattr(metrics_obj, "first_token_time", None):
                ttft = metrics_obj.first_token_time - metrics_obj.arrival_time
            else:
                ttft = float("nan")

            eq = evaluate(text, prompt["quality_checks"])

            all_metrics.append(RequestMetrics(
                prompt_id=prompt["id"],
                category=prompt["category"],
                ttft_sec=ttft,
                total_time_sec=batch_time,
                output_tokens=n_tokens,
                quality_passed=eq.passed,
                quality_pass_rate=eq.pass_rate,
            ))

    return all_metrics, total_wall, total_tokens


# CSV column definitions
DETAIL_COLS = [
    "method", "gpu", "concurrency", "prompt_id", "category",
    "ttft_sec", "total_time_sec", "output_tokens", "tps",
    "quality_passed", "quality_pass_rate",
]
SUMMARY_COLS = [
    "method", "gpu", "concurrency", "num_requests",
    "ttft_mean_sec", "ttft_p50_sec", "ttft_p95_sec",
    "tps_mean", "tps_total",
    "peak_gpu_memory_gb", "model_memory_gb", "kv_cache_memory_gb",
    "quality_pass_rate", "cost_per_1m_tokens_usd",
]


def _write_detail_csv(path, results):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=DETAIL_COLS)
        w.writeheader()
        for result in results:
            for req in result.requests:
                w.writerow({
                    "method": result.method, "gpu": result.gpu,
                    "concurrency": result.concurrency,
                    "prompt_id": req.prompt_id, "category": req.category,
                    "ttft_sec": round(req.ttft_sec, 6) if req.has_ttft else "",
                    "total_time_sec": round(req.total_time_sec, 6),
                    "output_tokens": req.output_tokens,
                    "tps": round(req.tps, 2),
                    "quality_passed": req.quality_passed,
                    "quality_pass_rate": round(req.quality_pass_rate, 4),
                })


def _write_summary_csv(path, results):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLS)
        w.writeheader()
        for result in results:
            w.writerow(result.summary_row())


def run_benchmark(method_name, gpu, config):
    bench_cfg = config["benchmark"]
    output_cfg = config["output"]
    concurrency_levels = bench_cfg["concurrency_levels"]

    prompts_path = PROJECT_ROOT / output_cfg["prompts_file"]
    with open(prompts_path) as f:
        prompts = json.load(f)["prompts"]
    print(f"Loaded {len(prompts)} prompts")

    llm, sampling_params = load_model(method_name, config)

    # Memory right after load = weights + CUDA context (no KV cache yet)
    model_memory_bytes = get_gpu_memory_bytes()
    print(f"Model memory: {model_memory_bytes / (1024**3):.2f} GB")

    tokenizer = llm.get_tokenizer()
    formatted = [_format_prompt(p, tokenizer) for p in prompts]

    _warmup(llm, sampling_params, formatted, bench_cfg.get("warmup_requests", 5))

    all_results = []
    for conc in concurrency_levels:
        print(f"\n--- Concurrency {conc} ---")
        reset_gpu_memory_stats()

        metrics, wall_time, tokens = _run_batch(
            llm, sampling_params, prompts, formatted, conc
        )

        # Peak memory after inference includes KV cache
        peak_mem = max(model_memory_bytes, get_gpu_memory_bytes())

        result = BenchmarkResult.from_requests(
            method=method_name, gpu=gpu, concurrency=conc,
            requests=metrics, peak_memory_bytes=peak_mem,
            model_memory_bytes=model_memory_bytes,
            total_wall_time_sec=wall_time, total_output_tokens=tokens,
        )

        s = result.summary_row()
        print(f"TTFT mean={s['ttft_mean_sec']}s p95={s['ttft_p95_sec']}s | "
              f"TPS total={s['tps_total']} | "
              f"Memory peak={s['peak_gpu_memory_gb']}GB model={s['model_memory_gb']}GB kv={s['kv_cache_memory_gb']}GB | "
              f"Quality={s['quality_pass_rate']} | Cost=${s['cost_per_1m_tokens_usd']}/1M")

        all_results.append(result)

    results_dir = PROJECT_ROOT / output_cfg["results_dir"]
    detail_path = results_dir / f"{method_name}_detail.csv"
    summary_path = results_dir / f"{method_name}_summary.csv"
    _write_detail_csv(detail_path, all_results)
    _write_summary_csv(summary_path, all_results)
    print(f"\nWritten: {detail_path}\n         {summary_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--config", default="scripts/benchmark_config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.method not in config["methods"]:
        print(f"Unknown method '{args.method}'. Valid: {', '.join(config['methods'])}")
        sys.exit(1)

    run_benchmark(args.method, args.gpu, config)


if __name__ == "__main__":
    main()
