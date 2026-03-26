"""Metric capture and aggregation for benchmark runs."""

import math
import statistics
import time
from dataclasses import dataclass, field

import numpy as np
import torch

# RunPod hourly rates (USD)
RUNPOD_HOURLY_RATES = {
    "RTX_A5000": 0.27,
    "A100_SXM": 1.49,
    "L4": 0.39,
}


def cost_per_million_tokens(gpu, tps_total):
    # $/hr / (tokens/sec * 3600 sec/hr) = $/token, then scale to 1M
    rate = RUNPOD_HOURLY_RATES.get(gpu, 0.0)
    if rate <= 0 or tps_total <= 0:
        return 0.0
    return (rate / (tps_total * 3600)) * 1_000_000


def get_gpu_memory_bytes():
    # mem_get_info gives (free, total) — we want used. This captures vLLM's
    # internal pool which torch.cuda.max_memory_allocated() misses.
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        return total - free
    return 0


def reset_gpu_memory_stats():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


@dataclass
class RequestMetrics:
    prompt_id: str
    category: str
    ttft_sec: float
    total_time_sec: float
    output_tokens: int
    quality_passed: bool
    quality_pass_rate: float

    @property
    def has_ttft(self):
        return not math.isnan(self.ttft_sec)

    @property
    def tps(self):
        # per-request throughput: how fast this single request generated tokens
        return self.output_tokens / self.total_time_sec if self.total_time_sec > 0 else 0.0


@dataclass
class BenchmarkResult:
    method: str
    gpu: str
    concurrency: int
    num_requests: int
    ttft_mean: float
    ttft_p50: float
    ttft_p95: float
    tps_mean: float
    tps_total: float
    # Memory breakdown: peak = model weights + KV cache + overhead
    peak_gpu_memory_gb: float
    model_memory_gb: float       # weights + CUDA context (measured right after load)
    kv_cache_memory_gb: float    # peak - model (KV cache + activations at runtime)
    quality_pass_rate: float
    cost_per_1m_tokens: float = 0.0
    requests: list[RequestMetrics] = field(default_factory=list)

    @staticmethod
    def from_requests(method, gpu, concurrency, requests, peak_memory_bytes,
                      model_memory_bytes, total_wall_time_sec, total_output_tokens):
        # TTFT percentiles — p50 for typical latency, p95 for tail latency
        ttfts = np.array([r.ttft_sec for r in requests if r.has_ttft])
        if len(ttfts):
            ttft_mean, ttft_p50, ttft_p95 = float(ttfts.mean()), float(np.percentile(ttfts, 50)), float(np.percentile(ttfts, 95))
        else:
            ttft_mean = ttft_p50 = ttft_p95 = 0.0

        # tps_total = system-level throughput (all tokens / wall clock), not per-request avg
        tps_total = total_output_tokens / total_wall_time_sec if total_wall_time_sec > 0 else 0.0
        tps_vals = [r.tps for r in requests]
        quality_passes = [r.quality_passed for r in requests]

        return BenchmarkResult(
            method=method, gpu=gpu, concurrency=concurrency,
            num_requests=len(requests),
            ttft_mean=ttft_mean, ttft_p50=ttft_p50, ttft_p95=ttft_p95,
            tps_mean=statistics.mean(tps_vals) if tps_vals else 0.0,
            tps_total=tps_total,
            peak_gpu_memory_gb=peak_memory_bytes / (1024 ** 3),
            model_memory_gb=model_memory_bytes / (1024 ** 3),
            kv_cache_memory_gb=max(0, peak_memory_bytes - model_memory_bytes) / (1024 ** 3),
            quality_pass_rate=sum(quality_passes) / len(quality_passes) if quality_passes else 0.0,
            cost_per_1m_tokens=cost_per_million_tokens(gpu, tps_total),
            requests=requests,
        )

    def summary_row(self):
        return {
            "method": self.method, "gpu": self.gpu,
            "concurrency": self.concurrency, "num_requests": self.num_requests,
            "ttft_mean_sec": round(self.ttft_mean, 4),
            "ttft_p50_sec": round(self.ttft_p50, 4),
            "ttft_p95_sec": round(self.ttft_p95, 4),
            "tps_mean": round(self.tps_mean, 2),
            "tps_total": round(self.tps_total, 2),
            "peak_gpu_memory_gb": round(self.peak_gpu_memory_gb, 2),
            "model_memory_gb": round(self.model_memory_gb, 2),
            "kv_cache_memory_gb": round(self.kv_cache_memory_gb, 2),
            "quality_pass_rate": round(self.quality_pass_rate, 4),
            "cost_per_1m_tokens_usd": round(self.cost_per_1m_tokens, 4),
        }


class TTFTTimer:
    """Records gap between request submission and first token back.
    Uses perf_counter for sub-ms precision. first_token() only latches once
    so subsequent tokens don't overwrite the measurement."""

    def __init__(self):
        self._start = None
        self._first_token = None

    def start(self):
        self._start = time.perf_counter()

    def first_token(self):
        if self._first_token is None:
            self._first_token = time.perf_counter()

    @property
    def ttft(self):
        if self._start and self._first_token:
            return self._first_token - self._start
        return 0.0
