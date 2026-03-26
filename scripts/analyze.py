"""
Aggregate benchmark CSVs and generate comparison charts.

Scans results/ for *_summary.csv and *_detail.csv, merges them into
combined files, then produces charts under results/charts/.

Usage:
    python scripts/analyze.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

RESULTS_DIR = Path(__file__).parent.parent / "results"
CHARTS_DIR = RESULTS_DIR / "charts"
PALETTE = sns.color_palette("tab10")


def _label(method):
    return method.replace("_", " ").title()


def _save(fig, name):
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHARTS_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}")


# -- Aggregation --------------------------------------------------------------

def aggregate():
    print(f"Scanning {RESULTS_DIR} ...")

    dfs = {}
    for kind, dedup in [("summary", ["method", "gpu", "concurrency"]),
                        ("detail", ["method", "gpu", "concurrency", "prompt_id"])]:
        files = [f for f in sorted(RESULTS_DIR.glob(f"*_{kind}.csv"))
                 if not f.name.startswith("combined_")]
        if not files:
            print(f"  No {kind} CSVs found")
            dfs[kind] = pd.DataFrame()
            continue

        frames = []
        for f in files:
            df = pd.read_csv(f)
            frames.append(df)
            print(f"  Loaded {f.name}  ({len(df)} rows)")

        combined = pd.concat(frames, ignore_index=True).drop_duplicates(subset=dedup)
        combined.to_csv(RESULTS_DIR / f"combined_{kind}.csv", index=False)
        print(f"  Written: combined_{kind}.csv  ({len(combined)} rows)")
        dfs[kind] = combined

    if dfs["summary"].empty:
        print("No summary CSVs found. Run benchmarks first.")
        sys.exit(1)

    return dfs["summary"], dfs["detail"]


# -- Charts -------------------------------------------------------------------

def _line_chart(summary, y_col, ylabel, title, filename, **fmt):
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, method in enumerate(summary["method"].unique()):
        df = summary[summary["method"] == method].sort_values("concurrency")
        ax.plot(df["concurrency"], df[y_col],
                marker="o", label=_label(method), color=PALETTE[i], linewidth=2)
    ax.set(xlabel="Concurrency", ylabel=ylabel, title=title)
    ax.set_xticks(sorted(summary["concurrency"].unique()))
    if "yformat" in fmt:
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter(fmt["yformat"]))
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, filename)


def chart_ttft(summary):
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, method in enumerate(summary["method"].unique()):
        df = summary[summary["method"] == method].sort_values("concurrency")
        ax.plot(df["concurrency"], df["ttft_mean_sec"],
                marker="o", label=_label(method), color=PALETTE[i], linewidth=2)
        if "ttft_p95_sec" in df.columns:
            ax.fill_between(df["concurrency"], df["ttft_mean_sec"], df["ttft_p95_sec"],
                            alpha=0.12, color=PALETTE[i])
    ax.set(xlabel="Concurrency", ylabel="TTFT (sec)",
           title="Time to First Token vs Concurrency\n(shaded area = p95)")
    ax.set_xticks(sorted(summary["concurrency"].unique()))
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, "ttft_vs_concurrency.png")


def chart_memory(summary):
    df = summary.sort_values("concurrency").groupby("method", sort=False).last().reset_index()
    df = df.sort_values("peak_gpu_memory_gb")

    fig, ax = plt.subplots(figsize=(8, max(3, len(df) * 0.8)))
    bars = ax.barh([_label(m) for m in df["method"]],
                   df["peak_gpu_memory_gb"], color=PALETTE[:len(df)], edgecolor="white")
    ax.bar_label(bars, fmt="%.1f GB", padding=4, fontsize=9)
    ax.set(xlabel="Peak GPU Memory (GB)", title="Peak GPU Memory by Method\n(at highest concurrency)")
    ax.grid(axis="x", alpha=0.3)
    _save(fig, "gpu_memory.png")


def chart_quality(summary):
    df = summary[summary["concurrency"] == summary["concurrency"].min()]
    df = df.sort_values("quality_pass_rate", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar([_label(m) for m in df["method"]],
                  df["quality_pass_rate"] * 100, color=PALETTE[:len(df)], edgecolor="white", width=0.5)
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    ax.set(ylabel="Quality Pass Rate (%)",
           title="Quality Pass Rate by Method\n(rule-based checks, >=80% threshold)")
    ax.set_ylim(0, 110)
    ax.axhline(80, linestyle="--", color="gray", linewidth=1, label="80% threshold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15, ha="right")
    _save(fig, "quality_pass_rate.png")


def chart_tradeoff(summary):
    max_conc = summary["concurrency"].max()
    df = summary[summary["concurrency"] == max_conc]

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (_, row) in enumerate(df.iterrows()):
        ax.scatter(row["peak_gpu_memory_gb"], row["tps_total"], s=120, color=PALETTE[i], zorder=3)
        ax.annotate(_label(row["method"]),
                    (row["peak_gpu_memory_gb"], row["tps_total"]),
                    textcoords="offset points", xytext=(8, 4), fontsize=8)
    ax.set(xlabel="Peak GPU Memory (GB)", ylabel=f"Aggregate TPS @ concurrency={max_conc}",
           title="Throughput vs Memory Tradeoff\n(ideal: upper-left = fast + small)")
    ax.grid(True, alpha=0.3)
    _save(fig, "tps_vs_memory_tradeoff.png")


def chart_quality_by_category(detail):
    if detail.empty:
        print("  Skipping quality_by_category.png (no detail CSV)")
        return

    df = detail[detail["concurrency"] == detail["concurrency"].min()]
    pivot = (df.groupby(["method", "category"])["quality_pass_rate"]
             .mean().unstack("category").fillna(0))

    methods = list(pivot.index)
    categories = list(pivot.columns)
    x = np.arange(len(categories))
    width = 0.8 / max(len(methods), 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, method in enumerate(methods):
        offset = (i - len(methods) / 2 + 0.5) * width
        vals = [pivot.loc[method, cat] * 100 for cat in categories]
        ax.bar(x + offset, vals, width * 0.9, label=_label(method),
               color=PALETTE[i], edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in categories], fontsize=9)
    ax.set(ylabel="Avg Quality Pass Rate (%)",
           title="Quality Pass Rate by Category & Method\n(concurrency=1)")
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "quality_by_category.png")


# -- Main ---------------------------------------------------------------------

def main():
    summary, detail = aggregate()
    print(f"\nGenerating charts to {CHARTS_DIR} ...")

    chart_ttft(summary)
    _line_chart(summary, "tps_total", "Aggregate Throughput (tok/sec)",
                "Total Throughput vs Concurrency", "tps_total_vs_concurrency.png")
    chart_memory(summary)
    chart_quality(summary)
    _line_chart(summary, "cost_per_1m_tokens_usd", "Cost per 1M Tokens (USD)",
                "Inference Cost vs Concurrency\n(based on RunPod hourly rates)",
                "cost_vs_concurrency.png", yformat="$%.4f")
    chart_tradeoff(summary)
    chart_quality_by_category(detail)
    print("\nDone.")


if __name__ == "__main__":
    main()
