"""Aggregate benchmark CSVs and generate comparison charts."""

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
SKIP_GPUS = {"RTX_A5000"}


def _label(method):
    return method.replace("_", " ").title()


def _gpu_label(gpu):
    return gpu.replace("_", " ")


def _save(fig, subdir, name):
    out = CHARTS_DIR / subdir
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {subdir}/{name}")


# -- Aggregation ---------------------------------------------------------------

def aggregate():
    print(f"Scanning {RESULTS_DIR} ...")
    dfs = {}
    for kind, dedup in [("summary", ["method", "gpu", "concurrency"]),
                        ("detail", ["method", "gpu", "concurrency", "prompt_id"])]:
        files = [f for f in sorted(RESULTS_DIR.glob(f"*_{kind}.csv"))
                 if not f.name.startswith("combined_")]
        if not files:
            dfs[kind] = pd.DataFrame()
            continue

        combined = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        combined = combined.drop_duplicates(subset=dedup)
        combined = combined[~combined["gpu"].isin(SKIP_GPUS)]
        combined.to_csv(RESULTS_DIR / f"combined_{kind}.csv", index=False)
        print(f"  combined_{kind}.csv  ({len(combined)} rows)")
        dfs[kind] = combined

    if dfs["summary"].empty:
        print("No summary CSVs found. Run benchmarks first.")
        sys.exit(1)
    return dfs["summary"], dfs["detail"]


# -- Chart helpers -------------------------------------------------------------

def _line(data, x, y, ylabel, title, subdir, filename, group_col="method", label_fn=_label, **fmt):
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, grp in enumerate(data[group_col].unique()):
        df = data[data[group_col] == grp].sort_values(x)
        ax.plot(df[x], df[y], marker="o", label=label_fn(grp), color=PALETTE[i], linewidth=2)
    ax.set(xlabel="Concurrency", ylabel=ylabel, title=title)
    ax.set_xticks(sorted(data[x].unique()))
    if "yformat" in fmt:
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter(fmt["yformat"]))
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, subdir, filename)


def _quality_bar(data, label_col, title, subdir, label_fn=_label):
    df = data[data["concurrency"] == data["concurrency"].min()].sort_values("quality_pass_rate", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar([label_fn(v) for v in df[label_col]],
                  df["quality_pass_rate"] * 100, color=PALETTE[:len(df)], edgecolor="white", width=0.5)
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    ax.set(ylabel="Quality Pass Rate (%)", title=title)
    ax.set_ylim(0, 110)
    ax.axhline(80, ls="--", color="gray", lw=1, label="80% threshold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15, ha="right")
    _save(fig, subdir, "quality_pass_rate.png")


def _quality_by_category(detail, title, subdir):
    if detail.empty or "category" not in detail.columns:
        return
    df = detail[detail["concurrency"] == detail["concurrency"].min()]
    pivot = df.groupby(["method", "category"])["quality_pass_rate"].mean().unstack("category").fillna(0)

    methods = list(pivot.index)
    cats = list(pivot.columns)
    x = np.arange(len(cats))
    w = 0.8 / max(len(methods), 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, m in enumerate(methods):
        offset = (i - len(methods) / 2 + 0.5) * w
        ax.bar(x + offset, [pivot.loc[m, c] * 100 for c in cats], w * 0.9,
               label=_label(m), color=PALETTE[i], edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in cats], fontsize=9)
    ax.set(ylabel="Avg Quality Pass Rate (%)", title=title)
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save(fig, subdir, "quality_by_category.png")


# -- Per-GPU charts ------------------------------------------------------------

def per_gpu_charts(summary, detail, gpu):
    subdir = gpu
    s = summary[summary["gpu"] == gpu]
    d = detail[detail["gpu"] == gpu] if not detail.empty else detail
    if s.empty:
        return

    print(f"\n--- {_gpu_label(gpu)} ({', '.join(sorted(s['method'].unique()))}) ---")

    _line(s, "concurrency", "ttft_mean_sec", "TTFT (sec)",
          f"Time to First Token — {_gpu_label(gpu)}", subdir, "ttft_vs_concurrency.png")
    _line(s, "concurrency", "tps_total", "Aggregate Throughput (tok/sec)",
          f"Total Throughput — {_gpu_label(gpu)}", subdir, "tps_total_vs_concurrency.png")
    _line(s, "concurrency", "cost_per_1m_tokens_usd", "Cost per 1M Tokens (USD)",
          f"Inference Cost — {_gpu_label(gpu)}", subdir, "cost_vs_concurrency.png", yformat="$%.4f")
    _quality_bar(s, "method", f"Quality Pass Rate — {_gpu_label(gpu)}", subdir)
    _quality_by_category(d, f"Quality by Category — {_gpu_label(gpu)}", subdir)


# -- Cross-GPU charts ----------------------------------------------------------

def cross_gpu_charts(summary, detail, method):
    s = summary[summary["method"] == method]
    gpus = sorted(s["gpu"].unique())
    if len(gpus) < 2:
        return

    subdir = f"cross_gpu_{method}"
    pretty = _label(method)
    print(f"\n--- Cross-GPU: {pretty} ({', '.join(_gpu_label(g) for g in gpus)}) ---")

    _line(s, "concurrency", "ttft_mean_sec", "TTFT (sec)",
          f"TTFT — {pretty}", subdir, "ttft_vs_concurrency.png", group_col="gpu", label_fn=_gpu_label)
    _line(s, "concurrency", "tps_total", "Aggregate Throughput (tok/sec)",
          f"Total Throughput — {pretty}", subdir, "tps_total_vs_concurrency.png", group_col="gpu", label_fn=_gpu_label)
    _line(s, "concurrency", "cost_per_1m_tokens_usd", "Cost per 1M Tokens (USD)",
          f"Inference Cost — {pretty}", subdir, "cost_vs_concurrency.png", group_col="gpu", label_fn=_gpu_label, yformat="$%.4f")
    _quality_bar(s, "gpu", f"Quality Pass Rate — {pretty}", subdir, label_fn=_gpu_label)


# -- Main ----------------------------------------------------------------------

def main():
    summary, detail = aggregate()
    print(f"\nGenerating charts to {CHARTS_DIR} ...")

    for gpu in sorted(summary["gpu"].unique()):
        per_gpu_charts(summary, detail, gpu)

    for method in sorted(summary["method"].unique()):
        cross_gpu_charts(summary, detail, method)

    print("\nDone.")


if __name__ == "__main__":
    main()
