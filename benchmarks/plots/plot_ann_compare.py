"""Plot recall-vs-QPS curves written by benchmarks/runners/ann_compare.py.

Usage:
    uv run python benchmarks/plots/plot_ann_compare.py \
        --results benchmarks/results/compare_sift1m.json \
        --output benchmarks/plots/compare_sift1m.png
"""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
LIBRARY_COLORS = {
    "hopfilter": "#2a78d6",
    "hnswlib": "#eb6834",
    "faiss": "#1baf7a",
    "hopfilter_scalar": "#eda100",
}
LIBRARY_LABELS = {
    "hopfilter": "hopfilter (AVX2)",
    "hnswlib": "hnswlib 0.8.0 (AVX2)",
    "faiss": "FAISS 1.15.1 (no SIMD, Windows wheel)",
    "hopfilter_scalar": "hopfilter (scalar)",
}
PANELS = (("qps_single_thread", "1 thread"), ("qps_all_threads", "all threads"))
LINE_WIDTH = 2.0
MARKER_SIZE = 6
FIGURE_SIZE = (12.0, 5.0)
DPI = 160
TARGET_RECALL = 0.95


def style_axes(axes: Axes, title: str) -> None:
    axes.set_facecolor(SURFACE)
    axes.set_title(title, loc="left", color=TEXT_PRIMARY, fontsize=12, pad=12)
    axes.set_xlabel("recall@10", color=TEXT_SECONDARY)
    axes.set_ylabel("queries per second (log scale)", color=TEXT_SECONDARY)
    axes.set_yscale("log")
    axes.grid(color=GRID, linewidth=0.8, which="both")
    axes.set_axisbelow(True)
    axes.tick_params(colors=TEXT_SECONDARY, length=0)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        axes.spines[side].set_color(GRID)


def plot_panel(axes: Axes, libraries: dict[str, Any], metric: str, title: str) -> None:
    for name, color in LIBRARY_COLORS.items():
        if name not in libraries:
            continue
        sweep = libraries[name]["sweep"]
        axes.plot(
            [row["recall"] for row in sweep],
            [row[metric] for row in sweep],
            color=color,
            linewidth=LINE_WIDTH,
            marker="o",
            markersize=MARKER_SIZE,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            label=LIBRARY_LABELS[name],
        )
    axes.axvline(TARGET_RECALL, color=TEXT_SECONDARY, linewidth=1, linestyle="dotted")
    style_axes(axes, title)
    axes.legend(frameon=False, labelcolor=TEXT_PRIMARY, loc="lower left")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    results = json.loads(arguments.results.read_text(encoding="utf-8"))
    figure, panels = plt.subplots(1, 2, figsize=FIGURE_SIZE, facecolor=SURFACE)
    threads = next(iter(results["libraries"].values()))["threads"]
    for axes, (metric, label) in zip(panels, PANELS, strict=True):
        suffix = f" ({threads})" if metric == "qps_all_threads" else ""
        plot_panel(axes, results["libraries"], metric, f"SIFT1M, {label}{suffix}")
    figure.suptitle(
        "Recall vs throughput at matched M=16, efConstruction=200 (up and right is better)",
        x=0.01,
        ha="left",
        color=TEXT_PRIMARY,
        fontsize=13,
    )
    figure.tight_layout()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=DPI, facecolor=SURFACE)
    print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
