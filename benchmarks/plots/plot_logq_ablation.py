"""Plot the logQ ablation results written by benchmarks/runners/logq_ablation.py.

Usage:
    uv run python benchmarks/plots/plot_logq_ablation.py \
        --results benchmarks/results/logq_ablation_ml1m.json \
        --output benchmarks/plots/logq_ablation_ml1m.png
"""

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
BASELINE_GRAY = "#8a8984"
SERIES_COLORS = {"in_batch_softmax": "#2a78d6", "logq_corrected": "#eb6834"}
SERIES_LABELS = {"in_batch_softmax": "In-batch softmax", "logq_corrected": "logQ corrected"}
SEGMENTS = (("recall", "All items"), ("head_recall", "Head items"), ("tail_recall", "Tail items"))
BAR_WIDTH = 0.26
BAR_GAP_POINTS = 2.0
LINE_WIDTH = 2.0
MARKER_SIZE = 8
FIGURE_SIZE = (12.0, 5.0)
DPI = 160


def load_results(path: Path) -> dict[str, Any]:
    results: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return results


def style_axes(axes: Axes, title: str, y_label: str) -> None:
    axes.set_facecolor(SURFACE)
    axes.set_title(title, loc="left", color=TEXT_PRIMARY, fontsize=12, pad=12)
    axes.set_ylabel(y_label, color=TEXT_SECONDARY)
    axes.grid(axis="y", color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)
    axes.tick_params(colors=TEXT_SECONDARY, length=0)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color(GRID)


@dataclass(frozen=True)
class BarSeries:
    label: str
    color: str
    values: list[float]
    spreads: list[float]


def draw_bar_series(axes: Axes, slot: int, series: BarSeries) -> None:
    centers = [position + (slot - 1) * BAR_WIDTH for position in range(len(series.values))]
    bars = axes.bar(
        centers,
        series.values,
        BAR_WIDTH,
        yerr=series.spreads,
        color=series.color,
        edgecolor=SURFACE,
        linewidth=BAR_GAP_POINTS,
        error_kw={"ecolor": TEXT_SECONDARY, "elinewidth": 1, "capsize": 3},
        label=series.label,
    )
    axes.bar_label(bars, fmt="%.3f", padding=4, color=TEXT_PRIMARY, fontsize=8)


def plot_test_recall(axes: Axes, results: dict[str, Any], k: str) -> None:
    summary, baseline = results["summary"], results["baselines"]["popularity"]
    for slot, name in enumerate(SERIES_COLORS):
        series = BarSeries(
            label=SERIES_LABELS[name],
            color=SERIES_COLORS[name],
            values=[summary[name][metric][k]["mean"] for metric, _ in SEGMENTS],
            spreads=[summary[name][metric][k]["std"] for metric, _ in SEGMENTS],
        )
        draw_bar_series(axes, slot, series)
    popularity = BarSeries(
        label="Most popular",
        color=BASELINE_GRAY,
        values=[baseline[metric][k] for metric, _ in SEGMENTS],
        spreads=[0.0] * len(SEGMENTS),
    )
    draw_bar_series(axes, len(SERIES_COLORS), popularity)
    labels = [label for _, label in SEGMENTS]
    axes.set_xticks(range(len(SEGMENTS)), labels, color=TEXT_PRIMARY)
    style_axes(axes, f"Test recall@{k} by item segment", f"recall@{k}")
    axes.legend(
        frameon=False,
        labelcolor=TEXT_PRIMARY,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=3,
    )


def plot_validation_curves(axes: Axes, results: dict[str, Any], k: int) -> None:
    for name, color in SERIES_COLORS.items():
        curves = [run["validation_recall_curve"] for run in results["runs"] if run["loss"] == name]
        epochs = range(1, len(curves[0]) + 1)
        means = [statistics.fmean(values) for values in zip(*curves, strict=True)]
        lows = [min(values) for values in zip(*curves, strict=True)]
        highs = [max(values) for values in zip(*curves, strict=True)]
        axes.fill_between(epochs, lows, highs, color=color, alpha=0.18, linewidth=0)
        axes.plot(epochs, means, color=color, linewidth=LINE_WIDTH, label=SERIES_LABELS[name])
        axes.plot(
            epochs[-1],
            means[-1],
            "o",
            color=color,
            markersize=MARKER_SIZE,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
        )
        axes.annotate(
            f"{means[-1]:.3f}",
            (epochs[-1], means[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            color=TEXT_PRIMARY,
            fontsize=9,
        )
    axes.set_xlabel("epoch (shaded: min-max over seeds)", color=TEXT_SECONDARY)
    axes.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes.set_ylim(bottom=0)
    axes.margins(x=0.08)
    style_axes(axes, f"Validation recall@{k} per epoch", f"recall@{k}")
    axes.legend(frameon=False, labelcolor=TEXT_PRIMARY, loc="lower right")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", default="50")
    arguments = parser.parse_args()
    results = load_results(arguments.results)
    figure, (test_axes, curve_axes) = plt.subplots(1, 2, figsize=FIGURE_SIZE, facecolor=SURFACE)
    plot_test_recall(test_axes, results, arguments.k)
    plot_validation_curves(curve_axes, results, results["config"]["evaluation"]["validation_k"])
    seeds = len(results["config"]["seeds"])
    seed_label = "seed" if seeds == 1 else "seeds"
    figure.suptitle(
        f"logQ ablation on MovieLens-1M (temporal split, {seeds} {seed_label}, mean ± std)",
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
