"""Plot recall-vs-QPS curves per selectivity written by benchmarks/runners/filter_sweep.py.

One row of panels per attribute (random, clustered) and one column per selectivity.
Pre-filtering is exact and does not depend on efSearch, so it is a single point.

Usage:
    uv run python benchmarks/plots/plot_filter_sweep.py \
        --results benchmarks/results/filter_sift1m.json \
        --output benchmarks/plots/filter_sift1m.png
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
STRATEGY_STYLES = {
    "post_filter": ("#eb6834", "post-filter", "solid"),
    "in_graph": ("#2a78d6", "in-graph", "solid"),
    "pre_filter": ("#1baf7a", "pre-filter (exact)", "solid"),
    "adaptive": ("#0b0b0b", "adaptive", "dashed"),
}
ATTRIBUTE_TITLES = {
    "random": "random attribute",
    "clustered": "clustered attribute",
}
LINE_WIDTH = 1.8
MARKER_SIZE = 5
PANEL_SIZE = (3.6, 3.2)
DPI = 150
RECALL_RANGE = (-0.03, 1.03)


def style_axes(axes: Axes, title: str, first_column: bool, last_row: bool) -> None:
    axes.set_facecolor(SURFACE)
    axes.set_title(title, loc="left", color=TEXT_PRIMARY, fontsize=10, pad=8)
    axes.set_yscale("log")
    axes.set_xlim(*RECALL_RANGE)
    axes.grid(color=GRID, linewidth=0.8, which="both")
    axes.set_axisbelow(True)
    axes.tick_params(colors=TEXT_SECONDARY, length=0, labelsize=8)
    if last_row:
        axes.set_xlabel("recall@10", color=TEXT_SECONDARY, fontsize=9)
    if first_column:
        axes.set_ylabel("QPS, all threads (log)", color=TEXT_SECONDARY, fontsize=9)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        axes.spines[side].set_color(GRID)


def plot_panel(axes: Axes, rows: list[dict[str, Any]]) -> None:
    for strategy, (color, label, line_style) in STRATEGY_STYLES.items():
        points = sorted(
            (row for row in rows if row["strategy"] == strategy), key=lambda row: row["ef_search"]
        )
        if not points:
            continue
        axes.plot(
            [row["recall"] for row in points],
            [row["qps"]["median"] for row in points],
            color=color,
            linewidth=LINE_WIDTH,
            linestyle=line_style,
            marker="o" if strategy != "pre_filter" else "D",
            markersize=MARKER_SIZE if strategy != "pre_filter" else MARKER_SIZE + 2,
            markeredgecolor=SURFACE,
            markeredgewidth=1.0,
            label=label,
        )


def selectivity_label(selectivity: float) -> str:
    return f"{selectivity * 100:g}% match"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    results = json.loads(arguments.results.read_text(encoding="utf-8"))
    sweep = results["sweep"]
    attributes = list(results["config"]["attributes"])
    selectivities = list(results["config"]["selectivities"])
    figure, panels = plt.subplots(
        len(attributes),
        len(selectivities),
        figsize=(PANEL_SIZE[0] * len(selectivities), PANEL_SIZE[1] * len(attributes)),
        facecolor=SURFACE,
        squeeze=False,
    )
    for row_index, attribute in enumerate(attributes):
        for column_index, selectivity in enumerate(selectivities):
            axes = panels[row_index][column_index]
            rows = [
                row
                for row in sweep
                if row["attribute"] == attribute and row["selectivity"] == selectivity
            ]
            plot_panel(axes, rows)
            title = f"{ATTRIBUTE_TITLES[attribute]}, {selectivity_label(selectivity)}"
            style_axes(axes, title, column_index == 0, row_index == len(attributes) - 1)
    handles, labels = panels[0][0].get_legend_handles_labels()
    threads = results["environment"]["logical_cpus"]
    figure.suptitle(
        f"SIFT1M filtered search, k=10, {threads} threads (up and right is better)",
        x=0.01,
        y=0.995,
        ha="left",
        color=TEXT_PRIMARY,
        fontsize=12,
    )
    figure.legend(
        handles,
        labels,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.005, 0.965),
        ncol=len(labels),
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=DPI, facecolor=SURFACE)
    print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
