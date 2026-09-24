#!/usr/bin/env python3
"""Plot MapWise test accuracy for joint, vision-only, and language-only LoRA."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


COLORS = {
    "Joint": "#315A7D",
    "Vision-only": "#B66A3C",
    "Language-only": "#3F7D5A",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    csv_path = args.csv_path.expanduser().resolve()
    output_dir = (args.output_dir or csv_path.parent).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with csv_path.open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    baseline_values = {float(row["baseline_accuracy_percent"]) for row in rows}
    if len(baseline_values) != 1:
        raise ValueError(f"Expected one shared baseline, got {sorted(baseline_values)}")
    baseline = baseline_values.pop()

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
    })
    fig, axis = plt.subplots(figsize=(8.2, 5.6))

    for scope in ("Joint", "Vision-only", "Language-only"):
        selected = sorted(
            (row for row in rows if row["scope"] == scope),
            key=lambda row: int(row["training_step"]),
        )
        axis.plot(
            [int(row["training_step"]) for row in selected],
            [float(row["accuracy_percent"]) for row in selected],
            color=COLORS[scope], marker="o", markersize=5.5,
            linewidth=2.3, label=scope, zorder=3,
        )

    axis.axhline(
        baseline, color="#6B6B6B", linestyle=(0, (5, 3)), linewidth=1.8,
        label="Baseline", zorder=2,
    )
    axis.set_xlim(1400, 3050)
    axis.set_xticks([1500, 2000, 2500, 3000])
    axis.set_ylim(30, 100)
    axis.set_yticks(list(range(30, 101, 10)))
    axis.set_xlabel("Training steps")
    axis.set_ylabel("Acc (%)")
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.75)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False, loc="best")

    # Indicate omitted x=0--1400 steps and y=0--30% without adding fake ticks.
    break_style = dict(transform=axis.transAxes, color="black", clip_on=False, linewidth=1.35)
    axis.plot((-0.011, 0.011), (0.018, 0.046), **break_style)
    axis.plot((-0.011, 0.011), (0.054, 0.082), **break_style)
    axis.plot((0.008, 0.020), (-0.012, 0.012), **break_style)
    axis.plot((0.026, 0.038), (-0.012, 0.012), **break_style)

    fig.tight_layout()
    stem = output_dir / "mapwise_scope_ablation_test_accuracy"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(stem)


if __name__ == "__main__":
    main()
