#!/usr/bin/env python3
"""Plot MapWise Correct/Wrong/Truncated composition across token budgets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter


COLORS = {
    "baseline": {"correct": "#315A7D", "wrong": "#B8CAD8"},
    "checkpoint-2500": {"correct": "#3F6B52", "wrong": "#BFD2C3"},
    "truncated": "#C7C7C7",
}


def parse_evaluation(value: str) -> tuple[int, Path]:
    try:
        token_text, path_text = value.split("=", 1)
        return int(token_text), Path(path_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected TOKENS=EVALUATION_DIR") from exc


def load_outcomes(evaluation_dir: Path, checkpoint: str) -> dict[str, float]:
    path = evaluation_dir.expanduser().resolve() / checkpoint / "responses.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    if len(rows) != 400:
        raise ValueError(f"Expected 400 samples in {path}, found {len(rows)}")

    counts = {"correct": 0, "wrong": 0, "truncated": 0}
    for row in rows:
        correct = bool(row.get("strict_exact_match", row.get("primary_score", 0)))
        terminated = bool(
            row.get("terminated", row.get("generation_status") == "complete")
        )
        if correct:
            counts["correct"] += 1
        elif not terminated:
            counts["truncated"] += 1
        else:
            counts["wrong"] += 1
    assert sum(counts.values()) == len(rows)
    return {key: 100.0 * value / len(rows) for key, value in counts.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", action="append", type=parse_evaluation,
                        required=True, help="Repeat TOKENS=EVALUATION_DIR.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    evaluations = sorted(args.evaluation)
    if [tokens for tokens, _ in evaluations] != [500, 1000, 1500]:
        raise ValueError("Expected exactly the token budgets 500, 1000, and 1500")

    checkpoints = ("baseline", "checkpoint-2500")
    labels = {"baseline": "Baseline", "checkpoint-2500": "Step 2500"}
    # Small within-pair spacing and visibly larger between-pair spacing.
    y_positions = {
        (500, "baseline"): 5.15, (500, "checkpoint-2500"): 4.55,
        (1000, "baseline"): 3.25, (1000, "checkpoint-2500"): 2.65,
        (1500, "baseline"): 1.35, (1500, "checkpoint-2500"): 0.75,
    }
    metrics = {
        (tokens, checkpoint): load_outcomes(path, checkpoint)
        for tokens, path in evaluations
        for checkpoint in checkpoints
    }

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10.5,
        "axes.labelsize": 11.5,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10.5,
    })
    fig, axis = plt.subplots(figsize=(9.4, 5.6))
    bar_height = 0.46

    for tokens, _ in evaluations:
        for checkpoint in checkpoints:
            values = metrics[(tokens, checkpoint)]
            y = y_positions[(tokens, checkpoint)]
            left = 0.0
            for outcome in ("correct", "wrong", "truncated"):
                value = values[outcome]
                color = (COLORS["truncated"] if outcome == "truncated"
                         else COLORS[checkpoint][outcome])
                axis.barh(y, value, left=left, height=bar_height, color=color,
                          edgecolor="white", linewidth=0.8, zorder=3)
                left += value

    ordered = [(tokens, checkpoint) for tokens, _ in evaluations
               for checkpoint in checkpoints]
    axis.set_yticks([y_positions[key] for key in ordered],
                    [labels[checkpoint] for _, checkpoint in ordered])

    # Higher-level token-budget labels centered beside each pair.
    y_transform = axis.get_yaxis_transform()
    for tokens, _ in evaluations:
        center = sum(y_positions[(tokens, checkpoint)] for checkpoint in checkpoints) / 2
        axis.text(-0.19, center, str(tokens), transform=y_transform,
                  ha="center", va="center", fontsize=11, fontweight="semibold")
    axis.text(-0.19, 5.78, "Max tokens", transform=y_transform,
              ha="center", va="bottom", fontsize=10, color="#444444")

    axis.set_xlim(0, 100)
    axis.set_ylim(0.35, 5.65)
    axis.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    axis.set_xlabel("Percentage of evaluation samples")
    axis.grid(axis="x", color="#D9D9D9", linewidth=0.8, alpha=0.8, zorder=0)
    axis.tick_params(axis="y", length=0, pad=7)
    for spine in ("top", "right", "left"):
        axis.spines[spine].set_visible(False)
    axis.spines["bottom"].set_color("#777777")

    correct_handle = (Patch(facecolor=COLORS["baseline"]["correct"]),
                      Patch(facecolor=COLORS["checkpoint-2500"]["correct"]))
    wrong_handle = (Patch(facecolor=COLORS["baseline"]["wrong"]),
                    Patch(facecolor=COLORS["checkpoint-2500"]["wrong"]))
    truncated_handle = Patch(facecolor=COLORS["truncated"])
    axis.legend([correct_handle, wrong_handle, truncated_handle],
                ["Correct (with answer)", "Wrong (with answer)",
                 "Truncated (without answer)"],
                handler_map={tuple: HandlerTuple(ndivide=None, pad=0.15)},
                loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=3,
                frameon=False, handlelength=2.0, columnspacing=1.8)

    fig.subplots_adjust(left=0.25, right=0.985, bottom=0.14, top=0.88)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "token_budget_outcome_composition"
    fig.savefig(stem.with_suffix(".png"), dpi=350, bbox_inches="tight",
                facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    for key in ordered:
        print(key, metrics[key])
    print(f"Wrote {stem}.[png|pdf|svg]")


if __name__ == "__main__":
    main()
