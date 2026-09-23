#!/usr/bin/env python3
"""Compare greedy checkpoint responses, separating truncation recovery from learned corrections."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt


CAPABILITIES_BY_TEMPLATE = {
    1: ["visual.legend_category_enumeration", "symbolic.counting"],
    2: ["visual.legend_order_reading", "symbolic.intensity_value_direction"],
    3: ["visual.legend_endpoint_reading", "symbolic.range_composition"],
    5: ["visual.palette_discrimination", "visual.legend_structure"],
    6: ["visual.legend_semantics", "language.quantity_type_classification"],
    9: ["visual.region_localization", "grounding.region_color_to_range", "symbolic.range_membership"],
    13: ["spatial.directional_subset", "grounding.region_color_to_value", "symbolic.argmin"],
    16: ["visual.region_localization", "grounding.region_color_to_range"],
    18: ["spatial.adjacency", "grounding.region_color_to_value", "symbolic.similarity"],
    19: ["visual.region_localization", "grounding.region_color_to_value", "symbolic.pairwise_comparison"],
    21: ["visual.region_localization", "grounding.region_color_to_value", "symbolic.equality"],
    34: ["spatial.directional_extremum", "spatial.adjacency", "symbolic.local_maximum"],
    39: ["spatial.region_set_selection", "spatial.adjacency_or_coast", "symbolic.universal_quantification"],
    41: ["visual.legend_endpoint_reading", "symbolic.interval_width_comparison"],
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def is_correct(row: dict) -> bool:
    return bool(row.get("strict_exact_match", row.get("primary_score", 0)))


def is_terminated(row: dict) -> bool:
    return bool(row.get("terminated", row.get("generation_status") == "complete"))


def checkpoint_metrics(rows: list[dict]) -> dict:
    total = len(rows)
    correct = sum(is_correct(row) for row in rows)
    terminated = [row for row in rows if is_terminated(row)]
    truncated = total - len(terminated)
    return {
        "total": total,
        "correct": correct,
        "accuracy_percent": 100 * correct / total,
        "terminated": len(terminated),
        "truncated": truncated,
        "truncated_percent": 100 * truncated / total,
        "correct_among_terminated": sum(is_correct(row) for row in terminated),
        "accuracy_among_terminated_percent": (
            100 * sum(is_correct(row) for row in terminated) / len(terminated)
            if terminated else None
        ),
        "mean_generated_tokens": mean(int(row.get("generated_tokens", 0)) for row in rows),
        "mean_generated_tokens_terminated": (
            mean(int(row.get("generated_tokens", 0)) for row in terminated)
            if terminated else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation_dir", type=Path)
    parser.add_argument("--checkpoints", nargs="+", default=["baseline", "checkpoint-2000", "checkpoint-3000"])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    evaluation_dir = args.evaluation_dir.resolve()
    output_dir = (args.output_dir or evaluation_dir / "reasoning_analysis").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records: dict[str, dict[str, dict]] = {}
    for checkpoint in args.checkpoints:
        path = evaluation_dir / checkpoint / "responses.jsonl"
        rows = read_jsonl(path)
        indexed = {str(row["qa_id"]): row for row in rows}
        if len(indexed) != len(rows):
            raise ValueError(f"Duplicate qa_id values in {path}")
        records[checkpoint] = indexed

    common_ids = set.intersection(*(set(rows) for rows in records.values()))
    if any(len(rows) != len(common_ids) for rows in records.values()):
        raise ValueError("Checkpoint response files do not contain the same QA IDs")

    baseline_name = args.checkpoints[0]
    baseline = records[baseline_name]
    summary = {
        "checkpoints": {
            checkpoint: checkpoint_metrics(list(records[checkpoint].values()))
            for checkpoint in args.checkpoints
        },
        "comparisons": {},
    }

    comparison_rows: list[dict] = []
    raw_path = output_dir / "per_qa_reasoning_comparison.jsonl"
    with raw_path.open("w", encoding="utf-8") as raw_handle:
        for qa_id in sorted(common_ids):
            base = baseline[qa_id]
            flat = {
                "qa_id": qa_id,
                "country": base.get("country", ""),
                "template_no": base.get("template_no", ""),
                "ground_truth_type": base.get("ground_truth_type", ""),
                "question": base.get("question", ""),
                "ground_truth": base.get("ground_truth", ""),
                "capabilities": "|".join(CAPABILITIES_BY_TEMPLATE.get(int(base.get("template_no", -1)), ["unclassified"])),
            }
            raw = {key: flat[key] for key in flat}
            for checkpoint in args.checkpoints:
                row = records[checkpoint][qa_id]
                prefix = checkpoint.replace("checkpoint-", "step_").replace("-", "_")
                flat[f"{prefix}_correct"] = int(is_correct(row))
                flat[f"{prefix}_terminated"] = int(is_terminated(row))
                flat[f"{prefix}_generated_tokens"] = int(row.get("generated_tokens", 0))
                flat[f"{prefix}_final_answer"] = row.get("final_answer", "")
                raw[checkpoint] = {
                    "correct": is_correct(row),
                    "terminated": is_terminated(row),
                    "generated_tokens": int(row.get("generated_tokens", 0)),
                    "final_answer": row.get("final_answer", ""),
                    "raw_response": row.get("raw_response", ""),
                    "evaluation_note": row.get("evaluation_note", ""),
                }
            comparison_rows.append(flat)
            raw_handle.write(json.dumps(raw, ensure_ascii=False) + "\n")

    for checkpoint in args.checkpoints[1:]:
        target = records[checkpoint]
        transitions = Counter()
        grouped = defaultdict(Counter)
        for qa_id in common_ids:
            before = baseline[qa_id]
            after = target[qa_id]
            before_correct = is_correct(before)
            after_correct = is_correct(after)
            before_term = is_terminated(before)
            after_term = is_terminated(after)
            accuracy_transition = (
                "correct_to_correct" if before_correct and after_correct else
                "correct_to_wrong" if before_correct else
                "wrong_to_correct" if after_correct else
                "wrong_to_wrong"
            )
            termination_transition = (
                f"{'terminated' if before_term else 'truncated'}_to_"
                f"{'terminated' if after_term else 'truncated'}"
            )
            transitions[accuracy_transition] += 1
            transitions[termination_transition] += 1
            if not before_correct and after_correct:
                transitions[
                    "wrong_to_correct_from_terminated_baseline"
                    if before_term else "wrong_to_correct_from_truncated_baseline"
                ] += 1
            if before_term and after_term:
                transitions[f"both_terminated_{accuracy_transition}"] += 1
            for field in ("country", "template_no", "ground_truth_type"):
                grouped[field, str(before.get(field, ""))][accuracy_transition] += 1

        summary["comparisons"][f"{baseline_name}_vs_{checkpoint}"] = {
            "transitions": dict(transitions),
            "net_correct_gain": transitions["wrong_to_correct"] - transitions["correct_to_wrong"],
            "strong_learning_candidates": transitions["both_terminated_wrong_to_correct"],
            "truncation_recovery_candidates": transitions["wrong_to_correct_from_truncated_baseline"],
            "grouped_accuracy_transitions": {
                f"{field}={value}": dict(counts)
                for (field, value), counts in sorted(grouped.items())
            },
        }

    (output_dir / "reasoning_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "per_qa_transition.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)

    evidence_rows = []
    for row in comparison_rows:
        if (row["baseline_terminated"] and not row["baseline_correct"]
                and row["step_3000_terminated"] and row["step_3000_correct"]):
            evidence_rows.append(row)
    with (output_dir / "strong_learning_evidence.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        # An empty result is meaningful: no baseline-wrong to target-correct
        # transition survived the stable-termination filter.  Still emit a
        # schema-valid CSV so downstream analysis does not need a special case.
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(evidence_rows)

    metric_rows = []
    for checkpoint in args.checkpoints:
        metric_rows.append({"checkpoint": checkpoint, **summary["checkpoints"][checkpoint]})
    with (output_dir / "checkpoint_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)

    transition_rows = []
    for comparison, payload in summary["comparisons"].items():
        transitions = payload["transitions"]
        transition_rows.append({
            "comparison": comparison,
            "wrong_to_correct": transitions.get("wrong_to_correct", 0),
            "correct_to_wrong": transitions.get("correct_to_wrong", 0),
            "net_correct_gain": payload["net_correct_gain"],
            "stable_terminated_wrong_to_correct": transitions.get("both_terminated_wrong_to_correct", 0),
            "stable_terminated_correct_to_wrong": transitions.get("both_terminated_correct_to_wrong", 0),
            "wrong_to_correct_from_truncated_baseline": transitions.get(
                "wrong_to_correct_from_truncated_baseline", 0
            ),
        })
    with (output_dir / "transition_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(transition_rows[0]))
        writer.writeheader()
        writer.writerows(transition_rows)

    labels = [checkpoint.replace("checkpoint-", "step ") for checkpoint in args.checkpoints]
    accuracy = [summary["checkpoints"][checkpoint]["accuracy_percent"] for checkpoint in args.checkpoints]
    truncated = [summary["checkpoints"][checkpoint]["truncated_percent"] for checkpoint in args.checkpoints]
    completed_accuracy = [
        summary["checkpoints"][checkpoint]["accuracy_among_terminated_percent"]
        for checkpoint in args.checkpoints
    ]
    fig, axis = plt.subplots(figsize=(10, 6))
    axis.plot(labels, accuracy, marker="o", linewidth=2.2, label="Overall accuracy")
    axis.plot(labels, truncated, marker="s", linewidth=2.2, label="Truncated fraction")
    axis.plot(labels, completed_accuracy, marker="^", linewidth=2.2, label="Accuracy among completed")
    axis.set_xlabel("Checkpoint")
    axis.set_ylabel("Percent")
    axis.set_ylim(0, 100)
    axis.set_title("MapWise Test Metrics Across GRPO Training")
    axis.grid(True, alpha=0.25)
    axis.legend()
    for series in (accuracy, truncated, completed_accuracy):
        for index, value in enumerate(series):
            axis.annotate(f"{value:.1f}", (index, value), xytext=(0, 7),
                          textcoords="offset points", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "checkpoint_metric_curves.png", dpi=200)
    fig.savefig(output_dir / "checkpoint_metric_curves.pdf")
    plt.close(fig)

    single_plots = (
        (accuracy, "Overall accuracy", "Acc (%)", "overall_accuracy_curve", "tab:blue"),
        (truncated, "Truncated fraction", "Truncated fraction", "truncated_fraction_curve", "tab:orange"),
    )
    for values, title, ylabel, filename, color in single_plots:
        fig, axis = plt.subplots(figsize=(8, 5.5))
        if filename == "overall_accuracy_curve":
            training_steps = [int(checkpoint.removeprefix("checkpoint-"))
                              for checkpoint in args.checkpoints[1:]]
            axis.axhline(values[0], color="tab:gray", linestyle="--", linewidth=2.2,
                         label="Baseline")
            axis.plot(training_steps, values[1:], marker="o", linewidth=2.4,
                      color=color, label="GRPO checkpoints")
            axis.set_xlabel("Training steps")
            axis.set_xlim(1000, 3050)
            axis.set_xticks(list(range(1000, 3001, 500)))
            axis.set_ylim(30, 100)
            # Indicate that x=0--1000 steps and y=0--30% are omitted.
            break_style = dict(transform=axis.transAxes, color="black",
                               clip_on=False, linewidth=1.5)
            # Break on the vertical axis.
            axis.plot((-0.010, 0.010), (0.012, 0.038), **break_style)
            axis.plot((-0.010, 0.010), (0.044, 0.070), **break_style)
            # Break on the horizontal axis.
            axis.plot((0.018, 0.038), (-0.012, 0.012), **break_style)
            axis.plot((0.050, 0.070), (-0.012, 0.012), **break_style)
            axis.legend()
        else:
            axis.plot(labels, values, marker="o", linewidth=2.4, color=color)
            axis.set_xlabel("Checkpoint")
        axis.set_ylabel(ylabel)
        if filename == "truncated_fraction_curve":
            axis.set_ylim(0, 50)
        axis.set_title(title)
        axis.grid(True, alpha=0.25)
        if filename != "overall_accuracy_curve":
            for index, value in enumerate(values):
                axis.annotate(f"{value:.1f}", (index, value), xytext=(0, 7),
                              textcoords="offset points", ha="center", fontsize=9)
        fig.tight_layout()
        fig.savefig(output_dir / f"{filename}.png", dpi=200)
        fig.savefig(output_dir / f"{filename}.pdf")
        plt.close(fig)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
