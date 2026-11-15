#!/usr/bin/env python3
"""Paired Relay3 vs adapter-only accuracy by L0--L5 ability tag."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


VLM_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = VLM_ROOT / "Training_outputs/Prototype/relay3_vs_adapter_by_ability"
DATASETS = {
    "mapverse": {
        "metadata": VLM_ROOT / "Datasets/Subset_MapVerse/scaling_1500/MapVerse_test_600_quality_screened.json",
        "adapter": VLM_ROOT / "Training_outputs/Prototype/adapter_only/evaluation_mapverse_test600_checkpoint_2000_4000/responses.jsonl",
        "relay": VLM_ROOT / "Training_outputs/Prototype/relay3/evaluation_mapverse_test600_initial_without_relay_2000_4000_5000/responses.jsonl",
    },
    "mapwise": {
        "metadata": VLM_ROOT / "Datasets/Processed_Mapwise/Test/mapwise_grpo_test_400.json",
        "adapter": VLM_ROOT / "Training_outputs/Prototype/adapter_only/evaluation_mapwise_test400_checkpoint_2000_4000/responses.jsonl",
        "relay": VLM_ROOT / "Training_outputs/Prototype/relay3/evaluation_mapwise_test400_initial_without_relay_2000_4000_5000/responses.jsonl",
    },
}
ABILITY_FOCUS = {
    "L0": "Legend and symbol reading",
    "L1": "Single-object lookup / grounding",
    "L2": "Single-layer element comparison / aggregation",
    "L3": "Single-layer spatial reasoning",
    "L4": "Complex spatial reasoning over multiple elements",
    "L5": "Reasoning / computation with complex spatial linkage",
}


def read_jsonl(path: Path, step: int) -> dict[int, dict]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if int(row["checkpoint_step"]) != step:
            continue
        index = int(row["dataset_index"]) - 1
        if index in rows:
            raise ValueError(f"Duplicate dataset_index={index + 1} at step {step}: {path}")
        rows[index] = row
    return rows


def exact_mcnemar_p(relay_win: int, adapter_win: int) -> float:
    """Two-sided exact binomial p for discordant paired outcomes."""
    n = relay_win + adapter_win
    if n == 0:
        return 1.0
    low = min(relay_win, adapter_win)
    tail = sum(math.comb(n, k) for k in range(low + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def paired_ci(differences: list[int]) -> tuple[float, float]:
    """Normal 95% CI for the paired accuracy difference, in proportions."""
    n = len(differences)
    mean = sum(differences) / n
    if n < 2:
        return mean, mean
    variance = sum((value - mean) ** 2 for value in differences) / (n - 1)
    half = 1.96 * math.sqrt(variance / n)
    return mean - half, mean + half


def median(values: list[int]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2


def bh_adjust(rows: list[dict]) -> None:
    """Benjamini-Hochberg adjustment within one dataset/checkpoint family."""
    ordered = sorted(enumerate(rows), key=lambda pair: pair[1]["mcnemar_p"])
    m = len(rows)
    adjusted = [1.0] * m
    running = 1.0
    for rank_from_end in range(m - 1, -1, -1):
        original_index, row = ordered[rank_from_end]
        rank = rank_from_end + 1
        running = min(running, row["mcnemar_p"] * m / rank)
        adjusted[original_index] = running
    for row, value in zip(rows, adjusted):
        row["mcnemar_p_bh"] = value


def tags(value) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if not value:
        return []
    try:
        parsed = ast.literal_eval(str(value))
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except (SyntaxError, ValueError):
        pass
    return [str(value)]


def top(counter: Counter, limit: int = 4) -> str:
    return ", ".join(f"{key} ({count})" for key, count in counter.most_common(limit))


def summarize_group(dataset: str, level: str, indices: list[int], metadata: list[dict],
                    adapter: dict[int, dict], relay: dict[int, dict], step: int) -> dict:
    paired = [(adapter[index], relay[index]) for index in indices]
    a_correct = sum(float(a["reward"]) for a, _ in paired)
    r_correct = sum(float(r["reward"]) for _, r in paired)
    relay_win = sum(float(a["reward"]) == 0 and float(r["reward"]) == 1 for a, r in paired)
    adapter_win = sum(float(a["reward"]) == 1 and float(r["reward"]) == 0 for a, r in paired)
    differences = [int(float(r["reward"]) == 1) - int(float(a["reward"]) == 1) for a, r in paired]
    low, high = paired_ci(differences)
    answer_key = "answer_type" if dataset == "mapverse" else "ground_truth_type"
    answer_types = Counter(str(metadata[index].get(answer_key, "Unknown")) for index in indices)
    question_types = Counter(
        tag for index in indices for tag in tags(metadata[index].get("question_type"))
    ) if dataset == "mapverse" else Counter()
    both_correct_terminated = [
        (a, r) for a, r in paired
        if float(a["reward"]) == 1 and float(r["reward"]) == 1
        and bool(a.get("terminated", True)) and bool(r.get("terminated", True))
    ]
    token_savings = [
        int(a["generated_tokens"]) - int(r["generated_tokens"])
        for a, r in both_correct_terminated
    ]
    return {
        "dataset": dataset,
        "checkpoint_step": step,
        "ability_level": level,
        "ability_focus": ABILITY_FOCUS.get(level, "Unknown"),
        "n": len(indices),
        "adapter_correct": int(a_correct),
        "relay_correct": int(r_correct),
        "adapter_accuracy": a_correct / len(indices),
        "relay_accuracy": r_correct / len(indices),
        "delta_percentage_points": 100 * (r_correct - a_correct) / len(indices),
        "paired_delta_ci95_low_pp": 100 * low,
        "paired_delta_ci95_high_pp": 100 * high,
        "relay_only_correct": relay_win,
        "adapter_only_correct": adapter_win,
        "net_paired_wins": relay_win - adapter_win,
        "discordant_pairs": relay_win + adapter_win,
        "mcnemar_p": exact_mcnemar_p(relay_win, adapter_win),
        "both_correct_terminated_n": len(both_correct_terminated),
        "adapter_mean_tokens_both_correct": (
            sum(int(a["generated_tokens"]) for a, _ in both_correct_terminated)
            / len(both_correct_terminated) if both_correct_terminated else None
        ),
        "relay_mean_tokens_both_correct": (
            sum(int(r["generated_tokens"]) for _, r in both_correct_terminated)
            / len(both_correct_terminated) if both_correct_terminated else None
        ),
        "mean_token_saving_both_correct": (
            sum(token_savings) / len(token_savings) if token_savings else None
        ),
        "median_token_saving_both_correct": median(token_savings),
        "relay_shorter_both_correct": sum(value > 0 for value in token_savings),
        "adapter_shorter_both_correct": sum(value < 0 for value in token_savings),
        "equal_length_both_correct": sum(value == 0 for value in token_savings),
        "answer_type_composition": top(answer_types),
        "question_type_composition": top(question_types) if question_types else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, nargs="+", default=[2000, 4000])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    all_rows = []
    dataset_info = {}
    for dataset, paths in DATASETS.items():
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        by_level = defaultdict(list)
        for index, item in enumerate(metadata):
            by_level[str(item.get("ability_level", "Unknown"))].append(index)
        dataset_info[dataset] = {level: len(indices) for level, indices in sorted(by_level.items())}
        for step in args.steps:
            adapter = read_jsonl(paths["adapter"], step)
            relay = read_jsonl(paths["relay"], step)
            expected = set(range(len(metadata)))
            if set(adapter) != expected or set(relay) != expected:
                raise ValueError(f"Incomplete or misaligned rows: {dataset}, step {step}")
            family = [
                summarize_group(dataset, level, indices, metadata, adapter, relay, step)
                for level, indices in sorted(by_level.items())
            ]
            bh_adjust(family)
            all_rows.extend(family)

    fields = list(all_rows[0])
    with (args.output / "accuracy_by_ability.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    (args.output / "accuracy_by_ability.json").write_text(
        json.dumps({"dataset_level_counts": dataset_info, "rows": all_rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "# Relay3 versus adapter-only by ability level",
        "",
        "Paired comparison on identical questions. `Relay-only` means Relay3 correct and adapter-only wrong; "
        "`Adapter-only` is the reverse. The p-value is exact McNemar, and q is Benjamini-Hochberg "
        "adjusted across levels within each dataset/checkpoint.",
        "",
        "MapWise contains L0-L3 only. MapVerse contains L0-L5. Their level taxonomies come from their "
        "respective dataset metadata, so the primary comparisons stay separated by dataset.",
        "",
        "Ability line used in this report:",
        "",
        *[f"- {level}: {focus}" for level, focus in ABILITY_FOCUS.items()],
    ]
    for step in args.steps:
        lines += ["", f"## Checkpoint {step}", ""]
        for dataset in DATASETS:
            rows = [row for row in all_rows if row["checkpoint_step"] == step and row["dataset"] == dataset]
            lines += [f"### {dataset.title()}", "",
                      "| Level | Focus | n | Adapter | Relay3 | Delta pp | Relay-only | Adapter-only | Net | 95% paired CI pp | p | q |",
                      "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
            for row in rows:
                lines.append(
                    f"| {row['ability_level']} | {row['ability_focus']} | {row['n']} | {100*row['adapter_accuracy']:.1f}% | "
                    f"{100*row['relay_accuracy']:.1f}% | {row['delta_percentage_points']:+.1f} | "
                    f"{row['relay_only_correct']} | {row['adapter_only_correct']} | {row['net_paired_wins']:+d} | "
                    f"[{row['paired_delta_ci95_low_pp']:.1f}, {row['paired_delta_ci95_high_pp']:.1f}] | "
                    f"{row['mcnemar_p']:.4f} | {row['mcnemar_p_bh']:.4f} |"
                )
            lines += ["", "Efficiency among paired, terminated responses where both systems are correct:", "",
                      "| Level | n | Adapter mean tokens | Relay3 mean tokens | Mean saving | Median saving | Relay3 shorter | Adapter shorter |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|"]
            for row in rows:
                if row["both_correct_terminated_n"]:
                    lines.append(
                        f"| {row['ability_level']} | {row['both_correct_terminated_n']} | "
                        f"{row['adapter_mean_tokens_both_correct']:.1f} | "
                        f"{row['relay_mean_tokens_both_correct']:.1f} | "
                        f"{row['mean_token_saving_both_correct']:+.1f} | "
                        f"{row['median_token_saving_both_correct']:+.1f} | "
                        f"{row['relay_shorter_both_correct']} | {row['adapter_shorter_both_correct']} |"
                    )
            lines += ["", "Tag composition:", ""]
            for row in rows:
                extra = f"; question tags: {row['question_type_composition']}" if row["question_type_composition"] else ""
                lines.append(f"- {row['ability_level']} (n={row['n']}): answers: {row['answer_type_composition']}{extra}")
    (args.output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output / "report.md")


if __name__ == "__main__":
    main()
