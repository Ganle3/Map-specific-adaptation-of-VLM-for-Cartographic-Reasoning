#!/usr/bin/env python3
"""Compare token efficiency and correctness of relay3 and adapter-only evals."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def load_rows(directory: Path) -> dict[tuple[int, int], dict]:
    path = directory / "responses.jsonl"
    rows: dict[tuple[int, int], dict] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        key = (int(row["checkpoint_step"]), int(row["dataset_index"]))
        if key in rows:
            raise ValueError(f"Duplicate {key} in {path}:{line_number}")
        if "generated_tokens" not in row:
            raise ValueError(f"Missing generated_tokens in {path}:{line_number}")
        rows[key] = row
    return rows


def mean(values: list[int]) -> float | None:
    return statistics.fmean(values) if values else None


def prediction(row: dict) -> str:
    evaluation = row.get("evaluation") or {}
    value = evaluation.get("canonical_prediction") or evaluation.get("prediction") or ""
    if value and len(str(value)) <= 200:
        return str(value)
    response = str(row.get("response", ""))
    marker = response.lower().rfind("final answer:")
    return response[marker + len("final answer:"):].strip() if marker >= 0 else response[-200:]


def summarize(name: str, dataset: str, rows: dict[tuple[int, int], dict]) -> list[dict]:
    result = []
    for step in sorted({key[0] for key in rows}):
        part = [row for (candidate, _), row in rows.items() if candidate == step]
        tokens = [int(row["generated_tokens"]) for row in part]
        correct = [int(row["generated_tokens"]) for row in part if float(row["reward"]) == 1.0]
        wrong = [int(row["generated_tokens"]) for row in part if float(row["reward"]) != 1.0]
        ordered = sorted(tokens)
        result.append({
            "dataset": dataset,
            "model": name,
            "checkpoint_step": step,
            "samples": len(part),
            "correct": len(correct),
            "accuracy": len(correct) / len(part),
            "mean_tokens": statistics.fmean(tokens),
            "median_tokens": statistics.median(tokens),
            "p90_tokens": ordered[max(0, int(0.9 * len(ordered) + 0.999999) - 1)],
            "min_tokens": min(tokens),
            "max_tokens": max(tokens),
            "mean_tokens_correct": mean(correct),
            "mean_tokens_wrong": mean(wrong),
            "truncated": sum(not bool(row.get("terminated", False)) for row in part),
        })
    return result


def compare(dataset: str, relay: dict, adapter: dict, step: int) -> tuple[dict, list[dict]]:
    relay_part = {index: row for (candidate, index), row in relay.items() if candidate == step}
    adapter_part = {index: row for (candidate, index), row in adapter.items() if candidate == step}
    if relay_part.keys() != adapter_part.keys():
        raise ValueError(f"{dataset} step {step}: evaluated question sets differ")
    differences = []
    jointly_correct = []
    relay_wins = []
    both_correct = both_wrong = adapter_wins = 0
    for index in sorted(relay_part):
        r, a = relay_part[index], adapter_part[index]
        if (str(r["question"]) != str(a["question"])
                or str(r["ground_truth"]) != str(a["ground_truth"])):
            raise ValueError(f"{dataset} step {step} index {index}: question mismatch")
        r_correct = float(r["reward"]) == 1.0
        a_correct = float(a["reward"]) == 1.0
        delta = int(r["generated_tokens"]) - int(a["generated_tokens"])
        differences.append(delta)
        if r_correct and a_correct:
            both_correct += 1
            jointly_correct.append(delta)
        elif r_correct and not a_correct:
            relay_wins.append({
                "dataset": dataset,
                "checkpoint_step": step,
                "dataset_index": index,
                "qa_id": r.get("qa_id", ""),
                "question": r["question"],
                "ground_truth": r["ground_truth"],
                "relay_prediction": prediction(r),
                "adapter_prediction": prediction(a),
                "relay_tokens": int(r["generated_tokens"]),
                "adapter_tokens": int(a["generated_tokens"]),
                "token_difference_relay_minus_adapter": delta,
                "relay_terminated": bool(r.get("terminated", False)),
                "adapter_terminated": bool(a.get("terminated", False)),
            })
        elif not r_correct and a_correct:
            adapter_wins += 1
        else:
            both_wrong += 1
    return {
        "dataset": dataset,
        "checkpoint_step": step,
        "samples": len(differences),
        "mean_paired_token_difference_relay_minus_adapter": statistics.fmean(differences),
        "median_paired_token_difference_relay_minus_adapter": statistics.median(differences),
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "relay_only_correct": len(relay_wins),
        "adapter_only_correct": adapter_wins,
        "jointly_correct_samples": len(jointly_correct),
        "jointly_correct_mean_token_difference_relay_minus_adapter": mean(jointly_correct),
        "jointly_correct_median_token_difference_relay_minus_adapter": (
            statistics.median(jointly_correct) if jointly_correct else None
        ),
    }, relay_wins


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-mapverse", type=Path, required=True)
    parser.add_argument("--relay-mapverse", type=Path, required=True)
    parser.add_argument("--adapter-mapwise", type=Path, required=True)
    parser.add_argument("--relay-mapwise", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    sources = {
        "mapverse": {
            "adapter_only": load_rows(args.adapter_mapverse),
            "relay3": load_rows(args.relay_mapverse),
        },
        "mapwise": {
            "adapter_only": load_rows(args.adapter_mapwise),
            "relay3": load_rows(args.relay_mapwise),
        },
    }
    summaries, comparisons, wins = [], [], []
    for dataset, models in sources.items():
        for name, rows in models.items():
            summaries.extend(summarize(name, dataset, rows))
        common_steps = sorted(
            {step for step, _ in models["adapter_only"]}
            & {step for step, _ in models["relay3"]}
        )
        for step in common_steps:
            comparison, step_wins = compare(
                dataset, models["relay3"], models["adapter_only"], step
            )
            comparisons.append(comparison)
            wins.extend(step_wins)

    unique_wins = {}
    for row in wins:
        key = (row["dataset"], row["dataset_index"])
        item = unique_wins.setdefault(key, {
            "dataset": row["dataset"], "dataset_index": row["dataset_index"],
            "qa_id": row["qa_id"], "question": row["question"],
            "ground_truth": row["ground_truth"], "relay_win_steps": [],
        })
        step = int(row["checkpoint_step"])
        item["relay_win_steps"].append(step)
        for name in ("relay_prediction", "adapter_prediction", "relay_tokens",
                     "adapter_tokens", "token_difference_relay_minus_adapter"):
            item[f"step_{step}_{name}"] = row[name]
    unique_rows = []
    for item in unique_wins.values():
        item["relay_win_steps"] = ";".join(map(str, sorted(item["relay_win_steps"])))
        unique_rows.append(item)
    unique_rows.sort(key=lambda row: (row["dataset"], row["dataset_index"]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summaries, "paired_comparison": comparisons,
               "relay_correct_adapter_wrong": wins,
               "relay_correct_adapter_wrong_unique": unique_rows}
    (args.output_dir / "analysis.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(args.output_dir / "token_summary.csv", summaries)
    write_csv(args.output_dir / "paired_comparison.csv", comparisons)
    write_csv(args.output_dir / "relay_correct_adapter_wrong.csv", wins)
    write_csv(args.output_dir / "relay_correct_adapter_wrong_unique.csv", unique_rows)
    print(json.dumps({
        "summary": summaries,
        "paired_comparison": comparisons,
        "relay_win_count": len(wins),
        "relay_win_unique_question_count": len(unique_rows),
        "output_dir": str(args.output_dir.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
