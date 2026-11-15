#!/usr/bin/env python3
"""Select aligned step-4000 evaluation cases without confusing IDs and indices."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


VLM_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = VLM_ROOT / "Training_outputs/Prototype/relay3_latent_evidence"
SOURCES = {
    "mapverse": {
        "dataset": "Datasets/Subset_MapVerse/scaling_1500/MapVerse_test_600_quality_screened.json",
        "adapter": "Training_outputs/Prototype/adapter_only/evaluation_mapverse_test600_checkpoint_2000_4000/responses.jsonl",
        "relay": "Training_outputs/Prototype/relay3/evaluation_mapverse_test600_initial_without_relay_2000_4000_5000/responses.jsonl",
    },
    "mapwise": {
        "dataset": "Datasets/Processed_Mapwise/Test/mapwise_grpo_test_400.json",
        "adapter": "Training_outputs/Prototype/adapter_only/evaluation_mapwise_test400_checkpoint_2000_4000/responses.jsonl",
        "relay": "Training_outputs/Prototype/relay3/evaluation_mapwise_test400_initial_without_relay_2000_4000_5000/responses.jsonl",
    },
}
CATEGORIES = {
    "relay_only_correct": (0, 1),
    "both_correct": (1, 1),
    "both_wrong": (0, 0),
}


def read_step(path: Path, step: int) -> dict[int, dict]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if int(row["checkpoint_step"]) == step:
            index = int(row["dataset_index"])
            if index in result:
                raise ValueError(f"Duplicate dataset_index={index} at step={step}: {path}")
            result[index] = row
    return result


def image_name(dataset: str, row: dict) -> str:
    if dataset == "mapverse":
        return str(row["image_name"])
    base = (VLM_ROOT / "Datasets/mapwise-dataset" / str(row["country"]) / "images"
            / str(row.get("map_type") or "with_annotations") / str(row["map_no"]))
    candidates = [base] if base.suffix else [base.with_suffix(ext) for ext in
                                             (".png", ".jpg", ".jpeg", ".webp")]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.relative_to(VLM_ROOT / "Datasets/mapwise-dataset").as_posix()
    raise FileNotFoundError(f"MapWise image missing: {base}")


def diverse_ten(candidates: list[dict], number: int) -> list[dict]:
    """Cover the dataset index range while avoiding repeated maps."""
    remaining = sorted(candidates, key=lambda row: row["dataset_index"])
    selected = []
    used_images = set()
    for target in [(i + 0.5) * len(remaining) / number for i in range(number)]:
        eligible = [(j, row) for j, row in enumerate(remaining)
                    if row["image_name"] not in used_images]
        if not eligible:
            break
        position, row = min(eligible, key=lambda item: (abs(item[0] - target), item[0]))
        selected.append(row)
        used_images.add(row["image_name"])
        remaining.pop(position)
    if len(selected) != number:
        raise ValueError(f"Only {len(selected)} distinct maps for requested {number}")
    return sorted(selected, key=lambda row: row["dataset_index"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=int, default=4000)
    parser.add_argument("--per-category-dataset", type=int, default=10)
    parser.add_argument("--include-truncated", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = []
    counts = {}
    for dataset, paths in SOURCES.items():
        dataset_rows = json.loads((VLM_ROOT / paths["dataset"]).read_text(encoding="utf-8"))
        adapter = read_step(VLM_ROOT / paths["adapter"], args.step)
        relay = read_step(VLM_ROOT / paths["relay"], args.step)
        expected = set(range(1, len(dataset_rows) + 1))
        if set(adapter) != expected or set(relay) != expected:
            raise ValueError(f"Incomplete or misindexed {dataset} evaluation")
        grouped = {name: [] for name in CATEGORIES}
        for dataset_index in sorted(expected):
            source = dataset_rows[dataset_index - 1]  # dataset_index is ONE-based
            a, r = adapter[dataset_index], relay[dataset_index]
            question = str(source["question"])
            truth = str(source.get("correct_answer", source.get("ground_truth")))
            if (a["question"] != question or r["question"] != question
                    or a["ground_truth"] != truth or r["ground_truth"] != truth
                    or a["qa_id"] != r["qa_id"]):
                raise ValueError(f"Row mismatch {dataset} dataset_index={dataset_index}")
            if not args.include_truncated and (not a["terminated"] or not r["terminated"]):
                continue
            label = (int(a["reward"]), int(r["reward"]))
            for name, pair in CATEGORIES.items():
                if label == pair:
                    grouped[name].append({
                        "category": name, "dataset": dataset,
                        "dataset_index": dataset_index,
                        "array_index_0based": dataset_index - 1,
                        "qa_id": str(a["qa_id"]),
                        "image_name": image_name(dataset, source),
                        "question": question, "ground_truth": truth,
                        "answer_type": source.get("answer_type", source.get("ground_truth_type")),
                        "adapter_reward": int(a["reward"]),
                        "relay_reward": int(r["reward"]),
                        "adapter_response": a["response"],
                        "relay_response": r["response"],
                        "adapter_generated_tokens": a["generated_tokens"],
                        "relay_generated_tokens": r["generated_tokens"],
                    })
        counts[dataset] = {name: len(rows) for name, rows in grouped.items()}
        for name, rows in grouped.items():
            if len(rows) < args.per_category_dataset:
                raise ValueError(f"Only {len(rows)} eligible {dataset}/{name} questions")
            selected.extend(diverse_ten(rows, args.per_category_dataset))
    manifest = {
        "checkpoint_step": args.step,
        "selection_rule": "equal-index, equal-question, equal-truth, equal-qa-id, both terminated; "
                          "even index coverage with distinct images",
        "available_counts": counts,
        "questions": selected,
    }
    destination = args.output_dir / f"selected_step{args.step}.json"
    destination.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Relay3 latent evidence question selection", "",
             f"Step {args.step}; both responses terminated; 1-based dataset_index and 0-based array index are shown separately.", ""]
    for category in CATEGORIES:
        lines += [f"## {category}", "", "| Dataset | Index (1-based) | qa_id | Image | Ground truth | Question |", "|---|---:|---|---|---|---|"]
        for row in selected:
            if row["category"] == category:
                cells = [row["dataset"], str(row["dataset_index"]), row["qa_id"],
                         row["image_name"], row["ground_truth"], row["question"]]
                lines.append("| " + " | ".join(str(v).replace("|", "\\|").replace("\n", " ") for v in cells) + " |")
        lines.append("")
    (args.output_dir / f"selected_step{args.step}.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"selection": str(destination), "counts": counts,
                      "selected": len(selected)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
