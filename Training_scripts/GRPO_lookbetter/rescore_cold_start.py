"""Re-evaluate a completed cold-start JSONL without regenerating model output."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from data import evaluate_answer, extract_final_answer, load_mapverse_questions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument("--mapverse-image-root", type=Path, required=True)
    parser.add_argument("--mapwise-root", type=Path, required=True)
    args = parser.parse_args()
    records = load_mapverse_questions(args.dataset_json, args.mapverse_image_root, mapwise_root=args.mapwise_root)
    rows = [json.loads(line) for line in args.input_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    corrected = 0
    for row in rows:
        item = records[int(row["dataset_index"]) - 1]
        rewards = []
        for run in row["rollouts"]:
            old = float(run["reward"])
            evaluation = evaluate_answer(run["response"], item)
            run["evaluation"] = evaluation
            run["reward"] = evaluation["reward"]
            rewards.append(run["reward"])
            corrected += int(old != run["reward"])
        row["summary"]["rewards"] = rewards
        row["summary"]["final_answers"] = [extract_final_answer(run["response"]) for run in row["rollouts"]]
        row["summary"]["mixed_reward"] = len(set(rewards)) > 1
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(f"Re-scored {len(rows)} questions; changed {corrected} rollout rewards: {args.output_jsonl}")


if __name__ == "__main__":
    main()
