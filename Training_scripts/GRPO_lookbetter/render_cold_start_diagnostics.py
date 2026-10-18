#!/usr/bin/env python3
"""Render compact human-readable cold-start diagnostics from JSONL rollouts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _bbox(region: dict) -> str:
    return f"({region['left']},{region['top']})-({region['right']},{region['bottom']}) score={region['score']:.4g}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index_rows = ["# Cold-start diagnostic index\n", "| Dataset index | Question | Rewards | LOOK attempts | Final answers | File |",
                  "| --- | --- | --- | --- | --- | --- |"]
    with args.input_jsonl.open(encoding="utf-8") as handle:
        for count, line in enumerate(handle, 1):
            if args.limit is not None and count > args.limit:
                break
            row = json.loads(line)
            number = int(row["dataset_index"])
            name = f"question_{number:04d}.md"
            summary = row["summary"]
            text = [f"# Question {number}: {row['sample_id']}\n", f"**Dataset:** {row.get('dataset')}  ",
                    f"**Image:** `{row['image']}`  ", f"**Question:** {row['question']}  ",
                    f"**Gold:** {row['correct_answer']}\n"]
            for run in row["rollouts"]:
                text += [f"## Rollout {run['rollout_index']}\n",
                         f"- Reward: `{run['reward']}`; seed: `{run['seed']}`; elapsed: `{run['elapsed_seconds']:.1f}s`",
                         f"- Parsed final answer: `{run['response'].split('Final answer:')[-1].strip() if 'Final answer:' in run['response'] else '(missing)'}`",
                         f"- LOOK attempts / successful crop insertions / total crops: "
                         f"`{run['look_attempt_count']} / {run['look_count']} / {run['selected_crop_count']}`",
                         f"- WHEN probes: `{run['when_probe_count']}` ({run['when_probe_seconds']:.2f}s)",
                         f"- Attention modulation: `{run['effective_attention_modulation']}`\n",
                         "### LOOK events\n"]
                if run["look_events"]:
                    for event_index, event in enumerate(run["look_events"], 1):
                        boxes = "; ".join(_bbox(region) for region in event["regions"])
                        text.append(f"{event_index}. reasoning token {event['token']}: {boxes}")
                else:
                    text.append("No crop was inserted.")
                text += ["\n### Full generated reasoning\n", "```text", run["response"], "```\n"]
            (args.output_dir / name).write_text("\n".join(text), encoding="utf-8")
            answers = " / ".join(answer or "(missing)" for answer in summary["final_answers"])
            question = row["question"].replace("|", "\\|")
            index_rows.append(f"| {number} | {question} | {summary['rewards']} | {summary['look_attempt_counts']} | {answers} | [{name}]({name}) |")
    (args.output_dir / "INDEX.md").write_text("\n".join(index_rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
