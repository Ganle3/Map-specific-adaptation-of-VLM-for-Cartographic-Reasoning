#!/usr/bin/env python3
"""Export playable HTML viewers for correct cold-start ROI trajectories."""
from __future__ import annotations

import argparse
import base64
import html
import json
import mimetypes
from pathlib import Path

from export_correct_training_roi_viewer import PLAYER


def _image_data(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.input_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    links = []
    for row in rows:
        image = Path(row["image"])
        correct = []
        for run in row["rollouts"]:
            if float(run.get("reward", 0.0)) != 1.0 or not run.get("look_events"):
                continue
            looks = [{"token": int(event["token"]),
                      "prefix": event.get("visible_trace_before_look", ""),
                      "prefix_source": "recorded token boundary" if "visible_trace_before_look" in event else "unavailable in legacy cold-start record",
                      "regions": event["regions"],
                      "roi_tokens": int(event.get("roi_visual_tokens", 0)),
                      "bias": float(event.get("roi_key_bias", 0.0))}
                     for event in run["look_events"]]
            correct.append({"epoch": "cold-start", "step": "—", "rollout": int(run["rollout_index"]),
                            "response": run["response"], "answer": run.get("parsed_answer", ""),
                            "budget": int(run.get("generation_budget_tokens", 0)),
                            "eos": bool(run.get("generation_terminated_by_eos", False)), "looks": looks})
        if not correct:
            continue
        index = int(row["dataset_index"])
        payload = {"question": {"index": index, "question": row["question"], "answer": row["correct_answer"]},
                   "image": _image_data(image), "runs": correct}
        name = f"question_{index:04d}_correct.html"
        args.output_dir.joinpath(name).write_text(
            PLAYER.replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")), encoding="utf-8")
        links.append((name, row, len(correct)))

    page = ["<!doctype html><meta charset=utf-8><style>body{font:16px system-ui;margin:32px;max-width:1100px}li{margin:12px 0}.meta{color:#566}</style>",
            "<h1>Correct cold-start ROI trajectories</h1>",
            f"<p>{sum(n for _, _, n in links)} correct rollouts with at least one LOOK.</p><ul>"]
    for name, row, count in links:
        page.append(f'<li><a href="{name}">Question {row["dataset_index"]}: {html.escape(row["question"])}</a> <span class="meta">— {count} correct rollouts</span></li>')
    page.append("</ul>")
    args.output_dir.joinpath("index.html").write_text("\n".join(page), encoding="utf-8")
    print(f"Exported {sum(n for _, _, n in links)} correct trajectories across {len(links)} questions to {args.output_dir}")


if __name__ == "__main__":
    main()
