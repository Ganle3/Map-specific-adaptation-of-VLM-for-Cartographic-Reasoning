#!/usr/bin/env python3
"""Render original-image crop boxes recorded by a cold-start JSONL rollout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PALETTE = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990"]


def _label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, color: str, font) -> None:
    box = draw.textbbox(xy, text, font=font)
    draw.rectangle((box[0] - 2, box[1] - 1, box[2] + 2, box[3] + 1), fill="white", outline=color, width=1)
    draw.text(xy, text, fill="black", font=font)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--dataset-index", type=int, required=True, help="1-based index recorded in cold-start JSONL.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    row = None
    for line in args.input_jsonl.read_text(encoding="utf-8").splitlines():
        candidate = json.loads(line)
        if int(candidate["dataset_index"]) == args.dataset_index:
            row = candidate
            break
    if row is None:
        raise ValueError(f"dataset index {args.dataset_index} is absent from {args.input_jsonl}")

    source = Path(row["image"])
    if not source.is_file():
        raise FileNotFoundError(source)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    original = Image.open(source).convert("RGB")
    font = ImageFont.load_default()
    lines = [f"# ROI visualization: question {row['dataset_index']}", "",
             f"- Image: `{source}`", f"- Question: {row['question']}", f"- Gold: `{row['correct_answer']}`", "",
             "Each color denotes one LOOK event. Rectangle labels have the form `LOOK.crop`.",
             "The individual `crop_*.png` files are original-resolution renderings of the selected ROI.",
             "In the current ROI-KV HOW these files are diagnostic only: no crop is re-encoded or sent through ViT;",
             "the matching global visual-token keys receive the recorded attention-logit bias.", ""]

    for run in row["rollouts"]:
        rollout_dir = args.output_dir / f"rollout_{run['rollout_index']:02d}"
        rollout_dir.mkdir(exist_ok=True)
        all_overlay = original.convert("RGBA")
        all_draw = ImageDraw.Draw(all_overlay)
        lines += [f"## Rollout {run['rollout_index']}", "",
                  f"- Final answer: `{run['response'].split('Final answer:')[-1].strip() if 'Final answer:' in run['response'] else '(missing)'}`",
                  f"- LOOK events: {len(run['look_events'])}", ""]
        for look_number, event in enumerate(run["look_events"], 1):
            color = PALETTE[(look_number - 1) % len(PALETTE)]
            per_look = original.convert("RGBA")
            draw = ImageDraw.Draw(per_look)
            for crop_number, region in enumerate(event["regions"], 1):
                box = (int(region["left"]), int(region["top"]), int(region["right"]), int(region["bottom"]))
                for target in (draw, all_draw):
                    target.rectangle(box, outline=color, width=4)
                    _label(target, (box[0] + 3, box[1] + 3), f"{look_number}.{crop_number}", color, font)
                original.crop(box).save(rollout_dir / f"look_{look_number:02d}_token_{event['token']:03d}_crop_{crop_number:02d}.png")
            per_look.convert("RGB").save(rollout_dir / f"look_{look_number:02d}_token_{event['token']:03d}_overlay.png")
            boxes = "; ".join(
                f"{look_number}.{crop_number}=({region['left']},{region['top']})-({region['right']},{region['bottom']}), score={region['score']:.4g}"
                for crop_number, region in enumerate(event["regions"], 1)
            )
            lines.append(f"- LOOK {look_number}, reasoning token {event['token']}, color `{color}`: {boxes}")
        all_overlay.convert("RGB").save(rollout_dir / "all_looks_overlay.png")
        lines += [f"- [All LOOK overlays](rollout_{run['rollout_index']:02d}/all_looks_overlay.png)", ""]
    (args.output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
