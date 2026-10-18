#!/usr/bin/env python3
"""Run 10 x 4 frozen-backbone ROI rollouts on visual_hard_1000.

This is deliberately a rollout-only smoke test: it validates dynamic actions,
attention-hook execution, cache progression, and reward calculation before any
GRPO optimiser update is attempted.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

from .data import evaluate_answer, extract_final_answer, load_mapverse_questions
from .roi_policy import DynamicROIPolicy, ROIPolicyConfig
from .rollout import RolloutConfig, collect_rollout
from .spatial_mask import SoftROIMask, VisualTokenGrid
from .train_roi_grpo import build_inputs


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = PROJECT_ROOT / "VLM_adaptation" / "Datasets" / "When_Where_Crop" / "visual_easier_1000.json"
DEFAULT_MAPVERSE_IMAGES = PROJECT_ROOT / "MapVerse" / "data" / "imgs"
DEFAULT_MAPWISE = PROJECT_ROOT / "VLM_adaptation" / "Datasets" / "mapwise-dataset"


def _json_safe(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Base model; defaults through existing loader to Qwen/Qwen3-VL-8B-Thinking.")
    parser.add_argument("--adapter", type=Path, default=None, help="Frozen existing reasoning LoRA, if applicable.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--mapverse-image-root", type=Path, default=DEFAULT_MAPVERSE_IMAGES)
    parser.add_argument("--mapwise-root", type=Path, default=DEFAULT_MAPWISE)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "VLM_adaptation" / "Training_outputs" / "GRPO_ROI_visual_hard_smoke")
    parser.add_argument("--max-new-tokens", type=int, default=1024,
                        help="Thinking models often need >=512 tokens to emit Final answer.")
    parser.add_argument("--questions", type=int, default=10, help="Use 20 for the required pretraining exploration diagnostic.")
    parser.add_argument("--rollouts", type=int, default=4, help="Use 4-8 stochastic ROI rollouts per question.")
    parser.add_argument("--attention-prior-alpha", type=float, default=.5)
    parser.add_argument("--print-reasoning", action=argparse.BooleanOptionalAction, default=True,
                        help="Print each complete generated reasoning trace to the terminal.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This Qwen3-VL smoke test requires a CUDA GPU; use smoke_test.py for CPU-only policy checks.")
    torch.manual_seed(args.seed)
    sys.path.insert(0, str(PROJECT_ROOT / "VLM_adaptation" / "SparseVLM_eliminateCONTEXT"))
    from rolling_sparse_qwen3vl import load_model
    model, processor = load_model(args.model, args.adapter)
    device = model.get_input_embeddings().weight.device
    hidden_size = model.config.text_config.hidden_size
    policy = DynamicROIPolicy(ROIPolicyConfig(hidden_size=hidden_size)).to(device).eval()
    if args.questions < 1 or args.rollouts < 2:
        raise ValueError("--questions must be positive and --rollouts must be at least 2.")
    questions = load_mapverse_questions(args.dataset, args.mapverse_image_root, mapwise_root=args.mapwise_root, limit=args.questions)
    args.output.mkdir(parents=True, exist_ok=True)
    rows, groups = [], []
    with ((args.output / "rollouts.jsonl").open("w", encoding="utf-8") as stream,
          (args.output / "readable_rollouts.md").open("w", encoding="utf-8") as readable):
        readable.write("# Dynamic ROI smoke-test rollouts\n\n")
        for question_index, item in enumerate(questions):
            with Image.open(item.image_path) as source:
                image = source.convert("RGB")
            inputs = build_inputs(processor, image, item.question, device)
            _time, raw_height, raw_width = inputs["image_grid_thw"][0].tolist()
            merge = model.config.vision_config.spatial_merge_size
            mask = SoftROIMask(VisualTokenGrid(raw_height // merge, raw_width // merge), max_bias=policy.cfg.max_bias)
            group = []
            readable.write(f"## Question {question_index + 1}: `{item.sample_id}`\n\n"
                           f"**Question:** {item.question}\n\n"
                           f"**Ground truth:** {item.answer}\n\n")
            for rollout_index in range(args.rollouts):
                evaluation_holder = {}
                def scored_reward(text, sample=item):
                    evaluation = evaluate_answer(text, sample)
                    evaluation_holder["value"] = evaluation
                    return evaluation["reward"]
                rollout = collect_rollout(model, processor, inputs, policy, mask, scored_reward,
                                          RolloutConfig(max_new_tokens=args.max_new_tokens,
                                                        eos_token_id=processor.tokenizer.eos_token_id,
                                                        attention_prior_alpha=args.attention_prior_alpha))
                evaluation = evaluation_holder["value"]
                row = {"question_index": question_index, "rollout_index": rollout_index,
                       "sample_id": item.sample_id, "dataset": item.metadata.get("dataset"),
                       "answer_type": item.answer_type, "ability_level": item.metadata.get("ability_level"),
                       "question": item.question, "ground_truth": item.answer,
                       "parsed_final_answer": extract_final_answer(rollout["text"]),
                       "evaluation": evaluation, **rollout}
                stream.write(json.dumps(row, default=_json_safe) + "\n"); stream.flush()
                rows.append(row); group.append(row)
                print(f"[{question_index + 1}/{len(questions)}, rollout {rollout_index + 1}/{args.rollouts}] reward={row['reward']:.0f} {item.sample_id}", flush=True)
                readable.write(f"### Rollout {rollout_index + 1} — reward: {row['reward']:.0f}\n\n"
                               f"- Parsed final answer: `{row['parsed_final_answer'] or '(missing)'}`\n"
                               f"- Evaluator: `{evaluation['evaluator']}` / `{evaluation['metric']}`\n"
                               f"- Canonical prediction / gold: `{evaluation['canonical_prediction']}` / `{evaluation['canonical_target']}`\n\n"
                               f"```text\n{row['text'].strip()}\n```\n\n")
                readable.flush()
                if args.print_reasoning:
                    print("----- generated reasoning -----")
                    print(row["text"].strip())
                    print("-------------------------------", flush=True)
            groups.append(group)
    strengths = [float(value) for row in rows for action in row["actions"] for value in action["strengths"].flatten()]
    masks = [torch.as_tensor(action["geometry"])[..., :2].flatten() for row in rows for action in row["actions"]]
    diversity = [float((masks[i] - masks[i - 1]).abs().mean()) for i in range(1, len(masks))]
    attention_shifts = [float((torch.as_tensor(actions[i]["attention_summary_after"]) - torch.as_tensor(actions[i - 1]["attention_summary_after"])).abs().mean())
                        for row in rows for actions in [row["actions"]] for i in range(1, len(actions))]
    answer_groups = [{row["text"].strip() for row in group} for group in groups]
    summary = {"dataset": str(args.dataset), "questions": len(groups), "rollouts": len(rows),
               "num_generations": args.rollouts, "mean_reward": sum(x["reward"] for x in rows) / len(rows),
               "groups_with_reward_variance": sum(len({x["reward"] for x in group}) > 1 for group in groups),
               "fraction_groups_reward_variance": sum(len({x["reward"] for x in group}) > 1 for group in groups) / len(groups),
               "fraction_groups_different_final_answers": sum(len(x) > 1 for x in answer_groups) / len(groups),
               "roi_strength": {"min": min(strengths), "mean": sum(strengths)/len(strengths), "max": max(strengths)},
               "mean_consecutive_roi_center_difference": sum(diversity)/max(1, len(diversity)),
               "fraction_actions_nonzero_bias": sum(action["bias_max"] > 1e-6 for row in rows for action in row["actions"]) / max(1, sum(len(row["actions"]) for row in rows)),
               "mean_visual_attention_step_shift": sum(attention_shifts)/max(1, len(attention_shifts)),
               "mean_rollout_tokens": sum(len(x["tokens"]) for x in rows) / len(rows),
               "mean_seconds": sum(x["elapsed_seconds"] for x in rows) / len(rows)}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
