#!/usr/bin/env python3
"""Train only nine region prototypes with group-relative final-answer reward."""
from __future__ import annotations

import argparse
import json
import sys
import math
from collections import Counter
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
for sibling in (ROOT / "SparseVLM_eliminateCONTEXT", ROOT / "Training_scripts" / "GRPO_lookbetter"):
    if str(sibling) not in sys.path:
        sys.path.insert(0, str(sibling))
from rolling_sparse_qwen3vl import load_model
from data import evaluate_answer, load_mapverse_questions
from live_rollout import prepare_crops, rollout
from region_grpo import region_grpo_loss
from region_policy import NineGridRegionPolicy, decoder_hidden_size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "Datasets" / "When_Where_Crop" / "visual_easier_1000.json")
    parser.add_argument("--mapverse-root", type=Path, required=True)
    parser.add_argument("--mapwise-root", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy-checkpoint", type=Path)
    parser.add_argument("--limit", type=int, default=None, help="Number of questions; all records by default.")
    parser.add_argument("--start-index", type=int, default=1, help="1-based inclusive index in the input JSON.")
    parser.add_argument("--end-index", type=int, default=None, help="1-based inclusive index in the input JSON.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--questions-per-update", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--look-every", type=int, default=64)
    parser.add_argument("--max-looks", type=int, default=3)
    parser.add_argument("--global-max-pixels", type=int, default=1_000_000)
    parser.add_argument("--crop-max-pixels", type=int, default=1_000_000)
    parser.add_argument("--crop-instruction", default="Continue the unfinished reasoning using this additional visual evidence.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--num-policy-updates", type=int, default=1,
                        help="PPO updates on each freshly generated rollout batch.")
    parser.add_argument("--off-policy-mask-threshold", type=float, default=None,
                        help="Discard an action in reused updates when |exp(logp-old_logp)-1| exceeds this value.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-after-epoch", action="store_true",
                        help="Run one deterministic region-policy rollout per selected question after each epoch.")
    parser.add_argument("--stop-at-eval-accuracy", type=float, default=None)
    parser.add_argument("--save-steps", type=int, default=1000,
                        help="Save a standalone region-policy checkpoint every N optimizer steps.")
    parser.add_argument("--save-every-epoch", action="store_true",
                        help="Also save an extra checkpoint at each epoch boundary.")
    parser.add_argument("--no-load-in-4bit", action="store_false", dest="load_in_4bit")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.set_defaults(load_in_4bit=True)
    args = parser.parse_args()
    if args.group_size < 2 and not args.eval_only:
        parser.error("Training requires --group-size >= 2")
    if (args.look_every <= 0 or args.max_looks < 0 or args.epochs <= 0 or args.start_index <= 0
            or (args.limit is not None and args.limit <= 0) or args.questions_per_update <= 0
            or args.num_policy_updates <= 0):
        parser.error("look-every and questions-per-update must be positive; max-looks must be nonnegative")
    if args.off_policy_mask_threshold is not None and args.off_policy_mask_threshold <= 0:
        parser.error("off-policy-mask-threshold must be positive when set")
    if args.save_steps <= 0:
        parser.error("save-steps must be positive")
    if args.end_index is not None and args.end_index < args.start_index:
        parser.error("end-index must be at least start-index")
    if args.stop_at_eval_accuracy is not None and not (0 < args.stop_at_eval_accuracy <= 1):
        parser.error("stop-at-eval-accuracy must be in (0, 1]")
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, processor = load_model(args.model, args.adapter, load_in_4bit=args.load_in_4bit)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    device = model.get_input_embeddings().weight.device
    policy = NineGridRegionPolicy(decoder_hidden_size(model), args.temperature).to(device)
    if args.policy_checkpoint:
        saved = torch.load(args.policy_checkpoint, map_location="cpu", weights_only=True)
        policy.load_state_dict(saved["policy"])
    optimizer = torch.optim.AdamW([policy.nine_blocks], lr=args.learning_rate, weight_decay=0.0)
    assert any(p is policy.nine_blocks for group in optimizer.param_groups for p in group["params"])
    original_blocks = policy.nine_blocks.detach().clone()
    all_records = load_mapverse_questions(args.dataset, args.mapverse_root,
                                          mapwise_root=args.mapwise_root, limit=args.end_index)
    records = all_records[args.start_index - 1:args.end_index]
    if args.limit is not None:
        records = records[:args.limit]
    if not records:
        parser.error("Selected question range is empty")
    wandb_run = None
    if args.wandb_project and args.wandb_mode != "disabled":
        import wandb
        wandb_config = {key: str(value) if isinstance(value, Path) else value
                        for key, value in vars(args).items()}
        wandb_run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                               name=args.wandb_run_name, mode=args.wandb_mode, config=wandb_config)
    trace_path = args.output_dir / "rollouts.jsonl"
    steps_per_epoch = math.ceil(len(records) / args.questions_per_update)
    last_completed_step = -1
    last_completed_epoch = 0

    def save_checkpoint(path: Path, step: int, epoch: int) -> None:
        torch.save({"policy": policy.state_dict(), "optimizer": optimizer.state_dict(),
                    "hidden_size": policy.hidden_size, "temperature": policy.temperature,
                    "step": step, "epoch": epoch}, path)

    with trace_path.open("w", encoding="utf-8") as trace:
        def evaluate_selected(epoch_number: int) -> dict[str, float]:
            eval_metrics = {}
            eval_rewards = []
            for offset, item in enumerate(records):
                with Image.open(item.image_path) as file:
                    eval_image = file.convert("RGB")
                eval_crops = prepare_crops(model, processor, eval_image, device, args.crop_max_pixels,
                                           args.crop_instruction)
                eval_row = rollout(model, processor, policy, item, eval_crops, rollout_id=0,
                                   global_max_pixels=args.global_max_pixels,
                                   max_new_tokens=args.max_new_tokens, look_every=args.look_every,
                                   max_looks=args.max_looks, sample=False)
                eval_result = evaluate_answer(eval_row["response"], item)
                eval_row.update({"reward": eval_result["reward"], "evaluation": eval_result,
                                 "epoch": epoch_number, "dataset_index": args.start_index + offset,
                                 "phase": "eval"})
                for event in eval_row["diagnostics"]:
                    event["reward"] = eval_row["reward"]
                    event["answer_correct"] = bool(eval_row["reward"])
                trace.write(json.dumps({key: value for key, value in eval_row.items() if key != "actions"},
                                       ensure_ascii=False) + "\n")
                trace.flush()
                eval_rewards.append(eval_row["reward"])
                eval_metrics[f"eval/reward_{args.start_index + offset}"] = eval_row["reward"]
                del eval_crops
            eval_metrics["eval/accuracy"] = sum(eval_rewards) / len(eval_rewards)
            return eval_metrics

        if args.eval_after_epoch and not args.eval_only:
            baseline_metrics = evaluate_selected(0)
            print(json.dumps({"step": -1, "epoch": 0, **baseline_metrics}), flush=True)
            if wandb_run:
                wandb_run.log(baseline_metrics, step=0)

        for step in range(steps_per_epoch * args.epochs):
            epoch = step // steps_per_epoch
            start = (step % steps_per_epoch) * args.questions_per_update
            batch = []
            for offset, item in enumerate(records[start:start + args.questions_per_update]):
                with Image.open(item.image_path) as file:
                    original = file.convert("RGB")
                prepared = prepare_crops(model, processor, original, device, args.crop_max_pixels,
                                         args.crop_instruction)
                group = []
                for rollout_id in range(1 if args.eval_only else args.group_size):
                    row = rollout(model, processor, policy, item, prepared, rollout_id=rollout_id,
                                  global_max_pixels=args.global_max_pixels, max_new_tokens=args.max_new_tokens,
                                  look_every=args.look_every, max_looks=args.max_looks, sample=not args.eval_only)
                    evaluation = evaluate_answer(row["response"], item)
                    row["reward"] = evaluation["reward"]
                    row["evaluation"] = evaluation
                    row["epoch"] = epoch + 1
                    row["dataset_index"] = args.start_index + start + offset
                    row["phase"] = "eval" if args.eval_only else "train"
                    for event in row["diagnostics"]:
                        event["reward"] = row["reward"]
                        event["answer_correct"] = bool(row["reward"])
                    trace.write(json.dumps({key: value for key, value in row.items() if key != "actions"},
                                           ensure_ascii=False) + "\n")
                    trace.flush()
                    group.append(row)
                del prepared
                batch.append((item, group))
            results, policy_losses, gradient_norms = [], [], []
            for _ in range(1 if args.eval_only else args.num_policy_updates):
                results = [region_grpo_loss(policy, group, args.clip_ratio, args.off_policy_mask_threshold)
                           if not args.eval_only else (None, {}) for _, group in batch]
                losses = [item_loss for item_loss, _ in results if item_loss is not None]
                if not losses:
                    break
                loss = torch.stack(losses).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norms.append(float(policy.nine_blocks.grad.norm()))
                optimizer.step()
                policy_losses.append(float(loss.detach()))
            loss = policy_losses[-1] if policy_losses else None
            gradient_norm = gradient_norms[-1] if gradient_norms else 0.0
            rows = [row for _, group in batch for row in group]
            frequency = Counter(event["selected_action"] for row in rows for event in row["diagnostics"])
            metrics = {"train/region_policy_loss": loss if loss is not None else 0.0,
                       "train/action_entropy": sum(info.get("action_entropy", 0.0) for _, info in results) / len(results),
                       "train/gradient_norm": gradient_norm,
                       "train/learning_rate": float(optimizer.param_groups[0]["lr"]),
                       "train/parameter_norm": float(policy.nine_blocks.detach().norm()),
                       "train/parameter_delta_norm": float((policy.nine_blocks.detach() - original_blocks).norm()),
                       "train/reward_mean": sum(row["reward"] for row in rows) / len(rows),
                       "train/active_rollouts": sum(info.get("active_rollouts", 0) for _, info in results),
                       "train/reward_std": sum(info.get("reward_std", 0.0) for _, info in results) / len(results),
                       "train/mixed_reward_question_fraction": sum(info.get("reward_std", 0.0) > 0
                                                                 for _, info in results) / len(results),
                       "train/policy_updates_completed": len(policy_losses),
                       "train/off_policy_mask_fraction": sum(info.get("off_policy_mask_fraction", 0.0)
                                                                for _, info in results) / len(results),
                       "train/mean_abs_action_ratio_drift": sum(info.get("mean_abs_ratio_drift", 0.0)
                                                                  for _, info in results) / len(results),
                       "train/active_region_actions": sum(info.get("active_action_count", 0) for _, info in results),
                       "train/crop_append_count": sum(row["crop_append_count"] for row in rows),
                       "train/questions_per_update": len(batch), "train/rollouts_per_update": len(rows)}
            if torch.cuda.is_available():
                metrics["system/peak_memory_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
                metrics["system/peak_memory_reserved_gib"] = torch.cuda.max_memory_reserved(device) / 2**30
                torch.cuda.reset_peak_memory_stats(device)
            action_total = sum(frequency.values())
            metrics.update({f"train/action_frequency_{i}": frequency[i] / action_total if action_total else 0.0
                            for i in range(9)})
            if args.eval_after_epoch and not args.eval_only and (step + 1) % steps_per_epoch == 0:
                metrics.update(evaluate_selected(epoch + 1))
            print(json.dumps({"step": step, "epoch": epoch + 1,
                              "dataset_indices": list(range(args.start_index + start,
                                                             args.start_index + start + len(batch))),
                              "question_ids": [item.sample_id for item, _ in batch],
                              **metrics}), flush=True)
            if wandb_run:
                wandb_run.log(metrics, step=step + 1)
            last_completed_step = step
            last_completed_epoch = epoch + 1
            if (step + 1) % args.save_steps == 0:
                save_checkpoint(args.output_dir / f"region_policy_step_{step + 1:06d}.pt", step, epoch + 1)
            if args.save_every_epoch and (step + 1) % steps_per_epoch == 0:
                save_checkpoint(args.output_dir / f"region_policy_epoch_{epoch + 1:03d}.pt", step, epoch + 1)
            if (args.stop_at_eval_accuracy is not None
                    and metrics.get("eval/accuracy", -1.0) >= args.stop_at_eval_accuracy):
                print(f"Reached evaluation accuracy {metrics['eval/accuracy']:.3f}; stopping.", flush=True)
                break
    if last_completed_step >= 0:
        save_checkpoint(args.output_dir / "region_policy_final.pt", last_completed_step, last_completed_epoch)
    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
