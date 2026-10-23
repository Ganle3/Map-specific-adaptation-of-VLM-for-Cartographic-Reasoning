#!/usr/bin/env python3
"""Train region prototypes while reusing one full-image prefill per rollout group."""
from __future__ import annotations

import argparse
import json
import sys
import math
import random
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
from live_rollout_cached_prefill import prepare_crops, prepare_global_prefill, rollout
from region_grpo import (capture_policy_update_diagnostics, finalize_policy_update_diagnostics,
                         region_grpo_loss)
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
    parser.add_argument("--resume-from-checkpoint", type=Path,
                        help="Resume policy, AdamW state, epoch and optimizer-step count; --epochs means additional epochs.")
    parser.add_argument("--limit", type=int, default=None, help="Number of questions; all records by default.")
    parser.add_argument("--start-index", type=int, default=1, help="1-based inclusive index in the input JSON.")
    parser.add_argument("--end-index", type=int, default=None, help="1-based inclusive index in the input JSON.")
    parser.add_argument("--exclude-indices", type=int, nargs="*", default=[],
                        help="1-based input rows to exclude after start/end selection.")
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
    parser.add_argument("--policy-hidden-layer", type=int, default=-1,
                        help="-1: final normalized decoder state; i>=0: Transformers hidden_states[i] at each decision.")
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--num-policy-updates", type=int, default=1,
                        help="PPO updates on each freshly generated rollout batch.")
    parser.add_argument("--off-policy-mask-threshold", type=float, default=None,
                        help="Discard an action in reused updates when |exp(logp-old_logp)-1| exceeds this value.")
    parser.add_argument("--print-policy-update-diagnostics", action="store_true",
                        help="Print reward/advantage, per-prototype gradient and parameter-update evidence after each policy update.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--preserve-question-order", action="store_true",
                        help="Disable the default reproducible per-epoch question shuffle.")
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
    if args.policy_checkpoint and args.resume_from_checkpoint:
        parser.error("Use either --policy-checkpoint or --resume-from-checkpoint, not both")
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
    resume_state = None
    if args.resume_from_checkpoint:
        resume_state = torch.load(args.resume_from_checkpoint, map_location="cpu", weights_only=True)
        if "optimizer" not in resume_state:
            parser.error("Resume checkpoint lacks optimizer state; use --policy-checkpoint for weights-only loading")
        policy.load_state_dict(resume_state["policy"])
        optimizer.load_state_dict(resume_state["optimizer"])
    assert any(p is policy.nine_blocks for group in optimizer.param_groups for p in group["params"])
    original_blocks = policy.nine_blocks.detach().clone()
    all_records = load_mapverse_questions(args.dataset, args.mapverse_root,
                                          mapwise_root=args.mapwise_root, limit=args.end_index)
    records = all_records[args.start_index - 1:args.end_index]
    if args.limit is not None:
        records = records[:args.limit]
    indexed_records = [(index, record) for index, record in
                       enumerate(records, start=args.start_index) if index not in set(args.exclude_indices)]
    records = [record for _, record in indexed_records]
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
    resumed_epoch = int(resume_state["epoch"]) if resume_state is not None else 0
    resumed_step = int(resume_state["step"]) if resume_state is not None else -1
    if resume_state is not None and resumed_step + 1 != resumed_epoch * steps_per_epoch:
        parser.error("Resume checkpoint is not at an epoch boundary for the selected data/batch configuration")
    last_completed_step = resumed_step
    last_completed_epoch = resumed_epoch

    def save_checkpoint(path: Path, step: int, epoch: int) -> None:
        torch.save({"policy": policy.state_dict(), "optimizer": optimizer.state_dict(),
                    "hidden_size": policy.hidden_size, "temperature": policy.temperature,
                    "step": step, "epoch": epoch}, path)

    with trace_path.open("a" if resume_state is not None else "w", encoding="utf-8") as trace:
        def evaluate_selected(epoch_number: int) -> dict[str, float]:
            eval_metrics = {}
            eval_rewards = []
            for dataset_index, item in indexed_records:
                with Image.open(item.image_path) as file:
                    eval_image = file.convert("RGB")
                eval_crops = prepare_crops(model, processor, eval_image, device, args.crop_max_pixels,
                                           args.crop_instruction)
                eval_row = rollout(model, processor, policy, item, eval_crops, rollout_id=0,
                                   global_max_pixels=args.global_max_pixels,
                                   max_new_tokens=args.max_new_tokens, look_every=args.look_every,
                                   max_looks=args.max_looks, sample=False,
                                   policy_hidden_layer=args.policy_hidden_layer)
                eval_result = evaluate_answer(eval_row["response"], item)
                eval_row.update({"reward": eval_result["reward"], "evaluation": eval_result,
                                 "epoch": epoch_number, "dataset_index": dataset_index,
                                 "phase": "eval"})
                for event in eval_row["diagnostics"]:
                    event["reward"] = eval_row["reward"]
                    event["answer_correct"] = bool(eval_row["reward"])
                trace.write(json.dumps({key: value for key, value in eval_row.items() if key != "actions"},
                                       ensure_ascii=False) + "\n")
                trace.flush()
                eval_rewards.append(eval_row["reward"])
                eval_metrics[f"eval/reward_{dataset_index}"] = eval_row["reward"]
                del eval_crops
            eval_metrics["eval/accuracy"] = sum(eval_rewards) / len(eval_rewards)
            return eval_metrics

        if args.eval_after_epoch and not args.eval_only and resume_state is None:
            baseline_metrics = evaluate_selected(0)
            print(json.dumps({"step": -1, "epoch": 0, **baseline_metrics}), flush=True)
            if wandb_run:
                wandb_run.log(baseline_metrics, step=0)

        for local_step in range(steps_per_epoch * args.epochs):
            step = resumed_step + 1 + local_step
            epoch = resumed_epoch + local_step // steps_per_epoch
            start = (local_step % steps_per_epoch) * args.questions_per_update
            epoch_records = list(indexed_records)
            if not args.preserve_question_order:
                random.Random(args.seed + epoch).shuffle(epoch_records)
            batch = []
            for dataset_index, item in epoch_records[start:start + args.questions_per_update]:
                with Image.open(item.image_path) as file:
                    original = file.convert("RGB")
                prepared = prepare_crops(model, processor, original, device, args.crop_max_pixels,
                                         args.crop_instruction)
                prefill = prepare_global_prefill(model, processor, item,
                                                 global_max_pixels=args.global_max_pixels,
                                                 policy_hidden_layer=args.policy_hidden_layer)
                group = []
                for rollout_id in range(1 if args.eval_only else args.group_size):
                    row = rollout(model, processor, policy, item, prepared, rollout_id=rollout_id,
                                  global_max_pixels=args.global_max_pixels, max_new_tokens=args.max_new_tokens,
                                  look_every=args.look_every, max_looks=args.max_looks, sample=not args.eval_only,
                                  prefill=prefill, policy_hidden_layer=args.policy_hidden_layer)
                    evaluation = evaluate_answer(row["response"], item)
                    row["reward"] = evaluation["reward"]
                    row["evaluation"] = evaluation
                    row["epoch"] = epoch + 1
                    row["dataset_index"] = dataset_index
                    row["phase"] = "eval" if args.eval_only else "train"
                    for event in row["diagnostics"]:
                        event["reward"] = row["reward"]
                        event["answer_correct"] = bool(row["reward"])
                    trace.write(json.dumps({key: value for key, value in row.items() if key != "actions"},
                                           ensure_ascii=False) + "\n")
                    trace.flush()
                    group.append(row)
                del prepared, prefill
                batch.append((dataset_index, item, group))
            results, policy_losses, gradient_norms = [], [], []
            for policy_update_index in range(1 if args.eval_only else args.num_policy_updates):
                snapshot = (capture_policy_update_diagnostics(policy, batch)
                            if args.print_policy_update_diagnostics and not args.eval_only else None)
                blocks_before_update = policy.nine_blocks.detach().clone() if snapshot is not None else None
                results = [region_grpo_loss(policy, group, args.clip_ratio, args.off_policy_mask_threshold)
                           if not args.eval_only else (None, {}) for _, _, group in batch]
                losses = [item_loss for item_loss, _ in results if item_loss is not None]
                if not losses:
                    break
                loss = torch.stack(losses).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norms.append(float(policy.nine_blocks.grad.norm()))
                per_prototype_gradient_norms = (policy.nine_blocks.grad.detach().norm(dim=1).cpu()
                                                if snapshot is not None else None)
                optimizer.step()
                if snapshot is not None:
                    diagnostic = finalize_policy_update_diagnostics(
                        policy, snapshot, per_prototype_gradient_norms,
                        (policy.nine_blocks.detach() - blocks_before_update).norm(dim=1).cpu())
                    diagnostic.update({"outer_step": step + 1, "epoch": epoch + 1,
                                       "policy_update": policy_update_index + 1,
                                       "region_policy_loss": float(loss.detach())})
                    print(f"[POLICY_UPDATE {step + 1:05d}.{policy_update_index + 1}] "
                          f"{json.dumps(diagnostic, ensure_ascii=False)}", flush=True)
                policy_losses.append(float(loss.detach()))
            loss = policy_losses[-1] if policy_losses else None
            gradient_norm = gradient_norms[-1] if gradient_norms else 0.0
            rows = [row for _, _, group in batch for row in group]
            reward_groups = []
            for dataset_index, item, group in batch:
                rewards = [float(row["reward"]) for row in group]
                correct_count = int(sum(rewards))
                payload = {"dataset_index": dataset_index, "qa_id": item.sample_id,
                           "group_valid": len(group) == args.group_size,
                           "rewards": rewards, "correct_count": correct_count,
                           "incorrect_count": len(group) - correct_count,
                           "reward_variance": len(set(rewards)) > 1,
                           "untruncated_count": sum(not row.get("truncated", False) for row in group),
                           "truncated_count": sum(row.get("truncated", False) for row in group),
                           "signal_group": len(set(rewards)) > 1}
                reward_groups.append(payload)
                print(f"[REWARD {step + 1:05d}] {json.dumps(payload, ensure_ascii=False)}", flush=True)
            events = [event for row in rows for event in row["diagnostics"]]
            frequency = Counter(event["selected_action"] for event in events)

            def behavior_metrics(prefix: str, subset: list[dict]) -> dict[str, float]:
                """Separate state-conditional sharpness from global action collapse."""
                if not subset:
                    return {f"{prefix}/decision_count": 0.0,
                            f"{prefix}/conditional_entropy": 0.0,
                            f"{prefix}/marginal_entropy": 0.0,
                            f"{prefix}/state_mutual_information": 0.0,
                            f"{prefix}/mean_probability_std": 0.0}
                probabilities = torch.tensor([event["probabilities"] for event in subset], dtype=torch.float64)
                conditional_entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1).mean()
                marginal = probabilities.mean(dim=0)
                marginal_entropy = -(marginal * marginal.clamp_min(1e-12).log()).sum()
                return {f"{prefix}/decision_count": float(len(subset)),
                        f"{prefix}/conditional_entropy": float(conditional_entropy),
                        f"{prefix}/marginal_entropy": float(marginal_entropy),
                        # H(E[p(a|h)]) - E[H(p(a|h))] estimates I(action; state).
                        f"{prefix}/state_mutual_information": float(marginal_entropy - conditional_entropy),
                        f"{prefix}/mean_probability_std": float(probabilities.std(dim=0, unbiased=False).mean())}

            behavior = behavior_metrics("train/behavior", events)
            if events:
                first_index = min(event["reasoning_token_index"] for event in events)
                behavior.update(behavior_metrics("train/first_look", [event for event in events
                                                                         if event["reasoning_token_index"] == first_index]))
            else:
                behavior.update(behavior_metrics("train/first_look", []))
            for look_slot in range(1, args.max_looks + 1):
                behavior.update(behavior_metrics(f"train/look_{look_slot}",
                                                 [event for event in events
                                                  if event["reasoning_token_index"] == look_slot * args.look_every]))
            metrics = {"train/region_policy_loss": loss if loss is not None else 0.0,
                       "train/action_entropy": sum(info.get("action_entropy", 0.0) for _, info in results) / len(results),
                       "train/gradient_norm": gradient_norm,
                       "train/learning_rate": float(optimizer.param_groups[0]["lr"]),
                       "train/parameter_norm": float(policy.nine_blocks.detach().norm()),
                       "train/parameter_delta_norm": float((policy.nine_blocks.detach() - original_blocks).norm()),
                       "train/reward_mean": sum(row["reward"] for row in rows) / len(rows),
                       "train/rollouts_correct": sum(row["reward"] for row in rows),
                       "train/rollouts_incorrect": sum(1 - row["reward"] for row in rows),
                       "train/questions_all_correct": sum(group["correct_count"] == len(group["rewards"])
                                                           for group in reward_groups),
                       "train/questions_all_incorrect": sum(group["correct_count"] == 0
                                                             for group in reward_groups),
                       "train/questions_mixed_correctness": sum(group["signal_group"] for group in reward_groups),
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
                       "train/global_prefill_reuse_rollouts": sum(row["global_prefill_reused"] for row in rows),
                       "train/questions_per_update": len(batch), "train/rollouts_per_update": len(rows)}
            metrics.update(behavior)
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
                              "dataset_indices": [dataset_index for dataset_index, _, _ in batch],
                              "question_ids": [item.sample_id for _, item, _ in batch],
                              **metrics}), flush=True)
            if wandb_run:
                reward_table = wandb.Table(
                    columns=["dataset_index", "qa_id", "rewards", "correct_count", "incorrect_count",
                             "reward_variance", "untruncated_count", "truncated_count", "signal_group"],
                    data=[[group[key] if key != "rewards" else json.dumps(group[key])
                           for key in ("dataset_index", "qa_id", "rewards", "correct_count", "incorrect_count",
                                       "reward_variance", "untruncated_count", "truncated_count", "signal_group")]
                          for group in reward_groups])
                wandb_run.log({**metrics, "train/reward_groups": reward_table}, step=step + 1)
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
