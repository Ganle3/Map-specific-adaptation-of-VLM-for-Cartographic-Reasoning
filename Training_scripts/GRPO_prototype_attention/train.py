#!/usr/bin/env python3
"""Prototype-cross-attention prefix training with language-token GRPO."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch
from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
VLM_ROOT = THIS_DIR.parents[1]
THESIS_ROOT = VLM_ROOT.parent
for path in (VLM_ROOT / "SparseVLM_eliminateCONTEXT",
             VLM_ROOT / "Training_scripts" / "GRPO_lookbetter"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from continuous_cache_qwen3vl import _content_inputs, _resize_to_pixel_cap
from data import build_cartographic_prompt, evaluate_answer, load_mapverse_questions
from rolling_sparse_qwen3vl import load_model

from language_grpo import (clipped_language_grpo_loss, completion_log_probs,
                           group_advantages, off_policy_sequence_keep)
from prototype_cross_attention_policy import PrototypeCrossAttentionPolicy, decoder_hidden_size
from prototype_prefill import (build_prototype_prefill, encode_global_image,
                               encode_question_context, generate_from_preencoded_vision)
from question_conditioned_policy import QuestionConditionedPrototypePolicy


_RECOMPUTE_INPUT_KEYS = ("input_ids", "mm_token_type_ids", "image_grid_thw")


def _cache_recompute_inputs(source_inputs: dict) -> dict:
    """Keep only the small frozen prompt tensors needed by policy recomputation.

    In particular, do not retain ``pixel_values`` for every question in an
    optimizer batch after its ViT features and rollouts have been produced.
    """
    missing = [key for key in _RECOMPUTE_INPUT_KEYS if key not in source_inputs]
    if missing:
        raise KeyError(f"Processor inputs are missing recomputation keys: {missing}")
    return {
        key: source_inputs[key].detach().to("cpu", copy=True)
        for key in _RECOMPUTE_INPUT_KEYS
    }


def _move_recompute_inputs(cached_inputs: dict, device) -> dict:
    return {key: value.to(device, non_blocking=True) for key, value in cached_inputs.items()}


def _move_visual_features(features, device):
    """Move the detached ViT/merger cache without changing its nested shape."""
    def move_group(value):
        tensors = (value,) if isinstance(value, torch.Tensor) else tuple(value)
        return tuple(tensor.detach().to(device, non_blocking=True) for tensor in tensors)

    return type(features)(
        pooler_output=move_group(features.pooler_output),
        deepstack_features=[move_group(level) for level in features.deepstack_features],
    )


def _eos_ids(tokenizer) -> set[int]:
    value = tokenizer.eos_token_id
    return set(map(int, value)) if isinstance(value, (list, tuple, set)) else {int(value)}


def generate_rollout(model, processor, policy, item, source_inputs, global_features, placement: str,
                     max_new_tokens: int, rollout_id: int,
                     question_tokens: torch.Tensor | None = None) -> dict:
    """Sample language tokens; no gradient is retained during generation."""
    augmented, attention = build_prototype_prefill(
        model, processor.tokenizer, source_inputs, global_features, policy, placement,
        with_grad=False, question_tokens=question_tokens,
    )
    prefix_length = augmented["input_ids"].shape[1]
    with torch.no_grad():
        output = generate_from_preencoded_vision(
            model,
            **augmented,
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            use_cache=True,
        )
    completion = output.sequences[0, prefix_length:].detach()
    if completion.numel() != len(output.scores):
        raise RuntimeError("Generation scores do not align with sampled completion tokens")
    old_logps = torch.stack([
        torch.log_softmax(score[0].float(), dim=-1)[token]
        for score, token in zip(output.scores, completion)
    ]).detach().cpu()
    response = processor.tokenizer.decode(completion, skip_special_tokens=True)
    terminated = bool(completion.numel() and int(completion[-1]) in _eos_ids(processor.tokenizer))
    # Match mask_truncated_completions=True from the established GRPO runs.
    completion_mask = torch.ones(completion.numel(), dtype=torch.float32) if terminated else torch.zeros(
        completion.numel(), dtype=torch.float32
    )
    evaluation = evaluate_answer(response, item)
    weights = attention.detach().float().cpu()
    return {
        "rollout_id": rollout_id,
        "completion_ids": completion.cpu(),
        "old_logps": old_logps,
        "completion_mask": completion_mask,
        "response": response,
        "reward": float(evaluation["reward"]),
        "evaluation": evaluation,
        "terminated": terminated,
        "truncated": not terminated,
        "prototype_attention_entropy": float(
            -(weights.clamp_min(1e-12) * weights.clamp_min(1e-12).log()).sum(dim=-1).mean()
        ),
        "prototype_attention_max": float(weights.max(dim=-1).values.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--placement", required=True, choices=("after_image", "after_question"))
    parser.add_argument("--policy-kind", choices=("image_only", "question_conditioned"),
                        default="image_only")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--mapverse-root", type=Path, required=True)
    parser.add_argument("--mapwise-root", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--end-index", type=int)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-steps", type=int,
                        help="Optimizer-update budget. Overrides --epochs when set.")
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--questions-per-update", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--max-pixels", type=int, default=1_000_000)
    parser.add_argument("--num-prototypes", type=int, default=9)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--num-policy-updates", type=int, default=1,
                        help="Optimizer updates made from each freshly generated rollout batch.")
    parser.add_argument("--off-policy-mask-threshold", type=float,
                        help="Drop negative-advantage trajectories above this mean old/current KL.")
    parser.add_argument("--lr-scheduler-type", choices=("constant", "constant_then_cosine"),
                        default="constant")
    parser.add_argument("--decay-start-step", type=int)
    parser.add_argument("--decay-final-factor", type=float, default=0.6)
    parser.add_argument("--save-steps", type=int,
                        help="Save policy/optimizer/scheduler every N optimizer updates.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--save-every-epoch", action="store_true")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--no-load-in-4bit", action="store_false", dest="load_in_4bit")
    parser.set_defaults(load_in_4bit=True)
    args = parser.parse_args()
    if args.group_size < 2 or args.questions_per_update < 1 or args.epochs < 1:
        parser.error("group-size must be >=2; questions-per-update and epochs must be positive")
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("max-steps must be positive")
    if args.num_policy_updates < 1:
        parser.error("num-policy-updates must be positive")
    if args.save_steps is not None and args.save_steps < 1:
        parser.error("save-steps must be positive")
    if args.off_policy_mask_threshold is not None and args.off_policy_mask_threshold <= 0:
        parser.error("off-policy-mask-threshold must be positive")
    if not 0 < args.decay_final_factor <= 1:
        parser.error("decay-final-factor must be in (0, 1]")
    if args.lr_scheduler_type == "constant_then_cosine":
        if args.max_steps is None:
            parser.error("constant_then_cosine requires --max-steps")
        if args.decay_start_step is None or not 0 <= args.decay_start_step < args.max_steps:
            parser.error("constant_then_cosine requires 0 <= decay-start-step < max-steps")
    if args.end_index is not None and args.end_index < args.start_index:
        parser.error("end-index must be >= start-index")
    if math.isqrt(args.num_prototypes) ** 2 != args.num_prototypes:
        parser.error("num-prototypes must currently be a square count such as 9, 16, or 36")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, processor = load_model(args.model, args.adapter, load_in_4bit=args.load_in_4bit)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    device = model.get_input_embeddings().weight.device
    policy_class = (PrototypeCrossAttentionPolicy if args.policy_kind == "image_only"
                    else QuestionConditionedPrototypePolicy)
    policy = policy_class(decoder_hidden_size(model), num_prototypes=args.num_prototypes,
                          num_heads=args.attention_heads, train_projections=False).to(device)
    optimizer = torch.optim.AdamW(policy.trainable_parameters(), lr=args.learning_rate, weight_decay=0.0)
    optimized = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    assert len(optimized) == 1 and optimized[0] is policy.prototypes
    initial_prototypes = policy.prototypes.detach().clone()
    if args.lr_scheduler_type == "constant_then_cosine":
        def lr_factor(update: int) -> float:
            if update <= args.decay_start_step:
                return 1.0
            progress = min(1.0, (update - args.decay_start_step)
                           / max(1, args.max_steps - args.decay_start_step))
            return args.decay_final_factor + 0.5 * (1.0 - args.decay_final_factor) * (
                1.0 + math.cos(math.pi * progress)
            )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    all_records = load_mapverse_questions(
        args.dataset, args.mapverse_root, mapwise_root=args.mapwise_root, limit=args.end_index
    )
    records = list(enumerate(all_records[args.start_index - 1:args.end_index], start=args.start_index))
    if not records:
        parser.error("Selected dataset range is empty")

    wandb_run = None
    if args.wandb_project and args.wandb_mode != "disabled":
        import wandb
        config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_run_name,
                               mode=args.wandb_mode, config=config)
    def save_checkpoint(path: Path, *, step: int, epoch: int) -> None:
        # Q/K/V/O projections are deterministic frozen identity matrices.
        # Store only the sole learned tensor rather than duplicating hundreds
        # of MB of frozen attention weights in every checkpoint.
        torch.save({
            "policy": {"prototypes": policy.prototypes.detach().cpu()},
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "epoch": epoch, "step": step,
            "placement": args.placement, "policy_kind": args.policy_kind,
            "num_prototypes": args.num_prototypes, "attention_heads": args.attention_heads,
            "learning_rate": args.learning_rate,
            "lr_scheduler_type": args.lr_scheduler_type,
            "decay_start_step": args.decay_start_step,
            "decay_final_factor": args.decay_final_factor,
            "num_policy_updates": args.num_policy_updates,
            "off_policy_mask_threshold": args.off_policy_mask_threshold,
        }, path)

    trace_path = args.output_dir / "rollouts.jsonl"
    step = 0
    epoch = 0
    generation_batch = 0
    save_checkpoint(args.output_dir / "checkpoint-0.pt", step=0, epoch=0)
    with trace_path.open("w", encoding="utf-8") as trace:
        while args.max_steps is None and epoch < args.epochs or (
                args.max_steps is not None and step < args.max_steps):
            epoch += 1
            epoch_records = list(records)
            random.Random(args.seed + epoch - 1).shuffle(epoch_records)
            for start in range(0, len(epoch_records), args.questions_per_update):
                if args.max_steps is not None and step >= args.max_steps:
                    break
                question_batch = epoch_records[start:start + args.questions_per_update]
                prepared_groups = []
                generation_batch += 1
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                for dataset_index, item in question_batch:
                    with Image.open(item.image_path) as image_file:
                        image = _resize_to_pixel_cap(image_file.convert("RGB"), args.max_pixels)
                    source_inputs = _content_inputs(
                        processor, [image], build_cartographic_prompt(item.question), device, args.max_pixels
                    )
                    global_features = encode_global_image(model, source_inputs)
                    question_tokens = (None if args.policy_kind == "image_only" else
                                       encode_question_context(model, processor.tokenizer, source_inputs,
                                                               global_features, item.question))
                    group = [generate_rollout(
                        model, processor, policy, item, source_inputs, global_features,
                        args.placement, args.max_new_tokens, rollout_id, question_tokens,
                    ) for rollout_id in range(args.group_size)]
                    rewards = [row["reward"] for row in group]
                    advantages = group_advantages(rewards, device).detach().cpu()
                    # The eight questions form one logical optimizer batch, but
                    # their frozen image pixels/features do not need to occupy
                    # GPU memory together.  Cache the minimal state on CPU and
                    # move back exactly one question during current-policy
                    # recomputation below.
                    cached_source_inputs = _cache_recompute_inputs(source_inputs)
                    cached_global_features = _move_visual_features(global_features, "cpu")
                    cached_question_tokens = (None if question_tokens is None else
                                              question_tokens.detach().to("cpu", copy=True))
                    prepared_groups.append((dataset_index, item, cached_source_inputs,
                                            cached_global_features, cached_question_tokens,
                                            group, advantages))
                    payload = {
                        "generation_batch": generation_batch, "optimizer_step_before_update": step,
                        "dataset_index": dataset_index, "qa_id": item.sample_id,
                        "rewards": rewards, "advantages": advantages.tolist(),
                        "correct_count": int(sum(rewards)),
                        "incorrect_count": len(rewards) - int(sum(rewards)),
                        "signal_group": len(set(rewards)) > 1,
                        "truncated_count": sum(row["truncated"] for row in group),
                    }
                    print(f"[REWARD {generation_batch:05d}] {json.dumps(payload, ensure_ascii=False)}", flush=True)
                    for row in group:
                        serializable = {key: value for key, value in row.items()
                                        if key not in ("completion_ids", "old_logps", "completion_mask")}
                        trace.write(json.dumps({"generation_batch": generation_batch,
                                                "optimizer_step_before_update": step,
                                                "epoch": epoch, "placement": args.placement,
                                                "dataset_index": dataset_index,
                                                "qa_id": item.sample_id, **serializable},
                                               ensure_ascii=False) + "\n")
                    trace.flush()
                    del source_inputs, global_features, question_tokens
                    del cached_source_inputs, cached_global_features, cached_question_tokens
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                total_trajectories = sum(len(group) for *_, group, _ in prepared_groups)
                rows = [row for *_, group, _ in prepared_groups for row in group]
                rewards = [row["reward"] for row in rows]
                for policy_update in range(1, args.num_policy_updates + 1):
                    if args.max_steps is not None and step >= args.max_steps:
                        break
                    optimizer.zero_grad(set_to_none=True)
                    losses, ratio_drifts, sequence_kls = [], [], []
                    active = masked_off_policy = zero_advantage = truncated_without_eos = 0
                    for _, _, cached_source_inputs, cached_global_features, cached_question_tokens, group, advantages in prepared_groups:
                        # Only one question's frozen visual cache is resident on
                        # the GPU while its trajectories are recomputed.
                        source_inputs = _move_recompute_inputs(cached_source_inputs, device)
                        global_features = _move_visual_features(cached_global_features, device)
                        question_tokens = (None if cached_question_tokens is None else
                                           cached_question_tokens.to(device, non_blocking=True))
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        for row, advantage in zip(group, advantages):
                            if not bool(row["completion_mask"].any()):
                                truncated_without_eos += 1
                                continue
                            # A uniform-reward GRPO group has exactly zero
                            # advantage and contributes identically zero policy
                            # gradient. Do not run an expensive Qwen recompute
                            # or call backward on such trajectories.
                            if float(advantage) == 0.0:
                                zero_advantage += 1
                                continue
                            # Be explicit because rollout generation runs under
                            # no_grad and older Euler torch/PEFT combinations
                            # have differed in their restoration of grad mode.
                            with torch.enable_grad():
                                augmented, _ = build_prototype_prefill(
                                    model, processor.tokenizer, source_inputs, global_features,
                                    policy, args.placement, with_grad=True,
                                    question_tokens=question_tokens,
                                )
                                current = completion_log_probs(
                                    model, augmented, row["completion_ids"]
                                )
                            if not current.requires_grad:
                                raise RuntimeError(
                                    "Current-policy log probabilities are detached despite a nonzero "
                                    f"advantage={float(advantage):.6g}; "
                                    f"grad_enabled={torch.is_grad_enabled()}, "
                                    f"prototype_requires_grad={policy.prototypes.requires_grad}"
                                )
                            keep, mean_kl = off_policy_sequence_keep(
                                current, row["old_logps"], advantage,
                                row["completion_mask"], args.off_policy_mask_threshold,
                            )
                            sequence_kls.append(mean_kl)
                            ratio_drifts.append(float((
                                torch.exp(current.detach().cpu() - row["old_logps"]) - 1
                            ).abs().mean()))
                            if not keep:
                                masked_off_policy += 1
                                del current, augmented
                                continue
                            loss = clipped_language_grpo_loss(
                                current, row["old_logps"], advantage.to(device),
                                row["completion_mask"], args.clip_ratio,
                            )
                            (loss / total_trajectories).backward()
                            losses.append(float(loss.detach()))
                            active += 1
                            # Backward has consumed this trajectory's graph.
                            # Releasing the final references here prevents the
                            # next 500-token forward from overlapping it.
                            del loss, current, augmented
                        del source_inputs, global_features, question_tokens
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    if active == 0:
                        payload = {
                            "generation_batch": generation_batch,
                            "optimizer_step_unchanged": step,
                            "epoch": epoch,
                            "policy_update": policy_update,
                            "reason": "no_usable_nonzero_advantage_trajectories",
                            "zero_advantage_terminated": zero_advantage,
                            "truncated_without_eos": truncated_without_eos,
                            "off_policy_masked": masked_off_policy,
                        }
                        print(f"[SKIP_UPDATE {generation_batch:05d}.{policy_update}] "
                              f"{json.dumps(payload, ensure_ascii=False)}", flush=True)
                        # A second off-policy pass over the same empty signal
                        # cannot create a gradient; generate a fresh batch.
                        break
                    gradient_norm = (float(policy.prototypes.grad.norm())
                                     if policy.prototypes.grad is not None else 0.0)
                    before = policy.prototypes.detach().clone()
                    lr_used = float(optimizer.param_groups[0]["lr"])
                    optimizer.step()
                    scheduler.step()
                    step += 1
                    update_norm = float((policy.prototypes.detach() - before).norm())
                    metrics = {
                        "train/language_grpo_loss": sum(losses) / len(losses) if losses else 0.0,
                        "train/reward_mean": sum(rewards) / len(rewards),
                        "train/rollouts_correct": sum(rewards),
                        "train/rollouts_incorrect": len(rewards) - sum(rewards),
                        "train/signal_question_fraction": sum(len(set(row["reward"] for row in group)) > 1
                                                               for *_, group, _ in prepared_groups) / len(prepared_groups),
                        "train/active_trajectories": active,
                        "train/zero_advantage_terminated": zero_advantage,
                        "train/truncated_without_eos": truncated_without_eos,
                        "train/off_policy_masked_trajectories": masked_off_policy,
                        "train/off_policy_mask_fraction": masked_off_policy / max(1, active + masked_off_policy),
                        "train/mean_sequence_forward_kl": sum(sequence_kls) / len(sequence_kls) if sequence_kls else 0.0,
                        "train/truncated_trajectories": sum(row["truncated"] for row in rows),
                        "train/prototype_gradient_norm": gradient_norm,
                        "train/prototype_parameter_norm": float(policy.prototypes.detach().norm()),
                        "train/prototype_delta_from_init": float((policy.prototypes.detach() - initial_prototypes).norm()),
                        "train/prototype_update_norm": update_norm,
                        "train/learning_rate": lr_used,
                        "train/mean_abs_ratio_drift": sum(ratio_drifts) / len(ratio_drifts) if ratio_drifts else 0.0,
                        "train/cross_attention_entropy": sum(row["prototype_attention_entropy"] for row in rows) / len(rows),
                        "train/cross_attention_max_weight": sum(row["prototype_attention_max"] for row in rows) / len(rows),
                        "train/policy_update_in_generation": policy_update,
                        "train/generation_batch": generation_batch,
                    }
                    if torch.cuda.is_available():
                        metrics.update({
                            "system/gpu_memory_allocated_gib":
                                torch.cuda.memory_allocated() / (1024 ** 3),
                            "system/gpu_peak_memory_allocated_gib":
                                torch.cuda.max_memory_allocated() / (1024 ** 3),
                            "system/gpu_peak_memory_reserved_gib":
                                torch.cuda.max_memory_reserved() / (1024 ** 3),
                        })
                    print(json.dumps({"step": step, "epoch": epoch, "placement": args.placement,
                                      "policy_kind": args.policy_kind,
                                      "dataset_indices": [entry[0] for entry in prepared_groups],
                                      **metrics}), flush=True)
                    if wandb_run:
                        wandb_run.log(metrics, step=step)
                    if args.save_steps and step % args.save_steps == 0:
                        save_checkpoint(args.output_dir / f"checkpoint-{step}.pt",
                                        step=step, epoch=epoch)
                del prepared_groups
            if args.save_every_epoch:
                save_checkpoint(args.output_dir / f"prototype_attention_epoch_{epoch:03d}.pt",
                                step=step, epoch=epoch)
    final_checkpoint = args.output_dir / f"checkpoint-{step}.pt"
    if not final_checkpoint.exists():
        save_checkpoint(final_checkpoint, step=step, epoch=epoch)
    save_checkpoint(args.output_dir / "prototype_attention_final.pt", step=step, epoch=epoch)
    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
