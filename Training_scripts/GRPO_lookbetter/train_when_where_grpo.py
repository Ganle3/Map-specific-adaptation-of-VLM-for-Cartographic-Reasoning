#!/usr/bin/env python3
"""Stage-1 GRPO for a frozen Qwen3-VL When--Where observation controller."""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SPARSE = ROOT / "SparseVLM_eliminateCONTEXT"
for path in (str(Path(__file__).resolve().parent), str(SPARSE)):
    if path not in sys.path:
        sys.path.insert(0, path)

from continuous_cache_qwen3vl import load_model
from data import load_mapverse_questions
from grpo import ControllerGRPO, GRPOConfig
from policy import ControllerConfig, WhenWhereController
from rollout import RolloutConfig, rollout


def calibrate_tau0(args, model, processor, records):
    """Calibrate from observed policy rollouts; never hard-code an arbitrary threshold."""
    # A neutral policy with very high threshold produces no crop actions while
    # preserving the exact frozen decoder trajectory used for measurements.
    neutral = WhenWhereController(ControllerConfig(tau0=1e6, temperature=args.temperature)).to(model.get_input_embeddings().weight.device)
    values = []
    # Keep decisions eligible so their JSD features are logged.  The neutral
    # threshold makes LOOK probability effectively zero, so no expensive crop
    # probe is executed during calibration.
    probe_cfg = RolloutConfig(max_reasoning_tokens=args.calibration_tokens, max_looks=999,
                              eligible_after_tokens=args.eligible_after_tokens, eligible_stride=args.eligible_stride)
    for item in records[:args.calibration_examples]:
        row = rollout(model, processor, item, neutral, probe_cfg)
        values.extend(action["features"][0] for action in row["actions"] if action["kind"] == "when")
    if not values:
        raise RuntimeError("Calibration recorded no eligible attention transitions; reduce eligible_after_tokens/stride.")
    return float(np.quantile(values, args.target_jsd_quantile)), {"jsd_count": len(values), "tau0": float(np.quantile(values, args.target_jsd_quantile))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True,
                        help="MapVerse image directory, e.g. ...\\MapVerse\\data\\imgs.")
    parser.add_argument("--mapwise-root", type=Path, default=None,
                        help="Root of mapwise-dataset; required when JSON contains dataset='mapwise'.")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path, default=None,
                        help="Resume controller/optimizer state. --epochs remains the total target epoch count.")
    parser.add_argument("--start-index", type=int, default=1,
                        help="1-based inclusive index in train-json; use with --end-index for a controlled overfit subset.")
    parser.add_argument("--end-index", type=int, default=None,
                        help="1-based inclusive end index in train-json (default: final record).")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=6)
    parser.add_argument("--questions-per-update", type=int, default=1,
                        help="Number of independently normalized QA rollout groups combined in one controller optimizer step.")
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--controller-architecture", choices=("linear", "residual_mlp"), default="linear",
                        help="linear reproduces the original controller; residual_mlp adds zero-initialized nonlinear residual heads.")
    parser.add_argument("--mlp-hidden-dim", type=int, default=16,
                        help="Hidden width used only by --controller-architecture residual_mlp.")
    parser.add_argument("--save-steps", type=int, default=1,
                        help="Write a versioned controller checkpoint every N optimizer steps (controller_last.pt is always overwritten).")
    parser.add_argument("--tau0", type=float, default=None, help="Use calibrated value when omitted.")
    parser.add_argument("--temperature", type=float, default=0.02)
    parser.add_argument("--target-jsd-quantile", type=float, default=0.85)
    parser.add_argument("--calibration-examples", type=int, default=4)
    parser.add_argument("--calibration-tokens", type=int, default=80)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--recoverable-only", action="store_true")
    parser.add_argument("--max-reasoning-tokens", type=int, default=320)
    parser.add_argument("--eligible-after-tokens", type=int, default=12)
    parser.add_argument("--eligible-stride", type=int, default=4)
    parser.add_argument("--max-looks", type=int, default=3)
    parser.add_argument("--cooldown-tokens", type=int, default=20)
    parser.add_argument("--revision-tokens", type=int, default=0)
    parser.add_argument("--how-refresh", choices=("roi_only", "roi_biased_full_anchor_trace"),
                        default="roi_biased_full_anchor_trace")
    parser.add_argument("--when-probe-stride", type=int, default=4)
    parser.add_argument("--when-attention-source", choices=("main", "ungated_probe"), default="main")
    parser.add_argument("--global-bias-mode", choices=("off", "look_snapshot"), default="off")
    parser.add_argument("--global-key-bias", type=float, default=-4.0)
    parser.add_argument("--global-retain-attention-mass", type=float, default=0.40)
    parser.add_argument("--roi-key-bias", type=float, default=2.079442)
    parser.add_argument("--global-max-pixels", type=int, default=1_000_000)
    parser.add_argument("--attention-mass", type=float, default=0.60,
                        help="Cumulative attribution mass retained for WHERE components; higher gives broader proposals.")
    parser.add_argument("--proposal-cap", type=int, default=6)
    parser.add_argument("--initial-size-scale", type=float, default=1.0,
                        help="Initial ROI width/height scale around each attention component.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--wandb-project", default=None,
                        help="Optional Weights & Biases project. Omit to keep training fully local.")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--allow-unvalidated-prototype", action="store_true",
                        help="Temporary escape hatch. The token-step rollout is disabled by default until the "
                             "deterministic compatibility suite is implemented and passes.")
    args = parser.parse_args()
    if args.group_size < 2 or args.questions_per_update < 1 or args.epochs < 1 or args.max_looks < 0 or args.save_steps < 1 or args.mlp_hidden_dim < 1:
        parser.error("group-size >=2, questions-per-update >=1, epochs >=1, and max-looks >=0 are required.")
    if not 0 < args.attention_mass <= 1 or args.proposal_cap < 1 or args.initial_size_scale <= 0:
        parser.error("attention-mass must be in (0,1], proposal-cap >=1, and initial-size-scale positive.")
    if not args.allow_unvalidated_prototype:
        parser.error(
            "Training is intentionally disabled: the current token-step prototype has not demonstrated "
            "equivalence to continuous_cache_qwen3vl.py. Run the forthcoming deterministic compatibility "
            "suite first; do not use --allow-unvalidated-prototype for reported experiments."
        )
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if args.resume_checkpoint is not None and not args.resume_checkpoint.is_file():
        parser.error(f"Resume checkpoint does not exist: {args.resume_checkpoint}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_records = load_mapverse_questions(args.train_json, args.image_root, mapwise_root=args.mapwise_root,
                                          recoverable_only=args.recoverable_only)
    end_index = args.end_index or len(all_records)
    if args.start_index < 1 or end_index < args.start_index or end_index > len(all_records):
        parser.error(f"Select a valid 1-based range within 1..{len(all_records)}.")
    records = all_records[args.start_index - 1:end_index]
    if args.max_examples is not None:
        records = records[:args.max_examples]
    model, processor = load_model(args.model, args.adapter)
    model.requires_grad_(False).eval()
    if args.tau0 is None:
        tau0, calibration = calibrate_tau0(args, model, processor, records)
    else:
        tau0, calibration = args.tau0, {"tau0": args.tau0, "source": "CLI"}
    requested_controller_cfg = ControllerConfig(
        tau0=tau0, temperature=args.temperature, initial_size_scale=args.initial_size_scale,
        architecture=args.controller_architecture, mlp_hidden_dim=args.mlp_hidden_dim,
    )
    resume_state = None
    if args.resume_checkpoint is not None:
        resume_state = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
        if "controller" not in resume_state or "controller_config" not in resume_state:
            parser.error("--resume-checkpoint must contain controller and controller_config.")
        checkpoint_cfg = ControllerConfig(**resume_state["controller_config"])
        if checkpoint_cfg != requested_controller_cfg:
            parser.error(
                "Resume controller configuration differs from requested arguments. "
                f"checkpoint={checkpoint_cfg}; requested={requested_controller_cfg}"
            )
    controller = WhenWhereController(requested_controller_cfg).to(model.get_input_embeddings().weight.device)
    trainer = ControllerGRPO(controller, GRPOConfig(learning_rate=args.learning_rate))
    if resume_state is not None:
        controller.load_state_dict(resume_state["controller"], strict=True)
        if "optimizer" in resume_state:
            trainer.optimizer.load_state_dict(resume_state["optimizer"])
    rollout_cfg = RolloutConfig(
        max_reasoning_tokens=args.max_reasoning_tokens,
        how_refresh=args.how_refresh,
        revision_tokens=args.revision_tokens,
        eligible_after_tokens=args.eligible_after_tokens,
        eligible_stride=args.eligible_stride,
        max_looks=args.max_looks,
        cooldown_tokens=args.cooldown_tokens,
        when_probe_stride=args.when_probe_stride,
        when_attention_source=args.when_attention_source,
        attention_mass=args.attention_mass,
        proposal_cap=args.proposal_cap,
        global_bias_mode=args.global_bias_mode,
        global_key_bias=args.global_key_bias,
        global_retain_attention_mass=args.global_retain_attention_mass,
        roi_key_bias=args.roi_key_bias,
        global_max_pixels=args.global_max_pixels,
    )
    (args.output_dir / "run_config.json").write_text(json.dumps({"args": vars(args), "tau0": tau0,
        "calibration": calibration, "rollout": asdict(rollout_cfg),
        "selected_source_indices": list(range(args.start_index, args.start_index + len(records)))}, default=str, indent=2), encoding="utf-8")
    wandb_run = None
    if args.wandb_project is not None and args.wandb_mode != "disabled":
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError("--wandb-project was set but wandb is not installed in this environment.") from exc
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            mode=args.wandb_mode,
            config={"args": vars(args), "tau0": tau0, "rollout": asdict(rollout_cfg),
                    "selected_source_indices": list(range(args.start_index, args.start_index + len(records)))},
        )
    trajectory_path = args.output_dir / "trajectories.json"
    if args.resume_checkpoint is not None and trajectory_path.is_file():
        history = json.loads(trajectory_path.read_text(encoding="utf-8"))
    else:
        history = []
    source_records = list(zip(range(args.start_index, args.start_index + len(records)), records))
    optimizer_step = int(resume_state.get("optimizer_step", 0)) if resume_state is not None else 0
    start_epoch = int(resume_state.get("epoch", -1)) + 1 if resume_state is not None else 0
    if start_epoch >= args.epochs:
        parser.error(f"Checkpoint already completed epoch {start_epoch}; --epochs={args.epochs} leaves no work to run.")
    for epoch in range(start_epoch, args.epochs):
        for first in range(0, len(source_records), args.questions_per_update):
            batch = source_records[first:first + args.questions_per_update]
            groups = [[rollout(model, processor, item, controller, rollout_cfg)
                       for _ in range(args.group_size)] for _, item in batch]
            update, group_metrics = trainer.update_groups(groups)
            optimizer_step += 1
            wandb_metrics = {"train/batch_loss": update["loss"], "train/grad_norm": update["grad_norm"],
                             "train/reward_mean": update["reward_mean"], "train/accuracy": update["accuracy"],
                             "train/correct_rollouts": update["correct_rollouts"],
                             "train/total_rollouts": update["total_rollouts"],
                             "train/question_groups": update["question_groups"],
                             "train/rollouts": update["rollouts"], "epoch": epoch}
            for (source_index, item), group, group_metric in zip(batch, groups, group_metrics):
                metrics = {**group_metric, "epoch": epoch, "optimizer_step": optimizer_step,
                           "batch_question_count": len(batch), "batch_loss": update["loss"],
                           "batch_grad_norm": update["grad_norm"], "source_index": source_index,
                           "sample_id": item.sample_id, "dataset": item.metadata.get("dataset", "mapverse"),
                           "question": item.question, "rollouts": group}
                history.append(metrics)
                print(f"epoch={epoch} step={optimizer_step} index={source_index} id={item.sample_id!r} "
                      f"reward={metrics['reward_mean']:.3f} mixed={metrics['mixed_rewards']} "
                      f"batch_grad_norm={metrics['batch_grad_norm']:.4g}")
                prefix = f"question_{source_index}"
                wandb_metrics.update({f"{prefix}/reward_mean": group_metric["reward_mean"],
                                      f"{prefix}/accuracy": group_metric["accuracy"],
                                      f"{prefix}/correct_rollouts": group_metric["correct_rollouts"],
                                      f"{prefix}/total_rollouts": group_metric["total_rollouts"],
                                      f"{prefix}/reward_std": group_metric["reward_std"],
                                      f"{prefix}/mixed_rewards": float(group_metric["mixed_rewards"]),
                                      f"{prefix}/ratio_mean": group_metric["ratio_mean"]})
            if wandb_run is not None:
                wandb_metrics.update({f"controller/{name}": float(parameter.detach().cpu())
                                      for name, parameter in controller.named_parameters()})
                wandb_run.log(wandb_metrics, step=optimizer_step)
            checkpoint = {"controller": controller.state_dict(), "controller_config": asdict(controller.cfg),
                          "optimizer": trainer.optimizer.state_dict(), "epoch": epoch,
                          "optimizer_step": optimizer_step, "last_update": update}
            torch.save(checkpoint, args.output_dir / "controller_last.pt")
            # Rollouts are expensive and are also the only way to inspect a
            # per-question GRPO signal.  Persist them after *every* optimizer
            # update instead of waiting for the complete multi-epoch run.
            # Write then replace so a reader never observes half-written JSON.
            temporary_path = args.output_dir / "trajectories.json.tmp"
            temporary_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary_path.replace(trajectory_path)
            if optimizer_step % args.save_steps == 0:
                checkpoint_dir = args.output_dir / "checkpoints"
                checkpoint_dir.mkdir(exist_ok=True)
                torch.save(checkpoint, checkpoint_dir / f"controller_step_{optimizer_step:05d}.pt")
    (args.output_dir / "trajectories.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    torch.save({"controller": controller.state_dict(), "controller_config": asdict(controller.cfg),
                "optimizer": trainer.optimizer.state_dict(), "optimizer_step": optimizer_step},
               args.output_dir / "controller_final.pt")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
