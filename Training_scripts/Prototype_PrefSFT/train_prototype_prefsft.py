#!/usr/bin/env python3
"""Train vision prototype tokens with fixed rollout preference pairs.

Example (question batch is distinct from trajectory micro-batch):
python train_prototype_prefsft.py --mapwise-image-root ... --mapverse-image-root ...
"""
from __future__ import annotations

import argparse
import json
import math
import random
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from preference_loss import prototype_preference_loss
from prototype_module import CachedVisionForward, PrototypeRoutingModule, VisionPrototypeHook
from utils import (collate_trajectory_inputs, cosine_metrics, image_for_question,
                   load_backbone, load_questions, make_trajectory_inputs, model_device,
                   normalized_logprob, qwen_visual, vision_cache_key)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rollouts", type=Path, required=True)
    p.add_argument("--baseline-cache", type=Path, required=True)
    p.add_argument("--mapwise-image-root", type=Path, required=True)
    p.add_argument("--mapverse-image-root", type=Path, required=True)
    p.add_argument("--adapter-path", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model-name", default="Qwen/Qwen3-VL-8B-Thinking")
    p.add_argument("--question-batch-size", type=int, default=2)
    p.add_argument("--trajectory-micro-batch-size", type=int, default=1)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--num-prototypes", type=int, default=8)
    p.add_argument("--alpha-init", type=float, default=0.01)
    p.add_argument("--tau1", type=float, default=0.1, help="Prototype-to-patch collection temperature.")
    p.add_argument("--tau2", type=float, default=0.1, help="Patch-to-prototype distribution temperature.")
    p.add_argument("--prototype-learning-rate", type=float, default=5e-4,
                   help="Learning rate for P tokens and residual gate alpha.")
    p.add_argument("--alpha-learning-rate", type=float, default=None,
                   help="Defaults to prototype LR. Kept separate for controlled gate tuning.")
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--save-interval", type=int, default=100)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--full-precision", action="store_true", help="Disable 4-bit frozen backbone loading.")
    p.add_argument("--wandb-project", default="prototype-prefsft")
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--vision-cache-dir", type=Path, default=None,
                   help="Precomputed frozen final-ViT states; skips ViT on every cache hit.")
    p.add_argument("--min-pixels", type=int, default=65536)
    p.add_argument("--max-pixels", type=int, default=1000000)
    p.add_argument("--min-correct", type=int, default=1); p.add_argument("--max-correct", type=int, default=7)
    return p


def _hidden_size(visual) -> int:
    for obj, name in ((getattr(visual, "config", None), "hidden_size"),
                      (getattr(visual, "config", None), "embed_dim"),
                      (getattr(getattr(visual, "merger", None), "linear_fc1", None), "in_features")):
        value = getattr(obj, name, None)
        if value:
            return int(value)
    raise AttributeError("Could not infer final ViT hidden width from Qwen visual module.")


def _scores(model, hook, rows, processor, *, micro_batch: int, with_grad: bool,
            retain_attention: bool = False, vision_cache=None, vision_key: str | None = None) -> torch.Tensor:
    values = []
    pad_id = processor.tokenizer.pad_token_id
    context = torch.enable_grad() if with_grad else torch.no_grad()
    with context:
        for start in range(0, len(rows), micro_batch):
            batch = collate_trajectory_inputs(rows[start:start + micro_batch], pad_id)
            grids = batch["image_grid_thw"]
            labels = batch.pop("labels")
            cache_context = vision_cache.use([vision_key] * len(batch["input_ids"])) if vision_cache else nullcontext()
            with cache_context:
                with hook.context(grids, capture_attention=retain_attention):
                    # KV cache is useful for generation, but only wastes memory during
                    # full teacher forcing and multiplies retained-graph memory here.
                    logits = model(**batch, use_cache=False).logits
            values.append(normalized_logprob(logits, labels))
    return torch.cat(values)


def _save(path: Path, prototype: PrototypeRoutingModule, optimizer, scheduler, args, step: int, epoch: int):
    path.mkdir(parents=True, exist_ok=True)
    torch.save({"prototypes": prototype.prototypes.detach().cpu(),
                "t_pro": {k: v.detach().cpu() for k, v in prototype.state_dict().items()
                          if k != "prototypes" and k != "alpha"},
                "alpha": prototype.alpha.detach().cpu(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "step": step, "epoch": epoch,
                "training_args": vars(args)}, path / "prototype_checkpoint.pt")
    (path / "training_args.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")


def train(args: argparse.Namespace) -> None:
    """Top-level training API requested in the experiment specification."""
    if args.question_batch_size < 1 or args.trajectory_micro_batch_size < 1:
        raise ValueError("Batch sizes must be positive.")
    random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    questions = load_questions(args.rollouts, args.baseline_cache, args.min_correct, args.max_correct)
    if not questions: raise RuntimeError("No complete mixed preference questions found.")
    model, processor = load_backbone(args.model_name, args.adapter_path, load_in_4bit=not args.full_precision,
                                     min_pixels=args.min_pixels, max_pixels=args.max_pixels)
    device = model_device(model)
    visual = qwen_visual(model)
    prototype = PrototypeRoutingModule(_hidden_size(visual), args.num_prototypes,
                                       alpha_init=args.alpha_init, tau1=args.tau1, tau2=args.tau2).to(
                                           device=device, dtype=torch.bfloat16)
    hook = VisionPrototypeHook(visual, prototype)
    vision_cache = CachedVisionForward(visual, args.vision_cache_dir) if args.vision_cache_dir else None
    alpha_lr = args.alpha_learning_rate if args.alpha_learning_rate is not None else args.prototype_learning_rate
    optimizer = AdamW([
        {"params": [prototype.prototypes], "lr": args.prototype_learning_rate, "weight_decay": 0.0},
        {"params": [prototype.alpha], "lr": alpha_lr, "weight_decay": 0.0},
    ])
    total_steps = math.ceil(len(questions) / args.question_batch_size) * args.epochs
    warmup = int(total_steps * args.warmup_ratio)
    scheduler = LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / max(1, warmup)) if s < warmup
                          else max(0.0, (total_steps - s) / max(1, total_steps - warmup)))
    step, start_epoch = 0, 0
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        prototype.prototypes.data.copy_(state["prototypes"].to(device)); prototype.alpha.data.copy_(state["alpha"].to(device))
        prototype.load_state_dict({**state["t_pro"], "prototypes": prototype.prototypes.data, "alpha": prototype.alpha.data}, strict=False)
        optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"])
        step, start_epoch = state["step"], state["epoch"]
    try:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
    except ImportError:
        wandb = None
    # Decoder checkpointing is required on a 24-GB GPU because preference loss
    # retains all eight trajectory graphs until same-question pairing. The
    # backbone stays frozen; Dropout is forced back to eval for deterministic
    # fixed-rollout scoring while model.train() activates HF checkpoint paths.
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "config"):
        model.config.use_cache = False
    model.train()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.eval()
    prototype.train()
    try:
        for epoch in range(start_epoch, args.epochs):
            random.shuffle(questions)
            for offset in range(0, len(questions), args.question_batch_size):
                group = questions[offset:offset + args.question_batch_size]
                optimizer.zero_grad(set_to_none=True)
                collected, logged = [], []
                # Keep every microbatch score connected to routing/P/alpha. We concatenate
                # the eight trajectory scores before forming same-question pairs; never
                # detach scores here, otherwise preference loss cannot reach prototypes.
                for question in group:
                    image = image_for_question(question, args.mapwise_image_root, args.mapverse_image_root)
                    rows = [make_trajectory_inputs(processor, question, t, image, device)
                            for t in question["trajectories"]]
                    baseline = torch.tensor([t["baseline_normalized_logprob"] for t in question["trajectories"]], device=device)
                    correct = torch.tensor([t["correct"] for t in question["trajectories"]], device=device)
                    proto_scores = _scores(model, hook, rows, processor,
                                           micro_batch=args.trajectory_micro_batch_size,
                                           with_grad=True, retain_attention=True,
                                           vision_cache=vision_cache,
                                           vision_key=vision_cache_key(question))
                    collect_attention_for_metrics = prototype.last_collect_attention
                    output = prototype_preference_loss(proto_scores.float(), baseline.float(), correct, args.beta)
                    output.loss.div(len(group)).backward()
                    collected.append((question, rows)); logged.append(output)
                def gradient_norm(parameters):
                    squared = [parameter.grad.detach().float().square().sum()
                               for parameter in parameters if parameter.grad is not None]
                    return torch.sqrt(torch.stack(squared).sum()) if squared else torch.zeros((), device=device)
                p_grad_norm = gradient_norm([prototype.prototypes])
                alpha_grad_norm = gradient_norm([prototype.alpha])
                grad_norm = torch.nn.utils.clip_grad_norm_(prototype.parameters(), args.max_grad_norm)
                optimizer.step(); scheduler.step(); step += 1
                metrics = {"loss/preference": float(torch.stack([x.loss for x in logged]).mean().detach().cpu()),
                           "pref/positive_gain": float(torch.stack([x.mean_positive_gain for x in logged]).mean().detach().cpu()),
                           "pref/negative_gain": float(torch.stack([x.mean_negative_gain for x in logged]).mean().detach().cpu()),
                           "pref/margin": float(torch.stack([x.mean_margin for x in logged]).mean().detach().cpu()),
                           "pref/positive_score_proto": float(torch.stack([x.positive_score_proto for x in logged]).mean().detach().cpu()),
                           "pref/negative_score_proto": float(torch.stack([x.negative_score_proto for x in logged]).mean().detach().cpu()),
                           "pref/positive_score_baseline": float(torch.stack([x.positive_score_baseline for x in logged]).mean().detach().cpu()),
                           "pref/negative_score_baseline": float(torch.stack([x.negative_score_baseline for x in logged]).mean().detach().cpu()),
                           "train/alpha": float(prototype.alpha.detach().cpu()), "train/lr": scheduler.get_last_lr()[0],
                           "train/lr_prototype": scheduler.get_last_lr()[0], "train/lr_alpha": scheduler.get_last_lr()[1],
                           "train/grad_norm": float(grad_norm.detach().cpu()),
                           "grad_norm/P": float(p_grad_norm.detach().cpu()),
                           "grad_norm/alpha": float(alpha_grad_norm.detach().cpu()),
                           "representation/residual_relative_norm": float(prototype.last_residual_relative_norm.cpu())}
                metrics.update(cosine_metrics(prototype.prototypes, collect_attention_for_metrics))
                if wandb: wandb.log(metrics, step=step)
                if step % args.save_interval == 0: _save(args.output_dir / f"checkpoint-{step}", prototype, optimizer, scheduler, args, step, epoch)
            _save(args.output_dir / f"epoch-{epoch + 1}", prototype, optimizer, scheduler, args, step, epoch + 1)
    finally:
        hook.remove()
        if vision_cache: vision_cache.remove()
        if wandb: wandb.finish()


if __name__ == "__main__":
    train(parser().parse_args())
