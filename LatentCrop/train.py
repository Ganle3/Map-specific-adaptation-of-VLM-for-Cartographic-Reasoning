#!/usr/bin/env python3
"""Train LatentVisualExtractor on frozen Qwen3-VL main-merger targets.

Only this extractor is optimized.  Qwen3-VL and the GRPO LoRA adapter are not
loaded in this process; their representations were persisted by the collector
and prepare_teacher.py.  No GRPO, answer loss, or decoder forward is used.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from data import LatentCropDataset, build_examples, collate_steps, split_by_map
from latent_visual_extractor import ExtractorConfig, LatentVisualExtractor


def reconstruction_and_stop_loss(output: dict, target: torch.Tensor, lengths: torch.Tensor,
                                 cosine_weight: float, stop_weight: float) -> dict:
    """Normalize visual errors by valid merger tokens and balance STOP per step.

    For a Teacher length N, positions 0..N-1 are CONTINUE, position N is STOP;
    all positions after N are ignored.  An empty crop contributes only STOP.
    """
    batch, queries = output["stop_logits"].shape
    if int(lengths.max()) >= queries:
        raise ValueError("The forward pass must include the STOP slot after the longest target.")
    max_tokens = target.shape[1]
    if max_tokens:
        valid = torch.arange(max_tokens, device=lengths.device)[None, :] < lengths[:, None]
        estimate = output["embeddings"][:, :max_tokens].float()
        truth = target.float()
        mse_per_token = (estimate - truth).square().mean(dim=-1)
        cosine_per_token = 1 - F.cosine_similarity(estimate, truth, dim=-1, eps=1e-8)
        denominator = valid.sum().clamp_min(1)
        mse = (mse_per_token * valid).sum() / denominator
        cosine = (cosine_per_token * valid).sum() / denominator
        prediction_norm = estimate.norm(dim=-1)[valid].mean()
        teacher_norm = truth.norm(dim=-1)[valid].mean()
    else:
        zero = output["embeddings"].sum() * 0
        mse = cosine = prediction_norm = teacher_norm = zero

    per_step_stop = []
    for i in range(batch):
        n = int(lengths[i])
        positive = F.softplus(-output["stop_logits"][i, n])
        if n:
            negative = F.softplus(output["stop_logits"][i, :n]).mean()
            per_step_stop.append((positive + negative) * 0.5)
        else:
            per_step_stop.append(positive)
    stop = torch.stack(per_step_stop).mean()
    total = mse + cosine_weight * cosine + stop_weight * stop
    return {"total": total, "mse": mse, "cosine": cosine, "stop": stop,
            "predicted_norm": prediction_norm, "teacher_norm": teacher_norm}


def run_epoch(model: LatentVisualExtractor, loader: DataLoader, optimizer, device: torch.device,
              cosine_weight: float, stop_weight: float, gradient_accumulation: int,
              grad_clip: float, train: bool, max_batches: int | None = None) -> dict:
    model.train(train)
    if train:
        optimizer.zero_grad(set_to_none=True)
    totals = Counter()
    examples_seen = 0
    positive = false_positive = false_negative = exact_length = no_crop_correct = no_crop_total = 0
    length_absolute_error = 0
    peak_memory = 0
    length_bins = {label: {"count": 0, "mse_sum": 0.0, "cosine_sum": 0.0}
                   for label in ("1-64", "65-128", "129-256", "257-384")}
    batch_limit = len(loader) if max_batches is None else min(len(loader), max_batches)
    for batch_index, batch in enumerate(loader):
        if batch_index >= batch_limit:
            break
        metadata = batch.pop("metadata")  # Only tensors are moved to the GPU.
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
        with torch.set_grad_enabled(train), context:
            # Training computes only the necessary prefix of the parallel query
            # bank. Validation checks the full K_max+1 bank and true first STOP.
            count = int(batch["lengths"].max()) + 1 if train else None
            output = model(batch["hidden"], batch["keys"], batch["values"], batch["memory_mask"],
                           batch["global_grid"], num_queries=count)
            losses = reconstruction_and_stop_loss(output, batch["target"], batch["lengths"],
                                                  cosine_weight, stop_weight)
        if train:
            remaining_in_group = batch_limit - (batch_index // gradient_accumulation) * gradient_accumulation
            group_size = min(gradient_accumulation, remaining_in_group)
            (losses["total"] / group_size).backward()
            if (batch_index + 1) % gradient_accumulation == 0 or batch_index + 1 == batch_limit:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        size = len(metadata)
        examples_seen += size
        for name, value in losses.items():
            totals[name] += float(value.detach()) * size
        similarity = output["stop_similarity"].detach()
        threshold = model.config.stop_threshold
        for i, actual in enumerate(batch["lengths"].tolist()):
            active = similarity[i] >= threshold
            predicted = int(active.nonzero(as_tuple=True)[0][0]) if active.any() else model.config.k_max
            exact_length += int(predicted == actual)
            length_absolute_error += abs(predicted - actual)
            positive += int(bool(active[actual]))
            false_negative += int(not bool(active[actual]))
            false_positive += int(active[:actual].sum())
            if actual == 0:
                no_crop_total += 1
                no_crop_correct += int(predicted == 0)
            elif not train:
                label = ("1-64" if actual <= 64 else "65-128" if actual <= 128 else
                         "129-256" if actual <= 256 else "257-384")
                estimate = output["embeddings"][i, :actual].float()
                truth = batch["target"][i, :actual].float()
                length_bins[label]["count"] += 1
                length_bins[label]["mse_sum"] += float((estimate - truth).square().mean())
                length_bins[label]["cosine_sum"] += float((1 - F.cosine_similarity(estimate, truth, dim=-1)).mean())
        if device.type == "cuda":
            peak_memory = max(peak_memory, torch.cuda.max_memory_allocated(device))

    if not examples_seen:
        raise ValueError("An epoch received no examples.")
    metrics = {key: value / examples_seen for key, value in totals.items()}
    metrics.update({"examples": examples_seen, "stop_precision": positive / max(1, positive + false_positive),
                    "stop_recall": positive / max(1, positive + false_negative),
                    "exact_length_accuracy": exact_length / examples_seen,
                    "length_mae": length_absolute_error / examples_seen,
                    "no_crop_accuracy": no_crop_correct / max(1, no_crop_total),
                    "no_crop_examples": no_crop_total,
                    "peak_allocated_gib": peak_memory / (1024 ** 3)})
    if not train:
        metrics["by_teacher_length"] = {
            label: {"count": values["count"],
                    "mse": values["mse_sum"] / max(1, values["count"]),
                    "cosine_error": values["cosine_sum"] / max(1, values["count"])}
            for label, values in length_bins.items()}
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--cosine-weight", type=float, default=0.1)
    parser.add_argument("--stop-weight", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--k-max", type=int, default=384)
    parser.add_argument("--latent-dim", type=int, default=None,
                        help="Default: actual flattened KV width (8 KV heads x 128 in this checkpoint).")
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--query-grid-width", type=int, default=16)
    parser.add_argument("--mlp-ratio", type=int, default=2)
    parser.add_argument("--stop-threshold", type=float, default=0.0)
    parser.add_argument("--stop-logit-scale", type=float, default=10.0)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=5, help="Epoch interval for numbered Student checkpoints.")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--allow-missing-targets", action="store_true",
                        help="Only for partial Teacher-cache smoke tests; otherwise missing targets are an error.")
    parser.add_argument("--no-map-balanced", action="store_false", dest="map_balanced")
    parser.add_argument("--inspect-only", action="store_true", help="Check data/dimensions/parameter count, then exit.")
    parser.add_argument("--max-train-batches", type=int, default=None,
                        help="Debug only: limit training batches per epoch; metrics are not full-epoch estimates.")
    parser.add_argument("--max-validation-batches", type=int, default=None,
                        help="Debug only: limit validation batches per epoch; do not compare checkpoints from this run.")
    parser.set_defaults(map_balanced=True)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.gradient_accumulation < 1 or args.save_every < 1:
        parser.error("Epochs, batch size, accumulation and save interval must be positive.")
    if any(value is not None and value < 1 for value in (args.max_train_batches, args.max_validation_batches)):
        parser.error("Debug batch limits must be positive if supplied.")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    examples, join_stats = build_examples(args.source_root, args.teacher_root, args.k_max,
                                          allow_missing_targets=args.allow_missing_targets)
    split = split_by_map(examples, args.seed, args.validation_fraction, args.test_fraction)
    datasets = {name: LatentCropDataset(parts, join_stats["merger_dim"]) for name, parts in split.items()}
    if any(len(dataset) == 0 for dataset in datasets.values()):
        raise ValueError("One map-level split has no usable steps; inspect Teacher target coverage.")
    first = datasets["train"][0]
    config = ExtractorConfig(reasoning_dim=first["hidden"].numel(), kv_dim=first["keys"].shape[-1],
                             merger_dim=join_stats["merger_dim"], decoder_layers=first["keys"].shape[0],
                             latent_dim=args.latent_dim or first["keys"].shape[-1],
                             attention_heads=args.attention_heads, k_max=args.k_max,
                             query_grid_width=args.query_grid_width, mlp_ratio=args.mlp_ratio,
                             stop_threshold=args.stop_threshold, stop_logit_scale=args.stop_logit_scale)
    model = LatentVisualExtractor(config)
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    summary = {"join": join_stats, "split_steps": {name: len(parts) for name, parts in split.items()},
               "split_maps": {name: len({example.map_id for example in parts}) for name, parts in split.items()},
               "model_config": asdict(config), "trainable_parameters": parameter_count,
               "effective_batch_size": args.batch_size * args.gradient_accumulation}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_config.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")
    (args.output_dir / "data_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "map_split.json").write_text(json.dumps(
        {name: sorted({example.map_id for example in parts}) for name, parts in split.items()}, indent=2), encoding="utf-8")
    if args.inspect_only:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    start_epoch, best_validation = 1, float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        if checkpoint["model_config"] != asdict(config):
            raise ValueError("Resume checkpoint extractor configuration differs from this run.")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation = float(checkpoint["best_validation"])

    generator = torch.Generator().manual_seed(args.seed)
    if args.map_balanced:
        counts = Counter(example.map_id for example in split["train"])
        weights = [1.0 / counts[example.map_id] for example in split["train"]]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True, generator=generator)
    else:
        sampler = None
    train_loader = DataLoader(datasets["train"], batch_size=args.batch_size, sampler=sampler,
                              shuffle=not args.map_balanced, collate_fn=collate_steps, num_workers=0,
                              pin_memory=device.type == "cuda")
    val_loader = DataLoader(datasets["validation"], batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_steps, num_workers=0, pin_memory=device.type == "cuda")
    metrics_file = args.output_dir / "metrics.jsonl"
    for epoch in range(start_epoch, args.epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        training = run_epoch(model, train_loader, optimizer, device, args.cosine_weight, args.stop_weight,
                             args.gradient_accumulation, args.grad_clip, train=True,
                             max_batches=args.max_train_batches)
        with torch.inference_mode():
            validation = run_epoch(model, val_loader, None, device, args.cosine_weight, args.stop_weight,
                                   1, args.grad_clip, train=False, max_batches=args.max_validation_batches)
        improved = validation["total"] < best_validation
        best_validation = min(best_validation, validation["total"])
        row = {"epoch": epoch, "train": training, "validation": validation,
               "best_validation": best_validation}
        with metrics_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(f"epoch={epoch} train_loss={training['total']:.4f} val_loss={validation['total']:.4f} "
              f"val_mse={validation['mse']:.4f} val_length_mae={validation['length_mae']:.2f} "
              f"val_no_crop_acc={validation['no_crop_accuracy']:.3f} GPU_peak={validation['peak_allocated_gib']:.2f} GiB")
        state = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "model_config": asdict(config), "best_validation": best_validation}
        torch.save(state, args.output_dir / "last.pt")
        if improved:
            torch.save(state, args.output_dir / "best.pt")
        if epoch % args.save_every == 0:
            torch.save(state, args.output_dir / f"epoch_{epoch:03d}.pt")
    print("The test split remains untouched; evaluate it only after selecting a checkpoint on validation maps.")


if __name__ == "__main__":
    main()
