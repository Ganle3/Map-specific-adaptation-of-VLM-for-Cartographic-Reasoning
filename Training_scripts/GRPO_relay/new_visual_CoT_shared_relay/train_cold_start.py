#!/usr/bin/env python3
"""Supervise Relay4 prototypes from Grad-Att hidden-state/crop pairs."""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
VLM_ROOT = THIS_DIR.parents[2]
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
from policy import NewVisualCoTRelay
from prepare_cold_start_cache import cache_name, records


def parse_args():
    default_teacher = VLM_ROOT / "Training_outputs/gradattn_teacher"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-root", type=Path, default=default_teacher)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=(
        VLM_ROOT / "Training_outputs/GRPO_relay/relay4_cold_start"
    ))
    parser.add_argument("--num-relays", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--injection-scale", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-accumulation", type=int, default=8)
    parser.add_argument("--diversity-weight", type=float, default=0.05)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--include-incorrect", action="store_true")
    parser.add_argument("--incorrect-weight", type=float, default=0.10)
    parser.add_argument("--exclude-sample-ids", nargs="*", default=[])
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def crop_target(crop: dict, grad_map: np.ndarray, original_size, grid):
    width, height = map(float, original_size)
    gh, gw = grid
    x0, y0, x1, y1 = map(float, (crop["left"], crop["top"], crop["right"], crop["bottom"]))
    rows = (np.arange(gh) + 0.5) * height / gh
    cols = (np.arange(gw) + 0.5) * width / gw
    mask = ((rows[:, None] >= y0) & (rows[:, None] < y1) &
            (cols[None, :] >= x0) & (cols[None, :] < x1))
    # Very small boundary boxes can miss all patch centers; include every
    # coarse cell overlapping the box in that rare case.
    if not mask.any():
        r0, r1 = max(0, int(y0 * gh / height)), min(gh, math.ceil(y1 * gh / height))
        c0, c1 = max(0, int(x0 * gw / width)), min(gw, math.ceil(x1 * gw / width))
        mask[r0:r1, c0:c1] = True
    target = np.maximum(grad_map, 0) * mask
    if target.sum() <= 0:
        target = mask.astype(np.float32)
    target = target.astype(np.float32).reshape(-1)
    target /= max(float(target.sum()), 1e-12)
    return torch.from_numpy(target), torch.from_numpy(mask.reshape(-1))


def build_sample(record, root: Path, cache_dir: Path):
    state = torch.load(root / record["hidden_state_file"], map_location="cpu", weights_only=False)
    maps = np.load(root / record["grad_map_file"])["maps"]
    hidden = state["hidden_states_bf16"].float()
    if hidden.shape[0] != len(record["steps"]) or maps.shape[0] != len(record["steps"]):
        raise RuntimeError(f"Step alignment mismatch for sample {record['sample_id']}")
    grid = tuple(map(int, record["global_image"]["grid"]))
    memory_payload = torch.load(cache_dir / cache_name(record), map_location="cpu", weights_only=False)
    memory = memory_payload["memory_bf16"].float()
    if memory.shape[0] != grid[0] * grid[1]:
        raise RuntimeError(f"Memory/grid mismatch for sample {record['sample_id']}")
    indices, targets, masks = [], [], []
    for index, step in enumerate(record["steps"]):
        components, component_masks = [], []
        for crop in step["crops"]:
            target, mask = crop_target(
                crop, maps[index], record["global_image"]["original_size"], grid
            )
            components.append(target)
            component_masks.append(mask)
        if components:
            indices.append(index)
            targets.append(torch.stack(components[:2]))
            masks.append(torch.stack(component_masks[:2]))
    if not indices:
        return None
    return {
        "hidden": hidden[indices], "memory": memory,
        "targets": targets, "masks": masks,
        "sample_id": str(record["sample_id"]),
        "weight": 1.0 if float(record["evaluation"]["reward"]) > 0 else None,
    }


def matched_loss(log_probs, probabilities, targets, masks, diversity_weight: float):
    # log_probs/probabilities: [steps, relays, patches]. Targets are a variable
    # number (1 or 2) of connected Grad-Att crop components per step.
    losses, hits, components = [], 0, 0
    for step_index, target in enumerate(targets):
        target = target.to(log_probs.device)
        mask = masks[step_index].to(log_probs.device)
        count = min(target.shape[0], log_probs.shape[1])
        costs = -(target[:, None, :] * log_probs[step_index][None, :, :]).sum(-1)
        assignments = itertools.permutations(range(log_probs.shape[1]), count)
        candidates = [sum(costs[target_index, relay_index]
                          for target_index, relay_index in enumerate(assignment)) / count
                      for assignment in assignments]
        losses.append(torch.stack(candidates).min())
        actions = probabilities[step_index].argmax(-1)
        for component_mask in mask:
            hits += int(any(bool(component_mask[action].item()) for action in actions))
            components += 1
    base = torch.stack(losses).mean()
    if probabilities.shape[1] > 1:
        unit = F.normalize(probabilities, dim=-1)
        overlap = torch.matmul(unit, unit.transpose(-2, -1))
        pairs = probabilities.shape[1] * (probabilities.shape[1] - 1)
        diversity = (overlap.sum(dim=(-2, -1)) - probabilities.shape[1]).mean() / pairs
    else:
        diversity = base.new_zeros(())
    return base + diversity_weight * diversity, float(hits / max(components, 1)), float(diversity.detach())


def run_epoch(policy, values, args, optimizer=None):
    training = optimizer is not None
    policy.train(training)
    total_loss = total_hit = total_diversity = 0.0
    used = 0
    if training:
        optimizer.zero_grad(set_to_none=True)
    for record_index, record in enumerate(values):
        sample = build_sample(record, args.teacher_root, args.cache_dir)
        if sample is None:
            continue
        hidden = sample["hidden"].to(args.device)
        memory = sample["memory"].to(args.device)
        queries = policy.conditioned_queries(hidden[:, None, :])
        _, logits, probabilities = policy._read(
            queries, memory[None, :, :].expand(hidden.shape[0], -1, -1)
        )
        loss, hit, diversity = matched_loss(
            logits.log_softmax(-1), probabilities, sample["targets"], sample["masks"],
            args.diversity_weight,
        )
        weight = 1.0 if float(record["evaluation"]["reward"]) > 0 else args.incorrect_weight
        loss = loss * weight
        if training:
            (loss / args.grad_accumulation).backward()
            if (record_index + 1) % args.grad_accumulation == 0 or record_index + 1 == len(values):
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        total_loss += float(loss.detach())
        total_hit += hit
        total_diversity += diversity
        used += 1
        del hidden, memory, queries, logits, probabilities
    return {"loss": total_loss / max(used, 1), "component_hit_rate": total_hit / max(used, 1),
            "relay_probability_overlap": total_diversity / max(used, 1), "samples": used}


def main():
    args = parse_args()
    args.cache_dir = args.cache_dir or args.teacher_root / "global_merger"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    values = records(args.teacher_root / "manifest.jsonl")
    excluded = set(map(str, args.exclude_sample_ids))
    values = [v for v in values if str(v["sample_id"]) not in excluded]
    if not args.include_incorrect:
        values = [v for v in values if float(v["evaluation"]["reward"]) > 0]
    missing = [cache_name(v) for v in values if not (args.cache_dir / cache_name(v)).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(set(missing))} merger caches. Run prepare_cold_start_cache.py first; "
            f"example: {missing[0]}"
        )
    random.shuffle(values)
    validation_count = max(1, round(len(values) * args.validation_fraction))
    validation, training = values[:validation_count], values[validation_count:]
    first = torch.load(args.teacher_root / training[0]["hidden_state_file"], map_location="cpu", weights_only=False)
    hidden_size = int(first["hidden_states_bf16"].shape[-1])
    policy_config = {
        "hidden_size": hidden_size, "num_relays": args.num_relays,
        "temperature": args.temperature, "injection_scale": args.injection_scale,
    }
    policy = NewVisualCoTRelay(**policy_config).to(args.device)
    initial = policy.prototypes.detach().cpu().clone()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    history, best_hit = [], -1.0
    for epoch in range(1, args.epochs + 1):
        random.shuffle(training)
        train_metrics = run_epoch(policy, training, args, optimizer)
        with torch.no_grad():
            validation_metrics = run_epoch(policy, validation, args)
        delta = float((policy.prototypes.detach().cpu() - initial).norm())
        row = {"epoch": epoch, "train": train_metrics, "validation": validation_metrics,
               "prototype_norm": float(policy.prototypes.detach().norm()),
               "prototype_delta_from_init": delta}
        history.append(row)
        print(json.dumps(row), flush=True)
        payload = {
            "policy_kind": "relay4_gradattn_cold_start",
            "policy_config": policy_config,
            "state_dict": {k: v.detach().cpu() for k, v in policy.state_dict().items()},
            "optimizer": optimizer.state_dict(), "epoch": epoch,
            "metrics": row, "history": history,
            "training_sample_ids": [str(v["sample_id"]) for v in training],
            "validation_sample_ids": [str(v["sample_id"]) for v in validation],
            "excluded_sample_ids": sorted(excluded),
            "args": vars(args),
        }
        torch.save(payload, args.output_dir / "relay4_cold_start_last.pt")
        if validation_metrics["component_hit_rate"] > best_hit:
            best_hit = validation_metrics["component_hit_rate"]
            torch.save(payload, args.output_dir / "relay4_cold_start_best.pt")
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
