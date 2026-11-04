#!/usr/bin/env python3
"""Measure how strongly a cold-start Relay4 selector depends on hidden state."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
from policy import NewVisualCoTRelay
from prepare_cold_start_cache import cache_name, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--split", choices=("training", "validation"), default="validation")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    policy = NewVisualCoTRelay(**payload["policy_config"])
    policy.load_state_dict(payload["state_dict"], strict=True)
    policy.eval()
    wanted = set(payload[f"{args.split}_sample_ids"])
    values = [v for v in records(args.teacher_root / "manifest.jsonl")
              if str(v["sample_id"]) in wanted]
    generator = torch.Generator().manual_seed(args.seed)
    changed = total = 0
    tv_sum = 0.0
    prototype_only_changed = 0
    unique_sum = torch.zeros(policy.num_relays)
    sample_count = 0
    cache_dir = args.teacher_root / "global_merger"
    with torch.no_grad():
        for record in values:
            state = torch.load(args.teacher_root / record["hidden_state_file"],
                               map_location="cpu", weights_only=False)
            hidden = state["hidden_states_bf16"].float()
            memory = torch.load(cache_dir / cache_name(record), map_location="cpu",
                                weights_only=False)["memory_bf16"].float()
            queries = policy.conditioned_queries(hidden[:, None, :])
            _, logits, probs = policy._read(
                queries, memory[None].expand(hidden.shape[0], -1, -1)
            )
            actions = logits.argmax(-1)
            permutation = torch.randperm(hidden.shape[0], generator=generator)
            shuffled_queries = policy.conditioned_queries(hidden[permutation, None, :])
            _, shuffled_logits, shuffled_probs = policy._read(
                shuffled_queries, memory[None].expand(hidden.shape[0], -1, -1)
            )
            shuffled_actions = shuffled_logits.argmax(-1)
            changed += int((actions != shuffled_actions).sum())
            total += actions.numel()
            tv_sum += float((probs - shuffled_probs).abs().sum(-1).mul(0.5).sum())

            prototype_queries = torch.nn.functional.normalize(
                policy.prototypes.float(), dim=-1
            )[None].expand(hidden.shape[0], -1, -1)
            _, prototype_logits, _ = policy._read(
                prototype_queries, memory[None].expand(hidden.shape[0], -1, -1)
            )
            prototype_actions = prototype_logits.argmax(-1)
            prototype_only_changed += int((actions != prototype_actions).sum())
            for relay in range(policy.num_relays):
                unique_sum[relay] += actions[:, relay].unique().numel()
            sample_count += 1
    print(json.dumps({
        "split": args.split,
        "samples": sample_count,
        "decisions": total,
        "hidden_shuffle_action_change_fraction": changed / max(total, 1),
        "hidden_shuffle_probability_total_variation_mean": tv_sum / max(total, 1),
        "normal_vs_prototype_only_action_change_fraction": prototype_only_changed / max(total, 1),
        "mean_unique_actions_per_image_by_relay": (unique_sum / max(sample_count, 1)).tolist(),
    }, indent=2))


if __name__ == "__main__":
    main()
