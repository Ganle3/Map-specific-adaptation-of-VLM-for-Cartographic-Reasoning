#!/usr/bin/env python3
"""Function-preserving expansion of an existing uniform-rank PEFT LoRA adapter.

The original channels are copied exactly. New A rows receive standard LoRA
Kaiming initialization and new B columns are zero, so the initial BA update is
unchanged while every added channel can begin learning through B immediately.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-rank", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def load_tensors(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, str] | None]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        tensors = {name: handle.get_tensor(name) for name in handle.keys()}
    return tensors, metadata


def expand_state(state: dict[str, torch.Tensor], source_rank: int,
                 target_rank: int, seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    output: dict[str, torch.Tensor] = {}
    a_keys = sorted(name for name in state if name.endswith(".lora_A.weight"))
    b_keys = sorted(name for name in state if name.endswith(".lora_B.weight"))
    if not a_keys or len(a_keys) != len(b_keys) or 2 * len(a_keys) != len(state):
        raise ValueError(
            "Expected a pure LoRA state containing paired lora_A/lora_B weights"
        )
    for a_name in a_keys:
        b_name = a_name[:-len(".lora_A.weight")] + ".lora_B.weight"
        if b_name not in state:
            raise KeyError(f"Missing pair for {a_name}")
        old_a, old_b = state[a_name], state[b_name]
        if old_a.ndim != 2 or old_b.ndim != 2:
            raise ValueError(f"LoRA tensors must be matrices: {a_name}, {b_name}")
        if old_a.shape[0] != source_rank or old_b.shape[1] != source_rank:
            raise ValueError(
                f"Source rank mismatch for {a_name}: {tuple(old_a.shape)}, "
                f"{tuple(old_b.shape)}"
            )
        new_a = torch.empty((target_rank, old_a.shape[1]), dtype=old_a.dtype)
        new_b = torch.zeros((old_b.shape[0], target_rank), dtype=old_b.dtype)
        new_a[:source_rank].copy_(old_a)
        new_b[:, :source_rank].copy_(old_b)
        extra_a = torch.empty(
            (target_rank - source_rank, old_a.shape[1]), dtype=torch.float32
        )
        torch.nn.init.kaiming_uniform_(
            extra_a, a=math.sqrt(5), generator=generator
        )
        new_a[source_rank:].copy_(extra_a.to(old_a.dtype))
        output[a_name] = new_a.contiguous()
        output[b_name] = new_b.contiguous()
    return output


def main():
    args = arguments()
    config_path = args.source / "adapter_config.json"
    weights_path = args.source / "adapter_model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(
            f"Expected adapter_config.json and adapter_model.safetensors in {args.source}"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source_rank = int(config["r"])
    if args.target_rank <= source_rank:
        raise ValueError(
            f"target-rank must exceed source rank: {args.target_rank} <= {source_rank}"
        )
    if config.get("rank_pattern") or config.get("alpha_pattern"):
        raise ValueError("Rank expansion requires uniform rank/alpha patterns")
    if bool(config.get("use_rslora", False)):
        raise ValueError("This expansion intentionally supports ordinary LoRA only")
    source_alpha = int(config["lora_alpha"])
    if source_alpha != source_rank:
        raise ValueError(
            f"Expected source alpha/r scaling 1, got {source_alpha}/{source_rank}"
        )
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    state, metadata = load_tensors(weights_path)
    expanded = expand_state(state, source_rank, args.target_rank, args.seed)
    config["r"] = args.target_rank
    config["lora_alpha"] = args.target_rank
    (args.output / "adapter_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    save_file(
        expanded, str(args.output / "adapter_model.safetensors"),
        metadata=metadata or {"format": "pt"},
    )
    manifest = {
        "source": str(args.source.resolve()),
        "source_rank": source_rank,
        "source_alpha": source_alpha,
        "target_rank": args.target_rank,
        "target_alpha": args.target_rank,
        "seed": args.seed,
        "lora_module_count": len(expanded) // 2,
        "preservation": "old A/B copied; added A Kaiming-uniform; added B zero",
    }
    (args.output / "rank_expansion.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
