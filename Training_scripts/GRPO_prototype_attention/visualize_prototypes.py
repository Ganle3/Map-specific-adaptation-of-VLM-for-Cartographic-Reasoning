#!/usr/bin/env python3
"""Visualize what image-only prototypes extract from Qwen merger tokens."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
from data import build_cartographic_prompt, load_mapverse_questions
from rolling_sparse_qwen3vl import load_model

from prototype_cross_attention_policy import PrototypeCrossAttentionPolicy, decoder_hidden_size
from prototype_prefill import build_prototype_prefill, encode_global_image, qwen_lm


def cosine_matrix(values: torch.Tensor) -> np.ndarray:
    values = torch.nn.functional.normalize(values.float(), dim=-1)
    return (values @ values.T).cpu().numpy()


def attention_metrics(attention: torch.Tensor, grid_h: int, grid_w: int) -> list[dict]:
    """Metrics from head-averaged attention [P,N]."""
    yy, xx = torch.meshgrid(
        torch.arange(grid_h, dtype=torch.float32),
        torch.arange(grid_w, dtype=torch.float32), indexing="ij",
    )
    xx = (xx.flatten() + 0.5) / grid_w
    yy = (yy.flatten() + 0.5) / grid_h
    result = []
    for slot, weights in enumerate(attention):
        weights = weights.float() / weights.float().sum().clamp_min(1e-12)
        entropy = float(-(weights.clamp_min(1e-12) * weights.clamp_min(1e-12).log()).sum())
        effective = float(torch.exp(torch.tensor(entropy)))
        top = torch.topk(weights, k=min(10, weights.numel()))
        cells = []
        for value, index in zip(top.values.tolist(), top.indices.tolist()):
            row, col = divmod(index, grid_w)
            cells.append({"row": row, "col": col, "weight": value,
                          "x_normalized": (col + 0.5) / grid_w,
                          "y_normalized": (row + 0.5) / grid_h})
        result.append({
            "prototype": slot + 1,
            "entropy": entropy,
            "effective_token_count": effective,
            "max_weight": float(weights.max()),
            "center_x_normalized": float((weights * xx).sum()),
            "center_y_normalized": float((weights * yy).sum()),
            "top_cells": cells,
        })
    return result


def show_matrix(matrix: np.ndarray, title: str, output: Path, labels: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, f"{matrix[row, col]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_average_overlays(image: Image.Image, maps: np.ndarray, metrics: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(15, 13))
    for slot, ax in enumerate(axes.flat):
        heat = maps[slot]
        heat_display = heat / max(float(heat.max()), 1e-12)
        ax.imshow(image)
        ax.imshow(heat_display, cmap="turbo", alpha=0.55, interpolation="bilinear",
                  extent=(0, image.width, image.height, 0), vmin=0, vmax=1)
        stat = metrics[slot]
        ax.set_title(
            f"R{slot + 1}: max={stat['max_weight']:.4f}, "
            f"eff={stat['effective_token_count']:.1f}\n"
            f"COM=({stat['center_x_normalized']:.2f}, {stat['center_y_normalized']:.2f})"
        )
        ax.axis("off")
    fig.suptitle("Prototype → image-merger attention (mean over 8 heads)", fontsize=16)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_head_maps(head_maps: np.ndarray, output_dir: Path) -> None:
    output_dir.mkdir(exist_ok=True)
    heads, prototypes, _, _ = head_maps.shape
    rows = 2
    cols = math.ceil(heads / rows)
    for slot in range(prototypes):
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 7))
        for head, ax in enumerate(np.asarray(axes).flat):
            if head >= heads:
                ax.axis("off")
                continue
            heat = head_maps[head, slot]
            shown = ax.imshow(heat, cmap="turbo", vmin=0, vmax=max(float(heat.max()), 1e-12))
            ax.set_title(f"head {head + 1}; max={heat.max():.4f}")
            ax.axis("off")
            fig.colorbar(shown, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(f"R{slot + 1}: per-head spatial attention")
        fig.tight_layout()
        fig.savefig(output_dir / f"R{slot + 1}_heads.png", dpi=160)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-indices", type=int, nargs="+", default=[245, 246, 247, 248])
    parser.add_argument("--mapverse-root", type=Path, required=True)
    parser.add_argument("--mapwise-root", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-pixels", type=int, default=1_000_000)
    parser.add_argument("--no-load-in-4bit", action="store_false", dest="load_in_4bit")
    parser.set_defaults(load_in_4bit=True)
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu")
    if state.get("placement") != "after_question":
        print(f"Warning: checkpoint placement is {state.get('placement')!r}, not 'after_question'", flush=True)
    policy_state = state["policy"]
    if any(key.startswith("question_attention.") for key in policy_state):
        raise ValueError("This visualizer targets image_only policy checkpoints")
    num_prototypes = int(state.get("num_prototypes", policy_state["prototypes"].shape[0]))
    num_heads = int(state.get("attention_heads", 8))

    questions = load_mapverse_questions(
        args.dataset, args.mapverse_root, mapwise_root=args.mapwise_root
    )
    for index in args.dataset_indices:
        if index < 1 or index > len(questions):
            parser.error(f"dataset index {index} is outside 1..{len(questions)}")

    model, processor = load_model(args.model, args.adapter, load_in_4bit=args.load_in_4bit)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    device = model.get_input_embeddings().weight.device
    policy = PrototypeCrossAttentionPolicy(
        decoder_hidden_size(model), num_prototypes=num_prototypes,
        num_heads=num_heads, train_projections=False,
    ).to(device).eval()
    missing, unexpected = policy.load_state_dict(policy_state, strict=False)
    allowed_missing = {"cross_attention.in_proj_weight", "cross_attention.out_proj.weight"}
    if unexpected or set(missing) - allowed_missing:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_summaries = []
    image_signatures: dict[str, np.ndarray] = {}
    contextual_signatures: dict[str, torch.Tensor] = {}
    for dataset_index in args.dataset_indices:
        item = questions[dataset_index - 1]
        with Image.open(item.image_path) as image_file:
            image = _resize_to_pixel_cap(image_file.convert("RGB"), args.max_pixels)
        source_inputs = _content_inputs(
            processor, [image], build_cartographic_prompt(item.question), device, args.max_pixels
        )
        global_features = encode_global_image(model, source_inputs)
        image_tokens = global_features.pooler_output[0].to(device)
        with torch.inference_mode():
            memory, per_head = policy(image_tokens)
            augmented, _ = build_prototype_prefill(
                model, processor.tokenizer, source_inputs, global_features,
                policy, "after_question", with_grad=False,
            )
            decoder_output = model(
                **augmented, use_cache=False, return_dict=True,
                output_hidden_states=True,
            )
        image_token_id = int(qwen_lm(model).config.image_token_id)
        image_positions = augmented["input_ids"][0].eq(image_token_id).nonzero().flatten()
        if image_positions.numel() < num_prototypes:
            raise RuntimeError("Augmented prompt does not contain all prototype visual positions")
        prototype_positions = image_positions[-num_prototypes:]
        contextual_memory = decoder_output.hidden_states[-1][0, prototype_positions].detach().cpu()

        grid_t, raw_h, raw_w = map(int, source_inputs["image_grid_thw"][0].tolist())
        merge = int(qwen_lm(model).config.vision_config.spatial_merge_size)
        if raw_h % merge or raw_w % merge:
            raise RuntimeError(f"Image grid {(grid_t, raw_h, raw_w)} is not divisible by merge={merge}")
        grid_h, grid_w = raw_h // merge, raw_w // merge
        token_count = grid_t * grid_h * grid_w
        if image_tokens.shape[0] != token_count:
            raise RuntimeError(
                f"Merger token count {image_tokens.shape[0]} != grid {grid_t}*{grid_h}*{grid_w}"
            )

        # [1,heads,P,N] -> [heads,P,T,H,W], then average temporal frames.
        head_maps = per_head[0].float().cpu().reshape(
            num_heads, num_prototypes, grid_t, grid_h, grid_w
        ).mean(dim=2).numpy()
        average_maps = head_maps.mean(axis=0)
        flat_average = torch.from_numpy(average_maps.reshape(num_prototypes, -1))
        metrics = attention_metrics(flat_average, grid_h, grid_w)
        memory_cosine = cosine_matrix(memory.detach().cpu())
        contextual_memory_cosine = cosine_matrix(contextual_memory)
        attention_cosine = cosine_matrix(flat_average)
        memory_norms = memory.detach().float().norm(dim=-1).cpu().tolist()
        contextual_memory_norms = contextual_memory.float().norm(dim=-1).tolist()

        item_dir = args.output_dir / f"dataset_{dataset_index:04d}_{item.sample_id}"
        item_dir.mkdir(parents=True, exist_ok=True)
        image.save(item_dir / "input_image.png")
        save_average_overlays(image, average_maps, metrics,
                              item_dir / "prototype_attention_overlay.png")
        save_head_maps(head_maps, item_dir / "per_head")
        labels = [f"R{i + 1}" for i in range(num_prototypes)]
        show_matrix(memory_cosine, "Cosine similarity between R1–R9",
                    item_dir / "R_cosine_similarity.png", labels)
        show_matrix(contextual_memory_cosine,
                    "Last-decoder-layer cosine between contextualized R1–R9",
                    item_dir / "contextualized_R_cosine_similarity.png", labels)
        show_matrix(attention_cosine, "Cosine overlap between prototype attention maps",
                    item_dir / "attention_overlap.png", labels)

        image_key = str(item.image_path.resolve()).casefold()
        repeated_image_delta = None
        if image_key in image_signatures:
            repeated_image_delta = float(np.max(np.abs(image_signatures[image_key] - average_maps)))
        else:
            image_signatures[image_key] = average_maps.copy()
        repeated_contextual_cosines = None
        if image_key in contextual_signatures:
            previous = torch.nn.functional.normalize(contextual_signatures[image_key].float(), dim=-1)
            current = torch.nn.functional.normalize(contextual_memory.float(), dim=-1)
            repeated_contextual_cosines = (previous * current).sum(dim=-1).tolist()
        else:
            contextual_signatures[image_key] = contextual_memory.clone()
        summary = {
            "dataset_index": dataset_index, "qa_id": item.sample_id,
            "image": str(item.image_path), "question": item.question,
            "answer": item.answer, "checkpoint": str(args.checkpoint),
            "checkpoint_epoch": state.get("epoch"), "checkpoint_step": state.get("step"),
            "policy_kind": "image_only", "placement": state.get("placement"),
            "important_interpretation": (
                "Prototype-to-image attention is conditioned on image tokens only. "
                "after_question changes decoder placement, not extraction attention."
            ),
            "input_image_size": [image.width, image.height],
            "premerge_grid_thw": [grid_t, raw_h, raw_w],
            "merger_grid_thw": [grid_t, grid_h, grid_w],
            "memory_norms": memory_norms,
            "contextualized_memory_norms": contextual_memory_norms,
            "prototype_metrics": metrics,
            "memory_cosine": memory_cosine.tolist(),
            "contextualized_memory_cosine": contextual_memory_cosine.tolist(),
            "attention_cosine": attention_cosine.tolist(),
            "same_image_previous_question_max_attention_delta": repeated_image_delta,
            "same_image_previous_question_contextualized_R_cosines": repeated_contextual_cosines,
        }
        (item_dir / "metrics.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        all_summaries.append(summary)
        print(f"Saved dataset index {dataset_index}: {item_dir}", flush=True)

    (args.output_dir / "summary.json").write_text(
        json.dumps(all_summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"COMPLETE: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
