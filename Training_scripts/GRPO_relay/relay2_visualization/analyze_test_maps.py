#!/usr/bin/env python3
"""Dataset-wide Relay2 analysis with image-side/question-side streams kept separate."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

HERE = Path(__file__).resolve().parent
RELAY_ROOT = HERE.parent
VLM_ROOT = HERE.parents[2]
THESIS_ROOT = VLM_ROOT.parent
SPARSE_ROOT = VLM_ROOT / "SparseVLM_eliminateCONTEXT"
for path in (HERE, RELAY_ROOT, SPARSE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from continuous_cache_qwen3vl import _content_inputs, _resize_to_pixel_cap
from data import build_cartographic_prompt
from evaluate_core import load_evaluation_questions, load_policy
from Image_question_shared_relay.layout import build_prefill
from relay1_visualization.analyze_test_maps import (
    crop_patch, heat_overlay, pooled_tensor, resize_map, select_spaced,
    spherical_kmeans, unique_sample,
)
from relay1_visualization.efficient_capture import EfficientRelayPrefillCapture
from relay_module import RelayModule
from rolling_sparse_qwen3vl import load_model
from utils import decoder_hidden_size, encode_global_image, generate_from_relays, qwen_lm

STREAMS = (
    "image_R1", "image_R2", "question_R1", "question_R2",
)
COLORS = {
    "image_R1": "#d73027", "image_R2": "#2166ac",
    "question_R1": "#fdae61", "question_R2": "#1a9850",
}


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mapverse-dataset", type=Path, default=VLM_ROOT / "Datasets/Subset_MapVerse/scaling_1500/MapVerse_test_600_quality_screened.json")
    p.add_argument("--mapwise-dataset", type=Path, default=VLM_ROOT / "Datasets/Processed_Mapwise/Test/mapwise_grpo_test_400.json")
    p.add_argument("--mapverse-root", type=Path, default=THESIS_ROOT / "MapVerse/data/imgs")
    p.add_argument("--mapwise-root", type=Path, default=VLM_ROOT / "Datasets/mapwise-dataset")
    p.add_argument("--run-dir", type=Path, default=VLM_ROOT / "Training_outputs/Prototype/relay2/checkpoint")
    p.add_argument("--checkpoint-step", type=int, default=2000)
    p.add_argument("--output", type=Path)
    p.add_argument("--mapverse-images", type=int, default=100)
    p.add_argument("--mapwise-images", type=int, default=100)
    p.add_argument("--top-patches", type=int, default=6)
    p.add_argument("--clusters", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-pixels", type=int, default=1_000_000)
    p.add_argument("--model")
    p.add_argument("--no-load-in-4bit", action="store_false", dest="load_in_4bit")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(load_in_4bit=True)
    return p.parse_args()


def normalized(array):
    value = np.maximum(array.astype(np.float64), 0)
    return value / max(float(value.sum()), 1e-12)


def shift_metrics(left, right):
    p, q = normalized(left).ravel(), normalized(right).ravel()
    m = (p + q) / 2
    js = .5 * np.sum(np.where(p > 0, p * np.log((p + 1e-12) / (m + 1e-12)), 0))
    js += .5 * np.sum(np.where(q > 0, q * np.log((q + 1e-12) / (m + 1e-12)), 0))
    cosine = float(np.dot(p, q) / max(np.linalg.norm(p) * np.linalg.norm(q), 1e-12))
    count = max(1, int(np.ceil(.1 * p.size)))
    top_p = set(np.argpartition(p, -count)[-count:].tolist())
    top_q = set(np.argpartition(q, -count)[-count:].tolist())
    overlap = len(top_p & top_q) / len(top_p | top_q)
    return float(js), cosine, float(overlap)


def montage(rows, destination, title, columns=5, tile=180):
    selected = rows[:25]
    height = ((len(selected) + columns - 1) // columns) * tile + 42
    canvas = Image.new("RGB", (columns * tile, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 10), title, fill="black")
    for index, row in enumerate(selected):
        image = Image.open(row["crop_path"]).convert("RGB")
        image = ImageOps.fit(image, (tile - 8, tile - 30), method=Image.Resampling.LANCZOS)
        x = (index % columns) * tile + 4
        y = (index // columns) * tile + 42
        canvas.paste(image, (x, y))
        color = COLORS[row["stream"]]
        draw.rectangle((x - 2, y, x + tile - 7, y + tile - 28), outline=color, width=4)
        draw.text((x + 1, y + tile - 27), f"{row['stream']} {row['dataset'][:2]}", fill=color)
    canvas.save(destination)


def composition_chart(stats, destination):
    width, row_h = 1050, 54
    canvas = Image.new("RGB", (width, 55 + row_h * len(stats)), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), "Cluster composition across Relay2 placement/token streams", fill="black")
    for row_index, stat in enumerate(stats):
        y = 52 + row_index * row_h
        draw.text((10, y + 10), f"cluster {stat['cluster']:02d}", fill="black")
        x = 120
        total = max(stat[stream] for stream in STREAMS)
        for stream in STREAMS:
            length = int(170 * stat[stream] / max(total, 1))
            draw.rectangle((x, y + 5, x + length, y + 18), fill=COLORS[stream])
            draw.text((x, y + 23), f"{stream} {stat[stream]}", fill=COLORS[stream])
            x += 220
    canvas.save(destination)


def main():
    args = arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    args.output = (args.output or VLM_ROOT / f"Training_outputs/Prototype/relay2_test_map_analysis_step{args.checkpoint_step}").resolve()
    checkpoint = args.run_dir / f"checkpoint-{args.checkpoint_step}.pt"
    adapter = args.run_dir / f"checkpoint-{args.checkpoint_step}"
    if not checkpoint.is_file() or not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Missing matched policy/adapter checkpoint: {checkpoint}, {adapter}")

    rng = random.Random(args.seed)
    pools = {
        "mapverse": load_evaluation_questions(args.mapverse_dataset, args.mapverse_root, args.mapwise_root),
        "mapwise": load_evaluation_questions(args.mapwise_dataset, args.mapverse_root, args.mapwise_root),
    }
    selected, unique_counts = {}, {}
    for dataset, requested in (("mapverse", args.mapverse_images), ("mapwise", args.mapwise_images)):
        selected[dataset], unique_counts[dataset] = unique_sample(pools[dataset], requested, rng)
        print(f"[selection] {dataset}: {len(selected[dataset])}/{unique_counts[dataset]} unique maps", flush=True)
    if args.output.exists() and args.overwrite:
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=False)
    for folder in ("raw", "patches", "clusters"):
        (args.output / folder).mkdir()

    model, processor = load_model(args.model, adapter, load_in_4bit=args.load_in_4bit)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    policy = RelayModule(decoder_hidden_size(model), num_relays=2).to(model.get_input_embeddings().weight.device).eval()
    state = load_policy(policy, checkpoint, "shared_dual_placement", "image_question_shared")
    if int(state.get("step", -1)) != args.checkpoint_step:
        raise ValueError("Checkpoint step mismatch")
    qwen = qwen_lm(model)
    spatial = {dataset: np.zeros((4, 32, 32), np.float64) for dataset in selected}
    patch_rows, embeddings, shifts = [], [], []
    processed = 0
    try:
        for dataset, records in selected.items():
            for local_index, item in enumerate(records):
                with Image.open(item.image_path) as handle:
                    original = handle.convert("RGB")
                    submitted = _resize_to_pixel_cap(original, args.max_pixels)
                source = _content_inputs(processor, [submitted], build_cartographic_prompt(item.question), model.get_input_embeddings().weight.device, args.max_pixels)
                features = encode_global_image(model, source)
                augmented, _ = build_prefill(model, processor.tokenizer, source, features, policy, with_grad=False)
                image_positions = augmented["input_ids"][0].eq(int(qwen.config.image_token_id)).nonzero().flatten()
                relay_positions = augmented["relay_positions"]
                _, h, w = map(int, augmented["image_grid_thw"][0].tolist())
                merge = int(qwen.config.vision_config.spatial_merge_size)
                grid = (h // merge, w // merge)
                with EfficientRelayPrefillCapture(model, image_positions, relay_positions.flatten(), grid) as capture:
                    generate_from_relays(model, **augmented, do_sample=False, num_beams=1, max_new_tokens=1, use_cache=True)
                flat = capture.arrays()
                maps = flat.reshape(2, 2, flat.shape[1], *grid)
                stream_maps = maps.reshape(4, maps.shape[2], *grid)
                mean_maps = stream_maps.mean(axis=1)
                pooled = pooled_tensor(features).detach().float().cpu().numpy()
                if pooled.shape[0] != grid[0] * grid[1]:
                    raise RuntimeError("Merger embedding grid mismatch")
                raw_name = f"{dataset}_{local_index:03d}"
                np.savez_compressed(args.output / "raw" / f"{raw_name}.npz", relay_attention=maps)

                for relay in range(2):
                    js, cosine, overlap = shift_metrics(mean_maps[relay], mean_maps[2 + relay])
                    shifts.append({"dataset": dataset, "image": str(item.image_path), "qa_id": item.sample_id,
                                   "relay": relay + 1, "js_divergence": js, "cosine": cosine,
                                   "top10_jaccard": overlap})
                for stream_index, stream in enumerate(STREAMS):
                    attention = normalized(mean_maps[stream_index])
                    spatial[dataset][stream_index] += resize_map(attention)
                    for rank, (y, x) in enumerate(select_spaced(attention, args.top_patches)):
                        crop_name = f"{raw_name}_{stream}_{rank}.jpg"
                        crop_path = args.output / "patches" / crop_name
                        crop_patch(original, y, x, grid).save(crop_path, quality=92)
                        flat_index = y * grid[1] + x
                        placement, relay = ("image", stream_index + 1) if stream_index < 2 else ("question", stream_index - 1)
                        patch_rows.append({
                            "dataset": dataset, "image": str(item.image_path), "qa_id": item.sample_id,
                            "placement": placement, "relay": relay, "stream": stream, "rank": rank,
                            "patch_y": y, "patch_x": x, "score": float(attention[y, x]),
                            "crop_path": str(crop_path),
                        })
                        embeddings.append(pooled[flat_index])
                processed += 1
                print(f"[{processed:03d}/{sum(map(len, selected.values()))}] {dataset} {item.image_path.name} grid={grid}", flush=True)
                del source, features, augmented, flat, maps, pooled
    finally:
        del policy, model, processor
        gc.collect()
        torch.cuda.empty_cache()

    values = np.stack(embeddings).astype(np.float32)
    labels, confidence = spherical_kmeans(values, args.clusters, args.seed)
    for row, label, score in zip(patch_rows, labels, confidence):
        row["cluster"] = int(label)
        row["cluster_cosine"] = float(score)
    with (args.output / "patch_assignments.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(patch_rows[0]))
        writer.writeheader(); writer.writerows(patch_rows)
    with (args.output / "placement_shift_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(shifts[0]))
        writer.writeheader(); writer.writerows(shifts)

    stats = []
    for cluster in range(args.clusters):
        rows = [row for row in patch_rows if row["cluster"] == cluster]
        rows.sort(key=lambda row: row["cluster_cosine"], reverse=True)
        montage(rows, args.output / "clusters" / f"cluster_{cluster:02d}.jpg", f"Cluster {cluster}: four Relay2 streams")
        stats.append({"cluster": cluster, "count": len(rows),
                      **{stream: sum(row["stream"] == stream for row in rows) for stream in STREAMS},
                      "mapverse": sum(row["dataset"] == "mapverse" for row in rows),
                      "mapwise": sum(row["dataset"] == "mapwise" for row in rows),
                      "manual_semantic_label": ""})
    composition_chart(stats, args.output / "cluster_stream_composition.png")
    for dataset, array in spatial.items():
        array /= len(selected[dataset])
        for stream_index, stream in enumerate(STREAMS):
            heat_overlay(array[stream_index], args.output / f"spatial_bias_{dataset}_{stream}.png")
    np.savez_compressed(args.output / "spatial_bias.npz", **spatial)

    shift_summary = {}
    for relay in (1, 2):
        rows = [row for row in shifts if row["relay"] == relay]
        shift_summary[f"R{relay}"] = {key: float(np.mean([row[key] for row in rows]))
                                      for key in ("js_divergence", "cosine", "top10_jaccard")}
    summary = {
        "checkpoint": str(checkpoint), "checkpoint_step": args.checkpoint_step,
        "shared_parameter_across_placements": True,
        "streams": list(STREAMS), "seed": args.seed,
        "available_unique": unique_counts,
        "analyzed_unique": {key: len(value) for key, value in selected.items()},
        "top_patches_per_stream_per_image": args.top_patches,
        "image_to_question_placement_shift": shift_summary,
        "clusters": stats,
        "interpretation": "Clusters are unlabeled. The question-side rows use the same prototype Parameters after causal contextualization by image, image-side relay, and question tokens.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    cards = "".join(f'<section><h2>Cluster {s["cluster"]}</h2><img src="clusters/cluster_{s["cluster"]:02d}.jpg"></section>' for s in stats)
    bias = "".join(f'<figure><img src="spatial_bias_{dataset}_{stream}.png"><figcaption>{dataset} {stream}</figcaption></figure>' for dataset in selected for stream in STREAMS)
    metrics = "".join(f'<li>{relay}: JS={values["js_divergence"]:.4f}, cosine={values["cosine"]:.4f}, top-10% Jaccard={values["top10_jaccard"]:.4f}</li>' for relay, values in shift_summary.items())
    (args.output / "report.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Relay2 dataset analysis</title><style>'
        'body{font:16px system-ui;max-width:1500px;margin:30px auto}img{max-width:100%}.bias{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}section{border-top:1px solid #ccc;padding:15px}</style>'
        '<h1>Relay2 shared dual-placement analysis</h1><p>The four streams remain separate: image-side R1/R2 and question-side R1/R2. '
        'The latter two are question-conditioned decoder states of the same shared Parameters.</p><h2>Image-to-question placement shift</h2><ul>'
        + metrics + '</ul><img src="cluster_stream_composition.png"><div class="bias">' + bias + '</div>' + cards,
        encoding="utf-8")
    print(args.output / "report.html")


if __name__ == "__main__":
    main()
