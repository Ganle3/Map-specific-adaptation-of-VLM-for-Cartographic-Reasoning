#!/usr/bin/env python3
"""Dataset-wide unsupervised analysis of Relay1 visual information preferences."""
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
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont, ImageOps

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
from efficient_capture import EfficientRelayPrefillCapture
from evaluate_core import load_evaluation_questions, load_policy
from only_visual_relay.layout import build_prefill
from relay_module import RelayModule
from rolling_sparse_qwen3vl import load_model
from utils import decoder_hidden_size, encode_global_image, generate_from_relays, qwen_lm


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapverse-dataset", type=Path, default=VLM_ROOT / "Datasets/Subset_MapVerse/scaling_1500/MapVerse_test_600_quality_screened.json")
    parser.add_argument("--mapwise-dataset", type=Path, default=VLM_ROOT / "Datasets/Processed_Mapwise/Test/mapwise_grpo_test_400.json")
    parser.add_argument("--mapverse-root", type=Path, default=THESIS_ROOT / "MapVerse/data/imgs")
    parser.add_argument("--mapwise-root", type=Path, default=VLM_ROOT / "Datasets/mapwise-dataset")
    parser.add_argument("--run-dir", type=Path, default=VLM_ROOT / "Training_outputs/Prototype/relay1/checkpoint")
    parser.add_argument("--output", type=Path, default=VLM_ROOT / "Training_outputs/Prototype/relay1_test_map_analysis_step4000")
    parser.add_argument("--mapverse-images", type=int, default=100)
    parser.add_argument("--mapwise-images", type=int, default=100)
    parser.add_argument("--top-patches", type=int, default=6)
    parser.add_argument("--clusters", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-pixels", type=int, default=1_000_000)
    parser.add_argument("--model")
    parser.add_argument("--no-load-in-4bit", action="store_false", dest="load_in_4bit")
    parser.add_argument("--overwrite", action="store_true")
    parser.set_defaults(load_in_4bit=True)
    return parser.parse_args()


def unique_sample(records, requested, rng):
    unique = {}
    for item in records:
        unique.setdefault(str(item.image_path.resolve()).lower(), item)
    rows = list(unique.values())
    rng.shuffle(rows)
    return rows[:min(requested, len(rows))], len(rows)


def pooled_tensor(features):
    value = features.pooler_output
    tensors = (value,) if isinstance(value, torch.Tensor) else tuple(value)
    return torch.cat(tensors, dim=0)


def resize_map(array, size=32):
    tensor = torch.from_numpy(array)[None, None]
    return F.interpolate(tensor, (size, size), mode="bilinear", align_corners=False)[0, 0].numpy()


def select_spaced(array, count, radius=1):
    h, w = array.shape
    chosen = []
    for flat in np.argsort(array.ravel())[::-1]:
        y, x = divmod(int(flat), w)
        if all(max(abs(y - py), abs(x - px)) > radius for py, px in chosen):
            chosen.append((y, x))
            if len(chosen) == count:
                break
    return chosen


def crop_patch(image, y, x, grid, margin=0.35):
    h, w = grid
    cell_w, cell_h = image.width / w, image.height / h
    x0 = max(0, int((x - margin) * cell_w)); x1 = min(image.width, int((x + 1 + margin) * cell_w))
    y0 = max(0, int((y - margin) * cell_h)); y1 = min(image.height, int((y + 1 + margin) * cell_h))
    return image.crop((x0, y0, max(x0 + 1, x1), max(y0 + 1, y1)))


def spherical_kmeans(values, clusters, seed, iterations=80):
    x = values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    rng = np.random.default_rng(seed)
    centers = x[rng.choice(len(x), clusters, replace=False)].copy()
    labels = np.zeros(len(x), dtype=np.int64)
    for _ in range(iterations):
        new_labels = np.argmax(x @ centers.T, axis=1)
        new_centers = []
        for index in range(clusters):
            members = x[new_labels == index]
            center = members.mean(axis=0) if len(members) else x[rng.integers(len(x))]
            center /= max(np.linalg.norm(center), 1e-12)
            new_centers.append(center)
        new_centers = np.stack(new_centers)
        if np.array_equal(labels, new_labels):
            centers, labels = new_centers, new_labels
            break
        centers, labels = new_centers, new_labels
    confidence = np.sum(x * centers[labels], axis=1)
    return labels, confidence


def heat_overlay(array, destination):
    normalized = array / max(float(np.quantile(array[array > 0], .99)) if np.any(array > 0) else 0, 1e-12)
    normalized = np.clip(normalized, 0, 1)
    rgb = np.stack((normalized, .15 + .75 * normalized, 1 - normalized), axis=-1)
    Image.fromarray(np.uint8(rgb * 255)).resize((640, 640), Image.Resampling.NEAREST).save(destination)


def montage(rows, destination, title, columns=5, tile=180):
    selected = rows[:25]
    canvas = Image.new("RGB", (columns * tile, ((len(selected)+columns-1)//columns)*tile + 42), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 10), title, fill="black")
    for index, row in enumerate(selected):
        image = Image.open(row["crop_path"]).convert("RGB")
        image = ImageOps.fit(image, (tile-8, tile-30), method=Image.Resampling.LANCZOS)
        x = (index % columns) * tile + 4
        y = (index // columns) * tile + 42
        canvas.paste(image, (x, y))
        color = "#d73027" if row["relay"] == 1 else "#2166ac"
        draw.rectangle((index%columns*tile+2, y, index%columns*tile+tile-3, y+tile-28), outline=color, width=4)
        draw.text((index%columns*tile+5, y+tile-27), f"R{row['relay']} {row['dataset'][:2]} {row['score']:.3g}", fill=color)
    canvas.save(destination)


def enrichment_chart(stats, destination):
    width, row_h, center = 1100, 42, 500
    canvas = Image.new("RGB", (width, 55 + row_h * len(stats)), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), "R1/R2 enrichment among top-attended patches (log2 count ratio)", fill="black")
    draw.line((center, 45, center, canvas.height - 10), fill="#555", width=2)
    maximum = max(abs(np.log2((s["relay1"]+1)/(s["relay2"]+1))) for s in stats)
    scale = 300 / max(maximum, 1e-6)
    for row, stat in enumerate(stats):
        y = 52 + row * row_h
        value = float(np.log2((stat["relay1"] + 1) / (stat["relay2"] + 1)))
        draw.text((10, y+7), f"cluster {stat['cluster']:02d}", fill="black")
        end = center + int(value * scale)
        color = "#d73027" if value >= 0 else "#2166ac"
        draw.rectangle((min(center, end), y+5, max(center, end), y+28), fill=color)
        draw.text((end + (7 if value >= 0 else -68), y+7), f"{value:+.2f}", fill=color)
        draw.text((width-180, y+7), f"R1 {stat['relay1']} / R2 {stat['relay2']}", fill="#444")
    canvas.save(destination)


def main():
    args = arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    rng = random.Random(args.seed)
    pools = {
        "mapverse": load_evaluation_questions(args.mapverse_dataset, args.mapverse_root, args.mapwise_root),
        "mapwise": load_evaluation_questions(args.mapwise_dataset, args.mapverse_root, args.mapwise_root),
    }
    selected = {}
    unique_counts = {}
    for dataset, requested in (("mapverse", args.mapverse_images), ("mapwise", args.mapwise_images)):
        selected[dataset], unique_counts[dataset] = unique_sample(pools[dataset], requested, rng)
        print(f"[selection] {dataset}: {len(selected[dataset])}/{unique_counts[dataset]} unique test images", flush=True)
    if args.output.exists() and args.overwrite:
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "raw").mkdir(); (args.output / "patches").mkdir(); (args.output / "clusters").mkdir()

    checkpoint = args.run_dir / "checkpoint-4000.pt"
    adapter = args.run_dir / "checkpoint-4000"
    model, processor = load_model(args.model, adapter, load_in_4bit=args.load_in_4bit)
    model.eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    policy = RelayModule(decoder_hidden_size(model), num_relays=2).to(model.get_input_embeddings().weight.device).eval()
    load_policy(policy, checkpoint, "after_vision_end", "only_visual")
    qwen = qwen_lm(model)
    spatial = {dataset: np.zeros((2, 32, 32), np.float64) for dataset in selected}
    patch_rows, embeddings = [], []
    processed = 0
    try:
        for dataset, records in selected.items():
            for local_index, item in enumerate(records):
                with Image.open(item.image_path) as handle:
                    original = handle.convert("RGB")
                    submitted = _resize_to_pixel_cap(original, args.max_pixels)
                source = _content_inputs(processor, [submitted], build_cartographic_prompt(item.question),
                                         model.get_input_embeddings().weight.device, args.max_pixels)
                features = encode_global_image(model, source)
                augmented, _ = build_prefill(model, processor.tokenizer, source, features, policy, with_grad=False)
                image_positions = augmented["input_ids"][0].eq(int(qwen.config.image_token_id)).nonzero().flatten()
                relay_positions = augmented["relay_positions"].flatten()
                t, h, w = map(int, augmented["image_grid_thw"][0].tolist())
                merge = int(qwen.config.vision_config.spatial_merge_size); grid = (h//merge, w//merge)
                with EfficientRelayPrefillCapture(model, image_positions, relay_positions, grid) as capture:
                    generate_from_relays(model, **augmented, do_sample=False, num_beams=1,
                                         max_new_tokens=1, use_cache=True)
                maps = capture.arrays()
                mean_maps = maps.mean(axis=1)
                pooled = pooled_tensor(features).detach().float().cpu().numpy()
                if pooled.shape[0] != grid[0] * grid[1]:
                    raise RuntimeError("Merger embedding grid mismatch")
                raw_name = f"{dataset}_{local_index:03d}"
                np.savez_compressed(args.output / "raw" / f"{raw_name}.npz", relay_attention=maps)
                for relay in range(2):
                    normalized = mean_maps[relay] / max(float(mean_maps[relay].sum()), 1e-12)
                    spatial[dataset][relay] += resize_map(normalized)
                    for rank, (y, x) in enumerate(select_spaced(normalized, args.top_patches)):
                        crop_name = f"{raw_name}_r{relay+1}_{rank}.jpg"
                        crop_path = args.output / "patches" / crop_name
                        crop_patch(original, y, x, grid).save(crop_path, quality=92)
                        flat = y * grid[1] + x
                        patch_rows.append({
                            "dataset": dataset, "image": str(item.image_path), "qa_id": item.sample_id,
                            "relay": relay+1, "rank": rank, "patch_y": y, "patch_x": x,
                            "score": float(normalized[y, x]), "crop_path": str(crop_path),
                        })
                        embeddings.append(pooled[flat])
                processed += 1
                print(f"[{processed:03d}/{sum(map(len, selected.values()))}] {dataset} {item.image_path.name} grid={grid}", flush=True)
                del source, features, augmented, maps, pooled
    finally:
        del policy, model, processor
        gc.collect(); torch.cuda.empty_cache()

    values = np.stack(embeddings).astype(np.float32)
    labels, confidence = spherical_kmeans(values, args.clusters, args.seed)
    for row, label, score in zip(patch_rows, labels, confidence):
        row["cluster"] = int(label); row["cluster_cosine"] = float(score)
    with (args.output / "patch_assignments.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(patch_rows[0])); writer.writeheader(); writer.writerows(patch_rows)
    stats = []
    for cluster in range(args.clusters):
        rows = [row for row in patch_rows if row["cluster"] == cluster]
        rows.sort(key=lambda row: row["cluster_cosine"], reverse=True)
        montage(rows, args.output / "clusters" / f"cluster_{cluster:02d}.jpg", f"Cluster {cluster}: red=R1, blue=R2")
        stats.append({
            "cluster": cluster, "count": len(rows),
            "relay1": sum(row["relay"] == 1 for row in rows),
            "relay2": sum(row["relay"] == 2 for row in rows),
            "mapverse": sum(row["dataset"] == "mapverse" for row in rows),
            "mapwise": sum(row["dataset"] == "mapwise" for row in rows),
            "manual_semantic_label": "",
        })
    enrichment_chart(stats, args.output / "relay_cluster_enrichment.png")
    for dataset, array in spatial.items():
        for relay in range(2):
            array[relay] /= len(selected[dataset])
            heat_overlay(array[relay], args.output / f"spatial_bias_{dataset}_r{relay+1}.png")
    np.savez_compressed(args.output / "spatial_bias.npz", **spatial)
    summary = {
        "checkpoint": str(checkpoint), "seed": args.seed,
        "requested": {"mapverse": args.mapverse_images, "mapwise": args.mapwise_images},
        "available_unique": unique_counts,
        "analyzed_unique": {key: len(value) for key, value in selected.items()},
        "top_patches_per_relay_per_image": args.top_patches,
        "clusters": stats,
        "interpretation": "Clusters are deliberately unlabeled. Inspect each montage, then fill manual_semantic_label; no legend/text ground truth was assumed.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    cards = "".join(
        f'<section><h2>Cluster {s["cluster"]} — R1 {s["relay1"]}, R2 {s["relay2"]}</h2>'
        f'<img src="clusters/cluster_{s["cluster"]:02d}.jpg"></section>' for s in stats
    )
    bias = "".join(f'<figure><img src="spatial_bias_{d}_r{r}.png"><figcaption>{d} R{r}</figcaption></figure>' for d in selected for r in (1,2))
    (args.output / "report.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Relay1 dataset analysis</title><style>'
        'body{font:16px system-ui;max-width:1400px;margin:30px auto}img{max-width:100%}'
        '.bias{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}section{border-top:1px solid #ccc;padding:15px}</style>'
        '<h1>Relay1 unsupervised test-map analysis</h1><p>Spatial maps show normalized positional bias. '
        'Cluster montages show top-attended patch types; red borders are R1 and blue borders are R2. '
        'Cluster names are intentionally left for manual semantic interpretation.</p>'
        '<img src="relay_cluster_enrichment.png"><div class="bias">'
        + bias + '</div>' + cards, encoding="utf-8"
    )
    print(args.output / "report.html")


if __name__ == "__main__":
    main()
