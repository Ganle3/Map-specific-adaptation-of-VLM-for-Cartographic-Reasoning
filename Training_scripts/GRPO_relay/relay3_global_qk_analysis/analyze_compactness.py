#!/usr/bin/env python3
"""Quantify whether Relay3 hidden-query clusters are more compact."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
VLM_ROOT = HERE.parents[2]
RELAY_ROOT = HERE.parent
if str(RELAY_ROOT) not in sys.path:
    sys.path.insert(0, str(RELAY_ROOT))

from data import MapQuestion, evaluate_answer


DEFAULT_INPUT = VLM_ROOT / "Training_outputs/Prototype/relay3_global_qk_step4000"
DEFAULT_OUTPUT = DEFAULT_INPUT / "analysis"
DEFAULT_MAPVERSE = VLM_ROOT / "Datasets/Subset_MapVerse/scaling_1500/MapVerse_test_600_quality_screened.json"
DEFAULT_MAPWISE = VLM_ROOT / "Datasets/Processed_Mapwise/Test/mapwise_grpo_test_400.json"
COLORS = {"adapter_only": "#3274b9", "relay3": "#d64a3a"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--mapverse-dataset", type=Path, default=DEFAULT_MAPVERSE)
    p.add_argument("--mapwise-dataset", type=Path, default=DEFAULT_MAPWISE)
    p.add_argument("--steps", type=int, nargs="+", default=(0, 4, 9, 19))
    p.add_argument("--layer-start", type=int, default=8)
    p.add_argument("--layer-end", type=int, default=32)
    return p.parse_args()


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def correlation(x, y, rank=False):
    x, y = np.asarray(x, float), np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    if rank:
        x = np.argsort(np.argsort(x, kind="mergesort"), kind="mergesort")
        y = np.argsort(np.argsort(y, kind="mergesort"), kind="mergesort")
    return float(np.corrcoef(x, y)[0, 1])


def questions(path: Path, dataset: str):
    rows = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for index, row in enumerate(rows, 1):
        metadata = dict(row); metadata["dataset"] = dataset
        if dataset == "mapverse":
            answer, answer_type = str(row["correct_answer"]), str(row.get("answer_type", "Single Entity"))
            sample_id = str(row.get("sample_id", index - 1))
        else:
            answer, answer_type = str(row["ground_truth"]), str(row["ground_truth_type"])
            sample_id = str(row.get("qa_id", f"mapwise_{row['country']}_{row['map_no']}_{index - 1}"))
            metadata.setdefault("qa_id", sample_id)
        result[index] = MapQuestion(sample_id, Path(), str(row["question"]), answer,
                                    answer_type, metadata)
    return result


def load_record(root, dataset, variant, index, item):
    folder = root / dataset / variant / f"index{index:04d}"
    meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    with np.load(folder / "qk_stats.npz") as z:
        arrays = {key: z[key].copy() for key in z.files}
    score = evaluate_answer(meta["response"], item)
    return {"meta": meta, "arrays": arrays, "reward": float(score["reward"])}


def selected_queries(record, requested_steps):
    available = {int(step): i for i, step in enumerate(record["arrays"]["projection_steps"])}
    steps = [step for step in requested_steps if step in available]
    return steps, np.stack([
        record["arrays"]["query_projection"][available[step]].astype(np.float64)
        for step in steps
    ]) if steps else None


def cluster_stats(vectors):
    vectors = unit(vectors.reshape(-1, vectors.shape[-1]))
    centroid = unit(vectors.mean(axis=0, keepdims=True))[0]
    cosine_radius = float(np.mean(1 - vectors @ centroid))
    centered = vectors - vectors.mean(axis=0)
    covariance = centered.T @ centered / max(len(vectors) - 1, 1)
    eigenvalues = np.linalg.eigvalsh(covariance).clip(min=0)
    mass = eigenvalues.sum()
    probabilities = eigenvalues / max(mass, 1e-30)
    nonzero = probabilities[probabilities > 0]
    effective_rank = float(np.exp(-(nonzero * np.log(nonzero)).sum()))
    participation_ratio = float(mass * mass / max(np.square(eigenvalues).sum(), 1e-30))
    return cosine_radius, effective_rank, participation_ratio


def temporal_stats(q):
    # q: fixed steps x layers x projection dim
    q = unit(q)
    centroid = unit(q.mean(axis=0))
    dispersion = (1 - np.einsum("sld,ld->sl", q, centroid)).mean(axis=0)
    if len(q) > 1:
        path = (1 - np.einsum("sld,sld->sl", q[:-1], q[1:])).sum(axis=0)
    else:
        path = np.zeros(q.shape[1])
    return dispersion, path


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    args = parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    source = {
        "mapverse": questions(args.mapverse_dataset, "mapverse"),
        "mapwise": questions(args.mapwise_dataset, "mapwise"),
    }
    records = []
    query_pool = {variant: defaultdict(list) for variant in COLORS}
    for dataset, items in source.items():
        for index, item in items.items():
            pair = {variant: load_record(args.input, dataset, variant, index, item)
                    for variant in COLORS}
            steps = sorted(set(selected_queries(pair["adapter_only"], args.steps)[0])
                           & set(selected_queries(pair["relay3"], args.steps)[0]))
            if not steps:
                continue
            per_variant = {}
            for variant in COLORS:
                available = {int(step): i for i, step in enumerate(
                    pair[variant]["arrays"]["projection_steps"]
                )}
                q = np.stack([pair[variant]["arrays"]["query_projection"][available[s]]
                              for s in steps]).astype(np.float64)
                dispersion, path = temporal_stats(q)
                per_variant[variant] = {
                    "q": q, "dispersion": dispersion, "path": path,
                }
                for layer in range(q.shape[1]):
                    query_pool[variant][layer].append(q[:, layer])
            a, r = pair["adapter_only"], pair["relay3"]
            layers = slice(args.layer_start, args.layer_end)
            records.append({
                "dataset": dataset, "dataset_index": index,
                "ability_level": item.metadata.get("ability_level", "unknown"),
                "adapter_reward": a["reward"], "relay_reward": r["reward"],
                "adapter_tokens": a["meta"]["generated_tokens"],
                "relay_tokens": r["meta"]["generated_tokens"],
                "token_saving": a["meta"]["generated_tokens"] - r["meta"]["generated_tokens"],
                "both_terminated": int(a["meta"]["terminated"] and r["meta"]["terminated"]),
                "both_correct": int(a["reward"] == 1 and r["reward"] == 1
                                    and a["meta"]["terminated"] and r["meta"]["terminated"]),
                "adapter_temporal_dispersion": float(per_variant["adapter_only"]["dispersion"][layers].mean()),
                "relay_temporal_dispersion": float(per_variant["relay3"]["dispersion"][layers].mean()),
                "dispersion_reduction": float((per_variant["adapter_only"]["dispersion"][layers]
                                               - per_variant["relay3"]["dispersion"][layers]).mean()),
                "adapter_path_length": float(per_variant["adapter_only"]["path"][layers].mean()),
                "relay_path_length": float(per_variant["relay3"]["path"][layers].mean()),
                "path_length_reduction": float((per_variant["adapter_only"]["path"][layers]
                                                - per_variant["relay3"]["path"][layers]).mean()),
            })

    layer_rows = []
    for layer in sorted(query_pool["adapter_only"]):
        for variant in COLORS:
            vectors = np.stack(query_pool[variant][layer])
            radius, rank, participation = cluster_stats(vectors)
            layer_rows.append({"layer": layer, "variant": variant,
                               "cosine_radius": radius, "effective_rank": rank,
                               "participation_ratio": participation,
                               "questions": len(vectors), "points": int(np.prod(vectors.shape[:2]))})
    write_csv(args.output / "query_cluster_compactness_by_layer.csv", layer_rows)
    write_csv(args.output / "per_question_query_dynamics.csv", records)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for metric, ax, title in [
        ("cosine_radius", axes[0], "Global query-cluster cosine radius"),
        ("effective_rank", axes[1], "Query covariance effective rank"),
        ("participation_ratio", axes[2], "Query covariance participation ratio"),
    ]:
        for variant in COLORS:
            subset = [row for row in layer_rows if row["variant"] == variant]
            ax.plot([row["layer"] for row in subset], [row[metric] for row in subset],
                    color=COLORS[variant], label=variant)
        ax.set_title(title); ax.set_xlabel("Decoder layer"); ax.grid(alpha=.2)
    axes[0].legend(frameon=False)
    fig.suptitle("Matched questions and fixed inference steps 0, 4, 9, 19")
    fig.tight_layout(); fig.savefig(args.output / "query_cluster_compactness.png", dpi=180)
    plt.close(fig)

    terminated = [row for row in records if row["both_terminated"]]
    correct = [row for row in records if row["both_correct"]]
    stats = {}
    for name, subset in (("both_terminated", terminated), ("both_correct", correct)):
        saving = [row["token_saving"] for row in subset]
        stats[name] = {"n": len(subset)}
        for metric in ("dispersion_reduction", "path_length_reduction"):
            values = [row[metric] for row in subset]
            stats[name][metric] = {
                "pearson_with_token_saving": correlation(values, saving),
                "spearman_with_token_saving": correlation(values, saving, rank=True),
                "mean": float(np.mean(values)),
            }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric, label in [
        (axes[0], "dispersion_reduction", "Temporal dispersion reduction"),
        (axes[1], "path_length_reduction", "Query path-length reduction"),
    ]:
        for dataset, marker in (("mapverse", "o"), ("mapwise", "s")):
            subset = [row for row in terminated if row["dataset"] == dataset]
            ax.scatter([row[metric] for row in subset], [row["token_saving"] for row in subset],
                       s=16, alpha=.3, marker=marker, label=dataset)
        ax.axhline(0, color="#888", lw=.7); ax.axvline(0, color="#888", lw=.7)
        ax.set_xlabel(label + " (adapter − Relay3)")
        ax.set_ylabel("Token saving (adapter − Relay3)"); ax.grid(alpha=.15)
    axes[0].legend(frameon=False)
    fig.suptitle("Does a more compact early query trajectory predict shorter answers?")
    fig.tight_layout(); fig.savefig(args.output / "query_compactness_vs_efficiency.png", dpi=180)
    plt.close(fig)

    (args.output / "query_compactness_statistics.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"questions": len(records), "statistics": stats,
                      "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

