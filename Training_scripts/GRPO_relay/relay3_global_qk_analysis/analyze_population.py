#!/usr/bin/env python3
"""Analyze population-level hidden-query/image-key alignment and efficiency."""
from __future__ import annotations

import argparse
import csv
import json
import math
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
STEP_BINS = ((0, 5), (5, 10), (10, 20), (20, 40), (40, 80),
             (80, 160), (160, 320), (320, 10_000))
COLORS = {"adapter_only": "#3274b9", "relay3": "#d64a3a"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--mapverse-dataset", type=Path, default=DEFAULT_MAPVERSE)
    p.add_argument("--mapwise-dataset", type=Path, default=DEFAULT_MAPWISE)
    p.add_argument("--bootstrap-samples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20261117)
    p.add_argument("--early-steps", type=int, default=20)
    p.add_argument("--early-layer-start", type=int, default=8)
    p.add_argument("--early-layer-end", type=int, default=32, help="Exclusive")
    p.add_argument("--cluster-layers", type=int, nargs="+", default=(8, 16, 24, 35))
    return p.parse_args()


def load_variant(root: Path, dataset: str, variant: str) -> dict[int, dict]:
    result = {}
    for folder in sorted((root / dataset / variant).glob("index[0-9][0-9][0-9][0-9]")):
        metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        with np.load(folder / "qk_stats.npz") as data:
            arrays = {name: data[name].copy() for name in data.files}
        index = int(metadata["dataset_index"])
        if index in result:
            raise ValueError(f"Duplicate {dataset}/{variant}/index{index}")
        result[index] = {"metadata": metadata, "arrays": arrays}
    if not result:
        raise FileNotFoundError(f"No captures below {root / dataset / variant}")
    return result


def validate_pair(adapter: dict, relay: dict, dataset: str) -> list[int]:
    if set(adapter) != set(relay):
        raise ValueError(f"Capture index mismatch for {dataset}")
    for index in adapter:
        a, r = adapter[index]["metadata"], relay[index]["metadata"]
        identity = ("qa_id", "question", "ground_truth")
        if any(str(a[k]) != str(r[k]) for k in identity):
            raise ValueError(f"Question mismatch at {dataset}/{index}")
        if tuple(adapter[index]["arrays"]["metric_names"]) != tuple(relay[index]["arrays"]["metric_names"]):
            raise ValueError("Metric schema mismatch")
    return sorted(adapter)


def scoring_questions(path: Path, dataset: str) -> dict[int, MapQuestion]:
    """Build evaluator inputs without reopening or re-encoding any images."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for index, row in enumerate(raw, start=1):
        metadata = dict(row)
        metadata["dataset"] = dataset
        if dataset == "mapverse":
            answer = str(row["correct_answer"])
            answer_type = str(row.get("answer_type", "Single Entity"))
            sample_id = str(row.get("sample_id", index - 1))
        else:
            answer = str(row["ground_truth"])
            answer_type = str(row["ground_truth_type"])
            sample_id = str(row.get(
                "qa_id", f"mapwise_{row['country']}_{row['map_no']}_{index - 1}"
            ))
            metadata.setdefault("qa_id", sample_id)
        result[index] = MapQuestion(
            sample_id=sample_id, image_path=Path(), question=str(row["question"]),
            answer=answer, answer_type=answer_type, metadata=metadata,
        )
    return result


def attach_fresh_scores(records: dict[int, dict], questions: dict[int, MapQuestion]):
    if set(records) != set(questions):
        raise ValueError("Captured rows and source scoring rows are not aligned")
    for index, record in records.items():
        item = questions[index]
        metadata = record["metadata"]
        if (metadata["question"] != item.question
                or str(metadata["ground_truth"]) != str(item.answer)):
            raise ValueError(f"Scoring row mismatch at dataset_index={index}")
        evaluation = evaluate_answer(metadata["response"], item)
        metadata["fresh_evaluation"] = evaluation
        metadata["fresh_reward"] = float(evaluation["reward"])


def metric_index(record: dict, name: str) -> int:
    names = [str(x) for x in record["arrays"]["metric_names"]]
    return names.index(name)


def question_layer_mean(record: dict, metric: str, steps: int) -> np.ndarray:
    values = record["arrays"]["metrics"][:, :, metric_index(record, metric)]
    return values[:min(steps, len(values))].mean(axis=0)


def bootstrap_mean_ci(values: np.ndarray, samples: int, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    mean = np.nanmean(values, axis=0)
    draws = []
    for _ in range(samples):
        selection = rng.integers(0, len(values), len(values))
        draws.append(np.nanmean(values[selection], axis=0))
    draws = np.asarray(draws)
    return mean, np.nanpercentile(draws, 2.5, axis=0), np.nanpercentile(draws, 97.5, axis=0)


def ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    result = np.empty(len(x), dtype=np.float64)
    cursor = 0
    while cursor < len(x):
        end = cursor + 1
        while end < len(x) and x[order[end]] == x[order[cursor]]:
            end += 1
        result[order[cursor:end]] = (cursor + end - 1) / 2
        cursor = end
    return result


def correlation(x: np.ndarray, y: np.ndarray, rank: bool = False) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    if rank:
        x, y = ranks(x), ranks(y)
    return float(np.corrcoef(x, y)[0, 1])


def correlation_ci(x, y, samples, rng, rank=False):
    point = correlation(x, y, rank)
    if not math.isfinite(point):
        return point, float("nan"), float("nan")
    draws = []
    for _ in range(samples):
        ids = rng.integers(0, len(x), len(x))
        draws.append(correlation(x[ids], y[ids], rank))
    return point, float(np.nanpercentile(draws, 2.5)), float(np.nanpercentile(draws, 97.5))


def early_distance(record, args, metric="attention_weighted_key_distance") -> float:
    values = record["arrays"]["metrics"][:, :, metric_index(record, metric)]
    values = values[:min(args.early_steps, len(values)),
                    args.early_layer_start:args.early_layer_end]
    return float(values.mean())


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_layer_curves(pairs, args, output, rng):
    metrics = ("nearest_key_distance", "attention_weighted_key_distance", "image_attention_mass")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    curve_rows = []
    for ax, metric in zip(axes, metrics):
        for variant in ("adapter_only", "relay3"):
            values = np.stack([
                question_layer_mean(pair[variant], metric, args.early_steps)
                for pair in pairs
            ])
            mean, low, high = bootstrap_mean_ci(values, args.bootstrap_samples, rng)
            layers = np.arange(len(mean))
            ax.plot(layers, mean, color=COLORS[variant], label=variant)
            ax.fill_between(layers, low, high, color=COLORS[variant], alpha=.18)
            for layer in layers:
                curve_rows.append({"metric": metric, "variant": variant,
                                   "layer": int(layer), "mean": float(mean[layer]),
                                   "ci_low": float(low[layer]), "ci_high": float(high[layer])})
        ax.set_title(metric.replace("_", " "))
        ax.set_xlabel("Decoder layer")
        ax.grid(alpha=.2)
    axes[0].set_ylabel(f"Mean over native greedy steps 0–{args.early_steps - 1}")
    axes[0].legend(frameon=False)
    fig.suptitle("Population hidden-query to image-key alignment (95% question bootstrap CI)")
    fig.tight_layout()
    fig.savefig(output / "early_qk_distance_by_layer.png", dpi=170)
    plt.close(fig)
    write_csv(output / "early_qk_distance_by_layer.csv", curve_rows)


def binned_matrix(record, metric: str) -> np.ndarray:
    values = record["arrays"]["metrics"][:, :, metric_index(record, metric)]
    rows = []
    for start, end in STEP_BINS:
        if start >= len(values):
            rows.append(np.full(values.shape[1], np.nan))
        else:
            rows.append(values[start:min(end, len(values))].mean(axis=0))
    return np.stack(rows)


def plot_delta_heatmap(pairs, output):
    metric = "attention_weighted_key_distance"
    stacked = np.stack([
        binned_matrix(pair["adapter_only"], metric) - binned_matrix(pair["relay3"], metric)
        for pair in pairs
    ])
    counts = np.isfinite(stacked).sum(axis=0)
    delta = np.divide(np.nansum(stacked, axis=0), counts,
                      out=np.full(stacked.shape[1:], np.nan), where=counts > 0)
    bound = max(abs(float(np.nanmin(delta))), abs(float(np.nanmax(delta))), 1e-5)
    fig, ax = plt.subplots(figsize=(12, 4.8))
    image = ax.imshow(delta, aspect="auto", cmap="RdBu_r", vmin=-bound, vmax=bound)
    ax.set_yticks(range(len(STEP_BINS)),
                  [f"{a}–{b - 1}" if b < 10_000 else f"{a}+" for a, b in STEP_BINS])
    ax.set_xlabel("Decoder layer")
    ax.set_ylabel("Native greedy inference step")
    ax.set_title("Adapter distance − Relay3 distance\npositive = Relay3 hidden query is closer to image keys")
    fig.colorbar(image, ax=ax, label="paired distance improvement")
    fig.tight_layout()
    fig.savefig(output / "paired_distance_improvement_step_layer.png", dpi=170)
    plt.close(fig)
    np.savetxt(output / "paired_distance_improvement_step_layer.csv", delta, delimiter=",")


def orthogonal_alignment(adapter_keys: np.ndarray, relay_keys: np.ndarray):
    a_mean, r_mean = adapter_keys.mean(axis=0), relay_keys.mean(axis=0)
    u, _, vt = np.linalg.svd((relay_keys - r_mean).T @ (adapter_keys - a_mean),
                             full_matrices=False)
    rotation = u @ vt
    return rotation, a_mean, r_mean


def pca_fit_transform(reference: np.ndarray, arrays: list[np.ndarray]):
    mean = reference.mean(axis=0)
    _, _, vt = np.linalg.svd(reference - mean, full_matrices=False)
    basis = vt[:2].T
    return [(array - mean) @ basis for array in arrays]


def plot_clusters(pairs, args, output):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    cluster_rows = []
    for ax, layer in zip(axes.flat, args.cluster_layers):
        a_keys, r_keys, a_queries, r_queries, query_steps = [], [], [], [], []
        for pair in pairs:
            aa, rr = pair["adapter_only"]["arrays"], pair["relay3"]["arrays"]
            if layer >= aa["key_projection"].shape[0]:
                raise ValueError(f"Invalid cluster layer {layer}")
            a_keys.append(aa["key_projection"][layer].astype(np.float64))
            r_keys.append(rr["key_projection"][layer].astype(np.float64))
            common = sorted(set(map(int, aa["projection_steps"]))
                            & set(map(int, rr["projection_steps"])))
            amap = {int(step): i for i, step in enumerate(aa["projection_steps"])}
            rmap = {int(step): i for i, step in enumerate(rr["projection_steps"])}
            for step in common:
                a_queries.append(aa["query_projection"][amap[step], layer].astype(np.float64))
                r_queries.append(rr["query_projection"][rmap[step], layer].astype(np.float64))
                query_steps.append(step)
        a_keys, r_keys = np.concatenate(a_keys), np.concatenate(r_keys)
        a_queries, r_queries = np.stack(a_queries), np.stack(r_queries)
        rotation, a_mean, r_mean = orthogonal_alignment(a_keys, r_keys)
        aligned_r_keys = (r_keys - r_mean) @ rotation + a_mean
        aligned_r_queries = (r_queries - r_mean) @ rotation + a_mean
        key_stride = max(1, len(a_keys) // 3000)
        reference = np.concatenate((a_keys[::key_stride], aligned_r_keys[::key_stride],
                                    a_queries, aligned_r_queries))
        ak2, rk2, aq2, rq2 = pca_fit_transform(
            reference, [a_keys[::key_stride], aligned_r_keys[::key_stride],
                        a_queries, aligned_r_queries]
        )
        ax.scatter(ak2[:, 0], ak2[:, 1], s=3, alpha=.10, color="#555555",
                   label="adapter image K")
        ax.scatter(rk2[:, 0], rk2[:, 1], s=3, alpha=.07, color="#b7a26a",
                   label="aligned Relay3 image K")
        steps = np.asarray(query_steps)
        sizes = np.where(steps < 20, 18, 9)
        ax.scatter(aq2[:, 0], aq2[:, 1], s=sizes, alpha=.32,
                   color=COLORS["adapter_only"], label="adapter hidden Q")
        ax.scatter(rq2[:, 0], rq2[:, 1], s=sizes, alpha=.32,
                   color=COLORS["relay3"], label="relay-affected hidden Q")
        ax.set_title(f"Decoder layer {layer}")
        ax.set_xticks([]); ax.set_yticks([])
        residual = float(np.sqrt(np.mean((aligned_r_keys - a_keys) ** 2)))
        cluster_rows.append({"layer": layer, "procrustes_key_rmse": residual,
                             "query_points_per_variant": len(a_queries),
                             "key_points_per_variant": len(a_keys)})
    axes[0, 0].legend(frameon=False, markerscale=2, fontsize=8)
    fig.suptitle("Shared per-layer Q/K space after image-key Procrustes alignment\n"
                 "larger query points are inference steps <20")
    fig.tight_layout()
    fig.savefig(output / "global_hidden_query_image_key_clusters.png", dpi=180)
    plt.close(fig)
    write_csv(output / "cluster_alignment_diagnostics.csv", cluster_rows)


def efficiency_rows(pairs, args):
    rows = []
    for pair in pairs:
        a, r = pair["adapter_only"], pair["relay3"]
        am, rm = a["metadata"], r["metadata"]
        ad, rd = early_distance(a, args), early_distance(r, args)
        rows.append({
            "dataset": am["dataset"], "dataset_index": am["dataset_index"],
            "qa_id": am["qa_id"], "ability_level": am.get("ability_level") or "unknown",
            "adapter_reward": am["fresh_reward"], "relay_reward": rm["fresh_reward"],
            "adapter_tokens": am["generated_tokens"], "relay_tokens": rm["generated_tokens"],
            "adapter_terminated": int(am["terminated"]), "relay_terminated": int(rm["terminated"]),
            "adapter_early_distance": ad, "relay_early_distance": rd,
            "relay_distance_improvement": ad - rd,
            "token_saving": am["generated_tokens"] - rm["generated_tokens"],
            "both_correct_terminated": int(am["fresh_reward"] == 1 and rm["fresh_reward"] == 1
                                           and am["terminated"] and rm["terminated"]),
        })
    return rows


def plot_efficiency(rows, args, output, rng):
    subset = [row for row in rows if row["adapter_terminated"] and row["relay_terminated"]]
    x = np.asarray([row["relay_distance_improvement"] for row in subset])
    y = np.asarray([row["token_saving"] for row in subset])
    pearson = correlation_ci(x, y, args.bootstrap_samples, rng, rank=False)
    spearman = correlation_ci(x, y, args.bootstrap_samples, rng, rank=True)
    correct_subset = [row for row in subset if row["both_correct_terminated"]]
    correct_x = np.asarray([row["relay_distance_improvement"] for row in correct_subset])
    correct_y = np.asarray([row["token_saving"] for row in correct_subset])
    correct_pearson = correlation_ci(
        correct_x, correct_y, args.bootstrap_samples, rng, rank=False
    )
    correct_spearman = correlation_ci(
        correct_x, correct_y, args.bootstrap_samples, rng, rank=True
    )
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for dataset, marker in (("mapverse", "o"), ("mapwise", "s")):
        mask = np.asarray([row["dataset"] == dataset for row in subset])
        ax.scatter(x[mask], y[mask], s=18, alpha=.35, marker=marker, label=dataset)
    if len(x) >= 2 and np.std(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        xx = np.linspace(x.min(), x.max(), 100)
        ax.plot(xx, slope * xx + intercept, color="black", lw=1.3)
    ax.axvline(0, color="#999", lw=.7); ax.axhline(0, color="#999", lw=.7)
    ax.set_xlabel("Early Q→image-K distance improvement\n(adapter − Relay3; positive means Relay3 closer)")
    ax.set_ylabel("Generated-token saving (adapter − Relay3)")
    ax.set_title(f"Alignment improvement versus answer efficiency\n"
                 f"Pearson r={pearson[0]:.3f}, Spearman ρ={spearman[0]:.3f}")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "early_alignment_vs_token_saving.png", dpi=180)
    plt.close(fig)
    stats = {
        "both_terminated": {
            "n": len(subset),
            "pearson": {"estimate": pearson[0], "bootstrap_ci95": pearson[1:]},
            "spearman": {"estimate": spearman[0], "bootstrap_ci95": spearman[1:]},
        },
        "both_correct_and_terminated": {
            "n": len(correct_subset),
            "pearson": {"estimate": correct_pearson[0],
                        "bootstrap_ci95": correct_pearson[1:]},
            "spearman": {"estimate": correct_spearman[0],
                         "bootstrap_ci95": correct_spearman[1:]},
        },
        "early_window": {"steps": [0, args.early_steps - 1],
                         "layers": [args.early_layer_start, args.early_layer_end - 1]},
    }
    (output / "efficiency_correlation.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    return stats


def ability_summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["ability_level"])].append(row)
    result = []
    for (dataset, level), cells in sorted(grouped.items()):
        correct = [x for x in cells if x["both_correct_terminated"]]
        result.append({
            "dataset": dataset, "ability_level": level, "questions": len(cells),
            "adapter_accuracy": np.mean([x["adapter_reward"] for x in cells]),
            "relay_accuracy": np.mean([x["relay_reward"] for x in cells]),
            "mean_distance_improvement": np.mean([x["relay_distance_improvement"] for x in cells]),
            "both_correct_terminated_n": len(correct),
            "mean_token_saving_both_correct": (np.mean([x["token_saving"] for x in correct])
                                                if correct else float("nan")),
        })
    return result


def plot_ability_summary(rows, output):
    levels = [f"L{i}" for i in range(6)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for column, dataset in enumerate(("mapverse", "mapwise")):
        cells = [row for row in rows if row["dataset"] == dataset]
        gains, distance, saving = [], [], []
        for level in levels:
            group = [row for row in cells if row["ability_level"] == level]
            gains.append(np.mean([row["relay_reward"] - row["adapter_reward"]
                                  for row in group]) if group else np.nan)
            distance.append(np.mean([row["relay_distance_improvement"]
                                     for row in group]) if group else np.nan)
            correct = [row for row in group if row["both_correct_terminated"]]
            saving.append(np.mean([row["token_saving"] for row in correct])
                          if correct else np.nan)
        x = np.arange(len(levels))
        axes[0, column].bar(x - .18, gains, width=.36, color="#4b8bc8",
                            label="accuracy gain")
        twin = axes[0, column].twinx()
        twin.plot(x, distance, color="#d84a3a", marker="o", label="Q-K distance improvement")
        axes[0, column].axhline(0, color="#777", lw=.7)
        twin.axhline(0, color="#d84a3a", lw=.5, alpha=.4)
        axes[0, column].set_title(dataset)
        axes[0, column].set_ylabel("Relay3 − adapter accuracy")
        twin.set_ylabel("adapter − Relay3 Q-K distance", color="#d84a3a")
        axes[1, column].bar(x, saving, color="#56a36c")
        axes[1, column].axhline(0, color="#777", lw=.7)
        axes[1, column].set_ylabel("Token saving when both correct")
        axes[1, column].set_xticks(x, levels)
        axes[1, column].set_xlabel("Map-reading ability level")
    fig.suptitle("Accuracy, visual alignment and efficiency by ability level")
    fig.tight_layout()
    fig.savefig(output / "ability_level_accuracy_alignment_efficiency.png", dpi=180)
    plt.close(fig)


def write_report(output, total, stats, ability):
    terminated = stats["both_terminated"]
    correct = stats["both_correct_and_terminated"]
    lines = [
        "# Relay3 population hidden-Q / image-K analysis", "",
        f"Aligned paired questions: **{total}**.", "",
        "The distance metrics use the actual post-RoPE language query and cached image keys at each decoder layer. "
        "Positive `adapter distance - Relay3 distance` means the relay-trained system's hidden query is closer to image keys.", "",
        "## Efficiency association", "",
        f"Among {terminated['n']} pairs where both models terminated, the correlation between early alignment improvement and token saving was "
        f"Pearson **{terminated['pearson']['estimate']:.3f}** and Spearman **{terminated['spearman']['estimate']:.3f}**. "
        f"Among the {correct['n']} pairs where both were also correct, it was Pearson "
        f"**{correct['pearson']['estimate']:.3f}** and Spearman **{correct['spearman']['estimate']:.3f}**.", "",
        "Accuracy in this report is rescored from each newly captured response with the shared dataset-aware exact evaluator; "
        "old evaluation rewards are retained only as provenance and are not joined onto a different greedy trajectory.", "",
        "This is system-level associational evidence. Adapter weights and relay tokens were jointly trained, so it should be described as the "
        "effect of the Relay3 training configuration rather than an isolated causal effect of the single relay parameter.", "",
        "## Ability-level summary", "",
        "| Dataset | Level | N | Adapter acc. | Relay3 acc. | Q-K distance improvement | Token saving (both correct) |", "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in ability:
        saving = row["mean_token_saving_both_correct"]
        lines.append(f"| {row['dataset']} | {row['ability_level']} | {row['questions']} | "
                     f"{row['adapter_accuracy']:.3f} | {row['relay_accuracy']:.3f} | "
                     f"{row['mean_distance_improvement']:.5f} | "
                     f"{saving:.2f} |" if math.isfinite(saving) else
                     f"| {row['dataset']} | {row['ability_level']} | {row['questions']} | "
                     f"{row['adapter_accuracy']:.3f} | {row['relay_accuracy']:.3f} | "
                     f"{row['mean_distance_improvement']:.5f} | n/a |")
    lines += ["", "## Files", "",
              "- `early_qk_distance_by_layer.png`: population layer curves with question bootstrap CIs.",
              "- `paired_distance_improvement_step_layer.png`: where in decoding and depth Relay3 changes image-key alignment.",
              "- `global_hidden_query_image_key_clusters.png`: per-layer cluster view after image-key Procrustes alignment.",
              "- `early_alignment_vs_token_saving.png`: relationship to inference efficiency.",
              "- CSV/JSON files contain the underlying values.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    pairs = []
    question_sets = {
        "mapverse": scoring_questions(args.mapverse_dataset, "mapverse"),
        "mapwise": scoring_questions(args.mapwise_dataset, "mapwise"),
    }
    for dataset in ("mapverse", "mapwise"):
        adapter = load_variant(args.input, dataset, "adapter_only")
        relay = load_variant(args.input, dataset, "relay3")
        attach_fresh_scores(adapter, question_sets[dataset])
        attach_fresh_scores(relay, question_sets[dataset])
        for index in validate_pair(adapter, relay, dataset):
            pairs.append({"adapter_only": adapter[index], "relay3": relay[index]})
    plot_layer_curves(pairs, args, args.output, rng)
    plot_delta_heatmap(pairs, args.output)
    plot_clusters(pairs, args, args.output)
    rows = efficiency_rows(pairs, args)
    write_csv(args.output / "per_question_efficiency.csv", rows)
    stats = plot_efficiency(rows, args, args.output, rng)
    ability = ability_summary(rows)
    write_csv(args.output / "ability_summary.csv", ability)
    plot_ability_summary(rows, args.output)
    write_report(args.output, len(pairs), stats, ability)
    print(json.dumps({"questions": len(pairs), "output": str(args.output),
                      "report": str(args.output / 'report.md')}, ensure_ascii=False))


if __name__ == "__main__":
    main()
