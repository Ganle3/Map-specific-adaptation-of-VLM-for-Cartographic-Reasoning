#!/usr/bin/env python3
"""Render matched-prefix layer trajectories and aggregate visual-sensitivity evidence."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "Training_outputs/Prototype/relay3_latent_evidence"


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selection", type=Path, default=DEFAULT_ROOT / "selected_step4000.json")
    p.add_argument("--trajectory-dir", type=Path, default=DEFAULT_ROOT / "trajectories_step4000")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT / "figures_step4000")
    return p.parse_args()


def normalized(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def pca_scores(x):
    centered = x - x.mean(axis=0, keepdims=True)
    gram = centered @ centered.T
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = [-1, -2]
    return eigenvectors[:, order] * np.sqrt(np.maximum(eigenvalues[order], 0))[None, :]


def load_pair(root, row):
    folder = root / f"{row['dataset']}_index{row['dataset_index']:04d}"
    files = [folder / kind / "metrics.json" for kind in ("adapter_only", "relay3")]
    if not all(f.is_file() for f in files):
        return None
    result = {}
    for kind in ("adapter_only", "relay3"):
        base = folder / kind
        result[kind] = (json.loads((base / "metrics.json").read_text(encoding="utf-8")),
                        np.load(base / "states.npz"))
    a, r = result["adapter_only"][0], result["relay3"][0]
    if (a["dataset_index"], a["qa_id"], a["question"], a["ground_truth"]) != (
            r["dataset_index"], r["qa_id"], r["question"], r["ground_truth"]):
        raise ValueError(f"Pair mismatch: {folder}")
    return result


def load_same_adapter_control(root, row):
    base = root / f"{row['dataset']}_index{row['dataset_index']:04d}" / "relay3_adapter_no_relay"
    path = base / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def plot_case(row, pair, output):
    ad_meta, ad_npz = pair["adapter_only"]
    re_meta, re_npz = pair["relay3"]
    ad_steps = {item["step"]: item for item in ad_meta["steps"]}
    re_steps = {item["step"]: item for item in re_meta["steps"]}
    steps = sorted(ad_steps.keys() & re_steps.keys())
    if not steps:
        return
    layers = len(ad_steps[steps[0]]["hidden_image_cosine_distance_by_layer"])
    a_sens = np.array([ad_steps[t]["hidden_image_cosine_distance_by_layer"] for t in steps])
    r_sens = np.array([re_steps[t]["hidden_image_cosine_distance_by_layer"] for t in steps])
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    vmax = float(max(a_sens.max(), r_sens.max(), 1e-4))
    for ax, heat, label in ((axes[0, 0], a_sens, "Adapter only"),
                            (axes[0, 1], r_sens, "Relay3")):
        im = ax.imshow(heat, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=vmax)
        ax.set_title(f"{label}: hidden response to visual ablation")
        ax.set_xlabel("Decoder layer output")
        ax.set_ylabel("Reference token step")
        ax.set_yticks(range(len(steps)), steps)
    fig.colorbar(im, ax=[axes[0, 0], axes[0, 1]], shrink=.75,
                 label="1 - cosine(hidden real image, hidden blank image)")
    delta = r_sens - a_sens
    scale = max(float(np.abs(delta).max()), 1e-5)
    im2 = axes[1, 0].imshow(delta, aspect="auto", origin="lower", cmap="RdBu_r",
                            vmin=-scale, vmax=scale)
    axes[1, 0].set_title("Relay3 minus adapter visual sensitivity")
    axes[1, 0].set_xlabel("Decoder layer output")
    axes[1, 0].set_ylabel("Reference token step")
    axes[1, 0].set_yticks(range(len(steps)), steps)
    fig.colorbar(im2, ax=axes[1, 0], shrink=.75)
    for kind, meta, color in (("Adapter only", ad_steps, "#2870b8"),
                               ("Relay3", re_steps, "#d35431")):
        axes[1, 1].plot(steps, [meta[t]["image_logp_effect"] for t in steps],
                        marker="o", color=color, label=kind)
    axes[1, 1].axhline(0, color="gray", linewidth=.8)
    axes[1, 1].set_title("Image support for the same next reference token")
    axes[1, 1].set_xlabel("Reference token step")
    axes[1, 1].set_ylabel("log p(token | real image) - log p(token | blank image)")
    axes[1, 1].legend()
    fig.suptitle(f"{row['dataset']} index {row['dataset_index']} | {row['category']}\n"
                 f"{row['question'][:110]}", fontsize=12)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)

    # Exploratory shared PCA of unit-normalized hidden states. PCA distance is
    # intentionally not interpreted as visual understanding or accuracy.
    point_rows = []
    for variant, (_, states) in pair.items():
        for step in (steps[0], steps[-1]):
            for layer, vec in enumerate(states[f"language_step{step}"]):
                point_rows.append((variant, step, layer, vec))
    matrix = normalized(np.stack([point[-1] for point in point_rows]).astype(np.float64))
    coords = pca_scores(matrix)
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    for variant, color in (("adapter_only", "#2870b8"), ("relay3", "#d35431")):
        for step, linestyle in ((steps[0], "-"), (steps[-1], "--")):
            ix = [i for i, (v, t, _, _) in enumerate(point_rows) if v == variant and t == step]
            ax.plot(coords[ix, 0], coords[ix, 1], linestyle=linestyle, color=color,
                    marker=".", markersize=3, label=f"{variant} step {step}")
            ax.scatter(coords[ix[0], 0], coords[ix[0], 1], c=color, marker="s", s=45)
            ax.scatter(coords[ix[-1], 0], coords[ix[-1], 1], c=color, marker="*", s=90)
    ax.set_title("Matched-prefix language hidden trajectory across decoder layers\n"
                 "Square = first layer; star = final layer; exploratory PCA")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.legend(fontsize=8)
    fig.savefig(output.with_name(output.stem + "_latent.png"), dpi=150)
    plt.close(fig)

    # Raw merger embeddings and decoder states share H but need not share an
    # interpretable metric space. Show this only as an exploratory overlay.
    if all("image_merger_sample" in states.files for _, states in pair.values()):
        image_rows = []
        for variant, (_, states) in pair.items():
            sample = normalized(states["image_merger_sample"].astype(np.float64))
            image_rows.extend((variant, vec) for vec in sample)
        all_vectors = np.vstack((matrix, np.stack([v for _, v in image_rows])))
        projected = pca_scores(all_vectors)
        lang_projected = projected[:len(point_rows)]
        image_projected = projected[len(point_rows):]
        fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
        for variant, color in (("adapter_only", "#2870b8"), ("relay3", "#d35431")):
            ix_img = [i for i, (v, _) in enumerate(image_rows) if v == variant]
            ax.scatter(image_projected[ix_img, 0], image_projected[ix_img, 1],
                       s=8, alpha=.18, color=color, marker="x",
                       label=f"{variant} merger image patches")
            ix = [i for i, (v, t, _, _) in enumerate(point_rows)
                  if v == variant and t == steps[-1]]
            ax.plot(lang_projected[ix, 0], lang_projected[ix, 1],
                    color=color, linewidth=2, label=f"{variant} language step {steps[-1]}")
        ax.set_title("Exploratory raw cross-modal PCA\n"
                     "Distances across merger/decoder representations are not validated")
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.legend(fontsize=8)
        fig.savefig(output.with_name(output.stem + "_crossmodal_exploratory.png"), dpi=150)
        plt.close(fig)


def main():
    args = arguments()
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    rows = []
    plotted = 0
    for question in selection["questions"]:
        pair = load_pair(args.trajectory_dir, question)
        if pair is None:
            continue
        stem = f"{question['dataset']}_{question['category']}_index{question['dataset_index']:04d}"
        plot_case(question, pair, args.output_dir / (stem + ".png"))
        plotted += 1
        a = {item["step"]: item for item in pair["adapter_only"][0]["steps"]}
        r = {item["step"]: item for item in pair["relay3"][0]["steps"]}
        control = load_same_adapter_control(args.trajectory_dir, question)
        c = ({item["step"]: item for item in control["steps"]} if control else {})
        for step in sorted(a.keys() & r.keys()):
            if a[step]["reference_next_token_id"] != r[step]["reference_next_token_id"]:
                raise ValueError("Reference token mismatch")
            for layer, (av, rv) in enumerate(zip(
                    a[step]["hidden_image_cosine_distance_by_layer"],
                    r[step]["hidden_image_cosine_distance_by_layer"])):
                rows.append({
                    "dataset": question["dataset"], "category": question["category"],
                    "dataset_index": question["dataset_index"], "qa_id": question["qa_id"],
                    "step": step, "next_token": a[step]["reference_next_token"],
                    "layer": layer, "adapter_image_sensitivity": av,
                    "relay_image_sensitivity": rv,
                    "relay_minus_adapter": rv - av,
                    "adapter_image_logp_effect": a[step]["image_logp_effect"],
                    "relay_image_logp_effect": r[step]["image_logp_effect"],
                    "same_adapter_no_relay_image_sensitivity": (
                        c[step]["hidden_image_cosine_distance_by_layer"][layer]
                        if step in c else ""),
                    "relay_minus_same_adapter_no_relay": (
                        rv - c[step]["hidden_image_cosine_distance_by_layer"][layer]
                        if step in c else ""),
                    "same_adapter_no_relay_image_logp_effect": (
                        c[step]["image_logp_effect"] if step in c else ""),
                })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        with (args.output_dir / "paired_metrics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        # Aggregate by question first: repeated layers or steps from one map
        # must not masquerade as independent examples.
        group_rows = []
        for dataset in ("mapverse", "mapwise"):
            for category in ("relay_only_correct", "both_correct", "both_wrong"):
                for step in sorted({r["step"] for r in rows}):
                    for layer in sorted({r["layer"] for r in rows}):
                        cells = [r for r in rows if r["dataset"] == dataset
                                 and r["category"] == category and r["step"] == step
                                 and r["layer"] == layer]
                        if not cells:
                            continue
                        diffs = np.array([r["relay_minus_adapter"] for r in cells], dtype=float)
                        # This is descriptive variability, not a causal CI for correctness.
                        group_rows.append({
                            "dataset": dataset, "category": category, "step": step,
                            "layer": layer, "question_count": len(cells),
                            "mean_relay_minus_adapter_image_sensitivity": float(diffs.mean()),
                            "median_relay_minus_adapter_image_sensitivity": float(np.median(diffs)),
                            "fraction_questions_relay_more_image_sensitive": float(np.mean(diffs > 0)),
                        })
        with (args.output_dir / "group_by_step_layer.csv").open(
                "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(group_rows[0]))
            writer.writeheader()
            writer.writerows(group_rows)
        fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True,
                                 constrained_layout=True)
        for row_index, dataset in enumerate(("mapverse", "mapwise")):
            for col_index, category in enumerate(("relay_only_correct", "both_correct", "both_wrong")):
                ax = axes[row_index, col_index]
                subset = [g for g in group_rows if g["dataset"] == dataset
                          and g["category"] == category]
                steps = sorted({g["step"] for g in subset})
                for step in steps:
                    layer_rows = sorted((g for g in subset if g["step"] == step),
                                        key=lambda g: g["layer"])
                    ax.plot([g["layer"] for g in layer_rows],
                            [g["mean_relay_minus_adapter_image_sensitivity"]
                             for g in layer_rows], label=f"step {step}")
                ax.axhline(0, color="gray", linewidth=.8)
                ax.set_title(f"{dataset} | {category}\n"
                             f"n={max((g['question_count'] for g in subset), default=0)}")
                ax.set_xlabel("Decoder layer output")
                ax.set_ylabel("Mean paired image-sensitivity difference")
                if row_index == 0 and col_index == 2:
                    ax.legend(fontsize=8)
        fig.suptitle("Relay3 minus adapter-only response to visual ablation\n"
                     "Positive means stronger image dependence under identical text; "
                     "it does not imply more accurate interpretation")
        fig.savefig(args.output_dir / "group_visual_sensitivity.png", dpi=150)
        plt.close(fig)
    summary = {"rendered_questions": plotted, "selected_questions": len(selection["questions"]),
               "metric_rows": len(rows),
               "questions_with_same_adapter_control": len({
                   (row["dataset"], row["dataset_index"]) for row in rows
                   if row["relay_minus_same_adapter_no_relay"] != ""
               }),
               "interpretation": "Visual-ablation sensitivity and reference-token log-probability "
                                 "measure image dependence under identical prefixes. Neither alone proves "
                                 "correct visual reasoning. PCA is exploratory only."}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
