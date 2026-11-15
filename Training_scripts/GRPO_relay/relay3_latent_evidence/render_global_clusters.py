#!/usr/bin/env python3
"""Across-question matched step-0 latent clusters at selected decoder layers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3] / "Training_outputs/Prototype/relay3_latent_evidence"
COLORS = {"relay_only_correct": "#2d9d78", "both_correct": "#3673bd",
          "both_wrong": "#cb593a"}


def pca(x):
    x = x.astype(np.float64)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    x -= x.mean(axis=0, keepdims=True)
    vals, vecs = np.linalg.eigh(x @ x.T)
    idx = [-1, -2]
    return vecs[:, idx] * np.sqrt(np.maximum(vals[idx], 0))[None, :]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selection", type=Path, default=ROOT / "selected_step4000.json")
    p.add_argument("--trajectory-dir", type=Path, default=ROOT / "trajectories_step4000")
    p.add_argument("--output-dir", type=Path, default=ROOT / "figures_step4000")
    p.add_argument("--layers", nargs="+", type=int, default=[8, 16, 24, 35])
    args = p.parse_args()
    selected = json.loads(args.selection.read_text(encoding="utf-8"))["questions"]
    rows = []
    for item in selected:
        folder = args.trajectory_dir / f"{item['dataset']}_index{item['dataset_index']:04d}"
        a, r = (folder / kind / "states.npz" for kind in ("adapter_only", "relay3"))
        if not a.is_file() or not r.is_file():
            continue
        with np.load(a) as af, np.load(r) as rf:
            rows.append((item, af["language_step0"], rf["language_step0"]))
    if not rows:
        raise RuntimeError("No complete step-0 language states")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for dataset in ("mapverse", "mapwise"):
        subset = [row for row in rows if row[0]["dataset"] == dataset]
        if not subset:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(13, 11))
        fig.subplots_adjust(left=.08, right=.98, top=.87, bottom=.13,
                            hspace=.38, wspace=.24)
        for ax, layer in zip(axes.flat, args.layers):
            if any(layer >= a.shape[0] or layer >= r.shape[0] for _, a, r in subset):
                raise ValueError(f"Layer {layer} outside captured state array")
            n = len(subset)
            vectors = np.concatenate((np.stack([a[layer] for _, a, _ in subset]),
                                      np.stack([r[layer] for _, _, r in subset])), axis=0)
            coords = pca(vectors)
            for i, (item, _, _) in enumerate(subset):
                color = COLORS[item["category"]]
                ax.plot([coords[i, 0], coords[i + n, 0]],
                        [coords[i, 1], coords[i + n, 1]],
                        color=color, alpha=.35, linewidth=.8)
                ax.scatter(*coords[i], edgecolor=color, facecolor="none", s=31,
                           linewidth=1)
                ax.scatter(*coords[i + n], color=color, s=17)
            ax.set_title(f"Decoder layer {layer} output | {n} matched questions")
            ax.set_xlabel("PCA 1")
            ax.set_ylabel("PCA 2")
        handles = [plt.Line2D([0], [0], color=color, marker="o", linestyle="none",
                              label=name) for name, color in COLORS.items()]
        handles += [plt.Line2D([0], [0], color="black", marker="o", markerfacecolor="none",
                               linestyle="none", label="adapter only"),
                    plt.Line2D([0], [0], color="black", marker="o", linestyle="none",
                               label="Relay3")]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .02),
                   ncol=5, fontsize=9)
        fig.suptitle(f"{dataset}: same prompt-end prediction position across questions\n"
                     "Line connects the two trained variants for the same question; "
                     "PCA is exploratory")
        fig.savefig(args.output_dir / f"{dataset}_step0_global_clusters.png", dpi=150)
        plt.close(fig)
    print(f"Rendered global clusters for {len(rows)} paired questions")


if __name__ == "__main__":
    main()
