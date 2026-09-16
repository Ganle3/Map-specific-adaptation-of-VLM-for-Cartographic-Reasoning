"""Plot saved diagnostic CSVs; does not import torch, TRL or training code."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_results(results_dir, dot_product=False):
    root = Path(results_dir)
    for name in ["cosine", *(["dot_product"] if dot_product else [])]:
        with (root / f"{name}_matrix.csv").open(encoding="utf-8") as stream:
            rows = list(csv.reader(stream))
        labels = rows[0][1:]
        matrix = np.array([[float(x) for x in r[1:]] for r in rows[1:]])
        cmap = plt.get_cmap("RdBu_r").copy()
        cmap.set_bad("#bdbdbd")
        limit = 1 if name == "cosine" else max(float(np.nanmax(np.abs(matrix))), 1e-15)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=-limit, vmax=limit)
        ax.set(xticks=range(len(labels)), yticks=range(len(labels)),
               xticklabels=labels, yticklabels=labels, title=f"Batch gradient {name.replace('_', ' ')}")
        for i, j in np.ndindex(matrix.shape):
            value = matrix[i, j]
            label = "N/A" if np.isnan(value) else (f"{value:.2f}" if name == "cosine" else f"{value:.2g}")
            ax.text(j, i, label, ha="center", va="center",
                    color="white" if np.isfinite(value) and abs(value) > .65 * limit else "black")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(root / f"{name}_matrix.png", dpi=200)
        plt.close(fig)
    with (root / "batch_metrics.csv").open(encoding="utf-8") as stream:
        metrics = list(csv.DictReader(stream))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([r["batch_id"] for r in metrics], [float(r["gradient_norm"]) for r in metrics])
    ax.set(ylabel="LoRA gradient L2 norm", title="Optimizer-batch gradient norms")
    fig.tight_layout()
    fig.savefig(root / "gradient_norms.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--dot-product", action="store_true")
    args = parser.parse_args()
    plot_results(args.results_dir, args.dot_product)
