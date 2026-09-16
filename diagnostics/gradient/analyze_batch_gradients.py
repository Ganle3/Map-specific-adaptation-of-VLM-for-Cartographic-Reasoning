"""CPU geometry and CSV output, independent of model and plotting libraries."""
import csv
from pathlib import Path

import numpy as np


def pairwise_geometry(gradients, eps=1e-12):
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if len(gradients) == 0:
        raise ValueError("No gradients")
    vectors = [np.asarray(g).reshape(-1) for g in gradients]
    if len({g.size for g in vectors}) != 1 or not all(np.isfinite(g).all() for g in vectors):
        raise ValueError("Gradients must have equal size and finite values")
    # Accumulate in float64 by chunks, without another full float64 gradient stack.
    dot = np.zeros((len(vectors), len(vectors)), dtype=np.float64)
    for start in range(0, vectors[0].size, 1_000_000):
        block = np.stack([g[start:start + 1_000_000] for g in vectors]).astype(np.float64)
        dot += block @ block.T
    norms = np.sqrt(np.maximum(np.diag(dot), 0))
    valid = norms >= eps
    denominator = np.outer(norms, norms)
    cosine = np.full_like(dot, np.nan)
    np.divide(dot, denominator, out=cosine, where=np.outer(valid, valid))
    cosine = np.clip(cosine, -1, 1)
    distance = np.sqrt(np.maximum(norms[:, None] ** 2 + norms[None, :] ** 2 - 2 * dot, 0))
    np.fill_diagonal(distance, 0)
    return dict(cosine=cosine, dot_product=dot, distance=distance), norms, valid


def write_results(output, labels, gradients, metrics, eps):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    matrices, norms, valid = pairwise_geometry(gradients, eps)
    for name, matrix in matrices.items():
        with (output / f"{name}_matrix.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["batch_id", *labels])
            writer.writerows([label, *row] for label, row in zip(labels, matrix))
    for row, norm, is_valid in zip(metrics, norms, valid):
        row.update(gradient_norm=float(norm), gradient_is_zero=bool(norm < eps),
                   valid_gradient=bool(is_valid))
    with (output / "batch_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    with (output / "valid_gradient_mask.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["batch_id", "valid_gradient"])
        writer.writerows(zip(labels, valid.tolist()))
