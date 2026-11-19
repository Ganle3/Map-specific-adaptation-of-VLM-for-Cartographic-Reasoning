#!/usr/bin/env python3
"""Paired outcome and efficiency analysis for Relay3 inference ablations."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


VARIANTS = ("full", "relay_off", "no_image_read", "no_current_x_read", "static_relay")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bootstrap-samples", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20261118)
    return p.parse_args()


def read_rows(root: Path):
    rows = {}
    for dataset in ("mapverse", "mapwise"):
        path = root / dataset / "responses.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            key = (dataset, int(row["dataset_index"]), str(row["variant"]))
            if key in rows:
                raise ValueError(f"Duplicate result {key}")
            rows[key] = row
    return rows


def ci(values, rng, samples):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return [None, None]
    means = np.empty(samples, dtype=np.float64)
    for i in range(samples):
        means[i] = values[rng.integers(0, values.size, values.size)].mean()
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def exact_mcnemar_p(full_only: int, ablation_only: int) -> float:
    """Two-sided exact binomial test on discordant accuracy pairs."""
    n = full_only + ablation_only
    if n == 0:
        return 1.0
    k = min(full_only, ablation_only)
    lower = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * lower)


def analyze_subset(rows, datasets, rng, samples):
    indices = sorted({(d, i) for d, i, v in rows if d in datasets and v == "full"})
    for key in indices:
        missing = [v for v in VARIANTS if (*key, v) not in rows]
        if missing:
            raise ValueError(f"Incomplete variants for {key}: {missing}")
    full = [rows[(*key, "full")] for key in indices]
    result = {
        "questions": len(indices),
        "variants": {},
        "contrasts_vs_full": {},
    }
    for variant in VARIANTS:
        part = [rows[(*key, variant)] for key in indices]
        correct = [float(r["reward"]) for r in part]
        terminated = [bool(r["terminated"]) for r in part]
        result["variants"][variant] = {
            "accuracy": float(np.mean(correct)),
            "terminated_fraction": float(np.mean(terminated)),
            "mean_tokens_all": float(np.mean([r["generated_tokens"] for r in part])),
            "mean_generation_seconds": float(np.mean([r["generation_seconds"] for r in part])),
        }
        if variant == "full":
            continue
        accuracy_effect = np.asarray([
            float(f["reward"]) - float(a["reward"]) for f, a in zip(full, part)
        ])
        full_only = sum(f["reward"] == 1.0 and a["reward"] == 0.0
                        for f, a in zip(full, part))
        ablation_only = sum(f["reward"] == 0.0 and a["reward"] == 1.0
                            for f, a in zip(full, part))
        paired = [(f, a) for f, a in zip(full, part)
                  if f["reward"] == a["reward"] == 1.0
                  and f["terminated"] and a["terminated"]]
        token_increase = np.asarray([
            a["generated_tokens"] - f["generated_tokens"] for f, a in paired
        ], dtype=np.float64)
        result["contrasts_vs_full"][variant] = {
            "accuracy_loss_when_ablated": float(accuracy_effect.mean()),
            "accuracy_loss_95ci": ci(accuracy_effect, rng, samples),
            "full_correct_ablation_wrong": int(full_only),
            "full_wrong_ablation_correct": int(ablation_only),
            "exact_mcnemar_p": exact_mcnemar_p(full_only, ablation_only),
            "both_correct_terminated_count": len(paired),
            "token_increase_when_ablated": (
                float(token_increase.mean()) if token_increase.size else None
            ),
            "token_increase_95ci": ci(token_increase, rng, samples),
        }
    return result


def fmt(value):
    return "NA" if value is None else f"{value:.4f}"


def main():
    args = parse_args()
    if args.bootstrap_samples < 100:
        raise ValueError("Use at least 100 bootstrap samples")
    rows = read_rows(args.input)
    rng = np.random.default_rng(args.seed)
    result = {
        "definitions": {
            "full_minus_relay_off": "total online relay contribution under the Relay3-trained adapter",
            "full_minus_no_image_read": "online relay-to-image-KV contribution",
            "full_minus_no_current_x_read": "current reasoning-state conditioning contribution",
            "full_minus_static_relay": "per-step dynamic relay refresh contribution",
        },
        "mapverse": analyze_subset(rows, {"mapverse"}, rng, args.bootstrap_samples),
        "mapwise": analyze_subset(rows, {"mapwise"}, rng, args.bootstrap_samples),
        "combined": analyze_subset(rows, {"mapverse", "mapwise"}, rng, args.bootstrap_samples),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "ablation_statistics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Relay3 inference ablation", "",
        "All rows use the same Relay3-trained adapter and relay checkpoint. Positive accuracy loss means the full dynamic Relay3 model was better. Positive token increase means the ablation needed more tokens on questions both variants answered correctly and terminated.", "",
    ]
    for dataset in ("mapverse", "mapwise", "combined"):
        section = result[dataset]
        lines += [f"## {dataset.title()}", "", "### Absolute results", "",
                  "| Variant | Accuracy | Terminated | Mean tokens | Mean generation seconds |",
                  "|---|---:|---:|---:|---:|"]
        for variant in VARIANTS:
            x = section["variants"][variant]
            lines.append(
                f"| {variant} | {x['accuracy']:.4f} | {x['terminated_fraction']:.4f} | "
                f"{x['mean_tokens_all']:.2f} | {x['mean_generation_seconds']:.3f} |"
            )
        lines += ["", "### Paired contrasts against full", "",
                  "| Ablation | Accuracy loss | 95% CI | Full-only correct | Ablation-only correct | McNemar p | Paired correct | Token increase | 95% CI |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for variant in VARIANTS[1:]:
            x = section["contrasts_vs_full"][variant]
            aci, tci = x["accuracy_loss_95ci"], x["token_increase_95ci"]
            lines.append(
                f"| {variant} | {x['accuracy_loss_when_ablated']:.4f} | "
                f"[{fmt(aci[0])}, {fmt(aci[1])}] | {x['full_correct_ablation_wrong']} | "
                f"{x['full_wrong_ablation_correct']} | {x['exact_mcnemar_p']:.4g} | "
                f"{x['both_correct_terminated_count']} | "
                f"{fmt(x['token_increase_when_ablated'])} | [{fmt(tci[0])}, {fmt(tci[1])}] |"
            )
        lines.append("")
    lines += [
        "## Causal interpretation", "",
        "- `full` versus `no_image_read` tests whether the temporary relay's extra online read from image KV matters. The normal language hidden state keeps its native access to image KV.",
        "- `full` versus `no_current_x_read` tests whether conditioning that read on the current reasoning state matters.",
        "- `full` versus `static_relay` tests whether refreshing the relay at every decoding step matters beyond a one-time relay memory.",
        "- `full` versus `relay_off` measures the complete online relay contribution while retaining the jointly trained Relay3 adapter.", "",
    ]
    (args.output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(args.output / "report.md")


if __name__ == "__main__":
    main()
