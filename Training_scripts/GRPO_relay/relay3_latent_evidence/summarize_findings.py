#!/usr/bin/env python3
"""Question-level paired summaries with bootstrap intervals, never layer pseudoreplication."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3] / "Training_outputs/Prototype/relay3_latent_evidence"


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selection", type=Path, default=ROOT / "selected_step4000.json")
    p.add_argument("--trajectory-dir", type=Path, default=ROOT / "trajectories_step4000")
    p.add_argument("--output-dir", type=Path, default=ROOT / "figures_step4000")
    p.add_argument("--layer-start", type=int, default=8)
    p.add_argument("--layer-end", type=int, default=31)
    return p.parse_args()


def interval(values, rng):
    x = np.asarray(values, dtype=np.float64)
    if len(x) == 0:
        return (float("nan"), float("nan"), float("nan"))
    draws = rng.choice(x, size=(4000, len(x)), replace=True).mean(axis=1)
    return (float(x.mean()), float(np.quantile(draws, .025)),
            float(np.quantile(draws, .975)))


def main():
    args = arguments()
    manifest = json.loads(args.selection.read_text(encoding="utf-8"))
    rng = np.random.default_rng(1234)
    question_rows = []
    relay_effect_rows = []
    for question in manifest["questions"]:
        folder = args.trajectory_dir / f"{question['dataset']}_index{question['dataset_index']:04d}"
        loaded = {}
        for variant in ("adapter_only", "relay3", "relay3_adapter_no_relay"):
            path = folder / variant / "metrics.json"
            if path.is_file():
                loaded[variant] = {s["step"]: s for s in json.loads(
                    path.read_text(encoding="utf-8"))["steps"]}
        if not {"adapter_only", "relay3"} <= loaded.keys():
            continue
        for step in sorted(loaded["adapter_only"].keys() & loaded["relay3"].keys()):
            a, r = loaded["adapter_only"][step], loaded["relay3"][step]
            def middle(s):
                return float(np.mean(s["hidden_image_cosine_distance_by_layer"]
                                     [args.layer_start:args.layer_end + 1]))
            c = loaded.get("relay3_adapter_no_relay", {}).get(step)
            if c is not None:
                with np.load(folder / "relay3" / "states.npz") as relay_states, np.load(
                        folder / "relay3_adapter_no_relay" / "states.npz") as control_states:
                    with_relay = relay_states[f"language_step{step}"].astype(np.float64)
                    without_relay = control_states[f"language_step{step}"].astype(np.float64)
                if with_relay.shape != without_relay.shape:
                    raise ValueError("Same-adapter language layer shape mismatch")
                cosine = np.sum(with_relay * without_relay, axis=-1) / np.maximum(
                    np.linalg.norm(with_relay, axis=-1)
                    * np.linalg.norm(without_relay, axis=-1), 1e-12)
                for layer, value in enumerate(1 - cosine):
                    relay_effect_rows.append({
                        "dataset": question["dataset"], "category": question["category"],
                        "dataset_index": question["dataset_index"], "step": step,
                        "layer": layer, "hidden_relay_effect_cosine_distance": float(value),
                    })
            question_rows.append({
                "dataset": question["dataset"], "category": question["category"],
                "dataset_index": question["dataset_index"], "qa_id": question["qa_id"],
                "step": step, "reference_next_token": a["reference_next_token"],
                "adapter_image_sensitivity": middle(a),
                "relay_image_sensitivity": middle(r),
                "relay_minus_adapter_sensitivity": middle(r) - middle(a),
                "same_adapter_no_relay_image_sensitivity": middle(c) if c else "",
                "relay_minus_same_adapter_sensitivity": middle(r) - middle(c) if c else "",
                "relay_minus_adapter_image_logp_effect": (
                    r["image_logp_effect"] - a["image_logp_effect"]),
                "relay_minus_same_adapter_image_logp_effect": (
                    r["image_logp_effect"] - c["image_logp_effect"] if c else ""),
            })
    if not question_rows:
        raise RuntimeError("No matched model trajectories found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "question_level_metrics.csv").open(
            "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(question_rows[0]))
        writer.writeheader()
        writer.writerows(question_rows)
    if relay_effect_rows:
        with (args.output_dir / "same_adapter_relay_effect_by_layer.csv").open(
                "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(relay_effect_rows[0]))
            writer.writeheader()
            writer.writerows(relay_effect_rows)
    group_rows = []
    for dataset in ("mapverse", "mapwise"):
        for category in ("relay_only_correct", "both_correct", "both_wrong"):
            for step in sorted({row["step"] for row in question_rows}):
                subset = [row for row in question_rows if row["dataset"] == dataset
                          and row["category"] == category and row["step"] == step]
                if not subset:
                    continue
                delta = [row["relay_minus_adapter_sensitivity"] for row in subset]
                control = [row["relay_minus_same_adapter_sensitivity"] for row in subset
                           if row["relay_minus_same_adapter_sensitivity"] != ""]
                mean, low, high = interval(delta, rng)
                cm, cl, ch = interval(control, rng)
                group_rows.append({
                    "dataset": dataset, "category": category, "step": step,
                    "question_count": len(subset), "relay_minus_adapter_mean": mean,
                    "relay_minus_adapter_bootstrap_low": low,
                    "relay_minus_adapter_bootstrap_high": high,
                    "fraction_positive": float(np.mean(np.asarray(delta) > 0)),
                    "same_adapter_control_count": len(control),
                    "relay_minus_same_adapter_mean": cm,
                    "relay_minus_same_adapter_bootstrap_low": cl,
                    "relay_minus_same_adapter_bootstrap_high": ch,
                })
    with (args.output_dir / "question_group_summary.csv").open(
            "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(group_rows[0]))
        writer.writeheader()
        writer.writerows(group_rows)
    lines = ["# Relay3 matched-prefix visual-sensitivity findings", "",
             f"Layer band: {args.layer_start}–{args.layer_end} (zero-based output layers).",
             "Positive difference means Relay3 hidden states change more when visual merger features are zeroed.",
             "Intervals are percentile bootstrap across **questions**, not across layers or steps.",
             "The selected categories condition on final-answer correctness; image sensitivity alone does not establish correctness.",
             "", "| Dataset | Outcome category | Step | N | Relay − adapter (mean, 95% bootstrap interval) | Relay − same adapter without relay |", "|---|---|---:|---:|---:|---:|"]
    for row in group_rows:
        a = f"{row['relay_minus_adapter_mean']:+.4f} [{row['relay_minus_adapter_bootstrap_low']:+.4f}, {row['relay_minus_adapter_bootstrap_high']:+.4f}]"
        c = (f"{row['relay_minus_same_adapter_mean']:+.4f} [{row['relay_minus_same_adapter_bootstrap_low']:+.4f}, {row['relay_minus_same_adapter_bootstrap_high']:+.4f}]"
             if row["same_adapter_control_count"] else "pending")
        lines.append(f"| {row['dataset']} | {row['category']} | {row['step']} | {row['question_count']} | {a} | {c} |")
    (args.output_dir / "findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if relay_effect_rows:
        fig_path = args.output_dir / "same_adapter_relay_effect.png"
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True,
                                 constrained_layout=True)
        for i, dataset in enumerate(("mapverse", "mapwise")):
            for j, category in enumerate(("relay_only_correct", "both_correct", "both_wrong")):
                ax = axes[i, j]
                cells = [r for r in relay_effect_rows if r["dataset"] == dataset
                         and r["category"] == category]
                for step in sorted({r["step"] for r in cells}):
                    by_layer = []
                    for layer in sorted({r["layer"] for r in cells}):
                        values = [r["hidden_relay_effect_cosine_distance"] for r in cells
                                  if r["step"] == step and r["layer"] == layer]
                        by_layer.append((layer, np.mean(values)))
                    ax.plot([x for x, _ in by_layer], [y for _, y in by_layer],
                            label=f"step {step}")
                ax.set_title(f"{dataset} | {category}")
                ax.set_xlabel("Decoder layer output")
                ax.set_ylabel("1 - cosine(x with relay, x without relay)")
                if i == 0 and j == 2:
                    ax.legend(fontsize=8)
        fig.suptitle("Direct effect of temporary relay using identical trained adapter and text")
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
    print(json.dumps({"question_step_rows": len(question_rows),
                      "groups": len(group_rows),
                      "same_adapter_relay_effect_rows": len(relay_effect_rows),
                      "same_adapter_control_rows": sum(
                          row["relay_minus_same_adapter_sensitivity"] != ""
                          for row in question_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
