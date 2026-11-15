#!/usr/bin/env python3
"""Compare decisive events on each model's own greedy trajectory.

Reads the existing step-4000 attention capture; no teacher forcing or model load.
"""
from __future__ import annotations

import argparse
import html
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


VLM = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = VLM / "Training_outputs/Prototype/relay3_visualization_step4000"
DEFAULT_OUTPUT = VLM / "Training_outputs/Prototype/relay3_native_step_evidence"
EVALUATION_PATHS = {
    "adapter_only": {
        "mapverse": VLM / "Training_outputs/Prototype/adapter_only/evaluation_mapverse_test600_checkpoint_2000_4000/responses.jsonl",
        "mapwise": VLM / "Training_outputs/Prototype/adapter_only/evaluation_mapwise_test400_checkpoint_2000_4000/responses.jsonl",
    },
    "relay3": {
        "mapverse": VLM / "Training_outputs/Prototype/relay3/evaluation_mapverse_test600_initial_without_relay_2000_4000_5000/responses.jsonl",
        "mapwise": VLM / "Training_outputs/Prototype/relay3/evaluation_mapwise_test400_initial_without_relay_2000_4000_5000/responses.jsonl",
    },
}

# The phrases are observations in each natural response, not a shared forced prefix.
# They mark a first concrete visual claim and the first answer claim respectively.
CASES = {
    "mapverse_qa232": {
        "adapter_only": ("The entry for S is", "So the answer is"),
        "relay3": ("The letter \"S\" corresponds", "So the answer is"),
    },
    "mapverse_qa273": {
        "adapter_only": ("Check the routes:", "That's 5 routes"),
        "relay3": ("The routes are F1", "That's 6 routes"),
    },
    "mapwise_qa023": {
        "adapter_only": ("Hainan's color is", "So the two markings are"),
        "relay3": ("Hainan's color is", "So the two markings are"),
    },
    "mapwise_qa101": {
        "adapter_only": ("Beijing is labeled as", "So count is 1"),
        "relay3": ("Beijing is in the North", "So count is 1"),
    },
}


def token_step(record: dict, phrase: str) -> int:
    raw = record["raw_response"]
    offset = raw.find(phrase)
    if offset < 0:
        raise ValueError(f"Phrase absent from {record['variant']}: {phrase!r}")
    accumulated = 0
    for step, token in enumerate(record["tokens"], 1):
        accumulated += len(token)
        if accumulated > offset:
            return step
    raise ValueError(f"Phrase outside token stream: {phrase!r}")


def mass_by_step(attention: np.ndarray) -> np.ndarray:
    """Mean over decoder layers of total image-key attention mass.

    Patch weights remain unnormalized here: a sum of 1 would mean all attention
    was paid to image keys. This is not a localization or correctness metric.
    """
    return attention.astype(np.float64).sum(axis=(-1, -2)).mean(axis=1)


def load_case(root: Path, name: str, variant: str) -> dict:
    folder = root / name / variant
    record = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    with np.load(folder / "attention.npz") as arrays:
        hidden = mass_by_step(arrays["hidden_attention"])
        relay = mass_by_step(arrays["relay_attention"]) if "relay_attention" in arrays else None
    if len(hidden) != record["steps"] or (relay is not None and len(relay) != record["steps"]):
        raise ValueError(f"Attention and token step mismatch: {name}/{variant}")
    visual_phrase, answer_phrase = CASES[name][variant]
    events = {
        "first_visual_claim": token_step(record, visual_phrase),
        "first_answer_claim": token_step(record, answer_phrase),
        "final_answer_label": token_step(record, "Final answer:"),
    }
    return {
        **record,
        "events": events,
        "event_phrases": {"first_visual_claim": visual_phrase, "first_answer_claim": answer_phrase},
        "mean_image_attention_mass": {"hidden": hidden.tolist(), "relay": relay.tolist() if relay is not None else None},
    }


def plot_case(name: str, rows: dict, path: Path) -> None:
    colors = {"adapter_only": "#2f6fb3", "relay3": "#cf4b4b"}
    fig, (ax, mass_ax) = plt.subplots(2, 1, figsize=(11, 5.2), sharex=True,
                                      gridspec_kw={"height_ratios": [1, 2]})
    for i, (variant, row) in enumerate(rows.items()):
        c = colors[variant]
        n = row["steps"]
        ax.hlines(i, 0, n, color=c, lw=5)
        ax.scatter([n], [i], marker="|", s=220, color=c)
        for event, marker in [("first_visual_claim", "o"), ("first_answer_claim", "s"),
                              ("final_answer_label", "^")]:
            step = row["events"][event]
            ax.scatter([step], [i], marker=marker, s=75, color=c,
                       edgecolor="white", linewidth=0.6, zorder=4)
            ax.annotate(str(step), (step, i), xytext=(0, 9), textcoords="offset points",
                        ha="center", fontsize=8)
        t = np.arange(1, n + 1)
        mass_ax.plot(t, row["mean_image_attention_mass"]["hidden"], color=c,
                     lw=1.3, alpha=0.85, label=f"{variant}: x image mass")
        if row["mean_image_attention_mass"]["relay"] is not None:
            mass_ax.plot(t, row["mean_image_attention_mass"]["relay"], color=c,
                         lw=1, alpha=0.65, linestyle="--", label="relay3: r image mass")
    ax.set_yticks([0, 1], ["adapter only", "relay3"])
    ax.set_ylim(-0.45, 1.55)
    ax.invert_yaxis()
    ax.set_title(f"{name}: greedy eager-attention replay at checkpoint 4000")
    ax.text(0.99, -0.25, "circle = visual claim   square = answer claim   triangle = Final answer label",
            transform=ax.transAxes, fontsize=8, ha="right")
    mass_ax.set_xlabel("Generated token step (each model's own trajectory)")
    mass_ax.set_ylabel("Mean image-key attention mass\nacross 36 layers")
    mass_ax.legend(fontsize=8, loc="upper right")
    mass_ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    evaluation = {
        variant: {
            dataset: {
                int(row["dataset_index"]) - 1: row
                for line in path.read_text(encoding="utf-8").splitlines()
                if (row := json.loads(line))["checkpoint_step"] == 4000
            }
            for dataset, path in paths.items()
        }
        for variant, paths in EVALUATION_PATHS.items()
    }
    summary = {}
    parts = ["<!doctype html><html lang='en'><meta charset='utf-8'>",
             "<title>Relay3 native-step evidence</title>",
             "<style>body{font:16px system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem}",
             "table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:.45rem}",
             "pre{white-space:pre-wrap;background:#f5f5f5;padding:1rem}img{max-width:100%}",
             "small{color:#555}</style><h1>Native greedy generation: step 4000</h1>",
             "<p><b>Primary result:</b> evaluator rows below are the original greedy evaluation. "
             "Attention capture runs are separate eager-attention replays and can produce different text. "
             "Their event markers must never be assigned to the evaluator response.</p>",
             "<p>Each captured model follows its own generated tokens. Event marks are the first concrete visual claim, "
             "first answer claim, and the final answer label. A visual claim can be wrong. "
             "Image attention mass measures how much attention goes to image keys; it does not establish "
             "which evidence is correct or causally used.</p>"]
    parts += ["<h2>Dataset-level token efficiency, step 4000</h2>",
              "<p>Matched questions where both models were correct and both responses terminated. "
              "This avoids rewarding a short but incorrect answer.</p>",
              "<table><tr><th>Dataset</th><th>Pairs</th><th>Relay3 shorter</th>"
              "<th>Adapter mean tokens</th><th>Relay3 mean tokens</th>"
              "<th>Mean paired saving</th><th>Median paired saving</th></tr>"]
    aggregate = {}
    for dataset in ("mapverse", "mapwise"):
        paired = [(a, evaluation["relay3"][dataset][index])
                  for index, a in evaluation["adapter_only"][dataset].items()
                  if index in evaluation["relay3"][dataset]
                  and a["reward"] == 1 and evaluation["relay3"][dataset][index]["reward"] == 1
                  and a.get("terminated", True)
                  and evaluation["relay3"][dataset][index].get("terminated", True)]
        savings = [a["generated_tokens"] - b["generated_tokens"] for a, b in paired]
        aggregate[dataset] = {
            "both_correct_terminated_pairs": len(paired),
            "relay_shorter_count": sum(value > 0 for value in savings),
            "ties": sum(value == 0 for value in savings),
            "adapter_mean_tokens": statistics.mean(a["generated_tokens"] for a, _ in paired),
            "relay_mean_tokens": statistics.mean(b["generated_tokens"] for _, b in paired),
            "mean_paired_token_saving": statistics.mean(savings),
            "median_paired_token_saving": statistics.median(savings),
        }
        row = aggregate[dataset]
        parts.append(f"<tr><td>{dataset}</td><td>{len(paired)}</td>"
                     f"<td>{row['relay_shorter_count']}</td>"
                     f"<td>{row['adapter_mean_tokens']:.1f}</td>"
                     f"<td>{row['relay_mean_tokens']:.1f}</td>"
                     f"<td>{row['mean_paired_token_saving']:.1f}</td>"
                     f"<td>{row['median_paired_token_saving']:.1f}</td></tr>")
    parts.append("</table>")
    for name in CASES:
        rows = {variant: load_case(args.input, name, variant)
                for variant in ("adapter_only", "relay3")}
        dataset = name.split("_qa")[0]
        index = int(name.split("_qa")[1])
        official = {variant: evaluation[variant][dataset][index]
                    for variant in ("adapter_only", "relay3")}
        summary[name] = {variant: {k: row[k] for k in ("qa_id", "question", "ground_truth", "status", "steps", "events", "event_phrases", "response")}
                         for variant, row in rows.items()}
        for variant in rows:
            summary[name][variant]["official_evaluation"] = {
                "reward": official[variant]["reward"],
                "generated_tokens": official[variant]["generated_tokens"],
                "response": official[variant]["response"],
                "matches_attention_capture": official[variant]["response"] == rows[variant]["response"],
            }
        plot_case(name, rows, args.output / f"{name}.png")
        first = rows["adapter_only"]
        parts += [f"<h2>{html.escape(name)}</h2>",
                  f"<p><b>{html.escape(first['question'])}</b><br>Ground truth: {html.escape(str(first['ground_truth']))}</p>",
                  "<h3>Original greedy evaluation</h3>",
                  "<table><tr><th>Model</th><th>Reward</th><th>Generated tokens</th><th>Matches attention replay?</th></tr>"]
        for variant in rows:
            off = summary[name][variant]["official_evaluation"]
            parts.append(f"<tr><td>{html.escape(variant)}</td><td>{off['reward']:.0f}</td>"
                         f"<td>{off['generated_tokens']}</td><td>{off['matches_attention_capture']}</td></tr>")
        parts += ["</table>", "<h3>Separate attention capture replay</h3>",
                  f"<img src='{name}.png' alt='Generated token timeline and attention mass'>",
                  "<table><tr><th>Model</th><th>Total tokens</th><th>First visual claim</th><th>First answer claim</th><th>Final answer label</th><th>Status</th></tr>"]
        for variant, row in rows.items():
            e = row["events"]
            parts.append(f"<tr><td>{html.escape(variant)}</td><td>{row['steps']}</td>"
                         f"<td>{e['first_visual_claim']}: {html.escape(row['event_phrases']['first_visual_claim'])}</td>"
                         f"<td>{e['first_answer_claim']}: {html.escape(row['event_phrases']['first_answer_claim'])}</td>"
                         f"<td>{e['final_answer_label']}</td><td>{html.escape(row['status'])}</td></tr>")
        parts.append("</table>")
        for variant, row in rows.items():
            parts.append(f"<h3>{html.escape(variant)}: original evaluator response</h3>"
                         f"<pre>{html.escape(official[variant]['response'])}</pre>")
            if official[variant]["response"] != row["response"]:
                parts.append(f"<h3>{html.escape(variant)}: attention replay response (different)</h3>"
                             f"<pre>{html.escape(row['response'])}</pre>")
            parts.append(f"<p><a href='../relay3_visualization_step4000/{name}/comparison.html'>Open attention viewer</a></p>")
    parts.append("</html>")
    (args.output / "report.html").write_text("\n".join(parts), encoding="utf-8")
    (args.output / "events.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps({n: {v: {"steps": r["steps"], **r["events"]} for v, r in vs.items()}
                      for n, vs in summary.items()}, ensure_ascii=False, indent=2))
    print(args.output / "report.html")


if __name__ == "__main__":
    main()
