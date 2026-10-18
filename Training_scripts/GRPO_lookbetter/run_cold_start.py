#!/usr/bin/env python3
"""Distributed, no-update cold-start evaluation for the When--Where policy."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SPARSE = ROOT / "SparseVLM_eliminateCONTEXT"
for directory in (str(Path(__file__).resolve().parent), str(SPARSE)):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from continuous_cache_qwen3vl import load_model
from data import extract_final_answer, load_mapverse_questions
from policy import ControllerConfig, WhenWhereController
from rollout import RolloutConfig, rollout


def _calibrate_tau0(model, processor, records, cfg: RolloutConfig, temperature: float, count: int, quantile: float):
    """Measure raw token JSD under an effectively never-LOOK policy."""
    neutral = WhenWhereController(ControllerConfig(tau0=1e6, temperature=temperature)).to(
        model.get_input_embeddings().weight.device
    )
    values = []
    for item in records[:count]:
        trajectory = rollout(model, processor, item, neutral, cfg)
        values.extend(row["features"][0] for row in trajectory["actions"] if row["kind"] == "when")
    if not values:
        raise RuntimeError("Calibration recorded no eligible WHEN boundaries.")
    return float(np.quantile(values, quantile)), {"jsd_count": len(values), "quantile": quantile,
                                                   "tau0": float(np.quantile(values, quantile))}


def _write_readable_question(row: dict, destination: Path) -> None:
    """Write the evidence needed to inspect one question without parsing JSONL."""
    lines = [f"# Question {row['dataset_index']}: {row['sample_id']}", "",
             f"- Dataset: `{row.get('dataset')}`", f"- Image: `{row['image']}`",
             f"- Question: {row['question']}", f"- Gold answer: `{row['correct_answer']}`", ""]
    lines += ["## Rollout summary", "",
              "| Rollout | Reward | Parsed final answer | Natural EOS | LOOK updates | Time |",
              "|---:|---:|---|:---:|---:|---:|"]
    for run in row["rollouts"]:
        answer = (extract_final_answer(run["response"]) or "(missing)").replace("|", "\\|").replace("\n", " ")
        eos = "yes" if run.get("generation_terminated_by_eos", False) else "no (budget)"
        lines.append(f"| {run['rollout_index']} | {run['reward']:.0f} | {answer} | {eos} | "
                     f"{run['look_count']} | {run['elapsed_seconds']:.1f}s |")
    lines.append("")
    for run in row["rollouts"]:
        answer = extract_final_answer(run["response"]) or "(missing)"
        response = run["response"]
        marker = "</think>"
        if marker in response:
            thinking, answer_channel = response.split(marker, 1)
            answer_channel = answer_channel.strip()
        else:
            thinking, answer_channel = response, ""
        lines += [f"## Rollout {run['rollout_index']}", "",
                  f"- Reward: `{run['reward']}`; parsed final answer: `{answer}`",
                  f"- Seed: `{run['seed']}`; elapsed: `{run['elapsed_seconds']:.1f}s`",
                  f"- Natural termination: `{'EOS' if run.get('generation_terminated_by_eos', False) else 'token budget reached'}`",
                  f"- Evaluator: `{run['evaluation']['evaluator']}` / `{run['evaluation']['metric']}`; "
                  f"canonical prediction: `{run['evaluation']['canonical_prediction']}`; "
                  f"canonical gold: `{run['evaluation']['canonical_target']}`",
                  "- LOOK attempts / successful ROI-focus updates / selected regions: "
                  f"`{run['look_attempt_count']} / {run['look_count']} / {run['selected_crop_count']}`",
                  f"- WHEN probes: `{run['when_probe_count']}` ({run['when_probe_seconds']:.2f}s)",
                  f"- Attention modulation: `{run['effective_attention_modulation']}`", "",
                  "### ROI boxes on the immutable global image KV", ""]
        if run["look_events"]:
            for event_number, event in enumerate(run["look_events"], 1):
                boxes = "; ".join(
                    f"({region['left']},{region['top']})-({region['right']},{region['bottom']}); score={region['score']:.4g}"
                    for region in event["regions"]
                )
                lines.append(f"{event_number}. reasoning token {event['token']}; "
                             f"ROI visual tokens `{event.get('roi_visual_tokens', 0)}`; "
                             f"logit bias `+{event.get('roi_key_bias', 0.0):.6g}`; "
                             f"HOW `{event.get('how_refresh', 'none')}`; "
                             f"teacher-forced replay `{event.get('teacher_forced_replay_tokens', 0)}` tokens; "
                             f"revised/regenerated `{event.get('revision_tokens', 0)}/{event.get('regenerated_tokens', 0)}`: {boxes}")
        else:
            lines.append("No ROI focus update was applied.")
        lines += ["", "### Natural answer channel", "", "```text",
                  answer_channel or "(The model did not naturally close </think> before EOS/budget.)", "```", "",
                  "<details>", "<summary>Thinking trace</summary>", "", "```text", thinking, "```", "",
                  "</details>", ""]
    destination.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument("--mapverse-image-root", type=Path, required=True)
    parser.add_argument("--mapwise-root", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--controller-checkpoint", type=Path, default=None,
                        help="Optional controller_final.pt/controller_last.pt from train_when_where_grpo.py.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--readable-output-dir", type=Path, default=None,
                        help="Directory for one Markdown reasoning report per question (default: OUTPUT_DIR/readable).")
    parser.add_argument("--start-index", type=int, required=True, help="1-based inclusive dataset index.")
    parser.add_argument("--end-index", type=int, required=True, help="1-based inclusive dataset index.")
    parser.add_argument("--rollouts-per-question", type=int, default=6)
    parser.add_argument("--tau0", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=0.02)
    parser.add_argument("--calibration-examples", type=int, default=8)
    parser.add_argument("--target-jsd-quantile", type=float, default=0.85)
    parser.add_argument("--max-reasoning-tokens", type=int, default=320)
    parser.add_argument("--revision-tokens", type=int, default=0,
                        help="After a successful LOOK, regenerate this recent reasoning suffix under its ROI-conditioned anchor refresh.")
    parser.add_argument("--max-looks", type=int, default=10)
    parser.add_argument("--cooldown-tokens", type=int, default=20)
    parser.add_argument("--eligible-after-tokens", type=int, default=12)
    parser.add_argument("--eligible-stride", type=int, default=4)
    parser.add_argument("--when-probe-stride", type=int, default=4)
    parser.add_argument("--when-attention-source", choices=("main", "ungated_probe"), default="main",
                        help="main is fast and is ungated when global-bias-mode=off; ungated_probe clones KV at probe boundaries.")
    parser.add_argument("--attention-mass", type=float, default=0.60,
                        help="Cumulative positive attribution mass retained before connected-component proposals. Higher means broader proposals.")
    parser.add_argument("--proposal-cap", type=int, default=6)
    parser.add_argument("--initial-size-scale", type=float, default=1.0,
                        help="Initial multiplicative centre for trainable ROI width/height actions; 1.5 starts 50%% larger than a component.")
    parser.add_argument("--global-max-pixels", type=int, default=1_000_000)
    parser.add_argument("--global-bias-mode", choices=("off", "look_snapshot"), default="off")
    parser.add_argument("--global-key-bias", type=float, default=-4.0)
    parser.add_argument("--global-retain-attention-mass", type=float, default=0.40)
    parser.add_argument("--roi-key-bias", type=float, default=2.079442,
                        help="Additive attention-logit bias for selected existing global visual KV keys.")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    if args.start_index < 1 or args.end_index < args.start_index or args.rollouts_per_question < 2:
        parser.error("Use 1-based start/end indices and at least two rollouts per question.")
    if min(args.max_looks, args.when_probe_stride, args.eligible_stride, args.global_max_pixels, args.proposal_cap) < 1:
        parser.error("LOOK/probe strides and pixel caps must be positive.")
    if not 0 < args.attention_mass <= 1:
        parser.error("attention-mass must be in (0, 1].")
    if args.initial_size_scale <= 0:
        parser.error("initial-size-scale must be positive.")
    if args.roi_key_bias < 0:
        parser.error("roi-key-bias must be non-negative.")
    if args.revision_tokens < 0:
        parser.error("revision-tokens must be non-negative.")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    records = load_mapverse_questions(args.dataset_json, args.mapverse_image_root, mapwise_root=args.mapwise_root)
    if args.end_index > len(records):
        parser.error(f"end-index {args.end_index} exceeds dataset size {len(records)}.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    readable_dir = args.readable_output_dir or args.output_dir / "readable"
    readable_dir.mkdir(parents=True, exist_ok=True)
    model, processor = load_model(args.model, args.adapter)
    model.requires_grad_(False).eval()
    rollout_cfg = RolloutConfig(
        max_reasoning_tokens=args.max_reasoning_tokens,
        revision_tokens=args.revision_tokens,
        max_looks=args.max_looks, cooldown_tokens=args.cooldown_tokens,
        eligible_after_tokens=args.eligible_after_tokens, eligible_stride=args.eligible_stride,
        attention_mass=args.attention_mass, proposal_cap=args.proposal_cap,
        when_probe_stride=args.when_probe_stride, global_max_pixels=args.global_max_pixels,
        when_attention_source=args.when_attention_source,
        global_bias_mode=args.global_bias_mode,
        global_key_bias=args.global_key_bias, global_retain_attention_mass=args.global_retain_attention_mass,
        roi_key_bias=args.roi_key_bias,
    )
    if args.tau0 is None:
        tau0, calibration = _calibrate_tau0(model, processor, records, rollout_cfg, args.temperature,
                                             args.calibration_examples, args.target_jsd_quantile)
    else:
        tau0, calibration = args.tau0, {"tau0": args.tau0, "source": "CLI"}
    if args.controller_checkpoint is not None:
        try:
            checkpoint = torch.load(args.controller_checkpoint, map_location="cpu", weights_only=True)
        except TypeError:  # PyTorch before weights_only support
            checkpoint = torch.load(args.controller_checkpoint, map_location="cpu")
        if "controller" not in checkpoint or "controller_config" not in checkpoint:
            parser.error("controller-checkpoint must contain controller and controller_config.")
        controller_cfg = ControllerConfig(**checkpoint["controller_config"])
    else:
        checkpoint = None
        controller_cfg = ControllerConfig(
            tau0=tau0, temperature=args.temperature, initial_size_scale=args.initial_size_scale
        )
    controller = WhenWhereController(controller_cfg).to(
        model.get_input_embeddings().weight.device
    ).eval()
    if checkpoint is not None:
        controller.load_state_dict(checkpoint["controller"], strict=True)
    settings = {"args": vars(args), "controller": asdict(controller.cfg), "rollout": asdict(rollout_cfg),
                "calibration": calibration, "dataset_size": len(records)}
    (args.output_dir / "cold_start_config.json").write_text(json.dumps(settings, default=str, indent=2), encoding="utf-8")
    output = args.output_dir / f"cold_start_{args.start_index:04d}_{args.end_index:04d}.jsonl"
    summaries, started = [], time.perf_counter()
    with output.open("w", encoding="utf-8") as handle:
        for index in range(args.start_index - 1, args.end_index):
            item = records[index]
            group = []
            for rollout_index in range(args.rollouts_per_question):
                run_seed = args.seed + index * 10_000 + rollout_index
                random.seed(run_seed); np.random.seed(run_seed); torch.manual_seed(run_seed)
                began = time.perf_counter()
                result = rollout(model, processor, item, controller, rollout_cfg)
                result["rollout_index"] = rollout_index
                result["seed"] = run_seed
                result["elapsed_seconds"] = time.perf_counter() - began
                group.append(result)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            row = {"dataset_index": index + 1, "sample_id": item.sample_id, "dataset": item.metadata.get("dataset"),
                   "image": str(item.image_path), "question": item.question, "correct_answer": item.answer,
                   "rollouts": group,
                   "summary": {"rewards": [run["reward"] for run in group],
                               "final_answers": [extract_final_answer(run["response"]) for run in group],
                               "look_attempt_counts": [run["look_attempt_count"] for run in group],
                               "look_counts": [run["look_count"] for run in group],
                               "crop_counts": [run["selected_crop_count"] for run in group],
                               "mixed_reward": len({run["reward"] for run in group}) > 1}}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n"); handle.flush()
            _write_readable_question(row, readable_dir / f"question_{index + 1:04d}.md")
            summaries.append(row["summary"])
            print(f"[{index + 1}/{args.end_index}] id={item.sample_id} rewards={row['summary']['rewards']} "
                  f"looks={row['summary']['look_attempt_counts']}")
    aggregate = {"questions": len(summaries), "mixed_reward_questions": sum(row["mixed_reward"] for row in summaries),
                 "mean_look_attempts": float(np.mean([value for row in summaries for value in row["look_attempt_counts"]])),
                 "mean_elapsed_per_question": (time.perf_counter() - started) / len(summaries), "jsonl": str(output.resolve()),
                 "readable_reports": str(readable_dir.resolve())}
    (args.output_dir / f"cold_start_{args.start_index:04d}_{args.end_index:04d}_summary.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
