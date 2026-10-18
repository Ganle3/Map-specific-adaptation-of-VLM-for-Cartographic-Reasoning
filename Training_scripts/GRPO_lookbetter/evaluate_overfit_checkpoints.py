#!/usr/bin/env python3
"""Fixed-seed, frozen-model comparison of initial/trained/no-LOOK controllers."""
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


def _controller(device: torch.device, checkpoint: dict | None, tau0: float, temperature: float,
                initial_size_scale: float) -> WhenWhereController:
    if checkpoint is None:
        cfg = ControllerConfig(tau0=tau0, temperature=temperature, initial_size_scale=initial_size_scale)
    else:
        cfg = ControllerConfig(**checkpoint["controller_config"])
    result = WhenWhereController(cfg).to(device).eval()
    if checkpoint is not None:
        result.load_state_dict(checkpoint["controller"], strict=True)
    return result


def _evaluate_mode(name: str, model, processor, records, controller, cfg: RolloutConfig, output: Path,
                   seed: int, rollouts_per_question: int) -> dict:
    rows, started = [], time.perf_counter()
    with output.open("w", encoding="utf-8") as handle:
        for dataset_index, item in records:
            runs = []
            for rollout_index in range(rollouts_per_question):
                run_seed = seed + dataset_index * 10_000 + rollout_index
                random.seed(run_seed); np.random.seed(run_seed); torch.manual_seed(run_seed)
                began = time.perf_counter()
                run = rollout(model, processor, item, controller, cfg)
                run.update({"rollout_index": rollout_index, "seed": run_seed,
                            "elapsed_seconds": time.perf_counter() - began})
                runs.append(run)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            row = {"mode": name, "dataset_index": dataset_index, "sample_id": item.sample_id,
                   "dataset": item.metadata.get("dataset"), "question": item.question,
                   "correct_answer": item.answer, "rollouts": runs}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n"); handle.flush()
            rows.append(row)
            print(f"[{name}] index={dataset_index} rewards={[run['reward'] for run in runs]}", flush=True)
    per_question = []
    for row in rows:
        rewards = [run["reward"] for run in row["rollouts"]]
        looks = [run["look_count"] for run in row["rollouts"]]
        per_question.append({"dataset_index": row["dataset_index"], "question": row["question"],
                             "accuracy": float(np.mean(rewards)), "correct": int(sum(rewards)),
                             "rollouts": len(rewards), "mean_looks": float(np.mean(looks)),
                             "answers": [extract_final_answer(run["response"]) for run in row["rollouts"]]})
    return {"mode": name, "mean_accuracy": float(np.mean([x["accuracy"] for x in per_question])),
            "mean_looks": float(np.mean([x["mean_looks"] for x in per_question])),
            "elapsed_seconds": time.perf_counter() - started, "per_question": per_question,
            "trajectory_jsonl": str(output.resolve())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument("--mapverse-image-root", type=Path, required=True)
    parser.add_argument("--mapwise-root", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-index", type=int, default=245)
    parser.add_argument("--end-index", type=int, default=248)
    parser.add_argument("--rollouts-per-question", type=int, default=32)
    parser.add_argument("--tau0", type=float, default=0.091112)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--max-reasoning-tokens", type=int, default=500)
    parser.add_argument("--max-looks", type=int, default=10)
    parser.add_argument("--cooldown-tokens", type=int, default=20)
    parser.add_argument("--eligible-after-tokens", type=int, default=12)
    parser.add_argument("--eligible-stride", type=int, default=4)
    parser.add_argument("--revision-tokens", type=int, default=0)
    parser.add_argument("--attention-mass", type=float, default=0.75)
    parser.add_argument("--proposal-cap", type=int, default=6)
    parser.add_argument("--initial-size-scale", type=float, default=1.5)
    parser.add_argument("--roi-key-bias", type=float, default=3.079442)
    parser.add_argument("--global-max-pixels", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=20261017)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_all = load_mapverse_questions(args.dataset_json, args.mapverse_image_root, mapwise_root=args.mapwise_root)
    records = list(enumerate(records_all[args.start_index - 1:args.end_index], start=args.start_index))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "controller" not in checkpoint or "controller_config" not in checkpoint:
        raise ValueError("Checkpoint lacks controller/controller_config; require a completed training checkpoint.")
    model, processor = load_model(None, args.adapter)
    model.requires_grad_(False).eval()
    device = model.get_input_embeddings().weight.device
    base = dict(max_reasoning_tokens=args.max_reasoning_tokens, revision_tokens=args.revision_tokens,
                max_looks=args.max_looks, cooldown_tokens=args.cooldown_tokens,
                eligible_after_tokens=args.eligible_after_tokens, eligible_stride=args.eligible_stride,
                attention_mass=args.attention_mass, proposal_cap=args.proposal_cap,
                global_bias_mode="off", roi_key_bias=args.roi_key_bias, global_max_pixels=args.global_max_pixels)
    modes = [
        ("initial_policy", _controller(device, None, args.tau0, args.temperature, args.initial_size_scale), RolloutConfig(**base)),
        ("trained_policy", _controller(device, checkpoint, args.tau0, args.temperature, args.initial_size_scale), RolloutConfig(**base)),
        ("no_look_adapter", _controller(device, None, 1e6, args.temperature, args.initial_size_scale), RolloutConfig(**{**base, "max_looks": 0})),
    ]
    result = {"evaluation_config": vars(args), "trained_controller_config": checkpoint["controller_config"],
              "modes": []}
    for name, controller, cfg in modes:
        result["modes"].append(_evaluate_mode(name, model, processor, records, controller, cfg,
                                                args.output_dir / f"{name}.jsonl", args.seed,
                                                args.rollouts_per_question))
    (args.output_dir / "comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
