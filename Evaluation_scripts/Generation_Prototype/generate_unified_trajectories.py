#!/usr/bin/env python3
"""Generate resumable GRPO-style rollouts for MapWise and MapVerse."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODEL_NAME = "Qwen/Qwen3-VL-8B-Thinking"
SCHEMA_VERSION = "unified_vlm_trajectories_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapwise-json", type=Path, required=True)
    parser.add_argument("--mapwise-image-root", type=Path, required=True)
    parser.add_argument("--mapverse-json", type=Path, required=True)
    parser.add_argument("--mapverse-image-root", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-rollouts", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--min-pixels", type=int, default=65536)
    parser.add_argument("--max-pixels", type=int, default=1000000)
    parser.add_argument("--limit", type=int, default=None,
                        help="Debug only: process this many questions per dataset.")
    return parser.parse_args()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_json_list(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise TypeError(f"Expected a JSON list of objects: {path}")
    return value


def normalize_question(dataset: str, row: dict[str, Any], index: int) -> dict[str, Any]:
    if dataset == "mapwise":
        qa_id = str(row.get("qa_id") or
                    f"mapwise_{row.get('country', 'unknown')}_{row.get('map_no', 'unknown')}_"
                    f"t{row.get('template_no', -1)}_idx{index:04d}")
        answer = str(row.get("ground_truth", "")).strip()
        answer_type = str(row.get("ground_truth_type", "")).strip()
        image_name = str(row.get("map_no", "")).strip()
        source_id = row.get("source_index", index)
    else:
        source_id = row.get("source_row", index)
        qa_id = str(row.get("qa_id") or f"mapverse_source_row_{source_id}")
        answer = str(row.get("correct_answer", "")).strip()
        answer_type = str(row.get("answer_type", "")).strip()
        image_name = str(row.get("image_name", "")).strip()
    if not str(row.get("question", "")).strip() or not answer or not image_name:
        raise ValueError(f"Incomplete {dataset} row {index}: {row}")
    return {
        "dataset": dataset,
        "dataset_index": index,
        "qa_id": qa_id,
        "source_id": source_id,
        "image_name": image_name,
        "question": str(row["question"]).strip(),
        "reference_answer": answer,
        "answer_type": answer_type,
        "ability_level": str(row.get("ability_level", "")).strip(),
        "source_metadata": dict(row),
    }


def score_rollout(dataset: str, scorer, question: dict[str, Any], raw: str,
                  final_answer: str) -> tuple[bool, dict[str, Any]]:
    if dataset == "mapverse":
        result = scorer.evaluate_exact(raw, question["reference_answer"], question["answer_type"])
    else:
        source = question["source_metadata"]
        record = {
            **source,
            "qa_id": question["qa_id"],
            "ground_truth": question["reference_answer"],
            "ground_truth_type": question["answer_type"],
            "raw_response": raw,
            "final_answer": final_answer,
        }
        result = scorer.evaluate_sample(record)
    correct = bool(result.get("correct", result.get("strict_exact_match", result.get("reward", 0))))
    keep = {
        key: value for key, value in result.items()
        if key in {
            "reward", "metric", "prediction_canonical", "ground_truth_canonical",
            "normalized_prediction", "normalized_ground_truth", "answer_extraction_method",
            "evaluation_note", "effective_ground_truth_type",
        }
    }
    return correct, keep


def load_journal(path: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    records: dict[tuple[str, str, int], dict[str, Any]] = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["dataset"]), str(row["qa_id"]), int(row["rollout_index"]))
            if key in records:
                raise ValueError(f"Duplicate journal key {key} at line {line_number}")
            records[key] = row
    return records


def build_output(manifest: dict[str, Any], questions: list[dict[str, Any]],
                 records: dict[tuple[str, str, int], dict[str, Any]], num_rollouts: int) -> dict[str, Any]:
    output_questions = []
    counts = defaultdict(int)
    completed = 0
    for question in questions:
        trajectories = [records[(question["dataset"], question["qa_id"], i)]
                        for i in range(num_rollouts)
                        if (question["dataset"], question["qa_id"], i) in records]
        scores = [bool(item["correct"]) for item in trajectories]
        if len(scores) < num_rollouts:
            outcome = "incomplete"
        elif all(scores):
            outcome = "all_correct"
        elif not any(scores):
            outcome = "all_incorrect"
        else:
            outcome = "mixed"
        counts[outcome] += 1
        completed += len(trajectories) == num_rollouts
        output_questions.append({
            **question,
            "rollout_outcome": outcome,
            "num_correct": sum(scores),
            "num_incorrect": len(scores) - sum(scores),
            "trajectories": trajectories,
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "metadata": copy.deepcopy(manifest),
        "summary": {
            "questions_total": len(questions),
            "questions_complete": completed,
            "trajectories_written": len(records),
            "outcome_counts": dict(counts),
        },
        "questions": output_questions,
    }


def main() -> None:
    args = parse_args()
    if args.num_rollouts < 1 or args.max_new_tokens < 1:
        raise ValueError("num-rollouts and max-new-tokens must be positive")

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[1]
    helper_dir = project_root / "Evaluation_scripts" / "GRPO_ablation"
    sys.path.insert(0, str(helper_dir))
    import torch
    from PIL import Image
    from transformers import set_seed
    import inference_mapwise_trl as inference
    from inference_mapwise_trl_e2 import install

    paths = {
        "mapwise_json": args.mapwise_json.expanduser().resolve(),
        "mapverse_json": args.mapverse_json.expanduser().resolve(),
        "mapwise_image_root": args.mapwise_image_root.expanduser().resolve(),
        "mapverse_image_root": args.mapverse_image_root.expanduser().resolve(),
        "adapter_path": args.adapter_path.expanduser().resolve(),
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name}: {path}")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    journal_path = output_dir / "rollouts.jsonl"
    output_path = output_dir / "unified_trajectories.json"
    manifest_path = output_dir / "manifest.json"

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "base_model": MODEL_NAME,
        "adapter_path": str(paths["adapter_path"]),
        "datasets": {
            "mapwise": {"path": str(paths["mapwise_json"]), "sha256": sha256(paths["mapwise_json"])},
            "mapverse": {"path": str(paths["mapverse_json"]), "sha256": sha256(paths["mapverse_json"])},
        },
        "generation_config": {
            "do_sample": True, "num_rollouts": args.num_rollouts,
            "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens, "num_beams": 1,
            "repetition_penalty": 1.0, "use_cache": True, "thinking_mode": "auto",
            "seed": args.seed,
        },
        "image_processor": {"min_pixels": args.min_pixels, "max_pixels": args.max_pixels},
        "limit_per_dataset": args.limit,
    }
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError(f"Run configuration differs from existing manifest: {manifest_path}")
    else:
        write_json_atomic(manifest_path, manifest)

    raw_questions = []
    for dataset, json_key in (("mapwise", "mapwise_json"), ("mapverse", "mapverse_json")):
        rows = read_json_list(paths[json_key])
        if args.limit is not None:
            rows = rows[:args.limit]
        raw_questions.extend(normalize_question(dataset, row, i) for i, row in enumerate(rows))

    mapwise_scorer = load_module("mapwise_exact", helper_dir / "mapwise_evaluation_exact.py")
    mapverse_scorer = load_module("mapverse_exact", helper_dir / "mapverse_evaluation_exact.py")
    scorers = {"mapwise": mapwise_scorer, "mapverse": mapverse_scorer}
    records = load_journal(journal_path)
    valid_keys = {(q["dataset"], q["qa_id"], i) for q in raw_questions for i in range(args.num_rollouts)}
    unexpected = set(records) - valid_keys
    if unexpected:
        raise ValueError(f"Journal contains records outside this run; first unexpected key: {next(iter(unexpected))}")

    print(f"Questions: {len(raw_questions)}; existing trajectories: {len(records)}", flush=True)
    install(inference, output_dir)
    model, processor, loaded_adapter = inference.load_model_and_processor(
        model_name=MODEL_NAME, adapter_path=paths["adapter_path"],
        min_pixels=args.min_pixels, max_pixels=args.max_pixels,
    )
    eos_ids = model.generation_config.eos_token_id
    eos_ids = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])

    for global_index, question in enumerate(raw_questions):
        dataset = question["dataset"]
        image_root = paths[f"{dataset}_image_root"]
        sample = dict(question["source_metadata"])
        if dataset == "mapverse":
            sample.update(country="mapverse", map_no=question["image_name"],
                          image_name=question["image_name"], qa_id=question["qa_id"])
        else:
            sample["qa_id"] = question["qa_id"]
        image_path = inference.resolve_mapwise_image(sample, image_root)
        with Image.open(image_path) as image:
            inputs = inference.prepare_multimodal_inputs(
                processor, image.convert("RGB"),
                inference.build_mapwise_prompt(question["question"]), thinking_mode="auto")
        inputs = inference.move_inputs_to_model_device(inputs, model)
        prompt_tokens = int(inputs["input_ids"].shape[1])

        for rollout_index in range(args.num_rollouts):
            key = (dataset, question["qa_id"], rollout_index)
            if key in records:
                continue
            seed = args.seed + global_index * args.num_rollouts + rollout_index
            set_seed(seed)
            started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    **inputs, do_sample=True, temperature=args.temperature,
                    top_p=args.top_p, top_k=args.top_k, min_p=None,
                    max_new_tokens=args.max_new_tokens, num_beams=1,
                    num_return_sequences=1, repetition_penalty=1.0, use_cache=True,
                )
            token_ids = generated[0, prompt_tokens:]
            elapsed = time.perf_counter() - started
            raw = processor.batch_decode(
                token_ids[None, :], skip_special_tokens=True,
                clean_up_tokenization_spaces=False)[0].strip()
            final_answer = inference.extract_final_answer(raw)
            terminated = bool(token_ids.numel() and int(token_ids[-1]) in eos_ids)
            status = inference.classify_generation_status(
                raw, final_answer, int(token_ids.numel()), args.max_new_tokens)
            correct, evaluation = score_rollout(
                dataset, scorers[dataset], question, raw, final_answer)
            record = {
                "dataset": dataset, "qa_id": question["qa_id"],
                "rollout_index": rollout_index, "seed": seed,
                "raw_response": raw, "final_answer": final_answer,
                "correct": correct, "reward": 1.0 if correct else 0.0,
                "generated_tokens": int(token_ids.numel()), "terminated": terminated,
                "generation_status": status, "inference_seconds": elapsed,
                "evaluation": evaluation,
            }
            with journal_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
            records[key] = record
            print(f"[{len(records)}/{len(valid_keys)}] {dataset} {question['qa_id']} "
                  f"rollout={rollout_index} correct={int(correct)} tokens={token_ids.numel()}", flush=True)
            del generated, token_ids
        del inputs

        # The JSONL is the per-rollout durability layer. Rebuilding the large
        # nested JSON less often avoids quadratic I/O over 2,000 questions.
        if (global_index + 1) % 25 == 0 or global_index + 1 == len(raw_questions):
            snapshot = build_output(manifest, raw_questions, records, args.num_rollouts)
            snapshot["metadata"]["snapshot_utc"] = datetime.now(timezone.utc).isoformat()
            write_json_atomic(output_path, snapshot)

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    final = build_output(manifest, raw_questions, records, args.num_rollouts)
    final["metadata"]["completed_utc"] = datetime.now(timezone.utc).isoformat()
    write_json_atomic(output_path, final)
    write_json_atomic(output_dir / "completed.json", final["summary"])
    print(f"COMPLETE: {output_path}", flush=True)


if __name__ == "__main__":
    main()
