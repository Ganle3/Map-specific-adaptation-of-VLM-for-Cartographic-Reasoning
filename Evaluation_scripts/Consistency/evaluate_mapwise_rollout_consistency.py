#!/usr/bin/env python3
"""Evaluate an adapted MapWise model with GRPO-style stochastic rollouts.

The Test split and image root are deliberately required CLI arguments.  This
avoids silently evaluating a different split when the repository is copied to
Euler or another machine.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


MODEL_NAME = "Qwen/Qwen3-VL-8B-Thinking"
NUM_ROLLOUTS = 4
# Keep these aligned with train_mapwise_grpo_visLoRA_trl.py and the submitted
# rollout job, unless explicitly overridden at launch.
TEMPERATURE = 0.8
TOP_P = 0.95
TOP_K = 0
MAX_NEW_TOKENS = 1536
SEED = 3407

SCRIPT_DIR = Path(__file__).resolve().parent
VLM_ROOT = SCRIPT_DIR.parents[1]
INFERENCE_DIR = VLM_ROOT / "Evaluation_scripts" / "GRPO_ablation"
DEFAULT_EVALUATOR = INFERENCE_DIR / "mapwise_evaluation_exact.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-json", type=Path, required=True,
                        help="MapWise Test JSON. Required: no split is hard-coded.")
    parser.add_argument("--image-root", type=Path, required=True,
                        help="Root containing the MapWise country image directories.")
    parser.add_argument("--adapter-path", type=Path, required=True,
                        help="PEFT/LoRA checkpoint to evaluate.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--evaluation-script", type=Path, default=DEFAULT_EVALUATOR,
                        help="Strict-exact scorer used by GRPO.")
    parser.add_argument("--num-rollouts", type=int, default=NUM_ROLLOUTS)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=TOP_P)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--min-pixels", type=int, default=65536)
    parser.add_argument("--max-pixels", type=int, default=1000000)
    parser.add_argument("--thinking", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional number of Test questions, for smoke tests only.")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-load-in-4bit", action="store_false", dest="load_in_4bit")
    return parser.parse_args()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_answer_key(value: Any) -> str:
    """A stable answer identity from the GRPO evaluator's canonical output."""
    if value is None:
        return "<unparseable>"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    text = str(value).strip()
    return text if text else "<unparseable>"


def answer_statistics(rollouts: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Return modal-answer agreement and Shannon entropy for one question."""
    answers = [normalized_answer_key(r.get("normalized_prediction")) for r in rollouts]
    counts = Counter(answers)
    total = len(answers)
    if not total:
        return {"answer_counts": {}, "unique_answers": 0,
                "answer_consistency": 0.0, "answer_entropy_bits": 0.0,
                "normalized_answer_entropy": 0.0}
    entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
    max_entropy = math.log2(total) if total > 1 else 0.0
    return {
        "answer_counts": dict(sorted(counts.items())),
        "unique_answers": len(counts),
        # Probability that a rollout agrees with the modal answer.
        "answer_consistency": max(counts.values()) / total,
        "answer_entropy_bits": entropy,
        "normalized_answer_entropy": entropy / max_entropy if max_entropy else 0.0,
    }


def summarize(groups: list[Mapping[str, Any]], manifest: Mapping[str, Any]) -> dict[str, Any]:
    complete = [g for g in groups if len(g.get("rollouts", [])) == manifest["generation_config"]["num_rollouts"]]
    rollouts = [r for g in complete for r in g["rollouts"]]
    correct = sum(int(r["correct"]) for r in rollouts)
    pass_count = sum(any(r["correct"] for r in g["rollouts"]) for g in complete)
    all_correct = sum(all(r["correct"] for r in g["rollouts"]) for g in complete)
    consistency = [float(g["answer_metrics"]["answer_consistency"]) for g in complete]
    entropy = [float(g["answer_metrics"]["answer_entropy_bits"]) for g in complete]
    normalized_entropy = [float(g["answer_metrics"]["normalized_answer_entropy"]) for g in complete]
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": dict(manifest),
        "completed_questions": len(complete),
        "total_rollouts": len(rollouts),
        "rollout_accuracy": correct / len(rollouts) if rollouts else 0.0,
        # GRPO-style best-of-N: the question passes if at least one rollout is correct.
        "pass_rate": pass_count / len(complete) if complete else 0.0,
        "all_rollouts_correct_rate": all_correct / len(complete) if complete else 0.0,
        "answer_consistency": sum(consistency) / len(consistency) if consistency else 0.0,
        "answer_entropy_bits": sum(entropy) / len(entropy) if entropy else 0.0,
        "normalized_answer_entropy": (
            sum(normalized_entropy) / len(normalized_entropy) if normalized_entropy else 0.0
        ),
    }


def load_journal(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    records: dict[tuple[str, int], dict[str, Any]] = {}
    if not path.exists():
        return records
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        key = (str(record["qa_id"]), int(record["rollout_index"]))
        if key in records:
            raise ValueError(f"Duplicate rollout in journal at line {line_number}: {key}")
        records[key] = record
    return records


def score_rollout(evaluator, sample: Mapping[str, Any], raw_response: str) -> dict[str, Any]:
    result = evaluator.evaluate_sample({**sample, "raw_response": raw_response, "final_answer": ""})
    return {
        "correct": bool(result["strict_exact_match"]),
        "evaluated_answer": result.get("evaluated_answer", ""),
        "normalized_prediction": result.get("normalized_prediction"),
        "normalized_ground_truth": result.get("normalized_ground_truth"),
        "metric": result.get("metric", ""),
        "answer_extraction_method": result.get("answer_extraction_method", ""),
        "evaluation_note": result.get("evaluation_note", ""),
    }


def main() -> None:
    args = parse_args()
    if args.num_rollouts != NUM_ROLLOUTS:
        raise ValueError(f"This consistency protocol requires exactly {NUM_ROLLOUTS} rollouts per question.")
    if args.max_new_tokens < 1 or args.temperature <= 0 or not 0 < args.top_p <= 1 or args.top_k < 0:
        raise ValueError("Invalid sampling configuration.")

    qa_json = args.qa_json.expanduser().resolve()
    image_root = args.image_root.expanduser().resolve()
    adapter_path = args.adapter_path.expanduser().resolve()
    evaluator_path = args.evaluation_script.expanduser().resolve()
    for name, path in {"qa JSON": qa_json, "image root": image_root,
                       "adapter": adapter_path, "evaluator": evaluator_path}.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name}: {path}")

    sys.path.insert(0, str(INFERENCE_DIR))
    import inference_mapwise_trl as inference
    evaluator = load_module("mapwise_grpo_exact_evaluator", evaluator_path)
    samples = inference.load_json_list(qa_json)
    if args.limit is not None:
        samples = samples[:args.limit]

    output_dir = args.output_dir.expanduser().resolve()
    journal_path = output_dir / "rollouts.jsonl"
    details_path = output_dir / "question_metrics.json"
    summary_path = output_dir / "summary.json"
    manifest_path = output_dir / "manifest.json"
    manifest = {
        "protocol": "mapwise_adapted_rollout_consistency_v1",
        "model_name": args.model_name,
        "adapter_path": str(adapter_path),
        "qa_json": str(qa_json),
        "qa_json_sha256": sha256(qa_json),
        "image_root": str(image_root),
        "evaluator": str(evaluator_path),
        "generation_config": {"num_rollouts": args.num_rollouts, "do_sample": True,
                              "temperature": args.temperature, "top_p": args.top_p,
                              "top_k": args.top_k, "max_new_tokens": args.max_new_tokens,
                              "seed": args.seed, "thinking": args.thinking},
        "image_processor": {"min_pixels": args.min_pixels, "max_pixels": args.max_pixels},
        "limit": args.limit,
    }
    if args.overwrite:
        for path in (journal_path, details_path, summary_path, manifest_path):
            if path.exists():
                path.unlink()
    elif args.no_resume and journal_path.exists():
        raise FileExistsError(
            f"Refusing to append duplicate rollouts with --no-resume: {journal_path}. "
            "Use --overwrite or a fresh --output-dir."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError(f"Existing output has a different configuration: {manifest_path}")
    else:
        atomic_write_json(manifest_path, manifest)

    records = load_journal(journal_path) if not args.no_resume else {}
    valid_keys = {(str(s["qa_id"]), i) for s in samples for i in range(args.num_rollouts)}
    if set(records) - valid_keys:
        raise ValueError("Journal contains rollout(s) outside the requested Test set/configuration.")

    print(f"Test questions: {len(samples)} | existing rollouts: {len(records)}", flush=True)
    model, processor, _ = inference.load_model_and_processor(
        model_name=args.model_name, adapter_path=adapter_path,
        min_pixels=args.min_pixels, max_pixels=args.max_pixels,
        load_in_4bit=args.load_in_4bit,
    )
    try:
        for question_index, sample in enumerate(samples):
            qa_id = str(sample["qa_id"])
            image_path = inference.resolve_mapwise_image(sample, image_root)
            prompt = inference.build_mapwise_prompt(str(sample["question"]))
            for rollout_index in range(args.num_rollouts):
                key = (qa_id, rollout_index)
                if key in records:
                    continue
                rollout_seed = args.seed + question_index * args.num_rollouts + rollout_index
                started = time.perf_counter()
                raw, generated_tokens = inference.generate_response(
                    model, processor, image_path, prompt,
                    max_new_tokens=args.max_new_tokens, do_sample=True,
                    temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                    seed=rollout_seed, thinking_mode=args.thinking,
                )
                scored = score_rollout(evaluator, sample, raw)
                record = {"qa_id": qa_id, "question_index": question_index,
                          "rollout_index": rollout_index, "seed": rollout_seed,
                          "raw_response": raw, "generated_tokens": generated_tokens,
                          "inference_seconds": round(time.perf_counter() - started, 4), **scored}
                with journal_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                records[key] = record
                print(f"[{len(records)}/{len(valid_keys)}] {qa_id} rollout={rollout_index + 1}/4 "
                      f"correct={int(record['correct'])}", flush=True)
            gc.collect()

    finally:
        groups = []
        for sample in samples:
            qa_id = str(sample["qa_id"])
            rollouts = [records[(qa_id, i)] for i in range(args.num_rollouts) if (qa_id, i) in records]
            groups.append({"qa_id": qa_id, "question": sample["question"],
                           "ground_truth": sample["ground_truth"],
                           "ground_truth_type": sample["ground_truth_type"],
                           "rollouts": rollouts, "answer_metrics": answer_statistics(rollouts)})
        atomic_write_json(details_path, groups)
        summary = summarize(groups, manifest)
        atomic_write_json(summary_path, summary)
        print(json.dumps({key: summary[key] for key in (
            "completed_questions", "rollout_accuracy", "pass_rate", "answer_consistency",
            "answer_entropy_bits")}, indent=2), flush=True)
        print(f"Saved: {summary_path}", flush=True)
        del model, processor
        if __import__("torch").cuda.is_available():
            __import__("torch").cuda.empty_cache()


if __name__ == "__main__":
    main()
