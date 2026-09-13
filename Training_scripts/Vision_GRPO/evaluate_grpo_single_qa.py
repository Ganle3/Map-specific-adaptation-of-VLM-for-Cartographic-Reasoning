"""Single-QA checkpoint evaluation, adapted from the existing debug50 evaluator."""
import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def discover(run_dir, reference):
    targets = []
    if reference:
        targets.append(("reference_before_debug", 0, reference))
    checkpoints = sorted(
        (p for p in run_dir.iterdir()
         if p.is_dir() and re.fullmatch(r"checkpoint-\d+", p.name)),
        key=lambda p: int(p.name.split("-")[1]),
    )
    targets.extend((p.name, int(p.name.split("-")[1]), p) for p in checkpoints)
    final = run_dir / "final_adapter"
    if final.is_dir():
        state = run_dir / "trainer_state.json"
        step = read_json(state).get("global_step", "") if state.exists() else ""
        targets.append(("final_adapter", step, final))
    if not checkpoints and not final.is_dir():
        raise ValueError(f"No saved adapters found in {run_dir}")
    for _, _, path in targets:
        if not (path / "adapter_config.json").is_file() or not any(
            (path / name).is_file()
            for name in ("adapter_model.safetensors", "adapter_model.bin")
        ):
            raise ValueError(f"Incomplete adapter: {path}")
    return targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "run-dir", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--reference-adapter", type=Path)
    parser.add_argument("--partition-debug44", action="store_true",
                        help="Also score retained44/excluded6 using the sibling partition scorer.")
    parser.add_argument("--baseline-predictions", type=Path,
                        help="Reuse previously generated raw-base predictions as step zero.")
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--thinking", choices=("auto", "on", "off"), default="auto")
    args = parser.parse_args()
    if args.partition_debug44:
        parser.error("Single-QA evaluation does not support 44/6 partition scoring")
    repo = args.repo.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    reference = args.reference_adapter.expanduser().resolve() if args.reference_adapter else None
    qa = repo / "Datasets/Processed_Mapwise/Train_Val/mapwise_grpo_debug_single_src2051.json"
    image_root = repo / "Datasets/mapwise-dataset"
    scripts = repo / "Evaluation_scripts/GRPO_ablation"
    inference = scripts / "inference_mapwise_trl.py"
    scorer = scripts / "mapwise_evaluation_exact.py"
    for path in (qa, image_root, inference, scorer):
        if not path.exists():
            raise FileNotFoundError(path)
    expected = len(read_json(qa))
    if expected != 1:
        raise ValueError(f"Expected 1 QA, found {expected}")
    targets = discover(run_dir, reference)
    baseline_predictions = args.baseline_predictions.expanduser().resolve() if args.baseline_predictions else None
    if baseline_predictions:
        if not baseline_predictions.is_file():
            raise FileNotFoundError(baseline_predictions)
        targets.insert(0, ("baseline", 0, None))
    partition_scorer = Path(__file__).with_name("score_debug44_partitions.py")
    if args.partition_debug44 and not partition_scorer.is_file():
        raise FileNotFoundError(partition_scorer)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_dir": str(run_dir), "qa_json": str(qa),
        "qa_sha256": hashlib.sha256(qa.read_bytes()).hexdigest(),
        "max_new_tokens": args.max_new_tokens, "thinking": args.thinking,
        "do_sample": False,
        "targets": [{"label": label, "debug_step": step, "adapter": str(path) if path else None}
                    for label, step, path in targets],
    }
    if args.partition_debug44:
        partition_data = repo / "Datasets/Processed_Mapwise/Train_Val"
        manifest["partitions"] = {name: hashlib.sha256((partition_data / name).read_bytes()).hexdigest()
                                  for name in ("mapwise_grpo_debug_filtered_44.json", "mapwise_grpo_debug_excluded_6.json")}
    if baseline_predictions:
        manifest["baseline_predictions"] = str(baseline_predictions)
        manifest["baseline_predictions_sha256"] = hashlib.sha256(baseline_predictions.read_bytes()).hexdigest()
    manifest_path = output / "evaluation_manifest.json"
    if manifest_path.exists() and read_json(manifest_path) != manifest:
        raise ValueError("Evaluation settings changed; use a new output directory.")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    rows = []
    for label, step, adapter in targets:
        destination = output / label
        destination.mkdir(exist_ok=True)
        predictions = destination / "predictions.json"
        row = {"checkpoint": label, "debug_step": step, "adapter_path": str(adapter) if adapter else "",
               "status": "failed", "total": "", "correct": "", "accuracy_percent": "",
               "mean_generated_tokens": "", "truncated": "", "inference_errors": "",
               "error": ""}
        if args.partition_debug44:
            row.update(retained44_correct="", retained44_accuracy_percent="",
                       excluded6_correct="", excluded6_accuracy_percent="")
        print(f"\nEvaluating {label}: {adapter}", flush=True)
        try:
            if adapter is None:
                records = read_json(baseline_predictions)
                if any(r.get("adapter_path") not in (None, "", "baseline") for r in records):
                    raise ValueError("Baseline predictions contain an adapter; expected raw base model.")
                predictions.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
            else:
                with (destination / "inference.log").open("a", encoding="utf-8") as log:
                    subprocess.run([
                        sys.executable, "-u", str(inference),
                        "--adapter-path", str(adapter), "--qa-json", str(qa),
                        "--image-root", str(image_root), "--output-json", str(predictions),
                        "--max-new-tokens", str(args.max_new_tokens),
                        "--thinking", args.thinking,
                    ], stdout=log, stderr=subprocess.STDOUT, check=True)
            records = read_json(predictions)
            expected_records = {(r['country'], r['source_index']): r for r in read_json(qa)}
            seen = set()
            for r in records:
                key = (r['country'], r['source_index'])
                if key in seen or key not in expected_records:
                    raise ValueError("Duplicate or unknown QA in predictions.")
                seen.add(key)
                if any(r.get(f) != expected_records[key].get(f) for f in
                       ('question', 'ground_truth', 'map_no', 'template_no', 'ground_truth_type')):
                    raise ValueError("Prediction QA metadata does not match evaluation dataset.")
                if r.get('max_new_tokens') != args.max_new_tokens or r.get('thinking_mode') != args.thinking:
                    raise ValueError("Prediction decoding settings differ from requested evaluation.")
            errors = sum(r.get("generation_status") == "error" for r in records)
            row.update(total=len(records), inference_errors=errors)
            if len(records) != expected or errors:
                raise ValueError(f"Expected {expected} predictions; found {len(records)}, errors={errors}")
            with (destination / "scoring.log").open("w", encoding="utf-8") as log:
                subprocess.run([
                    sys.executable, str(scorer), "--predictions-json", str(predictions),
                    "--output-dir", str(destination / "scores"),
                ], stdout=log, stderr=subprocess.STDOUT, check=True)
            summary = read_json(destination / "scores/evaluation_summary.json")
            if summary["total"] != expected:
                raise ValueError(f"Scorer evaluated {summary['total']} rather than {expected} QAs")
            row.update(
                status="ok", correct=summary["correct"],
                accuracy_percent=round(100 * summary["validation_accuracy"], 4),
                mean_generated_tokens=round(sum(r.get("generated_tokens", 0) for r in records) / expected, 2),
                truncated=sum(r.get("generation_status") == "truncated" for r in records),
            )
            if args.partition_debug44:
                with (destination / "partition_scoring.log").open("w", encoding="utf-8") as log:
                    subprocess.run([sys.executable, str(partition_scorer), "--repo", str(repo),
                                    "--predictions-json", str(predictions), "--output-dir", str(destination / "partitions")],
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
                partitions = read_json(destination / "partitions/partition_accuracy.json")
                for part in ('retained44', 'excluded6'):
                    row[part + '_correct'] = partitions[part]['correct']
                    row[part + '_accuracy_percent'] = round(100 * partitions[part]['validation_accuracy'], 4)
        except Exception as error:
            row['status'] = 'failed'
            row["error"] = str(error)
            print(f"FAILED: {error}; inspect {destination}", flush=True)
        rows.append(row)
        with (output / "checkpoint_accuracy.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
        (output / "checkpoint_accuracy.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"{label}: {row['correct']}/{row['total']} ({row['accuracy_percent']}%), status={row['status']}", flush=True)
        if args.partition_debug44 and row['status'] == 'ok':
            print(f"  training44: {row['retained44_correct']}/44 ({row['retained44_accuracy_percent']}%); "
                  f"excluded6: {row['excluded6_correct']}/6 ({row['excluded6_accuracy_percent']}%)", flush=True)
    print(f"\nSummary: {output / 'checkpoint_accuracy.csv'}", flush=True)
    return int(any(row["status"] != "ok" for row in rows))


if __name__ == "__main__":
    sys.exit(main())
