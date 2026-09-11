"""Isolated launch/configuration for the two RL experiments.

The byte-for-byte baseline snapshot is imported, never edited or run as a script.
Only its module-local Trainer/Config bindings change within this new process.
"""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import sys


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_experiment(mode):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--max-steps", type=int, default=220)
    parser.add_argument("--num-iterations", type=int, default=2 if mode == "iterations" else 1)
    parser.add_argument("--replay-capacity-per-qa", type=int, default=2)
    parser.add_argument("--replay-start-step", type=int, default=11)
    parser.add_argument("--replay-max-age", type=int, default=44)
    parser.add_argument("--replay-shaping-beta", type=float, default=0.1)
    if "--help" in sys.argv or "-h" in sys.argv:
        print("Additional isolated experiment options (max-steps overrides epochs):")
        parser.print_help()
    extra, remaining = parser.parse_known_args()
    if extra.max_steps < 1 or extra.num_iterations not in (1, 2):
        parser.error("max-steps must be positive; num-iterations must be 1 or 2")
    if mode == "replay" and extra.num_iterations != 1:
        parser.error("A fixes num-iterations=1 to isolate replay")
    if (extra.replay_capacity_per_qa < 1 or extra.replay_start_step < 0
            or extra.replay_max_age < 1 or not math.isfinite(extra.replay_shaping_beta)
            or extra.replay_shaping_beta <= 0):
        parser.error("Invalid replay settings")
    import _vislora_grpo_baseline_snapshot as base
    from _vislora_reuse_trainers import make_trainers
    sys.argv = [sys.argv[0], *remaining]
    args = base.parse_args()
    base.validate_args(args)
    if args.init_adapter_path is not None or args.resume_from_checkpoint is not None:
        parser.error("These controlled experiments start from the raw base; no init/resume")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or args.beta != 0:
        parser.error("This first implementation requires one GPU and beta=0")
    if not args.preserve_qa_order:
        parser.error("Use --preserve-qa-order and the same fixed QA file for A/B/control")
    output = Path(args.output_dir).expanduser().resolve()
    if (output / "run_config.json").exists() or list(output.glob("checkpoint-*")):
        parser.error("Use a fresh output directory; existing experiments must not be overwritten")
    version = importlib.metadata.version("trl")
    if version != "1.12.0":
        raise RuntimeError(f"This implementation was checked against TRL 1.12.0; found {version}")
    for key, value in vars(extra).items():
        setattr(args, key, value)
    args.reuse_experiment = mode
    Config = base.GRPOConfig
    native_trainer_file = inspect.getfile(base.GRPOTrainer)
    def configured(**kwargs):
        kwargs.update(max_steps=extra.max_steps, num_iterations=extra.num_iterations)
        cfg = Config(**kwargs)
        if cfg.steps_per_generation != cfg.gradient_accumulation_steps:
            raise ValueError("Expected one complete generation batch per accumulation window")
        (output / "effective_grpo_config.json").write_text(
            json.dumps(cfg.to_dict(), indent=2, default=str), encoding="utf-8")
        return cfg
    base.GRPOConfig = configured
    NativeReuse, Replay = make_trainers(base, args)
    base.GRPOTrainer = Replay if mode == "replay" else NativeReuse
    original_save = base.save_run_config
    def save_config(output_dir, args, dataset_size, **unused):
        original_save(output_dir, args, dataset_size, extra.max_steps,
                      math.ceil(extra.max_steps * args.warmup_fraction))
        path = output_dir / "run_config.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.update(step_counts_are_estimates=False, budget_unit="optimizer_updates",
                       max_steps_overrides_epochs=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        directory = Path(__file__).parent
        files = [directory / name for name in (
            "_vislora_grpo_baseline_snapshot.py", "_vislora_reuse_common.py",
            "_vislora_reuse_trainers.py", "_vislora_replay_buffer.py",
            "train_mapwise_grpo_visLoRA_replay_trl.py", "train_mapwise_grpo_visLoRA_iterations_trl.py")]
        manifest = dict(experiment=mode, arguments=payload,
                        source_sha256={p.name: sha256(p) for p in files},
                        native_trainer_sha256=sha256(native_trainer_file),
                        qa_sha256=sha256(args.qa_json), scorer_sha256=sha256(args.evaluation_script),
                        versions={p: importlib.metadata.version(p) for p in
                                  ("torch", "trl", "transformers", "peft", "accelerate")},
                        replay_method="ExGRPO-inspired fixed-QA mixed-group policy shaping",
                        replay_probability_convention="TRL temperature-scaled unfiltered model probabilities; not exact top-p sampling IS",
                        replay_selection="current-policy minimum mean completion token entropy; same fixed QA selection; max one old answer per group",
                        replay_entropy="TRL full-vocabulary Shannon entropy, temperature-scaled pre-top-p; completion only including EOS; single-candidate no-grad forwards",
                        generation_cost="A generates G fresh then discards one if replaying; no claimed generation speedup")
        (output_dir / "reuse_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"AUTHORITATIVE BUDGET: {extra.max_steps} optimizer updates; iterations={extra.num_iterations}.")
        print("The inherited preflight epoch estimate below is not the stopping budget.")
    base.save_run_config = save_config
    base.run_training(args)
    metrics_path = output / "training_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics.update(expected_optimizer_steps=extra.max_steps,
                   warmup_steps=math.ceil(extra.max_steps * args.warmup_fraction),
                   step_counts_are_estimates=False, budget_unit="optimizer_updates")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
