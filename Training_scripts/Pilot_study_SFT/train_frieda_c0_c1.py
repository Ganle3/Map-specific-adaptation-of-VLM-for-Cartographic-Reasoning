#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from pathlib import Path

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True",
)

# Unsloth must be imported before Transformers / TRL.
from unsloth import FastVisionModel, is_bfloat16_supported

import torch
from trl import SFTConfig, SFTTrainer

from cot_collator import PairedReasoningCollator
from dataset_utils import prepare_datasets, summarize_dataset
from debug_utils import inspect_target_and_mask, print_trainable_summary


MODEL_NAME = "unsloth/Qwen3-VL-8B-Thinking-unsloth-bnb-4bit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--validation-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument(
        "--experiment",
        choices=("c0", "c1"),
        required=True,
        help=(
            "c0 = mask full <think>...</think> from LM loss; "
            "c1 = supervise full verified CoT + final answer."
        ),
    )

    parser.add_argument(
        "--validation-dataset",
        default="FRIEDA",
        help="Filter validation rows by dataset name.",
    )
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--epochs", type=float, default=8.0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
    )
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=3407)

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use 8 train + 8 validation rows and train for 3 update steps.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Checkpoint path, or 'latest'.",
    )

    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument(
        "--wandb-project",
        default="VLM-Cartographic-CoT-Pilot",
    )
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    parser.add_argument(
        "--wandb-log-model",
        choices=("false", "end", "checkpoint"),
        default="false",
    )

    parser.add_argument(
        "--skip-target-debug",
        action="store_true",
        help="Skip pre-training target/loss-mask inspection.",
    )
    parser.add_argument(
        "--skip-token-positions",
        action="store_true",
        help="Do not print every supervised token during target inspection.",
    )

    return parser.parse_args()


def build_sft_config(args: argparse.Namespace) -> SFTConfig:
    default_run_name = (
        f"P1-FRIEDA101-"
        f"{'masked-CoT' if args.experiment == 'c0' else 'CoT-supervised'}"
    )
    run_name = args.wandb_run_name or default_run_name

    kwargs = dict(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=0.05,
        logging_steps=args.logging_steps,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=args.seed,
        data_seed=args.seed,

        report_to=("none" if args.wandb_mode == "disabled" else "wandb"),
        run_name=run_name,

        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,

        # C0 and C1 use the same answer-only validation loss for checkpointing.
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,

        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},

        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        gradient_checkpointing=True,
        dataloader_num_workers=0,
    )

    config_params = inspect.signature(SFTConfig.__init__).parameters
    if "max_length" in config_params:
        kwargs["max_length"] = args.max_length
    elif "max_seq_length" in config_params:
        kwargs["max_seq_length"] = args.max_length
    else:
        raise RuntimeError(
            "Neither max_length nor max_seq_length exists in SFTConfig."
        )

    if args.smoke_test:
        kwargs.pop("num_train_epochs", None)
        kwargs["max_steps"] = 3
        kwargs["logging_steps"] = 1
        kwargs["eval_strategy"] = "steps"
        kwargs["eval_steps"] = 3
        kwargs["save_strategy"] = "steps"
        kwargs["save_steps"] = 3

    return SFTConfig(**kwargs)


def find_latest_checkpoint(output_dir: Path) -> str | None:
    checkpoints = []
    for path in output_dir.glob("checkpoint-*"):
        try:
            step = int(path.name.split("-")[-1])
        except ValueError:
            continue
        checkpoints.append((step, path))

    if not checkpoints:
        return None

    return str(max(checkpoints)[1])


def configure_wandb(args: argparse.Namespace) -> None:
    os.environ["WANDB_PROJECT"] = args.wandb_project
    os.environ["WANDB_MODE"] = args.wandb_mode
    os.environ["WANDB_LOG_MODEL"] = args.wandb_log_model

    if args.wandb_entity:
        os.environ["WANDB_ENTITY"] = args.wandb_entity
    else:
        os.environ.pop("WANDB_ENTITY", None)


def main() -> int:
    args = parse_args()

    args.train_file = args.train_file.expanduser().resolve()
    args.validation_file = args.validation_file.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.train_file.is_file():
        raise FileNotFoundError(args.train_file)
    if not args.validation_file.is_file():
        raise FileNotFoundError(args.validation_file)

    configure_wandb(args)

    train_dataset, eval_dataset = prepare_datasets(
        train_file=args.train_file,
        validation_file=args.validation_file,
        experiment=args.experiment,
        validation_dataset=args.validation_dataset,
        smoke_test=args.smoke_test,
    )

    print("\nDataset summary")
    print("----------------")
    print(f"Experiment:       {args.experiment.upper()}")
    print(f"Train total:      {len(train_dataset)}")
    print(f"Validation total: {len(eval_dataset)}")
    print(f"Validation set:   {args.validation_dataset}")

    train_summary = summarize_dataset(train_dataset)
    eval_summary = summarize_dataset(eval_dataset)

    with (args.output_dir / "run_config.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                **vars(args),
                "train_file": str(args.train_file),
                "validation_file": str(args.validation_file),
                "output_dir": str(args.output_dir),
                "train_summary": train_summary,
                "validation_summary": eval_summary,
                "experimental_note": (
                    "C0 and C1 use the same 101 full-CoT training examples. "
                    "The intended treatment difference is whether the "
                    "<think>...</think> tokens receive supervised loss. "
                    "Validation is answer-only for both."
                ),
            },
            file,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU was not detected.")

    props = torch.cuda.get_device_properties(0)
    print(f"\nGPU:  {props.name}")
    print(f"VRAM: {props.total_memory / 1024**3:.2f} GiB")

    print(f"\nLoading model: {args.model_name}")
    model, processor = FastVisionModel.from_pretrained(
        args.model_name,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )

    # P1: language-model attention LoRA only.
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=False,
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        lora_dropout=0,
        bias="none",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )

    print_trainable_summary(model)
    FastVisionModel.for_training(model)

    training_args = build_sft_config(args)
    data_collator = PairedReasoningCollator(
        model,
        processor,
    )

    if not args.skip_target_debug:
        inspect_target_and_mask(
            example=train_dataset[0],
            data_collator=data_collator,
            processor=processor,
            experiment=args.experiment,
            print_token_positions=not args.skip_token_positions,
        )

    trainer_kwargs = dict(
        model=model,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )

    trainer_params = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = processor
    else:
        trainer_kwargs["tokenizer"] = processor

    trainer = SFTTrainer(**trainer_kwargs)

    resume = args.resume_from_checkpoint
    if resume == "latest":
        resume = find_latest_checkpoint(args.output_dir)
        if resume is None:
            print("No checkpoint found; starting from scratch.")
    elif resume:
        resume = str(Path(resume).expanduser().resolve())

    print(f"\nStarting P1 {args.experiment.upper()} training...")
    train_result = trainer.train(
        resume_from_checkpoint=resume
    )

    # load_best_model_at_end=True means trainer.model now corresponds to the
    # checkpoint with the lowest common answer-only validation loss.
    best_adapter_dir = args.output_dir / "best_adapter"
    trainer.save_model(str(best_adapter_dir))
    processor.save_pretrained(str(best_adapter_dir))

    metrics = dict(train_result.metrics)
    metrics["experiment"] = args.experiment
    metrics["best_model_checkpoint"] = trainer.state.best_model_checkpoint
    metrics["best_metric"] = trainer.state.best_metric
    metrics["peak_cuda_memory_gib"] = (
        torch.cuda.max_memory_reserved() / 1024**3
    )

    with (args.output_dir / "training_summary.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    print("\nTraining complete.")
    print(f"Experiment:      {args.experiment.upper()}")
    print(f"Best checkpoint: {trainer.state.best_model_checkpoint}")
    print(f"Best eval loss:  {trainer.state.best_metric}")
    print(f"Saved adapter:   {best_adapter_dir}")
    print(
        f"Peak reserved VRAM: "
        f"{torch.cuda.max_memory_reserved() / 1024**3:.2f} GiB"
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except torch.cuda.OutOfMemoryError:
        print(
            "\nCUDA out of memory.\n"
            "First retry with --max-length 1536. "
            "Keep per-device batch size at 1.",
            file=sys.stderr,
        )
        raise
