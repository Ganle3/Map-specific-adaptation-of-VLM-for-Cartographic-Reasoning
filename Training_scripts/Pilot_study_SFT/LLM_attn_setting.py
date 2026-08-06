#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from pathlib import Path

# Helps reduce CUDA memory fragmentation on long multimodal batches.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Import Unsloth before Transformers/TRL.
from unsloth import FastVisionModel, is_bfloat16_supported
from unsloth.trainer import UnslothVisionDataCollator

import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer


MODEL_NAME = "unsloth/Qwen3-VL-8B-Thinking-unsloth-bnb-4bit"
PROMPT_SUFFIX = (
    "\n\nAnswer the map question using the provided image or images. "
    "Reason about the map internally, then provide the final answer in this exact format:\n"
    "Final answer: <answer>"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use 16 train + 8 validation samples and run only 5 update steps.",
    )
    parser.add_argument(
        "--no-prompt-suffix",
        action="store_true",
        help="Do not append the fixed answer-format instruction to user prompts.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Checkpoint directory, or 'latest' to resume from the latest checkpoint.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=5,
        help="Log training metrics to the console and W&B every N update steps.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="VLM-Cartographic-SFT",
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default="P1-LLM-attn-answer-only",
        help="Weights & Biases run name.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Optional W&B username or team.",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
        help="W&B mode. Use disabled to turn W&B off.",
    )
    parser.add_argument(
        "--wandb-log-model",
        choices=("false", "end", "checkpoint"),
        default="false",
        help="Whether W&B uploads model artifacts. 'false' is recommended.",
    )
    parser.add_argument(
        "--skip-mask-debug",
        action="store_true",
        help="Skip printing the full supervised-token mask before training.",
    )
    return parser.parse_args()


def append_prompt_instruction(example: dict) -> dict:
    """Append the same answer-format instruction used for later inference."""
    messages = example["messages"]
    user_message = next((m for m in messages if m.get("role") == "user"), None)
    if user_message is None:
        raise ValueError(f"No user message in sample {example.get('sample_id')}")

    text_parts = [
        part for part in user_message.get("content", [])
        if part.get("type") == "text"
    ]
    if not text_parts:
        raise ValueError(f"No user text in sample {example.get('sample_id')}")

    text = text_parts[-1].get("text", "")
    if "Reason about the map internally, then provide the final answer in this exact format" not in text:
        text_parts[-1]["text"] = text.rstrip() + PROMPT_SUFFIX
    return example


def validate_dataset(dataset, name: str) -> None:
    required = {"messages", "sample_id", "dataset", "split"}
    missing_columns = required - set(dataset.column_names)
    if missing_columns:
        raise ValueError(f"{name} is missing columns: {sorted(missing_columns)}")

    for idx in range(min(5, len(dataset))):
        row = dataset[idx]
        if not row["messages"]:
            raise ValueError(f"{name}[{idx}] has empty messages")
        image_count = sum(
            part.get("type") == "image"
            for message in row["messages"]
            for part in message.get("content", [])
        )
        if image_count < 1:
            raise ValueError(f"{name}[{idx}] has no image item")


def print_dataset_summary(train_dataset, eval_dataset) -> None:
    def counts(ds):
        result = {}
        for source in sorted(set(ds["dataset"])):
            result[source] = sum(x == source for x in ds["dataset"])
        return result

    print("\nDataset summary")
    print("----------------")
    print(f"Train total:      {len(train_dataset)}  {counts(train_dataset)}")
    print(f"Validation total: {len(eval_dataset)}  {counts(eval_dataset)}")


def print_trainable_summary(model) -> None:
    trainable = 0
    total = 0
    trainable_names = []

    for name, parameter in model.named_parameters():
        total += parameter.numel()
        if parameter.requires_grad:
            trainable += parameter.numel()
            trainable_names.append(name)

    ratio = 100.0 * trainable / total
    print("\nTrainable parameter summary")
    print("---------------------------")
    print(f"Trainable: {trainable:,}")
    print(f"Total:     {total:,}")
    print(f"Ratio:     {ratio:.4f}%")
    print("First trainable parameter names:")
    for name in trainable_names[:20]:
        print(f"  {name}")

    suspicious = [
        name for name in trainable_names
        if any(token in name.lower() for token in ("visual", "vision", "merger"))
    ]
    if suspicious:
        print("\nWARNING: names suggesting visual/vision modules are trainable:")
        for name in suspicious[:20]:
            print(f"  {name}")
        print("Inspect these names before treating this run as a clean P1 ablation.")


def build_sft_config(args: argparse.Namespace) -> SFTConfig:
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
        run_name=args.wandb_run_name,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        gradient_checkpointing=True,
        dataloader_num_workers=0,  # safer on Windows
    )

    # Current Unsloth notebook uses max_length. This fallback keeps the script
    # usable with older TRL/Unsloth combinations that exposed max_seq_length.
    config_params = inspect.signature(SFTConfig.__init__).parameters
    if "max_length" in config_params:
        kwargs["max_length"] = args.max_length
    elif "max_seq_length" in config_params:
        kwargs["max_seq_length"] = args.max_length
    else:
        raise RuntimeError("Neither max_length nor max_seq_length exists in SFTConfig.")

    if args.smoke_test:
        kwargs.pop("num_train_epochs", None)
        kwargs["max_steps"] = 5
        kwargs["logging_steps"] = 1
        kwargs["eval_strategy"] = "steps"
        kwargs["eval_steps"] = 5
        kwargs["save_strategy"] = "steps"
        kwargs["save_steps"] = 5

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


def main() -> int:
    args = parse_args()
    args.data_dir = args.data_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Configure W&B before SFTTrainer is created.
    os.environ["WANDB_PROJECT"] = args.wandb_project
    os.environ["WANDB_MODE"] = args.wandb_mode
    os.environ["WANDB_LOG_MODEL"] = args.wandb_log_model
    if args.wandb_entity:
        os.environ["WANDB_ENTITY"] = args.wandb_entity
    else:
        os.environ.pop("WANDB_ENTITY", None)

    train_file = args.data_dir / "sft_train.jsonl"
    validation_file = args.data_dir / "sft_validation.jsonl"
    if not train_file.is_file() or not validation_file.is_file():
        raise FileNotFoundError(
            f"Expected:\n  {train_file}\n  {validation_file}"
        )

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(train_file),
            "validation": str(validation_file),
        },
    )
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]

    validate_dataset(train_dataset, "train")
    validate_dataset(eval_dataset, "validation")

    if not args.no_prompt_suffix:
        train_dataset = train_dataset.map(append_prompt_instruction)
        eval_dataset = eval_dataset.map(append_prompt_instruction)

    if args.smoke_test:
        train_dataset = train_dataset.select(range(min(16, len(train_dataset))))
        eval_dataset = eval_dataset.select(range(min(8, len(eval_dataset))))

    print_dataset_summary(train_dataset, eval_dataset)

    with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                **vars(args),
                "data_dir": str(args.data_dir),
                "output_dir": str(args.output_dir),
                "train_size": len(train_dataset),
                "validation_size": len(eval_dataset),
                "model_name": args.model_name,
            },
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"\nGPU: {props.name}")
        print(f"VRAM: {props.total_memory / 1024**3:.2f} GiB")
    else:
        raise RuntimeError("CUDA GPU was not detected.")

    print(f"\nLoading model: {args.model_name}")
    model, processor = FastVisionModel.from_pretrained(
        args.model_name,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )

    # P1: LLM attention only.
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,
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

    data_collator = UnslothVisionDataCollator(
        model,
        processor,
        train_on_responses_only=True,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
        completion_only_loss=True,
    )

    trainer_kwargs = dict(
        model=model,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )

    # Unsloth's pinned notebook uses tokenizer=; newer TRL uses processing_class=.
    trainer_params = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = processor
    else:
        trainer_kwargs["tokenizer"] = processor

    trainer = SFTTrainer(**trainer_kwargs)

    if not args.skip_mask_debug:
        # Inspect the exact batch produced by Trainer.
        debug_batch = next(iter(trainer.get_train_dataloader()))

        input_ids = debug_batch["input_ids"][0].detach().cpu()
        labels = debug_batch["labels"][0].detach().cpu()

        tokenizer = (
            processor.tokenizer
            if hasattr(processor, "tokenizer")
            else processor
        )

        supervised_mask = labels != -100
        supervised_ids = labels[supervised_mask].tolist()

        print("\n========== REAL TRAIN BATCH MASK ==========")
        print(f"Total tokens:      {labels.numel()}")
        print(f"Supervised tokens: {supervised_mask.sum().item()}")
        print(f"Masked tokens:     {(~supervised_mask).sum().item()}")
        print(
            f"Supervised ratio:  "
            f"{100 * supervised_mask.sum().item() / labels.numel():.2f}%"
        )

        print("\nText participating in loss:")
        print(
            tokenizer.decode(
                supervised_ids,
                skip_special_tokens=False,
            )
        )

        print("\nPer-token supervised positions:")
        for position, (input_id, label_id) in enumerate(
            zip(input_ids.tolist(), labels.tolist())
        ):
            if label_id != -100:
                token_text = tokenizer.decode(
                    [input_id],
                    skip_special_tokens=False,
                )
                token_text = token_text.replace("\n", "\\n")
                print(
                    f"[LOSS] {position:04d} | "
                    f"input={input_id} | label={label_id} | "
                    f"{token_text!r}"
                )

    print("===========================================\n")

    resume = args.resume_from_checkpoint
    if resume == "latest":
        resume = find_latest_checkpoint(args.output_dir)
        if resume is None:
            print("No checkpoint found; starting from scratch.")
    elif resume:
        resume = str(Path(resume).resolve())

    print("\nStarting P1 training...")
    train_result = trainer.train(resume_from_checkpoint=resume)

    final_adapter_dir = args.output_dir / "best_adapter"
    trainer.save_model(str(final_adapter_dir))
    processor.save_pretrained(str(final_adapter_dir))

    metrics = dict(train_result.metrics)
    metrics["best_model_checkpoint"] = trainer.state.best_model_checkpoint
    metrics["best_metric"] = trainer.state.best_metric
    metrics["peak_cuda_memory_gib"] = (
        torch.cuda.max_memory_reserved() / 1024**3
        if torch.cuda.is_available()
        else None
    )
    with (args.output_dir / "training_summary.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\nTraining complete.")
    print(f"Best checkpoint: {trainer.state.best_model_checkpoint}")
    print(f"Best eval loss:  {trainer.state.best_metric}")
    print(f"Saved adapter:   {final_adapter_dir}")
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
            "First retry with --max-length 1536. If it still fails, reduce the "
            "image pixel budget during data preprocessing. Keep batch size at 1.",
            file=sys.stderr,
        )
        raise