#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from collections import Counter
from typing import Any

from datasets import load_dataset


COMMON_PROMPT_SUFFIX = (
    "\n\nAnswer the map question using the provided image or images. "
    "Reason carefully using only evidence observable in the map, and avoid "
    "repeating earlier observations. At the end, provide the answer on a "
    "separate line using exactly this format:\n"
    "Final answer: <answer>"
)

# Suffixes used by earlier pilot scripts. We strip these before appending the
# common C0/C1 prompt so train and validation instructions are identical.
_OLD_PROMPT_MARKERS = (
    "\n\nAnswer the map question using the provided image or images.",
    "\n\nThis is a cartographic reasoning question from the FRIEDA dataset.",
)


def load_json_dataset(path):
    """Load one JSON/JSONL file independently.

    Train and validation are intentionally loaded separately so Hugging Face
    Datasets does not require both files to have identical source columns.
    """
    return load_dataset("json", data_files=str(path), split="train")


def _find_user_message(messages: list[dict[str, Any]]) -> dict[str, Any]:
    for message in messages:
        if message.get("role") == "user":
            return message
    raise ValueError("No user message was found.")


def _find_assistant_message(messages: list[dict[str, Any]]) -> dict[str, Any]:
    for message in messages:
        if message.get("role") == "assistant":
            return message
    raise ValueError("No assistant message was found.")


def _text_parts(message: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        part
        for part in message.get("content", [])
        if part.get("type") == "text"
    ]


def assistant_text(example: dict[str, Any]) -> str:
    message = _find_assistant_message(example["messages"])
    return "\n".join(
        str(part.get("text", ""))
        for part in _text_parts(message)
    ).strip()


def user_text(example: dict[str, Any]) -> str:
    message = _find_user_message(example["messages"])
    parts = _text_parts(message)
    if not parts:
        raise ValueError(
            f"No user text in sample {example.get('sample_id')}"
        )
    return str(parts[-1].get("text", ""))


def normalize_user_prompt(example: dict[str, Any]) -> dict[str, Any]:
    """Replace earlier suffixes with one common reasoning-oriented prompt."""
    messages = example["messages"]
    user_message = _find_user_message(messages)
    parts = _text_parts(user_message)
    if not parts:
        raise ValueError(
            f"No user text in sample {example.get('sample_id')}"
        )

    text = str(parts[-1].get("text", "")).rstrip()

    # Strip any older appended instruction while preserving the original QA.
    cut_positions = [
        text.find(marker)
        for marker in _OLD_PROMPT_MARKERS
        if marker in text
    ]
    if cut_positions:
        text = text[:min(cut_positions)].rstrip()

    parts[-1]["text"] = text + COMMON_PROMPT_SUFFIX
    return example


def _count_image_items(example: dict[str, Any]) -> int:
    return sum(
        part.get("type") == "image"
        for message in example["messages"]
        for part in message.get("content", [])
    )


def validate_train_dataset(dataset) -> None:
    required = {"messages", "sample_id", "dataset", "split", "answer"}
    missing = required - set(dataset.column_names)
    if missing:
        raise ValueError(
            f"Training file is missing required columns: {sorted(missing)}"
        )

    if len(dataset) == 0:
        raise ValueError("Training dataset is empty.")

    seen = set()
    for idx in range(len(dataset)):
        row = dataset[idx]
        sample_id = row.get("sample_id")

        if sample_id in seen:
            raise ValueError(f"Duplicate sample_id: {sample_id}")
        seen.add(sample_id)

        if row.get("dataset") != "FRIEDA":
            raise ValueError(
                f"Training sample {sample_id} is not FRIEDA."
            )

        if _count_image_items(row) < 1:
            raise ValueError(f"{sample_id} has no image item.")

        target = assistant_text(row)
        if "<think>" not in target or "</think>" not in target:
            raise ValueError(
                f"{sample_id} assistant target has no <think>...</think> block."
            )
        if "Final answer:" not in target:
            raise ValueError(
                f"{sample_id} assistant target has no Final answer marker."
            )

        start = target.find("<think>") + len("<think>")
        end = target.find("</think>", start)
        reasoning = target[start:end].strip()
        if not reasoning:
            raise ValueError(
                f"{sample_id} has an empty reasoning block; "
                "C0/C1 requires verified non-empty CoT."
            )


def validate_eval_dataset(dataset) -> None:
    required = {"messages", "sample_id", "dataset", "split", "answer"}
    missing = required - set(dataset.column_names)
    if missing:
        raise ValueError(
            f"Validation file is missing required columns: {sorted(missing)}"
        )

    for idx in range(len(dataset)):
        row = dataset[idx]
        sample_id = row.get("sample_id")
        if _count_image_items(row) < 1:
            raise ValueError(f"{sample_id} has no image item.")

        target = assistant_text(row)
        if "Final answer:" not in target:
            raise ValueError(
                f"{sample_id} validation target has no Final answer marker."
            )


def prepare_datasets(
    train_file,
    validation_file,
    experiment: str,
    validation_dataset: str = "FRIEDA",
    smoke_test: bool = False,
):
    """Load, validate, normalize prompts, filter FRIEDA validation, set loss mode."""
    train_dataset = load_json_dataset(train_file)
    eval_dataset = load_json_dataset(validation_file)

    validate_train_dataset(train_dataset)
    validate_eval_dataset(eval_dataset)

    train_dataset = train_dataset.map(normalize_user_prompt)
    eval_dataset = eval_dataset.map(normalize_user_prompt)

    if validation_dataset:
        eval_dataset = eval_dataset.filter(
            lambda x: x["dataset"] == validation_dataset
        )

    if len(eval_dataset) == 0:
        raise ValueError(
            f"No validation rows remained after filtering "
            f"dataset={validation_dataset!r}."
        )

    # C0/C1 difference is only the training loss mode.
    train_dataset = train_dataset.map(
        lambda _: {"_loss_mode": experiment}
    )

    # Checkpoint selection is answer-only in both experiments.
    eval_dataset = eval_dataset.map(
        lambda _: {"_loss_mode": "answer_only"}
    )

    if smoke_test:
        train_dataset = train_dataset.select(
            range(min(8, len(train_dataset)))
        )
        eval_dataset = eval_dataset.select(
            range(min(8, len(eval_dataset)))
        )

    return train_dataset, eval_dataset


def summarize_dataset(dataset) -> dict[str, Any]:
    task_counts = Counter(
        row.get("task_category") for row in dataset
    )
    answer_counts = Counter(
        row.get("answer_type") for row in dataset
    )
    return {
        "size": len(dataset),
        "dataset_counts": dict(Counter(row.get("dataset") for row in dataset)),
        "task_category_counts": dict(task_counts),
        "answer_type_counts": dict(answer_counts),
    }
