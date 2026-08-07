#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any

from unsloth.trainer import UnslothVisionDataCollator


def get_tokenizer(processor):
    return processor.tokenizer if hasattr(processor, "tokenizer") else processor


def find_subsequence(
    sequence: list[int],
    pattern: list[int],
    start: int = 0,
) -> int:
    if not pattern:
        return -1

    final_start = len(sequence) - len(pattern)
    for idx in range(start, final_start + 1):
        if sequence[idx:idx + len(pattern)] == pattern:
            return idx
    return -1


class PairedReasoningCollator:
    """C0/C1 loss masking on top of Unsloth's response-only VLM collator.

    C0 / answer_only:
        Full gold rationale remains in the teacher-forced token sequence, but
        every token from <think> through </think> is assigned label=-100.
        Therefore the rationale does not contribute supervised gradient.

    C1:
        Normal response-only supervision is retained for the entire assistant
        response: <think> rationale </think> + Final answer.

    User prompt and visual/prompt tokens are masked by
    UnslothVisionDataCollator(train_on_responses_only=True).
    """

    def __init__(self, model, processor):
        self.processor = processor
        self.tokenizer = get_tokenizer(processor)

        self.base = UnslothVisionDataCollator(
            model,
            processor,
            train_on_responses_only=True,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
            completion_only_loss=True,
        )

        self.think_start_ids = self.tokenizer.encode(
            "<think>",
            add_special_tokens=False,
        )
        self.think_end_ids = self.tokenizer.encode(
            "</think>",
            add_special_tokens=False,
        )

        if not self.think_start_ids or not self.think_end_ids:
            raise RuntimeError(
                "Tokenizer could not encode <think> / </think> tags."
            )

    def __call__(self, examples: list[dict[str, Any]]):
        modes = [
            str(example.get("_loss_mode", "c1"))
            for example in examples
        ]

        clean_examples = [
            {
                key: value
                for key, value in example.items()
                if key != "_loss_mode"
            }
            for example in examples
        ]

        batch = self.base(clean_examples)
        input_ids = batch["input_ids"]
        labels = batch["labels"]

        for row_idx, mode in enumerate(modes):
            if mode not in {"c0", "answer_only"}:
                continue

            ids = input_ids[row_idx].detach().cpu().tolist()
            cursor = 0
            found_any = False

            while True:
                start = find_subsequence(
                    ids,
                    self.think_start_ids,
                    cursor,
                )
                if start < 0:
                    break

                end_start = find_subsequence(
                    ids,
                    self.think_end_ids,
                    start + len(self.think_start_ids),
                )
                if end_start < 0:
                    raise RuntimeError(
                        "Found <think> but no matching </think>."
                    )

                end = end_start + len(self.think_end_ids)

                # Mask tags plus all rationale content. This intentionally
                # avoids training an immediate empty-think closure.
                labels[row_idx, start:end] = -100
                found_any = True
                cursor = end

            # C0 training examples must contain a verified CoT block.
            # Validation may be old answer-only data with no think block.
            if mode == "c0" and not found_any:
                raise RuntimeError(
                    "C0 training example contains no <think>...</think> span."
                )

        batch["labels"] = labels
        return batch
