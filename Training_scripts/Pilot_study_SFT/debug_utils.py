#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import torch

from dataset_utils import assistant_text
from cot_collator import get_tokenizer


def _supervised_text(labels: torch.Tensor, tokenizer) -> str:
    ids = labels[labels != -100].detach().cpu().tolist()
    return tokenizer.decode(ids, skip_special_tokens=False)


def inspect_target_and_mask(
    example,
    data_collator,
    processor,
    experiment: str,
    print_token_positions: bool = True,
) -> None:
    """Fail-fast inspection of the exact sequence and labels used by training."""
    tokenizer = get_tokenizer(processor)

    print("\n" + "=" * 88)
    print("PRE-TRAINING TARGET / LOSS-MASK CHECK")
    print("=" * 88)
    print(f"Experiment: {experiment.upper()}")
    print(f"Sample:     {example.get('sample_id')}")

    raw_target = assistant_text(example)

    print("\n[1] Assistant target stored in the training example:")
    print(raw_target)

    batch = data_collator([example])
    input_ids = batch["input_ids"][0].detach().cpu()
    labels = batch["labels"][0].detach().cpu()

    full_text = tokenizer.decode(
        input_ids.tolist(),
        skip_special_tokens=False,
    )
    supervised_text = _supervised_text(labels, tokenizer)

    print("\n[2] Full decoded model sequence:")
    print(full_text)

    print("\n[3] Text that ACTUALLY participates in loss:")
    print(supervised_text)

    total = labels.numel()
    supervised = int((labels != -100).sum().item())

    print("\n[4] Mask statistics:")
    print(f"Total tokens:      {total}")
    print(f"Supervised tokens: {supervised}")
    print(f"Masked tokens:     {total - supervised}")
    print(
        f"Supervised ratio:  "
        f"{100.0 * supervised / max(total, 1):.2f}%"
    )

    if "Final answer:" not in supervised_text:
        raise RuntimeError(
            "DEBUG CHECK FAILED: Final answer is not supervised."
        )

    if experiment == "c0":
        if "<think>" in supervised_text or "</think>" in supervised_text:
            raise RuntimeError(
                "DEBUG CHECK FAILED: C0 still supervises think-tag tokens."
            )

        raw_reasoning = raw_target.split("<think>", 1)[1].split(
            "</think>", 1
        )[0].strip()
        # Token decoding can alter whitespace, so use a short normalized probe.
        probe_words = raw_reasoning.split()[:6]
        if probe_words:
            probe = " ".join(probe_words)
            if probe in supervised_text:
                raise RuntimeError(
                    "DEBUG CHECK FAILED: C0 still supervises reasoning text."
                )

        print(
            "\nC0 CHECK PASSED: gold reasoning is present in the sequence, "
            "but the entire <think>...</think> span is masked from loss. "
            "Final-answer tokens remain supervised."
        )

    elif experiment == "c1":
        if "<think>" not in supervised_text or "</think>" not in supervised_text:
            raise RuntimeError(
                "DEBUG CHECK FAILED: C1 does not supervise the think block."
            )

        print(
            "\nC1 CHECK PASSED: verified reasoning AND final-answer tokens "
            "both participate in supervised loss."
        )

    if print_token_positions:
        print("\n[5] Per-token supervised positions:")
        for pos, (input_id, label_id) in enumerate(
            zip(input_ids.tolist(), labels.tolist())
        ):
            if label_id == -100:
                continue

            token_text = tokenizer.decode(
                [input_id],
                skip_special_tokens=False,
            ).replace("\n", "\\n")

            print(
                f"[LOSS] {pos:04d} | "
                f"input={input_id} | label={label_id} | "
                f"{token_text!r}"
            )

    print("=" * 88 + "\n")


def print_trainable_summary(model) -> None:
    trainable = 0
    total = 0
    names = []

    for name, parameter in model.named_parameters():
        total += parameter.numel()
        if parameter.requires_grad:
            trainable += parameter.numel()
            names.append(name)

    print("\nTrainable parameter summary")
    print("---------------------------")
    print(f"Trainable: {trainable:,}")
    print(f"Total:     {total:,}")
    print(f"Ratio:     {100.0 * trainable / total:.4f}%")
    print("First trainable parameter names:")
    for name in names[:20]:
        print(f"  {name}")

    suspicious = [
        name
        for name in names
        if any(
            token in name.lower()
            for token in ("visual", "vision", "merger")
        )
    ]
    if suspicious:
        print("\nWARNING: possible vision/merger parameters are trainable:")
        for name in suspicious[:20]:
            print(f"  {name}")
