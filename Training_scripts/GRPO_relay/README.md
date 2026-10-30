# Qwen3-VL relay experiments

This directory contains the relay-token experiments only. It does not import
relay implementations from `GRPO_prototype_attention` or `GRPO_lookbetter`.
The existing directories remain untouched as historical experiment copies.

## Layout

- `utils.py`: shared Qwen3-VL visual encoding, relay insertion, explicit
  generation, differentiable current-policy forward, and language GRPO loss.
- `train_core.py`: common training loop used by both relay placements.
- `evaluate_core.py`: common MapVerse/MapWise evaluator.
- `smoke_gradient_core.py`: real-model relay/LoRA gradient audit.
- `only_visual_relay/`: scheme 1.
- `Image_question_shared_relay/`: scheme 2A.

Both training entry points jointly train the loaded LoRA adapter and relay
parameters while leaving the quantized Qwen base weights frozen.

## Sequence layouts

Scheme 1:

```text
image <vision_end> P1 P2 question assistant
```

Scheme 2A:

```text
image <vision_end> P1 P2 question P1 P2 assistant
```

In scheme 2A both spans use the exact same `[2,H]` `nn.Parameter`. The second
span can causally read the image, first relay states, and question. It cannot
write information backward into the first span.

## Entry points

```powershell
# Scheme 1 four-question overfit
.\VLM_adaptation\Training_scripts\GRPO_relay\only_visual_relay\run_overfit_0245_0248.ps1

# Scheme 2A four-question overfit
.\VLM_adaptation\Training_scripts\GRPO_relay\Image_question_shared_relay\run_overfit_0245_0248.ps1
```

Use each directory's `smoke_gradient.py` before a cluster run. A passing audit
prints the relay positions, a nonzero shared relay gradient, nonzero visual,
merger and language LoRA gradients, and zero gradients on non-LoRA base
parameters.
