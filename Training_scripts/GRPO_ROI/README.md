# Token-level Dynamic ROI GRPO

This directory is an isolated experiment: the Qwen3-VL backbone and any
reasoning LoRA are frozen; only `DynamicROIPolicy` is optimised.  Each action
contains nine `cx, cy, width, height, lambda` ROIs.  A single linear policy
head consumes the preceding frozen decoder hidden state and a 3x3 pooled
summary of preceding visual attention.  It samples a Normal latent action,
stores its exact latent-space log probability, then converts it to valid ROI
geometry.  The soft rectangle masks are combined by bounded noisy-OR and are
injected only at original global-image key positions through Qwen eager
attention.

The temporal convention is intentional: the prompt produces the first token
without intervention.  That completed state chooses the ROI applied while the
first generated token is consumed, which affects prediction of the next token.
Thus an action never uses a hidden state influenced by itself.

## Checks and commands

Run CPU-only unit checks from `VLM_adaptation/Training_scripts`:

```powershell
python -m pytest GRPO_ROI/test_roi_components.py -q
python -m GRPO_ROI.train_roi_grpo --model <qwen-model> --adapter <reasoning-lora> --dataset <questions.json> --image-root <MapVerse-images> --output roi_pilot --limit 4
python -m GRPO_ROI.run_visual_hard_smoke --adapter <reasoning-lora>
python -m GRPO_ROI.run_visual_hard_smoke --adapter <reasoning-lora> --questions 20 --rollouts 4 --output roi_exploration_diagnostic
python -m GRPO_ROI.train_roi_grpo --model <qwen-model> --adapter <reasoning-lora> --dataset <questions.json> --image-root <MapVerse-images> --output roi_pilot --resume roi_pilot/checkpoint-10.pt
python -m GRPO_ROI.evaluate_roi --model <qwen-model> --adapter <reasoning-lora> --dataset <questions.json> --image-root <MapVerse-images> --checkpoint roi_pilot/final.pt --mode learned_roi --output learned.jsonl
```

`baseline`, `fixed_grid`, `learned_roi`, and `random_roi` are evaluation
modes.  Training logs are JSONL; checkpoints include policy/optimizer/RNG.

The smoke entry is also the pretraining exploration diagnostic. Its summary
reports group reward/answer variation, sampled strength range, spatial ROI
diversity, non-zero bias coverage, and per-step visual-attention shift. Do not
start a full GRPO run when reward variation is near zero: first adjust the
three exploration scales or select a more informative pilot subset.

Qwen3-VL Thinking traces frequently require more than 256 new tokens before
they write `Final answer:`. Use 1024 for reward diagnostics; a 256-token run
is only suitable for checking that the hook and cache execute. The ROI rollout
does not force `</think>` or an answer prefix: it preserves native Thinking
generation and alters only global visual-key attention logits.

## Limitations

The Qwen bridge explicitly forces eager attention and fails fast if this
installed Transformers implementation does not expose its eager hook. It is
not compatible with FlashAttention/SDPA for this experiment. The per-token
manual cache path must be GPU-smoke-tested with the exact installed Qwen3-VL
version before a pilot: Qwen's cache and M-RoPE APIs have varied between
Transformers releases. No vision encoding is repeated after the prompt pass;
only cached decoder K/V is used in later steps.
