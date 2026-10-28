# Prototype cross-attention language GRPO

This directory is independent of the mid-reasoning crop experiments.
No visual segment is appended during chain-of-thought generation.

## Policy and computational graph

The policy is implemented in `prototype_cross_attention_policy.py`.
The default trainable parameter is only:

```python
policy.prototypes  # nn.Parameter [9, hidden_size]
```

`nn.MultiheadAttention` performs head-wise cross-attention with the prototypes
as Query and the frozen Qwen global merger embeddings as Key and Value. Its
Q/K/V/O projections are identity-initialized and frozen for the clean
prototype-only ablation.

```text
P [9,H] -> multi-head CrossAttention(Q=P, K=Zimg, V=Zimg) -> R [9,H]
  -> one-time Qwen prefill -> sampled language tokens
  -> exact final-answer reward -> within-question GRPO advantage
  -> clipped sampled-token log-probability loss -> gradient to P
```

The evaluator is the established `data.evaluate_answer`, including the
MapWise country-aware exact evaluator and the MapVerse exact evaluator.

## Placement ablations

- `run_overfit_after_image.ps1`: image -> R1..R9 -> question -> assistant.
- `run_overfit_after_question.ps1`: image -> question -> R1..R9 -> assistant.

The second placement allows the Qwen decoder states at the prototype positions
to attend causally to both the image and the question. The explicit prototype
cross-attention itself reads the image merger tokens in both experiments.

The Qwen backbone and the loaded GRPO adapter are frozen. Language generation
is sampled at temperature 1.0 because language tokens are the GRPO actions.
Evaluation can use greedy decoding separately.

## Explicit question-conditioned query experiment

`question_conditioned_policy.py` implements a third architecture:

```text
C = CrossAttention(P, frozen_question_hidden, frozen_question_hidden)
Q = LayerNorm(P + C)
R = CrossAttention(Q, global_image_merger, global_image_merger)
```

The frozen question hidden states come from Qwen's final decoder layer on the
original image+question prompt. `R` is then inserted after the question. Both
cross-attention projection sets are frozen, so only the nine prototypes are
optimized. Launch it with `run_overfit_question_conditioned.ps1`.
