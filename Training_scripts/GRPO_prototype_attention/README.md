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

## Only visual relay

`only_visual_relay_policy.py` is a separate minimal policy with two trainable
decoder embeddings by default. `only_visual_relay_prefill.py` inserts them in
the native Qwen causal sequence:

```text
<vision_start> image holders <vision_end> R1 R2 question assistant-prefix
```

The relays use `mm_token_type_ids=0`. They do not add image holders, a visual
grid, or DeepStack tensors. During prefill, R1 can attend to the complete image
span, R2 can attend to that span and R1, and all later question/reasoning tokens
can attend to both relay K/V states. Only `policy.prototypes [2,H]` is optimized.

Use `train_only_visual_relay.py`; it fixes `--policy-kind only_visual_relay`
and `--placement after_vision_end`. Run the fast structural regression with:

```powershell
python .\smoke_only_visual_relay.py
```

### Joint relay + existing LoRA adapter

`train_only_visual_relay_joint.py` enables every LoRA A/B tensor in the loaded
adapter and optimizes it together with the relay tokens. Rollouts may use the
detached frozen visual cache, but every current-policy GRPO recomputation runs
ViT and merger again with autograd. This is required because the existing
adapter includes visual blocks, merger/deepstack merger and language layers.

Checkpoints contain `checkpoint-N.pt` for relay/optimizer/scheduler state and a
matching `checkpoint-N/` PEFT adapter directory. Joint WandB metrics separately
report relay, visual adapter, merger adapter and language adapter gradients.

Run the four-question experiment with:

```powershell
.\run_overfit_only_visual_relay_joint_0245_0248.ps1
```

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
