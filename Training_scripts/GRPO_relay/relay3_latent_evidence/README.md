# Relay3 latent-state evidence, step 4000

## Dataset alignment and selection

Run `select_questions.py` to join the saved adapter-only and Relay3 greedy
evaluations by **one-based `dataset_index`**. The script compares the question,
ground truth, and `qa_id` to the corresponding zero-based source dataset row.
Both responses must have terminated. It selects 10 distinct maps per dataset
and outcome category, for 60 questions total. The JSON stores both raw
responses and both identifiers.

Categories: Relay3 correct/adapter wrong, both correct, both wrong. These are
selected *after observing the outcomes*, so their visual statistics describe
the selected cases; they are not an unbiased estimate of test-set accuracy.

## Fixed-prefix capture

`capture_trajectories.py` uses the **saved Relay3 response** as a common
teacher-forced continuation for both step-4000 checkpoints. It captures
decoder-layer output states at reference token steps 0, 4, 9, 19, 39 (when
available). It repeats each pass with merger and DeepStack image embeddings
replaced by zeros while retaining the text, image-token positions, and RoPE.
The ViT is only run once per question and variant. Relay3 temporary relay K/V
are removed after each token, as in its normal inference path.

The principal layer/step metric is

`1 - cosine(hidden_real_image, hidden_zero_image)`.

This measures sensitivity to image features at a fixed text prefix; a larger
value is **not** automatically better. For each selected step, the code also
records `log p(next reference token | real image) - log p(next reference token |
zero image)`. Positive values indicate that the image supports that particular
token. Many early tokens are generic wording, so this does not prove correct
visual understanding. Zeroed image features are an ablation outside the model's
normal image distribution.

`render_evidence.py` produces per-question layer/step heatmaps, exploratory
shared-PCA language trajectories, a per-layer paired CSV, and a six-panel
dataset/category summary. PCA distance to image embeddings is intentionally
**not** interpreted as cross-modal alignment. The two variants share the same
token prefix, but they have separately trained adapters and thus different
representation functions. To isolate the relay mechanism from adapter training,
a same-checkpoint relay-disabled ablation would be needed.
The capture script supports this with
`--variants relay3_adapter_no_relay`: it reloads the **same step-4000 Relay3
adapter** and processes the same tokens without the temporary relay positions.
The resulting paired metrics appear in additional columns of
`paired_metrics.csv`. This isolates the inference-time effect of the relay
positions while preserving the trained adapter weights.

## Commands (local Windows VisionGRPO environment)

```powershell
& "C:\Users\junyhuang\.conda\envs\VisionGRPO\python.exe" `
  ".\VLM_adaptation\Training_scripts\GRPO_relay\relay3_latent_evidence\select_questions.py"

& "C:\Users\junyhuang\.conda\envs\VisionGRPO\python.exe" `
  ".\VLM_adaptation\Training_scripts\GRPO_relay\relay3_latent_evidence\capture_trajectories.py" `
  --max-reference-tokens 40 --capture-steps 0 4 9 19 39

& "C:\Users\junyhuang\.conda\envs\VisionGRPO\python.exe" `
  ".\VLM_adaptation\Training_scripts\GRPO_relay\relay3_latent_evidence\render_evidence.py"
```

Use `--limit 1` for a quick pilot. `capture_trajectories.py` requires CUDA.
