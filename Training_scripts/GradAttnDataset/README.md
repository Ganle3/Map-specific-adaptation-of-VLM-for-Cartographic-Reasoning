# Grad-Attention teacher-data collector

`collect_gradattn_dataset.py` runs the frozen NF4 Qwen3-VL + existing LoRA
adapter once per question.  It does **not** alter the generation with a crop.
It creates supervision for a later teacher:

```
global image + question -> ordinary greedy Qwen reasoning
                         -> token t hidden state
                         -> Grad-Att map at token t
                         -> optional original-resolution crop labels
```

For a generated token `t`, the map is the decoder adaptation of Grad-Att:

`A * ReLU(d log p(next_greedy_token) / d A)`.

Here `A` is Qwen's post-softmax text-decoder attention from token `t` to the
immutable full-image visual-KV keys.  It is not raw attention and it is not
the current ROI-bias policy.

## Output layout

```
OUTPUT/
  config.json
  manifest.jsonl                 # one self-contained record per question
  global_visual_kv/<hash>...pt   # saved once per unique source image
  steps/<sample>_<hash>.pt       # [collected steps, hidden_size] BF16 states
  grad_maps/<sample>_<hash>.npz  # optional normalized Grad-Att maps
  crops/<sample>_<hash>/...png   # exact source-pixel crops; never resized
```

Each manifest step always has a hidden-state row.  `crops: []` records that
Grad-Att was too diffuse/no-positive/no-legal-region at that reasoning step.
Crop paths are relative to `OUTPUT`; repeated near-identical regions reuse an
earlier crop path rather than duplicating image files.

`global-kv-mode=int8` is the default because complete FP16 decoder visual KV
can require hundreds of MB per large image.  It stores one symmetric int8
tensor plus scale for every key/value layer.  Use `fp16` only when exact cache
replay is required and storage has been budgeted.  Saved K/V contains visual
key positions only; it never serializes question or reasoning text K/V.

## Default localization constraints

- Grad-Att is computed at every one of the first 100 reasoning tokens.
- Mean across decoder layers; all positive Grad-Att is normalized to one.
- Reject diffuse maps: normalized concentration `1 - H(p)/log(N) < 0.12`.
- Keep connected top-mass components until 0.60 cumulative Grad-Att mass.
- Pad each selected component by one merged visual patch; enforce at least
  `2 x 2` merged patches and 64 original-image pixels per side.
- Reject crops over 25% of original visual-grid area and components with
  local contrast below 2.0 (region mass / region area fraction).
- Save at most two regions per token, and reuse a crop when IoU is at least
  0.90.
- Within one token step, proposals with IoU above 0.50 are unioned before the
  original-resolution crop is written.  This avoids two nearly identical crops
  while preserving the causal time boundary of teacher evidence.
- Before writing, remove a proposal only if it is both near-uniform in coarse
  RGB entropy (`<0.06`) **and** structurally blank in luma edge density
  (`<0.012`).  The detail calculation may use a 256-pixel diagnostic proxy;
  the saved teacher crop is never resized.  Rejected bboxes and measurements
  remain in `steps[*].metrics.filtered_low_detail_regions` for auditing.
- The default `--detail-filter-policy singleton_empty` writes `crops: []`
  when the token has exactly one selected proposal and it is low-detail. With
  multiple proposals, low-detail candidates are removed individually and
  useful regions remain. `all_candidates` is available only as a deliberately
  stricter ablation.

These are deliberately exposed as CLI values.  The first full collection
should preserve `steps[*].metrics` even where no crop is saved, then thresholds
can be selected from actual distributions rather than from a guessed absolute
Grad-Att scale.
