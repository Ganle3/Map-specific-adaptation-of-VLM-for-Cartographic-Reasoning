# SparseVLM-style rolling visual re-input

`rolling_sparse_qwen3vl.py` is a prototype for the existing Qwen3-VL-8B-Thinking + PEFT adapter setup. It deliberately does **not** modify Qwen's decoder or reuse a stale multimodal KV cache.

1. Window 0 receives the whole image and generates a bounded thinking segment.
2. Native, post-softmax decoder attention from that real generation ranks image patches.
3. Each later window re-encodes only the dynamically re-ranked high-priority crops and a bounded textual rolling state. Its current reasoning therefore determines the next ranking. The complete image is intentionally omitted by default so global context does not dilute local perception. Use `--include-full-image-after-first` only as an ablation.

## Optional full-image scout → crop-only focus

For later reasoning stages, enable both options below to use the original map
only to refresh crop selection. `3` is zero-based, so it starts at the fourth
reasoning window:

```powershell
--scout-full-image-from-window 3 --full-image-scout-only
```

At and after that threshold, each window performs (a) a short private
full-image visual-planning generation (`--scout-tokens`, default 80) with the
current rolling state, aggregates its real token-by-token attention to select
crops, then deletes its result/KV allocation; (b) generates using only the
scout-selected crop images. The full image is never an input to the formal
generation call. This adds one full-image generation per late window but
prevents full-image tokens from occupying the formal reasoning KV cache.

Use `--scout-full-image-interval N` to periodically return to the complete
image rather than recursively cropping crops forever. For example, start at
window 2 and use interval 2 to obtain: window 0 full image, window 1 crop
refinement, window 2 full-image scout + crop-only generation, window 3 crop
refinement, window 4 full-image scout + crop-only generation.

## Evidence ledger and final-answer mode

`--state-mode ledger` is the default. After every non-final visual window, a
small text-only model call compresses the provisional segment into:

```text
VERIFIED:
- visually supported facts
UNRESOLVED:
- facts still requiring inspection
NEXT:
- the next visual check
```

The next window receives this bounded ledger rather than a token-truncated raw
chain-of-thought. The final window uses `--final-window-tokens` (default 96)
and is instructed to emit exactly `Final answer: <answer>`. The JSON result has
both `final_text` and a parsed `final_answer`. Use `--state-mode raw` only for
the legacy raw-transcript ablation.

The first-window map uses native real-generation attention. Default aggregation is `last`, not max. High-attention patches are selected as variable-size four-connected components whose cumulative score reaches `--attention-mass` (default `0.60`), rather than as fixed-size boxes. Use `--aggregation mean`, `max`, or `weighted_mean` only as explicit ablations.

## Run

From `C:\Users\junyhuang\Thesis\VLM_adaptation`:

```powershell
python .\SparseVLM_eliminateCONTEXT\rolling_sparse_qwen3vl.py `
  --image .\Datasets\mapwise-dataset\india\images\with_annotations\YOUR_MAP.png `
  --question "YOUR QUESTION" `
  --adapter .\Training_outputs\YOUR_EXPERIMENT\checkpoint-2500 `
  --output .\Evaluation_results\rolling_sparse_example.json
```

When `--adapter` is given, its `adapter_config.json` supplies the base automatically. If the adapter was trained from the Unsloth 4-bit base, pass that exact base through `--model`; otherwise PEFT weights will not be compatible.

`--output` accepts either a JSON filename or a directory. With a directory such as `pilot_test_output`, the result is written to `pilot_test_output\rolling_result.json`.

Unless `--no-visualize` is supplied, the script also writes
`<output-stem>_windows\`. For every window it contains:

- `window_XX_input_rank_YY.png`: the exact crop image supplied to that window;
- `window_XX_next_rank_YY.png`: the attention-ranked crop supplied to the next window;
- `window_XX_input_overlay.png` and `window_XX_next_overlay.png`: those regions drawn on the unchanged original image.

## What SparseVLM itself does

The upstream [`score.py`](https://github.com/Gumpest/SparseVLMs/blob/main/llava/model/language_model/score.py) performs no cross-decoder-layer max, mean, or learned weighting. At each pruning layer (default 2, 6, 15), it averages attention over heads, selects informative text tokens, averages over those text queries, and top-k keeps that layer's visual tokens. Survivors are physically passed into the following layers. In SparseVLM+ v2, it first chooses the top half of heads by total text-to-vision attention; this still happens within one layer.

The existing project visualisation's `all_layer_step_max` is only a diagnostic rendering summary, not the model's pruning policy.

## Tests

```powershell
cd .\SparseVLM_eliminateCONTEXT
python -m unittest -v test_attention_policy.py
```

The test is CPU-only and verifies the single-layer SparseVLM scoring order, explicit aggregation choices, and spatial crop priority.

## Continuous-cache prototype

`continuous_cache_qwen3vl.py` is separate from the reset baseline. It retains
the initial full-image KV and every generated reasoning token, appends crop
turns to the same cache, and records global/crop attention mass per window.
Use `--global-key-bias -4.0` to suppress only the initial full-image key logits
(about `exp(-4)` before competing-key normalization), while preserving their
availability. `--full-reselect-from-window` and
`--full-reselect-interval` periodically choose the next crop from the cached
global-image attention map; the full image is not re-encoded.

This is an experimental cache-continuation implementation tied to the local
Qwen3-VL Transformers version. Start with a single image and inspect its JSON
`attention_mass.global_share` before drawing conclusions from the bias.
