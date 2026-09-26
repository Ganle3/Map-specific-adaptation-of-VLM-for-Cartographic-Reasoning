# Validation — 2026-09-25

Environment: Windows, NVIDIA RTX A4500 20 GB, VisionGRPO conda environment.

| Package | Version |
|---|---|
| torch | 2.11.0+cu126 |
| transformers | 5.16.1 |
| peft | 0.20.0 |
| accelerate | 1.14.0 |
| bitsandbytes | 0.50.0 |
| numpy | 2.4.4 |
| Pillow | 12.2.0 |
| matplotlib | 3.11.2 |

- Two CPU tests pass: real Qwen3-VL eager attention output equality, selected head ordering, image grid reshaping, prefill and cached decode alignment, hook/config restoration, invalid layer rejection.
- GPU smoke test: baseline and checkpoint-2500 each generated 4 tokens with 238 image tokens. Both attention traces and HTML/PNG output succeeded. These short outputs are deliberately marked `token_limit`.
- Full requested test: `mapwise_china_map20_0D_t16_idx0119`, default 1,638,400 max pixels, NF4 4-bit, layers 20/31, heads 0/1/2/3. Actual merged image grid 25 × 31 (775 image tokens). Baseline generated 276 tokens and adapted 92 tokens, both ending with EOS. Trace shapes respectively `[276, 2, 4, 25, 31]` and `[92, 2, 4, 25, 31]`.
- Baseline final answer: `3,216.0 - 5,812.0` (incorrect against dataset).
- Adapted final answer: `5,812.0 - 9,432.0` (matches dataset).
- This one example is not an aggregate accuracy claim or proof of why adaptation changed the answer.
- PNG previews were inspected. HTML payload and raw arrays were checked, but interactive browser automation was unavailable in this environment; browser playback/buttons have not been exercised automatically.

## Corrected full-head run

After the first diagnostic run, all 36 decoder layers were fixed to eager attention regardless of which layers were saved. This prevents the visualization selection itself from changing the inference kernels. The verified run is in `outputs/test400_verified/` and captures layers 8/16/24/35, all 32 query heads. Trace shapes are `[279, 4, 32, 25, 31]` for baseline and `[98, 4, 32, 25, 31]` for adapted; both end with EOS.

The spatial mapping is checked against Transformers' own Qwen preprocessing and position-index implementation: preprocessing orders raw patches as `(block_y, block_x, inner_y, inner_x)`, each consecutive 2×2 group is merged into one LLM image token, and the remaining merged tokens are raster ordered by `(block_y, block_x)`. A unit test now checks that mapping explicitly. The 1000×800 source becomes 992×800 model input, giving a 31×25 merged-token grid; mapping that grid back to 1000×800 introduces only the expected horizontal resize scaling.

Inspection of head-mean weights shows both meaningful local peaks and attention sinks. At layer 16, tokens around “legend” peak near grid rows 20–24 and columns 10–12, overlapping the legend. At layer 24, generation around “GX” includes a strong local peak around row 16, column 17, near Guangxi, but is dominated by a persistent top-row peak around columns 25–26. That top-row peak is a real attention sink in this layer, not a coordinate decoding shift, and should not be interpreted as task evidence.

Results are in `outputs/test400/mapwise_china_map20_0D_t16_idx0119/`. Attention was captured in the native forward calls of these generation runs, not reconstructed by replaying the responses. The full run preceded the addition of two metadata fields (`generation_config`, `model_commit`); these fields are written by future runs. Sampling-only defaults were subsequently reset to suppress irrelevant greedy-generation warnings; greedy decoding behavior is unchanged.
