# Qwen3-VL ILVAD single-case reproduction

This is a training-free port of ILVAD for the official
`Qwen/Qwen3-VL-8B-Thinking` checkpoint plus a local PEFT adapter. It does not
use Unsloth and does not merge, train, save, or alter model/LoRA weights.

The default is the project extension **rolling ILVAD**: after 12 native
attention steps, it replaces (never accumulates) the visual map every 4 tokens
using the preceding 8-token window. Maps are made from pre-enhancement
attention and take effect on the next step. Use `--mode fixed --fixed-t 10` only to
reproduce the upstream fixed-map method.

Run the minimal requested case (inside the same Python environment used for the
existing Qwen inference scripts):

```powershell
python run_single.py
python visualization/render_comparison.py --run outputs/mapverse_image2_checkpoint-3000
```

Outputs include `baseline.json`, `ilvad.json`, per-step/layer/head attention
arrays, `enhanced_map.npy`, a PNG, machine-readable metrics, and `analysis.md`.
In rolling mode, use `--window`, `--update-interval`, and `--warmup`; it has no
`T` parameter. `--fixed-t` applies only to the fixed-map reproduction mode.

The comparison is deterministic greedy decoding. Attention results are a useful
mechanistic diagnostic: higher late-step attention to ILVAD-selected evidence is
consistent with less forgetting, but hallucination mitigation needs answer-level
checking against the map and should not be claimed from attention alone.
