# Relay3 versus adapter-only attention visualization

This tool loads the **step-4000** checkpoints for the matched adapter-only and
one-token relay3 runs. It performs fresh greedy inference on the ten selected
questions and captures the real post-softmax decoder attention assigned to
merged image-token keys.

Each question receives its own directory and standalone `comparison.html` with
three views:

1. adapter-only language query `x_t -> image`;
2. relay3 language query `x_t -> image`;
3. relay3 temporary relay query `r_t -> image`.

The viewer exposes every inference step and all 36 decoder layers. Selecting
**All layers** computes the patch-wise maximum across raw layer maps first.
Its default robust p99 contrast is applied only after that maximum and can be
switched to literal maximum normalization. Individual-layer maps retain their
literal per-map maximum normalization. The full-inference view similarly takes
a temporal maximum before display scaling. Raw float32 arrays are retained in
`adapter_only/attention.npz` and `relay3/attention.npz`.

At relay3 layer 0, `x_t` has not yet consumed the current step's relay output.
At layer `l > 0`, `x_t` already contains relay interactions from preceding
layers. The `x_t` and `r_t` rows at one layer are simultaneous self-attention
queries; the updated states become inputs to the next layer.

## Run on Windows

From `C:\Users\junyhuang\Thesis`:

```powershell
.\VLM_adaptation\Training_scripts\GRPO_relay\relay3_visualization\run_step4000.ps1
```

To replace an incomplete prior output:

```powershell
.\VLM_adaptation\Training_scripts\GRPO_relay\relay3_visualization\run_step4000.ps1 -Overwrite
```

Defaults match the evaluations: NF4, greedy decoding, 500 generated tokens,
and a one-million-pixel cap. Attention capture requires eager decoder attention,
as SDPA does not expose attention probabilities. Eager and SDPA can differ by
small floating-point amounts, so a greedy trajectory may occasionally differ
from the earlier evaluation JSONL even with identical weights.

## Outputs

```text
relay3_visualization_step4000/
  manifest.json
  mapverse_qa232/
    comparison.html
    adapter_only/{result.json,response.txt,attention.npz}
    relay3/{result.json,response.txt,attention.npz}
  ...
```

The HTML embeds gzip-compressed, uint8-normalized display maps and the original
image, so it remains a standalone local file. NPZ files retain unnormalized
float32 attention for quantitative analysis.

Existing HTML pages can be rebuilt from their NPZ files without rerunning model
inference:

```powershell
& C:\Users\junyhuang\.conda\envs\VisionGRPO\python.exe `
  .\VLM_adaptation\Training_scripts\GRPO_relay\relay3_visualization\rerender_existing.py `
  .\VLM_adaptation\Training_outputs\Prototype\relay3_visualization_step4000
```

PowerPoint-compatible relay-query GIFs for MapVerse 273 and MapWise 294 can be
regenerated with:

```powershell
& C:\Users\junyhuang\.conda\envs\VisionGRPO\python.exe `
  .\VLM_adaptation\Training_scripts\GRPO_relay\relay3_visualization\export_relay_gifs.py
```

This exports individual layers 6, 9, 12, 15, 18, 25 and 30, a seven-panel
overview, and a per-step spatial-trajectory CSV. GIF contrast is computed per
frame at p99 after selecting the raw layer map.

## Test

```powershell
& C:\Users\junyhuang\.conda\envs\VisionGRPO\python.exe -m unittest `
  VLM_adaptation.Training_scripts.GRPO_relay.relay3_visualization.test_render
```
