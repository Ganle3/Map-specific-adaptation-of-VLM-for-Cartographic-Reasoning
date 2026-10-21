# Latent crop representation — phase 1

This directory trains `LatentVisualExtractor` to reconstruct the **main Qwen3-VL visual-merger embeddings** of Grad-Att-selected, original-pixel crop images. It does not yet insert predicted embeddings into the language decoder. That inference protocol and multi-crop M-RoPE layout are deliberately deferred until representation/STOP quality is measured.

## Files

- `latent_visual_extractor.py`: the trainable class, configuration and layer-aware visual-memory fusion.
- `prepare_teacher.py`: frozen NF4 Qwen3-VL + adapter pass over each **unique** crop file; saves the actual main-merger output and `image_grid_thw`. It does not save or predict crop DeepStack features.
- `data.py`: joins existing reasoning hidden states, all-layer global visual KV and Teacher embeddings; handles map-grouped splits and padded batches.
- `train.py`: reconstruction + learnable STOP training, validation, checkpoints and metrics.
- `evaluate.py`: held-out map evaluation after selecting a checkpoint.
- `tests/`: small CPU gradient, split and saved-GQA-order checks.

## Model

The model reads one final decoder hidden state and the saved global-image visual positions from **all 36 decoder KV layers**. It keeps K and V separate, applies independent per-layer softmax gates, and projects the fused memory into the latent space. One cross-attention block lets parallel latent queries read that memory. Queries receive reasoning conditioning and canonical 2-D row/column positional embeddings. A trainable `stop_token` in the same latent space is compared by cosine similarity with each candidate state; it is **not** an LM token or a separate WHEN classifier. The non-STOP states project to the actual main-merger width measured from Teacher files.

Defaults: `K_max=384`, latent width = the saved KV's flattened head width (1024 for this checkpoint), eight attention heads, one cross-attention block and a 2× residual MLP. For the inspected 4096/1024/4096, 36-layer checkpoint this is **19,340,360 trainable parameters**. No Qwen or LoRA parameters are trainable in `train.py`.

Multiple crop targets from a single step are concatenated **in the collector's rank order** and share a single final STOP label. An empty collector crop list gives a STOP target at position zero. Teacher totals above 384 are reported and **skipped rather than truncated**. The canonical query grid is an internal Student position encoding; it is not yet Qwen's per-crop M-RoPE grid.

## Run after Grad-Att collection finishes

From `C:\Users\junyhuang\Thesis` in `VisionGRPO` PowerShell:

```powershell
python .\VLM_adaptation\LatentCrop\prepare_teacher.py `
  --source-root .\VLM_adaptation\Training_outputs\gradattn_teacher `
  --output-dir .\VLM_adaptation\Training_outputs\latent_crop_teacher_main `
  --adapter .\VLM_adaptation\Training_outputs\MapVerse_scaling1000_GRPO_from_MapVerse5000_token500\checkpoint-2500 `
  --max-pixels 1000000
```

The input crop files remain unresized. Qwen's processor applies its dynamic-resolution resize **in memory**, as in the real crop append path. `--max-crops 16` can smoke-test encoding a subset, but normal training requires the complete target cache. Re-running the preparation command resumes already valid targets.

The collector's crop paths, hidden-state files and visual-KV paths are relative to `--source-root`, so these data can be copied to Euler. The preparation script accepts a Linux adapter path if its final experiment/checkpoint directory names and NF4 setting match the Windows collector configuration. This label check cannot prove the checkpoint bytes are identical; copy the actual same adapter. Use a fresh Teacher output directory when changing model weights or pixel budget.

Inspect joined data, inferred native dimensions, map splits and parameter count without starting optimization:

```powershell
python .\VLM_adaptation\LatentCrop\train.py `
  --source-root .\VLM_adaptation\Training_outputs\gradattn_teacher `
  --teacher-root .\VLM_adaptation\Training_outputs\latent_crop_teacher_main `
  --output-dir .\VLM_adaptation\Training_outputs\latent_crop_phase1 `
  --inspect-only
```

Train with four steps per forward and four forwards per optimizer update (effective batch 16), using BF16 autocast when CUDA is available:

```powershell
python .\VLM_adaptation\LatentCrop\train.py `
  --source-root .\VLM_adaptation\Training_outputs\gradattn_teacher `
  --teacher-root .\VLM_adaptation\Training_outputs\latent_crop_teacher_main `
  --output-dir .\VLM_adaptation\Training_outputs\latent_crop_phase1 `
  --epochs 20 `
  --batch-size 4 `
  --gradient-accumulation 4 `
  --k-max 384 `
  --seed 42
```

The default 80/10/10 split is by **image SHA**, not question or reasoning step. The training sampler gives each map equal expected weight. `metrics.jsonl` reports MSE, cosine error, embedding norms, STOP precision/recall, predicted length error and no-crop accuracy. Validation also breaks down reconstruction by Teacher length. `best.pt`, `last.pt` and epoch checkpoints are independent Student checkpoints. The test maps are not touched during training.

For a quick local memory/gradient test **after targets are available**, run `train.py` with a separate output directory and `--epochs 1 --batch-size 1 --gradient-accumulation 1 --max-train-batches 2 --max-validation-batches 2`. These truncated metrics are not usable for model selection. The training process does not load the 8B VLM; only `prepare_teacher.py` does.

After choosing the checkpoint using validation maps:

```powershell
python .\VLM_adaptation\LatentCrop\evaluate.py `
  --source-root .\VLM_adaptation\Training_outputs\gradattn_teacher `
  --teacher-root .\VLM_adaptation\Training_outputs\latent_crop_teacher_main `
  --run-dir .\VLM_adaptation\Training_outputs\latent_crop_phase1 `
  --checkpoint .\VLM_adaptation\Training_outputs\latent_crop_phase1\best.pt `
  --split test
```

## Interpretation limits

All collector-empty steps are deliberately used as first-position STOP labels in this first experiment. These are heuristic negatives, not independently verified no-look decisions. Reconstruction similarity alone cannot establish that generated embeddings improve final answers. No decoder injection or full-Qwen GPU run is claimed by the CPU tests in this directory.
