# VLM Adaptation for Cartographic Reasoning

This repository contains the training and evaluation code used for
adapting vision-language models (VLMs) to cartographic reasoning tasks.
The current experiments focus on Qwen3-VL-8B-Thinking,
parameter-efficient adaptation with LoRA, and GRPO training on MapWise.

## Project Structure

```text
VLM_adaptation/
├── Datasets/
│   ├── FRIEDA/
│   ├── mapwise-dataset/
│   ├── Processed_FRIEDA/
│   └── Processed_Mapwise/
│
├── Evaluation_results/
│
├── Evaluation_scripts/
│   ├── GRPO_ablation/
│   │   ├── inference_mapwise_trl.py
│   │   └── mapwise_evaluation_exact.py
│   │
│   ├── frieda_evaluation.py
│   ├── inference_frieda.py
│   ├── inference_mapwise.py
│   ├── mapwise_evaluation.py
│   ├── run_frieda.py
│   └── run_mapwise.py
│
├── Training_outputs/
│
├── Training_scripts/
│   ├── Debug_GRPO/
│   │   ├── frieda_adapter_rollout_diagnostic.py
│   │   ├── frieda_grpo_feasibility.py
│   │   ├── mapwise_rollout_diagnostic.py
│   │   ├── train_frieda_grpo_unsloth.py
│   │   └── train_mapwise_grpo_unsloth.py
│   │
│   ├── Pilot_study_SFT/
│   │   ├── cot_collator.py
│   │   ├── dataset_utils.py
│   │   ├── debug_utils.py
│   │   ├── LLM_attn_setting_debug.py
│   │   ├── LLM_attn_setting_masked_think.py
│   │   ├── LLM_attn_setting.py
│   │   └── train_frieda_c0_c1.py
│   │
│   └── Vision_GRPO/
│       ├── inspect_qwen3vl_vision_modules.py
│       ├── test_native_vision_grpo_gradient.py
│       ├── train_mapwise_grpo_native_minipilot.py
│       └── train_mapwise_grpo_visLoRA_trl.py
│
├── requirements.txt
├── requirements_visiongrpo.txt
└── requirements_lock.txt
```

`Debug_GRPO/` contains the earlier Unsloth-based GRPO implementation and
diagnostic experiments. `Vision_GRPO/` contains the native Hugging Face /
PEFT / TRL implementation used for Vision-LoRA GRPO experiments.

`Evaluation_scripts/GRPO_ablation/` contains the corresponding native
MapWise inference and strict exact-match validation pipeline.
```

The main experimental workflow is:

``` text
Processed dataset
      ↓
GRPO / SFT training
      ↓
Training_outputs/checkpoint-*
      ↓
MapWise inference
      ↓
Prediction JSON
      ↓
MapWise evaluation
      ↓
Evaluation_results/
```

## Environment

The experiments use Python, PyTorch, Unsloth, TRL, Hugging Face
Transformers/Datasets, and Weights & Biases.

Install the project dependencies with:

``` powershell
pip install -r requirements.txt
```

`requirements_lock.txt` can be used when reproducing the exact package
versions of the experimental environment.

A CUDA-capable NVIDIA GPU is required for the current training scripts.

## Data

Raw and processed datasets are stored under `Datasets/`.

The main MapWise resources used by the training and evaluation scripts
are:

``` text
Datasets/
├── mapwise-dataset/          # Map images
└── Processed_Mapwise/        # Processed QA splits
```

Training and evaluation scripts contain default paths for the
corresponding datasets. These paths can also be overridden through
command-line arguments where supported.

## GRPO Training

The main MapWise GRPO training script is:

``` text
Training_scripts/Debug_GRPO/train_mapwise_grpo_unsloth.py
```

### Current Vision-only GRPO experiment

The current diagnostic experiment adapts only the visual side of
Qwen3-VL while keeping the language layers frozen.

Main configuration:

  Setting                     Value
  --------------------------- -------------------------------------------------
  Base model                  `unsloth/Qwen3-VL-8B-Thinking-unsloth-bnb-4bit`
  Training method             GRPO
  Quantization                4-bit
  LoRA rank                   16
  LoRA alpha                  16
  Vision layers               Trainable with LoRA
  Language layers             Frozen
  Attention modules           Enabled within selected LoRA scope
  MLP modules                 Enabled within selected LoRA scope
  Reward                      Strict exact-match correctness only
  Correct reward              +1.0
  Incorrect reward            0.0
  Rollouts per prompt         6
  Temperature                 0.8
  Top-p                       0.95
  Learning rate               `5e-6`
  Epochs                      1
  Maximum completion length   1536

The reward intentionally excludes additional format rewards and
behavior/repetition penalties for this experiment. This isolates the
effect of task correctness during GRPO.

### Run training

## GRPO Training

Two GRPO implementations are retained in the repository. The earlier
experiments use Unsloth, while the native implementation uses Hugging Face
Transformers, PEFT, TRL, and bitsandbytes.

### Unsloth

The Unsloth implementation is located under:

```text
Training_scripts/Debug_GRPO/
```

Run MapWise GRPO with:

```powershell
python .\Training_scripts\Debug_GRPO\train_mapwise_grpo_unsloth.py `
  --lora-rank 16 `
  --lora-alpha 16 `
  --learning-rate 5e-6 `
  --num-generations 4 `
  --temperature 0.8 `
  --top-p 0.95 `
  --num-train-epochs 1 `
  --save-steps 50 `
  --wandb-project "MapWise-GRPO-Ablation-VisOnly" `
  --wandb-run-name "VisionOnly-R16-CorrectnessOnly"
```

This implementation uses the Unsloth Qwen3-VL model:

```text
unsloth/Qwen3-VL-8B-Thinking-unsloth-bnb-4bit
```

### Native Hugging Face / TRL

The native Vision-LoRA implementation is located under:

```text
Training_scripts/Vision_GRPO/
```

Run the formal MapWise Vision-LoRA GRPO experiment with:

```powershell
python .\Training_scripts\Vision_GRPO\train_mapwise_grpo_visLoRA_trl.py `
  --lora-rank 16 `
  --lora-alpha 16 `
  --learning-rate 5e-6 `
  --num-generations 4 `
  --temperature 0.8 `
  --top-p 0.95 `
  --num-train-epochs 1 `
  --save-steps 100 `
  --wandb-project "MapWise-GRPO-Ablation-VisOnly" `
  --wandb-run-name "VisionOnly-R16-CorrectnessOnly-TRL"
```

The native implementation uses:

```text
Qwen/Qwen3-VL-8B-Thinking
```

with 4-bit NF4 quantization and Vision-only LoRA through Hugging Face
Transformers, PEFT, and TRL.

The current GRPO reward is intentionally minimal:

```text
strict exact answer match = 1
otherwise                 = 0
```

No auxiliary format, completion, or repetition rewards are used.

Training checkpoints and the final adapter are written under
`Training_outputs/`. Each GRPO experiment should use a separate output
directory so that checkpoints from different implementations or ablations
are not mixed.

For a new ablation run, start from the same base model unless the experiment
explicitly requires checkpoint continuation.

## MapWise Evaluation

## MapWise GRPO Evaluation

Two evaluation paths are retained to match the corresponding GRPO
implementations.

### Unsloth

The original Unsloth evaluation entry point is:

```text
Evaluation_scripts/run_mapwise.py
```

`run_mapwise.py` implements a two-stage pipeline:

```text
Stage 1: inference
    ↓
prediction JSON
Stage 2: evaluation
    ↓
evaluation results
```

#### Baseline evaluation

If `--adapter-path` is omitted, the Unsloth base model is evaluated directly:

```powershell
python .\Evaluation_scripts\run_mapwise.py `
  --overwrite
```

The model can also be specified explicitly:

```powershell
python .\Evaluation_scripts\run_mapwise.py `
  --model-name "unsloth/Qwen3-VL-8B-Thinking-unsloth-bnb-4bit" `
  --overwrite
```

#### Evaluate an Unsloth GRPO checkpoint

```powershell
python .\Evaluation_scripts\run_mapwise.py `
  --adapter-path ".\Training_outputs\<experiment>\checkpoint-100" `
  --predictions-json ".\Evaluation_results\<experiment>\checkpoint-100_predictions.json" `
  --output-dir ".\Evaluation_results\<experiment>\checkpoint-100" `
  --overwrite
```

The adapter directory must contain `adapter_config.json`.

Existing prediction files can be evaluated without repeating inference:

```powershell
python .\Evaluation_scripts\run_mapwise.py `
  --evaluation-only `
  --predictions-json ".\Evaluation_results\<experiment>\checkpoint-100_predictions.json" `
  --output-dir ".\Evaluation_results\<experiment>\checkpoint-100"
```

### Native Hugging Face / TRL

The native evaluation scripts are located under:

```text
Evaluation_scripts/GRPO_ablation/
```

The pipeline deliberately separates model inference from scoring:

```text
inference_mapwise_trl.py
        ↓
mapwise_predictions.json
        ↓
mapwise_evaluation_exact.py
        ↓
strict exact-match accuracy
```

This allows prediction files to be re-evaluated without repeating expensive
VLM inference.

The current GRPO dataset excludes List and Rank questions requiring more
complex partial or structured metrics. All retained answer types are evaluated
using strict exact correctness:

```text
Binary  → exact match
Count   → exact match
Range   → exact match
Single  → exact match
```

For `Single` questions, equivalent administrative-region names and
abbreviations are normalized before comparison, but no partial credit or
recall score is assigned.

The headline validation metric is:

```text
Validation accuracy =
number of strictly correct answers / total validation questions
```

This is directly aligned with the correctness-only reward used during GRPO
training.

#### Baseline inference

Evaluate the native Qwen3-VL baseline with:

```powershell
python .\Evaluation_scripts\GRPO_ablation\inference_mapwise_trl.py `
  --qa-json ".\Datasets\Processed_Mapwise\Train_Val\mapwise_grpo_validation.json" `
  --overwrite
```

The baseline model is:

```text
Qwen/Qwen3-VL-8B-Thinking
```

Validation inference uses deterministic decoding (`do_sample=False`) to avoid
sampling variance when comparing checkpoints.

#### Evaluate a native Vision-LoRA checkpoint

Run inference with a specific PEFT checkpoint:

```powershell
python .\Evaluation_scripts\GRPO_ablation\inference_mapwise_trl.py `
  --qa-json ".\Datasets\Processed_Mapwise\Train_Val\mapwise_grpo_validation.json" `
  --adapter-path ".\Training_outputs\MapWise_GRPO_Qwen3-VL-8B-Thinking_visLoRA_TRL\checkpoint-500" `
  --overwrite
```

The adapter directory must contain `adapter_config.json`.

The same procedure can be applied to successive checkpoints:

```text
baseline
checkpoint-500
checkpoint-600
checkpoint-700
...
```

The base model, validation data, prompt, image preprocessing, and decoding
configuration remain fixed. The only model-side variable is the loaded
Vision-LoRA checkpoint.

#### Strict exact-match evaluation

Evaluate the resulting prediction file with:

```powershell
python .\Evaluation_scripts\GRPO_ablation\mapwise_evaluation_exact.py `
  --predictions-json ".\Evaluation_results\<experiment>\checkpoint-500\mapwise_predictions.json"
```

The evaluator reports overall strict exact-match accuracy together with
breakdowns by answer type and country.

#### Useful native inference options

`inference_mapwise_trl.py` supports:

```text
--model-name
--adapter-path
--qa-json
--image-root
--output-json
--max-new-tokens
--thinking {auto,on,off}
--start-index
--end-index
--overwrite
--no-resume
--save-every
--print-every
```

`--start-index` and `--end-index` are useful for short sanity checks before
running the complete validation set.

## Outputs

### Training outputs

Training artifacts are stored under:

```text
Training_outputs/
```

Each experiment should use its own output directory. For example:

```text
Training_outputs/
├── MapWise_GRPO_Qwen3-VL-8B-Thinking_only_VisLoRA/
├── MapWise_GRPO_Qwen3-VL-8B-Thinking_run1/
└── MapWise_GRPO_Qwen3-VL-8B-Thinking_visLoRA_TRL/
    ├── checkpoint-500/
    ├── checkpoint-600/
    ├── checkpoint-700/
    ├── README.md
    ├── run_config.json
    └── step_metrics.jsonl
```

The checkpoint directories contain the PEFT/LoRA adapter state required for
checkpoint-wise evaluation or training continuation.

### Evaluation outputs

Inference predictions and evaluation summaries should be stored under:

```text
Evaluation_results/
```

For the native GRPO validation pipeline, a typical checkpoint evaluation
contains:

```text
<experiment>/
└── checkpoint-500/
    ├── mapwise_predictions.json
    ├── evaluation_details.json
    ├── evaluation_details.csv
    └── evaluation_summary.json
```

`mapwise_predictions.json` preserves the raw model generations, while
`evaluation_summary.json` contains the strict exact-match validation accuracy.
The detailed JSON and CSV files retain per-question results for subsequent
error analysis.

Separate output directories should be used for the baseline and each
checkpoint so that every evaluation result remains traceable to the
corresponding training configuration.

## Experimental Workflow

For the current GRPO ablation study, the recommended workflow is:

1.  Train an adapter from the same Qwen3-VL baseline.
2.  Save checkpoints at fixed training intervals.
3.  Run deterministic MapWise inference for selected checkpoints.
4.  Evaluate predictions using the common MapWise evaluator.
5.  Compare checkpoint accuracy against the unchanged baseline.
6.  Use the result to determine the next LoRA ablation (e.g. lower
    vision rank or freeze vision and adapt the language side).

Keeping the dataset split, evaluation prompt, inference settings, and
evaluator fixed across ablations is important for controlled comparison.

## Reproducibility Notes

-   Do not resume a new ablation experiment from a checkpoint produced
    by a different LoRA configuration.
-   Use distinct output directories and W&B run names for different
    experiments.
-   Record the LoRA scope, rank, reward design, learning rate, rollout
    configuration, and evaluated checkpoint for every run.
-   Baseline and adapted-model comparisons should use the same MapWise
    test set and evaluation procedure.
