# VLM Adaptation for Cartographic Reasoning

This repository contains the training and evaluation code used for
adapting vision-language models (VLMs) to cartographic reasoning tasks.
The current experiments focus on Qwen3-VL-8B-Thinking,
parameter-efficient adaptation with LoRA, and GRPO training on MapWise.

## Project Structure

``` text
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
│   ├── inference_frieda.py
│   ├── inference_mapwise.py
│   ├── frieda_evaluation.py
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
│   └── Pilot_study_SFT/
│       ├── cot_collator.py
│       ├── dataset_utils.py
│       ├── debug_utils.py
│       ├── LLM_attn_setting_debug.py
│       ├── LLM_attn_setting_masked_think.py
│       ├── LLM_attn_setting.py
│       └── train_frieda_c0_c1.py
│
├── requirements.txt
└── requirements_lock.txt
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

From `Training_scripts/Debug_GRPO/`:

``` powershell
python .\train_mapwise_grpo_unsloth.py `
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

Training checkpoints and the final adapter are written under
`Training_outputs/`. Each GRPO experiment should use a separate output
directory so that checkpoints from different ablations are not mixed.

For a new ablation run, start from the same base model unless the
experiment explicitly requires checkpoint continuation.

## MapWise Evaluation

The main evaluation entry point is:

``` text
Evaluation_scripts/run_mapwise.py
```

`run_mapwise.py` implements a two-stage pipeline:

``` text
Stage 1: inference
    ↓
prediction JSON

Stage 2: evaluation
    ↓
evaluation results
```

The script supports both the original baseline model and a PEFT/LoRA
adapter checkpoint.

### Baseline evaluation

If `--adapter-path` is omitted, the base model is evaluated directly:

``` powershell
python .\Evaluation_scripts\run_mapwise.py `
  --overwrite
```

The base model can be changed explicitly with:

``` powershell
python .\Evaluation_scripts\run_mapwise.py `
  --model-name "unsloth/Qwen3-VL-8B-Thinking-unsloth-bnb-4bit" `
  --overwrite
```

### Evaluate a GRPO checkpoint

Use `--adapter-path` to select a specific LoRA/GRPO checkpoint:

``` powershell
python .\Evaluation_scripts\run_mapwise.py `
  --adapter-path ".\Training_outputs\<experiment>\checkpoint-100" `
  --predictions-json ".\Evaluation_results\<experiment>\checkpoint-100_predictions.json" `
  --output-dir ".\Evaluation_results\<experiment>\checkpoint-100" `
  --overwrite
```

The adapter directory must contain `adapter_config.json`.

The same mechanism can be used to evaluate later checkpoints:

``` text
checkpoint-100
checkpoint-200
checkpoint-300
...
```

This allows checkpoint-wise evaluation of adaptation performance over
the course of GRPO training.

### Evaluate an existing prediction file

Inference can be skipped when predictions have already been generated:

``` powershell
python .\Evaluation_scripts\run_mapwise.py `
  --evaluation-only `
  --predictions-json ".\Evaluation_results\<experiment>\checkpoint-100_predictions.json" `
  --output-dir ".\Evaluation_results\<experiment>\checkpoint-100"
```

### Useful evaluation options

`run_mapwise.py` also supports:

``` text
--model-name
--adapter-path
--qa-json
--image-root
--predictions-json
--output-dir
--max-new-tokens
--thinking {auto,on,off}
--start-index
--end-index
--overwrite
--evaluation-only
```

`--start-index` and `--end-index` are useful for partial evaluation or
debugging.

## Outputs

### Training outputs

Training artifacts are stored under:

``` text
Training_outputs/
```

A typical GRPO run contains:

``` text
<experiment>/
├── checkpoint-100/
├── checkpoint-200/
├── ...
├── final_adapter/
├── run_config.json
├── train_metrics.json
└── last_grpo_sample.json
```

### Evaluation outputs

Inference predictions and evaluation summaries should be stored under:

``` text
Evaluation_results/
```

Using separate subdirectories for each experiment/checkpoint is
recommended so that results remain traceable to the corresponding
training configuration.

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
