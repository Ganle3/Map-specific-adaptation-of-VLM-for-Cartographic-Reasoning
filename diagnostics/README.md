# Fixed20 batch gradient diagnostic

This measures five **optimizer batches at one fixed parameter state**, using
native TRL rollouts, correctness rewards, group advantages and `dr_grpo` loss.
It does not construct an optimizer, call its step, or change the training files.

## Run

Use the original single-GPU training container (TRL **1.12.0**, dependencies in
`../requirements_visiongrpo.txt` in the parent workspace). From `VLM_adaptation`:

```bash
python diagnostics/gradient/grpo_batch_gradient.py \
  --checkpoint /path/to/run/checkpoint-100 \
  --output-dir results/diagnostics/fixed20_gradient/checkpoint_100_seed42 \
  --seed 42

python diagnostics/visualization/plot_batch_gradient_matrix.py \
  --results-dir results/diagnostics/fixed20_gradient/checkpoint_100_seed42 \
  --dot-product
```

Supply `--image-root` and `--qa-json` if the original datasets are elsewhere.
`--training-config /path/to/effective_grpo_config.json` loads the original
configuration; it is automatically detected in the checkpoint's parent directory.
If absent, baseline generation defaults apply: temperature 0.8, top-p 0.95,
1536 completion tokens, group reward scaling, truncated-completion masking.
The effective settings and source hashes are saved. Use an empty output directory.
The adapter must have the fixed20 joint scope and rank/alpha 16/16.

The historical local `train_joint20.sbatch` omits explicit 2x8 and fixed-order
flags; it does not establish the actual remote training configuration. This
diagnostic uses the user-confirmed 2x8 setup and declared QA IDs. Conflicting
saved batch/objective settings fail explicitly. Match the base model revision,
container and evaluator to the original run before interpreting results.

## Reused code and data

- `Training_scripts/Debug_GRPO/grpo_joint_debug.py` delegates to
  `train_mapwise_grpo_joint4.py`; its `preset('joint_r16')` defines the scope:
  108 vision modules, 8 merger modules, 252 language modules; 736 LoRA tensors.
- `_vislora_grpo_baseline_snapshot.run_training` supplies processor/model loading,
  NF4/k-bit preparation, trainable adapter loading, dataset construction,
  evaluator initialization and trainer construction.
- `load_json_list` constructs missing `qa_id` from country/map/template/source.
  `build_train_dataset` produces conversational `prompt`, decoded PIL `images`,
  QA metadata, `ground_truth` and `ground_truth_type`. No alternate preprocessing.
- `mapwise_correctness_reward` calls `mapwise_evaluation_exact.evaluate_sample`
  and returns only binary strict exact correctness.
- The 20-QA JSON is the existing `mapwise_grpo_joint44_improved_20.json`.
  `Euler_results/VisionLoRA/debug50/analysis/select_joint4.py` retrospectively
  selected QAs whose last-five-epoch correct count exceeded their first-five
  count and ranked by late accuracy/truncation criteria. We validate exactly the
  declared IDs and map batches by ID, without reselecting from current rewards.
- Training checkpoints use Trainer/PEFT; final adapter uses `save_pretrained`.
  Measurement uses the baseline's `init_adapter_path` path with
  `PeftModel.from_pretrained(..., is_trainable=True)`, never optimizer resume.

The loader is monolithic. A temporary, process-local replacement of its trainer
class overrides `train()` to measure and raise a private completion exception
before the training/save continuation. Config and scope verification are also
replaced locally and restored on exit. No training source is patched. Native
generation and loss methods are inherited; reward interception only records
the returned rewards. Training audit callbacks are never dispatched. The
separate A/B reuse/replay trainers and Unsloth are not involved.

## Gradient definition and controls

Each batch contains four adjacent groups of four duplicated prompt rows. Native
`_prepare_inputs` generates once, scores group rewards/advantages, shuffles and
splits multimodal data into eight two-rollout microbatches. It retains the same
generated trajectories for all eight losses. `_step` is incremented manually
because no training loop is running; the generation buffer resets per batch.

TRL 1.12 `dr_grpo` already divides each microbatch loss by 8. Therefore

`g_B = sum_m autograd.grad(native_loss_m, trainable_LoRA)`

has the effective denominator `16 * max_completion_length`. There is **no
second division by 8**, clipping, Adam transformation, or optimizer update.
The reported loss is the sum of those native losses. Zero-valued policy loss
does not imply zero gradient. Unused parameters occupy zero entries in the
same named parameter order for every vector. Gradients accumulate on CPU in
float32; pairwise products accumulate in float64. Full gradients are not saved.

Accelerate prepares the model for the same mixed-precision forward behavior.
Non-reentrant gradient checkpointing enables `autograd.grad`; no backward
fallback silently changes behavior. If this fails in a different installation,
the run fails and should be diagnosed before introducing a fallback.

Frozen visual Linear4bit biases undergo their normal first-use BF16 conversion
before theta is fixed. Parameters and buffers are hashed after this preparation
and after every batch; changes abort the run. The preparation is recorded in
config. All batches retain training mode, and dropout is zero. No model update
occurs. CPU/GPU roundoff still prevents a promise of bitwise reproducibility
across different software/hardware.

`signal_group_count` counts reward groups with nonzero population standard
deviation; it does not guarantee a nonzero gradient after truncation masking.
`reward_std` is the population standard deviation across all 16 rewards.
Norms below `--eps` (default `1e-12`) have `valid_gradient=False` and NaN cosine
against every batch, including themselves. Dot products and distances retain
their mathematical values. Nonfinite gradients abort rather than become zeros.

## Outputs and interpretation

`config.json`, `batch_metrics.csv`, `valid_gradient_mask.csv`,
`cosine_matrix.csv`, `dot_product_matrix.csv`, `distance_matrix.csv`.
Plotting creates `cosine_matrix.png`, `gradient_norms.png` and optionally
`dot_product_matrix.png`. NaN cosine cells are grey and labelled N/A.
`config.json` says `status=complete` only after successful state checks.

Default: one realization, with batch seed `seed + batch_index`. Optional
`--num-repeats 5` writes independent `repeat_000` etc. directories, with seeds
`seed + 5 * repeat + batch_index`; plot each separately. No Monte Carlo inference
or repeat-summary statistics are implemented in this first version.

A negative cosine is descriptive evidence of opposing sampled gradient
directions at this state. One realization does not establish stable/statistically
significant conflict or catastrophic forgetting. Those require repeats and,
later, a separate B-update/A-degradation experiment.

CPU checks: `python -m pytest diagnostics/test_batch_gradients.py -q`.
These cover geometry, zero handling, file/plot outputs and fixed20 IDs; they do
not replace a real CUDA/TRL run.
