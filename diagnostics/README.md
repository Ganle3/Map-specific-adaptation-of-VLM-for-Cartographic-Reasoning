# Online optimizer-batch gradient diagnostic

**This replaces the former fixed20/offline diagnostic.** Train a random-order
joint20 or joint44 run for 40 epochs and observe actual accumulated gradients.
No per-question gradients, extra rollouts, extra backward passes, replay, gradient
surgery or fixed-state batch comparisons. No checkpoint or final adapter saves.

## Euler: submit both experiments

Upload the complete updated `VLM_adaptation` repo to `$HOME/VLM_adaptation`.
Use the existing `$SCRATCH/containers/ablationgrpo.sif` training container.

```bash
cd "$HOME/VLM_adaptation"
sbatch diagnostics/jobs/joint_online_gradient.sbatch
```

Array task 0 runs joint20; task 1 runs joint44, each with one RTX 4090, 64 GB CPU
RAM and a 24-hour wall-time request. This is an allocation, not a measured runtime
guarantee. Change resource directives for your available partition if necessary.
Test one epoch on both datasets first if the updated container has not been checked:

```bash
sbatch --export=ALL,EPOCHS=1 diagnostics/jobs/joint_online_gradient.sbatch
```

Results: `$SCRATCH/thesis/diagnostics/joint_gradient_<array_job_id>/joint20/`
and `joint44/`. Slurm logs are `joint-gradient-<array_job_id>_<task_id>.out`
in the submission directory. Each task automatically generates its own plots.
`REPO`, `CONTAINER`, `IMAGE_ROOT`, `RESULT_ROOT`, `SEED`, `EPOCHS` are overridable
environment variables. Use a fresh result directory. There is no resume because
weights are deliberately not saved. Timeouts can leave usable partial CSVs.

## Direct training and plotting

From `VLM_adaptation`, inside the training environment:

```bash
python diagnostics/gradient/grpo_batch_gradient.py \
  --dataset-size 20 --epochs 40 --seed 3407 \
  --output-dir results/diagnostics/joint20

python diagnostics/gradient/grpo_batch_gradient.py \
  --dataset-size 44 --epochs 40 --seed 3407 \
  --output-dir results/diagnostics/joint44

python diagnostics/visualization/plot_batch_gradient_matrix.py \
  --results-dir results/diagnostics/joint20 results/diagnostics/joint44 \
  --comparison-dir results/diagnostics/comparison
```

Plotting needs only NumPy and Matplotlib and can run on Windows after downloading
the result folders. The Euler script checks those imports before training. No
package installation is attempted inside the container. Train first/plot locally
with the direct training command if the container lacks Matplotlib.

CSV + automatically generated PNGs are the primary analysis outputs. Optional
WandB scalar logging is enabled by `--wandb` or
`sbatch --export=ALL,USE_WANDB=1 diagnostics/jobs/joint_online_gradient.sbatch`.
Authenticate beforehand in the container's environment. `WANDB_MODE=offline`
can be used when the compute node has no external connection. CSV outputs remain
available regardless of WandB; images are saved locally rather than uploaded.

## What is trained and measured?

- Fresh Qwen3-VL-8B-Thinking + NF4 + joint LoRA, default rank/alpha 16/16.
  `--rank` changes both consistently. Original loader, processor, preprocessing,
  correctness evaluator and joint target preset are imported and reused.
- Native TRL 1.12.0 / Transformers 5.5.0. Constant LR 5e-6, no warmup, group
  rewards, `dr_grpo`, beta=0, truncation masking, 1536 completion tokens,
  temperature 0.8, top-p 0.95; effective configuration saved in `config.json`.
- `G=4`, microbatch=2, accumulation=8: four QA / 16 rollouts per optimizer update.
  joint20 = 5 updates/epoch = 200 updates; joint44 = 11/epoch = 440 updates.
  Same epoch budgets mean equal visits per QA, not equal optimizer-update budgets.
- Randomized native sampling; actual IDs are logged. Every complete epoch is
  checked to contain every QA exactly once. The 20-QA subset is the existing
  retrospectively selected improved20 set, not a new random sample of the 44.
- Gradient readout is at the end of the eighth `training_step`, after native
  backward, BEFORE returning to the outer Trainer loop's clipping and update.
  `on_pre_optimizer_step` is not used for readout because it is after clipping.
  Native loss already handles accumulation scaling: do not divide again.
- `.grad` is only copied, never zeroed, normalized or overwritten by diagnostics.
  None gradients occupy zeros in a fixed named trainable-parameter order. There
  is no extra differentiation. The normal training loop performs its own updates.
- Copying uses one parameter at a time, avoiding a second full GPU gradient.
  CPU float32 history defaults to 11 steps for BOTH datasets. CPU memory is about
  `4 * trainable_parameter_count * (history_steps + 1)` bytes for vectors; dot
  products accumulate in float64. `--history-steps` changes the bounded window.
  Vectors are not written to disk, so larger windows cannot be recovered later.
- Each captured batch is compared with all available prior batches in the
  window. No complete 200x200/440x440 matrix is claimed. NaN cosine means either
  norm < eps (default 1e-12); nonfinite gradients fail the run.
- Native first-use frozen visual bias casts are audited using the joint trainer's
  policy. No unsolicited base dtype conversions or alternate loading path.
- Existing training files are unchanged. Process-local substitutions adapt the
  monolithic baseline configuration and Trainer class. A private completion
  exception exits before its final weight/state saving block. Original monitor
  callback still records `step_metrics.jsonl`; its norms are post-clipping,
  whereas `batch_metrics.csv` is authoritative for pre-clipping gradient geometry.

## Outputs

| File | Contents |
|---|---|
| `config.json` | Versions, source/data hashes, effective config, parameter inventory, status and budget |
| `batch_metrics.csv` | Actual QA IDs, loss, rewards, signal count, pre-clip norm, valid mask, template mix, GPU peaks |
| `gradient_pairs.csv` | Current/previous step, lag, shared QA count, cosine, dot product, distance, valid-pair mask |
| `qa_metrics.csv` | Rewards per group, QA visit number/gap, country/template/ability/answer-type metadata |
| `lag_summary.csv` | Valid pair counts, mean cosine, negative fraction per lag (generated by plotting) |
| `qa_summary.csv` | Early/late visit reward means, active fractions and actual visit gaps |
| `plots/training_overview.png` | Online rewards, gradient norms, signal and negative-cosine fractions |
| `plots/qa_learning_heatmap.png` | Per-QA reward across visits |
| `plots/gradient_lag_heatmap.png` | Step x lag cosine, fixed [-1,1], unavailable cells grey |
| `plots/gradient_by_lag.png` | Cosine/negative fraction with valid comparison counts |
| `plots/question_group_learning.png` | Existing ability-level or answer-type reward curves |

The multi-run plot command additionally creates `joint20_vs_joint44.png`, comparing
rewards by epoch and step and negative cosine rates at the same step lags.
Question-group plots are descriptive; no mixed-batch gradient is attributed to
an individual task. Use QA IDs/template compositions and pair shared-QA counts
for follow-up analysis. Different answer types do not establish different tasks.

## Interpretation and verification limits

These are `g_Bt(theta_t)` versus `g_Bs(theta_s)`: **different batches AND different
parameter states**. Negative cosine is not evidence by itself of fixed-state
conflict, causal forgetting, or a task incompatibility. Reward curves use four
online sampled responses per QA visit, not independent greedy evaluation.
Inactive groups remain in the batch; zero gradients are unavailable rather than
artificial zero cosines. Active rewards do not guarantee nonzero gradients after
truncation masking. Pair statistics use only valid gradients and report counts.

The 20-QA selection, different update budgets, changing batch composition, visit
gaps and sampling noise confound causal claims. Forty epochs and one seed are an
exploratory run, not a stability/significance guarantee.

`python -m pytest diagnostics/test_batch_gradients.py -q` checks geometry, bounded
history, QA coverage/gaps, zero handling, plots and read-only pre-clip capture in
a toy accumulated training loop. It does not validate CUDA, real GRPO backward,
container compatibility or GPU peak memory; run the one-epoch Euler smoke test.
