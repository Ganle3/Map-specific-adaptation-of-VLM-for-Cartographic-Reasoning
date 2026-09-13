# Fixed-group GRPO update diagnostic

Entry: `diagnose_grpo_one_update.py`. Keeps the original training scripts unchanged.
Uses `_vislora_grpo_baseline_snapshot.py` and native TRL 1.12.0 training loop.
Default experiment: raw Qwen3-VL-8B-Thinking, Vision blocks r16/alpha16,
same binary scorer and quantization. No SFT or replay.

## Deliberate diagnostic differences

- One QA, four responses, microbatch 2, accumulation 2, exactly one optimizer step.
- Constant learning rate 5e-6, no warmup (avoids testing a zero-LR first step).
- Search at most 12 QA groups in fixed order for binary mixed rewards and all
  four untruncated completions. Unsuccessful search performs no update.
- Native TRL loss, optimizer creation, gradient clipping, backward and optimizer
  step are retained. Before-update log probabilities anchor the fixed PPO ratio.
- Fixed-batch measurements use eval mode, no gradients, microbatch 2, and the
  training temperature. These are not top-p-renormalized sampling probabilities.
- Repeat the unchanged-model forward to quantify numerical variation.
- Save through the baseline final-adapter path, then load those files into a
  separate PEFT adapter on the same base model and match trained parameter dtype.
  This is not yet a fresh-process/base-model reload test.
- No W&B or 50-QA accuracy sweep: output is diagnostic evidence, not accuracy.

## Outputs

- `effective_grpo_config.json`, `run_config.json`: actual one-update configuration.
  The inherited console epoch estimates and final training_metrics estimates
  are baseline labels; `result.json` records the actual optimizer step count.
- `candidate_search.json`: accepted/rejected QA groups with actual rewards.
- `fixed_rollouts.json`: fixed prompt/response IDs, masks, response texts, actual
  binary rewards and advantages; row order matches result probability arrays.
- `fixed_batch.pt`: CPU copy including image tensors. Can be large.
- `token_logps.pt`: per-token before/after/reloaded log probabilities.
- `result.json`: per-response sum/mean log probabilities, changed parameter
  counts and norms, actual optimizer LR, clipped objective before/after,
  advantage-weighted log-probability change, repeated-forward and reload errors.

Larger clipped objective is better (loss is its negative). A positive
advantage-weighted change is local evidence, not proof that greedy accuracy
will improve. Do not require every positive-reward trajectory to improve:
parameters are shared. Evaluate effect sizes against repeated-forward noise.
No automatic pass threshold is imposed. A successful local check should be
followed by a separate single-QA fresh-rollout experiment, not presumed success
on all 44 questions.

## Synchronization and job

Commit/push these Python files, this README and the test through GitHub; on Euler
run `git pull --ff-only`. Local Windows jobs are outside this repository:
`Euler_results/VisionLoRA/debug50/jobs/upload_grpo_one_update.ps1` uploads only
`diagnose_grpo_one_update.sbatch` to scratch. Execute the PowerShell script locally.
In Euler Bash submit:

```bash
sbatch --output="$SCRATCH/thesis/logs/%x-%j.out" --error="$SCRATCH/thesis/logs/%x-%j.err" "$SCRATCH/thesis/jobs/diagnose_grpo_one_update.sbatch"
```

Optional positional argument is the candidate start index (default 0). A fresh
run is required for any retry. Output: `$SCRATCH/thesis/runs/grpo_one_update_JOBID/`.
The A100 job runs `test_grpo_one_update.py` before model loading; it checks the
objective gradient signs, padding mask, clipping saturation and zero-signal case.
GPU integration must still be checked in the actual container.
