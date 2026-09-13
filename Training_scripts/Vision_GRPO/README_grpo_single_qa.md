# Single-QA fresh-rollout experiment

Tests whether the local improvement from diagnostic 14037996 accumulates into
reliable answers on the same QA: USA map_15117, source_index 2051, template 3,
"What is the overall range of the legend?", ground truth "0-5".
The tracked one-row JSON is an exact subset of the mixed44 dataset.

`train_mapwise_grpo_single_qa.py` wraps the unchanged frozen Vision baseline.
Raw Qwen3-VL-8B-Thinking, fresh Vision-block LoRA r16/alpha16, native TRL 1.12.0,
existing quantization/BF16 behavior, binary exact reward, beta0, dr_grpo loss,
truncated-completion masking. BS2/GA2/G4, constant lr5e-6, zero warmup, 40 updates.
Each update uses four NEW samples; num_iterations1, no replay, no filtering of
all-correct/all-wrong groups. Does not initialize from the one-update adapter.
This is the same diagnostic optimization setup extended over new samples,
not a matched-budget comparison against the BS2/GA8 mixed44 runs.

All eight checkpoints (5,10,...,40) are retained plus final_adapter.
`fresh_group_history.jsonl` records every group's pre-update rewards and advantages.
`sampling_windows.json` aggregates five groups (20 responses) per window; retains
all-wrong and all-correct groups. Samples are correlated within groups and the
policy changes during a window; these are descriptive training statistics.

Euler job `train_grpo_single_qa.sbatch`, stored outside the repo under
Euler_results/VisionLoRA/debug50/jobs, performs:

1. Dependency-free subset/window tests.
2. Raw baseline greedy inference on the one QA.
3. Forty native GRPO optimizer updates with a new W&B run.
4. Sequential independent greedy inference and exact scoring for baseline,
   all eight checkpoints, final_adapter; CSV is written incrementally.

Greedy evaluation is post-training, not an online callback. It reports correctness
0/1 (accuracy 0/100%), generation length and truncation. Inspect checkpoint
predictions for the actual answer. A single correct greedy answer does not prove
stable mastery; compare several checkpoints and sampling windows. Baseline may
already answer correctly, so stability and fresh-sampling success are essential.

Outputs: `$SCRATCH/thesis/runs/grpo_single_src2051_JOBID/`.
Evaluation summary: `checkpoint_eval/checkpoint_accuracy.csv`.
CSV status must be ok; final_adapter duplicates the last checkpoint, not an
independent replication. The eight-hour allocation includes all evaluation.
The inherited console epoch estimate is not authoritative: actual config and
corrected training_metrics specify 40 optimizer updates / 160 fresh responses.

Sync Python, README, test and the one-QA JSON through GitHub. Execute local
`upload_grpo_single_qa.ps1` to upload ONLY the scratch job. In Euler Bash:

```bash
cd "$HOME/VLM_adaptation"
git pull --ff-only
sbatch --output="$SCRATCH/thesis/logs/%x-%j.out" --error="$SCRATCH/thesis/logs/%x-%j.err" "$SCRATCH/thesis/jobs/train_grpo_single_qa.sbatch"
```

Local subset/window tests and Python/PowerShell syntax checks can run without
GPU dependencies. Native TRL/GPU integration remains to be verified on Euler.
