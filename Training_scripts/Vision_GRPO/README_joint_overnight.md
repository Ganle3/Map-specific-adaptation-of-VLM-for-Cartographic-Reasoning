# Joint overnight experiments

Independent new jobs; previous entry points and results remain unchanged.
Both use raw Qwen3-VL-8B-Thinking and fresh rank16/alpha16 joint LoRA on all
368 previously verified vision/merger/language attention and MLP projections.
No replay, num_iterations1; quantization and optimizer inherited unchanged.

## Single QA: 120 updates

Scratch job `train_single_joint120.sbatch` uses the existing single-QA scope entry
point with max_steps120. Same BS2/GA2/G4, constant lr5e-6, no warmup, seed3407.
Starts over from the raw base, NOT continuation with restored optimizer/RNG.
Since schedule is constant, increasing max_steps does not alter the intended
early learning-rate schedule; exact replay of an old run is not guaranteed.
It retains checkpoints every5 steps through120. Baseline/all checkpoints/final
greedy evaluation follows training. Fixed checkpoint40,80,120 each receive
100-sample evaluation against freshly loaded baseline on the same seeds900000–900099.
Baseline is recomputed for each comparison; these are not independent replications.
Principal outputs: `sampling_eval_step40`, `sampling_eval_step80`, `sampling_eval_step120`.
Do not select the maximum score and treat it as an unbiased final estimate.
The inherited scope audit samples only steps1,5,20,40; no later update audit.

## Mixed44: 20 epochs / 220 updates

`train_mapwise_grpo_joint44.py` is an isolated copy of the existing 44-QA scope
entry, adding the joint union and accepting only joint_r16. BS2/GA8/G4,
lr5e-6 cosine, warmup5%, 220 updates (20 passes of 44 QA), seed3407, fixed mixed
order. This matches the prior44 scope ablations, NOT the single-QA constant-LR
schedule. Scope inventory/runtime audit checks remain in place.
Saves every20 updates, limit12, retaining all11 checkpoints. Job includes raw
baseline50, post-training all-checkpoint/final greedy evaluation, with retained44
training accuracy and excluded6 separately. No 100x-per-QA sampling for this job.
Twenty epochs matches the old44 comparison; it is not longer than those runs.

Both jobs default RTX4090, 24h including evaluation. Budget is an allocation
limit, not a completion-time guarantee; mixed44 takes substantially longer.
Use a larger GPU by sbatch override if needed rather than changing batch settings.
Repo files sync through GitHub. `upload_joint_overnight.ps1` uploads/checks scratch
jobs plus the existing evaluation/partition helpers only. No jobs auto-submitted.
