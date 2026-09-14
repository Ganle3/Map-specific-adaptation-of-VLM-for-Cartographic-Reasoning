# Four-QA joint GRPO diagnostic

This is a new diagnostic, leaving the previous training entry points unchanged.
Start from the original Qwen3-VL-8B-Thinking model with new joint r16/alpha16
adapters on vision attention/MLP, mergers, and language attention/MLP.

## Selection

The source is joint44 run 14052496. The 20 candidates have higher correctness
in their last five visits than their first five visits. These are retrospective
observations from 20 responses per window, not statistically established stable
improvements. Rank by the minimum correctness in the two final five-visit blocks,
then combined final-ten-visit correctness, fewer truncations, and improvement.

The training set contains src2051 as an anchor plus the top three candidates:
src0525, src2021, and src0646. The anchor is NOT among the improved 20.
The strict top-four ranking is saved separately; it is not the training input.
See mapwise_grpo_joint_debug4_selection.json for provenance and full rankings.

## Protocol

- 120 optimizer updates, constant learning rate 5e-6, zero warmup.
- Fixed four-QA order; four fresh responses per QA per update.
- Device batch size 2, gradient accumulation 8, one GPU, num_iterations=1.
- No replay or SFT; maximum completion length 1536; seed 3407.
- Save every 20 updates; retain all six checkpoints.
- Verify adapter scope, optimizer registration, generation group membership,
  and parameter dtypes across generation/scoring. Allow only a first-use FP32 to
  BF16 conversion of frozen visual bitsandbytes Linear4bit biases; retain all
  other dtype-change failures. Record installed Linear4bit.forward source and
  before/after inventories. Job 14187080 showed exactly 116 such bias casts,
  with no additional changes between generation and scoring. The original
  unconditional dtype guard incorrectly rejected this library behavior.

Each QA receives 120 sampled groups. Its contribution is averaged with the other
three QAs, so this does not reproduce the single-QA experiment's update strength.
Compared with joint44, both dataset size and schedule/budget change. This tests
whether a small joint task can learn and retain answers, not which single change
causes improvement. The three controls already have high observed correctness.

## Evaluation

Submit eval_joint4.sbatch with an afterok dependency on train_joint4.sbatch.
Evaluation uses E2 loading, with greedy evaluation at baseline and every saved
checkpoint. Baseline and checkpoints 40, 80, 120 also receive 100 independent
sampled answers per QA, using the same seed list and settings across models.
This totals 1600 sampled answers plus 28 greedy answers.

Results: RUN_DIR/evaluation_e2_joint4/per_qa_accuracy.csv.
Each checkpoint directory retains responses.jsonl, loading audits, and paired
sampling comparisons. Interpret per-QA changes, especially src2051, alongside
retention on the other three QAs; aggregate greedy accuracy can conceal changes.

Repository code/data are synchronized through Git. The local upload_joint4.ps1
uploads only the two scratch sbatch files and checks their hashes and Bash syntax.
CPU checks: python Training_scripts/Vision_GRPO/test_joint4_setup.py.
GPU execution must be validated on Euler; local CPU checks do not establish it.
