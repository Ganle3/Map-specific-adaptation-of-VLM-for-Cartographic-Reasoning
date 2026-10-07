# Adapted-model rollout consistency

The submitted-test route is `evaluate_mapwise_rollout_consistency.sbatch`.
It evaluates the MapWise-3000 adapter on both established held-out splits:
MapWise-400 and MapVerse-600. Its data locations mirror the existing jobs but
may be overridden with `MAPWISE_QA`, `MAPVERSE_QA`, `IMG_MAPWISE`, and
`IMG_MAPVERSE` at submission.

```bash
python evaluate_mapwise_rollout_consistency.py \
  --qa-json /path/to/mapwise_test.json \
  --image-root /path/to/mapwise-dataset \
  --adapter-path /path/to/checkpoint-3000 \
  --output-dir /path/to/consistency_results
```

The default protocol matches the MapWise GRPO training and submitted rollout
job: four stochastic rollouts per question, `temperature=0.8`, `top_p=0.95`,
`top_k=0`, `max_new_tokens=1536`, and seed `3407`.  The script rejects values
other than four rollouts to keep result files directly comparable.

The job delegates rollout generation to the existing
`Training_scripts/Debug_GRPO/evaluate_grpo.py` (`--sampling-draws 4` and
`--skip-greedy`), so loader, adapter handling, prompts, decoding, and strict
test scoring stay identical to the established evaluation pipeline.
`analyze_grpo_rollout_consistency.py` then only aggregates the saved
`responses.jsonl`; it is the canonical consistency result for the job.

The standalone `evaluate_mapwise_rollout_consistency.py` remains available for
an explicitly supplied MapWise Test JSON and image root, without hard-coded
dataset paths.

Outputs are resumable:

- `rollouts.jsonl`: durable per-rollout raw responses and strict-exact scores.
- `question_metrics.json`: all four rollouts grouped by question, including
  answer counts, consistency, and entropy.
- `summary.json`: aggregate metrics.

`rollout_accuracy` is strict-exact accuracy over all four rollouts.
`pass_rate` is the fraction of questions with at least one strictly correct
rollout (best-of-four). `answer_consistency` is the mean fraction agreeing
with a question's modal canonical answer. `answer_entropy_bits` is the mean
per-question Shannon entropy over canonical answers (lower is more
consistent).  Canonical answers and correctness are provided by the same
strict-exact evaluator used for GRPO.

For Euler, submit `evaluate_mapwise_rollout_consistency.sbatch`. The adapter is
preconfigured as `mapwise_scaling1000_iter2_mask0p2_retry/checkpoint-3000`.
