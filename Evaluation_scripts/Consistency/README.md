# Adapted-model rollout consistency

`evaluate_mapwise_rollout_consistency.py` evaluates a PEFT-adapted MapWise
model on a caller-supplied Test JSON.  It deliberately has no default Test
split path: pass `--qa-json` and `--image-root` explicitly.

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

For Euler, export `TEST_QA`, `MAPWISE_IMAGES`, `ADAPTER`, `OUTPUT`, and `SIF`,
then submit `evaluate_mapwise_rollout_consistency.sbatch`.
