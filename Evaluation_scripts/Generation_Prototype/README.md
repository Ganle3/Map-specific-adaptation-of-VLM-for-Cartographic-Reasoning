# Unified MapWise + MapVerse trajectory generation

Submit on Euler from the repository checkout:

```bash
sbatch Evaluation_scripts/Generation_Prototype/generate_mapwise_mapverse_8rollouts.sbatch
```

The job generates eight independent sampled trajectories per question with the
GRPO settings `temperature=0.8`, `top_p=0.95`, `top_k=0`, and 1536 maximum new
tokens. Both datasets use the same question and trajectory fields.

Output is written under
`$SCRATCH/thesis/runs/unified_mapwise1000_mapverse1000_checkpoint3000_rollouts8`:

- `rollouts.jsonl`: append-only recovery journal, written after every rollout.
- `unified_trajectories.json`: unified, readable snapshot, updated every 25
  questions and at completion. Each question has `rollout_outcome` equal to `all_correct`,
  `all_incorrect`, `mixed`, or `incomplete`.
- `manifest.json`: immutable input hashes and generation settings.
- `completed.json`: written only after all requested trajectories finish.

The run is safe to resume: resubmit the same job and already journaled rollouts
are skipped. Do not change settings or inputs in an existing output directory;
the manifest check intentionally rejects that.

For a quick end-to-end test, copy the Python command from the job and add
`--limit 1` while using a new output directory.
