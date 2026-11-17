# Relay3 population Q/K analysis

This analysis tests whether the **language hidden query after Relay3 interaction**
is better aligned with cached image keys than the matched adapter-only hidden
query, and whether that alignment is associated with shorter correct reasoning.

## What is captured

For every native greedy inference step, decoder layer and question:

- actual post-RoPE hidden query `Q`;
- actual cached image keys `K_img`;
- nearest-patch, top-5%-patch and mean cosine distance;
- image-conditional attention-weighted cosine distance;
- total full-cache attention mass assigned to image positions.

The collector leaves SDPA enabled. It stores scalar trajectories and fixed-step
64-dimensional random projections instead of full patch maps. Projection is
only used for cluster figures; reported distances are computed in the original
head space.

## Controlled comparison

The default Euler job compares checkpoint 4000 for:

1. adapter-only;
2. Relay3 adapter plus its one temporary dynamic relay.

Each model follows its own greedy trajectory. Absolute-step bins therefore test
whether the complete Relay3 system acquires image-aligned hidden queries earlier
in its own answer. The analysis does not claim that the single relay parameter
alone caused the difference because Relay3 and its LoRA adapter were jointly
trained.

## Euler

```bash
sbatch /cluster/home/junyhuang/VLM_adaptation/../Euler_results/VisionLoRA/debug50/jobs/capture_analyze_relay3_global_qk_step4000_4090.sbatch
```

If the `Euler_results` directory is not synced under `$HOME`, submit the local
job copy from the location used for the other Thesis jobs.

Default output:

```text
/cluster/scratch/junyhuang/thesis/analysis/relay3_global_qk_step4000/
```

The job is restartable: completed question files are skipped. Set `OUTPUT` to a
new directory for an independent run.

## Main outputs

- `early_qk_distance_by_layer.png`
- `paired_distance_improvement_step_layer.png`
- `global_hidden_query_image_key_clusters.png`
- `early_alignment_vs_token_saving.png`
- `ability_level_accuracy_alignment_efficiency.png`
- `per_question_efficiency.csv`
- `ability_summary.csv`
- `report.md`

