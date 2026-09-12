# Debug44: LoRA capacity/scope, plain GRPO

Entry: `train_mapwise_grpo_scope_ablation.py --scope-experiment NAME`.
Uses the unchanged `_vislora_grpo_baseline_snapshot.py` as the common training loop.
Does not import A/B trainer overrides or edit either original training script.

| NAME | Scope | rank/alpha | Target modules |
| --- | --- | --- | --- |
| vision_r64 | 27 visual blocks, attention qkv/proj + MLP fc1/fc2 | 64/64 | 108 |
| vision_merger_r16 | Same blocks + main merger and 3 deepstack mergers, fc1/fc2 | 16/16 | 116 |
| language_r16 | 36 language layers, attention q/k/v/o and MLP gate/up/down | 16/16 | 252 |

Language scope matches `train_mapwise_grpo_LanLoRA_trl.py`; using the common frozen
loop avoids importing unrelated differences from that older training script.
The full model stays quantized/frozen; only the specified LoRA matrices train.
Merger scope does not unfreeze its biases/norms. Module inventory is checked
against the actual loaded model; incompatible model layouts stop explicitly.

All three: original Qwen/Qwen3-VL-8B-Thinking start, same mixed44 QA order, no
init/resume, binary correctness, beta0, dr_grpo, group scaling, num_iterations1,
no historical replay/entropy selection. Single A100, BS2 GA8 G4, seed3407,
lr5e-6 cosine warmup0.05, max completion1536, 220 optimizer updates (~20 passes).
Keep the batch/quantization settings even though A100 has more memory.
Presets set rank AND alpha, overriding any separately provided rank flags.

Saving/evaluation: every20 updates, keep12; baseline50 first, then training, then
all checkpoints and final adapter greedy evaluation. Partition 44/6/50, output
`checkpoint_eval/checkpoint_accuracy.csv`. The entire sequence shares a24h job.
GPU type default is Euler `a100` (40GB); optionally override all three submissions
with `--gpus=a100_80gb:1` for 80GB. See
https://docs.hpc.ethz.ch/hardware/gpu_nodes/ .

Runtime evidence: `lora_scope_inventory.json` lists targets and parameter counts;
`optimizer_membership.json` checks every trainable tensor is in the optimizer and
records dtype after Trainer initialization; `parameter_update_audit.jsonl` records
changed elements and maximum deltas for the first five updates. Zero changes
can occur on signal-free batches; these checks alone do not prove functional
probability improvements or correct checkpoint reload. `effective_grpo_config.json`
and source/data/scorer hashes in `run_config.json` make settings reviewable.
Inherited console banners may still say Vision-only; the inventory and scope
manifest are authoritative for these new runs.

Interpretation: scopes do not match trainable parameter counts. Improvements in
merger/language arms support their whole configuration, not location alone.
The old vision-r16 control ran on4090; comparisons across hardware are not a
strict hardware-controlled ablation. These are initial overfit diagnostics.

## Sync and launch

Repository files: Windows Git commit/push, Euler `git pull --ff-only`.
Only scratch job/helpers are uploaded by the LOCAL PowerShell command:

```powershell
& "C:\Users\junyhuang\Thesis\Euler_results\VisionLoRA\debug50\jobs\upload_debug44_scope.ps1"
```

Inside Euler SSH after repository sync:

```bash
sbatch --output="$SCRATCH/thesis/logs/%x-%j.out" --error="$SCRATCH/thesis/logs/%x-%j.err" "$SCRATCH/thesis/jobs/train_debug44_scope.sbatch" vision_r64
sbatch --output="$SCRATCH/thesis/logs/%x-%j.out" --error="$SCRATCH/thesis/logs/%x-%j.err" "$SCRATCH/thesis/jobs/train_debug44_scope.sbatch" vision_merger_r16
sbatch --output="$SCRATCH/thesis/logs/%x-%j.out" --error="$SCRATCH/thesis/logs/%x-%j.err" "$SCRATCH/thesis/jobs/train_debug44_scope.sbatch" language_r16
```

Each job requests ONE GPU, creates a fresh W&B run in MapWise-GRPO-Scope44 and
uses `$SCRATCH/thesis/runs/scope_debug44_NAME_mixed_lr5e6_JOBID`. Jobs are independent.
No separate evaluation submission needed. Local validation covers scope tests,
Python syntax and uploader parsing; real A100 training has not been executed locally.
