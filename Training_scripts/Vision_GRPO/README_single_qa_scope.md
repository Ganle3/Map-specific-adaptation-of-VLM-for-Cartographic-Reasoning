# Single-QA Language and joint LoRA

`train_single_qa_scope.py --scope language_r16|joint_r16` wraps the unchanged
single-QA fresh-GRPO script and frozen baseline. Original experiments remain
unchanged. Same source2051 QA, raw base (no checkpoint initialization), rank16,
alpha16, BS2/GA2/G4, 40 updates, constant 5e-6, no warmup, same seed3407,
binary reward and truncation masking, same inherited quantization/dtype.

- language_r16: 252 language attention q/k/v/o and MLP gate/up/down projections.
- joint_r16: those 252 + 108 vision attention qkv/proj and MLP fc1/fc2 + 8
  merger/deepstack merger fc1/fc2 projections = 368 modules.

Joint means LoRA on both sides and connectors, NOT full-parameter finetuning.
Base parameters, norms, embeddings and output head remain frozen. Counts differ,
so this compares practical configurations, not parameter-count-matched location
effects. Gradient existence does not guarantee nonzero gradients each step:
zero-initialized LoRA B and zero-advantage groups can produce zero gradients.
Audits record optimizer registration/dtypes plus individual parameter gradients
and actual changes on steps 1,5,20,40. Inventory must match exact target names.
Inherited Vision-only banners are generic; inventory and run_config identify scope.

Scratch job `train_single_qa_scope.sbatch` takes the scope as its first argument.
Defaults to one RTX4090, 12h including baseline greedy, training, greedy scoring
of every checkpoint (5..40) and final adapter, then raw baseline/checkpoint40
100-sample comparison using seeds900000..900099 and existing sampling settings.
Models load sequentially. Joint-model 4090 peak memory is not locally verified;
do not silently change batch size if it OOMs, use a larger GPU for matched settings.

Output: `$SCRATCH/thesis/runs/grpo_single_SCOPE_src2051_JOBID/`
- `checkpoint_eval/checkpoint_accuracy.csv`
- `sampling_eval/sampling_accuracy.csv`, `sampling_eval/comparison.json`
- `sampling_windows.json`, `scope_update_audit.jsonl`, `lora_scope_inventory.json`

Sync repository through GitHub. Local PowerShell uploader
`Euler_results/VisionLoRA/debug50/jobs/upload_single_qa_scope.ps1` only uploads
the scratch sbatch, not repository code. Submit two independent jobs after pull:

```bash
sbatch --output="$SCRATCH/thesis/logs/%x-%j.out" --error="$SCRATCH/thesis/logs/%x-%j.err" "$SCRATCH/thesis/jobs/train_single_qa_scope.sbatch" language_r16
sbatch --output="$SCRATCH/thesis/logs/%x-%j.out" --error="$SCRATCH/thesis/logs/%x-%j.err" "$SCRATCH/thesis/jobs/train_single_qa_scope.sbatch" joint_r16
```

Each gets a new W&B run. Sampling is post-training and its CSV, not training
reward, is the principal comparison with the previous Vision run. Single-QA
results do not establish multi-QA or held-out generalization.
