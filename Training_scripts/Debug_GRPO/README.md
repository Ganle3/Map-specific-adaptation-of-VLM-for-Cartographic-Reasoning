# Vision GRPO experiments

This directory contains the maintained training and diagnostic entry points for
MapWise GRPO. Historical scripts remain when they are needed to reproduce an
existing run; do not edit them to change a new experiment.

## Maintained joint debug entry point

Use `grpo_joint_debug.py` for new joint experiments. It shares one implementation
for any validated dataset of 4, 20, 44, or a future size. The sbatch file supplies
the dataset and update budget; the Python entry point supplies the common model,
LoRA scope, GRPO objective, auditing and constant-LR diagnostic defaults.

Current joint scope `joint` adapts 368 projections across vision blocks,
mergers/deepstack mergers and language attention/MLP. Each QA has four fresh
rollouts per update; the batch contains four QAs. The script checks that every
batch has four rollouts per QA and that all QAs belong to the selected dataset.

Examples:

```text
4 QAs:  --qa-json .../mapwise_grpo_joint_debug4_src2051.json --max-steps 120
20 QAs: --qa-json .../mapwise_grpo_joint44_improved_20.json --max-steps 600
44 QAs: use the validated 44-QA file and the desired budget
```

The default diagnostic uses raw Qwen3-VL-8B-Thinking, rank/alpha 16, native
generation, binary exact reward, beta 0, truncated-completion masking, constant
learning rate 5e-6 and zero warmup. CLI arguments remain authoritative for
dataset, steps, learning rate, batch size, generation length and seed. A new run
must use a new output directory and cannot resume an older adapter.

`grpo_joint_debug.py` is the single implementation for joint debug runs. Pass a
different `--qa-json` to select 4, 20, 44, or another validated QA set. The
optional `--off-policy-mask-threshold 0.2` or `0.4` enables the TRL mask ablation;
omit it for the original behavior. `--max-completion-length 500` is likewise a
CLI override and does not require a second training script.

## Evaluation and audits

Use `evaluate_grpo.py` for the standard greedy evaluation of any completed joint
run. Provide `--run-dir` and the matching `--qa-json`; the evaluator discovers
available checkpoints and writes results under `evaluation_greedy/`. Optional
sampled draws are a diagnostic only and do not change the greedy criterion.

Report greedy (`do_sample=False`) accuracy as the deployment criterion. Sampled
accuracy is a diagnostic of probability mass and does not replace greedy evaluation.
Loading audits record adapter tensors, dtype preparation and active adapters;
training audits record optimizer membership, parameter changes and quantized
bias first-use casts. The bitsandbytes frozen visual bias FP32→BF16 cast is an
allowed first-use conversion; other unexpected dtype changes fail the run.

## Historical experiment families

- `train_mapwise_grpo_visLoRA_trl.py`: original baseline; frozen for comparison.
- `train_mapwise_grpo_visLoRA_iterations_trl.py` and
  `train_mapwise_grpo_visLoRA_replay_trl.py`: A/B reuse experiments.
- `train_mapwise_grpo_scope_ablation.py`: vision rank/merger/language scope runs.
- `train_mapwise_grpo_single_qa.py` and `train_single_qa_scope.py`: single-QA
  diagnostics and scope comparisons.
- `diagnose_grpo_one_update.py`: fixed rollout and local update audit.

The old families have separate historical configurations and should not be
silently merged into results from `grpo_joint_debug.py`.

## Local checks

Run the relevant standard-library checks before pushing:

```text
python Training_scripts/Debug_GRPO/test_grpo_joint_debug.py
python -m py_compile Training_scripts/Debug_GRPO/grpo_joint_debug.py
```

Windows upload scripts normalize sbatch files to LF and verify remote hashes and
`bash -n`. Git synchronizes repository code; scratch job files are uploaded by
their corresponding PowerShell script.
