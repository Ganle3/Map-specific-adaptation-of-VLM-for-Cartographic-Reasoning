# Four relay parameters across decoder depth

All three experiments use one `RelayModule.prototypes` parameter with shape
`[4, H]`. They use the same NF4 Qwen base, MapWise filtered 1058 training
data, GRPO reward, optimizer, rollout count, and evaluator. Each starts from
`mapwise_scaling1000_iter2_mask0p2_retry/checkpoints/checkpoint-3000`, expands
its rank-16 LoRA adapter to rank 32 with seed 1234, and initializes the relay
parameters with `xavier_normal`. This matches the starting configuration of
`train_relay_scheme3_r2_mapwise1058_xavier_rank32_q4r4_eval_4090.sbatch`.

| `--relay-layer-mode` | Relay state in each decoder layer |
|---|---|
| `continuous` | Four relays evolve continuously through all layers. |
| `reset_shared4` | Four relays are reset from the same four parameters at the start of layers 1, 9, 17, and 25. |
| `reset_stage1` | One relay is used per layer; layers 1–8 use parameter row 1, layers 9–16 row 2, layers 17–24 row 3, and layer 25 onward row 4. |

Layer numbering in this table is one based. The implementation uses zero based
layer indices `(0, 8, 16, 24)`. The final segment includes every remaining
decoder layer. At a segment boundary, only relay hidden states are reset;
language states continue. At the end of each decoding step, temporary relay KV
is cropped as in the original Relay3 implementation.

`continuous` versus `reset_shared4` isolates relay state accumulation across
layers while keeping both parameter count and per-layer relay width equal.
`reset_shared4` versus `reset_stage1` compares four parallel relays per layer
with one depth-specific relay per layer. The latter also has lower attention
compute, so parameter equality does not imply compute equality.

The three sbatch files in `Euler_results/VisionLoRA/debug50/jobs` call the
shared `run_euler.sh`. Each trains to 4000 optimizer steps and evaluates greedy
accuracy on the training set at 0 and 4000, and on MapWise test400 and MapVerse
test600 at 0, 2000, and 4000. The initial evaluation uses the original adapter
without randomly initialized relays. Evaluation rows include inference time.

After syncing the updated repository and copying the sbatch files to
`/cluster/scratch/junyhuang/thesis/jobs`, submit normally:

```bash
JOBS=/cluster/scratch/junyhuang/thesis/jobs
sbatch "$JOBS/train_relay3_r4_continuous_mapwise1058_rank32_4090.sbatch"
sbatch "$JOBS/train_relay3_r4_reset_shared4_mapwise1058_rank32_4090.sbatch"
sbatch "$JOBS/train_relay3_r4_reset_stage1_mapwise1058_rank32_4090.sbatch"
```

To have all three wait until **all** specified existing jobs finish
successfully, submit now with a Slurm dependency. Replace the example IDs:

```bash
WAIT_FOR=afterok:12345678:12345679
sbatch --dependency="$WAIT_FOR" "$JOBS/train_relay3_r4_continuous_mapwise1058_rank32_4090.sbatch"
sbatch --dependency="$WAIT_FOR" "$JOBS/train_relay3_r4_reset_shared4_mapwise1058_rank32_4090.sbatch"
sbatch --dependency="$WAIT_FOR" "$JOBS/train_relay3_r4_reset_stage1_mapwise1058_rank32_4090.sbatch"
```

`afterok` waits for successful completion. Use `afterany` if the new work
should start even when a predecessor fails. Slurm holds pending jobs without
allocating their GPU/CPU resources. For a calendar start, Slurm also supports
`sbatch --begin=YYYY-MM-DDTHH:MM ...`, though dependencies are better when the
actual finishing time is uncertain.
