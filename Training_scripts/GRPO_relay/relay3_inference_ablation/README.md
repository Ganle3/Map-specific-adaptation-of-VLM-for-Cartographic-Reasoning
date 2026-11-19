# Relay3 inference ablation

This experiment loads one trained Relay3 checkpoint and changes only the
inference information path. It does not retrain any parameter.

| Variant | Change |
|---|---|
| `full` | Exact native Relay3 decoder |
| `relay_off` | Same Relay3-trained LoRA adapter, no relay |
| `no_image_read` | Relay cannot query image KV; `x_t` retains native image access |
| `no_current_x_read` | Relay cannot query current `x_t`; `x_t` still queries relay |
| `static_relay` | Compute relay at the first decoding step, keep its per-layer KV, and never refresh it |

Primary paired contrasts:

- `full - no_image_read`: extra online visual read.
- `full - no_current_x_read`: reasoning-state-conditioned visual read.
- `full - static_relay`: per-step dynamic refresh.
- `full - relay_off`: complete online relay contribution under the jointly
  trained Relay3 adapter.

Run a short local smoke on the first two rows of each test set:

```powershell
& ".\VLM_adaptation\Training_scripts\GRPO_relay\relay3_inference_ablation\run_step4000_local.ps1" `
  -EndIndex 2
```

Run the complete local evaluation (resume-safe):

```powershell
& ".\VLM_adaptation\Training_scripts\GRPO_relay\relay3_inference_ablation\run_step4000_local.ps1"
```

Euler:

```bash
sbatch /cluster/scratch/junyhuang/thesis/jobs/evaluate_relay3_inference_ablation_step4000_4090.sbatch
```

