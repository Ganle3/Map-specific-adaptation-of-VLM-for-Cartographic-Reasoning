# GRPO When--Where Controller

> **Status: compatibility gate pending.** The current token-step prototype is
> disabled by default because it has not yet reproduced the validated
> `continuous_cache_qwen3vl.py` trajectory. It must not be used for an
> experiment or reported result before the deterministic compatibility suite
> passes.

This directory trains only the small visual-observation controller. Qwen3-VL,
the existing reasoning adapter, proposal extraction, crop encoding, anchor KV,
replay tail, and visual-key biases remain frozen.

The first run should be deliberately small because each rollout is token-boundary
controlled and therefore much slower than the existing windowed baseline:

```powershell
python .\Training_scripts\GRPO_lookbetter\train_when_where_grpo.py `
  --train-json .\Datasets\Subset_MapVerse\scaling_500\mapverse_scale500_train.json `
  --image-root C:\Users\junyhuang\Thesis\MapVerse\data\imgs `
  --adapter .\Training_outputs\MapVerse_scaling1000_GRPO_from_MapVerse5000_token500\checkpoint-2500 `
  --output-dir .\Training_outputs\when_where_smoke `
  --max-examples 2 --group-size 2 --epochs 1 --max-reasoning-tokens 96
```

Run a cold-start pilot before training by setting `--epochs 1 --learning-rate 0`
and inspect `trajectories.json`: groups need mixed answer rewards for GRPO to
have signal.  The controller checkpoint is separate from the reasoning adapter.

`tau0` is calibrated from token-level JSD when omitted. This calibration is
expensive; use `--tau0` only after recording and retaining a pilot value.
