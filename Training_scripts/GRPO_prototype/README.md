# Nine-grid region policy experiment

## Stage A: code review

- `live_crop_append_interactive.py` commits each emitted token into the live Qwen3-VL KV, then appends a crop as a new multimodal user segment. Its prefill and append use `_segment_positions`; this implementation uses the same helpers and preserves the existing full-image KV.
- With `output_hidden_states=True`, `output.hidden_states[-1][:, -1, :]` is the final normalized decoder state associated with `output.logits[:, -1]`. At a look point, the just emitted token is first committed; that forward's hidden state predicts the next token and is used for the region decision. This needs confirmation against the exact Transformers release on the GPU machine.
- Debug_GRPO uses TRL's language completion loss. It does not optimize region actions automatically. This directory implements a separate PPO clipped GRPO surrogate over region log probabilities only. Advantages come from within-question final-answer rewards. Each trajectory averages its own region decisions, then trajectories are averaged; language token length never enters.
- The exact MapWise evaluator and prompt are imported from existing project code. The PEFT adapter and backbone are frozen.

## Stage B: live inference

`region_policy.py` defines nine shared action prototypes and nonoverlapping row-major image boxes. `live_rollout.py` preprocesses and runs the native ViT/merger on each crop once per question, then substitutes its cached vision output when the selected crop is appended. The full image, existing reasoning, and every appended crop remain in decoder KV. The look schedule is a fixed reasoning-token window via `--look-every`; no WHEN policy is trained.

## Stage C: region GRPO

`train.py` defaults to 16 trajectories per question and four questions per optimizer update (64 trajectories). It generates trajectories one at a time, so there is no 8-rollout GPU microbatch: only compact detached decision states and policy graphs are retained for the region loss. Old action log probabilities are recorded, current probabilities are recomputed from those states, and only `nine_blocks` is updated. All action and reward traces go to `rollouts.jsonl`; scalar diagnostics also go to W&B when enabled. If a group has equal rewards, its advantages and gradient are zero. Trajectories with no looks contribute no region loss.

Example on the GPU host:

```bash
python train.py \
  --dataset /path/to/visual_easier_1000.json \
  --mapverse-root /path/to/mapverse/images \
  --mapwise-root /path/to/mapwise-dataset \
  --adapter /path/to/existing/grpo/adapter \
  --output-dir /path/to/region_run \
  --limit 4 --group-size 16 --questions-per-update 4 --wandb-project mapqa-region
```

For a deterministic untrained inference pass, add `--eval-only --limit 1`. Run `python self_check.py` for geometry and optimizer checks. Loading a saved region policy uses `--policy-checkpoint /path/to/region_policy.pt`.

For the local four-question overfit experiment, `run_overfit_0245_0248.ps1` selects 1-based JSON rows 245–248, repeats them for 10 epochs, samples 16 rollouts for each question, and updates once per four-question epoch. It uses a fixed LOOK every 20 reasoning tokens with at most 9 crop appends per rollout. It also runs one deterministic region-policy evaluation on each selected question after each epoch. Its default W&B mode is online; use `-WandbMode offline` when disconnected. The generated `rollouts.jsonl` records each rollout's epoch, phase and reward, while W&B/console reports training reward, deterministic evaluation accuracy, mixed-reward question fraction, gradient, parameter change and peak VRAM. The overfit criterion is improving deterministic evaluation accuracy across epochs, not a nonzero loss alone.

## Verification status

The base Windows Python environment lacks `torch`, `transformers`, and `peft`, but the local `VisionGRPO` environment has PyTorch 2.11, Transformers 5.19, and PEFT 0.20. Syntax compilation, CLI import, and `self_check.py` completed. A local NF4 Qwen3-VL-8B-Thinking run with the existing `checkpoint-2500` adapter completed crop append and full rollout on an RTX A4500 (20 GiB). A four-question, two-rollout, 64-token training smoke completed 16 crop appends, including reuse of the same region; rewards were all zero because generations were truncated. A 512-token deterministic evaluation of rows 245–248 returned 0/4 correct. A longer 800-token probe with four sampled rollouts per question produced mixed rewards in all four groups, gradient norm 0.272 on `nine_blocks`, parameter delta norm 0.384 after one update, and 7.00 GiB peak allocated VRAM. That verifies a real reward-driven region-policy update. The epoch-0 baseline, post-epoch evaluation, and per-epoch checkpoint path also completed in a separate short smoke. These tests do not establish overfitting, which requires several epochs and the deterministic evaluation trend. W&B online upload has not been checked.
