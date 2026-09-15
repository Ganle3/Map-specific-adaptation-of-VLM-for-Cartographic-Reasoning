"""Small TRL 1.12 overrides for recent reuse and successful-history replay.

A uses ExGRPO Eq. 4 mixed groups and section 4.2 shaping w/(w+0.1).
Deliberate adaptations: baseline group std scaling, fixed QA exposure,
finite age/capacity, no difficulty retirement. Replay selection uses current-policy
mean full-vocabulary token entropy on the valid completion, including EOS.
Successful selection and top-p decoding make this a biased surrogate, not an
unbiased importance-sampling estimator. No SFT or cross-entropy auxiliary loss.
"""
import inspect
import json
import math
from pathlib import Path
import time

from _vislora_replay_buffer import SuccessBuffer, minimum_entropy_index


FORWARD_KEYS = ("pixel_values", "image_grid_thw", "num_images", "pixel_attention_mask",
                "spatial_shapes", "num_tiles", "image_sizes", "token_type_ids",
                "mm_token_type_ids", "image_position_ids")


def replay_correction(torch, logps, old_logps, advantages, replay_mask, completion_mask,
                      epsilon_low, epsilon_high, shaping_beta, max_length, accumulation):
    """Replace native clipped replay terms, retaining the same forward/gradient graph."""
    log_ratio = logps.float() - old_logps.float()
    if not torch.isfinite(log_ratio).all() or (log_ratio.abs() > 60).any():
        raise FloatingPointError("Non-finite/extreme policy ratio: stop rather than silently corrupt replay loss")
    ratio = torch.exp(log_ratio.clamp(-60, 60))
    clipped = torch.minimum(ratio * advantages[:, None],
                            ratio.clamp(1 - epsilon_low, 1 + epsilon_high) * advantages[:, None])
    # sigmoid(log(w)-log(beta)) == w/(w+beta), without exp overflow.
    shaped = torch.sigmoid(log_ratio - math.log(shaping_beta)) * advantages[:, None]
    mask = replay_mask[:, None] * completion_mask
    return ((clipped - shaped) * mask).sum() / (logps.size(0) * max_length * accumulation)


def make_trainers(base, options):
    torch = base.torch

    class NativeReuseTrainer(base.GRPOTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.reuse_calls = 0
            self.fresh_count = 0
            self.fresh_tokens = 0
            self.replay_count = 0
            self.discarded_count = 0
            self.optimized_count = 0
            self.started_at = time.monotonic()
            self._reuse_path = Path(self.args.output_dir) / "reuse_metrics.jsonl"
            if self.accelerator.num_processes != 1 or self.use_vllm or self.tools:
                raise ValueError("Reuse experiments support single-GPU native, no tools")

        def _event(self, event, **values):
            data = dict(event=event, global_step=int(self.state.global_step),
                        generation_call=self.reuse_calls, num_iterations=self.num_iterations,
                        fresh_generated=self.fresh_count, fresh_tokens=self.fresh_tokens,
                        replayed=self.replay_count, discarded_fresh=self.discarded_count,
                        completion_training_uses=self.optimized_count,
                        elapsed_seconds=time.monotonic() - self.started_at, **values)
            with self._reuse_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(data) + "\n")

        def _generate_single_turn(self, *args, **kwargs):
            ids, logps = super()._generate_single_turn(*args, **kwargs)
            self.fresh_count += len(ids)
            self.fresh_tokens += sum(map(len, ids))
            return ids, logps

        def _generate_and_score_completions(self, inputs):
            result = super()._generate_and_score_completions(inputs)
            self.reuse_calls += 1
            self._event("generation", qa_ids=[x["qa_id"] for x in inputs[::self.num_generations]])
            return result

        def _compute_loss(self, model, inputs):
            result = super()._compute_loss(model, inputs)
            self.optimized_count += inputs["completion_ids"].shape[0]
            if self.accelerator.sync_gradients:
                self._event("update_loss_complete")
            return result

    class SuccessReplayTrainer(NativeReuseTrainer):
        def _generate(self, prompts):
            # These are TRL's prepared messages, already containing the actual images.
            self._entropy_prompts = prompts
            try:
                return super()._generate(prompts)
            finally:
                self._entropy_prompts = None

        def _candidate_entropies(self, row, expected_prompt_ids, entries):
            from trl.models.utils import disable_gradient_checkpointing
            ids, images, fields = self._tokenize_prompts([self._entropy_prompts[row]])
            if list(ids[0]) != list(expected_prompt_ids):
                raise RuntimeError("Entropy scoring prompt differs from rollout prompt")
            if not images or not images[0]:
                raise RuntimeError("Visual replay entropy scoring requires the original image")
            device = self.accelerator.device
            forward = {}
            for key, value in fields.items():
                if key not in FORWARD_KEYS:
                    raise RuntimeError(f"Unsupported entropy scoring multimodal field: {key}")
                forward[key] = torch.as_tensor(value, device=device)
            forward["num_images"] = [len(images[0])]
            scores = []
            was_training = self.model.training
            try:
                self.model.eval()
                with torch.no_grad(), disable_gradient_checkpointing(self.model, self.args.gradient_checkpointing_kwargs):
                    for entry in entries:
                        tokens = entry["tokens"]
                        # Exactly one unpadded trajectory per forward: only completion
                        # positions are scored; neither prompt nor padding enters mean.
                        input_ids = torch.tensor([list(ids[0]) + tokens], device=device)
                        kwargs = dict(forward)
                        for key in ("token_type_ids", "mm_token_type_ids"):
                            if key in kwargs:
                                prefix = kwargs[key]
                                if prefix.shape != (1, len(ids[0])):
                                    raise RuntimeError("Entropy token-type prefix length mismatch")
                                kwargs[key] = torch.cat([prefix, prefix.new_zeros((1, len(tokens)))], dim=1)
                        _, entropies, *_ = super()._get_per_token_logps_and_entropies(
                            self.model, input_ids, torch.ones_like(input_ids), len(tokens),
                            batch_size=1, compute_entropy=True, **kwargs)
                        if entropies is None or entropies.shape != (1, len(tokens)):
                            raise RuntimeError("Entropy output does not align with completion tokens")
                        scores.append(entropies.float().mean().item())
            finally:
                self.model.train(was_training)
            minimum_entropy_index(scores)  # fail on NaN/Inf before using a candidate
            return scores

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.successes = SuccessBuffer(options.replay_capacity_per_qa)
            self._replay_inputs = None
            self._capture_loss = False
            self._captured_logps = None
            self._pending_replays = {}
            self._pending_rewards = None
            self._pending_prompts = None
            self._pending_tokens = None
            if (self.num_iterations != 1 or self.beta != 0 or self.loss_type != "dr_grpo"
                    or self.use_liger_kernel or self.importance_sampling_level != "token"
                    or self.top_entropy_quantile != 1 or self.args.delta is not None
                    or self.off_policy_mask_threshold is not None
                    or getattr(self, "_entropy_bonus_enabled", False)):
                raise ValueError("A requires native token-ratio dr_grpo, beta=0, iterations=1, no extra loss modifiers")

        def _generate_single_turn(self, prompt_ids, images, multimodal_fields, **kwargs):
            if self._replay_inputs is None or self._pending_tokens is not None:
                raise RuntimeError("Expected exactly one native generation call per batch")
            g = self.num_generations
            if len(prompt_ids) != len(self._replay_inputs) or len(prompt_ids) % g:
                raise RuntimeError("Generation rows must align with complete QA groups")
            # Pick candidates BEFORE seeing current outcomes. No replacement conditioned
            # on a fresh answer being wrong (which would add another selection bias).
            groups_per_pass = len(self.train_dataset) // (len(prompt_ids) // g)
            pass_index = self.reuse_calls // groups_per_pass
            for group, start in enumerate(range(0, len(prompt_ids), g)):
                rows = self._replay_inputs[start:start + g]
                qa_id = rows[0]["qa_id"]
                if any(row["qa_id"] != qa_id for row in rows):
                    raise RuntimeError("Non-contiguous QA group; refusing cross-question replay")
                if any(p != prompt_ids[start] for p in prompt_ids[start:start + g]):
                    raise RuntimeError("Same-QA prompt tokenization differs across rollout rows")
                # Alternates the selected half each dataset pass; QA count cannot
                # increase just because a QA has more historical successes.
                if self.state.global_step >= options.replay_start_step and (group + pass_index) % 2 == 0:
                    entries = self.successes.candidates(qa_id, prompt_ids[start], self.state.global_step,
                                                       options.replay_max_age)
                    if entries:
                        begin = time.monotonic()
                        scores = self._candidate_entropies(start, prompt_ids[start], entries)
                        selected = minimum_entropy_index(scores)
                        self._pending_replays[start + g - 1] = entries[selected]
                        self._event("entropy_selection", qa_id=qa_id, candidate_entropies=scores,
                                    candidate_source_steps=[e["step"] for e in entries],
                                    candidate_lengths=[len(e["tokens"]) for e in entries],
                                    selected_index=selected, scoring_policy_step=int(self.state.global_step),
                                    scoring_seconds=time.monotonic() - begin)
            tokens, logps = super()._generate_single_turn(prompt_ids, images, multimodal_fields, **kwargs)
            if logps is not None:
                raise RuntimeError("Expected native HF generation without backend logprobs")
            self._pending_prompts = [list(p) for p in prompt_ids]
            # Generate G for unchanged native multimodal batching, replace the fixed
            # last slot only. Count the discarded sample in the compute budget.
            for row, entry in self._pending_replays.items():
                tokens[row] = list(entry["tokens"])
            self._pending_tokens = [list(t) for t in tokens]
            self.replay_count += len(self._pending_replays)
            self.discarded_count += len(self._pending_replays)
            return tokens, None

        def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
            rewards = super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)
            if rewards.shape != (len(inputs), 1):
                raise RuntimeError("A expects exactly one correctness reward on one GPU")
            self._pending_rewards = rewards[:, 0].detach().cpu().tolist()
            for row in self._pending_replays:
                if self._pending_rewards[row] != 1.0:
                    raise RuntimeError("Cached success no longer scores correct: prompt/scorer mismatch")
            fresh = [r for i, r in enumerate(self._pending_rewards) if i not in self._pending_replays]
            self._log_metric("reuse/fresh_correctness", sum(fresh) / len(fresh))
            self._log_metric("reuse/replay_fraction", len(self._pending_replays) / len(inputs))
            return rewards

        def _get_per_token_logps_and_entropies(self, *args, **kwargs):
            result = super()._get_per_token_logps_and_entropies(*args, **kwargs)
            if self._capture_loss:
                if self._captured_logps is not None:
                    raise RuntimeError("Expected one loss log-probability call")
                self._captured_logps = result[0]
            return result

        def _generate_and_score_completions(self, inputs):
            from trl.models.utils import disable_gradient_checkpointing
            self._replay_inputs = inputs
            self._pending_tokens = None
            self._pending_replays = {}
            self._pending_rewards = None
            try:
                # TRL decodes/rewards/recomputes advantages AFTER token substitution;
                # multimodal tensors and termination masks are rebuilt by TRL itself.
                result = super()._generate_and_score_completions(inputs)
                if self._pending_rewards is None or self._pending_tokens is None:
                    raise RuntimeError("TRL did not invoke expected generation/reward hooks")
                keys = inspect.signature(super()._get_per_token_logps_and_entropies).parameters
                forward = {key: result[key] for key in FORWARD_KEYS if key in result and key in keys}
                # Microbatched no-grad scoring at the SAME pre-update policy. This
                # is TRL's temperature-scaled probability convention (pre top-p).
                with torch.no_grad(), disable_gradient_checkpointing(self.model, self.args.gradient_checkpointing_kwargs):
                    old = super()._get_per_token_logps_and_entropies(
                        self.model, torch.cat([result["prompt_ids"], result["completion_ids"]], dim=1),
                        torch.cat([result["prompt_mask"], result["completion_mask"]], dim=1),
                        result["completion_ids"].size(1), batch_size=self.args.per_device_train_batch_size,
                        **forward)[0].detach().float()
                replay_mask = torch.zeros(len(inputs), device=old.device, dtype=torch.bool)
                eos = {self._tokenizer.eos_token_id, self._tokenizer.pad_token_id}
                history = []
                for row, sample in enumerate(inputs):
                    ids = self._pending_tokens[row]
                    reward = self._pending_rewards[row]
                    if row in self._pending_replays:
                        entry = self._pending_replays[row]
                        replay_mask[row] = True
                        old[row].zero_()
                        old[row, :len(ids)] = torch.tensor(entry["logps"], device=old.device)
                    else:
                        self.successes.add(sample["qa_id"], self._pending_prompts[row], ids,
                                           old[row, :len(ids)].cpu().tolist(), reward,
                                           ids[-1] in eos, self.state.global_step)
                    history.append(dict(qa_id=sample["qa_id"], reward=reward,
                                        replay=row in self._pending_replays,
                                        source_step=self._pending_replays[row]["step"] if row in self._pending_replays else int(self.state.global_step),
                                        length=len(ids), terminated=ids[-1] in eos))
                result["old_per_token_logps"] = old
                # Both are batch-leading tensors, so TRL shuffles/splits them with
                # exactly the same permutation as completion_ids and advantages.
                result["success_replay_mask"] = replay_mask
                self._event("replay_group_details", rows=history,
                            buffer_qas=len(self.successes.items),
                            buffer_trajectories=sum(map(len, self.successes.items.values())))
                return result
            finally:
                self._replay_inputs = None

        def _compute_loss(self, model, inputs):
            self._capture_loss = True
            self._captured_logps = None
            try:
                native_loss = super()._compute_loss(model, inputs)
                logps = self._captured_logps
                if logps is None:
                    raise RuntimeError("TRL loss forward hook was not invoked")
                if not inputs["success_replay_mask"].any():
                    return native_loss
                correction = replay_correction(
                    torch, logps, inputs["old_per_token_logps"], inputs["advantages"],
                    inputs["success_replay_mask"], inputs["completion_mask"],
                    self.epsilon_low, self.epsilon_high, options.replay_shaping_beta,
                    self.max_completion_length, self.current_gradient_accumulation_steps)
                return native_loss + correction
            finally:
                self._capture_loss = False
                self._captured_logps = None

        def _save_checkpoint(self, model, trial):
            super()._save_checkpoint(model, trial)
            path = Path(self._get_output_dir(trial=trial)) / f"checkpoint-{self.state.global_step}"
            (path / "success_replay_buffer.json").write_text(
                json.dumps(self.successes.state_dict()), encoding="utf-8")

    return NativeReuseTrainer, SuccessReplayTrainer
