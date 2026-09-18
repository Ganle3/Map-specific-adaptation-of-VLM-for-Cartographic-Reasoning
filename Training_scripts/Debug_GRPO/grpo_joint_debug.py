"""Unified joint GRPO debug entry point for any selected QA dataset.

Reuses the frozen baseline, not the A/B replay trainer or the older Language
training loop. Only scope and rank/alpha differ between presets.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import sys


EXPECTED_DEBUG_QA = [('usa', 2051), ('china', 525), ('usa', 2021), ('china', 646)]


def validate_debug_dataset(path):
    """Validate the canonical four-QA diagnostic anchor when it is selected."""
    rows = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if [(row['country'], row['source_index']) for row in rows] != EXPECTED_DEBUG_QA:
        raise ValueError('Expected the four selected diagnostic QAs in fixed order')
    canonical = (Path(__file__).resolve().parents[2]
                 / 'Datasets/Processed_Mapwise/Train_Val/mapwise_grpo_joint_debug4_src2051.json')
    if rows != json.loads(canonical.read_text(encoding='utf-8-sig')):
        raise ValueError('QA metadata differs from the canonical diagnostic dataset')
    return rows


def parameter_inventory(model):
    return {n: dict(dtype=str(p.dtype), shape=list(p.shape),
                    parameter_class=type(p).__name__, trainable=p.requires_grad)
            for n, p in model.named_parameters()}


def dtype_changes(before, after):
    return [dict(name=n, before=before.get(n), after=after.get(n))
            for n in sorted(before.keys() | after.keys())
            if n not in before or n not in after or before[n]['dtype'] != after[n]['dtype']]


def expected_bias_cast(change, allowed_names, already_cast):
    b, a = change['before'], change['after']
    return (change['name'] in allowed_names and change['name'] not in already_cast
            and b is not None and a is not None
            and b['dtype'] == 'torch.float32' and a['dtype'] == 'torch.bfloat16'
            and not b['trainable'] and not a['trainable']
            and b['shape'] == a['shape'] and b['parameter_class'] == a['parameter_class'])


def preset(name):
    """Return the default rank and exact LoRA target set for one scope."""
    if name == "joint":
        return 16, preset("vision_merger")[1] | preset("language")[1]
    blocks = {f"model.visual.blocks.{i}.{suffix}" for i in range(27)
              for suffix in ("attn.qkv", "attn.proj", "mlp.linear_fc1", "mlp.linear_fc2")}
    if name == "vision":
        return 16, blocks
    if name == "vision_merger":
        mergers = {f"model.visual.{module}.linear_fc{j}" for module in
                   ["merger", *[f"deepstack_merger_list.{i}" for i in range(3)]] for j in (1, 2)}
        return 16, blocks | mergers
    if name == "language":
        return 16, {f"model.language_model.layers.{i}.{suffix}" for i in range(36)
                    for suffix in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                                   "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")}
    raise ValueError(name)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scope-experiment", required=True,
                   choices=("joint", "vision", "language", "vision_merger"))
    p.add_argument("--rank", type=int, default=None,
                   help="LoRA rank; independent of the scope preset")
    p.add_argument("--max-steps", type=int, default=120)
    p.add_argument("--save-steps", type=int, default=20)
    p.add_argument("--save-total-limit", type=int, default=None)
    p.add_argument("--lr-scheduler-type", choices=("constant", "cosine", "constant_then_cosine"), default="constant")
    p.add_argument("--decay-start-step", type=int, default=None,
                   help="First step of cosine decay for constant_then_cosine.")
    p.add_argument("--decay-final-factor", type=float, default=0.8,
                   help="Final LR multiplier for constant_then_cosine.")
    p.add_argument("--warmup-fraction", type=float, default=0.0)
    p.add_argument("--num-iterations", type=int, default=1,
                   help="Number of policy updates per generated rollout batch.")
    p.add_argument("--off-policy-mask-threshold", type=float, default=None,
                   choices=(0.2, 0.4),
                   help="Optional TRL off-policy negative-sequence mask threshold.")
    extra, remaining = p.parse_known_args()
    preset_rank, targets = preset(extra.scope_experiment)
    rank = extra.rank if extra.rank is not None else preset_rank
    if rank < 1:
        p.error("--rank must be positive")
    import _vislora_grpo_baseline_snapshot as base
    sys.argv = [sys.argv[0], *remaining]
    args = base.parse_args()
    args.lora_rank = args.lora_alpha = rank
    args.warmup_fraction = extra.warmup_fraction
    if args.per_device_train_batch_size < 1 or args.gradient_accumulation_steps < 1:
        p.error('Batch size and gradient accumulation must be positive')
    args.num_generations = 4
    if not 0 <= extra.warmup_fraction < 1:
        p.error("--warmup-fraction must be in [0, 1)")
    if extra.lr_scheduler_type == "constant_then_cosine" and not (
            extra.decay_start_step is not None and 0 <= extra.decay_start_step < extra.max_steps):
        p.error("constant_then_cosine requires 0 <= --decay-start-step < --max-steps")
    if not 0 < extra.decay_final_factor <= 1:
        p.error("--decay-final-factor must be in (0, 1]")
    # Preserve the caller's ordering choice.  The fixed-order experiment
    # passes --preserve-qa-order explicitly; random-order runs leave it off.
    args.save_steps = extra.save_steps
    args.save_total_limit = (extra.save_total_limit if extra.save_total_limit is not None
                             else extra.max_steps // extra.save_steps + 1)
    if (extra.max_steps < 1 or extra.num_iterations < 1 or extra.save_steps < 1
            or args.learning_rate != 5e-6):
        p.error('Use positive updates and LR 5e-6')
    base.validate_args(args)
    if args.init_adapter_path is not None:
        p.error("Use a fresh raw base model; init adapters are not supported")
    if args.resume_from_checkpoint is not None and not Path(args.resume_from_checkpoint).is_dir():
        p.error("--resume-from-checkpoint must point to an existing checkpoint directory")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        p.error("This diagnostic requires one GPU per experiment")
    if extra.max_steps < 1 or args.beta != 0:
        p.error("Positive max-steps and beta=0 are required")
    if importlib.metadata.version("trl") != "1.12.0":
        raise RuntimeError("This entry point requires TRL 1.12.0")
    rows = (validate_debug_dataset(args.qa_json)
            if Path(args.qa_json).name == 'mapwise_grpo_joint_debug4_src2051.json'
            else json.loads(Path(args.qa_json).read_text(encoding='utf-8-sig')))
    if not rows or len({(r['country'], r['source_index']) for r in rows}) != len(rows):
        raise ValueError('Training dataset must contain unique non-empty QA rows')
    expected_keys = [(r['country'], r['source_index']) for r in rows]
    output = Path(args.output_dir).expanduser().resolve()
    if args.resume_from_checkpoint is None and ((output / "run_config.json").exists() or list(output.glob("checkpoint-*"))):
        p.error("Use a new output directory")
    args.scope_experiment = extra.scope_experiment
    args.max_steps = extra.max_steps
    args.num_iterations = extra.num_iterations
    # Explicit anchored names prevent accidentally adapting norms/embeddings/head.
    base.VISION_TARGET_REGEX = "^(?:" + "|".join(re.escape(n) for n in sorted(targets)) + ")$"
    base.EXPECTED_TARGET_MODULES = len(targets)
    base.EXPECTED_LORA_TENSORS = 2 * len(targets)

    def verify(model):
        found = set(model.targeted_module_names)
        if found != targets:
            raise RuntimeError(f"Scope mismatch: missing={sorted(targets-found)}, extra={sorted(found-targets)}")
        trainable = [(n, v) for n, v in model.named_parameters() if v.requires_grad]
        if len(trainable) != 2 * len(targets) or any(
                ".lora_A." not in n and ".lora_B." not in n for n, _ in trainable):
            raise RuntimeError("Unexpected trainable tensors outside the specified LoRA adapters")
        inventory = dict(experiment=extra.scope_experiment, rank=rank, alpha=rank,
                         targets=sorted(found), target_count=len(found),
                         trainable_parameters=sum(v.numel() for _, v in trainable),
                         tensors=[dict(name=n, shape=list(v.shape), dtype=str(v.dtype)) for n, v in trainable])
        (output / "lora_scope_inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
        print(f"VERIFIED {extra.scope_experiment}: {len(found)} modules, {inventory['trainable_parameters']} trainable parameters")
    base.verify_vision_only_lora = verify
    Config = base.GRPOConfig
    def config(**kwargs):
        config_scheduler = ('constant' if extra.lr_scheduler_type == 'constant_then_cosine'
                            else extra.lr_scheduler_type)
        kwargs.update(max_steps=extra.max_steps, num_iterations=extra.num_iterations,
                      lr_scheduler_type=config_scheduler,
                      warmup_steps=math.ceil(extra.max_steps * extra.warmup_fraction),
                      off_policy_mask_threshold=extra.off_policy_mask_threshold)
        result = Config(**kwargs)
        (output / "effective_grpo_config.json").write_text(json.dumps(result.to_dict(), indent=2, default=str), encoding="utf-8")
        return result
    base.GRPOConfig = config
    save = base.save_run_config
    def save_config(output_dir, args, dataset_size, **unused):
        if dataset_size != len(rows):
            raise ValueError(f"Expected {len(rows)} QAs, got {dataset_size}")
        save(output_dir, args, dataset_size, extra.max_steps, math.ceil(extra.max_steps * args.warmup_fraction))
        path = output_dir / "run_config.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        for key in ("vision_lora_target_modules", "vision_lora_trainable_tensors"):
            payload.pop(key, None)
        payload.update(lora_target_modules=len(targets), lora_trainable_tensors=2*len(targets),
                       step_counts_are_estimates=False, budget_unit="optimizer_updates",
                       replay=False, num_iterations=extra.num_iterations, lr_scheduler_type=extra.lr_scheduler_type,
                       off_policy_mask_threshold=extra.off_policy_mask_threshold,
                       qa_groups_per_update=len(rows), exposures_per_qa=extra.max_steps,
                       save_steps=extra.save_steps, save_total_limit=args.save_total_limit,
                       warmup_fraction=extra.warmup_fraction,
                       sha256={str(f): hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in
                               (args.qa_json, args.evaluation_script, Path(__file__), Path(base.__file__))},
                       versions={v: importlib.metadata.version(v) for v in
                                 ("torch", "trl", "transformers", "peft", "accelerate")})
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Authoritative budget: {extra.max_steps} updates. Scope: {extra.scope_experiment}.")
        print("Inherited Vision-only/epoch estimate banners below are baseline labels; consult scope inventory/config.")
    base.save_run_config = save_config

    class UpdateAudit(base.TrainerCallback):
        """Audit all optimizer memberships and actual LoRA tensor changes on first 5 updates.

        CPU snapshots; no optimizer/model changes and no claims about greedy gains.
        """
        def on_train_begin(self, args, state, control, model=None, optimizer=None, **kwargs):
            self.params = [(n, v) for n, v in model.named_parameters() if v.requires_grad]
            registered = {id(v) for group in optimizer.param_groups for v in group["params"]}
            missing = [n for n, v in self.params if id(v) not in registered]
            if missing:
                raise RuntimeError(f"Trainable tensors missing from optimizer: {missing}")
            self.before = None
            (output / "optimizer_membership.json").write_text(json.dumps({
                "all_trainable_registered": True,
                "trainable": [{"name": n, "dtype_after_trainer_init": str(v.dtype)} for n, v in self.params]
            }, indent=2), encoding="utf-8")

        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            if state.global_step < 5:
                self.before = {n: v.detach().cpu().clone() for n, v in self.params}

        def on_step_end(self, args, state, control, **kwargs):
            if self.before is None:
                return
            entries = []
            for n, v in self.params:
                diff = v.detach().cpu().float() - self.before[n].float()
                entries.append(dict(name=n, changed_elements=int((diff != 0).sum()),
                                    max_abs_delta=diff.abs().max().item()))
            with (output / "parameter_update_audit.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(dict(step=state.global_step, tensors=entries)) + "\n")
            self.before = None

    Trainer = base.GRPOTrainer
    class AuditedTrainer(Trainer):
        def create_scheduler(self, num_training_steps, optimizer=None):
            if extra.lr_scheduler_type != 'constant_then_cosine':
                return super().create_scheduler(num_training_steps, optimizer)
            from torch.optim.lr_scheduler import LambdaLR
            optimizer = optimizer or self.optimizer
            start = extra.decay_start_step
            span = max(1, num_training_steps - start)
            def schedule(step):
                if step <= start:
                    return 1.0
                progress = min(1.0, (step - start) / span)
                return extra.decay_final_factor + (1.0 - extra.decay_final_factor) * \
                    0.5 * (1.0 + math.cos(math.pi * progress))
            self.lr_scheduler = LambdaLR(optimizer, schedule)
            return self.lr_scheduler

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            from bitsandbytes.nn import Linear4bit
            import inspect
            self._expected_bias_casts = {
                n + '.bias' for n, m in self.model.named_modules()
                if isinstance(m, Linear4bit) and '.visual.' in n
                and m.bias is not None and not m.bias.requires_grad}
            self._seen_bias_casts = set()
            (output / 'quantized_bias_cast_policy.json').write_text(json.dumps(dict(
                allowed_parameters=sorted(self._expected_bias_casts),
                policy='Once per frozen visual Linear4bit bias: FP32 to BF16 only',
                bitsandbytes_version=importlib.metadata.version('bitsandbytes'),
                installed_forward_source=inspect.getsource(Linear4bit.forward)), indent=2), encoding='utf-8')
            self.add_callback(UpdateAudit())
            if self.use_vllm or self.args.use_transformers_continuous_batching:
                raise RuntimeError('This diagnostic requires native generation')

        def _write_dtype_audit(self, phase, before):
            after = parameter_inventory(self.model)
            changes = dtype_changes(before, after)
            record = dict(step=self.state.global_step, phase=phase, changes=changes,
                          before=before, after=after,
                          autocast_enabled=base.torch.is_autocast_enabled())
            with (output / 'rollout_dtype_audit.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(record) + '\n')
            if changes:
                print('DTYPE CHANGE ' + phase + ': ' + json.dumps(changes), flush=True)
            return after

        def _generate_single_turn(self, *a, **kw):
            before = parameter_inventory(self.model)
            try:
                return super()._generate_single_turn(*a, **kw)
            finally:
                self._write_dtype_audit('native_generation', before)

        def _generate_and_score_completions(self, inputs):
            from collections import Counter
            counts = Counter((r['country'], r['source_index']) for r in inputs)
            expected_batch = {key: 4 for key in dict.fromkeys(expected_keys)}
            if len(counts) < 1 or any(value != 4 for value in counts.values()) \
                    or not set(counts).issubset(expected_batch):
                raise RuntimeError(f'Expected four rollouts for four selected QAs: {counts}')
            before = parameter_inventory(self.model)
            try:
                result = super()._generate_and_score_completions(inputs)
            finally:
                after = self._write_dtype_audit('generation_and_scoring', before)
            changes = dtype_changes(before, after)
            unexpected = [c for c in changes if not expected_bias_cast(
                c, self._expected_bias_casts, self._seen_bias_casts)]
            if unexpected:
                raise RuntimeError(f'{len(unexpected)} unexpected parameter dtype/name changes during rollout; '
                                   f'see {output / "rollout_dtype_audit.jsonl"}. No update allowed.')
            if changes:
                self._seen_bias_casts.update(c['name'] for c in changes)
                print(f'Accepted {len(changes)} first-use frozen Linear4bit bias casts; '
                      'LoRA/base weight casts remain forbidden.', flush=True)
            return result
    base.GRPOTrainer = AuditedTrainer
    base.run_training(args)
    path = output / "training_metrics.json"
    metrics = json.loads(path.read_text(encoding="utf-8"))
    metrics.update(expected_optimizer_steps=extra.max_steps,
                   warmup_steps=math.ceil(extra.max_steps * args.warmup_fraction),
                   step_counts_are_estimates=False, scope_experiment=extra.scope_experiment)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
