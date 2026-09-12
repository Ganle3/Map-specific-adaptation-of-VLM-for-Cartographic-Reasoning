"""Three fixed-budget plain-GRPO LoRA scope/capacity experiments.

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


def preset(name):
    blocks = {f"model.visual.blocks.{i}.{suffix}" for i in range(27)
              for suffix in ("attn.qkv", "attn.proj", "mlp.linear_fc1", "mlp.linear_fc2")}
    if name == "vision_r64":
        return 64, blocks
    if name == "vision_merger_r16":
        mergers = {f"model.visual.{module}.linear_fc{j}" for module in
                   ["merger", *[f"deepstack_merger_list.{i}" for i in range(3)]] for j in (1, 2)}
        return 16, blocks | mergers
    if name == "language_r16":
        return 16, {f"model.language_model.layers.{i}.{suffix}" for i in range(36)
                    for suffix in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                                   "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")}
    raise ValueError(name)


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--scope-experiment", required=True,
                   choices=("vision_r64", "vision_merger_r16", "language_r16"))
    p.add_argument("--max-steps", type=int, default=220)
    extra, remaining = p.parse_known_args()
    rank, targets = preset(extra.scope_experiment)
    import _vislora_grpo_baseline_snapshot as base
    sys.argv = [sys.argv[0], *remaining]
    args = base.parse_args()
    args.lora_rank = args.lora_alpha = rank
    base.validate_args(args)
    if args.init_adapter_path is not None or args.resume_from_checkpoint is not None:
        p.error("Use a fresh raw base model, no init adapter or resume")
    if not args.preserve_qa_order or int(os.environ.get("WORLD_SIZE", "1")) != 1:
        p.error("Use fixed QA order on one GPU per experiment")
    if extra.max_steps < 1 or args.beta != 0:
        p.error("Positive max-steps and beta=0 are required")
    if importlib.metadata.version("trl") != "1.12.0":
        raise RuntimeError("This entry point requires TRL 1.12.0")
    output = Path(args.output_dir).expanduser().resolve()
    if (output / "run_config.json").exists() or list(output.glob("checkpoint-*")):
        p.error("Use a new output directory")
    args.scope_experiment = extra.scope_experiment
    args.max_steps = extra.max_steps
    args.num_iterations = 1
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
        kwargs.update(max_steps=extra.max_steps, num_iterations=1)
        result = Config(**kwargs)
        (output / "effective_grpo_config.json").write_text(json.dumps(result.to_dict(), indent=2, default=str), encoding="utf-8")
        return result
    base.GRPOConfig = config
    save = base.save_run_config
    def save_config(output_dir, args, dataset_size, **unused):
        if dataset_size != 44:
            raise ValueError(f"Expected the fixed 44-QA set, got {dataset_size}")
        save(output_dir, args, dataset_size, extra.max_steps, math.ceil(extra.max_steps * args.warmup_fraction))
        path = output_dir / "run_config.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        for key in ("vision_lora_target_modules", "vision_lora_trainable_tensors"):
            payload.pop(key, None)
        payload.update(lora_target_modules=len(targets), lora_trainable_tensors=2*len(targets),
                       step_counts_are_estimates=False, budget_unit="optimizer_updates",
                       replay=False, num_iterations=1,
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
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.add_callback(UpdateAudit())
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
