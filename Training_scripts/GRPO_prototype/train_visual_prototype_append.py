#!/usr/bin/env python3
"""Train image-conditioned continuous visual prototypes with GRPO.

The only trainable parameter is ``policy.prototypes``.  It reads cached global
Qwen merger tokens; a sampled Dirichlet gate is the policy action appended to
the frozen decoder as K weighted visual tokens.
"""
from __future__ import annotations
import argparse, json, math, random, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
for sibling in (ROOT / "SparseVLM_eliminateCONTEXT", ROOT / "Training_scripts" / "GRPO_lookbetter"):
    if str(sibling) not in sys.path: sys.path.insert(0, str(sibling))
from rolling_sparse_qwen3vl import load_model
from data import evaluate_answer, load_mapverse_questions
from live_rollout_visual_prototypes import prepare_global_visual_prefill, rollout
from visual_prototype_grpo import visual_slot_grpo_loss
from visual_prototype_policy import VisualPrototypeAppendPolicy, decoder_hidden_size


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=ROOT / "Datasets" / "When_Where_Crop" / "visual_easier_1000.json")
    p.add_argument("--mapverse-root", type=Path, required=True); p.add_argument("--mapwise-root", type=Path, required=True)
    p.add_argument("--adapter", type=Path, required=True); p.add_argument("--model", default=None)
    p.add_argument("--output-dir", type=Path, required=True); p.add_argument("--policy-checkpoint", type=Path)
    p.add_argument("--resume-from-checkpoint", type=Path); p.add_argument("--start-index", type=int, default=1)
    p.add_argument("--end-index", type=int); p.add_argument("--limit", type=int); p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--group-size", type=int, default=16); p.add_argument("--questions-per-update", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=500); p.add_argument("--look-every", type=int, default=20)
    p.add_argument("--max-looks", type=int, default=9); p.add_argument("--global-max-pixels", type=int, default=1_000_000)
    p.add_argument("--num-prototypes", type=int, default=9); p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--min-concentration", type=float, default=0.1); p.add_argument("--learning-rate", type=float, default=2e-3)
    p.add_argument("--clip-ratio", type=float, default=.2); p.add_argument("--num-policy-updates", type=int, default=1)
    p.add_argument("--off-policy-mask-threshold", type=float); p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--preserve-question-order", action="store_true"); p.add_argument("--eval-after-epoch", action="store_true")
    p.add_argument("--stop-at-eval-accuracy", type=float); p.add_argument("--save-steps", type=int, default=1000)
    p.add_argument("--save-every-epoch", action="store_true"); p.add_argument("--print-policy-update-diagnostics", action="store_true")
    p.add_argument("--no-load-in-4bit", action="store_false", dest="load_in_4bit")
    p.add_argument("--wandb-project"); p.add_argument("--wandb-entity"); p.add_argument("--wandb-run-name")
    p.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online"); p.set_defaults(load_in_4bit=True)
    a = p.parse_args()
    if a.policy_checkpoint and a.resume_from_checkpoint: p.error("Choose one checkpoint option")
    if a.group_size < 2 or min(a.epochs, a.questions_per_update, a.look_every, a.num_prototypes) <= 0: p.error("invalid training sizes")
    torch.manual_seed(a.seed); a.output_dir.mkdir(parents=True, exist_ok=True)
    model, processor = load_model(a.model, a.adapter, load_in_4bit=a.load_in_4bit); model.eval()
    for value in model.parameters(): value.requires_grad_(False)
    device = model.get_input_embeddings().weight.device
    policy = VisualPrototypeAppendPolicy(decoder_hidden_size(model), a.num_prototypes, a.temperature, a.min_concentration).to(device)
    opt = torch.optim.AdamW([policy.prototypes], lr=a.learning_rate, weight_decay=0.0)
    resume = None
    if a.policy_checkpoint:
        policy.load_state_dict(torch.load(a.policy_checkpoint, map_location="cpu", weights_only=True)["policy"])
    if a.resume_from_checkpoint:
        resume = torch.load(a.resume_from_checkpoint, map_location="cpu", weights_only=True)
        policy.load_state_dict(resume["policy"]); opt.load_state_dict(resume["optimizer"])
    initial = policy.prototypes.detach().clone()
    records_all = load_mapverse_questions(a.dataset, a.mapverse_root, mapwise_root=a.mapwise_root, limit=a.end_index)
    records = records_all[a.start_index - 1:a.end_index]
    if a.limit: records = records[:a.limit]
    if not records: p.error("Selected question range is empty")
    indexed = list(enumerate(records, start=a.start_index)); steps_per_epoch = math.ceil(len(records) / a.questions_per_update)
    start_epoch = int(resume["epoch"]) if resume else 0; start_step = int(resume["step"]) if resume else -1
    if resume and start_step + 1 != start_epoch * steps_per_epoch: p.error("Resume requires an epoch-boundary checkpoint")
    wandb_run = None
    if a.wandb_project and a.wandb_mode != "disabled":
        import wandb
        wandb_run = wandb.init(project=a.wandb_project, entity=a.wandb_entity, name=a.wandb_run_name,
                               mode=a.wandb_mode, config={k: str(v) if isinstance(v, Path) else v for k,v in vars(a).items()})
    trace_path = a.output_dir / "rollouts.jsonl"

    def save(path, step, epoch):
        torch.save({"policy": policy.state_dict(), "optimizer": opt.state_dict(), "hidden_size": policy.hidden_size,
                    "num_prototypes": policy.num_prototypes, "temperature": policy.temperature,
                    "step": step, "epoch": epoch}, path)
    def evaluate(epoch):
        out, rewards = {}, []
        for offset, item in enumerate(records):
            prefill = prepare_global_visual_prefill(model, processor, item, global_max_pixels=a.global_max_pixels)
            row = rollout(model, processor, policy, item, rollout_id=0, max_new_tokens=a.max_new_tokens,
                          look_every=a.look_every, max_looks=a.max_looks, sample=False, prefill=prefill)
            value = evaluate_answer(row["response"], item)["reward"]; rewards.append(value)
            out[f"eval/reward_{a.start_index + offset}"] = value
        out["eval/accuracy"] = sum(rewards) / len(rewards); return out

    with trace_path.open("a" if resume else "w", encoding="utf-8") as trace:
        if a.eval_after_epoch and not resume:
            base = evaluate(0); print(json.dumps({"step":-1,"epoch":0,**base}), flush=True)
            if wandb_run: wandb_run.log(base, step=0)
        for local_step in range(steps_per_epoch * a.epochs):
            step = start_step + 1 + local_step; epoch = start_epoch + local_step // steps_per_epoch
            order = list(indexed)
            if not a.preserve_question_order: random.Random(a.seed + epoch).shuffle(order)
            start = (local_step % steps_per_epoch) * a.questions_per_update; batch=[]
            for dataset_index, item in order[start:start+a.questions_per_update]:
                prefill = prepare_global_visual_prefill(model, processor, item, global_max_pixels=a.global_max_pixels)
                group=[]
                for rollout_id in range(a.group_size):
                    row = rollout(model, processor, policy, item, rollout_id=rollout_id, max_new_tokens=a.max_new_tokens,
                                  look_every=a.look_every, max_looks=a.max_looks, sample=True, prefill=prefill)
                    row["reward"] = evaluate_answer(row["response"], item)["reward"]; row["dataset_index"] = dataset_index
                    trace.write(json.dumps({k:v for k,v in row.items() if k not in ("actions","image_embeddings")}, ensure_ascii=False)+"\n")
                    trace.flush(); group.append(row)
                batch.append((dataset_index,item,group))
            losses=[]; info_list=[]; grad_norm=0.0
            for update in range(a.num_policy_updates):
                results=[visual_slot_grpo_loss(policy, group, a.clip_ratio, a.off_policy_mask_threshold) for _,_,group in batch]
                active=[x for x,_ in results if x is not None]
                if not active: break
                loss=torch.stack(active).mean(); before=policy.prototypes.detach().clone(); opt.zero_grad(set_to_none=True); loss.backward()
                grad_norm=float(policy.prototypes.grad.norm()); per_grad=policy.prototypes.grad.detach().norm(dim=1).cpu(); opt.step()
                losses.append(float(loss.detach())); info_list=[i for _,i in results]
                if a.print_policy_update_diagnostics:
                    payload={"outer_step":step+1,"policy_update":update+1,"loss":float(loss.detach()),
                             "prototype_gradient_norms":[float(x) for x in per_grad],
                             "prototype_delta_norms":[float(x) for x in (policy.prototypes.detach()-before).norm(dim=1).cpu()]}
                    print(f"[VISUAL_POLICY_UPDATE {step+1:05d}.{update+1}] {json.dumps(payload)}", flush=True)
            rows=[row for _,_,g in batch for row in g]; groups=[]
            for index,item,g in batch:
                rs=[float(x["reward"]) for x in g]; payload={"dataset_index":index,"qa_id":item.sample_id,"rewards":rs,
                    "correct_count":int(sum(rs)),"incorrect_count":len(rs)-int(sum(rs)),"signal_group":len(set(rs))>1}
                groups.append(payload); print(f"[REWARD {step+1:05d}] {json.dumps(payload)}", flush=True)
            events=[e for row in rows for e in row["diagnostics"]]
            gates=torch.tensor([e["gate_weights"] for e in events],dtype=torch.float64) if events else torch.empty((0,a.num_prototypes),dtype=torch.float64)
            gate_entropy=float(-(gates*gates.clamp_min(1e-12).log()).sum(-1).mean()) if len(gates) else 0.0
            gate_means=torch.tensor([e["gate_mean"] for e in events],dtype=torch.float64) if events else torch.empty((0,a.num_prototypes),dtype=torch.float64)
            if len(gate_means):
                conditional_gate_entropy=-(gate_means*gate_means.clamp_min(1e-12).log()).sum(-1).mean()
                marginal_gate=gate_means.mean(0)
                marginal_gate_entropy=-(marginal_gate*marginal_gate.clamp_min(1e-12).log()).sum()
                gate_argmax_frequency=torch.bincount(gate_means.argmax(-1),minlength=a.num_prototypes).double()/len(gate_means)
                gate_mean_std=gate_means.std(dim=0,unbiased=False).mean()
            else:
                conditional_gate_entropy=marginal_gate_entropy=gate_mean_std=torch.tensor(0.)
                marginal_gate=gate_argmax_frequency=torch.zeros(a.num_prototypes,dtype=torch.float64)
            metrics={"train/visual_policy_loss":losses[-1] if losses else 0.0,"train/gate_entropy":gate_entropy,
                "train/gate_conditional_entropy":float(conditional_gate_entropy),
                "train/gate_marginal_entropy":float(marginal_gate_entropy),
                "train/gate_state_mutual_information":float(marginal_gate_entropy-conditional_gate_entropy),
                "train/gate_mean_probability_std":float(gate_mean_std),
                "train/gate_max_slot_frequency":float(gate_argmax_frequency.max()),
                "train/gradient_norm":grad_norm,"train/learning_rate":float(opt.param_groups[0]["lr"]),
                "train/prototype_norm":float(policy.prototypes.detach().norm()),"train/prototype_delta_norm":float((policy.prototypes.detach()-initial).norm()),
                "train/reward_mean":sum(r["reward"] for r in rows)/len(rows),"train/rollouts_correct":sum(r["reward"] for r in rows),
                "train/questions_mixed_correctness":sum(g["signal_group"] for g in groups),"train/visual_slot_append_count":sum(r["visual_slot_append_count"] for r in rows),
                "train/global_prefill_reuse_rollouts":sum(r["global_prefill_reused"] for r in rows),"train/policy_updates_completed":len(losses)}
            metrics.update({f"train/gate_mean_slot_{i}":float(marginal_gate[i]) for i in range(a.num_prototypes)})
            metrics.update({f"train/gate_argmax_frequency_slot_{i}":float(gate_argmax_frequency[i]) for i in range(a.num_prototypes)})
            if info_list:
                metrics.update({"train/action_entropy":sum(i["action_entropy"] for i in info_list)/len(info_list),
                    "train/off_policy_mask_fraction":sum(i["off_policy_mask_fraction"] for i in info_list)/len(info_list),
                    "train/active_visual_actions":sum(i["active_action_count"] for i in info_list)})
            if a.eval_after_epoch and (step+1)%steps_per_epoch==0: metrics.update(evaluate(epoch+1))
            print(json.dumps({"step":step,"epoch":epoch+1,"dataset_indices":[x for x,_,_ in batch],**metrics}),flush=True)
            if wandb_run: wandb_run.log(metrics,step=step+1)
            if (step+1)%a.save_steps==0: save(a.output_dir/f"visual_policy_step_{step+1:06d}.pt",step,epoch+1)
            if a.save_every_epoch and (step+1)%steps_per_epoch==0: save(a.output_dir/f"visual_policy_epoch_{epoch+1:03d}.pt",step,epoch+1)
            if a.stop_at_eval_accuracy and metrics.get("eval/accuracy",-1)>=a.stop_at_eval_accuracy: break
    save(a.output_dir/"visual_policy_final.pt",step,epoch+1)
    if wandb_run: wandb_run.finish()

if __name__ == "__main__": main()
