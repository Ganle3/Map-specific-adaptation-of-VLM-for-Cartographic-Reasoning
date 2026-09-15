"""Single-QA Language or joint Vision/merger/Language LoRA; plain fresh GRPO."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from train_mapwise_grpo_scope_ablation import preset


def targets_for(scope):
    language = preset('language_r16')[1]
    if scope == 'language_r16':
        return language
    if scope == 'joint_r16':
        return language | preset('vision_merger_r16')[1]
    raise ValueError(scope)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--scope', choices=('language_r16','joint_r16'), required=True)
    extra, remaining = parser.parse_known_args()
    import _vislora_grpo_baseline_snapshot as base
    import train_mapwise_grpo_single_qa as single
    targets = targets_for(extra.scope)
    base.VISION_TARGET_REGEX = '^(?:'+'|'.join(re.escape(n) for n in sorted(targets))+')$'
    base.EXPECTED_TARGET_MODULES = len(targets)
    base.EXPECTED_LORA_TENSORS = 2*len(targets)
    output = None

    def verify(model):
        actual = set(model.targeted_module_names)
        trainable = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
        if actual != targets or len(trainable) != 2*len(targets) or any(
                '.lora_A.' not in n and '.lora_B.' not in n for n,_ in trainable):
            raise RuntimeError('Unexpected target modules or trainable base weights')
        (output/'lora_scope_inventory.json').write_text(json.dumps(dict(
            scope=extra.scope, rank=16, alpha=16, targets=sorted(actual),
            trainable_parameters=sum(p.numel() for _,p in trainable),
            tensors=[dict(name=n,shape=list(p.shape),dtype=str(p.dtype)) for n,p in trainable]),indent=2),encoding='utf-8')
        print(f'ACTUAL SCOPE: {extra.scope}, {len(actual)} modules; base weights frozen',flush=True)
    base.verify_vision_only_lora = verify

    class Audit(base.TrainerCallback):
        def on_train_begin(self,args,state,control,model=None,optimizer=None,**kw):
            self.params=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
            ids={id(p) for g in optimizer.param_groups for p in g['params']}
            if any(id(p) not in ids for _,p in self.params):
                raise RuntimeError('Trainable adapter missing from optimizer')
            self.before=None
            (output/'optimizer_membership.json').write_text(json.dumps(dict(all_trainable_registered=True,
                tensors=[dict(name=n,dtype=str(p.dtype)) for n,p in self.params]),indent=2),encoding='utf-8')

        def on_pre_optimizer_step(self,args,state,control,**kw):
            if state.global_step+1 in (1,5,20,40):
                self.before={n:p.detach().cpu().clone() for n,p in self.params}
                self.grads={n:None if p.grad is None else float(p.grad.detach().float().norm()) for n,p in self.params}

        def on_step_end(self,args,state,control,**kw):
            if self.before is None:
                return
            rows=[]
            for n,p in self.params:
                delta=p.detach().cpu().float()-self.before[n].float()
                rows.append(dict(name=n,grad_norm=self.grads[n],changed_elements=int((delta!=0).sum()),
                                 max_abs_delta=float(delta.abs().max())))
            with (output/'scope_update_audit.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(dict(step=state.global_step,tensors=rows))+'\n')
            self.before=None

    Native=base.GRPOTrainer
    class Audited(Native):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw)
            self.add_callback(Audit())
    base.GRPOTrainer=Audited
    save=base.save_run_config
    def save_config(output_dir,args,dataset_size,expected_optimizer_steps,warmup_steps):
        save(output_dir,args,dataset_size,expected_optimizer_steps,warmup_steps)
        path=Path(output_dir)/'run_config.json'
        payload=json.loads(path.read_text())
        payload.pop('vision_lora_target_modules',None)
        payload.pop('vision_lora_trainable_tensors',None)
        payload.update(scope=extra.scope,lora_target_modules=len(targets),lora_trainable_tensors=2*len(targets),
                       scope_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        path.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    base.save_run_config=save_config
    run=base.run_training
    def run_training(args):
        nonlocal output
        output=Path(args.output_dir)
        print('Inherited Vision-only banners are generic; scope inventory is authoritative.',flush=True)
        return run(args)
    base.run_training=run_training
    sys.argv=[sys.argv[0],*remaining]
    single.main()


if __name__=='__main__':
    main()
