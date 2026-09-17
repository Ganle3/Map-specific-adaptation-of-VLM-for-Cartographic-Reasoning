"""Streaming CPU measurements of actual optimizer-batch gradients."""
from collections import Counter, deque
import csv
import json
from pathlib import Path

import numpy as np


def dot64(left, right):
    return float(np.einsum('i,i->', left, right, dtype=np.float64))


def gradient_pair(left, right, eps=1e-12):
    left, right = np.asarray(left), np.asarray(right)
    if left.ndim != 1 or left.shape != right.shape:
        raise ValueError('Expected equal-length flat gradients')
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError('Nonfinite gradient')
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError('eps must be positive and finite')
    a, b = np.sqrt(dot64(left, left)), np.sqrt(dot64(right, right))
    dot = dot64(left, right)
    valid = a >= eps and b >= eps
    return dict(cosine=float(np.clip(dot/(a*b), -1, 1)) if valid else float('nan'),
                dot_product=dot, distance=float(np.sqrt(max(0, a*a+b*b-2*dot))), valid_pair=bool(valid))


class TrajectoryRecorder:
    """Bounded CPU history. Caller transfers ownership of an independent CPU vector.

    CSVs flush per update attempt. Cross-step pairs use different parameter states.
    """
    def __init__(self, output, metadata, history_steps=11, eps=1e-12):
        if history_steps < 1 or not np.isfinite(eps) or eps <= 0:
            raise ValueError('Invalid history/eps')
        if len(metadata) not in (20, 44):
            raise ValueError('Expected 20 or 44 unique QAs')
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.metadata, self.eps = metadata, eps
        self.steps_per_epoch = len(metadata)//4
        self.history = deque(maxlen=history_steps)
        self.visits, self.last_seen, self.epoch_seen = Counter(), {}, Counter()
        self.step = 0
        self.handles, self.writers = {}, {}

    def _write(self, name, row):
        if name not in self.writers:
            self.handles[name] = (self.output/f'{name}.csv').open('x', newline='', encoding='utf-8')
            self.writers[name] = csv.DictWriter(self.handles[name], fieldnames=list(row))
            self.writers[name].writeheader()
        self.writers[name].writerow(row)
        self.handles[name].flush()

    def record(self, step, gradient, groups, loss, **extra):
        if step != self.step+1:
            raise ValueError('Expected consecutive optimizer update attempts')
        ids = [g['qa_id'] for g in groups]
        if len(ids) != 4 or len(set(ids)) != 4 or not set(ids) <= self.metadata.keys():
            raise ValueError('Each optimizer batch must contain four distinct known QAs')
        rewards = np.asarray([g['rewards'] for g in groups], dtype=float)
        if rewards.shape != (4, 4) or not np.isin(rewards, [0, 1]).all():
            raise ValueError('Expected four groups of four binary rewards')
        gradient = np.asarray(gradient)
        if gradient.ndim != 1 or not np.isfinite(gradient).all():
            raise ValueError('Invalid flat gradient')
        norm = float(np.sqrt(dot64(gradient, gradient)))
        epoch = (step-1)//self.steps_per_epoch+1
        active = rewards.std(axis=1) > 0
        pairs = []
        for previous_step, previous, previous_ids in self.history:
            row = dict(step=step, previous_step=previous_step, lag=step-previous_step,
                       shared_qa_count=len(set(ids) & set(previous_ids)),
                       **gradient_pair(gradient, previous, self.eps))
            self._write('gradient_pairs', row)
            pairs.append(row)
        for i, qid in enumerate(ids):
            meta = self.metadata[qid]
            self.visits[qid] += 1
            self.epoch_seen[qid] += 1
            self._write('qa_metrics', dict(step=step, epoch=epoch, qa_id=qid,
                visit=self.visits[qid], steps_since_previous=step-self.last_seen[qid] if qid in self.last_seen else '',
                country=meta.get('country', ''), template_no=meta.get('template_no', ''),
                ability_level=meta.get('ability_level', ''), ground_truth_type=meta.get('ground_truth_type', ''),
                rewards=json.dumps(rewards[i].astype(int).tolist()), mean_reward=float(rewards[i].mean()),
                reward_active=bool(active[i])))
            self.last_seen[qid] = step
        if step % self.steps_per_epoch == 0:
            if self.epoch_seen != Counter({qid: 1 for qid in self.metadata}):
                raise RuntimeError('Sampler did not visit each QA exactly once in this epoch')
            self.epoch_seen.clear()
        valid = [r['cosine'] for r in pairs if r['valid_pair']]
        batch = dict(step=step, epoch=epoch, epoch_progress=step/self.steps_per_epoch,
                     qa_ids=json.dumps(ids), loss=float(loss), mean_reward=float(rewards.mean()),
                     reward_std=float(rewards.std()), signal_group_count=int(active.sum()),
                     signal_fraction=float(active.mean()), gradient_norm=norm,
                     gradient_is_zero=norm < self.eps, valid_gradient=norm >= self.eps,
                     num_trainable_params=gradient.size,
                     template_composition=json.dumps(dict(Counter(str(self.metadata[q]['template_no']) for q in ids))),
                     valid_pair_count=len(valid), negative_cosine_fraction=float(np.mean(np.array(valid) < 0)) if valid else float('nan'),
                     adjacent_cosine=pairs[-1]['cosine'] if pairs else float('nan'), **extra)
        self._write('batch_metrics', batch)
        self.history.append((step, gradient, ids))
        self.step = step
        return batch

    def close(self):
        for handle in self.handles.values():
            handle.close()
