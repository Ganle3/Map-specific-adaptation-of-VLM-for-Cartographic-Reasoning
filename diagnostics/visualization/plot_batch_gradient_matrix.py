"""Plot online diagnostic CSVs locally or on Euler; no torch/TRL/WandB required."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path):
    if not path.exists():
        return []
    with path.open(encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def grouped_mean(rows, group, value):
    buckets = defaultdict(list)
    for row in rows:
        number = float(row[value])
        if np.isfinite(number):
            buckets[int(row[group])].append(number)
    keys = sorted(buckets)
    return keys, [float(np.mean(buckets[k])) for k in keys]


def finish(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_results(results_dir):
    root = Path(results_dir)
    batches, pairs, qa = [read_csv(root/f'{name}.csv') for name in ('batch_metrics', 'gradient_pairs', 'qa_metrics')]
    if not batches or not qa:
        raise ValueError(f'No recorded batches/QAs in {root}')
    destination = root/'plots'
    destination.mkdir(exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(10, 9))
    x, y = grouped_mean(qa, 'epoch', 'mean_reward')
    axes[0].plot(x, y, marker='.')
    axes[0].set(ylabel='Online sampled correctness', ylim=(-.02, 1.02), xlabel='Epoch')
    steps = [int(r['step']) for r in batches]
    axes[1].plot(steps, [float(r['gradient_norm']) for r in batches])
    axes[1].set(ylabel='Pre-clip LoRA gradient norm', xlabel='Optimizer update')
    axes[2].plot(steps, [float(r['signal_fraction']) for r in batches], alpha=.4, label='Reward-active QA fraction')
    axes[2].plot(steps, [float(r['negative_cosine_fraction']) for r in batches], label='Negative fraction (valid history pairs)')
    axes[2].set(ylim=(-.02, 1.02), xlabel='Optimizer update')
    axes[2].legend()
    fig.suptitle('Training trajectory: sampled rewards, not independent evaluation')
    finish(fig, destination/'training_overview.png')

    ids = list(dict.fromkeys(r['qa_id'] for r in qa))
    id_index = {qid: i for i, qid in enumerate(ids)}
    values = np.full((len(ids), max(int(r['visit']) for r in qa)), np.nan)
    for row in qa:
        values[id_index[row['qa_id']], int(row['visit'])-1] = float(row['mean_reward'])
    fig, ax = plt.subplots(figsize=(12, max(5, len(ids)*.2)))
    im = ax.imshow(values, aspect='auto', vmin=0, vmax=1, cmap='viridis', extent=(.5, values.shape[1]+.5, len(ids)-.5, -.5))
    ax.set(yticks=range(len(ids)), yticklabels=ids, xlabel='QA visit number', title='Per-QA online sampled correctness (4 rollouts/visit)')
    ax.tick_params(axis='y', labelsize=6)
    fig.colorbar(im, ax=ax)
    finish(fig, destination/'qa_learning_heatmap.png')

    max_lag = max([int(r['lag']) for r in pairs], default=1)
    matrix = np.full((max(steps), max_lag), np.nan)
    for row in pairs:
        matrix[int(row['step'])-1, int(row['lag'])-1] = float(row['cosine'])
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad('#bdbdbd')
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(np.ma.masked_invalid(matrix), aspect='auto', cmap=cmap, vmin=-1, vmax=1,
                   extent=(.5, max_lag+.5, max(steps)+.5, .5))
    ax.set(xlabel='Lag in optimizer steps', ylabel='Current step', xticks=range(1, max_lag+1),
           title='Batch gradient cosine across changing model states\nGrey = unavailable or zero gradient')
    fig.colorbar(im, ax=ax)
    finish(fig, destination/'gradient_lag_heatmap.png')

    lag_summary = []
    for lag in range(1, max_lag+1):
        rows = [r for r in pairs if int(r['lag']) == lag]
        valid = [float(r['cosine']) for r in rows if r['valid_pair'] == 'True']
        lag_summary.append(dict(lag=lag, total_pairs=len(rows), valid_pairs=len(valid),
            mean_cosine=float(np.mean(valid)) if valid else float('nan'),
            negative_fraction=float(np.mean(np.array(valid) < 0)) if valid else float('nan')))
    write_csv(root/'lag_summary.csv', lag_summary)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot([r['lag'] for r in lag_summary], [r['mean_cosine'] for r in lag_summary], marker='o')
    axes[0].axhline(0, color='grey', linewidth=.8)
    axes[0].set(xlabel='Step lag', ylabel='Mean cosine', ylim=(-1, 1))
    axes[1].bar([r['lag'] for r in lag_summary], [r['negative_fraction'] for r in lag_summary])
    axes[1].set(xlabel='Step lag', ylabel='Negative fraction of valid pairs', ylim=(0, 1.15))
    for r in lag_summary:
        axes[1].text(r['lag'], 1.03, str(r['valid_pairs']), ha='center', fontsize=7)
    axes[1].set_title('Numbers = valid pair counts')
    finish(fig, destination/'gradient_by_lag.png')

    # Descriptive group labels, not inferred semantic tasks or per-task gradients.
    field = 'ability_level' if all(r['ability_level'] for r in qa) else 'ground_truth_type'
    labels = sorted({r[field] or 'unknown' for r in qa})
    fig, ax = plt.subplots(figsize=(9, 5))
    for label in labels:
        subset = [r for r in qa if (r[field] or 'unknown') == label]
        x, y = grouped_mean(subset, 'epoch', 'mean_reward')
        ax.plot(x, y, label=f"{label} ({len({r['qa_id'] for r in subset})} QA)")
    ax.set(xlabel='Epoch', ylabel='Online sampled correctness', ylim=(-.02, 1.02), title=f'Grouped by {field}; descriptive only')
    ax.legend(fontsize=8)
    finish(fig, destination/'question_group_learning.png')
    summary = []
    for qid in ids:
        rows = [r for r in qa if r['qa_id'] == qid]
        gaps = [int(r['steps_since_previous']) for r in rows if r['steps_since_previous']]
        summary.append(dict(qa_id=qid, visits=len(rows), template_no=rows[0]['template_no'],
            ability_level=rows[0]['ability_level'], ground_truth_type=rows[0]['ground_truth_type'],
            early_mean_reward=float(np.mean([float(r['mean_reward']) for r in rows[:5]])),
            late_mean_reward=float(np.mean([float(r['mean_reward']) for r in rows[-5:]])),
            active_fraction=float(np.mean([r['reward_active']=='True' for r in rows])),
            mean_visit_gap=float(np.mean(gaps)) if gaps else float('nan')))
    write_csv(root/'qa_summary.csv', summary)


def plot_comparison(roots, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for root in map(Path, roots):
        config = json.loads((root/'config.json').read_text(encoding='utf-8'))
        label = f"{config['dataset_size']} QA, seed {config['seed']}"
        qa = read_csv(root/'qa_metrics.csv')
        for ax, group in zip(axes[:2], ('epoch', 'step')):
            x, y = grouped_mean(qa, group, 'mean_reward')
            ax.plot(x, y, label=label, alpha=.85)
            ax.set(xlabel=group.capitalize(), ylabel='Online sampled correctness', ylim=(-.02, 1.02))
        rows = read_csv(root/'lag_summary.csv')
        axes[2].plot([int(r['lag']) for r in rows], [float(r['negative_fraction']) for r in rows], marker='.', label=label)
    axes[2].set(xlabel='Same step lag', ylabel='Negative cosine fraction', ylim=(-.02, 1.02))
    for ax in axes:
        ax.legend(fontsize=8)
    fig.suptitle('Descriptive comparison; different datasets and update budgets, no forgetting inference')
    finish(fig, output/'joint20_vs_joint44.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, nargs='+', required=True)
    parser.add_argument('--comparison-dir', type=Path)
    args = parser.parse_args()
    if len(args.results_dir) > 1 and args.comparison_dir is None:
        parser.error('Provide --comparison-dir when plotting multiple runs')
    for directory in args.results_dir:
        plot_results(directory)
    if len(args.results_dir) > 1:
        plot_comparison(args.results_dir, args.comparison_dir)
