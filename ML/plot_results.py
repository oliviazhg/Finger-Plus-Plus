'''
Plot Results Script

Generates visualisations from a trained model's results directory.

Always produced (from results.json + fold_scores.npy):
  per_class_metrics.png   — precision / recall / F1 per class
  fold_scores.png         — per-fold balanced accuracy (regenerated)

Produced if cv_results.json exists (saved by train_model_all_phases.py):
  param_n_estimators.png  — score vs n_estimators (marginalised over other params)
  param_max_features.png  — score vs max_features
  param_max_depth.png     — score vs max_depth
  param_min_samples_leaf.png — score vs min_samples_leaf
  param_heatmap.png       — n_estimators × max_depth mean score heatmap

Usage:
  python plot_results.py                    # defaults to results_all_phases/
  python plot_results.py results_steady
'''

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import defaultdict

RESULTS_DIR = sys.argv[1] if len(sys.argv) > 1 else 'results_all_phases'

# ── Load ──────────────────────────────────────────────────────────────────────

with open(os.path.join(RESULTS_DIR, 'results.json')) as f:
    results = json.load(f)

fold_scores = np.load(os.path.join(RESULTS_DIR, 'fold_scores.npy'))
classes     = results['class_order']
report      = results['classification_report']
best_params = results['best_params']

cv_results_path = os.path.join(RESULTS_DIR, 'cv_results.json')
cv_results = None
if os.path.exists(cv_results_path):
    with open(cv_results_path) as f:
        cv_results = json.load(f)
    print('cv_results.json found — parameter comparison plots will be generated.')
else:
    print('cv_results.json not found — only metric plots will be generated.')
    print('Re-run train_model_all_phases.py to produce it on the next training run.')


# ── Helpers ───────────────────────────────────────────────────────────────────

def save(fig, name):
    path = os.path.join(RESULTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


# ── Per-class metrics ─────────────────────────────────────────────────────────

def plot_per_class_metrics():
    metrics   = ['precision', 'recall', 'f1-score']
    n_classes = len(classes)
    x         = np.arange(n_classes)
    width     = 0.25
    colors    = ['#4C72B0', '#55A868', '#C44E52']

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        vals = [report[cls][metric] for cls in classes]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=metric.replace('-score', ''),
                      color=color, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel('Score')
    ax.set_ylim(0.88, 1.01)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.set_title('Per-class precision / recall / F1  (CV, train set)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    save(fig, 'per_class_metrics.png')


# ── Fold scores ───────────────────────────────────────────────────────────────

def plot_fold_scores():
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(1, len(fold_scores) + 1), fold_scores, color='steelblue', alpha=0.85)
    ax.axhline(fold_scores.mean(), color='red', linestyle='--',
               label=f'Mean {fold_scores.mean():.3f}  ±  {fold_scores.std():.3f}')
    for i, s in enumerate(fold_scores):
        ax.text(i + 1, s + 0.001, f'{s:.3f}', ha='center', va='bottom', fontsize=8)
    ax.set_xlabel('Fold')
    ax.set_ylabel('Balanced accuracy')
    ax.set_ylim(0.9, 1.0)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.set_title('Per-fold balanced accuracy (trial-aware CV)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    save(fig, 'fold_scores.png')


# ── Parameter comparison ──────────────────────────────────────────────────────

def _mean_score_by_param(param_key):
    '''
    For each unique value of param_key, collect all mean_test_scores
    (marginalised over the other parameters) and return value → scores dict.
    '''
    param_col = f'param_{param_key}'
    scores    = np.array(cv_results['mean_test_score'])
    values    = cv_results[param_col]
    grouped   = defaultdict(list)
    for v, s in zip(values, scores):
        grouped[str(v)].append(s)
    return grouped


def plot_param_marginal(param_key, xlabel, title, fname):
    grouped = _mean_score_by_param(param_key)
    labels  = sorted(grouped.keys(), key=lambda x: (x == 'None', x))
    means   = [np.mean(grouped[l]) for l in labels]
    stds    = [np.std(grouped[l])  for l in labels]

    # highlight best
    best_str = str(best_params.get(param_key, ''))
    colors   = ['#C44E52' if l == best_str else '#4C72B0' for l in labels]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, means, yerr=stds, capsize=5, color=colors, alpha=0.85,
                  error_kw={'elinewidth': 1.5})
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f'{m:.4f}', ha='center', va='bottom', fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Mean balanced accuracy (CV)')
    bottom = max(0.0, min(means) - 3 * max(stds) - 0.01)
    ax.set_ylim(bottom, min(1.0, max(means) + max(stds) + 0.02))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.set_title(title)
    ax.grid(axis='y', alpha=0.3)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color='#C44E52', label='best selected'),
                        Patch(color='#4C72B0', label='other')], fontsize=8)
    save(fig, fname)


def plot_heatmap():
    '''n_estimators (rows) × max_depth (cols) mean score, averaged over other params.'''
    estimators = sorted(set(str(v) for v in cv_results['param_n_estimators']))
    depths     = sorted(set(str(v) for v in cv_results['param_max_depth']),
                        key=lambda x: float('inf') if x == 'None' else float(x))
    scores_grid = np.zeros((len(estimators), len(depths)))
    counts_grid = np.zeros_like(scores_grid)

    for e_str, d_str, s in zip(
        [str(v) for v in cv_results['param_n_estimators']],
        [str(v) for v in cv_results['param_max_depth']],
        cv_results['mean_test_score']
    ):
        r, c = estimators.index(e_str), depths.index(d_str)
        scores_grid[r, c] += s
        counts_grid[r, c] += 1

    mean_grid = scores_grid / np.maximum(counts_grid, 1)

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(mean_grid, cmap='YlOrRd', aspect='auto',
                   vmin=mean_grid.min() - 0.002, vmax=mean_grid.max() + 0.002)
    plt.colorbar(im, ax=ax, label='Mean balanced accuracy')
    ax.set_xticks(range(len(depths)))
    ax.set_xticklabels([f'depth={d}' for d in depths])
    ax.set_yticks(range(len(estimators)))
    ax.set_yticklabels([f'n={e}' for e in estimators])
    for r in range(len(estimators)):
        for c in range(len(depths)):
            ax.text(c, r, f'{mean_grid[r, c]:.4f}', ha='center', va='center', fontsize=9,
                    color='black' if mean_grid[r, c] < 0.96 else 'white')
    ax.set_title('Mean CV score: n_estimators × max_depth\n(averaged over max_features, min_samples_leaf)')
    save(fig, 'param_heatmap.png')


# ── Run ───────────────────────────────────────────────────────────────────────

print(f'\n── Plotting results in {RESULTS_DIR}/ ───────────────────')
plot_per_class_metrics()
plot_fold_scores()

if cv_results is not None:
    plot_param_marginal('n_estimators',     'n_estimators',     'Effect of n_estimators (marginalised)',     'param_n_estimators.png')
    plot_param_marginal('max_features',     'max_features',     'Effect of max_features (marginalised)',     'param_max_features.png')
    plot_param_marginal('max_depth',        'max_depth',        'Effect of max_depth (marginalised)',        'param_max_depth.png')
    plot_param_marginal('min_samples_leaf', 'min_samples_leaf', 'Effect of min_samples_leaf (marginalised)', 'param_min_samples_leaf.png')
    plot_heatmap()

print('\nDone.')
