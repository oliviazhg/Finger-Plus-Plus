'''
Re-generate plots and metrics from an existing RF evaluation session.

Reads predictions.csv from the specified folder, recomputes all metrics
using the current code, and writes/overwrites all plots.

Works for static, dynamic, and rest sessions.

Usage:
  python replot_rf.py --dir inference_eval_rf/2026-03-16_12-00-00
  python replot_rf.py --dir inference_eval_rf/2026-03-16_12-00-00 --out /some/other/dir
'''

import sys
import os
import csv
import json
import argparse
from unittest.mock import MagicMock

# ── Mock hardware so evaluate_realtime_rf can be imported offline ─────────────
# (pyomyo requires a connected Myo; all hardware code is inside functions,
#  so mocking the module is safe for import-only use)
for _mod in ('pyomyo', 'struct'):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()
sys.modules['pyomyo'].Myo        = MagicMock()
sys.modules['pyomyo'].emg_mode   = MagicMock()

# Add server/ dir to path so the import works regardless of cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate_realtime_rf import (      # noqa: E402
    CLASSES, GROUP_TO_INT, REST_IDX,
    GROUP_VARIANTS, GROUPS_ORDERED, DYNAMIC_VARIANTS,
    N_SAMPLE_TRIALS, SMOOTH_N,
    compute_metrics, compute_dynamic_metrics,
    plot_confusion_matrix, plot_confidence_histogram,
    plot_group_accuracy, plot_subclass_accuracy, plot_dynamic_accuracy,
    plot_timeline, _write_csv,
)

# ── Rest-session metrics / plots (from evaluate_rest_rf) ─────────────────────
# Duplicated inline so this script has no extra dependency

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def _compute_rest_metrics(records):
    y_raw    = np.array([r['raw_pred']      for r in records])
    y_smooth = np.array([r['smoothed_pred'] for r in records])
    infer_ms = np.array([r['infer_ms']      for r in records])
    rest_idx = GROUP_TO_INT['rest']

    trials = sorted({r['trial'] for r in records})
    per_trial_acc = {
        t: float((np.array([r['raw_pred'] for r in records if r['trial'] == t])
                  == rest_idx).mean())
        for t in trials
    }
    pred_counts = {c: int((y_raw == i).sum()) for i, c in enumerate(CLASSES)}
    cm_row = confusion_matrix(
        [rest_idx] * len(records), y_raw,
        labels=range(len(CLASSES)), normalize='true'
    ).tolist()[rest_idx]

    return {
        'rest_acc_raw':      float((y_raw    == rest_idx).mean()),
        'rest_acc_smooth':   float((y_smooth == rest_idx).mean()),
        'per_trial_acc':     per_trial_acc,
        'pred_distribution': pred_counts,
        'confusion_row':     cm_row,
        'mean_infer_ms':     float(infer_ms.mean()),
        'std_infer_ms':      float(infer_ms.std()),
        'n_predictions':     len(records),
        'n_trials':          len(trials),
        'class_order':       CLASSES,
    }


def _plot_confusion_rest(metrics, path):
    row    = metrics['confusion_row']
    x      = np.arange(len(CLASSES))
    colors = ['steelblue' if c == 'rest' else 'tomato' for c in CLASSES]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x, row, color=colors, alpha=0.85, edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES)
    ax.set_ylabel('Fraction of predictions')
    ax.set_ylim(0, 1.1)
    ax.set_title(
        f'Predicted class distribution during rest\n'
        f'raw accuracy: {metrics["rest_acc_raw"]:.3f}  '
        f'smoothed: {metrics["rest_acc_smooth"]:.3f}'
    )
    for xi, v in enumerate(row):
        ax.text(xi, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  Saved {path}')


def _plot_per_trial_accuracy(metrics, path):
    per_trial = metrics['per_trial_acc']
    trials    = sorted(per_trial)
    accs      = [per_trial[t] for t in trials]
    mean_acc  = float(np.mean(accs))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(trials, accs, marker='o', color='steelblue', markersize=5)
    ax.axhline(mean_acc, color='tomato', linestyle='--',
               label=f'mean = {mean_acc:.3f}')
    ax.set_xlabel('Trial')
    ax.set_ylabel('Rest accuracy (raw)')
    ax.set_ylim(-0.05, 1.1)
    ax.set_title('Rest accuracy per trial')
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  Saved {path}')


# ── CSV loader ────────────────────────────────────────────────────────────────

def load_records(csv_path):
    '''Reconstruct internal record dicts from a predictions.csv file.'''
    records = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            true_group = row.get('true_group', 'rest')
            raw_pred   = row.get('raw_pred',   'rest')
            sm_pred    = row.get('smoothed_pred', raw_pred)

            true_idx = CLASSES.index(true_group) if true_group in CLASSES else REST_IDX
            raw_idx  = CLASSES.index(raw_pred)   if raw_pred   in CLASSES else REST_IDX
            sm_idx   = CLASSES.index(sm_pred)    if sm_pred    in CLASSES else REST_IDX

            proba = [
                float(row.get('conf_cyl',  0)),
                float(row.get('conf_lat',  0)),
                float(row.get('conf_palm', 0)),
                float(row.get('conf_rest', 0)),
            ]
            records.append({
                't':             float(row.get('t', 0)),
                'trial':         int(row.get('trial', 0)),
                'phase':         row.get('phase', 'hold'),
                'sub_class':     row.get('sub_class', ''),
                'true':          true_idx,
                'raw_pred':      raw_idx,
                'smoothed_pred': sm_idx,
                'proba':         proba,
                'infer_ms':      float(row.get('infer_ms', 0)),
            })
    return records


def detect_mode(records):
    '''Return "static", "dynamic", or "rest" based on phases in records.'''
    phases = {r['phase'] for r in records}
    if 'dynamic' in phases:
        return 'dynamic'
    if 'hold' in phases or 'transition_in' in phases:
        # check if there are sub_class values (static eval vs rest eval)
        has_subclass = any(r['sub_class'] for r in records)
        return 'static' if has_subclass else 'rest'
    return 'rest'


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Re-generate RF evaluation plots from an existing session folder'
    )
    parser.add_argument('--dir', required=True,
                        help='Path to the existing session folder containing predictions.csv')
    parser.add_argument('--out', default=None,
                        help='Output folder for plots (default: same as --dir)')
    args = parser.parse_args()

    csv_path = os.path.join(args.dir, 'predictions.csv')
    if not os.path.exists(csv_path):
        print(f'Error: {csv_path} not found.')
        return

    out_dir = args.out or args.dir
    os.makedirs(out_dir, exist_ok=True)

    print(f'Loading {csv_path}...')
    records = load_records(csv_path)
    print(f'  {len(records)} predictions loaded')

    mode = detect_mode(records)
    print(f'  Detected mode: {mode}')

    # ── Static mode ───────────────────────────────────────────────────────────
    if mode == 'static':
        metrics = compute_metrics(records)

        json_path = os.path.join(out_dir, 'results.json')
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f'  Saved {json_path}')

        hold_m = metrics.get('hold') or {}
        if hold_m.get('raw_cm'):
            plot_confusion_matrix(
                hold_m['raw_cm'],
                'Confusion matrix — raw predictions (hold phase)',
                os.path.join(out_dir, 'confusion_matrix_raw.png'))
        if hold_m.get('smoothed_cm'):
            plot_confusion_matrix(
                hold_m['smoothed_cm'],
                f'Confusion matrix — smoothed (n={SMOOTH_N}, hold phase)',
                os.path.join(out_dir, 'confusion_matrix_smoothed.png'))

        plot_confidence_histogram(records, os.path.join(out_dir, 'confidence_histogram.png'))
        plot_group_accuracy(metrics,       os.path.join(out_dir, 'group_accuracy.png'))
        plot_subclass_accuracy(metrics,    os.path.join(out_dir, 'subclass_accuracy.png'))
        plot_timeline(records,             os.path.join(out_dir, 'timeline.png'))

        print()
        print('── Summary ───────────────────────────────────────────')
        print(f'  Hold raw balanced acc.      : {hold_m.get("raw_balanced_acc", float("nan")):.3f}')
        print(f'  Hold smoothed balanced acc. : {hold_m.get("smoothed_balanced_acc", float("nan")):.3f}')
        print(f'  Mean inference time         : {metrics["mean_infer_ms"]:.1f} ± {metrics["std_infer_ms"]:.1f} ms')
        print(f'  Total predictions           : {metrics["n_predictions"]}')
        print()
        print(f'  Sub-class accuracy  (n≤{N_SAMPLE_TRIALS} trials/variant sampled)')
        print(f'  {"sub-class":<36}  {"hold":>6}  {"tr-in":>6}  {"tr-out":>7}  {"trials":>6}')
        print('  ' + '─' * 70)
        for group in GROUPS_ORDERED:
            for variant in GROUP_VARIANTS[group]:
                n_trials = len({r['trial'] for r in records
                                if r['sub_class'] == variant and r['phase'] == 'hold'})
                h  = metrics['hold_acc_per_subclass'].get(variant, float('nan'))
                ti = metrics['transition_in_acc_per_subclass'].get(variant, float('nan'))
                to = metrics['transition_out_acc_per_subclass'].get(variant, float('nan'))
                print(f'  {variant:<36}  {h:>6.3f}  {ti:>6.3f}  {to:>7.3f}  {n_trials:>6}')
            print()

        tr_out_overall = metrics.get('transition_out_rest_acc')
        if tr_out_overall is not None:
            print(f'  Overall transition-out → rest : {tr_out_overall:.3f}')

        print()
        print(f'  {"group":<14}  {"hold":>6}  {"tr-in":>6}  {"tr-out":>7}  (mean of variant accs)')
        print('  ' + '─' * 50)
        for group in GROUPS_ORDERED:
            h  = metrics['hold_acc_per_group'].get(group, float('nan'))
            ti = metrics['transition_in_acc_per_group'].get(group, float('nan'))
            to = metrics['transition_out_acc_per_group'].get(group, float('nan'))
            print(f'  {group:<14}  {h:>6.3f}  {ti:>6.3f}  {to:>7.3f}')

    # ── Dynamic mode ──────────────────────────────────────────────────────────
    elif mode == 'dynamic':
        metrics = compute_dynamic_metrics(records)

        json_path = os.path.join(out_dir, 'results.json')
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f'  Saved {json_path}')

        if metrics.get('raw_cm'):
            plot_confusion_matrix(
                metrics['raw_cm'],
                'Confusion matrix — dynamic mode (raw)',
                os.path.join(out_dir, 'confusion_matrix_dynamic.png'))
        plot_dynamic_accuracy(metrics, os.path.join(out_dir, 'dynamic_accuracy.png'))

        print()
        print('── Summary ───────────────────────────────────────────')
        print(f'  Raw balanced acc.   : {metrics["raw_balanced_acc"]:.3f}')
        print(f'  Mean inference time : {metrics["mean_infer_ms"]:.1f} ± {metrics["std_infer_ms"]:.1f} ms')
        print(f'  Total predictions   : {metrics["n_predictions"]}')
        print()
        print(f'  {"position":<42}  {"acc":>5}')
        print('  ' + '─' * 52)
        for group in GROUPS_ORDERED:
            for pos in DYNAMIC_VARIANTS[group]:
                acc = metrics['dynamic_acc_per_position'].get(pos, float('nan'))
                print(f'  {pos:<42}  {acc:>5.3f}')
            print()

    # ── Rest mode ─────────────────────────────────────────────────────────────
    elif mode == 'rest':
        metrics = _compute_rest_metrics(records)

        json_path = os.path.join(out_dir, 'results.json')
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f'  Saved {json_path}')

        _plot_confusion_rest(metrics,     os.path.join(out_dir, 'confusion_rest.png'))
        _plot_per_trial_accuracy(metrics, os.path.join(out_dir, 'accuracy_per_trial.png'))

        print()
        print('── Summary ───────────────────────────────────────────')
        print(f'  Rest accuracy (raw)     : {metrics["rest_acc_raw"]:.3f}')
        print(f'  Rest accuracy (smoothed): {metrics["rest_acc_smooth"]:.3f}')
        print(f'  Mean inference time     : {metrics["mean_infer_ms"]:.1f} ± {metrics["std_infer_ms"]:.1f} ms')
        print(f'  Total predictions       : {metrics["n_predictions"]}  ({metrics["n_trials"]} trials)')
        print()
        print('  Predicted class distribution:')
        for cls, count in metrics['pred_distribution'].items():
            pct = count / metrics['n_predictions'] * 100 if metrics['n_predictions'] else 0
            bar = '█' * int(pct / 2)
            print(f'    {cls:<12}  {count:>4}  ({pct:>5.1f}%)  {bar}')

    print('\nDone.')


if __name__ == '__main__':
    main()
