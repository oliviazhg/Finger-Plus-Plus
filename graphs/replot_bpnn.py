'''
Re-generate plots and metrics from an existing BPNN evaluation session.

Reads predictions.csv from a static session folder and a rest session folder,
merges them, and produces all plots with all 4 classes represented.

The merged confusion matrices always use labels=range(len(CLASSES)) so every
class appears even if it was absent from one session.

Outputs are saved to graphs/inference_eval_bpnn_merged/<timestamp>/ by default
(alongside this script), or to --out if specified.

Usage:
  python replot_bpnn.py --static <path> --rest <path>
  python replot_bpnn.py --static inference_eval_bpnn/2026-03-18_static --rest inference_eval_bpnn/2026-03-18_rest
  python replot_bpnn.py --static <path> --rest <path> --out graphs/my_output
'''

import sys
import os
import csv
import json
import argparse
from datetime import datetime
from unittest.mock import MagicMock

# ── Mock hardware so evaluate_realtime_bpnn can be imported offline ────────────
# Only mock modules that aren't actually installed. Mocking torch when scipy is
# present breaks scipy's is_torch_array check (issubclass against a MagicMock).

if 'pyomyo' not in sys.modules:
    try:
        import pyomyo  # noqa: F401
    except ImportError:
        _pyomyo_mock = MagicMock()
        _pyomyo_mock.Myo      = MagicMock()
        _pyomyo_mock.emg_mode = MagicMock()
        sys.modules['pyomyo'] = _pyomyo_mock

if 'torch' not in sys.modules:
    try:
        import torch  # noqa: F401
        import torch.nn  # noqa: F401
    except ImportError:
        # Provide a real class for Tensor so scipy's issubclass check doesn't crash
        class _FakeTensor:
            pass
        _torch_mock = MagicMock()
        _torch_mock.Tensor   = _FakeTensor
        _torch_mock.device   = MagicMock(return_value='cpu')
        _torch_mock.no_grad  = MagicMock()
        sys.modules['torch']    = _torch_mock
        sys.modules['torch.nn'] = MagicMock()

# Add server/ to path so the import works regardless of cwd
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SERVER_DIR = os.path.join(_SCRIPT_DIR, '..', 'server')
sys.path.insert(0, os.path.abspath(_SERVER_DIR))

import warnings
warnings.filterwarnings('ignore', message='.*single label.*')
warnings.filterwarnings('ignore', message='.*y_pred contains classes not in y_true.*')

from evaluate_realtime_bpnn import (      # noqa: E402
    CLASSES, GROUP_TO_INT, REST_IDX,
    GROUP_VARIANTS, GROUPS_ORDERED, DYNAMIC_VARIANTS,
    N_SAMPLE_TRIALS, SMOOTH_N, TRIALS_PER_GROUP,
    compute_metrics, compute_rest_metrics,
    plot_confusion_matrix, plot_confidence_histogram,
    plot_group_accuracy, plot_subclass_accuracy,
    plot_confusion_rest, plot_per_trial_accuracy,
    plot_timeline, _write_csv,
)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# ── CSV loader ────────────────────────────────────────────────────────────────

def _safe_float(val, default=0.0):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def load_records(csv_path):
    '''Reconstruct internal record dicts from a predictions.csv file.'''
    records = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if _safe_float(row.get('t'), None) is None:
                continue

            true_group = row.get('true_group', 'rest')
            raw_pred   = row.get('raw_pred',   'rest')
            sm_pred    = row.get('smoothed_pred', raw_pred)

            true_idx = CLASSES.index(true_group) if true_group in CLASSES else REST_IDX
            raw_idx  = CLASSES.index(raw_pred)   if raw_pred   in CLASSES else REST_IDX
            sm_idx   = CLASSES.index(sm_pred)    if sm_pred    in CLASSES else REST_IDX

            proba = [
                _safe_float(row.get('conf_cyl',  0)),
                _safe_float(row.get('conf_lat',  0)),
                _safe_float(row.get('conf_palm', 0)),
                _safe_float(row.get('conf_rest', 0)),
            ]
            records.append({
                't':             _safe_float(row.get('t', 0)),
                'trial':         int(_safe_float(row.get('trial', 0))),
                'phase':         row.get('phase', 'hold'),
                'sub_class':     row.get('sub_class', ''),
                'true':          true_idx,
                'raw_pred':      raw_idx,
                'smoothed_pred': sm_idx,
                'proba':         proba,
                'infer_ms':      _safe_float(row.get('infer_ms', 0)),
            })
    return records


def _merge_rest_records(gesture_records, rest_records):
    '''Normalise rest records to look like hold-phase records and append.

    trial numbers are offset to avoid collisions with gesture trial numbers.
    '''
    trial_offset = (max(r['trial'] for r in gesture_records) + 1
                    if gesture_records else 0)
    normalised = []
    for r in rest_records:
        nr = dict(r)
        nr['phase']     = 'hold'
        nr['sub_class'] = 'rest'
        nr['trial']     = r['trial'] + trial_offset
        normalised.append(nr)
    return gesture_records + normalised


def _plot_confusion_merged(y_true, y_pred, title, path):
    '''Full 4×4 confusion matrix (all classes always shown).'''
    cm = confusion_matrix(y_true, y_pred,
                          labels=range(len(CLASSES)), normalize='true')
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(
        ax=ax, colorbar=True, cmap='Blues', values_format='.2f')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  Saved {path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Re-generate BPNN evaluation plots from a static + rest session pair'
    )
    parser.add_argument('--static', required=True,
                        help='Path to static session folder (predictions.csv inside)')
    parser.add_argument('--rest', required=True,
                        help='Path to rest session folder (predictions.csv inside)')
    parser.add_argument('--out', default=None,
                        help='Output folder (default: graphs/inference_eval_bpnn_merged/<timestamp>)')
    args = parser.parse_args()

    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    # ── Load ──────────────────────────────────────────────────────────────────
    static_csv = os.path.join(args.static, 'predictions.csv')
    rest_csv   = os.path.join(args.rest,   'predictions.csv')

    for p in [static_csv, rest_csv]:
        if not os.path.exists(p):
            print(f'Error: {p} not found.')
            return

    print(f'Loading static session: {static_csv}')
    gesture_records = load_records(static_csv)
    print(f'  {len(gesture_records)} predictions loaded')

    print(f'Loading rest session  : {rest_csv}')
    rest_records = load_records(rest_csv)
    print(f'  {len(rest_records)} predictions loaded')

    # ── Output dir ────────────────────────────────────────────────────────────
    if args.out:
        out_dir = args.out
    else:
        ts      = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        out_dir = os.path.join(_SCRIPT_DIR, 'inference_eval_bpnn_merged', ts)
    os.makedirs(out_dir, exist_ok=True)
    print(f'\nOutput → {out_dir}/\n')

    # ── Part 1: merged 4×4 confusion matrices (hold phase, all classes) ───────
    merged_records = _merge_rest_records(gesture_records, rest_records)
    hold_recs = [r for r in merged_records if r['phase'] == 'hold']
    y_true   = [r['true']          for r in hold_recs]
    y_raw    = [r['raw_pred']      for r in hold_recs]
    y_smooth = [r['smoothed_pred'] for r in hold_recs]

    print('── Merged confusion matrices (all 4 classes) ─────────────')
    _plot_confusion_merged(
        y_true, y_raw,
        'Confusion matrix — merged (raw, hold phase)',
        os.path.join(out_dir, 'confusion_matrix_merged_raw.png'))
    _plot_confusion_merged(
        y_true, y_smooth,
        f'Confusion matrix — merged (smoothed n={SMOOTH_N}, hold phase)',
        os.path.join(out_dir, 'confusion_matrix_merged_smoothed.png'))

    # ── Part 2: gesture-only analysis ─────────────────────────────────────────
    print('\n── Gesture analysis ──────────────────────────────────────')
    g_metrics = compute_metrics(gesture_records)

    json_path = os.path.join(out_dir, 'results_gesture.json')
    with open(json_path, 'w') as f:
        json.dump(g_metrics, f, indent=2)
    print(f'  Saved {json_path}')

    hold_m = g_metrics.get('hold') or {}

    # Gesture confusion matrices use target_names so absent classes are dropped
    if hold_m.get('raw_cm'):
        plot_confusion_matrix(
            hold_m['raw_cm'],
            'Confusion matrix — gestures only, raw (hold phase)',
            os.path.join(out_dir, 'confusion_matrix_gesture_raw.png'),
            display_labels=hold_m.get('target_names'))
    if hold_m.get('smoothed_cm'):
        plot_confusion_matrix(
            hold_m['smoothed_cm'],
            f'Confusion matrix — gestures only, smoothed n={SMOOTH_N} (hold phase)',
            os.path.join(out_dir, 'confusion_matrix_gesture_smoothed.png'),
            display_labels=hold_m.get('target_names'))

    plot_confidence_histogram(gesture_records, os.path.join(out_dir, 'confidence_histogram.png'))
    plot_group_accuracy(g_metrics,             os.path.join(out_dir, 'group_accuracy.png'))
    plot_subclass_accuracy(g_metrics,          os.path.join(out_dir, 'subclass_accuracy.png'))
    plot_timeline(gesture_records,             os.path.join(out_dir, 'timeline.png'))

    # ── Part 3: rest-only analysis ────────────────────────────────────────────
    print('\n── Rest analysis ─────────────────────────────────────────')
    r_metrics = compute_rest_metrics(rest_records)

    json_path = os.path.join(out_dir, 'results_rest.json')
    with open(json_path, 'w') as f:
        json.dump(r_metrics, f, indent=2)
    print(f'  Saved {json_path}')

    plot_confusion_rest(r_metrics,     os.path.join(out_dir, 'confusion_rest.png'))
    plot_per_trial_accuracy(r_metrics, os.path.join(out_dir, 'accuracy_per_trial.png'))

    # ── Write merged CSV ──────────────────────────────────────────────────────
    csv_path = _write_csv(merged_records, out_dir)
    print(f'  Saved {csv_path}')

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print('── Summary ───────────────────────────────────────────────')
    print(f'  Gesture hold raw balanced acc.      : {hold_m.get("raw_balanced_acc", float("nan")):.3f}')
    print(f'  Gesture hold smoothed balanced acc. : {hold_m.get("smoothed_balanced_acc", float("nan")):.3f}')
    print(f'  Gesture predictions                 : {g_metrics["n_predictions"]}')
    print(f'  Rest accuracy (raw)                 : {r_metrics["rest_acc_raw"]:.3f}')
    print(f'  Rest accuracy (smoothed)            : {r_metrics["rest_acc_smooth"]:.3f}')
    print(f'  Rest predictions                    : {r_metrics["n_predictions"]}  ({r_metrics["n_trials"]} trials)')
    print(f'  Total predictions                   : {len(merged_records)}')
    infer_ms_all = np.array([r['infer_ms'] for r in merged_records])
    print(f'  Mean inference time                 : {infer_ms_all.mean():.1f} ± {infer_ms_all.std():.1f} ms')

    print()
    print(f'  Sub-class accuracy  (n≤{N_SAMPLE_TRIALS} trials/variant sampled)')
    print(f'  {"sub-class":<36}  {"hold":>6}  {"tr-in":>6}  {"tr-out":>7}  {"trials":>6}')
    print('  ' + '─' * 70)
    for group in GROUPS_ORDERED:
        for variant in GROUP_VARIANTS[group]:
            n_trials = len({r['trial'] for r in gesture_records
                            if r['sub_class'] == variant and r['phase'] == 'hold'})
            h  = g_metrics['hold_acc_per_subclass'].get(variant, float('nan'))
            ti = g_metrics['transition_in_acc_per_subclass'].get(variant, float('nan'))
            to = g_metrics['transition_out_acc_per_subclass'].get(variant, float('nan'))
            print(f'  {variant:<36}  {h:>6.3f}  {ti:>6.3f}  {to:>7.3f}  {n_trials:>6}')
        print()

    print(f'  {"group":<14}  {"hold":>6}  {"tr-in":>6}  {"tr-out":>7}  (mean of variant accs)')
    print('  ' + '─' * 50)
    for group in GROUPS_ORDERED:
        h  = g_metrics['hold_acc_per_group'].get(group, float('nan'))
        ti = g_metrics['transition_in_acc_per_group'].get(group, float('nan'))
        to = g_metrics['transition_out_acc_per_group'].get(group, float('nan'))
        print(f'  {group:<14}  {h:>6.3f}  {ti:>6.3f}  {to:>7.3f}')

    print()
    print('  Predicted class distribution during rest:')
    for cls, count in r_metrics['pred_distribution'].items():
        pct = count / r_metrics['n_predictions'] * 100 if r_metrics['n_predictions'] else 0
        bar = '█' * int(pct / 2)
        print(f'    {cls:<12}  {count:>4}  ({pct:>5.1f}%)  {bar}')

    print('\nDone.')


if __name__ == '__main__':
    main()
