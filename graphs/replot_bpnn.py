'''
Re-generate plots and metrics from an existing BPNN evaluation session.

Reads a static session folder and a rest session folder, merges them, and
produces a unified analysis across all 4 classes.

Confusion matrices:
  confusion_matrix_raw/smoothed.png         — hold phase (steady state)
  confusion_matrix_all_phases_raw/smoothed  — all phases (tr-in + hold + tr-out + rest)

Latency / stability (computed from confidence traces as onset proxy):
  Onset defined as first sample in transition_in where conf[correct] > conf[rest].
  Latency = onset → first STABLE_N consecutive correct smoothed predictions.
  Time-to-stable = trial start → first STABLE_N consecutive correct smoothed predictions.

Usage:
  python replot_bpnn.py --static <path> --rest <path>
  python replot_bpnn.py --static inference_eval_bpnn/2026-03-18_static \\
                        --rest   inference_eval_bpnn/2026-03-18_rest
  python replot_bpnn.py --static <path> --rest <path> --out graphs/my_output
  python replot_bpnn.py --static <path> --rest <path> --plot-trial 3
'''

import sys
import os
import csv
import json
import argparse
from collections import defaultdict
from datetime import datetime
from unittest.mock import MagicMock

# ── Mock hardware so evaluate_realtime_bpnn can be imported offline ────────────
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
        import torch      # noqa: F401
        import torch.nn   # noqa: F401
    except ImportError:
        class _FakeTensor:
            pass
        _torch_mock = MagicMock()
        _torch_mock.Tensor  = _FakeTensor
        _torch_mock.device  = MagicMock(return_value='cpu')
        _torch_mock.no_grad = MagicMock()
        sys.modules['torch']    = _torch_mock
        sys.modules['torch.nn'] = MagicMock()

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SERVER_DIR = os.path.join(_SCRIPT_DIR, '..', 'server')
sys.path.insert(0, os.path.abspath(_SERVER_DIR))

import warnings
warnings.filterwarnings('ignore', message='.*single label.*')
warnings.filterwarnings('ignore', message='.*y_pred contains classes not in y_true.*')

from evaluate_realtime_bpnn import (      # noqa: E402
    CLASSES, GROUP_TO_INT, REST_IDX,
    GROUP_VARIANTS, GROUPS_ORDERED,
    N_SAMPLE_TRIALS, SMOOTH_N,
    compute_metrics,
    plot_confidence_histogram,
    plot_group_accuracy, plot_subclass_accuracy,
    plot_timeline, _write_csv,
)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             classification_report, balanced_accuracy_score)

# ── Constants ─────────────────────────────────────────────────────────────────

STABLE_N   = 3     # consecutive correct smoothed predictions = "stable"
STRIDE_SEC = 20 / 200  # 100ms between predictions (STRIDE=20 @ 200Hz)

CLASS_COLORS = {
    'cylindrical': 'steelblue',
    'lateral':     'darkorange',
    'palm':        'seagreen',
    'rest':        'gray',
}
PHASE_COLORS = {
    'transition_in':  '#2196F3',
    'hold':           '#4CAF50',
    'transition_out': '#FF5722',
}

# ── CSV loader ────────────────────────────────────────────────────────────────

def _safe_float(val, default=0.0):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def load_records(csv_path):
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
            mav_raw = row.get('mav_max', '')
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
                'mav_max':       _safe_float(mav_raw) if mav_raw.strip() else None,
            })
    return records


def _merge_rest_records(gesture_records, rest_records):
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


def _group_by_trial(records):
    groups = defaultdict(list)
    for r in records:
        groups[r['trial']].append(r)
    for trial in groups:
        groups[trial].sort(key=lambda r: r['t'])
    return groups


# ── Confusion matrices ────────────────────────────────────────────────────────

def plot_confusion(y_true, y_pred, title, path):
    '''Full 4×4 confusion matrix — all classes always shown.'''
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


# ── Per-class metrics ─────────────────────────────────────────────────────────

def plot_per_class_metrics(y_true, y_pred, path):
    '''Grouped bar chart: precision / recall / F1 for each of the 4 classes.'''
    report = classification_report(
        y_true, y_pred,
        labels=range(len(CLASSES)), target_names=CLASSES,
        output_dict=True, zero_division=0)

    metrics_keys = ['precision', 'recall', 'f1-score']
    colors       = ['steelblue', 'darkorange', 'seagreen']
    x     = np.arange(len(CLASSES))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (key, color) in enumerate(zip(metrics_keys, colors)):
        vals = [report[cls].get(key, 0) for cls in CLASSES]
        bars = ax.bar(x + (i - 1) * width, vals, width,
                      label=key.replace('-score', ''), color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    bal_acc = balanced_accuracy_score(y_true, y_pred)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES)
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.18)
    ax.set_title(f'Per-class metrics — hold phase, all 4 classes\n'
                 f'balanced accuracy = {bal_acc:.3f}')
    ax.axhline(0.25, color='gray', linestyle=':', alpha=0.5, label='chance (1/4)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  Saved {path}')


# ── Latency / stability helpers ───────────────────────────────────────────────

def _baseline_mav(gesture_records, n_trials=5):
    '''Estimate MAV baseline (mean + 3*std) from the first N rest-phase records.

    Uses transition_out records (hand returning to rest) as a proxy for
    baseline activity when no dedicated rest MAV data is available.
    Falls back to a fixed default if insufficient data.
    '''
    rest_mavs = [r['mav_max'] for r in gesture_records
                 if r['phase'] == 'transition_out'
                 and r['mav_max'] is not None][:n_trials * 20]
    if len(rest_mavs) < 5:
        return None
    arr = np.array(rest_mavs)
    return float(arr.mean() + 3 * arr.std())


def _detect_onset(tr_in_recs, true_idx, mav_threshold=None):
    '''Detect EMG onset in a transition_in sequence.

    If mav_max is present in records and mav_threshold is provided, uses a
    3-SD threshold on peak-channel MAV — the standard signal-based method.

    Falls back to confidence proxy (conf[correct] > conf[rest]) for older
    CSVs without the mav_max column.

    Returns record index or None.
    '''
    has_mav = any(r['mav_max'] is not None for r in tr_in_recs)

    if has_mav and mav_threshold is not None:
        # Signal-based onset: MAV crosses baseline + 3*std threshold
        for i, r in enumerate(tr_in_recs):
            if r['mav_max'] is not None and r['mav_max'] > mav_threshold:
                return i
    else:
        # Confidence proxy fallback (biased late — see docstring in module)
        for i, r in enumerate(tr_in_recs):
            if r['proba'][true_idx] > r['proba'][REST_IDX]:
                return i
    return None


def _detect_stable(ordered_recs, true_idx):
    '''First index in ordered_recs where STABLE_N consecutive smoothed
    predictions equal true_idx. Returns record index or None.
    '''
    for i in range(len(ordered_recs) - STABLE_N + 1):
        if all(r['smoothed_pred'] == true_idx
               for r in ordered_recs[i:i + STABLE_N]):
            return i
    return None


def compute_latency_and_stability(gesture_records):
    '''For each gesture trial:
      - onset_t       : EMG onset (MAV threshold if available, else confidence proxy)
      - stable_t      : first sample where STABLE_N consecutive smoothed preds are correct
      - latency       : stable_t - onset_t  (model response after EMG onset)
      - time_to_stable: stable_t - trial_start_t  (total time from trial start)
      - hold_stability: fraction of hold-phase smoothed predictions that are correct
    '''
    mav_threshold  = _baseline_mav(gesture_records)
    onset_method   = 'mav_threshold' if mav_threshold is not None else 'confidence_proxy'
    trial_groups   = _group_by_trial(gesture_records)
    latencies      = []
    times_to_stable = []
    stabilities    = []
    n_no_onset     = 0
    n_no_stable    = 0
    example_trial  = None   # first trial with detected onset, for plotting

    for trial_num in sorted(trial_groups):
        recs     = trial_groups[trial_num]
        true_idx = recs[0]['true']
        if true_idx == REST_IDX:
            continue

        tr_in = [r for r in recs if r['phase'] == 'transition_in']
        hold  = [r for r in recs if r['phase'] == 'hold']

        if not tr_in:
            n_no_onset += 1
            continue

        onset_i = _detect_onset(tr_in, true_idx, mav_threshold)
        if onset_i is None:
            n_no_onset += 1
            continue

        onset_t       = tr_in[onset_i]['t']
        trial_start_t = recs[0]['t']

        # Stable detection over transition_in + hold
        active = tr_in + hold
        stable_i = _detect_stable(active, true_idx)
        if stable_i is None:
            n_no_stable += 1
        else:
            stable_t = active[stable_i]['t']
            lat = stable_t - onset_t
            tts = stable_t - trial_start_t
            if lat >= 0:   # guard against clock jitter
                latencies.append(lat)
                times_to_stable.append(tts)

        # Hold stability
        if hold:
            correct = sum(1 for r in hold if r['smoothed_pred'] == true_idx)
            stabilities.append(correct / len(hold))

        if example_trial is None:
            example_trial = (trial_num, recs, true_idx, onset_i,
                             None if stable_i is None else active[stable_i]['t'])

    return {
        'latencies':       latencies,
        'times_to_stable': times_to_stable,
        'stabilities':     stabilities,
        'n_no_onset':      n_no_onset,
        'n_no_stable':     n_no_stable,
        'example_trial':   example_trial,
        'onset_method':    onset_method,
        'mav_threshold':   mav_threshold,
    }


# ── Latency / stability plots ─────────────────────────────────────────────────

def plot_latency_distribution(data, path):
    lats = np.array(data['latencies']) * 1000   # convert to ms
    if len(lats) == 0:
        print('  Skipped latency plot (no valid trials)')
        return
    mean_l = lats.mean()
    std_l  = lats.std()
    worst  = lats.max()

    method = data.get('onset_method', 'confidence_proxy')
    if method == 'mav_threshold':
        onset_label = f'MAV threshold (baseline + 3σ = {data["mav_threshold"]:.3f})'
    else:
        onset_label = 'confidence proxy: conf[correct] > conf[rest]  (no mav_max in CSV)'

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(lats, bins=20, color='steelblue', alpha=0.8, edgecolor='white')
    ax.axvline(mean_l, color='tomato',   linestyle='--', linewidth=1.5,
               label=f'mean = {mean_l:.0f} ms')
    ax.axvline(worst,  color='firebrick', linestyle=':',  linewidth=1.5,
               label=f'worst = {worst:.0f} ms')
    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('Trial count')
    ax.set_title(
        f'Onset-to-detection latency  '
        f'(mean={mean_l:.0f} ms, std={std_l:.0f} ms, worst={worst:.0f} ms)\n'
        f'Onset: {onset_label}  |  '
        f'n={len(lats)} trials  '
        f'({data["n_no_onset"]} no onset, {data["n_no_stable"]} no stable detection)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  Saved {path}')


def plot_time_to_stable(data, path):
    tts = np.array(data['times_to_stable']) * 1000   # ms
    if len(tts) == 0:
        print('  Skipped time-to-stable plot (no valid trials)')
        return
    mean_t = tts.mean()
    std_t  = tts.std()
    worst  = tts.max()

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(tts, bins=20, color='seagreen', alpha=0.8, edgecolor='white')
    ax.axvline(mean_t, color='tomato',   linestyle='--', linewidth=1.5,
               label=f'mean = {mean_t:.0f} ms')
    ax.axvline(worst,  color='firebrick', linestyle=':',  linewidth=1.5,
               label=f'worst = {worst:.0f} ms')
    ax.set_xlabel('Time from trial start (ms)')
    ax.set_ylabel('Trial count')
    ax.set_title(
        f'Time to stable prediction from trial start\n'
        f'mean={mean_t:.0f} ms, std={std_t:.0f} ms, worst={worst:.0f} ms  '
        f'(stable = {STABLE_N} consecutive correct smoothed predictions)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  Saved {path}')


def plot_stability_distribution(data, path):
    stabs = np.array(data['stabilities'])
    if len(stabs) == 0:
        print('  Skipped stability plot (no hold data)')
        return
    mean_s = stabs.mean()
    worst  = stabs.min()

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(stabs, bins=20, range=(0, 1), color='darkorange', alpha=0.8, edgecolor='white')
    ax.axvline(mean_s, color='tomato',   linestyle='--', linewidth=1.5,
               label=f'mean = {mean_s:.3f}')
    ax.axvline(worst,  color='firebrick', linestyle=':',  linewidth=1.5,
               label=f'worst = {worst:.3f}')
    ax.set_xlabel('Fraction of hold predictions correct (smoothed)')
    ax.set_ylabel('Trial count')
    ax.set_title(
        f'Hold-phase prediction stability\n'
        f'mean={mean_s:.3f}, std={stabs.std():.3f}, worst case={worst:.3f}  '
        f'(n={len(stabs)} trials)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  Saved {path}')


# ── Trial confidence plot ─────────────────────────────────────────────────────

def plot_trial_confidence(gesture_records, trial_num, path):
    '''Plot model confidence traces over a single trial with onset and stable markers.

    Since raw EMG is not stored in predictions.csv, confidence traces serve as
    the model's view of the signal — they track EMG onset and decay closely.
    '''
    trial_groups = _group_by_trial(gesture_records)

    if trial_num not in trial_groups:
        available = sorted(trial_groups.keys())
        print(f'  Warning: trial {trial_num} not found. '
              f'Available: {available[:5]}{"..." if len(available) > 5 else ""}')
        if not available:
            return
        trial_num = available[0]

    recs     = trial_groups[trial_num]
    true_idx = recs[0]['true']
    cls_name = CLASSES[true_idx]

    t0 = recs[0]['t']
    t_rel = np.array([(r['t'] - t0) for r in recs])

    # Detect onset and stable for this trial
    tr_in  = [r for r in recs if r['phase'] == 'transition_in']
    hold   = [r for r in recs if r['phase'] == 'hold']
    active = tr_in + hold

    onset_i  = _detect_onset(tr_in, true_idx) if tr_in else None
    stable_i = _detect_stable(active, true_idx) if active else None

    onset_t  = (tr_in[onset_i]['t']  - t0) if onset_i  is not None else None
    stable_t = (active[stable_i]['t'] - t0) if stable_i is not None else None

    # Phase boundary times
    phase_seq = [r['phase'] for r in recs]
    phase_boundaries = {}
    for phase in ['transition_in', 'hold', 'transition_out']:
        indices = [i for i, p in enumerate(phase_seq) if p == phase]
        if indices:
            phase_boundaries[phase] = (t_rel[indices[0]], t_rel[indices[-1]])

    has_mav = any(r['mav_max'] is not None for r in recs)
    mav_threshold = _baseline_mav(gesture_records)

    if has_mav:
        fig, (ax_mav, ax) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                          gridspec_kw={'height_ratios': [1, 2]})
        # MAV trace (top panel)
        mavs = [r['mav_max'] if r['mav_max'] is not None else 0.0 for r in recs]
        ax_mav.plot(t_rel, mavs, color='saddlebrown', linewidth=1.2, alpha=0.85,
                    label='MAV (peak channel, normalised)')
        if mav_threshold is not None:
            ax_mav.axhline(mav_threshold, color='black', linestyle='--', linewidth=1.0,
                           label=f'onset threshold ({mav_threshold:.3f})')
        if onset_t is not None:
            ax_mav.axvline(onset_t, color='black', linestyle='-', linewidth=1.5)
        for phase, (t_start, t_end) in phase_boundaries.items():
            ax_mav.axvspan(t_start, t_end, alpha=0.08,
                           color=PHASE_COLORS.get(phase, 'white'))
        ax_mav.set_ylabel('MAV (norm.)')
        ax_mav.legend(fontsize=8)
    else:
        fig, ax = plt.subplots(figsize=(12, 5))

    # Shade phases (bottom / only panel)
    for phase, (t_start, t_end) in phase_boundaries.items():
        ax.axvspan(t_start, t_end, alpha=0.08, color=PHASE_COLORS.get(phase, 'white'),
                   label=phase.replace('_', '-'))
        ax.axvline(t_start, color='gray', linestyle='--', linewidth=0.7, alpha=0.5)

    # Confidence traces
    for cls_i, cls in enumerate(CLASSES):
        confs = [r['proba'][cls_i] for r in recs]
        lw    = 2.0 if cls_i == true_idx else 1.0
        alpha = 1.0 if cls_i == true_idx else 0.6
        ax.plot(t_rel, confs, color=CLASS_COLORS[cls], linewidth=lw,
                alpha=alpha, label=cls)

    # Onset marker
    onset_label = ('MAV threshold' if (has_mav and mav_threshold is not None)
                   else f'conf[{cls_name}] > conf[rest]')
    if onset_t is not None:
        ax.axvline(onset_t, color='black', linestyle='-', linewidth=1.5,
                   label=f'onset ({onset_label})')

    # Stable prediction marker
    if stable_t is not None:
        ax.axvline(stable_t, color='purple', linestyle='-.', linewidth=1.5,
                   label=f'stable ({STABLE_N} consec. correct)')

    ax.set_xlabel('Time from trial start (s)')
    ax.set_ylabel('Class probability')
    ax.set_ylim(-0.02, 1.05)
    subtitle = ('MAV onset detection' if has_mav
                else 'No MAV data — confidence proxy used for onset')
    ax.set_title(
        f'Prediction confidence over time — trial {trial_num} ({cls_name})  [{subtitle}]')
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  Saved {path}')


# ── Unified metrics ───────────────────────────────────────────────────────────

def compute_unified_metrics(hold_recs, gesture_records, all_phase_recs, infer_ms_all):
    y_true   = np.array([r['true']          for r in hold_recs])
    y_raw    = np.array([r['raw_pred']      for r in hold_recs])
    y_smooth = np.array([r['smoothed_pred'] for r in hold_recs])

    report_raw    = classification_report(
        y_true, y_raw,
        labels=range(len(CLASSES)), target_names=CLASSES,
        output_dict=True, zero_division=0)
    report_smooth = classification_report(
        y_true, y_smooth,
        labels=range(len(CLASSES)), target_names=CLASSES,
        output_dict=True, zero_division=0)

    cm_hold_raw    = confusion_matrix(y_true, y_raw,
                                      labels=range(len(CLASSES)), normalize='true').tolist()
    cm_hold_smooth = confusion_matrix(y_true, y_smooth,
                                      labels=range(len(CLASSES)), normalize='true').tolist()

    yap_true   = np.array([r['true']          for r in all_phase_recs])
    yap_raw    = np.array([r['raw_pred']      for r in all_phase_recs])
    yap_smooth = np.array([r['smoothed_pred'] for r in all_phase_recs])
    cm_all_raw    = confusion_matrix(yap_true, yap_raw,
                                     labels=range(len(CLASSES)), normalize='true').tolist()
    cm_all_smooth = confusion_matrix(yap_true, yap_smooth,
                                     labels=range(len(CLASSES)), normalize='true').tolist()

    # False activation (hold phase only)
    rest_mask = y_true == REST_IDX
    n_rest    = int(rest_mask.sum())
    fa_raw    = int((y_raw[rest_mask]    != REST_IDX).sum()) if rest_mask.any() else 0
    fa_smooth = int((y_smooth[rest_mask] != REST_IDX).sum()) if rest_mask.any() else 0

    # Cross-gesture confusions during transitions
    def _cross_gesture(recs, phase, pred_key):
        pr = [r for r in recs if r['phase'] == phase and r['true'] != REST_IDX]
        return (sum(1 for r in pr if r[pred_key] != r['true'] and r[pred_key] != REST_IDX),
                len(pr))

    cg_in_raw,     n_tr_in  = _cross_gesture(gesture_records, 'transition_in',  'raw_pred')
    cg_in_smooth,  _        = _cross_gesture(gesture_records, 'transition_in',  'smoothed_pred')
    cg_out_raw,    n_tr_out = _cross_gesture(gesture_records, 'transition_out', 'raw_pred')
    cg_out_smooth, _        = _cross_gesture(gesture_records, 'transition_out', 'smoothed_pred')

    return {
        'balanced_acc_raw':              float(balanced_accuracy_score(y_true, y_raw)),
        'balanced_acc_smoothed':         float(balanced_accuracy_score(y_true, y_smooth)),
        'false_activation_raw':          fa_raw,
        'false_activation_smooth':       fa_smooth,
        'n_rest_hold_predictions':       n_rest,
        'cross_gesture_transition_in_raw':     cg_in_raw,
        'cross_gesture_transition_in_smooth':  cg_in_smooth,
        'n_transition_in_predictions':         n_tr_in,
        'cross_gesture_transition_out_raw':    cg_out_raw,
        'cross_gesture_transition_out_smooth': cg_out_smooth,
        'n_transition_out_predictions':        n_tr_out,
        'per_class_raw':                 report_raw,
        'per_class_smoothed':            report_smooth,
        'confusion_hold_raw':            cm_hold_raw,
        'confusion_hold_smooth':         cm_hold_smooth,
        'confusion_all_phases_raw':      cm_all_raw,
        'confusion_all_phases_smooth':   cm_all_smooth,
        'n_hold_predictions':            len(hold_recs),
        'n_all_phase_predictions':       len(all_phase_recs),
        'mean_infer_ms':                 float(infer_ms_all.mean()),
        'std_infer_ms':                  float(infer_ms_all.std()),
        'class_order':                   CLASSES,
    }


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
                        help='Output folder '
                             '(default: graphs/inference_eval_bpnn_merged/<timestamp>)')
    parser.add_argument('--plot-trial', type=int, default=None,
                        help='Trial number to use for confidence trace plot '
                             '(default: first trial with detected onset)')
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

    # ── Build record sets ─────────────────────────────────────────────────────
    merged_records = _merge_rest_records(gesture_records, rest_records)
    hold_recs      = [r for r in merged_records if r['phase'] == 'hold']

    # All phases: gesture (all phases) + rest hold records
    all_phase_recs = gesture_records + [
        r for r in merged_records if r['sub_class'] == 'rest'
    ]

    infer_ms_all = np.array([r['infer_ms'] for r in merged_records])

    y_true_hold   = [r['true']          for r in hold_recs]
    y_raw_hold    = [r['raw_pred']      for r in hold_recs]
    y_smooth_hold = [r['smoothed_pred'] for r in hold_recs]

    yap_true   = [r['true']          for r in all_phase_recs]
    yap_raw    = [r['raw_pred']      for r in all_phase_recs]
    yap_smooth = [r['smoothed_pred'] for r in all_phase_recs]

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics   = compute_unified_metrics(hold_recs, gesture_records,
                                        all_phase_recs, infer_ms_all)
    g_metrics = compute_metrics(gesture_records)
    lat_data  = compute_latency_and_stability(gesture_records)

    lats = np.array(lat_data['latencies'])   * 1000  # ms
    tts  = np.array(lat_data['times_to_stable']) * 1000
    stabs = np.array(lat_data['stabilities'])

    metrics['gesture_phases'] = {
        'hold_acc_per_group':              g_metrics.get('hold_acc_per_group', {}),
        'transition_in_acc_per_group':     g_metrics.get('transition_in_acc_per_group', {}),
        'transition_out_acc_per_group':    g_metrics.get('transition_out_acc_per_group', {}),
        'hold_acc_per_subclass':           g_metrics.get('hold_acc_per_subclass', {}),
        'transition_in_acc_per_subclass':  g_metrics.get('transition_in_acc_per_subclass', {}),
        'transition_out_acc_per_subclass': g_metrics.get('transition_out_acc_per_subclass', {}),
        'transition_out_rest_acc':         g_metrics.get('transition_out_rest_acc'),
    }
    metrics['latency'] = {
        'mean_ms':   float(lats.mean())  if len(lats)  else None,
        'std_ms':    float(lats.std())   if len(lats)  else None,
        'worst_ms':  float(lats.max())   if len(lats)  else None,
        'n_trials':  len(lats),
        'n_no_onset':  lat_data['n_no_onset'],
        'n_no_stable': lat_data['n_no_stable'],
        'stable_n':  STABLE_N,
    }
    metrics['time_to_stable'] = {
        'mean_ms':  float(tts.mean())  if len(tts)  else None,
        'std_ms':   float(tts.std())   if len(tts)  else None,
        'worst_ms': float(tts.max())   if len(tts)  else None,
    }
    metrics['hold_stability'] = {
        'mean':  float(stabs.mean()) if len(stabs) else None,
        'std':   float(stabs.std())  if len(stabs) else None,
        'worst': float(stabs.min())  if len(stabs) else None,
    }

    json_path = os.path.join(out_dir, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'  Saved {json_path}')

    # ── Confusion matrices ────────────────────────────────────────────────────
    print('\n── Confusion matrices ────────────────────────────────────')

    plot_confusion(
        y_true_hold, y_raw_hold,
        'Confusion matrix — raw (hold / steady state, all 4 classes)',
        os.path.join(out_dir, 'confusion_matrix_raw.png'))
    plot_confusion(
        y_true_hold, y_smooth_hold,
        f'Confusion matrix — smoothed n={SMOOTH_N} (hold / steady state, all 4 classes)',
        os.path.join(out_dir, 'confusion_matrix_smoothed.png'))

    plot_confusion(
        yap_true, yap_raw,
        'Confusion matrix — raw (all phases incl. transitions)',
        os.path.join(out_dir, 'confusion_matrix_all_phases_raw.png'))
    plot_confusion(
        yap_true, yap_smooth,
        f'Confusion matrix — smoothed n={SMOOTH_N} (all phases incl. transitions)',
        os.path.join(out_dir, 'confusion_matrix_all_phases_smoothed.png'))

    # ── Per-class metrics ─────────────────────────────────────────────────────
    print('\n── Per-class metrics ─────────────────────────────────────')
    plot_per_class_metrics(
        y_true_hold, y_raw_hold,
        os.path.join(out_dir, 'per_class_metrics.png'))

    plot_confidence_histogram(
        merged_records,
        os.path.join(out_dir, 'confidence_histogram.png'))

    # ── Latency / stability ───────────────────────────────────────────────────
    print('\n── Latency and stability ─────────────────────────────────')
    plot_latency_distribution(lat_data,
                              os.path.join(out_dir, 'latency_distribution.png'))
    plot_time_to_stable(lat_data,
                        os.path.join(out_dir, 'time_to_stable.png'))
    plot_stability_distribution(lat_data,
                                os.path.join(out_dir, 'stability_distribution.png'))

    # ── Trial confidence plot ─────────────────────────────────────────────────
    print('\n── Trial confidence trace ────────────────────────────────')
    if args.plot_trial is not None:
        plot_trial_num = args.plot_trial
    elif lat_data['example_trial'] is not None:
        plot_trial_num = lat_data['example_trial'][0]
    else:
        plot_trial_num = min(_group_by_trial(gesture_records).keys())
    plot_trial_confidence(gesture_records, plot_trial_num,
                          os.path.join(out_dir, 'trial_confidence_example.png'))

    # ── Gesture phase breakdown ───────────────────────────────────────────────
    print('\n── Gesture-phase plots ───────────────────────────────────')
    plot_group_accuracy(g_metrics,    os.path.join(out_dir, 'group_accuracy.png'))
    plot_subclass_accuracy(g_metrics, os.path.join(out_dir, 'subclass_accuracy.png'))
    plot_timeline(gesture_records,    os.path.join(out_dir, 'timeline.png'))

    # ── Merged CSV ────────────────────────────────────────────────────────────
    csv_path = _write_csv(merged_records, out_dir)
    print(f'  Saved {csv_path}')

    # ── Summary ───────────────────────────────────────────────────────────────
    n_rest   = metrics['n_rest_hold_predictions']
    n_tr_in  = metrics['n_transition_in_predictions']
    n_tr_out = metrics['n_transition_out_predictions']

    print()
    print('── Summary ───────────────────────────────────────────────')
    print(f'  Balanced accuracy  (raw)     : {metrics["balanced_acc_raw"]:.3f}')
    print(f'  Balanced accuracy  (smoothed): {metrics["balanced_acc_smoothed"]:.3f}')
    print()
    print(f'  False activations  (raw)     : {metrics["false_activation_raw"]} / {n_rest}'
          f'  ← rest hold misfired as gesture')
    print(f'  False activations  (smoothed): {metrics["false_activation_smooth"]} / {n_rest}')
    print(f'  Cross-gesture tr-in  (raw)   : {metrics["cross_gesture_transition_in_raw"]} / {n_tr_in}'
          f'  ← wrong gesture during initiation')
    print(f'  Cross-gesture tr-in  (smooth): {metrics["cross_gesture_transition_in_smooth"]} / {n_tr_in}')
    print(f'  Cross-gesture tr-out (raw)   : {metrics["cross_gesture_transition_out_raw"]} / {n_tr_out}'
          f'  ← wrong gesture during release')
    print(f'  Cross-gesture tr-out (smooth): {metrics["cross_gesture_transition_out_smooth"]} / {n_tr_out}')
    print()

    if len(lats):
        print(f'  Onset-to-detection latency   : {lats.mean():.0f} ± {lats.std():.0f} ms'
              f'  (worst: {lats.max():.0f} ms,  n={len(lats)})')
    if len(tts):
        print(f'  Time to stable prediction    : {tts.mean():.0f} ± {tts.std():.0f} ms'
              f'  (worst: {tts.max():.0f} ms,  stable={STABLE_N} consec.)')
    if len(stabs):
        print(f'  Hold stability               : {stabs.mean():.3f} ± {stabs.std():.3f}'
              f'  (worst trial: {stabs.min():.3f})')
    print(f'  Mean inference time          : {metrics["mean_infer_ms"]:.1f}'
          f' ± {metrics["std_infer_ms"]:.1f} ms')
    print(f'  Hold predictions             : {metrics["n_hold_predictions"]}'
          f'  (gesture: {metrics["n_hold_predictions"] - n_rest}, rest: {n_rest})')

    print()
    print(f'  {"class":<14}  {"precision":>9}  {"recall":>6}  {"f1":>6}')
    print('  ' + '─' * 44)
    for cls in CLASSES:
        r = metrics['per_class_raw'].get(cls, {})
        print(f'  {cls:<14}  {r.get("precision", 0):>9.3f}'
              f'  {r.get("recall", 0):>6.3f}  {r.get("f1-score", 0):>6.3f}')

    print()
    print(f'  {"group":<14}  {"hold":>6}  {"tr-in":>6}  {"tr-out":>7}')
    print('  ' + '─' * 40)
    gp = metrics['gesture_phases']
    for group in GROUPS_ORDERED:
        h  = gp['hold_acc_per_group'].get(group, float('nan'))
        ti = gp['transition_in_acc_per_group'].get(group, float('nan'))
        to = gp['transition_out_acc_per_group'].get(group, float('nan'))
        print(f'  {group:<14}  {h:>6.3f}  {ti:>6.3f}  {to:>7.3f}')

    print('\nDone.')


if __name__ == '__main__':
    main()
