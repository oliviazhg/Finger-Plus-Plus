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
import matplotlib
from matplotlib.patches import Patch

matplotlib.rcParams.update({
    'font.family':     'sans-serif',
    'font.sans-serif': ['Roboto', 'Arial', 'Helvetica', 'DejaVu Sans'],
    'font.weight':     'normal',
})
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


def load_amp_samples(session_dir):
    '''Load emg_amplitude.csv if present. Returns list of dicts or None.'''
    path = os.path.join(session_dir, 'emg_amplitude.csv')
    if not os.path.exists(path):
        return None
    samples = []
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            t    = _safe_float(row.get('t'),    None)
            peak = _safe_float(row.get('peak'), None)
            if t is None or peak is None:
                continue
            samples.append({
                'trial': int(_safe_float(row.get('trial', 0))),
                'phase': row.get('phase', ''),
                't':     t,
                'peak':  peak,
            })
    return samples or None


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

def plot_per_class_metrics(y_true, y_pred, path, title_suffix='hold phase'):
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
    ax.set_title(f'Per-class metrics — {title_suffix}, all 4 classes\n'
                 f'balanced accuracy = {bal_acc:.3f}')
    ax.axhline(0.25, color='gray', linestyle=':', alpha=0.5, label='chance (1/4)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  Saved {path}')


# ── Latency / stability helpers ───────────────────────────────────────────────

AMP_SMOOTH_SAMPLES = 5   # rolling window for per-sample onset smoothing (~25 ms @ 200 Hz)


def _baseline_from_amp(amp_samples, n_samples=None):
    '''Estimate onset threshold (mean + 3*std) from transition_out amp samples.

    transition_out (hand returning to rest after a gesture) is used as a
    baseline proxy. Returns threshold float or None if insufficient data.
    '''
    rest_peaks = [s['peak'] for s in amp_samples if s['phase'] == 'transition_out']
    if n_samples:
        rest_peaks = rest_peaks[:n_samples]
    if len(rest_peaks) < 10:
        return None
    arr = np.array(rest_peaks)
    return float(arr.mean() + 3 * arr.std())


def _baseline_mav(gesture_records, n_trials=5):
    '''Fallback: estimate threshold from prediction-stride MAV values (100 ms resolution).'''
    rest_mavs = [r['mav_max'] for r in gesture_records
                 if r['phase'] == 'transition_out'
                 and r['mav_max'] is not None][:n_trials * 20]
    if len(rest_mavs) < 5:
        return None
    arr = np.array(rest_mavs)
    return float(arr.mean() + 3 * arr.std())


def _detect_onset_from_amp(trial_amp_in, threshold):
    '''High-resolution onset from per-sample amp data (~5 ms resolution).

    Applies a short rolling mean (AMP_SMOOTH_SAMPLES) then finds the first
    sample exceeding the threshold. Returns onset time (float seconds) or None.
    '''
    if not trial_amp_in:
        return None
    peaks = np.array([s['peak'] for s in trial_amp_in])
    # Rolling mean to suppress single-sample noise
    kernel = np.ones(AMP_SMOOTH_SAMPLES) / AMP_SMOOTH_SAMPLES
    smoothed = np.convolve(peaks, kernel, mode='same')
    for i, v in enumerate(smoothed):
        if v > threshold:
            return trial_amp_in[i]['t']
    return None


def _detect_onset(tr_in_recs, true_idx, mav_threshold=None):
    '''Prediction-stride onset fallback (100 ms resolution).

    Used when emg_amplitude.csv is not available. Uses mav_max from prediction
    records if present, otherwise falls back to confidence proxy.
    '''
    has_mav = any(r['mav_max'] is not None for r in tr_in_recs)
    if has_mav and mav_threshold is not None:
        for i, r in enumerate(tr_in_recs):
            if r['mav_max'] is not None and r['mav_max'] > mav_threshold:
                return i
    else:
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


def compute_latency_and_stability(gesture_records, amp_samples=None):
    '''For each gesture trial:
      - onset_t       : EMG onset time, detected at ~5 ms resolution from emg_amplitude.csv
                        when available, else falls back to prediction-stride MAV (100 ms)
                        or confidence proxy.
      - stable_t      : first prediction time where STABLE_N consecutive smoothed preds correct
      - latency       : stable_t - onset_t
      - time_to_stable: stable_t - trial_start_t
      - hold_stability: fraction of hold-phase smoothed predictions that are correct
    '''
    # Build per-trial amp lookup grouped by (trial, phase) if amp data available
    amp_by_trial_phase = defaultdict(list)
    if amp_samples:
        for s in amp_samples:
            amp_by_trial_phase[(s['trial'], s['phase'])].append(s)
        amp_threshold  = _baseline_from_amp(amp_samples)
        onset_method   = 'amp_5ms' if amp_threshold is not None else 'confidence_proxy'
    else:
        amp_threshold  = None
        mav_threshold  = _baseline_mav(gesture_records)
        onset_method   = 'mav_100ms' if mav_threshold is not None else 'confidence_proxy'

    trial_groups    = _group_by_trial(gesture_records)
    latencies       = []
    times_to_stable = []
    stabilities     = []
    n_no_onset      = 0
    n_no_stable     = 0
    example_trial   = None

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

        trial_start_t = recs[0]['t']

        # --- Onset detection ---
        if amp_samples and amp_threshold is not None:
            tr_in_amp = amp_by_trial_phase.get((trial_num, 'transition_in'), [])
            onset_t   = _detect_onset_from_amp(tr_in_amp, amp_threshold)
            onset_i   = None   # not used for amp-based onset
        else:
            thresh    = mav_threshold if not amp_samples else None
            onset_i   = _detect_onset(tr_in, true_idx, thresh)
            onset_t   = tr_in[onset_i]['t'] if onset_i is not None else None

        if onset_t is None:
            n_no_onset += 1
            continue

        # --- Stable detection (prediction-stride resolution) ---
        active   = tr_in + hold
        stable_i = _detect_stable(active, true_idx)
        if stable_i is None:
            n_no_stable += 1
        else:
            stable_t = active[stable_i]['t']
            lat = stable_t - onset_t
            tts = stable_t - trial_start_t
            if lat >= 0:
                latencies.append(lat)
                times_to_stable.append(tts)

        # Hold stability
        if hold:
            correct = sum(1 for r in hold if r['smoothed_pred'] == true_idx)
            stabilities.append(correct / len(hold))

        if example_trial is None:
            example_trial = (trial_num, recs, true_idx,
                             onset_i if onset_i is not None else 0,
                             None if stable_i is None else active[stable_i]['t'])

    threshold_val = amp_threshold if amp_samples else (mav_threshold if not amp_samples else None)
    return {
        'latencies':       latencies,
        'times_to_stable': times_to_stable,
        'stabilities':     stabilities,
        'n_no_onset':      n_no_onset,
        'n_no_stable':     n_no_stable,
        'example_trial':   example_trial,
        'onset_method':    onset_method,
        'mav_threshold':   threshold_val,
    }


# ── Latency / stability plots ─────────────────────────────────────────────────

LATENCY_OUTLIER_MS = 800   # trials above this are excluded from the plot (noted in title)


def plot_latency_distribution(data, path):
    lats_all   = np.array(data['latencies']) * 1000   # convert to ms
    lats       = lats_all[lats_all <= LATENCY_OUTLIER_MS]
    n_excluded = int((lats_all > LATENCY_OUTLIER_MS).sum())
    if len(lats) == 0:
        print('  Skipped latency plot (no valid trials)')
        return
    mean_l = lats_all.mean()   # stats computed on full distribution
    std_l  = lats_all.std()
    worst  = lats_all.max()

    method = data.get('onset_method', 'confidence_proxy')
    if method == 'amp_5ms':
        onset_label = f'per-sample peak amplitude, 25 ms smoothing, threshold={data["mav_threshold"]:.3f}'
    elif method == 'mav_100ms':
        onset_label = f'prediction-stride MAV (100 ms res.), threshold={data["mav_threshold"]:.3f}'
    else:
        onset_label = 'confidence proxy: conf[correct] > conf[rest]  (no amplitude data)'

    def _latency_plot(ax, arr, title_suffix):
        m, s, w = arr.mean(), arr.std(), arr.max()
        ax.hist(arr, bins=20, color='steelblue', alpha=0.8, edgecolor='white')
        ax.axvline(m, color='tomato',   linestyle='--', linewidth=1.5,
                   label=f'mean = {m:.0f} ms')
        ax.axvline(w, color='firebrick', linestyle=':',  linewidth=1.5,
                   label=f'worst = {w:.0f} ms')
        ax.set_xlabel('Latency (ms)')
        ax.set_ylabel('Trial count')
        ax.set_title(
            f'Onset-to-detection latency{title_suffix}  '
            f'(mean={m:.0f} ms, std={s:.0f} ms, worst={w:.0f} ms)\n'
            f'Onset: {onset_label}  |  n={len(arr)} trials  '
            f'({data["n_no_onset"]} no onset, {data["n_no_stable"]} no stable detection)')
        ax.legend()

    # Full distribution
    fig, ax = plt.subplots(figsize=(9, 4))
    _latency_plot(ax, lats_all, '')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  Saved {path}')

    # Outliers removed
    if n_excluded > 0:
        path_filt = path.replace('.png', '_no_outliers.png')
        fig, ax = plt.subplots(figsize=(9, 4))
        _latency_plot(ax, lats, f' (>{LATENCY_OUTLIER_MS} ms excluded, n={n_excluded})')
        plt.tight_layout()
        plt.savefig(path_filt, dpi=150)
        plt.close()
        print(f'  Saved {path_filt}')


TTS_OUTLIER_MS = 1000   # trials above this are excluded from the filtered plot


def plot_time_to_stable(data, path):
    tts_all    = np.array(data['times_to_stable']) * 1000   # ms
    tts        = tts_all[tts_all <= TTS_OUTLIER_MS]
    n_excluded = int((tts_all > TTS_OUTLIER_MS).sum())
    if len(tts_all) == 0:
        print('  Skipped time-to-stable plot (no valid trials)')
        return

    def _tts_plot(ax, arr, title_suffix):
        m, s, w = arr.mean(), arr.std(), arr.max()
        ax.hist(arr, bins=20, color='seagreen', alpha=0.8, edgecolor='white')
        ax.axvline(m, color='tomato',   linestyle='--', linewidth=1.5,
                   label=f'mean = {m:.0f} ms')
        ax.axvline(w, color='firebrick', linestyle=':',  linewidth=1.5,
                   label=f'worst = {w:.0f} ms')
        ax.set_xlabel('Time from trial start (ms)')
        ax.set_ylabel('Trial count')
        ax.set_title(
            f'Time to stable prediction from trial start{title_suffix}\n'
            f'mean={m:.0f} ms, std={s:.0f} ms, worst={w:.0f} ms  '
            f'(stable = {STABLE_N} consecutive correct smoothed predictions)')
        ax.legend()

    # Full distribution
    fig, ax = plt.subplots(figsize=(9, 4))
    _tts_plot(ax, tts_all, '')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  Saved {path}')

    # Outliers removed
    if n_excluded > 0 and len(tts) > 0:
        path_filt = path.replace('.png', '_no_outliers.png')
        fig, ax = plt.subplots(figsize=(9, 4))
        _tts_plot(ax, tts, f' (>{TTS_OUTLIER_MS} ms excluded, n={n_excluded})')
        plt.tight_layout()
        plt.savefig(path_filt, dpi=150)
        plt.close()
        print(f'  Saved {path_filt}')


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

def plot_trial_confidence(gesture_records, trial_num, path, amp_samples=None):
    '''Plot model confidence traces over a single trial with onset and stable markers.

    If amp_samples (from emg_amplitude.csv) are available, the top panel shows
    the per-sample amplitude trace at 5 ms resolution with the onset threshold.
    Otherwise falls back to prediction-stride MAV or confidence proxy.
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

    stable_i = _detect_stable(active, true_idx) if active else None
    stable_t = (active[stable_i]['t'] - t0) if stable_i is not None else None

    # Onset detection — prefer per-sample amp data when available
    if amp_samples:
        amp_threshold = _baseline_from_amp(amp_samples)
        tr_in_amp     = [s for s in amp_samples
                         if s['trial'] == trial_num and s['phase'] == 'transition_in']
        raw_onset_t   = _detect_onset_from_amp(tr_in_amp, amp_threshold) if amp_threshold else None
        onset_t       = (raw_onset_t - t0) if raw_onset_t is not None else None
    else:
        mav_threshold = _baseline_mav(gesture_records)
        onset_i       = _detect_onset(tr_in, true_idx, mav_threshold) if tr_in else None
        onset_t       = (tr_in[onset_i]['t'] - t0) if onset_i is not None else None
        amp_threshold = mav_threshold

    # Phase boundary times
    phase_seq = [r['phase'] for r in recs]
    phase_boundaries = {}
    for phase in ['transition_in', 'hold', 'transition_out']:
        indices = [i for i, p in enumerate(phase_seq) if p == phase]
        if indices:
            phase_boundaries[phase] = (t_rel[indices[0]], t_rel[indices[-1]])

    has_mav     = any(r['mav_max'] is not None for r in recs)
    mav_threshold = amp_threshold if amp_samples else _baseline_mav(gesture_records)

    show_amp_panel = amp_samples or has_mav
    if show_amp_panel:
        fig, (ax_mav, ax) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                          gridspec_kw={'height_ratios': [1, 2]})
        # Top panel: per-sample amplitude trace (preferred) or prediction-stride MAV
        if amp_samples:
            trial_amp = [s for s in amp_samples if s['trial'] == trial_num]
            amp_t     = np.array([s['t'] - t0 for s in trial_amp])
            amp_vals  = np.array([s['peak']    for s in trial_amp])
            # smoothed for display
            kernel    = np.ones(AMP_SMOOTH_SAMPLES) / AMP_SMOOTH_SAMPLES
            amp_smooth = np.convolve(amp_vals, kernel, mode='same')
            ax_mav.plot(amp_t, amp_vals,   color='saddlebrown', linewidth=0.6,
                        alpha=0.4, label='peak amplitude (raw)')
            ax_mav.plot(amp_t, amp_smooth, color='saddlebrown', linewidth=1.4,
                        alpha=0.9, label=f'smoothed ({AMP_SMOOTH_SAMPLES}-sample window)')
            amp_label = 'amplitude (norm.)'
        else:
            mavs = [r['mav_max'] if r['mav_max'] is not None else 0.0 for r in recs]
            ax_mav.plot(t_rel, mavs, color='saddlebrown', linewidth=1.2, alpha=0.85,
                        label='MAV (prediction-stride, 100 ms res.)')
            amp_label = 'MAV (norm.)'
        if mav_threshold is not None:
            ax_mav.axhline(mav_threshold, color='black', linestyle='--', linewidth=1.0,
                           label=f'onset threshold ({mav_threshold:.3f})')
        if onset_t is not None:
            ax_mav.axvline(onset_t, color='black', linestyle='-', linewidth=1.5)
        for phase, (t_start, t_end) in phase_boundaries.items():
            ax_mav.axvspan(t_start, t_end, alpha=0.08,
                           color=PHASE_COLORS.get(phase, 'white'))
        ax_mav.set_ylabel(amp_label)
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
    if amp_samples:
        onset_label = 'per-sample amplitude (5 ms res.)'
    elif has_mav and mav_threshold is not None:
        onset_label = 'prediction-stride MAV (100 ms res.)'
    else:
        onset_label = f'conf[{cls_name}] > conf[rest]'
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

    # Cross-gesture confusions during transitions — total and per-class
    GESTURE_IDXS = [i for i, c in enumerate(CLASSES) if i != REST_IDX]

    def _cross_gesture(recs, phase, pred_key):
        pr = [r for r in recs if r['phase'] == phase and r['true'] != REST_IDX]
        total_cross = sum(1 for r in pr
                          if r[pred_key] != r['true'] and r[pred_key] != REST_IDX)
        # Per-class breakdown: for each gesture class, count correct / wrong gesture / rest
        per_class = {}
        for cls_i in GESTURE_IDXS:
            cls_recs = [r for r in pr if r['true'] == cls_i]
            per_class[CLASSES[cls_i]] = {
                'correct':       sum(1 for r in cls_recs if r[pred_key] == cls_i),
                'wrong_gesture': sum(1 for r in cls_recs
                                     if r[pred_key] != cls_i and r[pred_key] != REST_IDX),
                'predicted_rest': sum(1 for r in cls_recs if r[pred_key] == REST_IDX),
                'total':         len(cls_recs),
            }
        return total_cross, len(pr), per_class

    cg_in_raw,     n_tr_in,  cg_in_raw_per_cls  = _cross_gesture(gesture_records, 'transition_in',  'raw_pred')
    cg_in_smooth,  _,        cg_in_sm_per_cls   = _cross_gesture(gesture_records, 'transition_in',  'smoothed_pred')
    cg_out_raw,    n_tr_out, cg_out_raw_per_cls  = _cross_gesture(gesture_records, 'transition_out', 'raw_pred')
    cg_out_smooth, _,        cg_out_sm_per_cls   = _cross_gesture(gesture_records, 'transition_out', 'smoothed_pred')

    return {
        'balanced_acc_raw':              float(balanced_accuracy_score(y_true, y_raw)),
        'balanced_acc_smoothed':         float(balanced_accuracy_score(y_true, y_smooth)),
        'false_activation_raw':          fa_raw,
        'false_activation_smooth':       fa_smooth,
        'n_rest_hold_predictions':       n_rest,
        'cross_gesture_transition_in_raw':          cg_in_raw,
        'cross_gesture_transition_in_smooth':       cg_in_smooth,
        'cross_gesture_transition_in_raw_per_cls':  cg_in_raw_per_cls,
        'cross_gesture_transition_in_sm_per_cls':   cg_in_sm_per_cls,
        'n_transition_in_predictions':              n_tr_in,
        'cross_gesture_transition_out_raw':         cg_out_raw,
        'cross_gesture_transition_out_smooth':      cg_out_smooth,
        'cross_gesture_transition_out_raw_per_cls': cg_out_raw_per_cls,
        'cross_gesture_transition_out_sm_per_cls':  cg_out_sm_per_cls,
        'n_transition_out_predictions':             n_tr_out,
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


def plot_transition_confusion(metrics, path, smoothed=False):
    '''Stacked bar chart of prediction outcomes during transition phases.

    For each gesture class, shows two grouped bars (tr-in / tr-out).
    Each bar is stacked: correct | predicted rest | wrong gesture.
    Uses raw predictions by default; pass smoothed=True for smoothed.
    '''
    suffix   = 'sm_per_cls' if smoothed else 'raw_per_cls'
    in_data  = metrics.get(f'cross_gesture_transition_in_{suffix}',  {})
    out_data = metrics.get(f'cross_gesture_transition_out_{suffix}', {})

    gesture_classes = [c for c in CLASSES if c != CLASSES[REST_IDX]]
    if not in_data and not out_data:
        print('  Skipped transition confusion plot (no data)')
        return

    n   = len(gesture_classes)
    x   = np.arange(n)
    w   = 0.35

    colors = {
        'correct':        'seagreen',
        'predicted_rest': 'steelblue',
        'wrong_gesture':  'tomato',
    }
    labels = {
        'correct':        'Correct',
        'predicted_rest': 'Predicted rest',
        'wrong_gesture':  'Wrong gesture',
    }

    fig, ax = plt.subplots(figsize=(9, 5))

    for bar_i, (phase_label, offset, data) in enumerate([
        ('tr-in',  -w / 2, in_data),
        ('tr-out',  w / 2, out_data),
    ]):
        bottoms = np.zeros(n)
        for stack_key in ('correct', 'predicted_rest', 'wrong_gesture'):
            vals = np.array([data.get(cls, {}).get(stack_key, 0)
                             for cls in gesture_classes], dtype=float)
            bars = ax.bar(x + offset, vals, w, bottom=bottoms,
                          color=colors[stack_key], alpha=0.85,
                          label=labels[stack_key] if bar_i == 0 else '_nolegend_')
            # Label non-zero segments
            for xi, (val, bot) in enumerate(zip(vals, bottoms)):
                if val > 0:
                    ax.text(xi + offset, bot + val / 2, str(int(val)),
                            ha='center', va='center', fontsize=8, color='white',
                            fontweight='bold')
            bottoms += vals

        # Phase label above bar group
        totals = np.array([data.get(cls, {}).get('total', 0)
                           for cls in gesture_classes], dtype=float)
        for xi, tot in enumerate(totals):
            ax.text(xi + offset, tot + 0.5, phase_label,
                    ha='center', va='bottom', fontsize=7.5, color='dimgray')

    ax.set_xticks(x)
    ax.set_xticklabels(gesture_classes)
    ax.set_ylabel('Prediction count')
    pred_type = 'smoothed' if smoothed else 'raw'
    ax.set_title(f'Transition phase prediction breakdown ({pred_type})\n'
                 f'Wrong gesture = cross-class confusion (excluding rest)')
    ax.legend()
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

    static_dir  = os.path.dirname(static_csv)
    amp_samples = load_amp_samples(static_dir)
    if amp_samples:
        print(f'  {len(amp_samples)} amplitude samples loaded (5 ms onset resolution)')
    else:
        print(f'  No emg_amplitude.csv found — onset will use prediction-stride fallback')

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
    lat_data  = compute_latency_and_stability(gesture_records, amp_samples)

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
        os.path.join(out_dir, 'per_class_metrics.png'),
        title_suffix='hold phase (steady state)')

    y_true_gest_all = [r['true']     for r in gesture_records]
    y_raw_gest_all  = [r['raw_pred'] for r in gesture_records]
    plot_per_class_metrics(
        y_true_gest_all, y_raw_gest_all,
        os.path.join(out_dir, 'per_class_metrics_all_phases.png'),
        title_suffix='all gesture phases (tr-in + hold + tr-out)')

    plot_transition_confusion(
        metrics,
        os.path.join(out_dir, 'transition_confusion_raw.png'),
        smoothed=False)
    plot_transition_confusion(
        metrics,
        os.path.join(out_dir, 'transition_confusion_smoothed.png'),
        smoothed=True)

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
                          os.path.join(out_dir, 'trial_confidence_example.png'),
                          amp_samples=amp_samples)

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
