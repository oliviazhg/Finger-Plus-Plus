'''
Structured Real-time Evaluation — Random Forest

Runs a prompted evaluation session with the Myo armband.
Each trial has three recorded phases:

  transition_in  (2s) — user moves from rest into the gesture
                        ground truth = target group (cylindrical/lateral/palm)
  hold           (Xs) — user holds gesture steady
                        ground truth = target group
  transition_out (2s) — user relaxes back to rest
                        ground truth = rest

20 trials are run per gesture group, distributed evenly across specific
sub-class variants within each group. Sub-class is recorded in every
prediction so per-variant accuracy can be analysed after the session.
Rest is not a hold target — only appears as transition_out ground truth.

Between trials there is a REST_GAP_SEC rest period.

Outputs saved to a timestamped folder under inference_eval_rf/:
  predictions.csv              — per-prediction log with sub_class + phase
  results.json                 — hold + transition metrics (group + sub-class)
  confusion_matrix_raw.png     — hold phase, group-level
  confusion_matrix_smoothed.png
  confidence_histogram.png     — hold phase
  subclass_accuracy.png        — hold + transition accuracy per sub-class variant
  timeline.png                 — prediction timeline per group

Usage:
  python evaluate_realtime_rf.py
  python evaluate_realtime_rf.py --results pre_demo_training_results_RF/results_rf_all_phases --hold 8 --rest 7
'''

import os
import csv
import json
import time
import queue
import struct
import threading
import argparse
import warnings
from collections import deque
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import joblib
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

from pyomyo import Myo, emg_mode

# ── Configuration ─────────────────────────────────────────────────────────────

RESULTS_DIR    = 'pre_demo_training_results_RF/results_rf_all_phases'

# Model output classes — must match training
CLASSES      = ['cylindrical', 'lateral', 'palm', 'rest']
GROUP_TO_INT = {g: i for i, g in enumerate(CLASSES)}
REST_IDX     = GROUP_TO_INT['rest']

# Sub-class variants to test, grouped by model class
GROUP_VARIANTS = {
    'cylindrical': [
        'cylindrical forward vertical',
        'cylindrical forward horizontal',
        'cylindrical by side down',
        'cylindrical by side outstretched horizontal',
        'cylindrical by side outstretched vertical',
    ],
    'lateral': [
        'lateral palm down',
        'lateral forward',
        'lateral by side down',
        'lateral by side outstretched vertical',
        'lateral by side outstretched horizontal',
    ],
    'palm': [
        'palm forward',
        'palm by side outstretched',
    ],
}
GROUPS_ORDERED   = ['cylindrical', 'lateral', 'palm']
TRIALS_PER_GROUP = 30   # 30÷5=6 per cyl/lat variant, 30÷2=15 per palm variant
N_SAMPLE_TRIALS  = 5    # trials sampled per variant for variant-level comparison

DYNAMIC_SEC = 2.5   # seconds per position in dynamic mode

# DYNAMIC_VARIANTS matches GROUP_VARIANTS (same positions, different recording protocol)
DYNAMIC_VARIANTS = {
    'cylindrical': [
        'cylindrical forward vertical',
        'cylindrical forward horizontal',
        'cylindrical by side down',
        'cylindrical by side outstretched horizontal',
        'cylindrical by side outstretched vertical',
    ],
    'lateral': [
        'lateral palm down',
        'lateral forward',
        'lateral by side down',
        'lateral by side outstretched vertical',
        'lateral by side outstretched horizontal',
    ],
    'palm': [
        'palm forward',
        'palm by side outstretched',
    ],
}

WINDOW_SIZE    = 40
STRIDE         = 20
WAMP_THRESH    = 10.0
SMOOTH_N       = 5
CALIB_SEC      = 2
HOLD_SEC       = 5
TRANSITION_SEC = 2
REST_GAP_SEC   = 7
WARMUP_SEC     = 0.5

# ── Myo background thread ─────────────────────────────────────────────────────

_emg_queue  = queue.Queue()
_stop_event = threading.Event()


def _myo_worker():
    m = Myo(mode=emg_mode.FILTERED)
    m.connect()
    m.add_emg_handler(
        lambda emg, moving: _emg_queue.put(np.array(emg, dtype=np.float32))
    )
    m.set_leds([0, 128, 255], [0, 128, 255])
    m.vibrate(1)
    while not _stop_event.is_set():
        try:
            m.run()
        except struct.error:
            pass
    m.set_leds([0, 0, 0], [0, 0, 0])
    m.disconnect()


# ── Calibration ───────────────────────────────────────────────────────────────

def calibrate():
    n = int(CALIB_SEC * 200)
    print(f'  Relax your hand — calibrating for {CALIB_SEC}s...', flush=True)
    while not _emg_queue.empty():
        _emg_queue.get_nowait()
    samples = []
    while len(samples) < n:
        try:
            samples.append(np.abs(_emg_queue.get(timeout=0.5)))
        except queue.Empty:
            print('  Warning: no EMG — check connection.')
    scale = np.array(samples).std(axis=0)
    scale[scale < 1.0] = 1.0
    print(f'  Scale: {scale.round(1)}')
    return scale


# ── Feature extraction (must match process_data.py) ───────────────────────────

def extract_features(window):
    diff = np.diff(window, axis=0)
    mav  = window.mean(axis=0)
    rms  = np.sqrt((window ** 2).mean(axis=0))
    var  = window.var(axis=0)
    wl   = np.abs(diff).sum(axis=0)
    ssc  = (np.diff(np.sign(diff), axis=0) != 0).sum(axis=0).astype(np.float32)
    wamp = (np.abs(diff) > WAMP_THRESH).sum(axis=0).astype(np.float32)
    return np.concatenate([mav, rms, var, wl, ssc, wamp])


# ── Inference ─────────────────────────────────────────────────────────────────

def infer(model, features):
    x     = features.reshape(1, -1)
    proba = model.predict_proba(x)[0]
    return int(proba.argmax()), proba


# ── Trial schedule ────────────────────────────────────────────────────────────

def build_trial_schedule(seed=42):
    '''
    Returns a list of (sub_class, group) tuples.
    Groups are presented in order (cylindrical → lateral → palm).
    Within each group, TRIALS_PER_GROUP trials are distributed as evenly as
    possible across variants and shuffled.
    '''
    rng      = np.random.default_rng(seed)
    schedule = []
    for group in GROUPS_ORDERED:
        variants = GROUP_VARIANTS[group]
        n        = TRIALS_PER_GROUP
        base, rem = divmod(n, len(variants))
        counts   = [base + (1 if i < rem else 0) for i in range(len(variants))]
        trials   = []
        for variant, count in zip(variants, counts):
            trials.extend([(variant, group)] * count)
        rng.shuffle(trials)
        schedule.extend(trials)
    return schedule


def build_dynamic_schedule():
    '''Returns [(group, [positions...])] with position order randomised per call.'''
    rng = np.random.default_rng()
    schedule = []
    for group in GROUPS_ORDERED:
        positions = list(DYNAMIC_VARIANTS[group])
        rng.shuffle(positions)
        schedule.append((group, positions))
    return schedule


# ── Session helpers ───────────────────────────────────────────────────────────

def countdown(label, seconds):
    for remaining in range(seconds, 0, -1):
        print(f'\r  {label} in {remaining}s...  ', end='', flush=True)
        time.sleep(1)
    print()


def rest_gap(seconds, next_label):
    quiet_sec = max(0, seconds - 3)
    if quiet_sec > 0:
        print(f'  Resting...', flush=True)
        time.sleep(quiet_sec)
    countdown(f'Get ready: {next_label.upper()}', min(3, seconds))


def drain_queue():
    while not _emg_queue.empty():
        try:
            _emg_queue.get_nowait()
        except queue.Empty:
            break


def record_phase(model, scale, true_idx, sub_class, duration, phase, trial_num=0):
    '''
    Record predictions for `duration` seconds.

    true_idx  : ground truth group index (0=cyl, 1=lat, 2=palm, 3=rest)
    sub_class : specific variant being performed (stored in every record)
    phase     : 'transition_in', 'hold', or 'transition_out'
    trial_num : trial counter used for subsampled variant-level analysis
    '''
    drain_queue()

    buf                = deque(maxlen=WINDOW_SIZE)
    samples_since_pred = 0
    recent_preds       = deque(maxlen=SMOOTH_N)
    records            = []
    t_start            = time.monotonic()
    t_warmup_end       = t_start + WARMUP_SEC

    while True:
        now     = time.monotonic()
        elapsed = now - t_start
        if elapsed >= duration:
            break

        frac = elapsed / duration
        bar  = '█' * int(frac * 20) + '░' * (20 - int(frac * 20))
        print(f'\r  ▶ [{bar}] {elapsed:.1f}s  ', end='', flush=True)

        try:
            sample = _emg_queue.get(timeout=0.05)
        except queue.Empty:
            continue

        buf.append(np.abs(sample) / scale)
        samples_since_pred += 1

        if len(buf) < WINDOW_SIZE or samples_since_pred < STRIDE:
            continue

        samples_since_pred = 0
        features = extract_features(np.array(buf))

        t0 = time.monotonic()
        raw_pred, proba = infer(model, features)
        infer_ms = (time.monotonic() - t0) * 1000

        recent_preds.append(raw_pred)
        smoothed = int(np.bincount(list(recent_preds), minlength=len(CLASSES)).argmax())

        if time.monotonic() >= t_warmup_end:
            records.append({
                't':             elapsed,
                'trial':         trial_num,
                'true':          true_idx,
                'sub_class':     sub_class,
                'phase':         phase,
                'raw_pred':      raw_pred,
                'smoothed_pred': smoothed,
                'proba':         proba.tolist(),
                'infer_ms':      infer_ms,
            })

    print()
    return records


def _phase_summary(records, true_idx, label):
    if not records:
        return
    n        = len(records)
    correct  = sum(r['raw_pred'] == true_idx for r in records)
    ok       = '✓' if correct > n / 2 else '✗'
    conf     = np.mean([max(r['proba']) for r in records])
    infer_ms = np.mean([r['infer_ms'] for r in records])
    print(f'  └ {label:<16} {correct:>2}/{n}  {ok}  conf: {conf:.2f}  infer: {infer_ms:.1f}ms')


# ── Metrics ───────────────────────────────────────────────────────────────────

def _phase_records(records, phase):
    return [r for r in records if r['phase'] == phase]


def _acc_metrics(recs):
    if not recs:
        return None
    y_true     = np.array([r['true']         for r in recs])
    y_raw      = np.array([r['raw_pred']      for r in recs])
    y_smoothed = np.array([r['smoothed_pred'] for r in recs])
    infer_ms   = np.array([r['infer_ms']      for r in recs])
    return {
        'raw_balanced_acc':      float(balanced_accuracy_score(y_true, y_raw)),
        'smoothed_balanced_acc': float(balanced_accuracy_score(y_true, y_smoothed)),
        'raw_report':      classification_report(y_true, y_raw,
                               target_names=CLASSES, output_dict=True,
                               zero_division=0),
        'smoothed_report': classification_report(y_true, y_smoothed,
                               target_names=CLASSES, output_dict=True,
                               zero_division=0),
        'raw_cm':      confusion_matrix(y_true, y_raw,
                           labels=range(len(CLASSES)), normalize='true').tolist(),
        'smoothed_cm': confusion_matrix(y_true, y_smoothed,
                           labels=range(len(CLASSES)), normalize='true').tolist(),
        'n_predictions':  len(recs),
        'mean_infer_ms':  float(infer_ms.mean()),
        'std_infer_ms':   float(infer_ms.std()),
    }


def _subclass_acc(records, phase, true_idx_fn, seed=0):
    '''
    Per-sub-class accuracy using N_SAMPLE_TRIALS randomly sampled trials per variant.
    Subsampling equalises the comparison across variants regardless of trial count.
    '''
    all_variants = [v for g in GROUPS_ORDERED for v in GROUP_VARIANTS[g]]
    recs_phase   = _phase_records(records, phase)
    rng          = np.random.default_rng(seed)
    out = {}
    for variant in all_variants:
        v_recs = [r for r in recs_phase if r['sub_class'] == variant]
        if not v_recs:
            continue
        trial_nums  = list({r['trial'] for r in v_recs})
        n           = min(N_SAMPLE_TRIALS, len(trial_nums))
        sampled     = set(rng.choice(trial_nums, size=n, replace=False).tolist())
        s_recs      = [r for r in v_recs if r['trial'] in sampled]
        expected    = true_idx_fn(variant)
        correct     = sum(r['raw_pred'] == expected for r in s_recs)
        out[variant] = float(correct / len(s_recs)) if s_recs else 0.0
    return out


def compute_metrics(records):
    hold_recs   = _phase_records(records, 'hold')
    tr_in_recs  = _phase_records(records, 'transition_in')
    tr_out_recs = _phase_records(records, 'transition_out')
    infer_ms    = np.array([r['infer_ms'] for r in records])

    def group_idx(variant):
        return GROUP_TO_INT[next(g for g, vs in GROUP_VARIANTS.items() if variant in vs)]

    def _group_mean_acc(phase_recs, expected_fn):
        '''Group accuracy = mean of per-variant accuracies (all trials, equal weight per variant).'''
        out = {}
        for group in GROUPS_ORDERED:
            cls_idx      = GROUP_TO_INT[group]
            variant_accs = []
            for variant in GROUP_VARIANTS[group]:
                v_recs = [r for r in phase_recs if r['sub_class'] == variant]
                if v_recs:
                    correct = sum(r['raw_pred'] == expected_fn(cls_idx) for r in v_recs)
                    variant_accs.append(correct / len(v_recs))
            if variant_accs:
                out[group] = float(np.mean(variant_accs))
        return out

    # Overall transition-out → rest accuracy (pooled, for reference)
    tr_out_acc = None
    if tr_out_recs:
        correct    = sum(r['raw_pred'] == REST_IDX for r in tr_out_recs)
        tr_out_acc = float(correct / len(tr_out_recs))

    return {
        'hold':                            _acc_metrics(hold_recs),
        'transition_in':                   _acc_metrics(tr_in_recs),
        'transition_out':                  _acc_metrics(tr_out_recs),
        'hold_acc_per_group':              _group_mean_acc(hold_recs,   lambda i: i),
        'transition_in_acc_per_group':     _group_mean_acc(tr_in_recs,  lambda i: i),
        'transition_out_acc_per_group':    _group_mean_acc(tr_out_recs, lambda _: REST_IDX),
        'transition_out_rest_acc':         tr_out_acc,
        'hold_acc_per_subclass':           _subclass_acc(records, 'hold',
                                               lambda v: group_idx(v)),
        'transition_in_acc_per_subclass':  _subclass_acc(records, 'transition_in',
                                               lambda v: group_idx(v)),
        'transition_out_acc_per_subclass': _subclass_acc(records, 'transition_out',
                                               lambda v: REST_IDX),
        'n_sample_trials': N_SAMPLE_TRIALS,
        'mean_infer_ms':   float(infer_ms.mean()),
        'std_infer_ms':    float(infer_ms.std()),
        'n_predictions':   len(records),
        'class_order':     CLASSES,
    }


def compute_dynamic_metrics(records):
    pos_to_group  = {v: g for g, vs in DYNAMIC_VARIANTS.items() for v in vs}
    all_positions = [v for g in GROUPS_ORDERED for v in DYNAMIC_VARIANTS[g]]

    acc_per_pos = {}
    for pos in all_positions:
        recs = [r for r in records if r['sub_class'] == pos]
        if recs:
            expected = GROUP_TO_INT[pos_to_group[pos]]
            correct  = sum(r['raw_pred'] == expected for r in recs)
            acc_per_pos[pos] = float(correct / len(recs))

    acc_per_group = {}
    for group in GROUPS_ORDERED:
        cls_idx = GROUP_TO_INT[group]
        recs = [r for r in records if r['true'] == cls_idx]
        if recs:
            correct = sum(r['raw_pred'] == cls_idx for r in recs)
            acc_per_group[group] = float(correct / len(recs))

    y_true   = np.array([r['true']     for r in records])
    y_raw    = np.array([r['raw_pred'] for r in records])
    infer_ms = np.array([r['infer_ms'] for r in records])
    return {
        'dynamic_acc_per_position': acc_per_pos,
        'dynamic_acc_per_group':    acc_per_group,
        'raw_balanced_acc':  float(balanced_accuracy_score(y_true, y_raw)),
        'raw_cm':  confusion_matrix(y_true, y_raw,
                       labels=range(len(CLASSES)), normalize='true').tolist(),
        'mean_infer_ms':  float(infer_ms.mean()),
        'std_infer_ms':   float(infer_ms.std()),
        'n_predictions':  len(records),
        'class_order':    CLASSES,
    }


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm_data, title, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(np.array(cm_data), display_labels=CLASSES).plot(
        ax=ax, colorbar=True, cmap='Blues', values_format='.2f')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  Saved {path}')


def plot_confidence_histogram(records, path):
    recs = _phase_records(records, 'hold')
    if not recs:
        return
    probas   = np.array([r['proba']    for r in recs])
    y_true   = np.array([r['true']     for r in recs])
    y_raw    = np.array([r['raw_pred'] for r in recs])
    max_conf = probas.max(axis=1)
    correct  = y_raw == y_true

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, 1, 21)
    ax.hist(max_conf[correct],  bins=bins, alpha=0.7, label='Correct',   color='steelblue')
    ax.hist(max_conf[~correct], bins=bins, alpha=0.7, label='Incorrect', color='tomato')
    ax.set_xlabel('Max class probability')
    ax.set_ylabel('Prediction count')
    ax.set_title('Prediction confidence — correct vs incorrect (hold phase)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  Saved {path}')


def plot_subclass_accuracy(metrics, path):
    '''Three subplots: hold, transition-in, transition-out accuracy per sub-class.'''
    GROUP_COLORS = {'cylindrical': 'steelblue', 'lateral': 'darkorange', 'palm': 'seagreen'}
    all_variants     = [v for g in GROUPS_ORDERED for v in GROUP_VARIANTS[g]]
    variant_to_group = {v: g for g, vs in GROUP_VARIANTS.items() for v in vs}
    bar_colors       = [GROUP_COLORS[variant_to_group[v]] for v in all_variants]
    x                = np.arange(len(all_variants))
    short_labels     = [v.replace('cylindrical ', 'cyl\n')
                         .replace('lateral ', 'lat\n')
                         .replace('palm ', 'palm\n')
                        for v in all_variants]

    datasets = [
        (metrics['hold_acc_per_subclass'],
         f'Hold accuracy per sub-class  (n={N_SAMPLE_TRIALS} trials/variant, subsampled)'),
        (metrics['transition_in_acc_per_subclass'],
         f'Transition-in accuracy per sub-class  (n={N_SAMPLE_TRIALS} trials/variant, subsampled)'),
        (metrics['transition_out_acc_per_subclass'],
         f'Transition-out → rest accuracy per sub-class  (n={N_SAMPLE_TRIALS} trials/variant, subsampled)'),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    for ax, (acc_dict, title) in zip(axes, datasets):
        accs = [acc_dict.get(v, 0) for v in all_variants]
        ax.bar(x, accs, color=bar_colors, alpha=0.85, edgecolor='white')
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, fontsize=8)
        ax.set_ylabel('Raw accuracy')
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5, label='chance')
        for xi, acc in enumerate(accs):
            ax.text(xi, acc + 0.02, f'{acc:.2f}', ha='center', va='bottom', fontsize=7)

    legend_patches = [Patch(color=c, label=g) for g, c in GROUP_COLORS.items()]
    axes[0].legend(handles=legend_patches, loc='lower right', fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  Saved {path}')


def plot_group_accuracy(metrics, path):
    '''Grouped bar chart: hold / transition-in / transition-out accuracy per gesture group.'''
    phases = [
        ('hold_acc_per_group',           'Hold'),
        ('transition_in_acc_per_group',  'Transition-in'),
        ('transition_out_acc_per_group', 'Transition-out → rest'),
    ]
    x     = np.arange(len(GROUPS_ORDERED))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (key, label) in enumerate(phases):
        acc_dict = metrics.get(key, {})
        accs     = [acc_dict.get(g, 0) for g in GROUPS_ORDERED]
        ax.bar(x + (i - 1) * width, accs, width, label=label, alpha=0.85)
        for xi, acc in enumerate(accs):
            ax.text(xi + (i - 1) * width, acc + 0.02, f'{acc:.2f}',
                    ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(GROUPS_ORDERED)
    ax.set_ylabel('Accuracy (mean of variant accuracies)')
    ax.set_ylim(0, 1.15)
    ax.set_title('Group accuracy by phase (mean of per-variant accuracies, all trials)')
    ax.axhline(0.25, color='gray', linestyle=':', alpha=0.5, label='chance (1/4)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  Saved {path}')


def plot_dynamic_accuracy(metrics, path):
    '''Bar chart of per-position accuracy in dynamic mode, coloured by group.'''
    GROUP_COLORS  = {'cylindrical': 'steelblue', 'lateral': 'darkorange', 'palm': 'seagreen'}
    all_positions = [v for g in GROUPS_ORDERED for v in DYNAMIC_VARIANTS[g]]
    pos_to_group  = {v: g for g, vs in DYNAMIC_VARIANTS.items() for v in vs}
    acc_dict      = metrics.get('dynamic_acc_per_position', {})
    accs          = [acc_dict.get(p, 0) for p in all_positions]
    colors        = [GROUP_COLORS[pos_to_group[p]] for p in all_positions]
    x             = np.arange(len(all_positions))
    short_labels  = [p.replace('cylindrical ', 'cyl\n')
                      .replace('lateral ', 'lat\n')
                      .replace('palm ', 'palm\n')
                     for p in all_positions]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x, accs, color=colors, alpha=0.85, edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=8)
    ax.set_ylabel('Raw accuracy')
    ax.set_ylim(0, 1.05)
    ax.set_title('Dynamic mode — accuracy per position (raw)')
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    for xi, acc in enumerate(accs):
        ax.text(xi, acc + 0.02, f'{acc:.2f}', ha='center', va='bottom', fontsize=7)
    legend_patches = [Patch(color=c, label=g) for g, c in GROUP_COLORS.items()]
    ax.legend(handles=legend_patches, loc='lower right', fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  Saved {path}')


def plot_timeline(records, path):
    '''One row per gesture group — markers differ by phase, colour by correct/incorrect.'''
    MARKERS  = {'transition_in': '^', 'hold': 'o', 'transition_out': 'v'}
    row_defs = [(g, GROUP_TO_INT[g]) for g in GROUPS_ORDERED] + [('rest (tr-out)', REST_IDX)]

    fig, axes = plt.subplots(len(row_defs), 1, figsize=(14, 6), sharex=False)
    for ax, (label, cls_idx) in zip(axes, row_defs):
        recs = [r for r in records if r['true'] == cls_idx]
        ax.set_ylabel(label, rotation=0, labelpad=70, va='center', fontsize=9)
        ax.set_yticks([])
        ax.set_ylim(0.5, 1.5)
        if not recs:
            continue
        for phase, marker in MARKERS.items():
            ph_recs = [r for r in recs if r['phase'] == phase]
            if not ph_recs:
                continue
            ts      = np.array([r['t']        for r in ph_recs])
            correct = np.array([r['raw_pred'] for r in ph_recs]) == cls_idx
            if correct.any():
                ax.scatter(ts[correct],  np.ones(correct.sum()),
                           color='steelblue', s=12, marker=marker)
            if (~correct).any():
                ax.scatter(ts[~correct], np.ones((~correct).sum()),
                           color='tomato',    s=12, marker=marker)
        ax.set_xlim(0, max(r['t'] for r in recs) + 0.2)
        ax.set_xlabel('Time within phase (s)', fontsize=8)

    axes[0].set_title(
        'Prediction timeline  ▲=transition_in  ●=hold  ▼=transition_out  '
        'blue=correct  red=incorrect'
    )
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  Saved {path}')


# ── Save ──────────────────────────────────────────────────────────────────────

def save_all(records, metrics, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, 'predictions.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['t', 'phase', 'sub_class', 'true_group', 'raw_pred', 'smoothed_pred',
                    'conf_cyl', 'conf_lat', 'conf_palm', 'conf_rest', 'infer_ms'])
        for r in records:
            w.writerow([
                f'{r["t"]:.4f}',
                r['phase'],
                r['sub_class'],
                CLASSES[r['true']],
                CLASSES[r['raw_pred']],
                CLASSES[r['smoothed_pred']],
                *[f'{p:.4f}' for p in r['proba']],
                f'{r["infer_ms"]:.2f}',
            ])
    print(f'  Saved {csv_path}')

    json_path = os.path.join(out_dir, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'  Saved {json_path}')

    hold_m = metrics.get('hold') or {}
    if hold_m.get('raw_cm'):
        plot_confusion_matrix(hold_m['raw_cm'],
                              'Confusion matrix — raw predictions (hold phase)',
                              os.path.join(out_dir, 'confusion_matrix_raw.png'))
    if hold_m.get('smoothed_cm'):
        plot_confusion_matrix(hold_m['smoothed_cm'],
                              f'Confusion matrix — smoothed (n={SMOOTH_N}, hold phase)',
                              os.path.join(out_dir, 'confusion_matrix_smoothed.png'))

    plot_confidence_histogram(records,  os.path.join(out_dir, 'confidence_histogram.png'))
    plot_group_accuracy(metrics,        os.path.join(out_dir, 'group_accuracy.png'))
    plot_subclass_accuracy(metrics,     os.path.join(out_dir, 'subclass_accuracy.png'))
    plot_timeline(records,              os.path.join(out_dir, 'timeline.png'))


# ── Save (dynamic) ────────────────────────────────────────────────────────────

def save_dynamic(records, metrics, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, 'predictions.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['t', 'phase', 'sub_class', 'true_group', 'raw_pred', 'smoothed_pred',
                    'conf_cyl', 'conf_lat', 'conf_palm', 'conf_rest', 'infer_ms'])
        for r in records:
            w.writerow([
                f'{r["t"]:.4f}',
                r['phase'],
                r['sub_class'],
                CLASSES[r['true']],
                CLASSES[r['raw_pred']],
                CLASSES[r['smoothed_pred']],
                *[f'{p:.4f}' for p in r['proba']],
                f'{r["infer_ms"]:.2f}',
            ])
    print(f'  Saved {csv_path}')

    json_path = os.path.join(out_dir, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'  Saved {json_path}')

    if metrics.get('raw_cm'):
        plot_confusion_matrix(metrics['raw_cm'],
                              'Confusion matrix — dynamic mode (raw)',
                              os.path.join(out_dir, 'confusion_matrix_dynamic.png'))
    plot_dynamic_accuracy(metrics, os.path.join(out_dir, 'dynamic_accuracy.png'))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Real-time RF evaluation')
    parser.add_argument('--mode', choices=['static', 'dynamic'], default='static',
                        help='static: trial schedule with hold phases; '
                             'dynamic: continuous position sweep')
    parser.add_argument('--results', default=RESULTS_DIR,
                        help=f'Results directory (default: {RESULTS_DIR})')
    parser.add_argument('--hold', type=int, default=HOLD_SEC,
                        help='Hold duration in seconds (static mode)')
    parser.add_argument('--rest', type=int, default=REST_GAP_SEC,
                        help='Rest gap between trials/sweeps in seconds')
    args = parser.parse_args()

    print('Loading model...')
    model = joblib.load(os.path.join(args.results, 'model.joblib'))
    with open(os.path.join(args.results, 'results.json')) as f:
        meta = json.load(f)
    print(f'  Results dir  : {args.results}')
    print(f'  Best params  : {meta.get("best_params")}')

    myo_thread = threading.Thread(target=_myo_worker, daemon=True)
    myo_thread.start()
    print('Connecting to Myo (vibration confirms)...')
    time.sleep(1.5)

    print('\n── Calibration ───────────────────────────────────────')
    scale = calibrate()

    # ── Dynamic mode ──────────────────────────────────────────────────────────
    if args.mode == 'dynamic':
        dyn_schedule = build_dynamic_schedule()
        total_pos    = sum(len(ps) for _, ps in dyn_schedule)
        total_sec    = sum(args.rest + len(ps) * DYNAMIC_SEC for _, ps in dyn_schedule)

        print(f'\n── Dynamic session plan ──────────────────────────────')
        for group, positions in dyn_schedule:
            print(f'  {group}: {len(positions)} positions × {DYNAMIC_SEC}s each')
            for p in positions:
                print(f'    {p}')
        print(f'  Total positions : {total_pos}')
        print(f'  Total time      : ~{total_sec:.0f}s ({int(total_sec)//60}m {int(total_sec)%60}s)')
        input('\n  Press Enter to begin...')

        all_records = []

        try:
            for group, positions in dyn_schedule:
                cls_idx = GROUP_TO_INT[group]
                rest_gap(args.rest, f'{group} dynamic sweep')
                print(f'\n  ── {group.upper()} DYNAMIC SWEEP ──────────────────────────')
                short_seq = ' → '.join(p.split(group + ' ', 1)[-1] for p in positions)
                print(f'  Sequence: {short_seq}')
                for i, pos in enumerate(positions):
                    if i > 0:
                        countdown(f'→ {pos.upper()}', 2)
                    else:
                        print(f'  START: {pos.upper()}', flush=True)
                    recs = record_phase(model, scale, cls_idx, pos, DYNAMIC_SEC, 'dynamic')
                    all_records.extend(recs)
                    _phase_summary(recs, cls_idx, pos)

        except KeyboardInterrupt:
            print('\n  Interrupted — saving partial results...')
        finally:
            _stop_event.set()
            myo_thread.join(timeout=3)

        if not all_records:
            print('No predictions recorded.')
            return

        print('\n── Dynamic results ───────────────────────────────────')
        metrics = compute_dynamic_metrics(all_records)
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

        out_dir = os.path.join('inference_eval_rf',
                               datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_dynamic')
        print(f'\n── Saving to {out_dir}/ ──────────────────────────────')
        save_dynamic(all_records, metrics, out_dir)
        print('\nDone.')
        return

    # ── Static mode ───────────────────────────────────────────────────────────
    schedule  = build_trial_schedule()
    trial_sec = TRANSITION_SEC + args.hold + TRANSITION_SEC
    total_sec = len(schedule) * (args.rest + trial_sec)

    print(f'\n── Session plan ──────────────────────────────────────')
    for group in GROUPS_ORDERED:
        variants = GROUP_VARIANTS[group]
        base, rem = divmod(TRIALS_PER_GROUP, len(variants))
        counts = [base + (1 if i < rem else 0) for i in range(len(variants))]
        print(f'  {group} ({TRIALS_PER_GROUP} trials):')
        for v, c in zip(variants, counts):
            print(f'    {v:<36} {c} trials')
    print(f'  Total trials : {len(schedule)}')
    print(f'  Per trial    : {TRANSITION_SEC}s in + {args.hold}s hold + {TRANSITION_SEC}s out')
    print(f'  Between      : {args.rest}s rest gap')
    print(f'  Total time   : ~{total_sec}s ({total_sec // 60}m {total_sec % 60}s)')
    input('\n  Press Enter to begin...')

    all_records = []

    try:
        for trial_num, (sub_cls, group) in enumerate(schedule, 1):
            cls_idx = GROUP_TO_INT[group]

            rest_gap(args.rest, sub_cls)
            print(f'\n  [{trial_num:>2}/{len(schedule)}] {sub_cls.upper()}  ({group})')

            print(f'  TRANSITION IN', flush=True)
            tr_in = record_phase(model, scale,
                                 cls_idx, sub_cls, TRANSITION_SEC, 'transition_in',
                                 trial_num=trial_num)
            all_records.extend(tr_in)
            _phase_summary(tr_in, cls_idx, 'transition in')

            print(f'  HOLD', flush=True)
            hold = record_phase(model, scale,
                                cls_idx, sub_cls, args.hold, 'hold',
                                trial_num=trial_num)
            all_records.extend(hold)
            _phase_summary(hold, cls_idx, 'hold')

            print(f'  RELEASE back to rest', flush=True)
            tr_out = record_phase(model, scale,
                                  REST_IDX, sub_cls, TRANSITION_SEC, 'transition_out',
                                  trial_num=trial_num)
            all_records.extend(tr_out)
            _phase_summary(tr_out, REST_IDX, 'transition out')

    except KeyboardInterrupt:
        print('\n  Interrupted — saving partial results...')
    finally:
        _stop_event.set()
        myo_thread.join(timeout=3)

    if not all_records:
        print('No predictions recorded.')
        return

    print('\n── Results ───────────────────────────────────────────')
    metrics = compute_metrics(all_records)

    hold_m = metrics.get('hold') or {}
    print(f'  Hold raw balanced acc.      : {hold_m.get("raw_balanced_acc", 0):.3f}')
    print(f'  Hold smoothed balanced acc. : {hold_m.get("smoothed_balanced_acc", 0):.3f}')
    print(f'  Mean inference time         : {metrics["mean_infer_ms"]:.1f} ± {metrics["std_infer_ms"]:.1f} ms')
    print(f'  Total predictions           : {metrics["n_predictions"]}')
    print()
    print(f'  {"sub-class":<36}  {"hold":>6}  {"tr-in":>6}  {"tr-out":>7}')
    print('  ' + '─' * 62)
    for group in GROUPS_ORDERED:
        for variant in GROUP_VARIANTS[group]:
            h  = metrics['hold_acc_per_subclass'].get(variant, float('nan'))
            ti = metrics['transition_in_acc_per_subclass'].get(variant, float('nan'))
            to = metrics['transition_out_acc_per_subclass'].get(variant, float('nan'))
            print(f'  {variant:<36}  {h:>6.3f}  {ti:>6.3f}  {to:>7.3f}')
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
    print()

    out_dir = os.path.join('inference_eval_rf',
                           datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    print(f'\n── Saving to {out_dir}/ ──────────────────────────────')
    save_all(all_records, metrics, out_dir)
    print('\nDone.')


if __name__ == '__main__':
    main()
