'''
Structured Real-time Evaluation — Random Forest

Identical protocol to evaluate_realtime_bpnn.py.
Each trial has three recorded phases:

  transition_in  (2s) — user moves from rest into the gesture
  hold           (Xs) — user holds gesture steady
  transition_out (2s) — user relaxes back to rest

Loads model.joblib from the results directory (pass --results to switch
between results_steady and results_all_phases).

Output is written to a timestamped folder under inference_eval_rf/.

Usage:
  python evaluate_realtime_rf.py
  python evaluate_realtime_rf.py --results results_steady --reps 5 --hold 8 --rest 7
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
import joblib
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

from pyomyo import Myo, emg_mode

# ── Configuration ─────────────────────────────────────────────────────────────

RESULTS_DIR    = 'results_steady'   # overridable via --results
CLASSES        = ['cylindrical', 'lateral', 'palm', 'rest']
REST_IDX       = CLASSES.index('rest')
WINDOW_SIZE    = 40
STRIDE         = 20
WAMP_THRESH    = 10.0
SMOOTH_N       = 5
CALIB_SEC      = 2
HOLD_SEC       = 5
TRANSITION_SEC = 2
REST_GAP_SEC   = 7
WARMUP_SEC     = 0.5
N_REPS         = 5

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


def record_phase(model, scale, true_idx, duration, phase):
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
                'true':          true_idx,
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


def compute_metrics(records):
    hold_recs   = _phase_records(records, 'hold')
    tr_in_recs  = _phase_records(records, 'transition_in')
    tr_out_recs = _phase_records(records, 'transition_out')
    infer_ms    = np.array([r['infer_ms'] for r in records])

    tr_in_per_class = {}
    for cls_idx, cls in enumerate(CLASSES):
        recs = [r for r in tr_in_recs if r['true'] == cls_idx]
        if recs:
            correct = sum(r['raw_pred'] == cls_idx for r in recs)
            tr_in_per_class[cls] = float(correct / len(recs))

    tr_out_acc = None
    if tr_out_recs:
        correct    = sum(r['raw_pred'] == REST_IDX for r in tr_out_recs)
        tr_out_acc = float(correct / len(tr_out_recs))

    return {
        'hold':           _acc_metrics(hold_recs),
        'transition_in':  _acc_metrics(tr_in_recs),
        'transition_out': _acc_metrics(tr_out_recs),
        'transition_in_acc_per_class': tr_in_per_class,
        'transition_out_rest_acc':     tr_out_acc,
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


def plot_transition_accuracy(metrics, path):
    tr_in  = metrics['transition_in_acc_per_class']
    tr_out = metrics['transition_out_rest_acc']

    gesture_classes = [c for c in CLASSES if c != 'rest']
    in_accs = [tr_in.get(c, 0) for c in gesture_classes]
    out_acc = tr_out if tr_out is not None else 0

    x = np.arange(len(gesture_classes))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, in_accs, 0.5, label='Transition in (→ gesture)', color='steelblue')
    ax.axhline(out_acc, color='tomato', linestyle='--',
               label=f'Transition out (→ rest): {out_acc:.2f}')
    ax.set_xticks(x)
    ax.set_xticklabels(gesture_classes)
    ax.set_ylabel('Raw accuracy')
    ax.set_ylim(0, 1.05)
    ax.set_title('Transition accuracy — into gesture and back to rest')
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  Saved {path}')


def plot_timeline(records, path):
    MARKERS = {'transition_in': '^', 'hold': 'o', 'transition_out': 'v'}
    fig, axes = plt.subplots(len(CLASSES), 1, figsize=(14, 6), sharex=False)

    for cls_idx, (ax, cls) in enumerate(zip(axes, CLASSES)):
        recs = [r for r in records if r['true'] == cls_idx]
        ax.set_ylabel(cls, rotation=0, labelpad=65, va='center', fontsize=9)
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
        w.writerow(['t', 'phase', 'true_class', 'raw_pred', 'smoothed_pred',
                    'conf_cyl', 'conf_lat', 'conf_palm', 'conf_rest', 'infer_ms'])
        for r in records:
            w.writerow([
                f'{r["t"]:.4f}',
                r['phase'],
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

    plot_confidence_histogram(records, os.path.join(out_dir, 'confidence_histogram.png'))
    plot_transition_accuracy(metrics,  os.path.join(out_dir, 'transition_accuracy.png'))
    plot_timeline(records,             os.path.join(out_dir, 'timeline.png'))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Structured real-time RF evaluation')
    parser.add_argument('--results', default=RESULTS_DIR,
                        help='Results directory (default: results_steady)')
    parser.add_argument('--reps', type=int, default=N_REPS,       help='Repetitions per class')
    parser.add_argument('--hold', type=int, default=HOLD_SEC,     help='Hold duration in seconds')
    parser.add_argument('--rest', type=int, default=REST_GAP_SEC, help='Rest gap between trials in seconds')
    args = parser.parse_args()

    trial_sec = TRANSITION_SEC + args.hold + TRANSITION_SEC
    total_sec = args.reps * len(CLASSES) * (args.rest + trial_sec)

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

    print(f'\n── Session plan ──────────────────────────────────────')
    print(f'  Classes     : {", ".join(CLASSES)}')
    print(f'  Reps        : {args.reps}  ×  {len(CLASSES)} classes')
    print(f'  Per trial   : {TRANSITION_SEC}s transition in  +  {args.hold}s hold  +  {TRANSITION_SEC}s transition out')
    print(f'  Between     : {args.rest}s rest gap')
    print(f'  Total time  : ~{total_sec}s ({total_sec // 60}m {total_sec % 60}s)')
    input('\n  Press Enter to begin...')

    all_records = []

    try:
        for rep in range(1, args.reps + 1):
            print(f'\n  ── Rep {rep} / {args.reps} ────────────────────────────────')
            for cls_idx, cls in enumerate(CLASSES):

                rest_gap(args.rest, cls)

                print(f'  TRANSITION INTO {cls.upper()}', flush=True)
                tr_in = record_phase(model, scale, cls_idx, TRANSITION_SEC, 'transition_in')
                all_records.extend(tr_in)
                _phase_summary(tr_in, cls_idx, 'transition in')

                print(f'  HOLD {cls.upper()}', flush=True)
                hold = record_phase(model, scale, cls_idx, args.hold, 'hold')
                all_records.extend(hold)
                _phase_summary(hold, cls_idx, 'hold')

                print(f'  RELEASE back to rest', flush=True)
                tr_out = record_phase(model, scale, REST_IDX, TRANSITION_SEC, 'transition_out')
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

    print(f'  {"class":<12}  {"hold recall":>11}  {"transition in":>13}')
    print('  ' + '─' * 42)
    for cls in CLASSES:
        hold_r  = hold_m.get('raw_report', {}).get(cls, {})
        tr_in_a = metrics['transition_in_acc_per_class'].get(cls, float('nan'))
        print(f'  {cls:<12}  {hold_r.get("recall", 0):>11.3f}  {tr_in_a:>13.3f}')

    tr_out = metrics.get('transition_out_rest_acc')
    if tr_out is not None:
        print(f'\n  Transition-out (→ rest) acc. : {tr_out:.3f}')

    out_dir = os.path.join('inference_eval_rf',
                           datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    print(f'\n── Saving to {out_dir}/ ──────────────────────────────')
    save_all(all_records, metrics, out_dir)
    print('\nDone.')


if __name__ == '__main__':
    main()
