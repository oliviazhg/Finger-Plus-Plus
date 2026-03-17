'''
Rest-class evaluation — Random Forest

Records N_TRIALS trials of rest and measures how reliably the model
predicts 'rest' when no gesture is performed.  Useful as a standalone
check of specificity / false-activation rate.

Each trial: brief gap → HOLD_SEC seconds of rest recording.
Ground truth is always 'rest'.  Pause / redo support is included.

Outputs saved to a timestamped folder under inference_eval_rf/:
  predictions.csv          — per-prediction log
  results.json             — accuracy + per-trial breakdown
  confusion_rest.png       — what the model predicts during rest
  accuracy_per_trial.png   — rest accuracy over time (drift check)

Usage:
  python evaluate_rest_rf.py
  python evaluate_rest_rf.py --results pre_demo_training_results_RF/results_rf_all_phases
  python evaluate_rest_rf.py --trials 30 --hold 5 --rest 3
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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

from pyomyo import Myo, emg_mode

# ── Configuration ─────────────────────────────────────────────────────────────

RESULTS_DIR  = 'pre_demo_training_results_RF/results_rf_all_phases'

CLASSES      = ['cylindrical', 'lateral', 'palm', 'rest']
GROUP_TO_INT = {g: i for i, g in enumerate(CLASSES)}
REST_IDX     = GROUP_TO_INT['rest']

N_TRIALS     = 30
HOLD_SEC     = 5
REST_GAP_SEC = 3
WINDOW_SIZE  = 40
STRIDE       = 20
WAMP_THRESH  = 10.0
SMOOTH_N     = 5
CALIB_SEC    = 2
WARMUP_SEC   = 0.5

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


# ── Feature extraction ────────────────────────────────────────────────────────

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
    proba = model.predict_proba(features.reshape(1, -1))[0]
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
    countdown(f'Get ready: {next_label}', min(3, seconds))


def drain_queue():
    while not _emg_queue.empty():
        try:
            _emg_queue.get_nowait()
        except queue.Empty:
            break


def _post_trial_action():
    '''Prompt after each trial. Returns "continue", "redo", or "pause".'''
    while True:
        resp = input('  → [Enter] continue  [r] redo  [p] pause : ').strip().lower()
        if resp == '':
            return 'continue'
        if resp == 'r':
            return 'redo'
        if resp == 'p':
            return 'pause'


# ── Recording ─────────────────────────────────────────────────────────────────

def record_rest(model, scale, duration, trial_num):
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
                'true':          REST_IDX,
                'raw_pred':      raw_pred,
                'smoothed_pred': smoothed,
                'proba':         proba.tolist(),
                'infer_ms':      infer_ms,
            })

    print()
    n       = len(records)
    correct = sum(r['raw_pred'] == REST_IDX for r in records)
    ok      = '✓' if n > 0 and correct / n >= 0.5 else '✗'
    conf    = np.mean([max(r['proba']) for r in records]) if records else 0.0
    print(f'  └ rest  {correct:>2}/{n}  {ok}  conf: {conf:.2f}')
    return records


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(records):
    if not records:
        return {}

    y_raw    = np.array([r['raw_pred']      for r in records])
    y_smooth = np.array([r['smoothed_pred'] for r in records])
    infer_ms = np.array([r['infer_ms']      for r in records])

    rest_acc_raw    = float((y_raw    == REST_IDX).mean())
    rest_acc_smooth = float((y_smooth == REST_IDX).mean())

    trials = sorted({r['trial'] for r in records})
    per_trial_acc = {}
    for t in trials:
        recs  = [r for r in records if r['trial'] == t]
        preds = np.array([r['raw_pred'] for r in recs])
        per_trial_acc[t] = float((preds == REST_IDX).mean())

    pred_counts = {c: int((y_raw == i).sum()) for i, c in enumerate(CLASSES)}

    # One-row confusion: rest predicted as ...
    cm_row = confusion_matrix(
        [REST_IDX] * len(records), y_raw,
        labels=range(len(CLASSES)), normalize='true'
    ).tolist()[REST_IDX]

    return {
        'rest_acc_raw':      rest_acc_raw,
        'rest_acc_smooth':   rest_acc_smooth,
        'per_trial_acc':     per_trial_acc,
        'pred_distribution': pred_counts,
        'confusion_row':     cm_row,
        'mean_infer_ms':     float(infer_ms.mean()),
        'std_infer_ms':      float(infer_ms.std()),
        'n_predictions':     len(records),
        'n_trials':          len(trials),
        'class_order':       CLASSES,
    }


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_confusion_rest(metrics, path):
    '''Bar chart — what does the model predict when the input is rest?'''
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


def plot_per_trial_accuracy(metrics, path):
    '''Line plot of rest accuracy per trial — detects drift over session.'''
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


# ── Save ──────────────────────────────────────────────────────────────────────

def _write_csv(records, out_dir):
    csv_path = os.path.join(out_dir, 'predictions.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['t', 'trial', 'true_group', 'raw_pred', 'smoothed_pred',
                    'conf_cyl', 'conf_lat', 'conf_palm', 'conf_rest', 'infer_ms'])
        for r in records:
            w.writerow([
                f'{r["t"]:.4f}',
                r['trial'],
                CLASSES[r['true']],
                CLASSES[r['raw_pred']],
                CLASSES[r['smoothed_pred']],
                *[f'{p:.4f}' for p in r['proba']],
                f'{r["infer_ms"]:.2f}',
            ])
    return csv_path


def _save_incremental(records, out_dir):
    _write_csv(records, out_dir)
    if records:
        json_path = os.path.join(out_dir, 'results.json')
        with open(json_path, 'w') as f:
            json.dump(compute_metrics(records), f, indent=2)


def save_all(records, metrics, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = _write_csv(records, out_dir)
    print(f'  Saved {csv_path}')
    json_path = os.path.join(out_dir, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'  Saved {json_path}')
    plot_confusion_rest(metrics,      os.path.join(out_dir, 'confusion_rest.png'))
    plot_per_trial_accuracy(metrics,  os.path.join(out_dir, 'accuracy_per_trial.png'))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Rest-class RF evaluation')
    parser.add_argument('--results', default=RESULTS_DIR,
                        help=f'Results directory (default: {RESULTS_DIR})')
    parser.add_argument('--trials', type=int, default=N_TRIALS,
                        help=f'Number of rest trials (default: {N_TRIALS})')
    parser.add_argument('--hold',   type=int, default=HOLD_SEC,
                        help=f'Hold duration in seconds (default: {HOLD_SEC})')
    parser.add_argument('--rest',   type=int, default=REST_GAP_SEC,
                        help=f'Rest gap between trials (default: {REST_GAP_SEC})')
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

    out_dir = os.path.join('inference_eval_rf',
                           datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_rest')
    os.makedirs(out_dir, exist_ok=True)
    print(f'  Output → {out_dir}/')

    total_sec = args.trials * (args.rest + args.hold)
    print(f'\n── Session plan ──────────────────────────────────────')
    print(f'  {args.trials} rest trials × {args.hold}s hold  ({args.rest}s gap between)')
    print(f'  Total time : ~{total_sec}s ({total_sec // 60}m {total_sec % 60}s)')
    input('\n  Press Enter to begin...')

    all_records = []
    idx         = 0

    try:
        while idx < args.trials:
            trial_num = idx + 1
            rest_gap(args.rest, f'REST trial {trial_num}/{args.trials}')
            print(f'\n  [{trial_num:>2}/{args.trials}] REST', flush=True)

            recs = record_rest(model, scale, args.hold, trial_num)
            all_records.extend(recs)
            _save_incremental(all_records, out_dir)

            action = _post_trial_action()
            if action == 'pause':
                print('\n  ── PAUSED ──────────────────────────────────────────')
                resp = input('  [Enter] resume  [r] redo this trial : ').strip().lower()
                action = 'redo' if resp == 'r' else 'continue'

            if action == 'redo':
                all_records = [r for r in all_records if r['trial'] != trial_num]
                _save_incremental(all_records, out_dir)
                print(f'  ↺ Redoing trial {trial_num}...')
            else:
                idx += 1

    except KeyboardInterrupt:
        print('\n  Interrupted — results saved to disk.')
    finally:
        _stop_event.set()
        myo_thread.join(timeout=3)

    if not all_records:
        print('No predictions recorded.')
        return

    print('\n── Results ───────────────────────────────────────────')
    metrics = compute_metrics(all_records)

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

    print(f'\n── Saving final results to {out_dir}/ ───────────────')
    save_all(all_records, metrics, out_dir)
    print('\nDone.')


if __name__ == '__main__':
    main()
