'''
Structured Real-time Evaluation — BPNN

Runs a prompted evaluation session with the Myo armband.
For each gesture class the script counts down, then records predictions
for HOLD_SEC seconds while you hold the gesture steady.
Repeats N_REPS times per class.

After all repetitions the script computes accuracy metrics and saves:
  predictions.csv              — per-prediction log with ground truth
  results.json                 — summary metrics (raw + smoothed accuracy)
  confusion_matrix_raw.png
  confusion_matrix_smoothed.png
  confidence_histogram.png
  timeline.png                 — prediction timeline coloured correct/incorrect

Output is written to a timestamped folder under inference_eval/.

Usage:
  python evaluate_realtime_bpnn.py
  python evaluate_realtime_bpnn.py --reps 5 --hold 8 --prep 3
'''

import os
import csv
import json
import time
import queue
import struct
import threading
import argparse
from collections import deque
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import joblib
import torch
import torch.nn as nn
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from pyomyo import Myo, emg_mode

# ── Configuration ─────────────────────────────────────────────────────────────

RESULTS_DIR  = 'results_bpnn'
CLASSES      = ['cylindrical', 'lateral', 'palm', 'rest']
WINDOW_SIZE  = 40
STRIDE       = 20
WAMP_THRESH  = 10.0
SMOOTH_N     = 5
CALIB_SEC    = 2
PREP_SEC     = 3     # countdown before each hold
HOLD_SEC     = 5     # seconds to record predictions per hold
WARMUP_SEC   = 0.5   # discard predictions at very start of each hold (buffer settling)
N_REPS       = 3     # repetitions per class

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Model (must match train_model_bpnn.py) ────────────────────────────────────

class BPNN(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        layers = [nn.Linear(48, 128), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(128, 4))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

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

def infer(model, scaler, features):
    x   = scaler.transform(features.reshape(1, -1))
    x_t = torch.tensor(x, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        logits = model(x_t)
        proba  = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return int(proba.argmax()), proba


# ── Session helpers ───────────────────────────────────────────────────────────

def countdown(label, seconds):
    for remaining in range(seconds, 0, -1):
        print(f'\r  {label} in {remaining}s...  ', end='', flush=True)
        time.sleep(1)
    print()


def drain_queue():
    while not _emg_queue.empty():
        try:
            _emg_queue.get_nowait()
        except queue.Empty:
            break


def record_hold(model, scaler, scale, true_idx, hold_sec):
    '''
    Record predictions for hold_sec seconds.
    Predictions within the first WARMUP_SEC are discarded (buffer settling).
    Smoothing buffer resets at the start of each hold.

    Returns a list of dicts:
      t, true, raw_pred, smoothed_pred, proba (list), infer_ms
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
        if elapsed >= hold_sec:
            break

        frac = elapsed / hold_sec
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
        raw_pred, proba = infer(model, scaler, features)
        infer_ms = (time.monotonic() - t0) * 1000

        recent_preds.append(raw_pred)
        smoothed = int(np.bincount(list(recent_preds), minlength=len(CLASSES)).argmax())

        if time.monotonic() >= t_warmup_end:
            records.append({
                't':             elapsed,
                'true':          true_idx,
                'raw_pred':      raw_pred,
                'smoothed_pred': smoothed,
                'proba':         proba.tolist(),
                'infer_ms':      infer_ms,
            })

    print()
    return records


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(records):
    y_true     = np.array([r['true']          for r in records])
    y_raw      = np.array([r['raw_pred']       for r in records])
    y_smoothed = np.array([r['smoothed_pred']  for r in records])
    infer_ms   = np.array([r['infer_ms']       for r in records])

    return {
        'raw_balanced_acc':      float(balanced_accuracy_score(y_true, y_raw)),
        'smoothed_balanced_acc': float(balanced_accuracy_score(y_true, y_smoothed)),
        'raw_report':      classification_report(y_true, y_raw,
                               target_names=CLASSES, output_dict=True),
        'smoothed_report': classification_report(y_true, y_smoothed,
                               target_names=CLASSES, output_dict=True),
        'raw_cm':      confusion_matrix(y_true, y_raw,
                           labels=range(len(CLASSES)), normalize='true').tolist(),
        'smoothed_cm': confusion_matrix(y_true, y_smoothed,
                           labels=range(len(CLASSES)), normalize='true').tolist(),
        'mean_infer_ms': float(infer_ms.mean()),
        'std_infer_ms':  float(infer_ms.std()),
        'n_predictions': len(records),
        'class_order':   CLASSES,
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
    probas   = np.array([r['proba']    for r in records])
    y_true   = np.array([r['true']     for r in records])
    y_raw    = np.array([r['raw_pred'] for r in records])
    max_conf = probas.max(axis=1)
    correct  = y_raw == y_true

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, 1, 21)
    ax.hist(max_conf[correct],  bins=bins, alpha=0.7, label='Correct',   color='steelblue')
    ax.hist(max_conf[~correct], bins=bins, alpha=0.7, label='Incorrect', color='tomato')
    ax.set_xlabel('Max class probability')
    ax.set_ylabel('Prediction count')
    ax.set_title('Prediction confidence — correct vs incorrect (live session)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  Saved {path}')


def plot_timeline(records, path):
    '''One row per class — dots coloured correct (blue) or incorrect (red).'''
    fig, axes = plt.subplots(len(CLASSES), 1, figsize=(12, 5), sharex=False)

    for cls_idx, (ax, cls) in enumerate(zip(axes, CLASSES)):
        recs = [r for r in records if r['true'] == cls_idx]
        ax.set_ylabel(cls, rotation=0, labelpad=65, va='center', fontsize=9)
        ax.set_yticks([])
        ax.set_ylim(0.5, 1.5)
        if not recs:
            continue
        ts      = np.array([r['t']        for r in recs])
        correct = np.array([r['raw_pred'] for r in recs]) == cls_idx
        if correct.any():
            ax.scatter(ts[correct],   np.ones(correct.sum()),
                       color='steelblue', s=12, label='correct')
        if (~correct).any():
            ax.scatter(ts[~correct],  np.ones((~correct).sum()),
                       color='tomato',    s=12, label='incorrect')
        ax.set_xlim(0, max(ts) + 0.2)
        ax.set_xlabel('Time within hold (s)', fontsize=8)

    axes[0].set_title('Prediction timeline per class (raw, blue=correct, red=incorrect)')
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
        w.writerow(['t', 'true_class', 'raw_pred', 'smoothed_pred',
                    'conf_cyl', 'conf_lat', 'conf_palm', 'conf_rest', 'infer_ms'])
        for r in records:
            w.writerow([
                f'{r["t"]:.4f}',
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

    plot_confusion_matrix(metrics['raw_cm'],
                          'Confusion matrix — raw predictions (live)',
                          os.path.join(out_dir, 'confusion_matrix_raw.png'))
    plot_confusion_matrix(metrics['smoothed_cm'],
                          f'Confusion matrix — smoothed predictions (n={SMOOTH_N}, live)',
                          os.path.join(out_dir, 'confusion_matrix_smoothed.png'))
    plot_confidence_histogram(records, os.path.join(out_dir, 'confidence_histogram.png'))
    plot_timeline(records, os.path.join(out_dir, 'timeline.png'))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Structured real-time BPNN evaluation')
    parser.add_argument('--reps', type=int, default=N_REPS,  help='Repetitions per class')
    parser.add_argument('--hold', type=int, default=HOLD_SEC, help='Hold duration in seconds')
    parser.add_argument('--prep', type=int, default=PREP_SEC, help='Prep countdown in seconds')
    args = parser.parse_args()

    total_holds = args.reps * len(CLASSES)
    total_sec   = total_holds * (args.prep + args.hold)

    print(f'Device : {DEVICE}')
    print('Loading model and scaler...')
    with open(os.path.join(RESULTS_DIR, 'results.json')) as f:
        meta = json.load(f)
    dropout = meta.get('best_config', {}).get('dropout', 0.0)

    model = BPNN(dropout=dropout).to(DEVICE)
    model.load_state_dict(
        torch.load(os.path.join(RESULTS_DIR, 'model.pt'), map_location=DEVICE)
    )
    model.eval()
    scaler = joblib.load(os.path.join(RESULTS_DIR, 'scaler.joblib'))
    print(f'  Architecture : {meta.get("architecture", "48→128→4")}')
    print(f'  Config       : {meta.get("best_config")}')

    myo_thread = threading.Thread(target=_myo_worker, daemon=True)
    myo_thread.start()
    print('Connecting to Myo (vibration confirms)...')
    time.sleep(1.5)

    print('\n── Calibration ───────────────────────────────────────')
    scale = calibrate()

    print(f'\n── Session plan ──────────────────────────────────────')
    print(f'  Classes    : {", ".join(CLASSES)}')
    print(f'  Reps       : {args.reps}  ×  {len(CLASSES)} classes')
    print(f'  Hold time  : {args.hold}s per gesture')
    print(f'  Total time : ~{total_sec}s ({total_sec // 60}m {total_sec % 60}s)')
    input('\n  Press Enter to begin...')

    all_records = []

    try:
        for rep in range(1, args.reps + 1):
            print(f'\n  ── Rep {rep} / {args.reps} ────────────────────────────────')
            for cls_idx, cls in enumerate(CLASSES):
                countdown(
                    f'[rep {rep}/{args.reps}] Get ready: {cls.upper()}',
                    args.prep
                )
                print(f'  HOLD {cls.upper()}', flush=True)
                records = record_hold(model, scaler, scale, cls_idx, args.hold)
                all_records.extend(records)

                if records:
                    n       = len(records)
                    correct = sum(r['raw_pred'] == cls_idx for r in records)
                    print(f'  └ {n} predictions  |  raw acc: {correct/n:.0%}  '
                          f'|  mean conf: {np.mean([max(r["proba"]) for r in records]):.2f}')

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
    print(f'  Raw balanced acc.      : {metrics["raw_balanced_acc"]:.3f}')
    print(f'  Smoothed balanced acc. : {metrics["smoothed_balanced_acc"]:.3f}')
    print(f'  Mean inference time    : {metrics["mean_infer_ms"]:.1f} ± {metrics["std_infer_ms"]:.1f} ms')
    print(f'  Total predictions      : {metrics["n_predictions"]}')
    print()
    print(f'  {"class":<12}  {"raw recall":>10}  {"smooth recall":>13}')
    print('  ' + '─' * 40)
    for cls in CLASSES:
        raw_r = metrics['raw_report'].get(cls, {})
        smo_r = metrics['smoothed_report'].get(cls, {})
        print(f'  {cls:<12}  {raw_r.get("recall", 0):>10.3f}  {smo_r.get("recall", 0):>13.3f}')

    out_dir = os.path.join('inference_eval',
                           datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    print(f'\n── Saving to {out_dir}/ ──────────────────────────────')
    save_all(all_records, metrics, out_dir)
    print('\nDone.')


if __name__ == '__main__':
    main()
