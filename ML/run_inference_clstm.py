'''
Real-time C-LSTM Inference

Loads results_clstm/model.pt and runs live grip classification using the Myo armband.

Unlike the RF and BPNN scripts, no hand-crafted feature extraction is performed.
The raw rectified EMG window (40 samples × 8 channels) is fed directly to the model.
Normalisation is handled by BatchNorm inside the network — no external scaler needed.

Startup calibration:
  - Records 2s of relaxed signal, computes per-channel std
  - Raw EMG is divided by this scale before windowing
  - Makes amplitude-based input session-invariant

Display updates every 200ms. Smoothing: majority vote over last SMOOTH_N predictions.
Dwell-time filter: committed class only changes after candidate holds for DWELL_TIME seconds.

Run: python run_inference_clstm.py
'''

import threading
import queue
import time
import struct
import warnings
import numpy as np
from collections import deque

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import joblib

from pyomyo import Myo, emg_mode

# ── Configuration ─────────────────────────────────────────────────────────────

RESULTS_DIR      = 'results_clstm'
CLASSES          = ['cylindrical', 'lateral', 'palm', 'rest']
WINDOW_SIZE      = 40        # 200ms at 200Hz
STRIDE           = 20        # 50% overlap → predict every 100ms
SMOOTH_N         = 5
DWELL_TIME       = 0.3       # seconds
CALIB_SEC        = 2
DISPLAY_INTERVAL = 0.2       # seconds

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Model (must match train_model_clstm.py) ───────────────────────────────────

class CLSTM(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(8, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.lstm1 = nn.LSTM(input_size=32, hidden_size=64, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=32, batch_first=True)
        self.drop  = nn.Dropout(dropout)
        self.fc    = nn.Linear(32, 4)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]
        x = self.drop(x)
        return self.fc(x)

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
    print(f'  Scale (per-channel std): {scale.round(1)}')
    return scale


# ── Inference ─────────────────────────────────────────────────────────────────

def infer(model, window):
    '''
    window: (WINDOW_SIZE, 8) rectified, amplitude-normalised float32
    returns: (pred_index, proba array)
    '''
    # Model expects (batch, time, channels) → add batch dim
    x_t = torch.tensor(window[np.newaxis], dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        logits = model(x_t)
        proba  = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return int(proba.argmax()), proba


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import os, json

    print(f'Device : {DEVICE}')
    print('Loading model...')

    with open(os.path.join(RESULTS_DIR, 'results.json')) as f:
        meta = json.load(f)
    dropout = meta.get('best_config', {}).get('dropout', 0.0)

    model = CLSTM(dropout=dropout).to(DEVICE)
    model.load_state_dict(
        torch.load(os.path.join(RESULTS_DIR, 'model.pt'), map_location=DEVICE)
    )
    model.eval()
    print(f'  Architecture : {meta.get("architecture")}')
    print(f'  Config       : {meta.get("best_config")}')

    myo_thread = threading.Thread(target=_myo_worker, daemon=True)
    myo_thread.start()
    print('Connecting to Myo (vibration confirms)...')
    time.sleep(1.5)

    print('\n── Calibration ───────────────────────────────────────')
    scale = calibrate()

    print('\nRunning — press Ctrl+C to stop.\n')
    print(f'  {"CLASS":<12}  {"CONF":>5}   {"cyl":>5} {"lat":>5} {"palm":>5} {"rest":>5}   {"infer":>7}')
    print('  ' + '─' * 58)

    buf                = deque(maxlen=WINDOW_SIZE)
    samples_since_pred = 0
    recent_preds       = deque(maxlen=SMOOTH_N)
    last_display       = 0.0
    last_proba         = np.zeros(len(CLASSES))

    committed_class = CLASSES[0]
    candidate_class = CLASSES[0]
    candidate_since = time.monotonic()

    try:
        while True:
            try:
                sample = _emg_queue.get(timeout=0.5)
            except queue.Empty:
                print('\n  Warning: no EMG data — check Myo connection.')
                continue

            buf.append(np.abs(sample) / scale)   # rectify + amplitude-normalise
            samples_since_pred += 1

            if len(buf) < WINDOW_SIZE or samples_since_pred < STRIDE:
                continue

            samples_since_pred = 0
            window = np.array(buf, dtype=np.float32)  # (40, 8)

            t0 = time.monotonic()
            pred, proba = infer(model, window)
            infer_ms = (time.monotonic() - t0) * 1000

            recent_preds.append(pred)
            smoothed       = int(np.bincount(list(recent_preds), minlength=len(CLASSES)).argmax())
            smoothed_label = CLASSES[smoothed]
            last_proba     = proba

            now = time.monotonic()
            if smoothed_label != candidate_class:
                candidate_class = smoothed_label
                candidate_since = now
            elif now - candidate_since >= DWELL_TIME:
                committed_class = candidate_class

            if now - last_display >= DISPLAY_INTERVAL:
                last_display = now
                p = last_proba
                pending = f'→{candidate_class}' if candidate_class != committed_class else ''
                print(
                    f'\r  {committed_class:<12}  {p[smoothed]:>4.0%}'
                    f'   {p[0]:>5.2f} {p[1]:>5.2f} {p[2]:>5.2f} {p[3]:>5.2f}'
                    f'   {infer_ms:>5.1f}ms  {pending:<16}',
                    end='', flush=True
                )

    except KeyboardInterrupt:
        pass
    finally:
        _stop_event.set()
        print('\nDisconnecting...')
        myo_thread.join(timeout=3)
        print('Done.')


if __name__ == '__main__':
    main()
