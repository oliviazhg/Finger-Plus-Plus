'''
Real-time BPNN Inference

Loads results_bpnn/model.pt and results_bpnn/scaler.joblib and runs live
grip classification using the Myo armband.

Feature extraction matches process_data.py exactly:
  - 200ms window (40 samples at 200Hz), 50% stride (20 samples)
  - Full-wave rectification + MAV, RMS, VAR, WL, SSC, WAMP × 8 channels = 48 features

Features are standardised using the scaler saved during training before
being passed to the network.

Startup calibration:
  - Records 2s of relaxed signal, computes per-channel std
  - Raw EMG is divided by this scale before feature extraction
  - Makes amplitude-based features session-invariant

Locking logic:
  - Committed class locks to a gesture after LOCK_STREAK consecutive identical
    smoothed predictions (non-rest).
  - Once locked, stays locked until any smoothed prediction of rest, which
    immediately resets committed class to rest.
  - Every committed class change is published to MQTT (MQTT_TOPIC).

Run: python run_inference_bpnn.py
'''

import threading
import queue
import time
import struct
import warnings
import numpy as np
import joblib
from collections import deque
import os
import json
import paho.mqtt.client as mqtt
from dotenv import load_dotenv

try:
    import paho.mqtt.client as mqtt
    _MQTT_AVAILABLE = True
except ImportError:
    _MQTT_AVAILABLE = False
    print('Warning: paho-mqtt not installed — MQTT publishing disabled.')

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

import torch
import torch.nn as nn

from pyomyo import Myo, emg_mode

# ── Configuration ─────────────────────────────────────────────────────────────

RESULTS_DIR      = 'results_bpnn/2026-03-18_01-00-53_GOOD'
CLASSES          = ['cylindrical', 'lateral', 'palm', 'rest']
REST_LABEL       = 'rest'
WINDOW_SIZE      = 40        # 200ms at 200Hz
STRIDE           = 20        # 50% overlap → predict every 100ms
WAMP_THRESH      = 10.0
SMOOTH_N         = 5         # majority-vote over last N predictions
LOCK_STREAK      = 10         # consecutive identical smoothed predictions to lock a gesture
CALIB_SEC        = 2         # seconds of rest for amplitude calibration
DISPLAY_INTERVAL = 0.2       # seconds between display updates

MQTT_BROKER      = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT        = 1883
MQTT_TOPIC       = 'sensor/myo/state'

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
    '''
    Collect CALIB_SEC seconds of relaxed EMG and return per-channel std.
    Used to normalise raw signal so amplitude-based features are session-invariant.
    '''
    n = int(CALIB_SEC * 200)
    print(f'  Relax your hand — calibrating for {CALIB_SEC}s...', flush=True)

    while not _emg_queue.empty():
        _emg_queue.get_nowait()

    samples = []
    while len(samples) < n:
        try:
            samples.append(np.abs(_emg_queue.get(timeout=0.5)))
        except queue.Empty:
            print('  Warning: no EMG during calibration — check connection.')

    calib = np.array(samples)          # (n, 8)
    scale = calib.std(axis=0)
    scale[scale < 1.0] = 1.0           # floor to avoid division by near-zero noise
    print(f'  Scale (per-channel std): {scale.round(1)}')
    return scale


# ── Feature extraction (must match process_data.py) ───────────────────────────

def extract_features(window):
    '''
    window: (WINDOW_SIZE, 8) normalised rectified float32
    returns: (48,) feature vector
    '''
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
    '''
    features: (48,) numpy array (raw, pre-standardisation)
    returns: (pred_index, proba array)
    '''
    x = scaler.transform(features.reshape(1, -1))
    x_t = torch.tensor(x, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        logits = model(x_t)
        proba  = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return int(proba.argmax()), proba


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import os, json

    print(f'Device : {DEVICE}')
    print('Loading model and scaler...')

    meta_path = os.path.join(RESULTS_DIR, 'results.json')
    with open(meta_path) as f:
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

    # ── MQTT ──────────────────────────────────────────────────────────────────
    mqtt_client = None
    if _MQTT_AVAILABLE:
        try:
            mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
            mqtt_client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
            mqtt_client.loop_start()
            print(f'MQTT connected → {MQTT_BROKER}:{MQTT_PORT}  topic: {MQTT_TOPIC}')
        except Exception as e:
            mqtt_client = None
            print(f'MQTT connection failed ({e}) — publishing disabled.')

    def _publish(cls_name):
        if mqtt_client is not None:
            mqtt_client.publish(MQTT_TOPIC, cls_name)

    myo_thread = threading.Thread(target=_myo_worker, daemon=True)
    myo_thread.start()
    print('Connecting to Myo (vibration confirms)...')
    time.sleep(1.5)

    print('\n── Calibration ───────────────────────────────────────')
    scale = calibrate()

    print('\nRunning — press Ctrl+C to stop.\n')
    print(f'  {"CLASS":<12}  {"CONF":>5}   {"cyl":>5} {"lat":>5} {"palm":>5} {"rest":>5}   {"infer":>7}  {"streak":>6}')
    print('  ' + '─' * 66)

    buf                = deque(maxlen=WINDOW_SIZE)
    samples_since_pred = 0
    recent_preds       = deque(maxlen=SMOOTH_N)
    last_display       = 0.0
    last_proba         = np.zeros(len(CLASSES))

    committed_class = REST_LABEL   # start unlocked
    streak_class    = REST_LABEL   # class currently being counted
    streak_count    = 0            # consecutive predictions of streak_class

    try:
        while True:
            try:
                sample = _emg_queue.get(timeout=0.5)
            except queue.Empty:
                print('\n  Warning: no EMG data — check Myo connection.')
                continue

            buf.append(np.abs(sample) / scale)   # rectify + normalise
            samples_since_pred += 1

            if len(buf) < WINDOW_SIZE or samples_since_pred < STRIDE:
                continue

            samples_since_pred = 0
            features = extract_features(np.array(buf))

            t0 = time.monotonic()
            pred, proba = infer(model, scaler, features)
            infer_ms = (time.monotonic() - t0) * 1000

            recent_preds.append(pred)
            smoothed       = int(np.bincount(recent_preds, minlength=len(CLASSES)).argmax())
            smoothed_label = CLASSES[smoothed]
            last_proba     = proba

            # ── Streak counter ─────────────────────────────────────────────
            if smoothed_label == streak_class:
                streak_count += 1
            else:
                streak_class  = smoothed_label
                streak_count  = 1

            # ── Locking logic ──────────────────────────────────────────────
            prev_committed = committed_class

            if committed_class != REST_LABEL:
                # Locked to a gesture — release immediately on any rest prediction
                if smoothed_label == REST_LABEL:
                    committed_class = REST_LABEL
            else:
                # Unlocked — lock after LOCK_STREAK consecutive non-rest predictions
                if streak_count >= LOCK_STREAK and smoothed_label != REST_LABEL:
                    committed_class = smoothed_label

            if committed_class != prev_committed:
                _publish(committed_class)
                print(f'\n  ▶ Published: {committed_class}', flush=True)

            if time.monotonic() - last_display >= DISPLAY_INTERVAL:
                last_display = time.monotonic()
                p = last_proba
                streak_disp = f'{streak_class[:3]}×{streak_count}'
                print(
                    f'\r  {committed_class:<12}  {p[smoothed]:>4.0%}'
                    f'   {p[0]:>5.2f} {p[1]:>5.2f} {p[2]:>5.2f} {p[3]:>5.2f}'
                    f'   {infer_ms:>5.1f}ms  {streak_disp:>6}',
                    end='', flush=True
                )

    except KeyboardInterrupt:
        pass
    finally:
        _stop_event.set()
        if mqtt_client is not None:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()
        print('\nDisconnecting...')
        myo_thread.join(timeout=3)
        print('Done.')


if __name__ == '__main__':
    main()
