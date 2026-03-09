'''
Real-time Random Forest Inference

Loads results/model.joblib and runs live grip classification using the Myo armband.
Feature extraction matches process_data.py exactly:
  - 200ms window (40 samples at 200Hz), 50% stride (20 samples)
  - Full-wave rectification + MAV, RMS, VAR, WL, SSC, WAMP × 8 channels = 48 features

Startup calibration:
  - Records 3s of relaxed signal, computes per-channel std
  - Raw EMG is divided by this scale before feature extraction
  - Makes amplitude-based features session-invariant

Display updates every 200ms. Smoothing: majority vote over last SMOOTH_N predictions.
Dwell-time filter: committed class only changes after candidate holds for DWELL_TIME seconds.

Run: python run_inference.py
'''

import threading
import queue
import time
import struct
import warnings
import numpy as np
import joblib
from collections import deque

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

from pyomyo import Myo, emg_mode

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_PATH   = 'results_all_phases/model.joblib'   # or 'results_steady/model.joblib'
CLASSES      = ['cylindrical', 'lateral', 'palm', 'rest']
WINDOW_SIZE  = 40        # 200ms at 200Hz
STRIDE       = 20        # 50% overlap → predict every 100ms
WAMP_THRESH  = 10.0
SMOOTH_N         = 5    # majority-vote over last N predictions
DWELL_TIME       = 0.3  # seconds candidate must hold before becoming committed class
CALIB_SEC        = 2    # seconds of rest for amplitude calibration
DISPLAY_INTERVAL = 0.2  # seconds between display updates

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

    # drain any stale samples
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print('Loading model...')
    model = joblib.load(MODEL_PATH)

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

    # Dwell-time state
    committed_class    = CLASSES[0]   # currently active output class
    candidate_class    = CLASSES[0]   # class being evaluated
    candidate_since    = time.monotonic()

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
            features = extract_features(np.array(buf)).reshape(1, -1)

            t0    = time.monotonic()
            pred  = int(model.predict(features)[0])
            proba = model.predict_proba(features)[0]
            infer_ms = (time.monotonic() - t0) * 1000

            recent_preds.append(pred)
            smoothed       = int(np.bincount(recent_preds, minlength=len(CLASSES)).argmax())
            smoothed_label = CLASSES[smoothed]
            last_proba     = proba

            # Dwell-time filter: only commit when candidate holds for DWELL_TIME
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
