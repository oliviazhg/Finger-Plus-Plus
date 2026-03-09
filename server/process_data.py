'''
EMG Data Processing Script

Loads raw EMG from data_collection/, applies:
  1. Rectification (full-wave abs)
  2. Feature extraction per 200ms window, 50% overlap

Features per window per channel (6 × 8 = 48 dimensions):
  MAV  — mean absolute value
  RMS  — root mean square
  VAR  — variance
  WL   — waveform length (cumulative signal change)
  SSC  — slope sign changes
  WAMP — Willison amplitude (spike count above threshold)

Windows are extracted within each trial (no cross-trial bleed).
Output saved to data_processed/{class}_{phase}.npy, shape (N_trials * N_windows, 48).

Run: python process_data.py
'''

import os
import numpy as np

# ── Configuration ─────────────────────────────────────────────────────────────

CLASSES = [
    "cylindrical forward",
    "cylindrical by side",
    "lateral palm up",
    "lateral palm down",
    "lateral forward",
    "lateral by side",
    "palm",
    "rest",
]

SAMPLE_RATE      = 200   # Hz
INIT_DURATION    = 2.0   # seconds
STEADY_DURATION  = 4.0   # seconds
RELEASE_DURATION = 2.0   # seconds

INIT_SAMPLES    = int(INIT_DURATION    * SAMPLE_RATE)  # 400
STEADY_SAMPLES  = int(STEADY_DURATION  * SAMPLE_RATE)  # 800
RELEASE_SAMPLES = int(RELEASE_DURATION * SAMPLE_RATE)  # 400

PHASE_SAMPLES = {
    "init":    INIT_SAMPLES,
    "steady":  STEADY_SAMPLES,
    "release": RELEASE_SAMPLES,
}

WINDOW_SIZE  = int(0.200 * SAMPLE_RATE)  # 200ms = 40 samples
STRIDE       = WINDOW_SIZE // 2          # 50% overlap = 20 samples
WAMP_THRESH  = 10.0

DATA_DIR      = "data_collection"
PROCESSED_DIR = "data_processed"

# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(window):
    '''
    window: (WINDOW_SIZE, 8) rectified EMG
    returns: (48,) feature vector — 6 features × 8 channels
    '''
    diff = np.diff(window, axis=0)   # (W-1, 8)

    mav  = window.mean(axis=0)
    rms  = np.sqrt((window ** 2).mean(axis=0))
    var  = window.var(axis=0)
    wl   = np.abs(diff).sum(axis=0)
    ssc  = (np.diff(np.sign(diff), axis=0) != 0).sum(axis=0).astype(np.float32)
    wamp = (np.abs(diff) > WAMP_THRESH).sum(axis=0).astype(np.float32)

    return np.concatenate([mav, rms, var, wl, ssc, wamp])


def extract_windows(trial):
    '''
    trial: (phase_samples, 8) single-trial EMG
    returns: (N_windows, 48)
    '''
    windows = []
    for start in range(0, len(trial) - WINDOW_SIZE + 1, STRIDE):
        windows.append(extract_features(trial[start:start + WINDOW_SIZE]))
    return np.array(windows, dtype=np.float32)


# ── Processing ────────────────────────────────────────────────────────────────

def _in_path(cls, phase):
    return os.path.join(DATA_DIR, f"{cls.replace(' ', '_')}_{phase}.npy")

def _out_path(cls, phase):
    return os.path.join(PROCESSED_DIR, f"{cls.replace(' ', '_')}_{phase}.npy")


def process_file(cls, phase):
    src = _in_path(cls, phase)
    if not os.path.exists(src):
        return None

    raw = np.load(src)                          # (N_trials * phase_samples, 8)
    n_samples = PHASE_SAMPLES[phase]
    n_trials  = raw.shape[0] // n_samples

    if raw.shape[0] % n_samples != 0:
        print(f"  [warn] {cls} {phase}: {raw.shape[0]} samples not divisible by {n_samples}, truncating")
        raw = raw[:n_trials * n_samples]

    trials = raw.reshape(n_trials, n_samples, 8)

    # Rectify then extract features per trial
    all_windows = []
    for trial in trials:
        rectified = np.abs(trial)
        all_windows.append(extract_windows(rectified))

    result = np.vstack(all_windows)             # (N_trials * N_windows, 48)

    dst = _out_path(cls, phase)
    np.save(dst, result)

    windows_per_trial = all_windows[0].shape[0]
    return n_trials, windows_per_trial, result.shape


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print(f"Window: {WINDOW_SIZE} samples ({WINDOW_SIZE/SAMPLE_RATE*1000:.0f}ms)  "
          f"Stride: {STRIDE} samples ({STRIDE/SAMPLE_RATE*1000:.0f}ms overlap)")
    print(f"Output: {PROCESSED_DIR}/\n")

    for cls in CLASSES:
        found_any = False
        for phase in ("init", "steady", "release"):
            result = process_file(cls, phase)
            if result is None:
                continue
            found_any = True
            n_trials, n_win, shape = result
            print(f"  {cls:<22}  {phase:<8}  {n_trials} trials × {n_win} windows  →  {shape}")
        if not found_any:
            print(f"  {cls:<22}  — no data")

    print("\nDone.")
