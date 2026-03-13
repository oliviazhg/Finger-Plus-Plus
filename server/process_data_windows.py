'''
EMG Raw Window Processing Script (for C-LSTM)

Loads filtered EMG from data_collection/, applies rectification only
(no feature extraction), and saves sliding windows of raw signal.

This produces the input format required by the C-LSTM model:
  each sample is a (WINDOW_SIZE, N_CHANNELS) = (40, 8) raw EMG window

Contrast with process_data.py which extracts hand-crafted features,
collapsing each window to a (48,) feature vector.

Output saved to data_windowed/{class}_{phase}.npy
  shape: (N_trials * N_windows, 40, 8)  dtype: float32

Run: python process_data_windows.py
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

SAMPLE_RATE      = 200
INIT_DURATION    = 2.0
STEADY_DURATION  = 4.0
RELEASE_DURATION = 2.0

PHASE_SAMPLES = {
    "init":    int(INIT_DURATION    * SAMPLE_RATE),  # 400
    "steady":  int(STEADY_DURATION  * SAMPLE_RATE),  # 800
    "release": int(RELEASE_DURATION * SAMPLE_RATE),  # 400
}

WINDOW_SIZE = int(0.200 * SAMPLE_RATE)  # 200ms = 40 samples
STRIDE      = WINDOW_SIZE // 2          # 50% overlap = 20 samples
N_CHANNELS  = 8

DATA_DIR    = "data_collection"
WINDOWED_DIR = "data_windowed"

# ── Windowing ─────────────────────────────────────────────────────────────────

def extract_windows(trial):
    '''
    trial: (phase_samples, 8) rectified EMG for one trial
    returns: (N_windows, WINDOW_SIZE, 8)
    '''
    windows = []
    for start in range(0, len(trial) - WINDOW_SIZE + 1, STRIDE):
        windows.append(trial[start:start + WINDOW_SIZE])
    return np.array(windows, dtype=np.float32)


# ── Processing ────────────────────────────────────────────────────────────────

def _in_path(cls, phase):
    return os.path.join(DATA_DIR, f"{cls.replace(' ', '_')}_{phase}.npy")

def _out_path(cls, phase):
    return os.path.join(WINDOWED_DIR, f"{cls.replace(' ', '_')}_{phase}.npy")


def process_file(cls, phase):
    src = _in_path(cls, phase)
    if not os.path.exists(src):
        return None

    raw       = np.load(src)                      # (N_trials * phase_samples, 8)
    n_samples = PHASE_SAMPLES[phase]
    n_trials  = raw.shape[0] // n_samples

    if raw.shape[0] % n_samples != 0:
        print(f"  [warn] {cls} {phase}: {raw.shape[0]} samples not divisible "
              f"by {n_samples}, truncating")
        raw = raw[:n_trials * n_samples]

    trials = raw.reshape(n_trials, n_samples, N_CHANNELS)

    all_windows = []
    for trial in trials:
        rectified = np.abs(trial)                 # full-wave rectification only
        all_windows.append(extract_windows(rectified))

    result = np.vstack(all_windows)               # (N_trials * N_windows, 40, 8)

    dst = _out_path(cls, phase)
    np.save(dst, result)

    windows_per_trial = all_windows[0].shape[0]
    return n_trials, windows_per_trial, result.shape


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs(WINDOWED_DIR, exist_ok=True)

    print(f"Window : {WINDOW_SIZE} samples ({WINDOW_SIZE / SAMPLE_RATE * 1000:.0f}ms)")
    print(f"Stride : {STRIDE} samples ({STRIDE / SAMPLE_RATE * 1000:.0f}ms)")
    print(f"Output : {WINDOWED_DIR}/   shape per file: (N_windows, {WINDOW_SIZE}, {N_CHANNELS})\n")

    for cls in CLASSES:
        found_any = False
        for phase in ("init", "steady", "release"):
            result = process_file(cls, phase)
            if result is None:
                continue
            found_any = True
            n_trials, n_win, shape = result
            print(f"  {cls:<22}  {phase:<8}  "
                  f"{n_trials} trials × {n_win} windows  →  {shape}")
        if not found_any:
            print(f"  {cls:<22}  — no data")

    print("\nDone.")
