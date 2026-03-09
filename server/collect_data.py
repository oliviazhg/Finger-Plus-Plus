'''
EMG Data Collection Script

Collects labelled EMG data in three phases per trial:
  - Initiation (2s): transitioning into the grip
  - Steady     (4s): holding the grip
  - Release    (2s): releasing back to rest

After every trial an automated 8s rest period is collected and saved
to the rest class (2s → rest_init, 4s → rest_steady, 2s → rest_release).
This keeps rest data balanced with other classes without manual recording.

Each phase is saved to a separate file:
  data_collection/{class}_init.npy
  data_collection/{class}_steady.npy
  data_collection/{class}_release.npy

Files store float32 samples, shape (N_trials * phase_samples, 8).
New trials are appended by load-concatenate-save.

Usage:
  python collect_data.py
'''

import threading
import queue
import time
import os
import struct
import numpy as np

from pyomyo import Myo, emg_mode

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

INIT_DURATION    = 2.0   # seconds
STEADY_DURATION  = 4.0   # seconds
RELEASE_DURATION = 2.0   # seconds
SAMPLE_RATE      = 200   # Hz (FILTERED mode)

DATA_DIR = "data_collection"

INIT_SAMPLES    = int(INIT_DURATION    * SAMPLE_RATE)  # 300
STEADY_SAMPLES  = int(STEADY_DURATION  * SAMPLE_RATE)  # 600
RELEASE_SAMPLES = int(RELEASE_DURATION * SAMPLE_RATE)  # 300

# ── Myo background thread ─────────────────────────────────────────────────────

_emg_queue  = queue.Queue()
_stop_event = threading.Event()


def _myo_worker():
    m = Myo(mode=emg_mode.FILTERED)
    m.connect()

    def on_emg(emg, moving):
        _emg_queue.put(np.abs(np.array(emg, dtype=np.float32)))

    m.add_emg_handler(on_emg)
    m.set_leds([128, 128, 0], [128, 128, 0])
    m.vibrate(1)

    while not _stop_event.is_set():
        try:
            m.run()
        except struct.error:
            pass

    m.set_leds([0, 0, 0], [0, 0, 0])
    m.disconnect()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _flush_queue():
    while not _emg_queue.empty():
        try:
            _emg_queue.get_nowait()
        except queue.Empty:
            break


def _collect(n_samples):
    '''Block until exactly n_samples have been received from the Myo.'''
    samples = []
    while len(samples) < n_samples:
        try:
            samples.append(_emg_queue.get(timeout=0.5))
        except queue.Empty:
            print("  Warning: no EMG data received — check Myo connection.")
    return np.array(samples, dtype=np.float32)


def _countdown(seconds):
    for i in range(seconds, 0, -1):
        print(f"  {i}...", end=' ', flush=True)
        time.sleep(1)
    print("GO!", flush=True)


def _fname(class_name, phase):
    return os.path.join(DATA_DIR, f"{class_name.replace(' ', '_')}_{phase}.npy")


def _save(class_name, phase, data):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = _fname(class_name, phase)
    if os.path.exists(path):
        data = np.vstack([np.load(path), data])
    np.save(path, data)
    print(f"    -> {path}  ({data.shape[0]} samples total)")


def _summary():
    print("\n── Data summary ──────────────────────────────────────")
    for cls in CLASSES:
        for phase in ("init", "steady", "release"):
            path = _fname(cls, phase)
            if os.path.exists(path):
                d = np.load(path)
                print(f"  {cls:12s} {phase:8s}  {d.shape[0]:4d} samples  shape={d.shape}")
            else:
                print(f"  {cls:12s} {phase:8s}     — no data")
    print()


# ── Trial ─────────────────────────────────────────────────────────────────────

def _run_trial(class_name):
    print(f"\n── {class_name.upper()} ──────────────────────────────────────────")
    input("  Press Enter when ready...")
    print("  Starting in:", end=' ')
    _countdown(3)

    _flush_queue()

    print(f"\n  [INITIATION]  Transition into {class_name}...  ({INIT_DURATION}s)")
    init_data = _collect(INIT_SAMPLES)

    print(f"  [STEADY]      Hold {class_name}...  ({STEADY_DURATION}s)")
    steady_data = _collect(STEADY_SAMPLES)

    print(f"  [RELEASE]     Release back to rest...  ({RELEASE_DURATION}s)")
    release_data = _collect(RELEASE_SAMPLES)

    # Per-channel mean amplitude summary
    ch = "  ch:  " + "  ".join(f"{i+1:>6}" for i in range(8))
    print(f"\n{ch}")
    for label, d in (("init   ", init_data), ("steady ", steady_data), ("release", release_data)):
        means = "  ".join(f"{v:6.1f}" for v in d.mean(axis=0))
        print(f"  {label}  {means}")

    print("\n  Saving...")
    _save(class_name, "init",    init_data)
    _save(class_name, "steady",  steady_data)
    _save(class_name, "release", release_data)

    # ── Automated rest collection ──────────────────────────────────────────
    # Collect one rest trial immediately after without user intervention.
    # Phases mirror the grip trial durations so rest stays balanced.
    total = INIT_DURATION + STEADY_DURATION + RELEASE_DURATION
    print(f"\n  [REST]  Relax for {total:.0f}s — collecting rest data automatically...")

    _flush_queue()
    rest_init    = _collect(INIT_SAMPLES)
    rest_steady  = _collect(STEADY_SAMPLES)
    rest_release = _collect(RELEASE_SAMPLES)

    ch = "  ch:  " + "  ".join(f"{i+1:>6}" for i in range(8))
    print(f"\n{ch}")
    for label, d in (("rest-i ", rest_init), ("rest-s ", rest_steady), ("rest-r ", rest_release)):
        means = "  ".join(f"{v:6.1f}" for v in d.mean(axis=0))
        print(f"  {label}  {means}")

    _save("rest", "init",    rest_init)
    _save("rest", "steady",  rest_steady)
    _save("rest", "release", rest_release)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("═" * 52)
    print("  EMG Data Collection")
    print(f"  Classes : {', '.join(CLASSES)}")
    print(f"  Phases  : init {INIT_DURATION}s  |  steady {STEADY_DURATION}s  |  release {RELEASE_DURATION}s")
    print(f"  Output  : {DATA_DIR}/")
    print("═" * 52)

    myo_thread = threading.Thread(target=_myo_worker, daemon=True)
    myo_thread.start()
    print("\nConnecting to Myo (vibration confirms)...")
    time.sleep(2)

    try:
        while True:
            print("\n── Menu ──────────────────────────────────────────────")
            for i, cls in enumerate(CLASSES, 1):
                print(f"  {i}. Record {cls}")
            print("  s. Summary")
            print("  q. Quit")

            choice = input("\n> ").strip().lower()

            if choice == 'q':
                break
            elif choice == 's':
                _summary()
            elif choice.isdigit() and 1 <= int(choice) <= len(CLASSES):
                cls = CLASSES[int(choice) - 1]
                while True:
                    _run_trial(cls)
                    again = input(f"\n  Another {cls} trial? (y/n): ").strip().lower()
                    if again != 'y':
                        break
            else:
                print("  Invalid choice.")

    except KeyboardInterrupt:
        pass
    finally:
        _stop_event.set()
        print("\nDisconnecting...")
        myo_thread.join(timeout=3)
        print("Done.")


if __name__ == '__main__':
    main()
