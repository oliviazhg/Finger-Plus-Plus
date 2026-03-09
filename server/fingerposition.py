'''
Hand Position Detection using XGBoost

Classes:
  0 = rest
  1 = cylindrical
  2 = ball
  3 = lateral
  4 = flat

Modes:
  USE_IMU = True  — 18 features: 8 EMG (rectified) + 10 IMU (quat 4 + acc 3 + gyro 3)
  USE_IMU = False —  8 features: 8 EMG (rectified) only
  EMG at 200Hz (FILTERED mode). IMU at 50Hz linearly interpolated to 200Hz when used.

Training Mode:
1. Run with TRAINING_MODE = True
2. Press '0'-'4' while holding the corresponding grip
3. Exit when enough data collected (saved to data_emg/ or data_imu/ depending on USE_IMU)

Training Tips:
- Collect 250-300 samples per class (hold each pose for 5-6 seconds)
- Vary hand position slightly during collection for robustness
- Data files are not compatible between USE_IMU = True and False — erase when switching

Classification Mode:
1. Set TRAINING_MODE = False
2. Call get_hand_position() to get current hand state
'''

import pygame
from pygame.locals import *
import numpy as np
import os
import struct
import time

from pyomyo import Myo, emg_mode
from pyomyo.Classifier import Live_Classifier, MyoClassifier, EMGHandler
from xgboost import XGBClassifier

TRAINING_MODE = False
USE_IMU = False  # False = EMG only (8 features), True = EMG + IMU (18 features)

CLASSES = {
    0: "rest",
    1: "cylindrical",
    2: "ball",
    3: "lateral",
    4: "flat",
}
NUM_CLASSES = len(CLASSES)
FEATURE_DIM = 18 if USE_IMU else 8  # 8 EMG + 10 IMU, or 8 EMG only
DATA_DIR = "data_imu" if USE_IMU else "data_emg"

# IMU interpolation state — hardware delivers at 50Hz, interpolated to 200Hz
_imu_prev = np.zeros(10, dtype=np.float32)
_imu_next = np.zeros(10, dtype=np.float32)
_imu_prev_t: float = 0.0
_imu_next_t: float = 0.0


def _imu_handler(quat, acc, gyro):
    global _imu_prev, _imu_next, _imu_prev_t, _imu_next_t
    _imu_prev = _imu_next
    _imu_prev_t = _imu_next_t
    _imu_next = np.array(list(quat) + list(acc) + list(gyro), dtype=np.float32)
    _imu_next_t = time.monotonic()


def _get_imu():
    '''Linearly interpolate IMU to the current timestamp.'''
    dt = _imu_next_t - _imu_prev_t
    if dt <= 0:
        return _imu_next
    t = np.clip((time.monotonic() - _imu_prev_t) / dt, 0.0, 1.0)
    return _imu_prev + t * (_imu_next - _imu_prev)


def _build_features(emg_rect):
    '''Concatenate rectified EMG with IMU if enabled.'''
    if USE_IMU:
        return np.concatenate([emg_rect, _get_imu()])
    return np.array(emg_rect, dtype=np.float32)


class GripClassifier(Live_Classifier):
    '''XGBoost multi-class classifier using EMG (+ optional IMU) features'''

    def __init__(self):
        model = XGBClassifier(
            eval_metric='mlogloss',
            num_class=NUM_CLASSES,
            objective='multi:softmax'
        )
        super().__init__(model, name="Grip_Detector", color=(50, 150, 255))

    def store_data(self, cls, vals):
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(f'{DATA_DIR}/vals%d.dat' % cls, 'ab') as f:
            f.write(struct.pack('<%df' % FEATURE_DIM, *vals))
        self.train(np.vstack([self.X, vals]), np.hstack([self.Y, [cls]]))

    def read_data(self):
        X, Y = [], []
        for i in range(NUM_CLASSES):
            try:
                data = np.fromfile(f'{DATA_DIR}/vals%d.dat' %i, dtype=np.float32).reshape(-1, FEATURE_DIM)
            except ValueError:
                print(f"Warning: data/vals{i}.dat has incompatible format — press 'e' in training mode to erase.")
                data = np.zeros((0, FEATURE_DIM), dtype=np.float32)
            X.append(data)
            Y.append(np.full(data.shape[0], i))
        self.train(np.vstack(X), np.hstack(Y))

    def delete_data(self):
        for i in range(NUM_CLASSES):
            open(f'{DATA_DIR}/vals%d.dat' %i, 'wb').close()
        self.read_data()

    def classify(self, emg):
        if self.X.shape[0] == 0 or self.model is None:
            return 0
        features = _build_features(np.abs(emg)).reshape(1, -1)
        try:
            return int(self.model.predict(features)[0])
        except:
            return 0


class GripEMGHandler(EMGHandler):
    '''EMG handler — rectifies and optionally appends IMU before storing training data'''

    def __call__(self, emg, moving):
        emg_rect = np.abs(emg)
        if self.recording >= 0:
            self.m.cls.store_data(self.recording, _build_features(emg_rect))
        self.emg = tuple(emg_rect)  # 8-value tuple for GUI display


# Global position tracker
_current_hand_position = 0


def _pose_handler(pose):
    global _current_hand_position
    _current_hand_position = pose
    try:
        with open('hand_position.txt', 'w') as f:
            f.write(str(pose))
    except:
        pass


def get_hand_position():
    '''Returns the current grip class (0=rest, 1=cylindrical, 2=ball, 3=lateral, 4=flat)'''
    return _current_hand_position


def _setup_myo(clr):
    m = MyoClassifier(clr, mode=emg_mode.FILTERED, hist_len=6)
    if USE_IMU:
        m.add_imu_handler(_imu_handler)
    return m


def run_training_mode():
    print("=== TRAINING MODE ===")
    print(f"  IMU: {'enabled' if USE_IMU else 'disabled'} ({FEATURE_DIM} features)")
    for k, v in CLASSES.items():
        print(f"  Press '{k}' while holding: {v}")
    print()
    print("TIP: Hold each pose for 5-6 seconds to collect ~250-300 samples")
    print("Press 'q' to quit and save data | 'e' to erase existing data")

    pygame.init()
    w, h = 800, 320
    scr = pygame.display.set_mode((w, h))
    font = pygame.font.Font(None, 30)

    clr = GripClassifier()
    m = _setup_myo(clr)

    hnd = GripEMGHandler(m)
    m.add_emg_handler(hnd)
    m.connect()

    m.set_leds(m.cls.color, m.cls.color)
    pygame.display.set_caption(m.cls.name + " - TRAINING MODE")

    try:
        while True:
            try:
                m.run()
            except struct.error:
                pass
            m.run_gui(hnd, scr, font, w, h)

    except KeyboardInterrupt:
        pass
    finally:
        try:
            m.disconnect()
        except:
            pass
        print("\nTraining data saved!")
        pygame.quit()


def run_classification_mode():
    print("=== CLASSIFICATION MODE ===")
    print(f"  IMU: {'enabled' if USE_IMU else 'disabled'} ({FEATURE_DIM} features)")
    for k, v in CLASSES.items():
        print(f"  {k} = {v}")
    print("Press Ctrl+C to quit")

    clr = GripClassifier()
    m = _setup_myo(clr)
    m.add_raw_pose_handler(_pose_handler)

    hnd = GripEMGHandler(m)
    m.add_emg_handler(hnd)
    m.connect()

    m.set_leds([0, 255, 0], [0, 255, 0])
    print("\nStarting classification...")

    try:
        while True:
            try:
                m.run()
            except struct.error:
                pass
            current_pos = get_hand_position()
            print(f"Hand position: {current_pos} ({CLASSES.get(current_pos, 'UNKNOWN')})")
            time.sleep(0.01)

    except KeyboardInterrupt:
        pass
    finally:
        m.disconnect()
        print("\nDisconnected.")


def run_classification_mode_with_shared_value(shared_position):
    '''Run classification mode with multiprocessing shared value.'''
    print("=== CLASSIFICATION MODE (Background Process) ===")
    print(f"  IMU: {'enabled' if USE_IMU else 'disabled'} ({FEATURE_DIM} features)")
    for k, v in CLASSES.items():
        print(f"  {k} = {v}")

    def shared_pose_handler(pose):
        global _current_hand_position
        _current_hand_position = pose
        shared_position.value = pose

    clr = GripClassifier()
    m = _setup_myo(clr)
    m.add_raw_pose_handler(shared_pose_handler)

    hnd = GripEMGHandler(m)
    m.add_emg_handler(hnd)
    m.connect()

    m.set_leds([0, 255, 0], [0, 255, 0])
    print("\nStarting classification...")

    try:
        while True:
            try:
                m.run()
            except struct.error:
                pass
            current_pos = get_hand_position()
            print(f"[Classifier] {current_pos} ({CLASSES.get(current_pos, 'UNKNOWN')})")
            time.sleep(0.01)

    except KeyboardInterrupt:
        pass
    finally:
        m.disconnect()
        print("[Classifier] Disconnected")


if __name__ == '__main__':
    if TRAINING_MODE:
        run_training_mode()
    else:
        run_classification_mode()
