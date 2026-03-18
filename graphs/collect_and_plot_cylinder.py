'''
Collect a single cylindrical grasp trial from the Myo armband and
immediately plot the filtered (bipolar) and rectified signals side by side.

Phases:
  Initiation (2s) → Steady (4s) → Release (2s)

Usage:
  python collect_and_plot_cylinder.py
'''

import threading
import queue
import time
import struct
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pyomyo import Myo, emg_mode

# ── Config ────────────────────────────────────────────────────────────────────

SAMPLE_RATE      = 200   # Hz (FILTERED mode)
INIT_DURATION    = 2.0
STEADY_DURATION  = 4.0
RELEASE_DURATION = 2.0

INIT_SAMPLES    = int(INIT_DURATION    * SAMPLE_RATE)  # 400
STEADY_SAMPLES  = int(STEADY_DURATION  * SAMPLE_RATE)  # 800
RELEASE_SAMPLES = int(RELEASE_DURATION * SAMPLE_RATE)  # 400

OUT_DIR = os.path.dirname(os.path.abspath(__file__))  # graphs/ (this file's location)
os.makedirs(OUT_DIR, exist_ok=True)

STAGE_COLORS = {
    "Initiation": "#2196F3",
    "Steady":     "#4CAF50",
    "Release":    "#FF5722",
}

# ── Myo thread ────────────────────────────────────────────────────────────────

# Each item in queue: (raw_sample, rectified_sample) both shape (8,) float32
_emg_queue  = queue.Queue()
_stop_event = threading.Event()


def _myo_worker():
    m = Myo(mode=emg_mode.FILTERED)
    m.connect()

    def on_emg(emg, moving):
        raw  = np.array(emg, dtype=np.float32)
        rect = np.abs(raw)
        _emg_queue.put((raw, rect))

    m.add_emg_handler(on_emg)
    m.set_leds([0, 128, 255], [0, 128, 255])
    m.vibrate(1)

    while not _stop_event.is_set():
        try:
            m.run()
        except struct.error:
            pass

    m.set_leds([0, 0, 0], [0, 0, 0])
    m.disconnect()


# ── Collection helpers ────────────────────────────────────────────────────────

def _flush():
    while not _emg_queue.empty():
        try:
            _emg_queue.get_nowait()
        except queue.Empty:
            break


def _collect(n_samples):
    '''Collect exactly n_samples, returns (raw, rectified) both shape (n, 8).'''
    raw_buf  = []
    rect_buf = []
    while len(raw_buf) < n_samples:
        try:
            raw, rect = _emg_queue.get(timeout=0.5)
            raw_buf.append(raw)
            rect_buf.append(rect)
        except queue.Empty:
            print("  Warning: no data received — check Myo connection.")
    return (np.array(raw_buf,  dtype=np.float32),
            np.array(rect_buf, dtype=np.float32))


def _countdown(seconds):
    for i in range(seconds, 0, -1):
        print(f"  {i}...", end=' ', flush=True)
        time.sleep(1)
    print("GO!", flush=True)


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot(raw, rect):
    t_init_end   = INIT_SAMPLES    / SAMPLE_RATE
    t_steady_end = (INIT_SAMPLES + STEADY_SAMPLES) / SAMPLE_RATE
    t = np.arange(len(raw)) / SAMPLE_RATE

    fig, axes = plt.subplots(8, 2, figsize=(18, 12), sharex=True)
    fig.suptitle("Cylindrical Grasp — Filtered (Bipolar) vs Rectified EMG",
                 fontsize=32, fontweight='bold')

    axes[0, 0].set_title("Filtered — raw bipolar (±128)", fontsize=27, pad=8)
    axes[0, 1].set_title("Rectified — absolute value (0–128)", fontsize=27, pad=8)

    for ch in range(8):
        ax_f = axes[ch, 0]
        ax_r = axes[ch, 1]

        ax_f.plot(t, raw[:, ch],  color='#1565C0', linewidth=0.6, alpha=0.85)
        ax_r.plot(t, rect[:, ch], color='#B71C1C', linewidth=0.6, alpha=0.85)

        ax_f.set_ylim(-130, 130)
        ax_f.set_yticks([-128, 0, 128])
        ax_f.axhline(0, color='black', linewidth=0.4, alpha=0.4)

        ax_r.set_ylim(0, 130)
        ax_r.set_yticks([0, 64, 128])

        for ax in [ax_f, ax_r]:
            ax.axvspan(0,           t_init_end,   alpha=0.10, color=STAGE_COLORS["Initiation"])
            ax.axvspan(t_init_end,  t_steady_end, alpha=0.10, color=STAGE_COLORS["Steady"])
            ax.axvspan(t_steady_end, t[-1],       alpha=0.10, color=STAGE_COLORS["Release"])
            ax.axvline(t_init_end,   color='gray', linestyle='--', linewidth=0.7, alpha=0.6)
            ax.axvline(t_steady_end, color='gray', linestyle='--', linewidth=0.7, alpha=0.6)
            ax.tick_params(labelsize=17)
            ax.grid(True, axis='y', alpha=0.25, linewidth=0.5)

        axes[ch, 0].set_ylabel(f"Ch {ch+1}", fontsize=20, rotation=0, labelpad=28)

    # Stage labels on top row
    for col in range(2):
        for label, x0, x1 in [
            ("Initiation", 0,           t_init_end),
            ("Steady",     t_init_end,  t_steady_end),
            ("Release",    t_steady_end, t[-1]),
        ]:
            axes[0, col].text(
                (x0 + x1) / 2, 115,
                label, ha='center', va='top',
                fontsize=20, fontweight='bold',
                color=STAGE_COLORS[label]
            )

    for col in range(2):
        axes[-1, col].set_xlabel("Time (s)", fontsize=25)

    patches = [mpatches.Patch(color=c, alpha=0.4, label=l)
               for l, c in STAGE_COLORS.items()]
    fig.legend(handles=patches, loc='lower center', ncol=3, fontsize=22,
               bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout(rect=[0, 0.08, 1, 0.97])

    out_path = os.path.join(OUT_DIR, "cylindrical_live_filtered_vs_rectified.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved → {out_path}")
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("═" * 52)
    print("  Cylindrical Grasp — Live Collection & Plot")
    print(f"  Phases: init {INIT_DURATION}s | steady {STEADY_DURATION}s | release {RELEASE_DURATION}s")
    print("═" * 52)

    myo_thread = threading.Thread(target=_myo_worker, daemon=True)
    myo_thread.start()
    print("\nConnecting to Myo (vibration confirms)...")
    time.sleep(2)

    input("\nPress Enter when ready to begin the trial...")
    print("\nStarting in:", end=' ')
    _countdown(3)

    _flush()

    print(f"\n[INITIATION]  Transition into cylindrical grasp...  ({INIT_DURATION}s)")
    raw_init,    rect_init    = _collect(INIT_SAMPLES)

    print(f"[STEADY]      Hold the grip...  ({STEADY_DURATION}s)")
    raw_steady,  rect_steady  = _collect(STEADY_SAMPLES)

    print(f"[RELEASE]     Release back to rest...  ({RELEASE_DURATION}s)")
    raw_release, rect_release = _collect(RELEASE_SAMPLES)

    print("\nDone! Disconnecting...")
    _stop_event.set()
    myo_thread.join(timeout=3)

    # Concatenate all phases
    raw  = np.vstack([raw_init,  raw_steady,  raw_release])
    rect = np.vstack([rect_init, rect_steady, rect_release])

    print(f"Collected {len(raw)} samples ({len(raw)/SAMPLE_RATE:.1f}s)")
    print("\nPlotting...")
    _plot(raw, rect)


if __name__ == '__main__':
    main()
