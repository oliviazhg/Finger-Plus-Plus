'''
Comparison plot: Filtered (bipolar) vs Rectified EMG
Since all stored data is already rectified (np.abs applied during collection),
the bipolar signal is simulated by multiplying the rectified envelope by a
carrier wave that mimics the frequency content of real filtered EMG (~20-200Hz).
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "server" / "data_collection"
OUT_DIR  = Path(__file__).parent

SAMPLE_RATE    = 200
INIT_SAMPLES   = 400
STEADY_SAMPLES = 800
RELEASE_SAMPLES = 400
TRIAL_IDX = 0

STAGE_COLORS = {
    "Initiation": "#2196F3",
    "Steady":     "#4CAF50",
    "Release":    "#FF5722",
}

# Load first trial
init    = np.load(DATA_DIR / "cylindrical_forward_init.npy")
steady  = np.load(DATA_DIR / "cylindrical_forward_steady.npy")
release = np.load(DATA_DIR / "cylindrical_forward_release.npy")

trial_init    = init   [TRIAL_IDX*INIT_SAMPLES    : (TRIAL_IDX+1)*INIT_SAMPLES]
trial_steady  = steady [TRIAL_IDX*STEADY_SAMPLES  : (TRIAL_IDX+1)*STEADY_SAMPLES]
trial_release = release[TRIAL_IDX*RELEASE_SAMPLES : (TRIAL_IDX+1)*RELEASE_SAMPLES]

rectified = np.vstack([trial_init, trial_steady, trial_release])  # (1600, 8)
t = np.arange(len(rectified)) / SAMPLE_RATE

# Simulate bipolar filtered signal: multiply envelope by carrier
# EMG motor unit firing frequency ~80Hz, modulated by a mix of frequencies
rng = np.random.default_rng(42)
carrier_freqs = [40, 80, 120, 160]  # Hz — representative EMG spectral content
carrier = np.zeros(len(t))
for f in carrier_freqs:
    phase = rng.uniform(0, 2*np.pi)
    carrier += np.sin(2 * np.pi * f * t + phase)
carrier /= np.abs(carrier).max()  # normalise to ±1

bipolar = rectified * carrier[:, None]  # apply per-sample carrier to all channels

# Stage boundaries
t_init_end   = INIT_SAMPLES / SAMPLE_RATE
t_steady_end = (INIT_SAMPLES + STEADY_SAMPLES) / SAMPLE_RATE

def shade_stages(axes):
    for ax in axes:
        ax.axvspan(0,           t_init_end,   alpha=0.10, color=STAGE_COLORS["Initiation"])
        ax.axvspan(t_init_end,  t_steady_end, alpha=0.10, color=STAGE_COLORS["Steady"])
        ax.axvspan(t_steady_end, t[-1],       alpha=0.10, color=STAGE_COLORS["Release"])
        ax.axvline(t_init_end,   color='gray', linestyle='--', linewidth=0.7, alpha=0.6)
        ax.axvline(t_steady_end, color='gray', linestyle='--', linewidth=0.7, alpha=0.6)

# ---------- Plot ----------
fig, axes = plt.subplots(8, 2, figsize=(18, 12), sharex=True)
fig.suptitle("Cylindrical Forward Grasp — Filtered vs Rectified EMG (Trial 1)",
             fontsize=13, fontweight='bold')

axes[0, 0].set_title("Filtered (bipolar, simulated)", fontsize=11, pad=8)
axes[0, 1].set_title("Rectified (absolute value, stored)", fontsize=11, pad=8)

for ch in range(8):
    # Bipolar (filtered)
    ax_f = axes[ch, 0]
    ax_f.plot(t, bipolar[:, ch], color='#1565C0', linewidth=0.6, alpha=0.85)
    ax_f.set_ylim(-130, 130)
    ax_f.set_yticks([-128, 0, 128])
    ax_f.axhline(0, color='black', linewidth=0.4, alpha=0.4)
    ax_f.set_ylabel(f"Ch {ch+1}", fontsize=8, rotation=0, labelpad=28)

    # Rectified
    ax_r = axes[ch, 1]
    ax_r.plot(t, rectified[:, ch], color='#B71C1C', linewidth=0.6, alpha=0.85)
    ax_r.set_ylim(0, 130)
    ax_r.set_yticks([0, 64, 128])

    for ax in [ax_f, ax_r]:
        shade_stages([ax])
        ax.tick_params(labelsize=7)
        ax.grid(True, axis='y', alpha=0.25, linewidth=0.5)

# Stage labels on row 0
for col, label_y, ylim in [(0, 110, 130), (1, 110, 130)]:
    for label, x0, x1 in [
        ("Initiation", 0, t_init_end),
        ("Steady",     t_init_end, t_steady_end),
        ("Release",    t_steady_end, t[-1]),
    ]:
        axes[0, col].text(
            (x0 + x1) / 2, label_y, label,
            ha='center', va='top', fontsize=8, fontweight='bold',
            color=STAGE_COLORS[label]
        )

for col in range(2):
    axes[-1, col].set_xlabel("Time (s)", fontsize=10)

# Legend
patches = [mpatches.Patch(color=c, alpha=0.4, label=l)
           for l, c in STAGE_COLORS.items()]
fig.legend(handles=patches, loc='lower center', ncol=3, fontsize=9,
           bbox_to_anchor=(0.5, 0.01))

plt.tight_layout(rect=[0, 0.04, 1, 0.97])

out_path = OUT_DIR / "cylindrical_forward_filtered_vs_rectified.png"
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved to {out_path}")
plt.show()
