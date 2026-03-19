import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "server" / "data_collection"
OUT_DIR = Path(__file__).parent

SAMPLE_RATE = 200  # Hz
INIT_SAMPLES = 400    # 2s
STEADY_SAMPLES = 800  # 4s
RELEASE_SAMPLES = 400 # 2s

TRIAL_IDX = 0  # Which trial to plot

# Load first trial from each phase
init    = np.load(DATA_DIR / "cylindrical_forward_init.npy")
steady  = np.load(DATA_DIR / "cylindrical_forward_steady.npy")
release = np.load(DATA_DIR / "cylindrical_forward_release.npy")

trial_init    = init   [TRIAL_IDX * INIT_SAMPLES    : (TRIAL_IDX + 1) * INIT_SAMPLES]
trial_steady  = steady [TRIAL_IDX * STEADY_SAMPLES  : (TRIAL_IDX + 1) * STEADY_SAMPLES]
trial_release = release[TRIAL_IDX * RELEASE_SAMPLES : (TRIAL_IDX + 1) * RELEASE_SAMPLES]

data = np.vstack([trial_init, trial_steady, trial_release])
time = np.arange(len(data)) / SAMPLE_RATE  # seconds

# Stage boundaries (in seconds)
t_init_end    = INIT_SAMPLES / SAMPLE_RATE
t_steady_end  = (INIT_SAMPLES + STEADY_SAMPLES) / SAMPLE_RATE

# Stage colors
STAGE_COLORS = {
    "Initiation": "#2196F3",  # blue
    "Steady":     "#4CAF50",  # green
    "Release":    "#FF5722",  # orange-red
}

fig, axes = plt.subplots(8, 1, figsize=(14, 12), sharex=True)
fig.suptitle("Cylindrical Forward Grasp — Trial 1\n8-Channel EMG (200Hz, FILTERED + Rectified)",
             fontsize=14, fontweight='bold')

for ch in range(8):
    ax = axes[ch]
    ax.plot(time, data[:, ch], color='#333333', linewidth=0.7, alpha=0.9)

    # Shade each stage
    ax.axvspan(0,           t_init_end,   alpha=0.12, color=STAGE_COLORS["Initiation"])
    ax.axvspan(t_init_end,  t_steady_end, alpha=0.12, color=STAGE_COLORS["Steady"])
    ax.axvspan(t_steady_end, time[-1],    alpha=0.12, color=STAGE_COLORS["Release"])

    # Stage boundary lines
    ax.axvline(t_init_end,   color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.axvline(t_steady_end, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

    ax.set_ylabel(f"Ch {ch+1}", fontsize=8, rotation=0, labelpad=28)
    ax.set_ylim(0, 130)
    ax.set_yticks([0, 64, 128])
    ax.tick_params(labelsize=7)
    ax.grid(True, axis='y', alpha=0.3, linewidth=0.5)

axes[-1].set_xlabel("Time (s)", fontsize=10)

# Stage labels on top plot
for label, x_start, x_end in [
    ("Initiation", 0, t_init_end),
    ("Steady",     t_init_end, t_steady_end),
    ("Release",    t_steady_end, time[-1]),
]:
    axes[0].text(
        (x_start + x_end) / 2, 118,
        label, ha='center', va='top',
        fontsize=9, fontweight='bold',
        color=STAGE_COLORS[label]
    )

# Legend
patches = [mpatches.Patch(color=c, alpha=0.4, label=l)
           for l, c in STAGE_COLORS.items()]
fig.legend(handles=patches, loc='upper right', fontsize=9,
           bbox_to_anchor=(0.99, 0.99))

plt.tight_layout(rect=[0, 0, 1, 0.96])

out_path = OUT_DIR / "cylindrical_forward_trial1.png"
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved to {out_path}")
plt.show()
