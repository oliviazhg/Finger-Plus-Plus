'''
Rectified EMG plot for cylindrical forward grasp trial.
Data is stored as rectified (np.abs applied during collection).
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "server" / "data_collection"
OUT_DIR  = Path(__file__).parent

SAMPLE_RATE     = 200
INIT_SAMPLES    = 400
STEADY_SAMPLES  = 800
RELEASE_SAMPLES = 400
TRIAL_IDX = 0

STAGE_COLORS = {
    "Initiation": "#2196F3",
    "Steady":     "#4CAF50",
    "Release":    "#FF5722",
}

plt.rcParams.update({
    'font.family':     'sans-serif',
    'font.sans-serif': ['Roboto', 'Arial', 'Helvetica', 'DejaVu Sans'],
    'font.weight':     'normal',
})

# Load first trial
init    = np.load(DATA_DIR / "cylindrical_forward_init.npy")
steady  = np.load(DATA_DIR / "cylindrical_forward_steady.npy")
release = np.load(DATA_DIR / "cylindrical_forward_release.npy")

trial_init    = init   [TRIAL_IDX*INIT_SAMPLES    : (TRIAL_IDX+1)*INIT_SAMPLES]
trial_steady  = steady [TRIAL_IDX*STEADY_SAMPLES  : (TRIAL_IDX+1)*STEADY_SAMPLES]
trial_release = release[TRIAL_IDX*RELEASE_SAMPLES : (TRIAL_IDX+1)*RELEASE_SAMPLES]

rectified = np.vstack([trial_init, trial_steady, trial_release])  # (1600, 8)
t = np.arange(len(rectified)) / SAMPLE_RATE

t_init_end   = INIT_SAMPLES / SAMPLE_RATE
t_steady_end = (INIT_SAMPLES + STEADY_SAMPLES) / SAMPLE_RATE

fig, axes = plt.subplots(8, 1, figsize=(12, 10), sharex=True)
fig.subplots_adjust(hspace=0.08)
fig.suptitle("Cylindrical Forward Grasp — Rectified EMG (Trial 1)",
             fontsize=24, fontweight='normal', y=0.995)

for ch in range(8):
    ax = axes[ch]
    ax.plot(t, rectified[:, ch], color='#B71C1C', linewidth=0.6, alpha=0.85)
    ax.set_ylim(0, 130)
    ax.set_yticks([0, 64, 128])
    ax.axvspan(0,            t_init_end,   alpha=0.10, color=STAGE_COLORS["Initiation"])
    ax.axvspan(t_init_end,   t_steady_end, alpha=0.10, color=STAGE_COLORS["Steady"])
    ax.axvspan(t_steady_end, t[-1],        alpha=0.10, color=STAGE_COLORS["Release"])
    ax.axvline(t_init_end,   color='gray', linestyle='--', linewidth=0.7, alpha=0.6)
    ax.axvline(t_steady_end, color='gray', linestyle='--', linewidth=0.7, alpha=0.6)
    ax.tick_params(labelsize=11)
    ax.grid(True, axis='y', alpha=0.25, linewidth=0.5)
    ax.set_ylabel(f"Ch {ch+1}", fontsize=13, rotation=0, labelpad=28)

# Stage labels on top row
for label, x0, x1 in [
    ("Initiation", 0,            t_init_end),
    ("Steady",     t_init_end,   t_steady_end),
    ("Release",    t_steady_end, t[-1]),
]:
    axes[0].text(
        (x0 + x1) / 2, 115, label,
        ha='center', va='top', fontsize=13, fontweight='normal',
        color=STAGE_COLORS[label]
    )

axes[-1].set_xlabel("Time (s)", fontsize=17)

patches = [mpatches.Patch(color=c, alpha=0.4, label=l)
           for l, c in STAGE_COLORS.items()]
fig.legend(handles=patches, loc='lower center', ncol=3, fontsize=15,
           bbox_to_anchor=(0.5, 0.0), frameon=False)

plt.tight_layout(rect=[0, 0.05, 1, 0.96])

out_path = OUT_DIR / "cylindrical_forward_rectified.png"
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved to {out_path}")
plt.show()
