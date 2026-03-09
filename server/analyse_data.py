'''
EMG Data Analysis Script

Loads steady-state data from data_collection/ and produces:
  1. Per-class mean EMG per channel (terminal table)
  2. Pairwise Euclidean distance matrix between class centroids (feature space)
  3. PCA scatter plot of all windows coloured by class

Features extracted per 0.5s window per channel (6 × 8 = 48 dimensions):
  MAV  — mean absolute value
  RMS  — root mean square
  VAR  — variance
  WL   — waveform length (cumulative signal change)
  SSC  — slope sign changes
  WAMP — Willison amplitude (spike rate above threshold)

Run: python analyse_data.py
'''

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

DATA_DIR     = "data_collection"
STEADY_SAMPLES = 800          # 4s × 200Hz
WINDOW_SIZE  = 40            # 0.2s window
STRIDE       = 20             # 50% overlap
WAMP_THRESH  = 10.0           # Willison amplitude threshold

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

# Mapping from sub-class → merged group for combined mode
CLASS_GROUPS = {
    "cylindrical forward": "cylindrical",
    "cylindrical by side": "cylindrical",
    "lateral palm up":     "lateral",
    "lateral palm down":   "lateral",
    "lateral forward":     "lateral",
    "lateral by side":     "lateral",
    "palm":                "palm",
    "rest":                "rest",
}
GROUPS = ["cylindrical", "lateral", "palm", "rest"]

# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(window):
    '''
    window: (W, 8) array of rectified EMG samples
    returns: (48,) feature vector
    '''
    diff = np.diff(window, axis=0)          # (W-1, 8)
    diff2 = np.diff(diff, axis=0)           # (W-2, 8)  slope differences

    mav  = window.mean(axis=0)
    rms  = np.sqrt((window ** 2).mean(axis=0))
    var  = window.var(axis=0)
    wl   = np.abs(diff).sum(axis=0)
    ssc  = (np.diff(np.sign(diff), axis=0) != 0).sum(axis=0).astype(np.float32)
    wamp = (np.abs(diff) > WAMP_THRESH).sum(axis=0).astype(np.float32)

    return np.concatenate([mav, rms, var, wl, ssc, wamp])


def extract_all_windows(data):
    '''
    data: (N_samples, 8) flat array for one class
    Returns: (N_windows, 48) feature matrix
    '''
    windows = []
    for start in range(0, len(data) - WINDOW_SIZE + 1, STRIDE):
        w = data[start:start + WINDOW_SIZE]
        windows.append(extract_features(w))
    return np.array(windows, dtype=np.float32)


# ── Load ──────────────────────────────────────────────────────────────────────

def _fname(cls, phase):
    return os.path.join(DATA_DIR, f"{cls.replace(' ', '_')}_{phase}.npy")


def load_features():
    feat_data, labels, available = [], [], []
    raw_data = {}
    for cls in CLASSES:
        path = _fname(cls, "steady")
        if os.path.exists(path):
            raw = np.load(path)
            feats = extract_all_windows(raw)
            feat_data.append(feats)
            labels.extend([cls] * len(feats))
            available.append(cls)
            raw_data[cls] = raw
        else:
            print(f"  [skip] {cls} — no steady data found")
    return np.vstack(feat_data), labels, available, raw_data


# ── Analysis ──────────────────────────────────────────────────────────────────

def print_channel_means(raw_data, classes):
    print("\n── Per-class mean EMG per channel (MAV) ─────────────")
    header = f"  {'class':<22}" + "".join(f"  ch{i+1:>2}" for i in range(8))
    print(header)
    print("  " + "─" * (len(header) - 2))
    for cls in classes:
        means = raw_data[cls].mean(axis=0)
        row = "  ".join(f"{v:5.1f}" for v in means)
        print(f"  {cls:<22}  {row}")
    print()


def print_distance_matrix(feat_data, labels, classes):
    centroids = np.array([
        feat_data[np.array(labels) == cls].mean(axis=0)
        for cls in classes
    ])
    dists = cdist(centroids, centroids, metric='euclidean')

    short = [c.replace("cylindrical", "cyl").replace("lateral", "lat") for c in classes]
    col_w = max(len(s) for s in short) + 2

    print("── Pairwise centroid distance (48-dim feature space) — lower = more similar ─")
    print("  " + " " * col_w + "  " + "  ".join(f"{s:>{col_w}}" for s in short))
    for i, s in enumerate(short):
        row = "  ".join(f"{dists[i, j]:>{col_w}.1f}" for j in range(len(classes)))
        print(f"  {s:>{col_w}}  {row}")
    print()

    pairs = [
        (dists[i, j], classes[i], classes[j])
        for i in range(len(classes))
        for j in range(i + 1, len(classes))
    ]
    pairs.sort()
    print("── Most similar class pairs ──────────────────────────")
    for dist, a, b in pairs[:5]:
        print(f"  {dist:7.1f}  {a}  ↔  {b}")
    print()


def plot_pca(feat_data, labels, classes):
    pca = PCA(n_components=2)
    proj = pca.fit_transform(feat_data)
    var = pca.explained_variance_ratio_

    cmap = plt.get_cmap("tab10")
    colours = {cls: cmap(i) for i, cls in enumerate(classes)}

    fig, ax = plt.subplots(figsize=(11, 8))
    for cls in classes:
        mask = np.array(labels) == cls
        ax.scatter(proj[mask, 0], proj[mask, 1],
                   c=[colours[cls]], alpha=0.25, s=10, rasterized=True)
        cx, cy = proj[mask, 0].mean(), proj[mask, 1].mean()
        ax.scatter(cx, cy, c=[colours[cls]], s=140, marker='X',
                   edgecolors='black', linewidths=0.8, zorder=5)
        ax.annotate(cls, (cx, cy), textcoords="offset points",
                    xytext=(6, 4), fontsize=8, color=colours[cls])

    legend = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=colours[cls], markersize=8, label=cls)
        for cls in classes
    ]
    ax.legend(handles=legend, fontsize=8, loc='upper right')
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% variance)")
    ax.set_title(
        "PCA of steady-state EMG — 48-dim feature space (MAV, RMS, VAR, WL, SSC, WAMP)\n"
        "X = class centroid  |  each point = 0.5s window"
    )
    plt.tight_layout()
    plt.savefig("pca_steady.png", dpi=150)
    print("PCA plot saved to pca_steady.png")
    plt.show()


# ── Group merging ─────────────────────────────────────────────────────────────

def merge_to_groups(feat_data, labels, raw_data):
    '''Remap sub-class labels → group labels and merge raw data accordingly.'''
    merged_labels = [CLASS_GROUPS[l] for l in labels]

    merged_raw = {}
    for cls, group in CLASS_GROUPS.items():
        if cls in raw_data:
            if group in merged_raw:
                merged_raw[group] = np.vstack([merged_raw[group], raw_data[cls]])
            else:
                merged_raw[group] = raw_data[cls].copy()

    available_groups = [g for g in GROUPS if g in merged_raw]
    return feat_data, merged_labels, available_groups, merged_raw


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("\n  1. Detailed — all sub-classes")
    print("  2. Combined — cylindrical / lateral / palm / rest")
    mode = input("\n> ").strip()

    print("\nLoading and extracting features from steady-state data...")
    feat_data, labels, available, raw_data = load_features()

    if mode == '2':
        feat_data, labels, available, raw_data = merge_to_groups(feat_data, labels, raw_data)
        pca_file = "pca_steady_combined.png"
        title_tag = "combined groups"
    else:
        pca_file = "pca_steady.png"
        title_tag = "sub-classes"

    n_windows = len(feat_data) // max(len(available), 1)
    print(f"  {len(available)} classes  |  ~{n_windows} windows/class  |  {feat_data.shape[1]}-dim features\n")

    print_channel_means(raw_data, available)
    print_distance_matrix(feat_data, labels, available)

    def _plot(feat_data, labels, classes):
        pca = PCA(n_components=2)
        proj = pca.fit_transform(feat_data)
        var = pca.explained_variance_ratio_
        cmap = plt.get_cmap("tab10")
        colours = {cls: cmap(i) for i, cls in enumerate(classes)}
        _, ax = plt.subplots(figsize=(11, 8))
        for cls in classes:
            mask = np.array(labels) == cls
            ax.scatter(proj[mask, 0], proj[mask, 1],
                       c=[colours[cls]], alpha=0.25, s=10, rasterized=True)
            cx, cy = proj[mask, 0].mean(), proj[mask, 1].mean()
            ax.scatter(cx, cy, c=[colours[cls]], s=140, marker='X',
                       edgecolors='black', linewidths=0.8, zorder=5)
            ax.annotate(cls, (cx, cy), textcoords="offset points",
                        xytext=(6, 4), fontsize=8, color=colours[cls])
        legend = [Line2D([0], [0], marker='o', color='w',
                         markerfacecolor=colours[cls], markersize=8, label=cls)
                  for cls in classes]
        ax.legend(handles=legend, fontsize=8, loc='upper right')
        ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% variance)")
        ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% variance)")
        ax.set_title(
            f"PCA of steady-state EMG ({title_tag}) — 48-dim feature space\n"
            "X = class centroid  |  each point = 0.2s window"
        )
        plt.tight_layout()
        plt.savefig(pca_file, dpi=150)
        print(f"PCA plot saved to {pca_file}")
        plt.show()

    _plot(feat_data, labels, available)
