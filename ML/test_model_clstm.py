'''
C-LSTM Test Script

Evaluates the trained C-LSTM on its held-out test set (15% of trials
set aside during training, never seen by the model).

Loads model.pt from the results directory and runs inference directly
on the raw windowed EMG (shape: N × 40 × 8). No external scaler needed —
normalisation is handled by BatchNorm inside the model.

Usage:
  python test_model_clstm.py              # defaults to results_clstm/
  python test_model_clstm.py results_clstm
'''

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

GROUPS = ['cylindrical', 'lateral', 'palm', 'rest']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Model (must match train_model_clstm.py) ───────────────────────────────────

class CLSTM(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(8, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.lstm1 = nn.LSTM(input_size=32, hidden_size=64, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=32, batch_first=True)
        self.drop  = nn.Dropout(dropout)
        self.fc    = nn.Linear(32, 4)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]
        x = self.drop(x)
        return self.fc(x)

# ── Load ──────────────────────────────────────────────────────────────────────

def load(results_dir):
    meta_path = os.path.join(results_dir, 'results.json')
    with open(meta_path) as f:
        meta = json.load(f)

    dropout = meta.get('best_config', {}).get('dropout', 0.0)
    model   = CLSTM(dropout=dropout).to(DEVICE)
    model.load_state_dict(
        torch.load(os.path.join(results_dir, 'model.pt'), map_location=DEVICE)
    )
    model.eval()

    X_test = np.load(os.path.join(results_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(results_dir, 'y_test.npy'))

    return model, X_test, y_test, meta

# ── Evaluate ──────────────────────────────────────────────────────────────────

def evaluate(model, X_test, y_test):
    X_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        logits = model(X_t)
        proba  = torch.softmax(logits, dim=1).cpu().numpy()
        y_pred = logits.argmax(dim=1).cpu().numpy()

    bal_acc = balanced_accuracy_score(y_test, y_pred)
    report  = classification_report(y_test, y_pred,
                                    target_names=GROUPS, output_dict=True)
    cm      = confusion_matrix(y_test, y_pred, normalize='true')
    return y_pred, proba, bal_acc, report, cm

# ── Display ───────────────────────────────────────────────────────────────────

def print_report(bal_acc, report, meta):
    cv_acc = meta.get('oof_balanced_acc', meta.get('best_gs_cv_score'))
    print(f'  CV balanced acc.   : {cv_acc:.3f}  (train set, out-of-fold)')
    print(f'  Test balanced acc. : {bal_acc:.3f}  (held-out 15%)')

    gap = cv_acc - bal_acc
    if gap > 0.05:
        print(f'  [!] Gap of {gap:.3f} suggests some overfitting to the training distribution')
    elif gap < -0.02:
        print(f'  [?] Test > CV — may be chance variation with a small test set')
    else:
        print(f'  [ok] CV and test scores are consistent')

    print()
    print(f'  {"class":<12}  {"precision":>9}  {"recall":>9}  {"f1":>9}  {"support":>9}')
    print('  ' + '─' * 56)
    for cls in GROUPS:
        r = report[cls]
        print(f'  {cls:<12}  {r["precision"]:>9.3f}  {r["recall"]:>9.3f}  '
              f'{r["f1-score"]:>9.3f}  {int(r["support"]):>9}')
    print()


def plot_confusion_matrix(cm, title, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=GROUPS).plot(
        ax=ax, colorbar=True, cmap='Oranges', values_format='.2f')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  Saved {path}')


def plot_confidence_histogram(proba, y_test, y_pred, path):
    max_conf = proba.max(axis=1)
    correct  = y_pred == y_test

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, 1, 21)
    ax.hist(max_conf[correct],  bins=bins, alpha=0.7, label='Correct',   color='steelblue')
    ax.hist(max_conf[~correct], bins=bins, alpha=0.7, label='Incorrect', color='tomato')
    ax.set_xlabel('Max class probability')
    ax.set_ylabel('Window count')
    ax.set_title('Prediction confidence — correct vs incorrect (test set)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  Saved {path}')

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    results_dir = sys.argv[1] if len(sys.argv) > 1 else 'results_clstm'

    if not os.path.isdir(results_dir):
        print(f'Error: directory not found: {results_dir}')
        print('Usage: python test_model_clstm.py [results_clstm]')
        sys.exit(1)

    for required in ('model.pt', 'X_test.npy', 'y_test.npy', 'results.json'):
        if not os.path.exists(os.path.join(results_dir, required)):
            print(f'Error: {required} not found in {results_dir}')
            print('Re-run train_model_clstm.py to regenerate.')
            sys.exit(1)

    print(f'── Testing C-LSTM in {results_dir}/ ─────────────────')
    model, X_test, y_test, meta = load(results_dir)
    print(f'  Architecture : {meta.get("architecture")}')
    print(f'  Best config  : {meta.get("best_config")}')
    print(f'  Test windows : {len(y_test)}  shape: {X_test.shape}')
    print(f'  Device       : {DEVICE}')
    print(f'  Class dist.  : { {GROUPS[i]: int((y_test == i).sum()) for i in range(len(GROUPS))} }')
    print()

    print('── Running predictions ───────────────────────────────')
    y_pred, proba, bal_acc, report, cm = evaluate(model, X_test, y_test)

    print('── Results ───────────────────────────────────────────')
    print_report(bal_acc, report, meta)

    print('── Saving plots ──────────────────────────────────────')
    plot_confusion_matrix(
        cm,
        'Test set confusion matrix (C-LSTM)',
        os.path.join(results_dir, 'confusion_matrix_test.png')
    )
    plot_confidence_histogram(
        proba, y_test, y_pred,
        os.path.join(results_dir, 'confidence_histogram_test.png')
    )

    print('\nDone.')
