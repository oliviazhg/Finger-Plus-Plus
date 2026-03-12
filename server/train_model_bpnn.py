'''
BPNN Training Script — Steady Phase Only

Loads processed steady-state features from data_processed/,
merges sub-classes into 4 groups, and trains a 3-layer backpropagation
neural network using Adam optimisation.

Architecture : 48 inputs → 128 hidden (ReLU) → 4 outputs (softmax)
Optimiser    : Adam with CrossEntropyLoss
Normalisation: StandardScaler fitted on training fold only (no leakage)

Features are the same 48 as the RF scripts:
  MAV, RMS, VAR, WL, SSC, WAMP × 8 channels

15% of trials are held out as a test set before any training occurs.
Trial-aware GroupKFold prevents window leakage between folds.

Hyperparameter grid searched: learning rate, dropout, batch size.

Outputs saved to results_bpnn/:
  model.pt                — final model weights (PyTorch state dict)
  scaler.joblib           — StandardScaler fitted on all training data
  X_test.npy / y_test.npy — held-out test set for test_model_bpnn.py
  results.json            — metrics, best config, fold scores
  confusion_matrix.png    — normalised CV confusion matrix
  fold_scores.png         — per-fold balanced accuracy
  cv_predictions.npy      — (y_true, y_pred) stacked array from CV folds

Run: python train_model_bpnn.py
'''

import os
import copy
import json
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import joblib

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

# ── Configuration ─────────────────────────────────────────────────────────────

PROCESSED_DIR = 'data_processed'
RESULTS_DIR   = 'results_bpnn'

CLASS_GROUPS = {
    'cylindrical forward': 'cylindrical',
    'cylindrical by side': 'cylindrical',
    'lateral palm up':     'lateral',
    'lateral palm down':   'lateral',
    'lateral forward':     'lateral',
    'lateral by side':     'lateral',
    'palm':                'palm',
    'rest':                'rest',
}
GROUPS       = ['cylindrical', 'lateral', 'palm', 'rest']
GROUP_TO_INT = {g: i for i, g in enumerate(GROUPS)}

STEADY_SAMPLES    = 800
WINDOW_SIZE       = 40
STRIDE            = 20
WINDOWS_PER_TRIAL = (STEADY_SAMPLES - WINDOW_SIZE) // STRIDE + 1  # 39

CV_FOLDS  = 5
TEST_SIZE = 0.15

MAX_EPOCHS = 300
PATIENCE   = 30   # early-stopping patience (epochs without val-loss improvement)

PARAM_GRID = {
    'lr':         [1e-3, 3e-4, 1e-4],
    'dropout':    [0.0, 0.2, 0.3],
    'batch_size': [64, 256],
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Model ─────────────────────────────────────────────────────────────────────

class BPNN(nn.Module):
    '''48 → 128 (ReLU [+ Dropout]) → 4 (raw logits for CrossEntropyLoss).'''

    def __init__(self, dropout=0.0):
        super().__init__()
        layers = [nn.Linear(48, 128), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(128, 4))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ── Data loading ──────────────────────────────────────────────────────────────

def _fpath(cls, phase):
    return os.path.join(PROCESSED_DIR, f"{cls.replace(' ', '_')}_{phase}.npy")


def load_data():
    X_parts, y_parts, g_parts = [], [], []
    trial_counter = 0
    meta = {'class_counts': {}, 'trial_counts': {}}

    for sub_cls, group in CLASS_GROUPS.items():
        path = _fpath(sub_cls, 'steady')
        if not os.path.exists(path):
            continue
        data     = np.load(path)
        n_trials = data.shape[0] // WINDOWS_PER_TRIAL
        data     = data[:n_trials * WINDOWS_PER_TRIAL]

        trial_ids = np.repeat(
            np.arange(trial_counter, trial_counter + n_trials),
            WINDOWS_PER_TRIAL
        )
        trial_counter += n_trials

        label = GROUP_TO_INT[group]
        X_parts.append(data)
        y_parts.append(np.full(len(data), label, dtype=np.int32))
        g_parts.append(trial_ids)

        meta['trial_counts'][sub_cls] = n_trials
        meta['class_counts'][group]   = meta['class_counts'].get(group, 0) + len(data)

    return np.vstack(X_parts), np.concatenate(y_parts), np.concatenate(g_parts), meta


# ── Train / test split ────────────────────────────────────────────────────────

def split_data(X, y, groups):
    '''Hold out 15% of trials as a test set (trial-aware, no window leakage).'''
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    return (X[train_idx], y[train_idx], groups[train_idx],
            X[test_idx],  y[test_idx])

# ── Training utilities ────────────────────────────────────────────────────────

def _to_tensors(X, y):
    return (torch.tensor(X, dtype=torch.float32).to(DEVICE),
            torch.tensor(y, dtype=torch.long).to(DEVICE))


def train_one(X_tr, y_tr, X_val, y_val, config, verbose=False):
    '''
    Train a single BPNN with early stopping on validation loss.
    Returns the model loaded with the best (lowest val-loss) weights.
    '''
    model   = BPNN(dropout=config['dropout']).to(DEVICE)
    opt     = torch.optim.Adam(model.parameters(), lr=config['lr'])
    loss_fn = nn.CrossEntropyLoss()

    Xtr_t,  ytr_t  = _to_tensors(X_tr,  y_tr)
    Xval_t, yval_t = _to_tensors(X_val, y_val)

    loader = DataLoader(
        TensorDataset(Xtr_t, ytr_t),
        batch_size=config['batch_size'],
        shuffle=True
    )

    best_val_loss = float('inf')
    best_state    = None
    patience_cnt  = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xval_t), yval_t).item()

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state    = copy.deepcopy(model.state_dict())
            patience_cnt  = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                if verbose:
                    print(f'    Early stop at epoch {epoch + 1}')
                break

    model.load_state_dict(best_state)
    return model


def predict(model, X):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32).to(DEVICE))
        return logits.argmax(dim=1).cpu().numpy()


def predict_proba(model, X):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32).to(DEVICE))
        return torch.softmax(logits, dim=1).cpu().numpy()

# ── Grid search ───────────────────────────────────────────────────────────────

def _cv_score_for_config(X_tr, y_tr, groups_tr, config):
    '''Mean balanced accuracy across CV folds for one hyperparameter config.'''
    gkf         = GroupKFold(n_splits=CV_FOLDS)
    fold_scores = []
    for tr_idx, val_idx in gkf.split(X_tr, y_tr, groups_tr):
        scaler = StandardScaler().fit(X_tr[tr_idx])
        Xf_tr  = scaler.transform(X_tr[tr_idx])
        Xf_val = scaler.transform(X_tr[val_idx])
        model  = train_one(Xf_tr, y_tr[tr_idx], Xf_val, y_tr[val_idx], config)
        fold_scores.append(
            balanced_accuracy_score(y_tr[val_idx], predict(model, Xf_val))
        )
    return float(np.mean(fold_scores))


def run_grid_search(X_tr, y_tr, groups_tr):
    grid        = list(ParameterGrid(PARAM_GRID))
    best_score  = -1.0
    best_config = None

    all_results = []
    for config in tqdm(grid, desc='  Grid search', ncols=72):
        score = _cv_score_for_config(X_tr, y_tr, groups_tr, config)
        tqdm.write(f'    {config}  →  {score:.3f}')
        all_results.append({'config': config, 'cv_score': score})
        if score > best_score:
            best_score  = score
            best_config = config

    all_results.sort(key=lambda r: r['cv_score'], reverse=True)
    return best_config, best_score, all_results

# ── Full CV with best config ──────────────────────────────────────────────────

def run_cv(X_tr, y_tr, groups_tr, config):
    '''
    Re-run CV with the chosen config to collect OOF predictions and per-fold
    scores for reporting. (Separate from grid search to avoid double-counting.)
    '''
    gkf = GroupKFold(n_splits=CV_FOLDS)
    fold_scores          = []
    all_true, all_pred   = [], []

    for tr_idx, val_idx in tqdm(
            gkf.split(X_tr, y_tr, groups_tr),
            desc='  CV predict ', total=CV_FOLDS, ncols=72):
        scaler = StandardScaler().fit(X_tr[tr_idx])
        Xf_tr  = scaler.transform(X_tr[tr_idx])
        Xf_val = scaler.transform(X_tr[val_idx])

        model  = train_one(Xf_tr, y_tr[tr_idx], Xf_val, y_tr[val_idx], config)
        y_pred = predict(model, Xf_val)

        fold_scores.append(balanced_accuracy_score(y_tr[val_idx], y_pred))
        all_true.extend(y_tr[val_idx])
        all_pred.extend(y_pred)

    return fold_scores, np.array(all_true), np.array(all_pred)

# ── Final model ───────────────────────────────────────────────────────────────

def train_final(X_tr, y_tr, config):
    '''
    Train the deployment model on all training data.

    A random 15% window split (not trial-aware) is used purely as an
    early-stopping validation set; it is not used for evaluation.
    The StandardScaler is fitted on the full training set.
    '''
    print('  Fitting scaler on full training set...')
    scaler   = StandardScaler().fit(X_tr)
    X_scaled = scaler.transform(X_tr)

    rng      = np.random.default_rng(42)
    val_mask = rng.random(len(X_scaled)) < 0.15
    tr_mask  = ~val_mask

    print('  Training final model (with early stopping)...')
    model = train_one(
        X_scaled[tr_mask], y_tr[tr_mask],
        X_scaled[val_mask], y_tr[val_mask],
        config, verbose=True
    )
    return model, scaler

# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm, labels, title, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(
        ax=ax, colorbar=True, cmap='Blues', values_format='.2f')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  Saved {path}')


def plot_fold_scores(fold_scores, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(1, len(fold_scores) + 1), fold_scores, color='steelblue')
    ax.axhline(np.mean(fold_scores), color='red', linestyle='--',
               label=f'Mean {np.mean(fold_scores):.3f}')
    ax.set_xlabel('Fold')
    ax.set_ylabel('Balanced accuracy')
    ax.set_title('Per-fold balanced accuracy (trial-aware CV, steady only)')
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  Saved {path}')

# ── Save ──────────────────────────────────────────────────────────────────────

def save_results(model, scaler, best_config, best_gs_score, all_gs_results,
                 fold_scores, y_tr, y_oof, y_pred_oof,
                 X_test, y_test, meta):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(RESULTS_DIR, 'model.pt'))
    print(f'  Saved {RESULTS_DIR}/model.pt')

    joblib.dump(scaler, os.path.join(RESULTS_DIR, 'scaler.joblib'))
    print(f'  Saved {RESULTS_DIR}/scaler.joblib')

    np.save(os.path.join(RESULTS_DIR, 'X_test.npy'), X_test)
    np.save(os.path.join(RESULTS_DIR, 'y_test.npy'), y_test)
    print(f'  Saved {RESULTS_DIR}/X_test.npy  y_test.npy  ({len(y_test)} windows)')

    np.save(os.path.join(RESULTS_DIR, 'cv_predictions.npy'),
            np.stack([y_oof, y_pred_oof]))
    np.save(os.path.join(RESULTS_DIR, 'fold_scores.npy'),
            np.array(fold_scores))

    bal_acc = balanced_accuracy_score(y_oof, y_pred_oof)
    cm      = confusion_matrix(y_oof, y_pred_oof, normalize='true')
    report  = classification_report(y_oof, y_pred_oof,
                                    target_names=GROUPS, output_dict=True)

    results = {
        'model':                   'BPNN 48→128→4',
        'architecture':            '48 → 128 (ReLU) → 4 (softmax)',
        'optimizer':               'Adam',
        'best_config':             best_config,
        'best_gs_cv_score':        float(best_gs_score),
        'grid_search_results':     all_gs_results,
        'oof_balanced_acc':        float(bal_acc),
        'fold_scores':             [float(s) for s in fold_scores],
        'fold_mean':               float(np.mean(fold_scores)),
        'fold_std':                float(np.std(fold_scores)),
        'classification_report':   report,
        'confusion_matrix':        cm.tolist(),
        'class_order':             GROUPS,
        'phases_used':             ['steady'],
        'test_size_fraction':      TEST_SIZE,
        'test_windows':            int(len(y_test)),
        'train_windows':           int(len(y_tr)),
        'windows_per_trial':       WINDOWS_PER_TRIAL,
        'trial_counts':            meta['trial_counts'],
        'class_window_counts':     meta['class_counts'],
        'cv_folds':                CV_FOLDS,
        'feature_dim':             48,
        'max_epochs':              MAX_EPOCHS,
        'early_stopping_patience': PATIENCE,
        'device':                  str(DEVICE),
        'param_grid':              {k: list(v) for k, v in PARAM_GRID.items()},
    }
    json_path = os.path.join(RESULTS_DIR, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'  Saved {json_path}')

    plot_confusion_matrix(
        cm, GROUPS,
        'CV confusion matrix (normalised, steady only)',
        os.path.join(RESULTS_DIR, 'confusion_matrix.png')
    )
    plot_fold_scores(fold_scores, os.path.join(RESULTS_DIR, 'fold_scores.png'))

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print(f'── Device : {DEVICE} ─────────────────────────────────')
    print()

    print('── Loading data ──────────────────────────────────────')
    X, y, groups, meta = load_data()
    print(f'  Total windows : {len(X)}')
    print(f'  Unique trials : {len(np.unique(groups))}')
    for sub_cls, n in meta['trial_counts'].items():
        g = CLASS_GROUPS[sub_cls]
        print(f'    {sub_cls:<24} ({g:<12}) {n:>3} trials')
    print()

    print('── Train / test split (15% trials held out) ──────────')
    X_tr, y_tr, groups_tr, X_te, y_te = split_data(X, y, groups)
    print(f'  Train : {len(X_tr):>6} windows  ({len(np.unique(groups_tr))} trials)')
    print(f'  Test  : {len(X_te):>6} windows')
    print()

    print('── Grid search (trial-aware 5-fold CV on train set) ──')
    n_configs = len(list(ParameterGrid(PARAM_GRID)))
    print(f'  {n_configs} configs × {CV_FOLDS} folds = {n_configs * CV_FOLDS} training runs')
    best_config, best_gs_score, all_gs_results = run_grid_search(X_tr, y_tr, groups_tr)
    print(f'  Best config       : {best_config}')
    print(f'  Best CV bal. acc. : {best_gs_score:.3f}')
    print()

    print('── Full CV with best config ──────────────────────────')
    fold_scores, y_oof, y_pred_oof = run_cv(X_tr, y_tr, groups_tr, best_config)
    bal_acc = balanced_accuracy_score(y_oof, y_pred_oof)
    print(f'  OOF balanced acc. : {bal_acc:.3f}')
    print(f'  Fold mean ± std   : {np.mean(fold_scores):.3f} ± {np.std(fold_scores):.3f}')
    print()
    print(classification_report(y_oof, y_pred_oof, target_names=GROUPS))

    print('── Training final model ──────────────────────────────')
    final_model, scaler = train_final(X_tr, y_tr, best_config)
    print()

    print('── Saving ────────────────────────────────────────────')
    save_results(final_model, scaler, best_config, best_gs_score, all_gs_results,
                 fold_scores, y_tr, y_oof, y_pred_oof, X_te, y_te, meta)

    print(f'\nDone. Run python test_model_bpnn.py to evaluate on the held-out test set.')
