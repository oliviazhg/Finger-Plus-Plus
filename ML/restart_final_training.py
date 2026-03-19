'''
Restart Final Model Training Script

Loads the best config from previous results_clstm/results.json
and retrains only the final model without redoing grid search/CV.
'''

import os
import copy
import json
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# ── Configuration ─────────────────────────────────────────────────────────────

WINDOWED_DIR = 'data_windowed'
RESULTS_DIR  = 'results_clstm'

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
N_CHANNELS        = 8

MAX_EPOCHS = 200
PATIENCE   = 20

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Model ─────────────────────────────────────────────────────────────────────

class CLSTM(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(N_CHANNELS, 32, kernel_size=5, padding=2),
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

# ── Data loading ──────────────────────────────────────────────────────────────

def _fpath(cls, phase):
    return os.path.join(WINDOWED_DIR, f"{cls.replace(' ', '_')}_{phase}.npy")

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

    return (np.concatenate(X_parts, axis=0),
            np.concatenate(y_parts),
            np.concatenate(g_parts),
            meta)

def split_data(X, y, groups):
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    return (X[train_idx], y[train_idx], groups[train_idx],
            X[test_idx],  y[test_idx])

# ── Training utilities ────────────────────────────────────────────────────────

def _to_tensors(X, y):
    return (torch.tensor(X, dtype=torch.float32).to(DEVICE),
            torch.tensor(y, dtype=torch.long).to(DEVICE))

def train_one(X_tr, y_tr, X_val, y_val, config, verbose=False):
    '''Train a single C-LSTM with early stopping. Returns best model.'''
    model   = CLSTM(dropout=config['dropout']).to(DEVICE)
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

def train_final(X_tr, y_tr, config):
    '''
    Train deployment model on all training data.
    A random 15% window split is used for early stopping only.
    '''
    rng      = np.random.default_rng(42)
    val_mask = rng.random(len(X_tr)) < 0.15
    tr_mask  = ~val_mask

    print('  Training final model (with early stopping)...')
    model = train_one(X_tr[tr_mask], y_tr[tr_mask],
                      X_tr[val_mask], y_tr[val_mask],
                      config, verbose=True)
    return model

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print(f'── Device : {DEVICE} ─────────────────────────────────')

    # Load best config from previous results
    results_path = os.path.join(RESULTS_DIR, 'results.json')
    if not os.path.exists(results_path):
        print(f'Error: {results_path} not found. Run full training first.')
        exit(1)

    with open(results_path, 'r') as f:
        results = json.load(f)

    best_config = results['best_config']
    print(f'Loaded best config: {best_config}')

    # Load data
    print('── Loading data ──────────────────────────────────────')
    X, y, groups, meta = load_data()
    print(f'  Total windows : {len(X)}  shape: {X.shape}')

    # Split data (same as before)
    print('── Train / test split (15% trials held out) ──────────')
    X_tr, y_tr, groups_tr, X_te, y_te = split_data(X, y, groups)
    print(f'  Train : {len(X_tr):>6} windows')

    # Train final model
    print('── Training final model ──────────────────────────────')
    final_model = train_final(X_tr, y_tr, best_config)

    # Save the new model
    torch.save(final_model.state_dict(), os.path.join(RESULTS_DIR, 'model_final.pt'))
    print(f'  Saved {RESULTS_DIR}/model_final.pt')

    print('\nDone. Final model retrained and saved as model_final.pt')