'''
Random Forest Training Script — All Phases (init + steady + release)

Same as train_model_rf_steady.py but trains on all three phases combined.
Windows from the same trial share a group ID across phases so
GroupKFold never splits a trial, even across different phases.

15% of trials are held out as a test set before any training occurs.

Outputs saved to results_all_phases/:
  model.joblib            — final model trained on 85% train trials
  X_test.npy / y_test.npy — held-out test set for test_model_rf.py
  results.json            — metrics, params, fold scores, class counts
  confusion_matrix.png    — normalised CV confusion matrix
  feature_importance.png  — top-20 feature importances
  cv_predictions.npy      — (y_true, y_pred) from CV folds (train set only)
  fold_scores.npy         — balanced_accuracy per fold

Run: python train_model_rf_all_phases.py
'''

import os
import json
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import joblib
from contextlib import contextmanager

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (GroupKFold, GroupShuffleSplit,
                                     GridSearchCV, cross_val_predict,
                                     ParameterGrid)
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

# ── Configuration ─────────────────────────────────────────────────────────────

PROCESSED_DIR = 'data_processed'
RESULTS_DIR   = 'results_rf_all_phases'

CLASS_GROUPS = {
    'cylindrical forward': 'cylindrical',
    'cylindrical by side': 'cylindrical',
    'lateral palm down':   'lateral',
    'lateral forward':     'lateral',
    'lateral by side':     'lateral',
    'palm':                'palm',
    'rest':                'rest',
}
GROUPS = ['cylindrical', 'lateral', 'palm', 'rest']
GROUP_TO_INT = {g: i for i, g in enumerate(GROUPS)}

WINDOW_SIZE = 40
STRIDE      = 20
PHASE_SAMPLES = {'init': 400, 'steady': 800, 'release': 400}
WINDOWS_PER_PHASE = {
    phase: (n - WINDOW_SIZE) // STRIDE + 1
    for phase, n in PHASE_SAMPLES.items()
}  # init: 19, steady: 39, release: 19

FEATURE_NAMES = (
    [f'MAV_ch{i+1}'  for i in range(8)] +
    [f'RMS_ch{i+1}'  for i in range(8)] +
    [f'VAR_ch{i+1}'  for i in range(8)] +
    [f'WL_ch{i+1}'   for i in range(8)] +
    [f'SSC_ch{i+1}'  for i in range(8)] +
    [f'WAMP_ch{i+1}' for i in range(8)]
)

CV_FOLDS  = 5
TEST_SIZE = 0.15

PARAM_GRID = {
    'n_estimators':     [100, 300],
    'max_features':     ['sqrt', 0.3],
    'max_depth':        [10, 20, None],
    'min_samples_leaf': [1, 5],
}

# ── Progress bar for joblib ────────────────────────────────────────────────────

@contextmanager
def tqdm_joblib(tqdm_object):
    '''Patch joblib to report progress into a tqdm bar.'''
    class TqdmCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)
    old = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old
        tqdm_object.close()

# ── Load ──────────────────────────────────────────────────────────────────────

def _fpath(cls, phase):
    return os.path.join(PROCESSED_DIR, f"{cls.replace(' ', '_')}_{phase}.npy")


def load_data():
    X_parts, y_parts, g_parts = [], [], []
    trial_counter = 0
    meta = {'class_counts': {}, 'trial_counts': {}, 'phase_counts': {}}

    for sub_cls, group in CLASS_GROUPS.items():
        steady_path = _fpath(sub_cls, 'steady')
        if not os.path.exists(steady_path):
            continue

        wpt_steady = WINDOWS_PER_PHASE['steady']
        n_trials   = np.load(steady_path).shape[0] // wpt_steady
        base_ids   = np.arange(trial_counter, trial_counter + n_trials)
        trial_counter += n_trials

        label = GROUP_TO_INT[group]
        phase_window_counts = {}

        for phase in ('init', 'steady', 'release'):
            path = _fpath(sub_cls, phase)
            if not os.path.exists(path):
                continue
            wpt  = WINDOWS_PER_PHASE[phase]
            data = np.load(path)[:n_trials * wpt]
            trial_ids = np.repeat(base_ids, wpt)
            X_parts.append(data)
            y_parts.append(np.full(len(data), label, dtype=np.int32))
            g_parts.append(trial_ids)
            phase_window_counts[phase] = len(data)

        meta['trial_counts'][sub_cls] = n_trials
        meta['class_counts'][group]   = meta['class_counts'].get(group, 0) + sum(phase_window_counts.values())
        meta['phase_counts'][sub_cls] = phase_window_counts

    return np.vstack(X_parts), np.concatenate(y_parts), np.concatenate(g_parts), meta


# ── Train / test split ────────────────────────────────────────────────────────

def split_data(X, y, groups):
    '''Hold out 15% of trials as a test set (trial-aware, no window leakage).'''
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    return (X[train_idx], y[train_idx], groups[train_idx],
            X[test_idx],  y[test_idx])


# ── Training ──────────────────────────────────────────────────────────────────

def train(X_tr, y_tr, groups_tr):
    n_fits = CV_FOLDS * len(list(ParameterGrid(PARAM_GRID)))
    rf  = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)
    gkf = GroupKFold(n_splits=CV_FOLDS)
    gs  = GridSearchCV(rf, PARAM_GRID, cv=gkf, scoring='balanced_accuracy',
                       refit=True, n_jobs=-1, return_train_score=True, verbose=0)

    with tqdm_joblib(tqdm(desc='  Grid search', total=n_fits, ncols=72)):
        gs.fit(X_tr, y_tr, groups=groups_tr)
    return gs


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(gs, X_tr, y_tr, groups_tr):
    best = gs.best_estimator_
    gkf  = GroupKFold(n_splits=CV_FOLDS)

    with tqdm_joblib(tqdm(desc='  CV predict ', total=CV_FOLDS, ncols=72)):
        y_pred = cross_val_predict(best, X_tr, y_tr, cv=gkf, groups=groups_tr,
                                   method='predict', n_jobs=-1)

    bal_acc = balanced_accuracy_score(y_tr, y_pred)
    report  = classification_report(y_tr, y_pred, target_names=GROUPS, output_dict=True)
    cm      = confusion_matrix(y_tr, y_pred, normalize='true')

    fold_scores = []
    for tr_idx, te_idx in tqdm(gkf.split(X_tr, y_tr, groups_tr),
                                desc='  Fold scores', total=CV_FOLDS, ncols=72):
        best.fit(X_tr[tr_idx], y_tr[tr_idx])
        fold_scores.append(balanced_accuracy_score(y_tr[te_idx], best.predict(X_tr[te_idx])))

    final_model = RandomForestClassifier(**gs.best_params_, class_weight='balanced',
                                         random_state=42, n_jobs=-1)
    print('  Fitting final model on train set...')
    final_model.fit(X_tr, y_tr)

    return {'y_pred': y_pred, 'bal_acc': bal_acc, 'report': report,
            'cm': cm, 'fold_scores': fold_scores, 'final_model': final_model}


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


def plot_feature_importance(model, path, top_n=20):
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(top_n), importances[idx])
    ax.set_xticks(range(top_n))
    ax.set_xticklabels([FEATURE_NAMES[i] for i in idx], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Mean decrease in impurity')
    ax.set_title(f'Top {top_n} feature importances (all phases)')
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
    ax.set_title('Per-fold balanced accuracy (trial-aware CV, all phases)')
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  Saved {path}')


# ── Save ──────────────────────────────────────────────────────────────────────

def save_results(gs, eval_out, meta, y_tr, X_test, y_test):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    joblib.dump(eval_out['final_model'], os.path.join(RESULTS_DIR, 'model.joblib'))
    print(f"  Saved {RESULTS_DIR}/model.joblib")

    np.save(os.path.join(RESULTS_DIR, 'X_test.npy'), X_test)
    np.save(os.path.join(RESULTS_DIR, 'y_test.npy'), y_test)
    print(f"  Saved {RESULTS_DIR}/X_test.npy  y_test.npy  ({len(y_test)} windows)")

    np.save(os.path.join(RESULTS_DIR, 'cv_predictions.npy'),
            np.stack([y_tr, eval_out['y_pred']]))
    np.save(os.path.join(RESULTS_DIR, 'fold_scores.npy'),
            np.array(eval_out['fold_scores']))

    cv_results_path = os.path.join(RESULTS_DIR, 'cv_results.json')
    with open(cv_results_path, 'w') as f:
        json.dump({k: (v.tolist() if hasattr(v, 'tolist') else list(v))
                   for k, v in gs.cv_results_.items()}, f, indent=2)
    print(f'  Saved {cv_results_path}')

    results = {
        'best_params':            gs.best_params_,
        'best_cv_score':          float(gs.best_score_),
        'oof_balanced_acc':       float(eval_out['bal_acc']),
        'fold_scores':            [float(s) for s in eval_out['fold_scores']],
        'fold_mean':              float(np.mean(eval_out['fold_scores'])),
        'fold_std':               float(np.std(eval_out['fold_scores'])),
        'classification_report':  eval_out['report'],
        'confusion_matrix':       eval_out['cm'].tolist(),
        'class_order':            GROUPS,
        'phases_used':            ['init', 'steady', 'release'],
        'windows_per_phase':      WINDOWS_PER_PHASE,
        'test_size_fraction':     TEST_SIZE,
        'test_windows':           int(len(y_test)),
        'train_windows':          int(len(y_tr)),
        'trial_counts':           meta['trial_counts'],
        'class_window_counts':    meta['class_counts'],
        'phase_window_counts':    meta['phase_counts'],
        'cv_folds':               CV_FOLDS,
        'feature_dim':            48,
        'param_grid':             {k: [str(v) for v in vs] for k, vs in PARAM_GRID.items()},
    }
    json_path = os.path.join(RESULTS_DIR, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'  Saved {json_path}')

    plot_confusion_matrix(eval_out['cm'], GROUPS,
                          'CV confusion matrix (normalised, all phases)',
                          os.path.join(RESULTS_DIR, 'confusion_matrix.png'))
    plot_feature_importance(eval_out['final_model'],
                            os.path.join(RESULTS_DIR, 'feature_importance.png'))
    plot_fold_scores(eval_out['fold_scores'],
                     os.path.join(RESULTS_DIR, 'fold_scores.png'))


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('── Loading data (init + steady + release) ────────────')
    X, y, groups, meta = load_data()
    print(f'  Total windows : {len(X)}')
    print(f'  Unique trials : {len(np.unique(groups))}')
    for sub_cls, n in meta['trial_counts'].items():
        g  = CLASS_GROUPS[sub_cls]
        pc = meta['phase_counts'][sub_cls]
        print(f'    {sub_cls:<24} ({g:<12}) {n:>3} trials  '
              f'init:{pc.get("init",0)} steady:{pc.get("steady",0)} release:{pc.get("release",0)}')
    print()

    print('── Train / test split (15% trials held out) ──────────')
    X_tr, y_tr, groups_tr, X_te, y_te = split_data(X, y, groups)
    print(f'  Train : {len(X_tr):>6} windows  ({len(np.unique(groups_tr))} trials)')
    print(f'  Test  : {len(X_te):>6} windows')
    print()

    print('── Grid search (trial-aware 5-fold CV on train set) ──')
    gs = train(X_tr, y_tr, groups_tr)
    print(f'  Best params       : {gs.best_params_}')
    print(f'  Best CV bal. acc. : {gs.best_score_:.3f}')
    print()

    print('── Evaluating on train set (CV) ──────────────────────')
    eval_out = evaluate(gs, X_tr, y_tr, groups_tr)
    print(f'  OOF balanced acc. : {eval_out["bal_acc"]:.3f}')
    print(f'  Fold mean ± std   : {np.mean(eval_out["fold_scores"]):.3f} ± {np.std(eval_out["fold_scores"]):.3f}')
    print()
    print(classification_report(y_tr, eval_out['y_pred'], target_names=GROUPS))

    print('── Saving ────────────────────────────────────────────')
    save_results(gs, eval_out, meta, y_tr, X_te, y_te)
    print(f'\nDone. Run python test_model_rf.py {RESULTS_DIR} to evaluate on the held-out test set.')
