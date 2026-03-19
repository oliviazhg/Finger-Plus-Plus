'''
Model Test Script

Evaluates a trained model on its held-out test set (15% of trials
set aside during training, never seen by the model).

Usage:
  python test_model_rf.py                        # defaults to results_all_phases/
  python test_model_rf.py results_steady         # test the steady-only RF model
  python test_model_rf.py results_all_phases     # test the all-phases RF model
'''

import os
import sys
import json
import warnings
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

import joblib
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

GROUPS = ['cylindrical', 'lateral', 'palm', 'rest']

# ── Load ──────────────────────────────────────────────────────────────────────

def load(results_dir):
    model  = joblib.load(os.path.join(results_dir, 'model.joblib'))
    X_test = np.load(os.path.join(results_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(results_dir, 'y_test.npy'))

    meta_path = os.path.join(results_dir, 'results.json')
    with open(meta_path) as f:
        meta = json.load(f)

    return model, X_test, y_test, meta


# ── Evaluate ──────────────────────────────────────────────────────────────────

def evaluate(model, X_test, y_test):
    y_pred  = model.predict(X_test)
    proba   = model.predict_proba(X_test)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    report  = classification_report(y_test, y_pred, target_names=GROUPS, output_dict=True)
    cm      = confusion_matrix(y_test, y_pred, normalize='true')
    return y_pred, proba, bal_acc, report, cm


# ── Display ───────────────────────────────────────────────────────────────────

def print_report(bal_acc, report, meta):
    cv_acc = meta.get('oof_balanced_acc', meta.get('best_cv_score'))
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
    '''Histogram of max-class confidence, split by correct vs incorrect.'''
    max_conf   = proba.max(axis=1)
    correct    = y_pred == y_test

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
    results_dir = sys.argv[1] if len(sys.argv) > 1 else 'results_all_phases'

    if not os.path.isdir(results_dir):
        print(f'Error: directory not found: {results_dir}')
        print('Usage: python test_model_rf.py [results_steady | results_all_phases]')
        sys.exit(1)

    for required in ('model.joblib', 'X_test.npy', 'y_test.npy', 'results.json'):
        if not os.path.exists(os.path.join(results_dir, required)):
            print(f'Error: {required} not found in {results_dir}')
            print('Re-run the training script to regenerate the test set.')
            sys.exit(1)

    print(f'── Testing model in {results_dir}/ ──────────────────')
    model, X_test, y_test, meta = load(results_dir)
    phases = meta.get('phases_used', ['steady'])
    print(f'  Phases used in training : {phases}')
    print(f'  Test windows            : {len(y_test)}')
    print(f'  Class distribution      : { {GROUPS[i]: int((y_test==i).sum()) for i in range(len(GROUPS))} }')
    print()

    print('── Running predictions ───────────────────────────────')
    y_pred, proba, bal_acc, report, cm = evaluate(model, X_test, y_test)

    print('── Results ───────────────────────────────────────────')
    print_report(bal_acc, report, meta)

    print('── Saving plots ──────────────────────────────────────')
    phase_tag = '+'.join(phases)
    plot_confusion_matrix(
        cm,
        f'Test set confusion matrix ({phase_tag})',
        os.path.join(results_dir, 'confusion_matrix_test.png')
    )
    plot_confidence_histogram(
        proba, y_test, y_pred,
        os.path.join(results_dir, 'confidence_histogram_test.png')
    )

    print('\nDone.')
