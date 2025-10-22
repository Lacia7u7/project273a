import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, confusion_matrix, balanced_accuracy_score, brier_score_loss

def compute_metrics(y_true, y_prob):
    """Compute classification metrics from labels and predicted probabilities."""
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {}
    if len(np.unique(y_true)) == 2:
        metrics['auroc'] = roc_auc_score(y_true, y_prob)
        metrics['auprc'] = average_precision_score(y_true, y_prob)
    metrics['f1_pos'] = f1_score(y_true, y_pred, pos_label=1)
    metrics['precision_pos'] = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    metrics['recall_pos'] = recall_score(y_true, y_pred, pos_label=1)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    metrics['brier'] = brier_score_loss(y_true, y_prob)
    # ECE (expected calibration error)
    n_bins = 10
    bins = np.linspace(0,1,n_bins+1)
    binids = np.digitize(y_prob, bins) - 1
    ece = 0.0
    for i in range(n_bins):
        mask = binids == i
        if np.any(mask):
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            ece += np.abs(bin_conf - bin_acc) * (np.sum(mask) / len(y_true))
    metrics['ece'] = ece
    return metrics

def find_best_threshold(y_true, y_prob, optimize_for='f1_pos', grid=None):
    """Find best probability threshold based on a metric."""
    if grid is None:
        grid = [i/100 for i in range(1,100)]
    best_thr = 0.5
    best_val = -np.inf
    for thr in grid:
        y_pred = (y_prob >= thr).astype(int)
        if optimize_for == 'f1_pos':
            val = f1_score(y_true, y_pred, pos_label=1)
        elif optimize_for == 'precision_pos':
            val = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        elif optimize_for == 'recall_pos':
            val = recall_score(y_true, y_pred, pos_label=1)
        else:
            val = f1_score(y_true, y_pred, pos_label=1)
        if val > best_val:
            best_val = val
            best_thr = thr
    return best_thr, best_val
