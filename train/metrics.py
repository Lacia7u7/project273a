import math
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


SUPPORTED_METRICS: Mapping[str, str] = {
    "auroc": "prob",
    "auprc": "prob",
    "brier": "prob",
    "ece": "prob",
    "f1_pos": "pred",
    "precision_pos": "pred",
    "recall_pos": "pred",
    "balanced_accuracy": "pred",
}


def _ensure_supported(metric_names: Iterable[str]) -> List[str]:
    names = []
    for name in metric_names:
        if name not in SUPPORTED_METRICS:
            raise KeyError(
                f"Unsupported metric '{name}'. Supported metrics: {sorted(SUPPORTED_METRICS.keys())}"
            )
        names.append(name)
    return names


def compute_metrics(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    *,
    threshold: float = 0.5,
    metrics: Optional[Iterable[str]] = None,
    num_calibration_bins: int = 10,
) -> Dict[str, float]:
    """Compute classification metrics from labels and predicted probabilities.

    Parameters
    ----------
    y_true, y_prob:
        Iterable of ground-truth labels and predicted probabilities.
    threshold:
        Probability threshold used to derive hard predictions for metrics that
        operate on class labels (e.g. precision, recall, F1).
    metrics:
        Optional iterable limiting the metrics to compute. When ``None`` all
        supported metrics are returned.
    num_calibration_bins:
        Number of bins to use for the expected calibration error (ECE).
    """

    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)
    y_pred = (y_prob_arr >= threshold).astype(int)

    if metrics is None:
        requested = list(SUPPORTED_METRICS.keys())
    else:
        requested = _ensure_supported(metrics)

    results: Dict[str, float] = {}

    # Probability-based metrics -------------------------------------------------
    if any(SUPPORTED_METRICS[m] == "prob" for m in requested):
        # Binary metrics (AUROC/AUPRC) are undefined if only one class present.
        binary = np.unique(y_true_arr).size == 2
        if "auroc" in requested:
            results["auroc"] = roc_auc_score(y_true_arr, y_prob_arr) if binary else math.nan
        if "auprc" in requested:
            results["auprc"] = (
                average_precision_score(y_true_arr, y_prob_arr) if binary else math.nan
            )
        if "brier" in requested:
            results["brier"] = brier_score_loss(y_true_arr, y_prob_arr)
        if "ece" in requested:
            bins = np.linspace(0.0, 1.0, num_calibration_bins + 1)
            binids = np.digitize(y_prob_arr, bins) - 1
            ece = 0.0
            total = len(y_true_arr)
            for i in range(num_calibration_bins):
                mask = binids == i
                if not np.any(mask):
                    continue
                bin_acc = np.mean(y_true_arr[mask])
                bin_conf = np.mean(y_prob_arr[mask])
                ece += abs(bin_conf - bin_acc) * (np.sum(mask) / total)
            results["ece"] = float(ece)

    # Prediction-based metrics --------------------------------------------------
    if any(SUPPORTED_METRICS[m] == "pred" for m in requested):
        if "f1_pos" in requested:
            results["f1_pos"] = f1_score(y_true_arr, y_pred, pos_label=1, zero_division=0)
        if "precision_pos" in requested:
            results["precision_pos"] = precision_score(
                y_true_arr, y_pred, pos_label=1, zero_division=0
            )
        if "recall_pos" in requested:
            results["recall_pos"] = recall_score(y_true_arr, y_pred, pos_label=1, zero_division=0)
        if "balanced_accuracy" in requested:
            tp = float(np.sum((y_true_arr == 1) & (y_pred == 1)))
            fn = float(np.sum((y_true_arr == 1) & (y_pred == 0)))
            tn = float(np.sum((y_true_arr == 0) & (y_pred == 0)))
            fp = float(np.sum((y_true_arr == 0) & (y_pred == 1)))
            tpr = tp / (tp + fn) if (tp + fn) else 0.0
            tnr = tn / (tn + fp) if (tn + fp) else 0.0
            results["balanced_accuracy"] = 0.5 * (tpr + tnr)

    return results


def find_best_threshold(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    *,
    optimize_for: str = "f1_pos",
    grid: Optional[Iterable[float]] = None,
    num_calibration_bins: int = 10,
) -> Dict[str, float]:
    """Find the best probability threshold for the requested metric.

    Returns a dictionary with the chosen threshold and the metric value.
    """

    metric_name = _ensure_supported([optimize_for])[0]
    thresholds = list(grid) if grid is not None else [i / 100 for i in range(1, 100)]
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)

    best_thr = 0.5
    best_val = -math.inf
    for thr in thresholds:
        metrics = compute_metrics(
            y_true_arr,
            y_prob_arr,
            threshold=float(thr),
            metrics=[metric_name],
            num_calibration_bins=num_calibration_bins,
        )
        val = metrics.get(metric_name)
        if np.isnan(val):
            continue
        if val > best_val:
            best_val = float(val)
            best_thr = float(thr)

    return {"threshold": best_thr, "metric": metric_name, "metric_value": best_val}
