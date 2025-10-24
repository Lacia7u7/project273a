from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def calibrate_probabilities(y_prob, y_true, method: str = "isotonic"):
    """Train a calibration model (Platt scaling or isotonic) on given probabilities."""
    y_prob = np.array(y_prob)
    y_true = np.array(y_true)
    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(y_prob, y_true)
        return iso
    if method == "platt":
        lr = LogisticRegression(solver="lbfgs")
        lr.fit(y_prob.reshape(-1, 1), y_true)
        return lr
    return None


def apply_calibration(model, y_prob):
    """Apply a fitted calibration model (isotonic or logistic) to probabilities."""
    if model is None:
        return y_prob
    if isinstance(model, IsotonicRegression):
        return model.predict(y_prob)
    if isinstance(model, LogisticRegression):
        return model.predict_proba(y_prob.reshape(-1, 1))[:, 1]
    return y_prob


def fit_calibrator_from_config(
    y_true: Iterable[int],
    y_prob: Iterable[float],
    config,
):
    """Fit a calibration model based on ``config.evaluation.threshold_tuning`` settings."""

    cal_name: Optional[str] = getattr(
        getattr(getattr(config, "evaluation", None), "threshold_tuning", None),
        "calibration",
        None,
    )
    if not cal_name:
        return None

    cal_name = cal_name.lower()
    if cal_name in {"isotonic", "isotonic_regression"}:
        method = "isotonic"
    elif cal_name in {"platt", "logistic", "platt_scaling"}:
        method = "platt"
    else:
        raise ValueError(
            "Unknown calibration method '{}' (expected 'isotonic' or 'platt').".format(cal_name)
        )

    return calibrate_probabilities(y_prob, y_true, method=method)
