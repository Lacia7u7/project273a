from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd

from train.calibration import apply_calibration, fit_calibrator_from_config
from train.metrics import compute_metrics, find_best_threshold


@dataclass
class EvaluationResult:
    """Structured container for evaluation artefacts."""

    split: str
    threshold: float
    metrics_primary: Dict[str, float]
    metrics_secondary: Dict[str, float]
    metrics_all: Dict[str, float]
    threshold_tuning: Optional[Dict[str, float]]
    calibration_model: Any = None
    subgroup_metrics: Optional[Dict[str, Dict[Any, Dict[str, float]]]] = None


def _select_metrics(source: Mapping[str, float], wanted: Iterable[str]) -> Dict[str, float]:
    return {name: source.get(name, float("nan")) for name in wanted}


def _prepare_metadata(metadata: Optional[Any], n_samples: int) -> Optional[pd.DataFrame]:
    if metadata is None:
        return None
    if isinstance(metadata, pd.DataFrame):
        df = metadata.copy()
    else:
        df = pd.DataFrame(metadata)
    if len(df) != n_samples:
        raise ValueError(
            f"Metadata length ({len(df)}) does not match number of samples ({n_samples})."
        )
    return df.reset_index(drop=True)


def tune_threshold_from_config(
    y_true: Iterable[int],
    y_prob: Iterable[float],
    config,
) -> Optional[Dict[str, float]]:
    eval_cfg = getattr(config, "evaluation", None)
    if eval_cfg is None or getattr(eval_cfg, "threshold_tuning", None) is None:
        return None

    thr_cfg = eval_cfg.threshold_tuning
    grid = getattr(thr_cfg, "grid", None)
    grid = grid if grid else None
    optimize_for = getattr(thr_cfg, "optimize_for", "f1_pos")

    return find_best_threshold(y_true, y_prob, optimize_for=optimize_for, grid=grid)


def evaluate_predictions(
    y_true: Iterable[int],
    y_prob: Iterable[float],
    config,
    *,
    split: str = "val",
    metadata: Optional[Any] = None,
    apply_config_calibration: bool = False,
    tune_threshold: bool = False,
    existing_threshold: Optional[float] = None,
) -> EvaluationResult:
    """Evaluate predictions according to the configuration settings."""

    eval_cfg = getattr(config, "evaluation", None)
    if eval_cfg is None:
        raise ValueError("Configuration is missing the 'evaluation' section.")

    probs = np.asarray(y_prob, dtype=float)
    labels = np.asarray(y_true)

    calibration_model = None
    if apply_config_calibration:
        calibration_model = fit_calibrator_from_config(labels, probs, config)
        probs = np.asarray(apply_calibration(calibration_model, probs))

    threshold_info: Optional[Dict[str, float]] = None
    threshold_value: float = existing_threshold if existing_threshold is not None else 0.5
    if tune_threshold:
        threshold_info = tune_threshold_from_config(labels, probs, config)
        if threshold_info is not None:
            threshold_value = float(threshold_info["threshold"])

    metrics_all = compute_metrics(labels, probs, threshold=threshold_value)
    metrics_primary = _select_metrics(metrics_all, getattr(eval_cfg, "metrics_primary", []))
    metrics_secondary = _select_metrics(
        metrics_all, getattr(eval_cfg, "metrics_secondary", [])
    )

    subgroup_results: Optional[Dict[str, Dict[Any, Dict[str, float]]]] = None
    df_meta = _prepare_metadata(metadata, len(labels))
    subgroup_cols = getattr(eval_cfg, "subgroup_metrics", [])
    if df_meta is not None and subgroup_cols:
        subgroup_results = {}
        for col in subgroup_cols:
            if col not in df_meta.columns:
                raise KeyError(f"Metadata is missing subgroup column '{col}'.")
            subgroup_results[col] = {}
            for value, idxs in df_meta.groupby(col).groups.items():
                mask = df_meta.index.isin(idxs)
                subgroup_metrics = compute_metrics(
                    labels[mask], probs[mask], threshold=threshold_value
                )
                subgroup_results[col][value] = subgroup_metrics

    return EvaluationResult(
        split=split,
        threshold=threshold_value,
        metrics_primary=metrics_primary,
        metrics_secondary=metrics_secondary,
        metrics_all=metrics_all,
        threshold_tuning=threshold_info,
        calibration_model=calibration_model,
        subgroup_metrics=subgroup_results,
    )
