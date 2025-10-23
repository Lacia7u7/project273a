import numpy as np
from types import SimpleNamespace

from train.evaluation import EvaluationResult, evaluate_predictions, tune_threshold_from_config
from train.metrics import compute_metrics, find_best_threshold


def _make_eval_config():
    threshold_cfg = SimpleNamespace(optimize_for="f1_pos", grid=[0.25, 0.5, 0.75], calibration="platt")
    plots_cfg = SimpleNamespace()
    eval_cfg = SimpleNamespace(
        metrics_primary=["f1_pos", "precision_pos"],
        metrics_secondary=["auroc"],
        threshold_tuning=threshold_cfg,
        plots=plots_cfg,
        subgroup_metrics=["group"],
    )
    return SimpleNamespace(evaluation=eval_cfg)


def test_compute_metrics_threshold_dependency():
    y_true = [0, 0, 1, 1]
    y_prob = [0.1, 0.4, 0.6, 0.9]

    metrics_default = compute_metrics(y_true, y_prob)
    metrics_high_threshold = compute_metrics(y_true, y_prob, threshold=0.8)

    assert metrics_default["f1_pos"] > metrics_high_threshold["f1_pos"]
    assert metrics_default["auroc"] == metrics_high_threshold["auroc"]


def test_find_best_threshold_returns_dict():
    y_true = [0, 0, 1, 1]
    y_prob = [0.2, 0.3, 0.7, 0.8]

    result = find_best_threshold(y_true, y_prob, optimize_for="precision_pos", grid=[0.4, 0.5, 0.6])

    assert set(result.keys()) == {"threshold", "metric", "metric_value"}
    assert result["metric"] == "precision_pos"
    assert 0.4 <= result["threshold"] <= 0.6


def test_tune_threshold_from_config_uses_grid():
    config = _make_eval_config()
    y_true = [0, 0, 1, 1]
    y_prob = [0.2, 0.4, 0.7, 0.9]

    tuning = tune_threshold_from_config(y_true, y_prob, config)

    assert tuning is not None
    assert tuning["threshold"] in config.evaluation.threshold_tuning.grid


def test_evaluate_predictions_returns_dataclass():
    config = _make_eval_config()
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.4, 0.7, 0.9])
    metadata = {"group": ["A", "A", "B", "B"]}

    result = evaluate_predictions(
        y_true,
        y_prob,
        config,
        split="validation",
        metadata=metadata,
        apply_config_calibration=True,
        tune_threshold=True,
    )

    assert isinstance(result, EvaluationResult)
    assert result.split == "validation"
    assert "f1_pos" in result.metrics_primary
    assert "auroc" in result.metrics_secondary
    assert result.threshold_tuning is not None
    assert result.subgroup_metrics is not None
    assert set(result.subgroup_metrics["group"].keys()) == {"A", "B"}
