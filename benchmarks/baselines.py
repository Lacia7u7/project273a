import numpy as np
from typing import Dict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from train.evaluation import EvaluationResult, evaluate_predictions
from train.metrics import compute_metrics


def _predict_proba_or_score(model, X):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] > 1:
            return proba[:, 1]
        return proba.reshape(-1)
    return model.decision_function(X)


def train_and_eval_baselines(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    *,
    config=None,
    val_metadata=None,
    test_metadata=None,
):
    """Train multiple baseline ML models and evaluate on validation and test sets."""

    results = {}
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
        "LightGBM": LGBMClassifier(random_state=42),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)

        val_output: Dict[str, EvaluationResult | Dict[str, float]]
        if X_val is not None and y_val is not None:
            val_prob = _predict_proba_or_score(model, X_val)
            if config is not None:
                val_eval = evaluate_predictions(
                    y_val,
                    val_prob,
                    config,
                    split="val",
                    metadata=val_metadata,
                    tune_threshold=True,
                    apply_config_calibration=True,
                )
                val_output = val_eval
                threshold_to_use = val_eval.threshold
                calibrator = val_eval.calibration_model
            else:
                val_output = compute_metrics(y_val, val_prob)
                threshold_to_use = 0.5
                calibrator = None
        else:
            val_output = {}
            threshold_to_use = 0.5
            calibrator = None

        test_prob = _predict_proba_or_score(model, X_test)
        if config is not None:
            test_eval = evaluate_predictions(
                y_test,
                test_prob,
                config,
                split="test",
                metadata=test_metadata,
                apply_config_calibration=calibrator is not None,
                existing_threshold=threshold_to_use,
            )
            # Reuse fitted calibrator if available
            if calibrator is not None and test_eval.calibration_model is None:
                test_eval.calibration_model = calibrator
        else:
            test_eval = compute_metrics(y_test, test_prob)

        results[name] = {
            "val": val_output,
            "test": test_eval,
        }

    return results
