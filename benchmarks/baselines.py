import numpy as np
from typing import Dict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from train.evaluation import EvaluationResult, evaluate_predictions
from train.metrics import compute_metrics
from tqdm.auto import tqdm


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
        # Fast, strong baseline; scaling helps most linear models
        "LogisticRegression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                solver="lbfgs",  # robust for dense ~60 features
                max_iter=300,  # lighter than 1000
                class_weight="balanced",
                n_jobs=None  # (only used by some solvers; lbfgs ignores)
            )
        ),

        # Much lighter trees: shallower + fewer leaves
        "RandomForest": RandomForestClassifier(
            n_estimators=500,  # keep modest
            max_depth=12,  # cap tree depth
            min_samples_leaf=5,  # coarser leaves
            max_features="sqrt",  # standard for classification
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42
        ),

        # Histogram-based GBM + early stopping in fit()
        "XGBoost": XGBClassifier(
            tree_method="hist",  # fast on tabular
            max_depth=12,  # shallow trees
            learning_rate=0.1,  # pair with early stopping
            n_estimators=1000,  # large cap; will stop early
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            eval_metric="aucpr",
            n_jobs=-1,
            random_state=42,
        ),
        # Optional: KNN is heavy to predict on 100k; keep only if you really want it.
        "KNN": make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(
                n_neighbors=11,
                weights="distance",
                algorithm="ball_tree",  # faster than brute in moderate dims
                leaf_size=50,
                n_jobs=-1
            )
        ),
    }
    for name, model in tqdm(list(models.items()), desc="Training (light) baselines", unit="model"):
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
