from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


@dataclass
class BaselineSearchReport:
    results: pd.DataFrame
    best_model: BaseEstimator
    best_params: Dict[str, Any]
    best_score: float


BaselineFactory = Callable[[], BaseEstimator]


def _logreg_factory() -> Pipeline:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            solver="lbfgs",
            max_iter=300,
            class_weight="balanced",
        ),
    )


def _knn_factory() -> Pipeline:
    return make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(
            algorithm="auto",
            n_neighbors=11,
            weights="distance",
            n_jobs=-1,
        ),
    )


def _rf_factory() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=4,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )


def _xgb_factory() -> XGBClassifier:
    return XGBClassifier(
        tree_method="hist",
        eval_metric="aucpr",
        subsample=0.8,
        colsample_bytree=0.8,
        learning_rate=0.1,
        max_depth=10,
        n_estimators=600,
        n_jobs=-1,
        random_state=42,
        reg_lambda=1.0,
    )


DEFAULT_BASELINE_GRIDS: Dict[str, Tuple[BaselineFactory, Dict[str, List[Any]]]] = {
    "logistic_regression": (
        _logreg_factory,
        {
            "logisticregression__C": [0.25, 1.0, 4.0],
            "logisticregression__solver": ["lbfgs"],
        },
    ),
    "random_forest": (
        _rf_factory,
        {
            "n_estimators": [200, 400],
            "max_depth": [8, 12],
            "min_samples_leaf": [3, 5],
        },
    ),
    "xgboost": (
        _xgb_factory,
        {
            "learning_rate": [0.05, 0.1],
            "max_depth": [6, 10],
            "subsample": [0.7, 0.85],
            "colsample_bytree": [0.6, 0.8],
        },
    ),
    "knn": (
        _knn_factory,
        {
            "kneighborsclassifier__n_neighbors": [7, 11, 17],
            "kneighborsclassifier__weights": ["uniform", "distance"],
        },
    ),
}


class BaselineGridSearch:
    """Lightweight grid search utility for scikit-learn baseline models."""

    def __init__(
        self,
        baselines: Optional[Dict[str, Tuple[BaselineFactory, Dict[str, List[Any]]]]] = None,
        *,
        scoring: Optional[Callable[[BaseEstimator, np.ndarray, np.ndarray], float]] = None,
        metric_name: str = "auprc",
        greater_is_better: bool = True,
    ) -> None:
        self.baselines = baselines or DEFAULT_BASELINE_GRIDS
        self.scoring = scoring
        self.metric_name = metric_name
        self.greater_is_better = greater_is_better

    def _evaluate(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        prob = model.predict_proba(X)
        if prob.ndim == 2 and prob.shape[1] > 1:
            prob = prob[:, 1]
        else:
            prob = prob.reshape(-1)
        pred = (prob >= 0.5).astype(int)
        return {
            "roc_auc": roc_auc_score(y, prob),
            "auprc": average_precision_score(y, prob),
            "accuracy": accuracy_score(y, pred),
            "f1": f1_score(y, pred),
        }

    def run(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        sample_size: Optional[int] = None,
        random_state: int = 42,
    ) -> BaselineSearchReport:
        X_val = X_val if X_val is not None else X_train
        y_val = y_val if y_val is not None else y_train

        if sample_size is not None and X_train.shape[0] > sample_size:
            rng = np.random.default_rng(random_state)
            idx = rng.choice(X_train.shape[0], size=sample_size, replace=False)
            X_train = X_train[idx]
            y_train = y_train[idx]

        rows: List[Dict[str, Any]] = []
        best_score = float("-inf") if self.greater_is_better else float("inf")
        best_model: Optional[BaseEstimator] = None
        best_params: Dict[str, Any] = {}

        for name, (factory, grid) in self.baselines.items():
            param_grid = ParameterGrid(grid)
            for params in param_grid:
                model = factory()
                model = clone(model)
                model.set_params(**params)
                model.fit(X_train, y_train)

                if self.scoring is not None:
                    score = float(self.scoring(model, X_val, y_val))
                else:
                    metrics = self._evaluate(model, X_val, y_val)
                    score = float(metrics[self.metric_name])
                row = {"model": name, **params, "score": score}
                rows.append(row)

                better = score > best_score if self.greater_is_better else score < best_score
                if better or best_model is None:
                    best_score = score
                    best_model = model
                    best_params = {"model": name, **params}

        results_df = pd.DataFrame(rows)
        if best_model is None or results_df.empty:
            raise RuntimeError("Grid search did not evaluate any baseline combinations.")

        return BaselineSearchReport(
            results=results_df.sort_values("score", ascending=not self.greater_is_better).reset_index(drop=True),
            best_model=best_model,
            best_params=best_params,
            best_score=float(best_score),
        )


__all__ = ["BaselineGridSearch", "BaselineSearchReport", "DEFAULT_BASELINE_GRIDS"]
