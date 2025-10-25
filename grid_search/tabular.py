from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import math
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
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
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# Progress bar
from tqdm.auto import tqdm
# robust tqdm_joblib import (works with tqdm>=4.64 or the external package)
try:
    from tqdm.contrib.tqdm_joblib import tqdm_joblib
except Exception:  # pragma: no cover
    from tqdm_joblib import tqdm_joblib  # pip install tqdm-joblib

# Optional CatBoost
try:
    from catboost import CatBoostClassifier  # pip install catboost
    _HAVE_CATBOOST = True
except Exception:  # pragma: no cover
    CatBoostClassifier = None  # type: ignore
    _HAVE_CATBOOST = False


@dataclass
class BaselineSearchReport:
    results: pd.DataFrame
    best_model: BaseEstimator
    best_params: Dict[str, Any]
    best_score: float


BaselineFactory = Callable[[], BaseEstimator]


# ---------- helpers to expand search spaces by ~10% ----------
def _expand_numeric(values: List[float], pct: float = 0.10, *, integer: bool = False,
                    bounds: Optional[Tuple[float, float]] = None) -> List[float]:
    vals = [float(v) for v in values]
    lo, hi = min(vals), max(vals)
    new_lo = lo * (1.0 - pct)
    new_hi = hi * (1.0 + pct)
    if bounds is not None:
        lb, ub = bounds
        new_lo = max(lb, new_lo)
        new_hi = min(ub, new_hi)
    if integer:
        new_lo = max(1, int(math.floor(new_lo)))
        new_hi = int(math.ceil(new_hi))
        out = sorted(set([int(v) for v in values] + [new_lo, new_hi]))
    else:
        out = sorted(set(values + [new_lo, new_hi]))
    return out


# ---------- factories ----------
def _logreg_factory() -> Pipeline:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(solver="lbfgs", max_iter=300, class_weight="balanced"),
    )

def _knn_factory() -> Pipeline:
    return make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(algorithm="auto", n_neighbors=11, weights="distance", n_jobs=-1),
    )

def _rf_factory() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=400, max_depth=12, min_samples_leaf=4,
        class_weight="balanced_subsample", n_jobs=-1, random_state=42,
    )

def _xgb_factory() -> XGBClassifier:
    return XGBClassifier(
        tree_method="hist", eval_metric="aucpr",
        subsample=0.8, colsample_bytree=0.8,
        learning_rate=0.1, max_depth=10, n_estimators=600,
        n_jobs=-1, random_state=42, reg_lambda=1.0,
    )

def _linear_svc_factory() -> Pipeline:
    # fast linear SVM; scaler helps a lot
    return make_pipeline(
        StandardScaler(),
        LinearSVC(loss="squared_hinge"),  # decision_function for scoring
    )

def _mlp_factory() -> MLPClassifier:
    return MLPClassifier(random_state=42)

def _extratrees_factory() -> ExtraTreesClassifier:
    return ExtraTreesClassifier(n_estimators=400, max_depth=None, n_jobs=-1, random_state=42)

def _gb_factory() -> GradientBoostingClassifier:
    return GradientBoostingClassifier(random_state=42)

def _ada_factory() -> AdaBoostClassifier:
    return AdaBoostClassifier(random_state=42)

def _cat_factory() -> CatBoostClassifier:
    # use_logloss; silent; let thread_count be tuned/overridden
    return CatBoostClassifier(
        loss_function="Logloss",
        verbose=0,
        allow_writing_files=False,
        random_state=42,
    )


# ---------- expanded grids (~50% more params, ~10% wider numeric ranges) ----------
# Logistic Regression
_logreg_C = _expand_numeric([0.25, 1.0, 4.0], pct=0.10)
_logreg_max_iter = _expand_numeric([200, 300, 400], pct=0.10, integer=True)
_logreg_grid = {
    "logisticregression__C": _logreg_C,
    "logisticregression__solver": ["lbfgs"],
    "logisticregression__max_iter": _logreg_max_iter,     # + param
    "logisticregression__fit_intercept": [True, False],    # + param
}

# Random Forest
_rf_n_estimators = _expand_numeric([200, 400], pct=0.10, integer=True)
_rf_max_depth = _expand_numeric([8, 12], pct=0.10, integer=True)
_rf_min_samples_leaf = _expand_numeric([3, 5], pct=0.10, integer=True)
_rf_grid = {
    "n_estimators": _rf_n_estimators,
    "max_depth": _rf_max_depth,
    "min_samples_leaf": _rf_min_samples_leaf,
    "min_samples_split": _expand_numeric([2, 4], pct=0.10, integer=True, bounds=(2, 100000000)),  # + param
    "max_features": ["sqrt", "log2", 0.8],                                 # + param
}

# XGBoost
_xgb_lr  = _expand_numeric([0.05, 0.1], pct=0.10)
_xgb_md  = _expand_numeric([6, 10], pct=0.10, integer=True)
_xgb_sub = _expand_numeric([0.7, 0.85], pct=0.10, bounds=(0.0, 1.0))
_xgb_col = _expand_numeric([0.6, 0.8], pct=0.10, bounds=(0.0, 1.0))
_xgb_grid = {
    "learning_rate": _xgb_lr,
    "max_depth": _xgb_md,
    "subsample": _xgb_sub,
    "colsample_bytree": _xgb_col,
    "min_child_weight": _expand_numeric([1, 3], pct=0.10, integer=True),     # + param
    "reg_lambda": _expand_numeric([0.5, 1.0, 2.0], pct=0.10),                # + param
}

# KNN
_knn_neighbors = _expand_numeric([7, 11, 17], pct=0.10, integer=True)
_knn_grid = {
    "kneighborsclassifier__n_neighbors": _knn_neighbors,
    "kneighborsclassifier__weights": ["uniform", "distance"],
    "kneighborsclassifier__p": [1, 2],  # + param
}

# Linear SVC (fast)
_lsvc_C = _expand_numeric([0.5, 1.0, 2.0], pct=0.10)
_lsvc_max_iter = _expand_numeric([1000, 1500], pct=0.10, integer=True)
_lsvc_grid = {
    "linearsvc__C": _lsvc_C,
    "linearsvc__dual": [True, False],             # + param
    "linearsvc__class_weight": [None, "balanced"],# + param
    "linearsvc__max_iter": _lsvc_max_iter,        # + param
    "linearsvc__fit_intercept": [True, False],    # + param
}

# MLP (neural network)
_mlp_lr = _expand_numeric([1e-3, 1e-2], pct=0.10)
_mlp_alpha = _expand_numeric([1e-4, 1e-3], pct=0.10)
_mlp_max_iter = _expand_numeric([200, 300], pct=0.10, integer=True)
_mlp_grid = {
    "hidden_layer_sizes": [(64,), (128,), (64, 64), (128, 64)],  # variety
    "learning_rate_init": _mlp_lr,
    "alpha": _mlp_alpha,
    "activation": ["relu", "tanh"],           # + param
    "solver": ["adam", "lbfgs"],              # + param (note: lbfgs ignores batch params)
    "max_iter": _mlp_max_iter,                # + param
}

# ExtraTrees
_et_n_estimators = _expand_numeric([200, 400], pct=0.10, integer=True)
_et_max_depth_vals = _expand_numeric([8, 12], pct=0.10, integer=True)
_et_min_leaf = _expand_numeric([1, 2, 4], pct=0.10, integer=True)
_et_grid = {
    "n_estimators": _et_n_estimators,
    "max_depth": [None] + _et_max_depth_vals,   # include unlimited depth
    "min_samples_leaf": _et_min_leaf,
    "min_samples_split": _expand_numeric([2, 4], pct=0.10, integer=True),
    "max_features": ["sqrt", "log2", 0.8],
}

# Gradient Boosting
_gb_n_estimators = _expand_numeric([200, 400], pct=0.10, integer=True)
_gb_lr = _expand_numeric([0.05, 0.1], pct=0.10)
_gb_max_depth = _expand_numeric([2, 3], pct=0.10, integer=True)
_gb_grid = {
    "n_estimators": _gb_n_estimators,
    "learning_rate": _gb_lr,
    "max_depth": _gb_max_depth,
    "subsample": _expand_numeric([0.8, 1.0], pct=0.10, bounds=(0.0, 1.0)),
    "max_features": ["sqrt", "log2", None],
}

# AdaBoost
_ab_n_estimators = _expand_numeric([100, 200], pct=0.10, integer=True)
_ab_lr = _expand_numeric([0.5, 1.0], pct=0.10)
_ab_grid = {
    "n_estimators": _ab_n_estimators,
    "learning_rate": _ab_lr,
    "algorithm": ["SAMME.R", "SAMME"],
}

# CatBoost (if available)
if _HAVE_CATBOOST:
    _cat_lr = _expand_numeric([0.05, 0.1], pct=0.10)
    _cat_depth = _expand_numeric([6, 8], pct=0.10, integer=True)
    _cat_l2 = _expand_numeric([1.0, 3.0], pct=0.10)
    _cat_iters = _expand_numeric([300, 600], pct=0.10, integer=True)
    _cat_subsample = _expand_numeric([0.8, 1.0], pct=0.10, bounds=(0.0, 1.0))
    _cat_grid = {
        "learning_rate": _cat_lr,
        "depth": _cat_depth,
        "l2_leaf_reg": _cat_l2,
        "iterations": _cat_iters,
        "subsample": _cat_subsample,
        "random_strength": _expand_numeric([1.0, 2.0], pct=0.10),  # + param
        # thread_count handled by outer parallelization control
    }


# ---------- registry ----------
DEFAULT_BASELINE_GRIDS: Dict[str, Tuple[BaselineFactory, Dict[str, List[Any]]]] = {
    "logistic_regression": (_logreg_factory, _logreg_grid),
    "random_forest": (_rf_factory, _rf_grid),
    "xgboost": (_xgb_factory, _xgb_grid),
    "knn": (_knn_factory, _knn_grid),
    "linear_svc": (_linear_svc_factory, _lsvc_grid),
    "mlp": (_mlp_factory, _mlp_grid),
    "extra_trees": (_extratrees_factory, _et_grid),
    "gradient_boosting": (_gb_factory, _gb_grid),
    "adaboost": (_ada_factory, _ab_grid),
}
if _HAVE_CATBOOST:
    DEFAULT_BASELINE_GRIDS["catboost"] = (_cat_factory, _cat_grid)


class BaselineGridSearch:
    """Lightweight grid search utility for scikit-learn baseline models.

    Paraleliza con joblib (por defecto, todos los cores con n_jobs=-1).
    Evita sobre-suscripción forzando 'n_jobs'/'thread_count' internos a 1
    cuando se usa paralelismo externo.

    Muestra barra de progreso con tqdm que avanza al completar cada job.

    Nuevo:
      - max_configs_per_model: si hay más combinaciones para un modelo,
        se hace un submuestreo uniforme hasta ese máximo.
    """

    def __init__(
        self,
        baselines: Optional[Dict[str, Tuple[BaselineFactory, Dict[str, List[Any]]]]] = None,
        *,
        scoring: Optional[Callable[[BaseEstimator, np.ndarray, np.ndarray], float]] = None,
        metric_name: str = "auprc",
        greater_is_better: bool = True,
        n_jobs: int = -1,
        backend: Optional[str] = None,
        verbose: int = 0,
        show_progress: bool = True,
        tqdm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.baselines = baselines or DEFAULT_BASELINE_GRIDS
        self.scoring = scoring
        self.metric_name = metric_name
        self.greater_is_better = greater_is_better
        self.n_jobs = n_jobs
        self.backend = backend
        self.verbose = verbose
        self.show_progress = show_progress
        self.tqdm_kwargs = tqdm_kwargs or {}

    def _evaluate(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        # Preferir probabilidades; luego decision_function; último recurso: labels
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X)
            if prob.ndim == 2 and prob.shape[1] > 1:
                scores = prob[:, 1]
            else:
                scores = prob.reshape(-1)
            preds = (scores >= 0.5).astype(int)
        elif hasattr(model, "decision_function"):
            scores = np.asarray(model.decision_function(X)).reshape(-1)
            preds = (scores >= 0.0).astype(int)
        else:
            preds = model.predict(X)
            scores = preds.astype(float)

        return {
            "roc_auc": roc_auc_score(y, scores),
            "auprc": average_precision_score(y, scores),
            "accuracy": accuracy_score(y, preds),
            "f1": f1_score(y, preds),
        }

    @staticmethod
    def _set_all_inner_n_jobs(model: BaseEstimator, n_jobs_value: int) -> BaseEstimator:
        """Fuerza todos los paths que terminan en 'n_jobs' o 'thread_count'."""
        params = model.get_params(deep=True)
        to_set = {k: n_jobs_value for k in params.keys()
                  if k.endswith("n_jobs") or k.endswith("thread_count")}
        if to_set:
            model.set_params(**to_set)
        return model

    @staticmethod
    def _even_subsample_dicts(items: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
        """Submuestreo uniforme y determinista de 'items' a k elementos (k < len(items))."""
        n = len(items)
        if k >= n or k <= 0:
            return items
        out = []
        for i in range(k):
            start = (i * n) // k
            end = ((i + 1) * n) // k - 1
            idx = (start + end) // 2  # centro del bin
            out.append(items[idx])
        return out

    def _fit_and_score_single(
        self,
        name: str,
        factory: BaselineFactory,
        params: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        outer_parallel_active: bool,
    ) -> Dict[str, Any]:
        model = clone(factory())
        model.set_params(**params)

        if outer_parallel_active:
            model = self._set_all_inner_n_jobs(model, 1)

        # CatBoost: por si no pasó thread_count en params
        if "catboost" in type(model).__name__.lower():
            try:
                model.set_params(thread_count=1 if outer_parallel_active else -1)
            except Exception:
                pass

        model.fit(X_train, y_train)

        if self.scoring is not None:
            score = float(self.scoring(model, X_val, y_val))
        else:
            metrics = self._evaluate(model, X_val, y_val)
            score = float(metrics[self.metric_name])

        return {"model_name": name, "model": model, "params": params, "score": score}

    def run(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        sample_size: Optional[int] = None,
        random_state: int = 42,
        max_configs_per_model: Optional[int] = None,  # <--- NUEVO
    ) -> BaselineSearchReport:
        X_val = X_val if X_val is not None else X_train
        y_val = y_val if y_val is not None else y_train

        if sample_size is not None and X_train.shape[0] > sample_size:
            rng = np.random.default_rng(random_state)
            idx = rng.choice(X_train.shape[0], size=sample_size, replace=False)
            X_train = X_train[idx]
            y_train = y_train[idx]

        # Construir (modelo, params) con posible submuestreo por modelo
        configs: List[Tuple[str, BaselineFactory, Dict[str, Any]]] = []
        for name, (factory, grid) in self.baselines.items():
            full_list = list(ParameterGrid(grid))
            if max_configs_per_model is not None and max_configs_per_model > 0 and len(full_list) > max_configs_per_model:
                chosen = self._even_subsample_dicts(full_list, max_configs_per_model)
            else:
                chosen = full_list
            for params in chosen:
                configs.append((name, factory, params))

        if not configs:
            raise RuntimeError("Grid search did not evaluate any baseline combinations.")

        outer_parallel_active = (self.n_jobs is not None and self.n_jobs != 1)
        parallel = Parallel(n_jobs=self.n_jobs, backend=self.backend, verbose=self.verbose)

        if self.show_progress:
            tk = dict(self.tqdm_kwargs)
            tk.setdefault("desc", "Evaluating baselines")
            total = len(configs)
            from tqdm.auto import tqdm  # asegurar import local si se usa en otros entornos
            try:
                from tqdm.contrib.tqdm_joblib import tqdm_joblib
            except Exception:
                from tqdm_joblib import tqdm_joblib
            with tqdm_joblib(tqdm(total=total, **tk)):
                results = parallel(
                    delayed(self._fit_and_score_single)(
                        name, factory, params, X_train, y_train, X_val, y_val, outer_parallel_active
                    )
                    for (name, factory, params) in configs
                )
        else:
            results = parallel(
                delayed(self._fit_and_score_single)(
                    name, factory, params, X_train, y_train, X_val, y_val, outer_parallel_active
                )
                for (name, factory, params) in configs
            )

        # Agregar filas y elegir el mejor
        rows: List[Dict[str, Any]] = []
        best_score = float("-inf") if self.greater_is_better else float("inf")
        best_result: Optional[Dict[str, Any]] = None

        for res in results:
            rows.append({"model": res["model_name"], **res["params"], "score": res["score"]})
            better = res["score"] > best_score if self.greater_is_better else res["score"] < best_score
            if better or best_result is None:
                best_score = res["score"]
                best_result = res

        results_df = pd.DataFrame(rows).sort_values(
            "score", ascending=not self.greater_is_better
        ).reset_index(drop=True)

        if best_result is None:
            raise RuntimeError("Grid search did not evaluate any baseline combinations.")

        return BaselineSearchReport(
            results=results_df,
            best_model=best_result["model"],
            best_params={"model": best_result["model_name"], **best_result["params"]},
            best_score=float(best_score),
        )



__all__ = ["BaselineGridSearch", "BaselineSearchReport", "DEFAULT_BASELINE_GRIDS"]
