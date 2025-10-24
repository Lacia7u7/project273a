from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional

import pandas as pd
from sklearn.model_selection import ParameterGrid

from models.wrapper import SklearnLikeGNN


@dataclass
class GridSearchReport:
    results: pd.DataFrame
    best_params: Dict[str, Any]
    best_score: float
    best_model: SklearnLikeGNN


def _clone_config(config: Any) -> Any:
    if hasattr(config, "copy"):
        try:
            return config.copy(deep=True)
        except TypeError:
            return config.copy()
    return copy.deepcopy(config)


def _set_nested_attr(obj: Any, path: str, value: Any) -> None:
    parts = path.split(".")
    target = obj
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], value)


class GridSearchGNN:
    """Hyper-parameter search helper for :class:`SklearnLikeGNN` models."""

    def __init__(
        self,
        model_cls: type[SklearnLikeGNN],
        *,
        base_config: Any,
        model_kwargs: Optional[Dict[str, Any]] = None,
        train_kwargs: Optional[Dict[str, Any]] = None,
        scoring: Optional[Callable[[SklearnLikeGNN, Any], float]] = None,
        metric_name: str = "auprc",
        greater_is_better: bool = True,
    ) -> None:
        self.model_cls = model_cls
        self.base_config = base_config
        self.model_kwargs = model_kwargs or {}
        self.train_kwargs = train_kwargs or {"epochs": 5}
        self.scoring = scoring
        self.metric_name = metric_name
        self.greater_is_better = greater_is_better

        self.param_grid = self._extract_param_grid()

    def _extract_param_grid(self) -> Dict[str, List[Any]]:
        grid: Dict[str, List[Any]] = {}
        for name, param in self.model_cls.suggested_grid().items():
            grid[name] = list(param.values)
        return grid

    def _prepare_config(self, params: Mapping[str, Any]) -> Any:
        config = _clone_config(self.base_config)
        for path, value in params.items():
            _set_nested_attr(config, path, value)
        return config

    def _score_model(self, model: SklearnLikeGNN, data: Any) -> float:
        if self.scoring is not None:
            return float(self.scoring(model, data))
        metrics = model.evaluate_loader(data)
        return float(metrics.get(self.metric_name, float("nan")))

    def run(
        self,
        train_data: Any,
        *,
        val_data: Optional[Any] = None,
        device: Optional[str] = None,
    ) -> GridSearchReport:
        if not self.param_grid:
            raise ValueError("Model does not declare any suggested parameters for grid search.")

        parameter_grid = ParameterGrid(self.param_grid)
        results: List[Dict[str, Any]] = []
        best_score = float("-inf") if self.greater_is_better else float("inf")
        best_model: Optional[SklearnLikeGNN] = None
        best_params: Dict[str, Any] = {}

        evaluation_data = val_data if val_data is not None else train_data

        for params in parameter_grid:
            config = self._prepare_config(params)
            model_kwargs = {**self.model_kwargs, "config": config}
            model = self.model_cls(**model_kwargs)
            model.fit(train_data, val_data=val_data, device=device, **self.train_kwargs)
            score = self._score_model(model, evaluation_data)
            row = {**params, "score": score}
            results.append(row)

            better = score > best_score if self.greater_is_better else score < best_score
            if better or best_model is None:
                best_score = score
                best_model = model
                best_params = dict(params)

        results_df = pd.DataFrame(results)
        if results_df.empty:
            raise RuntimeError("Grid search did not evaluate any parameter combinations.")

        assert best_model is not None
        return GridSearchReport(
            results=results_df.sort_values("score", ascending=not self.greater_is_better).reset_index(drop=True),
            best_params=best_params,
            best_score=float(best_score),
            best_model=best_model,
        )


__all__ = ["GridSearchGNN", "GridSearchReport"]
