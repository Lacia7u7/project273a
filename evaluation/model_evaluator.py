from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)

from evaluation.metrics import compute_metrics

try:
    import torch
except Exception:  # pragma: no cover - torch is optional at evaluation time
    torch = None

import random
import statistics


@dataclass
class ModelPerformance:
    name: str
    y_true: np.ndarray
    y_pred: np.ndarray
    y_prob: np.ndarray
    metrics: Dict[str, float]


class Evaluator:
    """Sklearn-style evaluation harness for comparing multiple models."""

    def __init__(self, models: Mapping[str, Any]) -> None:
        self.models: Dict[str, Any] = dict(models)
        self.performances: List[ModelPerformance] = []

    # ------------------------------------------------------------------
    # Core evaluation workflow
    # ------------------------------------------------------------------
    def fit(self, X: Any, y: Any) -> "Evaluator":
        for model in self.models.values():
            model.fit(X, y)
        return self

    def evaluate(self, X: Any, y: Any) -> "Evaluator":
        self.performances = []
        for name, model in self.models.items():
            prob = np.asarray(model.predict_proba(X))
            if prob.ndim == 2 and prob.shape[1] > 1:
                prob = prob[:, 1]
            else:
                prob = prob.reshape(-1)
            pred = np.asarray(model.predict(X)).reshape(-1)
            base_metrics = compute_metrics(y, prob, threshold=0.5)
            metrics = {
                "roc_auc": float(base_metrics.get("auroc", float("nan"))),
                "auprc": float(base_metrics.get("auprc", float("nan"))),
                "accuracy": float(np.mean(pred == y)),
                "balanced_accuracy": float(base_metrics.get("balanced_accuracy", float("nan"))),
                "precision": float(base_metrics.get("precision_pos", float("nan"))),
                "recall": float(base_metrics.get("recall_pos", float("nan"))),
                "f1": float(base_metrics.get("f1_pos", float("nan"))),
                "brier": float(base_metrics.get("brier", float("nan"))),
                "ece": float(base_metrics.get("ece", float("nan"))),
            }
            self.performances.append(ModelPerformance(name=name, y_true=np.asarray(y), y_pred=pred, y_prob=prob, metrics=metrics))
        return self

    # ------------------------------------------------------------------
    # Metric tables
    # ------------------------------------------------------------------
    def metrics_summary_table(self) -> pd.DataFrame:
        rows = []
        for perf in self.performances:
            rows.append({"model": perf.name, **perf.metrics})
        return pd.DataFrame(rows)

    def threshold_metrics_table(self, thresholds: Sequence[float] = (0.3, 0.5, 0.7)) -> pd.DataFrame:
        records: List[Dict[str, Any]] = []
        for perf in self.performances:
            for thr in thresholds:
                preds = (perf.y_prob >= thr).astype(int)
                records.append(
                    {
                        "model": perf.name,
                        "threshold": thr,
                        "precision": precision_score(perf.y_true, preds, zero_division=0),
                        "recall": recall_score(perf.y_true, preds, zero_division=0),
                        "f1": f1_score(perf.y_true, preds, zero_division=0),
                        "specificity": self._specificity(perf.y_true, preds),
                    }
                )
        return pd.DataFrame(records)

    def confusion_table(self) -> pd.DataFrame:
        rows = []
        for perf in self.performances:
            tn, fp, fn, tp = confusion_matrix(perf.y_true, perf.y_pred).ravel()
            rows.append({"model": perf.name, "tn": tn, "fp": fp, "fn": fn, "tp": tp})
        return pd.DataFrame(rows)

    def calibration_summary_table(self, n_bins: int = 10) -> pd.DataFrame:
        rows = []
        for perf in self.performances:
            prob_true, prob_pred = calibration_curve(perf.y_true, perf.y_prob, n_bins=n_bins)
            diff = np.abs(prob_true - prob_pred)
            rows.append(
                {
                    "model": perf.name,
                    "ece": float(np.average(diff, weights=np.ones_like(diff))),
                    "mean_pred": float(prob_pred.mean()),
                    "mean_true": float(prob_true.mean()),
                }
            )
        return pd.DataFrame(rows)

    def ranking_table(self, metric: str = "auprc") -> pd.DataFrame:
        tbl = self.metrics_summary_table()
        if metric not in tbl.columns:
            raise KeyError(f"Metric '{metric}' not found in summary table.")
        tbl = tbl.sort_values(metric, ascending=False).reset_index(drop=True)
        tbl["rank"] = np.arange(1, len(tbl) + 1)
        return tbl

    # ------------------------------------------------------------------
    # Required plots
    # ------------------------------------------------------------------
    def plot_roc(self) -> go.Figure:
        fig = go.Figure()
        for perf in self.performances:
            fpr, tpr, _ = roc_curve(perf.y_true, perf.y_prob)
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=perf.name))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="chance", line=dict(dash="dash")))
        fig.update_layout(title="ROC Curves", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        return fig

    def plot_precision_recall(self) -> go.Figure:
        fig = go.Figure()
        for perf in self.performances:
            precision, recall, _ = precision_recall_curve(perf.y_true, perf.y_prob)
            fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=perf.name))
        fig.update_layout(title="Precision-Recall Curves", xaxis_title="Recall", yaxis_title="Precision")
        return fig

    def plot_calibration(self, n_bins: int = 10) -> go.Figure:
        fig = go.Figure()
        for perf in self.performances:
            prob_true, prob_pred = calibration_curve(perf.y_true, perf.y_prob, n_bins=n_bins)
            fig.add_trace(go.Scatter(x=prob_pred, y=prob_true, mode="lines+markers", name=perf.name))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect", line=dict(dash="dash")))
        fig.update_layout(title="Calibration Curves", xaxis_title="Predicted probability", yaxis_title="Observed frequency")
        return fig

    def plot_confusion_matrices(self, normalize: bool = False) -> go.Figure:
        cols = len(self.performances)
        fig = make_subplots(rows=1, cols=cols, subplot_titles=[p.name for p in self.performances])
        for idx, perf in enumerate(self.performances, start=1):
            cm = confusion_matrix(perf.y_true, perf.y_pred)
            if normalize:
                cm = cm.astype(float) / cm.sum()
            fig.add_trace(go.Heatmap(z=cm, x=["Pred 0", "Pred 1"], y=["True 0", "True 1"], colorscale="Blues", showscale=idx == cols), row=1, col=idx)
        fig.update_layout(title="Confusion Matrices")
        return fig

    def plot_net_benefit(self, thresholds: Sequence[float] = np.linspace(0.01, 0.99, 25)) -> go.Figure:
        fig = go.Figure()
        for perf in self.performances:
            nb = [self._net_benefit(perf.y_true, perf.y_prob, thr) for thr in thresholds]
            fig.add_trace(go.Scatter(x=list(thresholds), y=nb, mode="lines", name=perf.name))
        fig.add_trace(go.Scatter(x=list(thresholds), y=[0] * len(thresholds), name="Treat none", line=dict(dash="dash")))
        prevalence = self.performances[0].y_true.mean() if self.performances else 0.0
        treat_all = [prevalence - (1 - prevalence) * (thr / (1 - thr)) for thr in thresholds]
        fig.add_trace(go.Scatter(x=list(thresholds), y=treat_all, name="Treat all", line=dict(dash="dot")))
        fig.update_layout(title="Decision Curve Analysis", xaxis_title="Threshold", yaxis_title="Net benefit")
        return fig

    # ------------------------------------------------------------------
    # Additional comparison plots
    # ------------------------------------------------------------------
    def plot_metric_bars(self, metric: str = "auprc") -> go.Figure:
        tbl = self.metrics_summary_table()
        if metric not in tbl.columns:
            raise KeyError(f"Metric '{metric}' not available for plotting.")
        fig = px.bar(tbl, x="model", y=metric, title=f"Model comparison: {metric}")
        return fig

    def plot_metric_radar(self, metrics: Sequence[str] = ("roc_auc", "auprc", "accuracy", "f1", "recall")) -> go.Figure:
        tbl = self.metrics_summary_table()
        fig = go.Figure()
        for _, row in tbl.iterrows():
            values = [row[m] for m in metrics]
            fig.add_trace(go.Scatterpolar(r=values + [values[0]], theta=list(metrics) + [metrics[0]], name=row["model"], fill="toself", opacity=0.3))
        fig.update_layout(title="Metric radar comparison", polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
        return fig

    def plot_probability_histogram(self, bins: int = 25) -> go.Figure:
        fig = go.Figure()
        for perf in self.performances:
            fig.add_trace(go.Histogram(x=perf.y_prob, name=perf.name, nbinsx=bins, opacity=0.5, histnorm="probability"))
        fig.update_layout(barmode="overlay", title="Predicted probability distribution", xaxis_title="Probability", yaxis_title="Density")
        return fig

    def plot_gain_chart(self, steps: int = 20) -> go.Figure:
        fig = go.Figure()
        for perf in self.performances:
            sorted_idx = np.argsort(perf.y_prob)[::-1]
            y_sorted = perf.y_true[sorted_idx]
            cumulative = np.cumsum(y_sorted)
            proportions = np.linspace(1 / len(y_sorted), 1.0, len(y_sorted))
            step_indices = np.linspace(0, len(y_sorted) - 1, steps).astype(int)
            fig.add_trace(go.Scatter(x=proportions[step_indices], y=(cumulative[step_indices] / cumulative[-1]), mode="lines", name=perf.name))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="baseline", line=dict(dash="dash")))
        fig.update_layout(title="Cumulative gain chart", xaxis_title="Fraction of population", yaxis_title="Fraction of positives captured")
        return fig

    def plot_threshold_sweep(self, metric: str = "f1") -> go.Figure:
        fig = go.Figure()
        thresholds = np.linspace(0.05, 0.95, 30)
        for perf in self.performances:
            values = []
            for thr in thresholds:
                preds = (perf.y_prob >= thr).astype(int)
                if metric == "precision":
                    values.append(precision_score(perf.y_true, preds, zero_division=0))
                elif metric == "recall":
                    values.append(recall_score(perf.y_true, preds, zero_division=0))
                elif metric == "specificity":
                    values.append(self._specificity(perf.y_true, preds))
                else:
                    values.append(f1_score(perf.y_true, preds, zero_division=0))
            fig.add_trace(go.Scatter(x=list(thresholds), y=values, mode="lines", name=perf.name))
        fig.update_layout(title=f"Threshold sweep ({metric})", xaxis_title="Threshold", yaxis_title=metric.capitalize())
        return fig

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0

    @staticmethod
    def _net_benefit(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> float:
        preds = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        n = len(y_true)
        if n == 0 or threshold in (0, 1):
            return 0.0
        return (tp / n) - (fp / n) * (threshold / (1 - threshold))


class RepeatedTrainingStudy:
    """Run repeated train/evaluate cycles to assess stability."""

    def __init__(
        self,
        model_factory: Callable[[], Any],
        *,
        metric_names: Optional[Sequence[str]] = None,
        threshold: float = 0.5,
    ) -> None:
        self.model_factory = model_factory
        self.metric_names = list(metric_names) if metric_names is not None else None
        self.threshold = float(threshold)
        self.runs_: Optional[pd.DataFrame] = None
        self.summary_: Optional[pd.DataFrame] = None

    def run(
        self,
        train_X: Any,
        train_y: Any,
        test_X: Any,
        test_y: Any,
        *,
        n_runs: int = 5,
        random_state: Optional[int] = None,
        callbacks: Optional[Sequence[Callable[[int, Mapping[str, float]], None]]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Execute ``n_runs`` fits and evaluations, returning per-run and summary metrics."""

        if n_runs <= 0:
            raise ValueError("n_runs must be a positive integer")

        callbacks = list(callbacks) if callbacks is not None else []

        seeds = self._generate_seeds(n_runs, random_state)
        run_records: List[Dict[str, float]] = []

        for idx, seed in enumerate(seeds, start=1):
            model = self.model_factory()
            self._seed_everything(seed)
            self._apply_model_seed(model, seed)

            model.fit(train_X, train_y)

            prob = np.asarray(model.predict_proba(test_X))
            if prob.ndim == 2 and prob.shape[1] > 1:
                prob = prob[:, 1]
            else:
                prob = prob.reshape(-1)
            pred = np.asarray(model.predict(test_X)).reshape(-1)

            metrics = compute_metrics(test_y, prob, threshold=self.threshold)
            metrics = self._select_metrics(metrics)
            metrics["accuracy"] = float(np.mean(pred == np.asarray(test_y)))

            record = {"run": idx, **metrics}
            run_records.append(record)

            for cb in callbacks:
                cb(idx, record)

        runs_df = pd.DataFrame(run_records)
        summary_df = self._summarise_runs(runs_df)

        self.runs_ = runs_df
        self.summary_ = summary_df
        return runs_df, summary_df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _select_metrics(self, metrics: Mapping[str, float]) -> Dict[str, float]:
        if not self.metric_names:
            return dict(metrics)
        return {name: float(metrics.get(name, float("nan"))) for name in self.metric_names}

    @staticmethod
    def _generate_seeds(n_runs: int, random_state: Optional[int]) -> Iterable[int]:
        if random_state is None:
            return range(1, n_runs + 1)
        rng = np.random.default_rng(random_state)
        return rng.integers(0, 2**32 - 1, size=n_runs)

    @staticmethod
    def _seed_everything(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        if torch is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():  # pragma: no cover - depends on runtime
                torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _apply_model_seed(model: Any, seed: int) -> None:
        params_getter = getattr(model, "get_params", None)
        if callable(params_getter):
            try:
                params = params_getter()
            except TypeError:
                params = params_getter(deep=True)
            if isinstance(params, Mapping) and "random_state" in params and hasattr(model, "set_params"):
                try:
                    model.set_params(random_state=int(seed))
                    return
                except Exception:
                    pass
        if hasattr(model, "random_state"):
            try:
                setattr(model, "random_state", int(seed))
            except Exception:
                pass

    @staticmethod
    def _summarise_runs(runs: pd.DataFrame) -> pd.DataFrame:
        metric_cols = [col for col in runs.columns if col != "run"]
        summaries: List[Dict[str, float]] = []
        for metric in metric_cols:
            values = runs[metric].dropna().to_numpy(dtype=float)
            if values.size == 0:
                continue
            summary = {
                "metric": metric,
                "mean": float(np.mean(values)),
                "variance": float(np.var(values, ddof=1)) if values.size > 1 else 0.0,
                "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                "median": float(np.median(values)),
                "mode": float(RepeatedTrainingStudy._mode(values)),
            }
            summaries.append(summary)
        return pd.DataFrame(summaries)

    @staticmethod
    def _mode(values: Sequence[float]) -> float:
        modes = statistics.multimode(values)
        return float(modes[0]) if modes else float("nan")


__all__ = ["Evaluator", "ModelPerformance", "RepeatedTrainingStudy"]
